"""AgenticEngine - the agentic (PDP) HTTP front door.

Unlike the standalone ``deepintshield_agents`` client it was migrated from,
the engine does not take its own ``gateway_url``/``virtual_key`` - it binds
to the parent :class:`~deepintshield.client.DeepintShield` and reuses its
``base_url``, ``virtual_key`` and shared httpx client. The user therefore
passes *only* virtual key + base URL to ``DeepintShield`` and everything
agentic (discovery, credential selection, agent-token minting) happens
automatically.

What it does at decide() time:
    1. POST /api/agentic-security/decide with two auth headers:
       - Authorization: Bearer <virtual_key>  (platform VK)
       - X-Agent-Token: <fresh agent token from the credential>  (best-effort)
    2. Returns the Decision; the gate translates verdicts into exceptions.

Discovery + credential build are lazy (first decide(), not at construction)
so building a ``DeepintShield`` never performs network I/O.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Optional

import httpx

from .credentials.base import AgentCredential, StaticAgentCredential
from .errors import DeepIntShieldError, GatewayUnavailable
from .types import Decision, DelegationContext, Verdict, VKCredentialInfo

if TYPE_CHECKING:
    from ..client import DeepintShield

log = logging.getLogger(__name__)

_DISCOVERY_TTL_SECONDS = 3600.0


class AgenticEngine:
    """Handles every interaction with the gateway's agentic-security
    endpoints - discovery, decide, approval polling."""

    def __init__(
        self,
        parent: "DeepintShield",
        *,
        agent_credential: Optional[AgentCredential] = None,
        approval_poll_timeout_seconds: float = 300.0,
        approval_poll_interval_seconds: float = 2.0,
    ) -> None:
        self._parent = parent
        self._cred_info: Optional[VKCredentialInfo] = None
        self._cred_info_fetched_at: float = 0.0
        self._agent_credential: Optional[AgentCredential] = agent_credential
        self._approval_timeout = approval_poll_timeout_seconds
        self._approval_interval = approval_poll_interval_seconds
        # Per-process run id (OTel/Langfuse style). Stamped on every decide so all
        # of this client's steps group into ONE Agent Execution; a fresh process
        # (new engine) → a new execution. Override per call via dc.session_id, or
        # set engine.session_id to bind a longer-lived agent run to one execution.
        self.session_id: str = "sdk-" + uuid.uuid4().hex[:12]

    # ──────────────────────────────────────────────────────────────────
    # Parent-backed connection details
    # ──────────────────────────────────────────────────────────────────

    @property
    def gateway_url(self) -> str:
        return self._parent.base_url

    @property
    def virtual_key(self) -> str:
        return self._parent.virtual_key_or_raise()

    @property
    def _http(self) -> httpx.Client:
        return self._parent._client

    # ──────────────────────────────────────────────────────────────────
    # Discovery + agent-credential bootstrap
    # ──────────────────────────────────────────────────────────────────

    @property
    def credential_info(self) -> VKCredentialInfo:
        """Lazy-loaded discovery info. Cached for 1h; auto-refreshed on a
        401/403 from /decide."""
        if (
            self._cred_info is None
            or time.time() - self._cred_info_fetched_at > _DISCOVERY_TTL_SECONDS
        ):
            self._cred_info = self._fetch_credential_info()
            self._cred_info_fetched_at = time.time()
        return self._cred_info

    def set_agent_credential(self, cred: AgentCredential) -> None:
        """Override the auto-built credential (advanced; usually for tests
        where you want a known-good static token)."""
        self._agent_credential = cred

    @property
    def agent_credential(self) -> Optional[AgentCredential]:
        """Returns the active credential, building one lazily from the
        discovered info if no override was set."""
        if self._agent_credential is not None:
            return self._agent_credential
        info = self.credential_info
        if not info.agent_configured:
            # LLM-only VK - no agent token needed.
            return None
        self._agent_credential = self._build_credential_for(info)
        return self._agent_credential

    # ──────────────────────────────────────────────────────────────────
    # PDP calls
    # ──────────────────────────────────────────────────────────────────

    def decide(self, dc: DelegationContext) -> Decision:
        """Synchronously call the PDP. Returns the Decision object.

        Retries once with fresh discovery info on 401/403 to handle the case
        where a platform admin re-bound the VK mid-process."""
        # Stamp the per-process session id unless the caller set their own, so
        # the run's steps group into one execution server-side.
        if not dc.session_id:
            dc.session_id = self.session_id
        try:
            return self._post_decide(dc)
        except GatewayUnavailable:
            raise
        except DeepIntShieldError:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                self._cred_info = None
                self._agent_credential = None
                return self._post_decide(dc)
            raise

    def register_blueprint(self, manifest: object) -> Optional[str]:
        """Register the agent's declared tool surface (manifest) with the server
        BEFORE the run - the server stores the declared topology for full-graph
        visualization, policy pre-validation, and declared-vs-observed drift.

        Best-effort and NON-fatal: a registration failure (offline gateway, older
        server) must never break the agent, so this never raises. Returns the
        server-assigned blueprint id, or None."""
        try:
            payload = manifest.to_dict() if hasattr(manifest, "to_dict") else manifest
            # Stamp the per-process session + principal so the blueprint binds to
            # the same execution the decisions will carry.
            if isinstance(payload, dict):
                payload.setdefault("session_id", self.session_id)
            resp = self._http.post(
                f"{self.gateway_url}/api/agentic-security/blueprints",
                json=payload,
                headers=self._headers(include_agent_token=False),
            )
            if resp.status_code < 300:
                try:
                    return resp.json().get("blueprint_id")
                except Exception:
                    return None
        except Exception as exc:  # never break the run on a registration hiccup
            log.warning("blueprint registration skipped: %s", exc)
        return None

    def poll_approval(self, decision_id: str) -> Decision:
        """Block waiting for a REQUIRE_APPROVAL decision to be resolved by a
        human. Returns the final Decision (ALLOW or DENY)."""
        deadline = time.time() + self._approval_timeout
        while time.time() < deadline:
            resp = self._http.get(
                f"{self.gateway_url}/api/agentic-security/approvals/{decision_id}",
                headers=self._headers(include_agent_token=False),
            )
            if resp.status_code == 200:
                # Be defensive: a transient/non-JSON body (proxy error page, SPA
                # fallback) must not crash the poll - treat it as "still pending"
                # and keep waiting until the deadline (→ GuardrailApprovalPending).
                try:
                    state = resp.json().get("state", "pending")
                except Exception:  # noqa: BLE001
                    state = "pending"
                if state == "approved":
                    return Decision(verdict=Verdict.ALLOW, decision_id=decision_id)
                if state in ("denied", "expired"):
                    return Decision(
                        verdict=Verdict.DENY,
                        decision_id=decision_id,
                        reason="approval denied",
                    )
            time.sleep(self._approval_interval)
        raise TimeoutError(f"approval timeout after {self._approval_timeout}s")

    # ──────────────────────────────────────────────────────────────────
    # Token injection (consumed by the L1 transport too)
    # ──────────────────────────────────────────────────────────────────

    def agent_token(self) -> Optional[str]:
        """Return a fresh agent token, or None if this VK has no agent
        identity configured / the optional credential dep is missing. Never
        raises - identity is a strengthening signal, not a hard requirement."""
        try:
            cred = self.agent_credential
        except ImportError as exc:
            log.warning("agent credential unavailable: %s", exc)
            return None
        except Exception as exc:  # discovery failure shouldn't break LLM traffic
            log.warning("agent credential discovery failed: %s", exc)
            return None
        if cred is None:
            return None
        try:
            return cred.get_token()
        except Exception as exc:
            log.warning("agent token acquisition failed: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _post_decide(self, dc: DelegationContext) -> Decision:
        payload = dc.model_dump(mode="json")
        resp = self._http.post(
            f"{self.gateway_url}/api/agentic-security/decide",
            json=payload,
            headers=self._headers(include_agent_token=True),
        )
        if resp.status_code >= 500:
            raise GatewayUnavailable(reason=f"http {resp.status_code}")
        resp.raise_for_status()
        return Decision.model_validate(resp.json())

    def _fetch_credential_info(self) -> VKCredentialInfo:
        resp = self._http.get(
            f"{self.gateway_url}/api/agentic-security/vk-credential-info",
            headers=self._headers(include_agent_token=False),
        )
        resp.raise_for_status()
        return VKCredentialInfo.model_validate(resp.json())

    def _build_credential_for(self, info: VKCredentialInfo) -> AgentCredential:
        if info.provider_type == "entra_agent_id":
            from .credentials.entra import EntraAgentCredential

            return EntraAgentCredential(
                authority=info.authority,
                blueprint_client_id=info.blueprint_client_id,
                gateway_audience=info.gateway_audience,
                scopes=info.scopes,
                fic_audience=info.fic_audience,
                exchange_endpoint=info.exchange_endpoint,
            )
        if info.provider_type == "zeroid":
            from .credentials.zeroid import ZeroIDCredential

            return ZeroIDCredential(
                exchange_endpoint=info.exchange_endpoint,
                gateway_audience=info.gateway_audience,
                scopes=info.scopes,
            )
        if info.provider_type == "generic_oidc":
            from .credentials.oidc import OIDCCredential

            return OIDCCredential(
                exchange_endpoint=info.exchange_endpoint,
                client_id=info.blueprint_client_id,
                gateway_audience=info.gateway_audience,
                scopes=info.scopes,
            )
        # Dev shortcut: respect a known-good token from env for offline
        # iteration where the gateway is reachable but Azure isn't.
        if dev_token := os.environ.get("DEEPINTSHIELD_DEV_AGENT_TOKEN", ""):
            return StaticAgentCredential(dev_token)
        raise DeepIntShieldError(
            f"unknown provider_type: {info.provider_type!r}; cannot build credential"
        )

    def _headers(self, *, include_agent_token: bool) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.virtual_key}",
            "Content-Type": "application/json",
            "User-Agent": f"deepintshield-python/{_sdk_version()}",
        }
        if include_agent_token:
            token = self.agent_token()
            if token:
                h["X-Agent-Token"] = token
        return h


def _sdk_version() -> str:
    try:
        from ..version import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0.0.0"


__all__ = ["AgenticEngine"]
