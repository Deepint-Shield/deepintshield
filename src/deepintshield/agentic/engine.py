"""AgenticEngine - the low-level Agentic (PDP) transport/control primitive.

Applications use ``shield.agentic`` or a native framework adapter. Those
supported boundaries translate the engine's internal typed failures into
marked built-in Python exceptions with trusted details. Direct engine use and
its code-only typed errors remain available for SDK internals and backward
compatibility, but are not the supported application error contract.

Unlike the standalone ``deepintshield_agents`` client it was migrated from,
the engine does not take its own ``gateway_url``/``virtual_key`` - it binds
to the parent :class:`~deepintshield.client.DeepintShield` and reuses its
``base_url``, ``virtual_key`` and shared httpx client. The user therefore
passes *only* virtual key + base URL to ``DeepintShield`` and everything
agentic (discovery, credential selection, agent-token minting) happens
automatically.

What it does at decide() time:
    1. POST /api/agentic-new/decide for the authoritative OpenFGA/GAF verdict.
    2. Return that verdict directly. The SDK never asks a hidden second PDP to
       replace or tighten canonical evidence after GAF succeeds.
    3. Only when both canonical Agentic-New routes are genuinely absent (an
       older gateway returning 404/405/501) use the legacy PDP as an explicitly
       marked compatibility source. Canonical errors always fail closed.
    4. Authorization calls use the authenticated key plus the Agentic selector
       and, when configured, workload proof:
       - Authorization: Bearer <virtual_key>  (platform VK)
       - X-Agent-Subject: agent:<registry-key> (untrusted profile selector)
       - X-Agent-Token: <fresh agent token from the credential>, when configured
    5. Return the Decision; the gate translates blocking verdicts
       into exceptions.

Discovery + credential build are lazy (first decide(), not at construction)
so building a ``DeepintShield`` never performs network I/O.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import threading
import time
import uuid
import weakref
from typing import TYPE_CHECKING, Optional

import httpx

from ..errors import DeepintShieldError as SDKConfigurationError
from .actions import infer_action_class
from .credentials.base import AgentCredential, StaticAgentCredential
from .errors import (
    AGENT_APPROVAL_PENDING,
    AGENT_CONFIGURATION_ERROR,
    AGENT_GATEWAY_UNAVAILABLE,
    AGENT_INVALID_GATEWAY_RESPONSE,
    DeepIntShieldError,
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    normalize_agentic_error_code,
)
from .identity import PrincipalBinding
from .registry import _registry_key as _manifest_registry_key
from .types import Decision, DelegationContext, GAFDecision, Verdict, VKCredentialInfo

if TYPE_CHECKING:
    from ..client import DeepintShield

log = logging.getLogger(__name__)
_PRINCIPAL_UNSET = object()

_DISCOVERY_TTL_SECONDS = 3600.0
# Several execution/identity consumers may request the same credential binding
# concurrently during one gated call. Cache only fail-closed configuration
# codes for a very short window so a pending registration produces one gateway
# read, while an operator approval becomes visible on the next normal retry.
_CREDENTIAL_FAILURE_TTL_SECONDS = 0.25
# Blueprint transport probing stays bounded so a genuinely absent/older gateway
# cannot freeze ``compile()``. Once a modern server answers authoritatively,
# coverage and scanner failures are governance decisions and fail closed.
_BLUEPRINT_TIMEOUT_SECONDS = 5.0
_BLUEPRINT_CONFIGURATION_CODES = frozenset(
    {
        "invalid_blueprint_manifest",
        "blueprint_manifest_too_large",
        "blueprint_coverage_incomplete",
        "blueprint_remote_tool_unverified",
        "blueprint_mcp_inventory_unavailable",
        "blueprint_model_unavailable",
        "blueprint_registration_failed",
        "blueprint_scan_unavailable",
    }
)
_ENDPOINT_PROBE_TTL_SECONDS = 60.0
_GAF_FIELDS = frozenset(
    {
        "agent",
        "user",
        "permission",
        "object",
        "delegation_id",
        "action",
        "action_class",
    }
)
class AgenticEngine:
    """Low-level gateway runtime for the canonical Agentic-New GAF data plane.

    GAF is the sole verdict whenever available. The legacy PDP is reachable
    only as an older-gateway compatibility fallback and never participates in
    a successful canonical decision. Applications normally reach this through
    ``shield.agentic`` so internal typed errors are translated to built-ins.
    """

    def __init__(
        self,
        parent: "DeepintShield",
        *,
        agent_credential: Optional[AgentCredential] = None,
        approval_poll_timeout_seconds: float = 300.0,
        approval_poll_interval_seconds: float = 2.0,
    ) -> None:
        # Avoid a client → surface → engine → client reference cycle. Framework
        # objects retain their concrete selected engine, while connection access
        # fails closed if the owning SDK client has been released.
        self._parent_ref = weakref.ref(parent)
        self._cred_info: Optional[VKCredentialInfo] = None
        self._cred_info_fetched_at: float = 0.0
        self._cred_info_failure_code = ""
        self._cred_info_failure_at = 0.0
        self._agent_credential: Optional[AgentCredential] = agent_credential
        # Credential discovery/build is lazy and shared by every gated call.
        # An RLock makes the first fetch/build single-flight under concurrent
        # agent invocations while allowing agent_credential → credential_info.
        self._credential_lock = threading.RLock()
        # A DeepintShield client represents one agent identity, even when
        # several clients share the same virtual key.  Snapshot the selector
        # from the existing per-client agent_name so credential discovery,
        # token acquisition, and every later Agentic-New call cannot drift to
        # another binding if the parent attribute is mutated mid-run.
        raw_agent_name = str(getattr(parent, "agent_name", "") or "").strip()
        agent_key = _registry_key(raw_agent_name) if raw_agent_name else ""
        self._agent_subject_selector = f"agent:{agent_key}" if agent_key else ""
        requester = str(getattr(parent, "requester", "") or "").strip()
        self._requester_hint = "" if requester == "sdk-user" else requester
        self._default_principal: Optional[PrincipalBinding] = None
        self._default_principal_lock = threading.Lock()
        self._approval_timeout = approval_poll_timeout_seconds
        self._approval_interval = approval_poll_interval_seconds
        # ContextVars make one long-lived SDK client safe to share across async
        # requests and worker threads.  The stable default preserves the old
        # per-client session grouping; ``surface.run()`` / ``start_run()`` binds
        # a request-local run id without mutating other callers.
        self._default_session_id = self.new_session_id()
        self._session_context: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"deepintshield_session_{id(self)}", default=None
            )
        )
        self._principal_context: contextvars.ContextVar[object] = (
            contextvars.ContextVar(
                f"deepintshield_principal_{id(self)}", default=_PRINCIPAL_UNSET
            )
        )
        # Discovery dedupe is scoped to this client instance. Identical graphs
        # owned by another VK/client must still register independently.
        self.registry_scope_id = uuid.uuid4().hex
        # Native framework run/compile reporters normally complete discovery
        # before authorization. The shared gate uses these fields for a
        # single-flight minimal fallback when a framework version exposes only
        # its final tool boundary.
        self._registration_capture_lock = threading.Lock()
        self._registration_capture_ready = False
        self._registration_captured_tools: set[str] = set()
        self._registration_captured_tool_attestations: dict[str, str] = {}
        self._registration_minimal_tools: set[str] = set()
        self._registration_capture_retry_at = 0.0
        self._registration_capture_retry_by_tool: dict[str, float] = {}
        self._registration_capture_error_code = ""
        self._gaf_unavailable_until = 0.0
        self._legacy_unavailable_until = 0.0
        # None until credential discovery runs, False only when the canonical
        # route explicitly returned a compatibility status (404/405/501).
        # This prevents a missing decide handler on a modern/GAF-aware gateway
        # from silently falling through to the retired PDP.
        self._canonical_credential_info_available: Optional[bool] = None

    @staticmethod
    def new_session_id() -> str:
        return "sdk-" + uuid.uuid4().hex[:12]

    @property
    def session_id(self) -> str:
        return self._session_context.get() or self._default_session_id

    @property
    def bound_session_id(self) -> str:
        """The request-local session override, excluding the client fallback.

        Automatic framework execution uses this to preserve an explicit
        ``shield.agentic.run(session_id=...)`` while still assigning a fresh id
        to every ordinary top-level framework invocation.
        """
        return str(self._session_context.get() or "")

    @property
    def execution_id(self) -> str:
        """Current first-class workflow execution id, when inside a run."""
        from .execution import current_execution_id

        return current_execution_id(self)

    @session_id.setter
    def session_id(self, value: str) -> None:
        clean = str(value or "").strip()
        self._session_context.set(clean or None)

    def bind_session(self, session_id: str = ""):
        """Bind a session in the current context and return a reset token."""
        clean = str(session_id or "").strip() or self.new_session_id()
        return self._session_context.set(clean)

    def reset_session(self, token: object) -> None:
        self._session_context.reset(token)  # type: ignore[arg-type]

    @property
    def principal_binding(self) -> Optional[PrincipalBinding]:
        scoped = self._principal_context.get()
        if scoped is not _PRINCIPAL_UNSET:
            return scoped if isinstance(scoped, PrincipalBinding) else None
        return self._requester_principal()

    def _requester_principal(self) -> Optional[PrincipalBinding]:
        """Lazily resolve ``DeepintShield(requester=...)`` on first gated use."""
        if not self._requester_hint:
            return None
        if self._default_principal is not None:
            return self._default_principal
        with self._default_principal_lock:
            if self._default_principal is not None:
                return self._default_principal
            from .identity import resolve_principal

            hint = self._requester_hint
            email = hint if "@" in hint else None
            username = None if email else hint
            subject = resolve_principal(self, email=email, username=username)
            self._default_principal = PrincipalBinding(
                subject=subject,
                email=(email or "").lower(),
                username=username or "",
                display_name=hint,
                kind="user",
            )
            return self._default_principal

    def bind_principal(self, binding: Optional[PrincipalBinding]):
        """Bind an acting principal in the current context; return reset token."""
        return self._principal_context.set(binding)

    def reset_principal(self, token: object) -> None:
        self._principal_context.reset(token)  # type: ignore[arg-type]

    @property
    def acting_user(self) -> str:
        binding = self.principal_binding
        return binding.subject if binding is not None else ""

    @acting_user.setter
    def acting_user(self, value: str) -> None:
        """Backward-compatible subject setter, now request-local."""
        subject = str(value or "").strip()
        self._principal_context.set(
            PrincipalBinding(subject=subject) if subject else None
        )

    # ──────────────────────────────────────────────────────────────────
    # Parent-backed connection details
    # ──────────────────────────────────────────────────────────────────

    def _parent(self) -> "DeepintShield":
        parent = self._parent_ref()
        if parent is None:
            raise GatewayUnavailable(reason="DeepintShield client has been released")
        return parent

    @property
    def gateway_url(self) -> str:
        return self._parent().base_url

    @property
    def virtual_key(self) -> str:
        parent = self._parent()
        try:
            return parent.virtual_key_or_raise()
        except SDKConfigurationError:
            raise GovernanceConfigurationError(
                framework="agentic",
                reason="virtual key is missing",
                code="virtual_key_missing",
            ) from None

    @property
    def _http(self) -> httpx.Client:
        return self._parent()._client

    # ──────────────────────────────────────────────────────────────────
    # Discovery + agent-credential bootstrap
    # ──────────────────────────────────────────────────────────────────

    @property
    def credential_info(self) -> VKCredentialInfo:
        """Lazy-loaded discovery info. Cached for 1h; auto-refreshed on a
        401/403 from /decide."""
        failure_now = time.monotonic()
        if (
            self._cred_info_failure_code
            and failure_now - self._cred_info_failure_at
            <= _CREDENTIAL_FAILURE_TTL_SECONDS
        ):
            raise GovernanceConfigurationError(
                code=self._cred_info_failure_code,
            ) from None
        now = time.time()
        if (
            self._cred_info is not None
            and now - self._cred_info_fetched_at <= _DISCOVERY_TTL_SECONDS
        ):
            return self._cred_info
        with self._credential_lock:
            failure_now = time.monotonic()
            if (
                self._cred_info_failure_code
                and failure_now - self._cred_info_failure_at
                <= _CREDENTIAL_FAILURE_TTL_SECONDS
            ):
                raise GovernanceConfigurationError(
                    code=self._cred_info_failure_code,
                ) from None
            now = time.time()
            if (
                self._cred_info is None
                or now - self._cred_info_fetched_at > _DISCOVERY_TTL_SECONDS
            ):
                try:
                    fetched = self._fetch_credential_info()
                except GovernanceConfigurationError as exc:
                    self._cred_info_failure_code = exc.code
                    self._cred_info_failure_at = time.monotonic()
                    raise
                self._cred_info = fetched
                self._cred_info_fetched_at = time.time()
                self._cred_info_failure_code = ""
                self._cred_info_failure_at = 0.0
            return self._cred_info

    def set_agent_credential(self, cred: AgentCredential) -> None:
        """Override the auto-built credential (advanced; usually for tests
        where you want a known-good static token)."""
        with self._credential_lock:
            self._agent_credential = cred

    def _clear_credential_failure_cache(self) -> None:
        """Forget a short-lived fail-closed lookup after discovery changed state."""
        with self._credential_lock:
            self._cred_info_failure_code = ""
            self._cred_info_failure_at = 0.0

    @property
    def agent_subject(self) -> str:
        """Canonical principal selected from this VK's Registry associations.

        The selector header asks the gateway for this client's Registry profile,
        but the returned credential-info value remains authoritative. Older
        gateways omit the field; GAF requests then send an empty claim so the
        server can fill it from the authenticated virtual-key context.
        """
        return str(self.credential_info.agent_subject or "").strip()

    @property
    def agent_credential(self) -> Optional[AgentCredential]:
        """Returns the active credential, building one lazily from the
        discovered info if no override was set."""
        if self._agent_credential is not None:
            return self._agent_credential
        with self._credential_lock:
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
        """Synchronously obtain the canonical Agentic-New authorization result.

        Retries once with fresh discovery info on 401/403 to handle a Registry
        profile or credential change made mid-process. A legacy decision is
        accepted only from a gateway that does not implement either canonical
        Agentic-New route."""
        # Stamp the request-local session unless the caller supplied one.
        if not dc.session_id:
            dc.session_id = self.session_id
        for attempt in range(2):
            try:
                return self._decide_once(dc)
            except DeepIntShieldError:
                raise
            except httpx.HTTPStatusError as exc:
                if attempt == 0 and exc.response.status_code in (401, 403):
                    with self._credential_lock:
                        self._cred_info = None
                        self._cred_info_fetched_at = 0.0
                        self._cred_info_failure_code = ""
                        self._cred_info_failure_at = 0.0
                        self._agent_credential = None
                    continue
                self._raise_agentic_http_error(exc.response)
            except httpx.HTTPError as exc:
                raise GatewayUnavailable(reason=type(exc).__name__) from None
            except Exception as exc:
                # Framework applications should never need to understand an
                # httpx/Pydantic/credential implementation exception. Retain
                # the type for diagnostics, but expose only the public code.
                raise GatewayUnavailable(reason=type(exc).__name__) from None
        raise GatewayUnavailable(code=AGENT_GATEWAY_UNAVAILABLE)  # pragma: no cover

    def register_blueprint(
        self,
        manifest: object,
        *,
        timeout: float = _BLUEPRINT_TIMEOUT_SECONDS,
    ) -> Optional[str]:
        """Register the agent's declared tool surface (manifest) with the server
        BEFORE the run - the server stores the declared topology for full-graph
        visualization, policy pre-validation, and declared-vs-observed drift.

        Transport failure and an older gateway without this compatibility route
        remain non-fatal. Any response from an implemented route is
        authoritative: incomplete source coverage, an unverified MCP claim or a
        required scanner failure raises a stable code-only Agentic exception.
        Returns the server-assigned blueprint id, or None for the two legacy
        compatibility cases."""
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
                timeout=timeout,
            )
        except DeepIntShieldError:
            raise
        except httpx.HTTPError:
            # Optional compatibility transport: do not emit exception prose to
            # stderr. The canonical first-run discovery/decision boundary still
            # fails closed independently when the gateway is required.
            log.debug("blueprint_registration_transport_unavailable")
            return None
        except Exception:
            raise GovernanceConfigurationError(
                framework="agentic",
                code="invalid_blueprint_manifest",
            ) from None

        if self._route_not_implemented(resp):
            log.debug("blueprint_registration_route_unavailable")
            return None
        if resp.status_code >= 400:
            code = self._response_error_code(resp)
            if code in _BLUEPRINT_CONFIGURATION_CODES:
                raise self._agentic_configuration_error(resp) from None
            self._raise_agentic_http_error(resp)
        try:
            body = resp.json()
        except Exception:
            raise DeepIntShieldError(code=AGENT_INVALID_GATEWAY_RESPONSE) from None
        blueprint_id = body.get("blueprint_id") if isinstance(body, dict) else None
        if not isinstance(blueprint_id, str) or not blueprint_id.strip():
            raise DeepIntShieldError(code=AGENT_INVALID_GATEWAY_RESPONSE) from None
        return blueprint_id.strip()

    def poll_approval(self, decision_id: str) -> Decision:
        """Poll an approval produced by the older-gateway compatibility PDP.

        Canonical Agentic-New approvals never call this method: they return
        control immediately with their canonical approval id so the workflow
        can persist and resume without a second authorization data source.
        """
        deadline = time.time() + self._approval_timeout
        while time.time() < deadline:
            try:
                resp = self._http.get(
                    f"{self.gateway_url}/api/agentic-security/approvals/{decision_id}",
                    headers=self._headers(include_agent_token=False),
                )
            except DeepIntShieldError:
                raise
            except Exception as exc:
                raise GatewayUnavailable(reason=type(exc).__name__) from None
            if resp.status_code >= 500 or resp.status_code in (400, 401, 403, 409):
                self._raise_agentic_http_error(resp)
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
        raise GuardrailApprovalPending(
            decision_id=decision_id,
            code=AGENT_APPROVAL_PENDING,
        ) from None

    # ──────────────────────────────────────────────────────────────────
    # Token injection (consumed by the L1 transport too)
    # ──────────────────────────────────────────────────────────────────

    def agent_token(self, *, fail_closed: bool = False) -> Optional[str]:
        """Return a fresh agent token, or None if this VK has no agent
        identity configured.

        Transparent LLM transport keeps the historical best-effort behavior.
        Authorization calls pass ``fail_closed=True``: if a VK is configured
        for workload identity, a credential/setup/token failure must block
        before the PDP request rather than silently dropping the agent proof.
        """
        if fail_closed:
            try:
                cred = self.agent_credential
                if cred is None:
                    return None
                token = str(cred.get_token() or "").strip()
                if not token:
                    raise GovernanceConfigurationError(
                        framework="agent-credential",
                        reason="configured credential returned an empty agent token",
                    )
                return token
            except ImportError as exc:
                raise GovernanceConfigurationError(
                    framework="agent-credential",
                    reason=type(exc).__name__,
                ) from None
            except DeepIntShieldError:
                raise
            except Exception as exc:
                raise GatewayUnavailable(
                    reason=type(exc).__name__
                ) from None
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

    def _decide_once(self, dc: DelegationContext) -> Decision:
        """One canonical decision attempt, with an older-gateway fallback.

        Once GAF returns a valid response, its verdict and evidence are final.
        The legacy route is contacted only after the canonical route is absent
        *and* canonical credential discovery already established that this is
        an older gateway.
        """
        gaf_call = self._post_gaf_decide(dc)
        if gaf_call is not None:
            gaf, gaf_decision_id = gaf_call
            return self._decision_from_gaf(gaf, gaf_decision_id)

        legacy = self._post_legacy_compat_decide(dc)
        if legacy is None:
            raise GatewayUnavailable(
                reason=(
                    "canonical Agentic-New is not implemented by this older "
                    "gateway and its compatibility PDP is unavailable"
                )
            )
        proceed = (
            legacy.proceed
            if legacy.proceed is not None
            else legacy.verdict in (Verdict.ALLOW, Verdict.MASK)
        )
        return legacy.model_copy(
            update={
                # Preserve truthful evidence: no GAF verdict exists on this
                # compatibility path, and callers can identify it explicitly.
                "gaf_checked": False,
                "gaf_verdict": "",
                "mode": "legacy-compat",
                "proceed": bool(proceed),
            }
        )

    def _post_gaf_decide(
        self, dc: DelegationContext
    ) -> Optional[tuple[GAFDecision, str]]:
        if time.monotonic() < self._gaf_unavailable_until:
            return None
        payload = self._gaf_payload(dc)
        resp = self._http.post(
            f"{self.gateway_url}/api/agentic-new/decide",
            json=payload,
            headers=self._headers(include_agent_token=True),
        )
        if self._route_not_implemented(resp):
            # _gaf_payload() normally runs credential discovery first. The
            # explicit check also covers callers that supplied both a custom
            # agent principal and credential override.
            if self._canonical_credential_info_available is None:
                _ = self.credential_info
            if self._canonical_credential_info_available is not False:
                raise GovernanceConfigurationError(
                    framework="agentic-new",
                    reason=(
                        "the canonical credential-info route is available but "
                        "the canonical decision route is missing"
                    ),
                )
            self._gaf_unavailable_until = (
                time.monotonic() + _ENDPOINT_PROBE_TTL_SECONDS
            )
            return None
        if self._endpoint_absent(resp, "GAF PDP"):
            # An explicitly unconfigured GAF route is a modern gateway
            # configuration error, not evidence of an older gateway.
            raise GovernanceConfigurationError(
                framework="agentic-new",
                reason="the canonical Agentic-New PDP is not configured",
            )
        if resp.status_code >= 500:
            self._raise_agentic_http_error(resp)
        resp.raise_for_status()
        try:
            decision = GAFDecision.model_validate(resp.json())
        except Exception as exc:
            raise DeepIntShieldError(
                type(exc).__name__,
                code=AGENT_INVALID_GATEWAY_RESPONSE,
            ) from None
        decision_id = (
            decision.decision_id
            or decision.approval_id
            or resp.headers.get("x-request-id", "").strip()
            or "gaf-" + uuid.uuid4().hex[:16]
        )
        return decision, decision_id

    def _post_legacy_compat_decide(
        self, dc: DelegationContext
    ) -> Optional[Decision]:
        """Call the retired PDP only for a proven older-gateway fallback."""
        if self._canonical_credential_info_available is not False:
            raise GovernanceConfigurationError(
                framework="agentic-new",
                reason=(
                    "legacy authorization fallback was refused because this "
                    "gateway exposes canonical Agentic-New capabilities"
                ),
            )
        if time.monotonic() < self._legacy_unavailable_until:
            return None
        # Agentic-New-only fields are never sent to the legacy Go contract.
        payload = dc.model_dump(mode="json", exclude=_GAF_FIELDS)
        resp = self._http.post(
            f"{self.gateway_url}/api/agentic-security/decide",
            json=payload,
            headers=self._headers(include_agent_token=True),
        )
        if self._route_not_implemented(resp) or self._endpoint_absent(
            resp, "legacy PDP"
        ):
            self._legacy_unavailable_until = (
                time.monotonic() + _ENDPOINT_PROBE_TTL_SECONDS
            )
            return None
        if resp.status_code >= 500:
            self._raise_agentic_http_error(resp)
        resp.raise_for_status()
        try:
            return Decision.model_validate(resp.json())
        except Exception as exc:
            raise DeepIntShieldError(
                type(exc).__name__,
                code=AGENT_INVALID_GATEWAY_RESPONSE,
            ) from None

    # Backward-compatible private name, now preserving canonical semantics
    # instead of exposing a hidden legacy authorization path.
    def _post_decide(self, dc: DelegationContext) -> Decision:
        return self._decide_once(dc)

    def _gaf_payload(self, dc: DelegationContext) -> dict[str, str]:
        raw_tool = str(dc.tool or "").strip()
        tool_name = (
            raw_tool.split(":", 1)[1]
            if raw_tool.lower().startswith("tool:")
            else raw_tool
        )
        tool_key = _registry_key(tool_name)
        action = str(dc.action or "").strip()
        action_class = (
            str(dc.action_class or "").strip().lower()
            # A named operation is more specific than its containing tool. This
            # matters for multi-operation tools such as read_records/delete:
            # without an explicit class, the destructive action must select the
            # write permission/HITL path rather than inherit the tool's read name.
            or _action_class(action or tool_key)
        )
        # Only an explicit GAF claim or the server-selected Registry principal is
        # sent. The Go data plane rejects subjects that do not match a persisted
        # association and fills an empty claim when selection is unambiguous.
        agent = str(dc.agent or "").strip() or self.agent_subject

        user = str(dc.user or "").strip() or self.acting_user
        if not user:
            user = next(
                (
                    str(subject).strip()
                    for subject in dc.actor_chain
                    if str(subject).strip().startswith(("user:", "service_account:"))
                ),
                "",
            )

        permission = str(dc.permission or "").strip() or "holder"
        object_id = str(dc.object or "").strip()
        if not object_id:
            # Safe, deterministic default: an operator must explicitly grant
            # permission key "<tool>:<read|write>" in Organization → Permissions.
            # Match Go PermissionSubject exactly. Missing tuples deny.
            object_id = _permission_subject(f"{tool_key}:{action_class}")
        tool_subject = f"tool:{tool_key}"
        return {
            "agent": agent,
            "user": user,
            "permission": permission,
            "object": object_id,
            "tool": tool_subject,
            # Empty is meaningful only when exactly one registered action
            # exists: the server selects that sole action. Actionless tools are
            # denied; discovery always supplies a conservative `invoke` action.
            "action": action,
            "delegation_id": str(dc.delegation_id or "").strip(),
            "action_class": action_class,
            "args_digest": str(dc.args_digest or "").strip(),
            # One ordinary framework invocation owns one execution/session.
            # The server also accepts these values in headers, but carrying
            # them in the body keeps MockTransport/custom transports and audit
            # replay deterministic.
            "execution_id": self.execution_id or str(dc.session_id or "").strip(),
            "session_id": str(dc.session_id or self.session_id).strip(),
        }

    @staticmethod
    def _decision_from_gaf(gaf: GAFDecision, decision_id: str) -> Decision:
        proceed = gaf.proceed if gaf.proceed is not None else bool(gaf.allow)
        if proceed:
            effective_verdict = Verdict.ALLOW
        elif gaf.verdict == Verdict.REQUIRE_APPROVAL:
            effective_verdict = Verdict.REQUIRE_APPROVAL
        else:
            # ``proceed`` is the authoritative execution bit. A malformed or
            # forward-version response such as verdict=ALLOW/proceed=false must
            # not pass through merely because the enum looks non-blocking.
            effective_verdict = Verdict.DENY
        return Decision(
            verdict=effective_verdict,
            reason=gaf.reason,
            decision_id=decision_id,
            mode=gaf.mode,
            latency_us=gaf.latency_us,
            gaf_checked=True,
            gaf_verdict=gaf.verdict.value,
            failed_check=gaf.failed_check,
            checks=gaf.checks,
            proceed=proceed,
            would_block=gaf.would_block,
            approval_id=gaf.approval_id,
        )

    @staticmethod
    def _route_not_implemented(resp: httpx.Response) -> bool:
        """True only for statuses that identify an older route surface."""
        return resp.status_code in (404, 405, 501)

    @staticmethod
    def _response_error_detail(resp: httpx.Response) -> str:
        """Extract a bounded message from flat or DeepIntShield error envelopes."""

        def find_message(value: object, seen: set[int]) -> str:
            if isinstance(value, str):
                return value.strip()
            if not isinstance(value, dict) or id(value) in seen:
                return ""
            seen.add(id(value))
            for key in ("message", "detail", "error"):
                message = find_message(value.get(key), seen)
                if message:
                    return message
            return ""

        try:
            detail = find_message(resp.json(), set())
        except Exception:
            detail = resp.text.strip()
        # Do not let an intermediary HTML page or oversized upstream error flood
        # exception logs. Agentic server messages are intentionally short.
        return detail[:500]

    @staticmethod
    def _response_error_code(resp: httpx.Response) -> str:
        """Read only documented public-code envelope fields.

        Arbitrary nested objects may contain third-party ``code`` attributes;
        recursively promoting those values would turn payload data into the
        SDK's public exception contract.
        """
        try:
            body = resp.json()
        except Exception:
            return ""
        if not isinstance(body, dict):
            return ""
        error = body.get("error")
        if isinstance(error, dict):
            code = normalize_agentic_error_code(error.get("code"), "")
            if code:
                return code
        return normalize_agentic_error_code(body.get("code"), "")

    def _agentic_configuration_error(
        self, resp: httpx.Response
    ) -> GovernanceConfigurationError:
        """Preserve the server code and leave human remediation to Agentic UI."""
        detail = self._response_error_detail(resp)
        return GovernanceConfigurationError(
            framework="agentic-new",
            reason=detail,
            code=self._response_error_code(resp) or AGENT_CONFIGURATION_ERROR,
            status_code=resp.status_code,
            payload=self._response_json_object(resp),
        )

    def _raise_agentic_http_error(self, resp: httpx.Response) -> None:
        """Translate every Agentic HTTP failure to the public code contract."""
        if resp.status_code < 500:
            raise self._agentic_configuration_error(resp) from None
        raise GatewayUnavailable(
            reason=self._response_error_detail(resp),
            code=self._response_error_code(resp) or AGENT_GATEWAY_UNAVAILABLE,
            status_code=resp.status_code,
            payload=self._response_json_object(resp),
        ) from None

    @staticmethod
    def _response_json_object(resp: httpx.Response) -> dict:
        try:
            body = resp.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _endpoint_absent(resp: httpx.Response, label: str) -> bool:
        if resp.status_code != 503:
            return False
        # The Agentic-New handler uses 503 to say that no OpenFGA store is
        # configured. A generic 503 is an outage and must fail closed.
        try:
            body = resp.json()
            detail = " ".join(
                str(body.get(k, "")) for k in ("error", "message", "detail")
            )
        except Exception:
            detail = resp.text
        absent = "not enabled" in detail.lower() or "not configured" in detail.lower()
        if absent:
            log.debug("%s unavailable: %s", label, detail)
        return absent

    def _fetch_credential_info(self) -> VKCredentialInfo:
        # Agentic-New is the canonical identity-provider store. Keep the old
        # discovery route as a one-release compatibility fallback so an updated
        # SDK can still run against an older gateway, but never fall back after
        # an authentication/configuration failure (403/409/5xx): those must stay
        # fail closed.
        try:
            resp = self._http.get(
                f"{self.gateway_url}/api/agentic-new/credential-info",
                headers=self._headers(include_agent_token=False),
            )
        except DeepIntShieldError:
            raise
        except Exception as exc:
            raise GatewayUnavailable(reason=type(exc).__name__) from None
        canonical_route = not self._route_not_implemented(resp)
        if not canonical_route:
            self._canonical_credential_info_available = False
            try:
                resp = self._http.get(
                    f"{self.gateway_url}/api/agentic-security/vk-credential-info",
                    headers=self._headers(include_agent_token=False),
                )
            except DeepIntShieldError:
                raise
            except Exception as exc:
                raise GatewayUnavailable(reason=type(exc).__name__) from None
        else:
            # Even an auth/config error proves the modern route exists; record
            # this before raise_for_status so decide can never downgrade.
            self._canonical_credential_info_available = True
        if resp.status_code >= 400:
            self._raise_agentic_http_error(resp)
        try:
            return VKCredentialInfo.model_validate(resp.json())
        except Exception as exc:
            raise DeepIntShieldError(
                type(exc).__name__,
                code=AGENT_INVALID_GATEWAY_RESPONSE,
            ) from None

    def _build_credential_for(self, info: VKCredentialInfo) -> AgentCredential:
        if info.provider_type == "entra_agent_id":
            from .credentials.entra import EntraAgentCredential

            return EntraAgentCredential(
                authority=info.authority,
                blueprint_client_id=info.blueprint_client_id,
                agent_identity_client_id=info.agent_identity_client_id,
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
        raise GovernanceConfigurationError(
            framework="agent-credential",
            reason=f"unknown provider_type: {info.provider_type!r}",
            code="credential_provider_unsupported",
        )

    def _headers(self, *, include_agent_token: bool) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.virtual_key}",
            "Content-Type": "application/json",
            "User-Agent": f"deepintshield-python/{_sdk_version()}",
        }
        if self._agent_subject_selector:
            h["X-Agent-Subject"] = self._agent_subject_selector
        if include_agent_token:
            token = self.agent_token(fail_closed=True)
            if token:
                h["X-Agent-Token"] = token
        return h


def _registry_key(value: str) -> str:
    """Match Go ``AgenticNewSanitizeKey`` for OpenFGA agent/tool ids."""
    return _manifest_registry_key(value) or "tool"


def _permission_slug(value: str) -> str:
    """Match Go ``AgenticNewSlugify`` used by ``PermissionSubject``."""
    out: list[str] = []
    last_dash = False
    for char in str(value or "").strip().lower():
        if "a" <= char <= "z" or "0" <= char <= "9":
            out.append(char)
            last_dash = False
        elif char in "_-.:":
            out.append("-")
            last_dash = True
        elif char in " \t/\\" and not last_dash and out:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "tool-read"


def _permission_subject(value: str) -> str:
    """Match Go's versioned, collision-resistant ``PermissionSubject``."""
    canonical = str(value or "").strip().lower()
    readable = _permission_slug(canonical)
    readable = readable[:64].rstrip("-._") or readable[:64]

    payload = bytearray(b"deepintshield/openfga/object-id/v1")
    for component in ("permission", canonical):
        encoded = component.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    suffix = hashlib.sha256(payload).hexdigest()[:32]
    return f"permission:v1-{readable}--{suffix}"


def _action_class(tool: str) -> str:
    return infer_action_class(tool)


def _sdk_version() -> str:
    try:
        from ..version import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0.0.0"


__all__ = ["AgenticEngine"]
