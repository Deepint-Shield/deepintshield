"""Entra Agent ID credential - Federated Identity Credential exchange.

Flow per call (only when the cached token is missing or near expiry):

    1. Get a self-token from the local Managed Identity for audience
       ``api://AzureADTokenExchange``. Uses DefaultAzureCredential so the MI
       binding is auto-detected from the host (AKS workload identity, App
       Service identity, VM identity, etc.) - no client_id needs to appear
       in user code.

    2. POST that as the ``client_assertion`` for the blueprint, with
       ``fmi_path`` naming the child Agent Identity. Entra validates the FIC
       and returns an intermediate Agent ID assertion (T1).

    3. Exchange T1 as the child identity for the gateway resource token (T2).
       Cache T2 and refresh ~5 min before expiry.

``azure-identity`` is an optional install (``pip install
deepintshield[azure]``). If the import fails we surface a clear error
pointing at the install command rather than letting Python die with a
generic ImportError on first use.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from ..errors import (
    DeepIntShieldError,
    GatewayUnavailable,
    GovernanceConfigurationError,
)
from .base import _CachedToken

log = logging.getLogger(__name__)

_ENTRA_TOKEN_EXCHANGE_SCOPE = "api://AzureADTokenExchange/.default"

try:
    from azure.identity import DefaultAzureCredential
    _AZURE_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when extra is missing
    DefaultAzureCredential = None  # type: ignore[assignment]
    _AZURE_AVAILABLE = False


class EntraAgentCredential:
    """Concrete AgentCredential for Microsoft Entra Agent ID blueprints.

    Constructed via :class:`AgenticEngine` from the values discovered at
    ``GET /api/agentic-new/credential-info`` - developers should not instantiate
    this directly except for overrides.
    """

    def __init__(
        self,
        authority: str,
        blueprint_client_id: str,
        agent_identity_client_id: str,
        gateway_audience: str,
        scopes: List[str],
        fic_audience: str = "api://AzureADTokenExchange",
        exchange_endpoint: str = "",
        mi_credential: Optional[object] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not _AZURE_AVAILABLE and mi_credential is None:
            raise GovernanceConfigurationError(
                framework="agent-credential",
                reason="azure-identity dependency is unavailable",
                code="credential_dependency_missing",
            ) from None
        self._authority = authority.rstrip("/")
        self._blueprint_client_id = blueprint_client_id
        self._agent_identity_client_id = agent_identity_client_id
        self._gateway_audience = gateway_audience
        self._scopes = list(scopes) or ["tools:invoke"]
        self._fic_audience = fic_audience
        self._exchange_endpoint = (
            exchange_endpoint or f"{self._authority}/oauth2/v2.0/token"
        )
        # DefaultAzureCredential discovers the MI from the host (AKS workload
        # identity, App Service MI, VM-MI via IMDS, env vars, az cli for local
        # dev). The user almost never needs to override.
        self._mi_credential = mi_credential or DefaultAzureCredential()
        self._http = http_client or httpx.Client(timeout=10.0)
        self._cache = _CachedToken()

    @property
    def provider_type(self) -> str:
        return "entra_agent_id"

    def get_token(self) -> str:
        try:
            return self._cache.get_or_refresh(self._exchange)
        except DeepIntShieldError:
            raise
        except Exception as exc:
            raise GatewayUnavailable(
                reason=type(exc).__name__,
                code="credential_exchange_failed",
            ) from None

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _exchange(self) -> tuple[str, float]:
        """Perform the autonomous Agent ID T1→T2 exchange."""
        if not self._blueprint_client_id or not self._agent_identity_client_id:
            raise GovernanceConfigurationError(
                framework="agent-credential",
                reason="Entra Agent ID identifiers are incomplete",
                code="credential_configuration_error",
            )
        log.debug("entra FIC exchange to %s", self._exchange_endpoint)
        # 1. Get the MI self-token used as the blueprint assertion.
        mi_token = self._get_mi_token()

        # 2. Blueprint + fmi_path -> intermediate child-agent assertion (T1).
        t1_data = {
            "client_id": self._blueprint_client_id,
            "fmi_path": self._agent_identity_client_id,
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": mi_token,
            "grant_type": "client_credentials",
            # This is the Entra Agent ID token-exchange resource, not a
            # provider/downstream scope and therefore is never caller-defined.
            "scope": _ENTRA_TOKEN_EXCHANGE_SCOPE,
        }
        t1_resp = self._http.post(
            self._exchange_endpoint,
            data=t1_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if t1_resp.status_code >= 400:
            raise RuntimeError(
                f"Entra Agent ID T1 exchange failed (HTTP {t1_resp.status_code}): "
                f"{t1_resp.text[:300]}"
            )
        t1 = str(t1_resp.json().get("access_token") or "").strip()
        if not t1:
            raise RuntimeError("Entra Agent ID T1 exchange returned no access_token")

        # 3. Child identity + T1 -> final gateway resource token (T2).
        t2_data = {
            "client_id": self._agent_identity_client_id,
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": t1,
            "grant_type": "client_credentials",
            "scope": self._default_scope(self._gateway_audience),
        }
        t2_resp = self._http.post(
            self._exchange_endpoint,
            data=t2_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if t2_resp.status_code >= 400:
            raise RuntimeError(
                f"Entra Agent ID T2 exchange failed (HTTP {t2_resp.status_code}): "
                f"{t2_resp.text[:300]}"
            )
        body = t2_resp.json()
        token = str(body.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("Entra Agent ID T2 exchange returned no access_token")
        return token, float(body.get("expires_in", 3600))

    @staticmethod
    def _default_scope(resource: str) -> str:
        resource = str(resource or "").strip().rstrip("/")
        if resource.endswith("/.default"):
            return resource
        return f"{resource}/.default"

    def _get_mi_token(self) -> str:
        """Ask the local Managed Identity for a self-token to the FIC
        audience. The Azure SDK handles the IMDS roundtrip + caching."""
        result = self._mi_credential.get_token(self._default_scope(self._fic_audience))
        return result.token
