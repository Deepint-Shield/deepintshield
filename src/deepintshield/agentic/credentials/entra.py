"""Entra Agent ID credential - Federated Identity Credential exchange.

Flow per call (only when the cached token is missing or near expiry):

    1. Get a self-token from the local Managed Identity for audience
       ``api://AzureADTokenExchange``. Uses DefaultAzureCredential so the MI
       binding is auto-detected from the host (AKS workload identity, App
       Service identity, VM identity, etc.) - no client_id needs to appear
       in user code.

    2. POST that as the ``client_assertion`` to the blueprint's token
       endpoint with the configured scope. Entra validates the FIC and
       issues an agent token whose audience is the gateway.

    3. Cache the token; refresh ~5 min before expiry.

``azure-identity`` is an optional install (``pip install
deepintshield[azure]``). If the import fails we surface a clear error
pointing at the install command rather than letting Python die with a
generic ImportError on first use.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from .base import _CachedToken

log = logging.getLogger(__name__)

try:
    from azure.identity import DefaultAzureCredential
    _AZURE_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when extra is missing
    DefaultAzureCredential = None  # type: ignore[assignment]
    _AZURE_AVAILABLE = False


class EntraAgentCredential:
    """Concrete AgentCredential for Microsoft Entra Agent ID blueprints.

    Constructed via :class:`AgenticEngine` from the values discovered at
    GET /api/agentic-security/vk-credential-info - developers should not
    instantiate this directly except for overrides.
    """

    def __init__(
        self,
        authority: str,
        blueprint_client_id: str,
        gateway_audience: str,
        scopes: List[str],
        fic_audience: str = "api://AzureADTokenExchange",
        exchange_endpoint: str = "",
        mi_credential: Optional[object] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not _AZURE_AVAILABLE and mi_credential is None:
            raise ImportError(
                "azure-identity is required for EntraAgentCredential. "
                "Install with: pip install deepintshield[azure]"
            )
        self._authority = authority.rstrip("/")
        self._blueprint_client_id = blueprint_client_id
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
        cached = self._cache.get()
        if cached:
            return cached
        with self._cache.lock():
            # Re-check inside lock - another thread may have refreshed.
            cached = self._cache.get()
            if cached:
                return cached
            token, ttl = self._exchange()
            self._cache.set(token, ttl)
            return token

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _exchange(self) -> tuple[str, float]:
        """Perform the FIC exchange and return (token, ttl_seconds)."""
        log.debug("entra FIC exchange to %s", self._exchange_endpoint)
        # 1. Get the MI self-token (the client_assertion).
        mi_token = self._get_mi_token()

        # 2. POST to Entra's token endpoint.
        data = {
            "client_id": self._blueprint_client_id,
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": mi_token,
            "grant_type": "client_credentials",
            "scope": f"{self._gateway_audience}/.default",
        }
        resp = self._http.post(
            self._exchange_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Entra FIC exchange failed (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        body = resp.json()
        token = body["access_token"]
        ttl = float(body.get("expires_in", 3600))
        return token, ttl

    def _get_mi_token(self) -> str:
        """Ask the local Managed Identity for a self-token to the FIC
        audience. The Azure SDK handles the IMDS roundtrip + caching."""
        result = self._mi_credential.get_token(f"{self._fic_audience}/.default")
        return result.token
