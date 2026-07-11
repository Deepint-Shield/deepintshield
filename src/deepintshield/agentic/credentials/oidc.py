"""Generic OIDC credential - Client-Credentials grant against any OIDC IdP
that publishes a JWKS endpoint (Okta, Auth0, Keycloak, …).

Used when the platform VK is bound to a ``generic_oidc`` identity provider on
the gateway. The shape is intentionally similar to the Entra adapter so
swap-in is mechanical.
"""

from __future__ import annotations

import os
from typing import List, Optional

import httpx

from .base import _CachedToken


class OIDCCredential:
    """OIDC client-credentials credential."""

    def __init__(
        self,
        exchange_endpoint: str,
        client_id: str,
        client_secret_env: str = "OIDC_CLIENT_SECRET",
        gateway_audience: str = "",
        scopes: Optional[List[str]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._endpoint = exchange_endpoint
        self._client_id = client_id
        self._client_secret_env = client_secret_env
        self._audience = gateway_audience
        self._scopes = list(scopes) if scopes else []
        self._http = http_client or httpx.Client(timeout=10.0)
        self._cache = _CachedToken()

    @property
    def provider_type(self) -> str:
        return "generic_oidc"

    def get_token(self) -> str:
        cached = self._cache.get()
        if cached:
            return cached
        with self._cache.lock():
            cached = self._cache.get()
            if cached:
                return cached
            token, ttl = self._exchange()
            self._cache.set(token, ttl)
            return token

    def _exchange(self) -> tuple[str, float]:
        secret = os.environ.get(self._client_secret_env, "")
        if not secret:
            raise RuntimeError(
                f"OIDC client secret not found in env {self._client_secret_env}"
            )
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": secret,
            "scope": " ".join(self._scopes) if self._scopes else "",
        }
        if self._audience:
            data["audience"] = self._audience
        resp = self._http.post(self._endpoint, data=data)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OIDC exchange failed (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        body = resp.json()
        return body["access_token"], float(body.get("expires_in", 3600))
