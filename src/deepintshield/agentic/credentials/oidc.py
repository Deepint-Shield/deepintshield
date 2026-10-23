"""Generic OIDC credential using the client-credentials grant.

Used when the platform VK is bound to a ``generic_oidc`` identity provider on
the gateway. The shape is intentionally similar to the Entra adapter so
swap-in is mechanical.
"""

from __future__ import annotations

import os
import math
from typing import List, Optional
from urllib.parse import urlsplit

import httpx

from ..errors import (
    DeepIntShieldError,
    GatewayUnavailable,
)
from .base import _CachedToken
from .oidc_auth import ClientAuthentication, configuration_error, secure_url


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
        *,
        issuer: str = "",
        require_proof_of_possession: bool = False,
        client_auth_method: str | None = None,
        audience: str | None = None,
        resource: str | None = None,
    ) -> None:
        self._endpoint = exchange_endpoint
        self._client_id = client_id
        self._client_secret_env = client_secret_env
        self._audience = os.environ.get("OIDC_AUDIENCE", gateway_audience) if audience is None else audience
        self._resource = os.environ.get("OIDC_RESOURCE", "") if resource is None else resource
        self._issuer = issuer
        self._scopes = list(scopes) if scopes else []
        self._http = http_client or httpx.Client(timeout=10.0)
        self._cache = _CachedToken()
        self._client_auth = ClientAuthentication(client_id, client_secret_env, client_auth_method)
        self._dpop_exchange = None
        if require_proof_of_possession:
            from .dpop_oauth import BoundTokenExchange

            self._dpop_exchange = BoundTokenExchange(
                self._http, issuer, exchange_endpoint, client_authentication=self._client_auth,
            )

    @property
    def provider_type(self) -> str:
        return "generic_oidc"

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

    def _exchange(self) -> tuple[str, float]:
        data = {"grant_type": "client_credentials"}
        if self._scopes:
            data["scope"] = " ".join(self._scopes)
        if self._audience:
            data["audience"] = self._audience
        if self._resource:
            parsed = urlsplit(self._resource)
            if not parsed.scheme or parsed.fragment:
                raise configuration_error("OIDC resource must be an absolute URI without a fragment")
            data["resource"] = self._resource
        if self._dpop_exchange is not None:
            with self._client_auth.http_client(self._http) as client:
                return self._dpop_exchange.exchange(data, http=client)
        self._ensure_endpoint()
        form, auth = self._client_auth.prepare(self._endpoint, data)
        with self._client_auth.http_client(self._http) as client:
            resp = client.post(self._endpoint, data=form, auth=auth, follow_redirects=False)
        if resp.status_code != 200:
            raise GatewayUnavailable(reason="OIDC issuer rejected the token request", code="credential_exchange_failed")
        try:
            body = resp.json()
            token = body["access_token"]
            raw_ttl = body.get("expires_in", 3600)
            ttl = float(raw_ttl)
            if (not isinstance(token, str) or not token.strip() or not token.isascii()
                    or len(token) > 16384 or any(character.isspace() for character in token)
                    or isinstance(raw_ttl, bool) or not math.isfinite(ttl) or ttl <= 0):
                raise ValueError("invalid token or lifetime")
            # TLS client authentication is supported, but this transport does
            # not present that certificate to the gateway as token proof.
            if str(body.get("token_type", "Bearer")).lower() != "bearer":
                raise ValueError("unexpected token type")
        except (KeyError, TypeError, ValueError, AttributeError):
            raise GatewayUnavailable(reason="OIDC issuer returned an invalid token response", code="credential_exchange_failed") from None
        return token, ttl

    def _ensure_endpoint(self) -> None:
        if self._endpoint:
            secure_url(self._endpoint)
            return
        issuer = secure_url(self._issuer, issuer=True)
        response = self._http.get(
            issuer.rstrip("/") + "/.well-known/openid-configuration", follow_redirects=False,
        )
        if response.status_code != 200:
            raise configuration_error("OIDC issuer discovery failed")
        metadata = response.json()
        if not isinstance(metadata, dict) or metadata.get("issuer") != issuer:
            raise configuration_error("OIDC discovery issuer does not match")
        endpoint = self._client_auth.metadata_endpoint(metadata)
        methods = metadata.get("token_endpoint_auth_methods_supported", ["client_secret_basic"])
        if not isinstance(methods, list) or not all(isinstance(method, str) for method in methods):
            raise configuration_error("OIDC discovery client authentication metadata is invalid")
        grants = metadata.get("grant_types_supported")
        if grants is not None and (not isinstance(grants, list) or "client_credentials" not in grants):
            raise configuration_error("issuer does not support client-credentials grant")
        if not self._client_auth.explicit_method:
            if "client_secret_post" in methods:
                self._client_auth.method = "client_secret_post"
            elif "client_secret_basic" in methods:
                self._client_auth.method = "client_secret_basic"
        if self._client_auth.method not in methods:
            raise configuration_error("issuer does not support configured client authentication")
        self._endpoint = endpoint
