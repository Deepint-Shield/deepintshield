"""RFC 9449 client-credentials acquisition for JWT workload tokens.

Discovery capabilities are hints. A successful response must also declare DPoP
and bind its JWT to the process key. The gateway independently verifies the
issuer signature, audience, claims and proof on every authenticated request.
"""

from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import math
import time
from urllib.parse import urlsplit

import httpx

from ..dpop import process_dpop_key
from ..errors import GatewayUnavailable, GovernanceConfigurationError


def _configuration_error(reason: str) -> GovernanceConfigurationError:
    return GovernanceConfigurationError(
        framework="agent-credential", reason=reason,
        code="credential_dpop_configuration_error",
    )


def _secure_url(value: str, *, allow_query: bool = False) -> str:
    """Require authenticated transport, allowing numeric loopback for local tests."""
    try:
        url = urlsplit(value)
        loopback = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        loopback = False
    url = urlsplit(value)
    if (not url.hostname or url.username or url.password or url.fragment
            or (url.query and not allow_query) or url.scheme not in ("https", "http")
            or (url.scheme != "https" and not loopback)):
        raise _configuration_error("DPoP issuer and token endpoint must use HTTPS")
    return value


class BoundTokenExchange:
    """Used under the credential's single-flight refresh lock, including nonces."""

    def __init__(self, http: httpx.Client, issuer: str, endpoint: str, *, client_authentication=None) -> None:
        self._http = http
        self._issuer = _secure_url(issuer)
        self._endpoint = _secure_url(endpoint, allow_query=True) if endpoint else ""
        self._key = process_dpop_key()
        if self._key is None:
            raise _configuration_error("DPoP requires the cryptography dependency")
        self._metadata_until = 0.0
        self._client_auth = "client_secret_post"
        self._nonce: str | None = None
        self._client_authentication = client_authentication

    def _discover(self) -> None:
        if time.monotonic() < self._metadata_until:
            return
        response = self._http.get(
            self._issuer.rstrip("/") + "/.well-known/openid-configuration",
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise _configuration_error("DPoP requires issuer discovery metadata")
        metadata = response.json()
        if not isinstance(metadata, dict) or metadata.get("issuer") != self._issuer:
            raise _configuration_error("discovery issuer does not match the configured issuer")
        if "ES256" not in metadata.get("dpop_signing_alg_values_supported", []):
            raise _configuration_error("issuer does not advertise the required DPoP algorithm")
        if "client_credentials" not in metadata.get("grant_types_supported", []):
            raise _configuration_error("issuer does not advertise the client-credentials grant")
        endpoint = (
            self._client_authentication.metadata_endpoint(metadata)
            if self._client_authentication is not None
            else _secure_url(metadata.get("token_endpoint", ""), allow_query=True)
        )
        if self._endpoint and endpoint != self._endpoint:
            raise _configuration_error("configured token endpoint differs from issuer discovery")
        self._endpoint = endpoint
        methods = metadata.get("token_endpoint_auth_methods_supported", ["client_secret_basic"])
        if not isinstance(methods, list) or not all(isinstance(method, str) for method in methods):
            raise _configuration_error("issuer returned invalid client authentication metadata")
        explicit_method = getattr(self._client_authentication, "explicit_method", "")
        if explicit_method:
            if explicit_method not in methods:
                raise _configuration_error("issuer does not support configured client authentication")
            self._client_auth = explicit_method
        elif "client_secret_post" in methods:
            self._client_auth = "client_secret_post"
        elif "client_secret_basic" in methods:
            self._client_auth = "client_secret_basic"
        else:
            raise _configuration_error("issuer requires an unsupported client authentication method")
        self._metadata_until = time.monotonic() + 300

    def exchange(self, data: dict[str, str], *, http: httpx.Client | None = None) -> tuple[str, float]:
        self._discover()
        form = dict(data)
        if not form.get("scope"):
            form.pop("scope", None)
        auth = None
        if self._client_authentication is None and self._client_auth == "client_secret_basic":
            # OAuth requires form encoding each credential before HTTP Basic.
            from urllib.parse import quote_plus

            auth = httpx.BasicAuth(
                quote_plus(form.pop("client_id")), quote_plus(form.pop("client_secret"))
            )
        for attempt in range(2):
            if self._client_authentication is not None:
                form, auth = self._client_authentication.prepare(
                    self._endpoint, data, method=self._client_auth,
                )
            proof = self._key.proof("POST", self._endpoint, nonce=self._nonce)
            response = (http or self._http).post(
                self._endpoint, data=form, auth=auth, headers={"DPoP": proof},
                follow_redirects=False,
            )
            nonce = response.headers.get("DPoP-Nonce")
            if nonce is not None:
                if not nonce or len(nonce) > 2048 or not nonce.isascii():
                    raise _configuration_error("issuer returned an invalid DPoP nonce")
                self._nonce = nonce
            body = response.json()
            if not isinstance(body, dict):
                raise _configuration_error("issuer returned an invalid token response")
            if response.status_code == 400 and body.get("error") == "use_dpop_nonce":
                if nonce and attempt == 0:
                    continue
                raise GatewayUnavailable(
                    reason="issuer nonce challenge could not be satisfied",
                    code="credential_dpop_nonce_failed",
                )
            if response.status_code != 200:
                raise GatewayUnavailable(
                    reason="issuer rejected the DPoP token request",
                    code="credential_exchange_failed",
                )
            return self._bound_response(body)
        raise AssertionError("unreachable")

    def _bound_response(self, body: dict) -> tuple[str, float]:
        if str(body.get("token_type", "")).lower() != "dpop":
            raise _configuration_error("issuer did not return a DPoP-bound token")
        token = body.get("access_token")
        try:
            if not isinstance(token, str) or not token.isascii() or len(token) > 16384:
                raise ValueError("invalid JWT")
            parts = token.split(".")
            if len(parts) != 3 or not all(parts):
                raise ValueError("JWT required")
            claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
            thumbprint = claims["cnf"]["jkt"]
            if not isinstance(thumbprint, str) or not hmac.compare_digest(thumbprint, self._key.thumbprint):
                raise ValueError("incorrect key binding")
            if claims.get("iss") != self._issuer:
                raise ValueError("incorrect issuer")
            raw_ttl = body.get("expires_in", 3600)
            if isinstance(raw_ttl, bool):
                raise ValueError("invalid token lifetime")
            expires_in = float(raw_ttl)
            remaining = float(claims["exp"]) - time.time()
            if not all(math.isfinite(value) and value > 0 for value in (expires_in, remaining)):
                raise ValueError("invalid token lifetime")
            ttl = min(expires_in, remaining)
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise _configuration_error("issuer returned an invalid JWT key binding or lifetime") from None
        return token, ttl
