"""OAuth client authentication; private material stays in the workload.

JWT assertions follow RFC 7523; TLS client authentication follows RFC 8705.
TLS client authentication alone does not implement certificate-bound access
tokens at the gateway. DPoP is the supported access-token binding mechanism.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import ipaddress
import json
import os
from pathlib import Path
import ssl
import time
from urllib.parse import quote_plus, urlsplit
import uuid

import httpx

from ..errors import GovernanceConfigurationError


def configuration_error(reason: str) -> GovernanceConfigurationError:
    return GovernanceConfigurationError(
        framework="agent-credential", reason=reason,
        code="credential_configuration_error",
    )


def secure_url(value: str, *, issuer: bool = False) -> str:
    try:
        if not isinstance(value, str) or any(character.isspace() or ord(character) < 32 for character in value):
            raise ValueError("invalid URL")
        parsed = urlsplit(value)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if (parsed.scheme not in {"https", "http"}
                or (parsed.scheme != "https" and not loopback)
                or not parsed.hostname or parsed.username
                or parsed.password or parsed.fragment or (issuer and parsed.query)):
            raise ValueError("HTTPS endpoint required")
        _ = parsed.port
    except (TypeError, ValueError):
        raise configuration_error("OIDC issuer and token endpoint must use HTTPS") from None
    return value


class ClientAuthentication:
    METHODS = frozenset({
        "client_secret_post", "client_secret_basic", "private_key_jwt",
        "tls_client_auth", "self_signed_tls_client_auth",
    })

    def __init__(self, client_id: str, secret_env: str, method: str | None = None):
        self.client_id = client_id
        self.secret_env = secret_env
        self.explicit_method = method or os.environ.get("OIDC_CLIENT_AUTH_METHOD", "")
        self.method = self.explicit_method or "client_secret_post"
        if self.method not in self.METHODS:
            raise configuration_error("unsupported OIDC client authentication method")

    def prepare(self, endpoint: str, data: dict[str, str], *, method: str | None = None):
        secure_url(endpoint)
        selected = method or self.method
        if not self.client_id.strip() or selected not in self.METHODS:
            raise configuration_error("OIDC client authentication is incomplete")
        form = {**data, "client_id": self.client_id}
        auth = None
        if selected in {"client_secret_post", "client_secret_basic"}:
            secret = os.environ.get(self.secret_env, "")
            if not secret:
                raise configuration_error("OIDC client secret is not configured")
            if selected == "client_secret_basic":
                form.pop("client_id")
                # RFC 6749 requires form encoding each credential first.
                auth = httpx.BasicAuth(quote_plus(self.client_id), quote_plus(secret))
            else:
                form["client_secret"] = secret
        elif selected == "private_key_jwt":
            form["client_assertion_type"] = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            form["client_assertion"] = self._assertion(endpoint)
        return form, auth

    def metadata_endpoint(self, metadata: dict) -> str:
        endpoint = metadata.get("token_endpoint", "")
        if self.method in {"tls_client_auth", "self_signed_tls_client_auth"}:
            aliases = metadata.get("mtls_endpoint_aliases", {})
            if not isinstance(aliases, dict):
                raise configuration_error("OIDC TLS endpoint aliases are invalid")
            # RFC 8705 section 5 requires the advertised TLS endpoint alias
            # when present. An explicit endpoint is never silently rewritten.
            endpoint = aliases.get("token_endpoint", endpoint)
        return secure_url(endpoint)

    def _assertion(self, endpoint: str) -> str:
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils
        except ImportError:
            raise GovernanceConfigurationError(
                framework="agent-credential", reason="private_key_jwt requires cryptography",
                code="credential_dependency_missing",
            ) from None
        try:
            filename = os.environ.get("OIDC_PRIVATE_KEY_FILE", "")
            if not filename:
                raise ValueError("missing private key file")
            # Read on refresh, allowing atomic file rotation while warm tokens
            # remain cached. Never copy PEM material into discovery metadata.
            with Path(filename).open("rb") as source:
                pem = source.read(1024 * 1024 + 1)
            if len(pem) > 1024 * 1024:
                raise ValueError("private key file too large")
            password = os.environ.get("OIDC_PRIVATE_KEY_PASSWORD", "")
            key = serialization.load_pem_private_key(
                pem, password=password.encode() if password else None,
            )
            algorithm = os.environ.get("OIDC_PRIVATE_KEY_ALG", "RS256")
            if algorithm in {"RS256", "PS256"}:
                if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 2048:
                    raise ValueError("RSA key required")
            elif algorithm == "ES256":
                if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
                    raise ValueError("P-256 key required")
            else:
                raise ValueError("unsupported assertion algorithm")
            header = {"alg": algorithm, "typ": "JWT"}
            if kid := os.environ.get("OIDC_PRIVATE_KEY_KID", ""):
                header["kid"] = kid
            now = int(time.time())
            claims = {
                "iss": self.client_id, "sub": self.client_id,
                "aud": os.environ.get("OIDC_ASSERTION_AUDIENCE", "") or endpoint,
                "iat": now, "exp": now + 60, "jti": str(uuid.uuid4()),
            }
            def encoded(value):
                return base64.urlsafe_b64encode(value).rstrip(b"=")
            unsigned = b".".join(encoded(json.dumps(value, separators=(",", ":")).encode())
                                 for value in (header, claims))
            if algorithm == "ES256":
                r, s = utils.decode_dss_signature(key.sign(unsigned, ec.ECDSA(hashes.SHA256())))
                signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
            else:
                pad = padding.PKCS1v15() if algorithm == "RS256" else padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()), salt_length=32,
                )
                signature = key.sign(unsigned, pad, hashes.SHA256())
            return (unsigned + b"." + encoded(signature)).decode("ascii")
        except (OSError, TypeError, ValueError):
            raise configuration_error("OIDC signing key or assertion settings are invalid") from None

    @contextmanager
    def http_client(self, default: httpx.Client):
        if self.method not in {"tls_client_auth", "self_signed_tls_client_auth"}:
            yield default
            return
        try:
            cert = os.environ.get("OIDC_CLIENT_CERT_FILE", "")
            key = os.environ.get("OIDC_CLIENT_KEY_FILE", "")
            if not cert or not key:
                raise ValueError("certificate and key files required")
            context = ssl.create_default_context(cafile=os.environ.get("OIDC_CA_BUNDLE") or None)
            context.load_cert_chain(
                certfile=cert, keyfile=key,
                password=os.environ.get("OIDC_CLIENT_KEY_PASSWORD") or None,
            )
        except (OSError, TypeError, ValueError):
            raise configuration_error("OIDC TLS client credentials are invalid") from None
        # A fresh pool only on token refresh reloads rotated certificates.
        # Default verification remains enabled; redirects are never followed.
        with httpx.Client(verify=context, timeout=10.0, follow_redirects=False) as client:
            yield client
