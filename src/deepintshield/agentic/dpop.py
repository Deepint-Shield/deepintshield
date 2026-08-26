"""DPoP (RFC 9449) proof-of-possession for agent workload tokens.

An agent token is a bearer credential: anyone holding it can use it until it
expires. The gateway's replay cache already refuses a token PRESENTED twice,
which covers one recovered from a log and reused - but nothing helps when a
token is intercepted before its first use, because there is nothing tying the
token to the sender.

DPoP ties them. This process holds a private key, the issuer binds the public
key's thumbprint into the token as ``cnf.jkt``, and every request carries a
fresh proof signed by that key over the method and URI. A stolen token is then
useless without the key.

The key is EPHEMERAL and per-process by design. Persisting it would create
exactly the long-lived secret-on-disk this is meant to remove, and a restarted
workload simply obtains a token bound to its new key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import uuid
from typing import Optional

try:  # pragma: no cover - exercised by whichever branch the environment takes
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    _CRYPTOGRAPHY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _CRYPTOGRAPHY_AVAILABLE = False


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class DPoPKey:
    """An ephemeral P-256 signing key and the proofs it produces.

    P-256/ES256 because it is the one algorithm every DPoP implementation is
    required to support, so a proof from here verifies anywhere.
    """

    def __init__(self) -> None:
        if not _CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "DPoP proof-of-possession requires the 'cryptography' package; "
                "install it or leave proof-of-possession disabled"
            )
        self._key = ec.generate_private_key(ec.SECP256R1())
        numbers = self._key.public_key().public_numbers()
        # A JWK's thumbprint is computed over its members in lexicographic
        # order with no whitespace (RFC 7638). Building the dict in that order
        # and serialising compactly is what makes the thumbprint reproducible
        # by the verifier.
        self.public_jwk = {
            "crv": "P-256",
            "kty": "EC",
            "x": _b64url(numbers.x.to_bytes(32, "big")),
            "y": _b64url(numbers.y.to_bytes(32, "big")),
        }
        canonical = json.dumps(self.public_jwk, separators=(",", ":"), sort_keys=True)
        self.thumbprint = _b64url(hashlib.sha256(canonical.encode("utf-8")).digest())

    def proof(self, method: str, uri: str) -> str:
        """Return a compact JWS proof binding this key to one request.

        ``htu`` deliberately excludes query and fragment, as the RFC requires -
        including them would make every proof fail the moment a caller added a
        query parameter.
        """
        header = {"typ": "dpop+jwt", "alg": "ES256", "jwk": self.public_jwk}
        target = str(uri or "").split("?", 1)[0].split("#", 1)[0]
        payload = {
            "jti": uuid.uuid4().hex,
            "htm": str(method or "").upper(),
            "htu": target,
            "iat": int(time.time()),
        }
        signing_input = (
            _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
            + "."
            + _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        )
        der = self._key.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256()))
        # JWS wants the raw r||s pair, not the DER structure the library emits.
        r, s = utils.decode_dss_signature(der)
        raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return signing_input + "." + _b64url(raw)


_key_lock = threading.Lock()
_process_key: Optional[DPoPKey] = None


def process_dpop_key() -> Optional[DPoPKey]:
    """The process-wide key, or None when proofs cannot be produced.

    Returning None rather than raising is deliberate: a workload without the
    optional dependency should keep working as a plain bearer client, exactly
    as it did before. The gateway refuses a bound token with no proof, so the
    failure surfaces there with a message that names the cause - which is far
    more useful than an import error at startup.
    """
    global _process_key
    if not _CRYPTOGRAPHY_AVAILABLE:
        return None
    with _key_lock:
        if _process_key is None:
            _process_key = DPoPKey()
        return _process_key


def dpop_headers(method: str, uri: str) -> dict[str, str]:
    """Headers carrying a fresh proof, or {} when proofs are unavailable."""
    key = process_dpop_key()
    if key is None:
        return {}
    try:
        return {"DPoP": key.proof(method, uri)}
    except Exception:
        # A proof that cannot be built must not break the call: the request
        # proceeds as a bearer and the gateway decides whether that is
        # acceptable for this token.
        return {}


__all__ = ["DPoPKey", "process_dpop_key", "dpop_headers"]
