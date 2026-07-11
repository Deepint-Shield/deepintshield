"""ZeroID credential - RFC 8693 token exchange.

ZeroID's wire format follows OAuth 2.0 Token Exchange (RFC 8693) with a
standard ``act`` claim chain and SPIFFE/WIMSE identity URIs. The gateway's
broker verifies the resulting JWT against ZeroID's JWKS the same way it does
for Entra - the SDK side is just the exchange dance.

For a full integration you'll typically swap in the vendor SDK's own token
provider here; this implementation is the minimal HTTP shape that works for
any RFC 8693 issuer.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx

from .base import _CachedToken

log = logging.getLogger(__name__)


class ZeroIDCredential:
    """RFC 8693 token exchange against a ZeroID issuer."""

    def __init__(
        self,
        exchange_endpoint: str,
        gateway_audience: str,
        scopes: List[str],
        subject_token_env: str = "ZEROID_SUBJECT_TOKEN",
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._endpoint = exchange_endpoint
        self._audience = gateway_audience
        self._scopes = list(scopes) or ["tools:invoke"]
        self._subject_token_env = subject_token_env
        self._http = http_client or httpx.Client(timeout=10.0)
        self._cache = _CachedToken()

    @property
    def provider_type(self) -> str:
        return "zeroid"

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
        subject_token = os.environ.get(self._subject_token_env, "")
        if not subject_token:
            raise RuntimeError(
                f"ZeroID subject token not found in env {self._subject_token_env}. "
                "Provide one via the workload identity / sidecar that issued it."
            )
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "audience": self._audience,
            "scope": " ".join(self._scopes),
        }
        resp = self._http.post(self._endpoint, data=data)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"ZeroID exchange failed (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        body = resp.json()
        return body["access_token"], float(body.get("expires_in", 3600))
