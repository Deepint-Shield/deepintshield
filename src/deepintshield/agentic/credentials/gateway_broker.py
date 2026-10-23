"""Explicit gateway runtime authentication, bound to an approved workload key.

The virtual key authorizes enrollment requests; it does not approve a new key.
An administrator must authorize the key before the gateway issues a
short-lived opaque session. A pending key is reported to the developer once
per process by the engine/registry (reference, thumbprint, notification
e-mail); this adapter only raises the code with its payload. Neither Entra
credentials nor sessions are saved on disk. Only the per-workload signing key
is durable.

A gateway offers a pending registration its provisional runtime key only to a
client that says it can hold one. :func:`runtime_key_capable` is that local
judgement and :func:`provisional_key_headers` the one request header that
states it, on credential discovery and on enrolment and nowhere else. A client
that does not send it keeps the earlier registration flow.
"""

from __future__ import annotations

import ipaddress
import math
import random
import re
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..errors import GovernanceConfigurationError, GatewayUnavailable
from ..types import RuntimeBrokerInfo
from .base import _CachedToken
from .runtime_key import load_runtime_key, runtime_key_available

_PREFIX = "/api/agentic-new/runtime-auth/"
_SESSION_TOKEN = re.compile(r"^dsrs_[A-Za-z0-9_-]{43}$")
# Broker protocol codes plus the one registry code the challenge/session
# routes emit for a key whose agent identity has not been declared yet.
_ERROR_CODE = re.compile(r"^(?:agent_runtime_[a-z0-9_]{1,48}|runtime_binding_identity_missing)$")
_REGISTRATION_ID_LIMIT = 64
# Asks the gateway for the provisional runtime key of a pending registration.
_PROVISIONAL_KEY_HEADER = "X-DeepintShield-Runtime-Key"
_PROVISIONAL_KEY_VALUE = "provisional"
_sleep = time.sleep


class _CapacityError(GatewayUnavailable):
    def __init__(self, retry_after: float) -> None:
        super().__init__(code="agent_runtime_capacity_exceeded", status_code=429)
        self.retry_after = retry_after


def _config(code: str = "agent_runtime_configuration_error") -> GovernanceConfigurationError:
    return GovernanceConfigurationError(framework="gateway-broker", code=code)


def _gateway_origin(value: str) -> str:
    try:
        url = urlsplit(value)
        if (
            not url.hostname or url.username is not None or url.password is not None
            or url.query or url.fragment or url.path not in ("", "/")
        ):
            raise ValueError
        if url.scheme != "https":
            # HTTP is only for local development, never a LAN/cloud gateway.
            if url.scheme != "http" or not (
                url.hostname == "localhost" or ipaddress.ip_address(url.hostname).is_loopback
            ):
                raise ValueError
        host = url.hostname.lower()
        host = f"[{host}]" if ":" in host else host
        port = url.port
        if port is not None and port != (443 if url.scheme == "https" else 80):
            host += f":{port}"
        return urlunsplit((url.scheme, host, "", "", ""))
    except Exception:
        raise _config() from None


def runtime_key_capable(gateway_url: object) -> bool:
    """Whether this process can hold and use a runtime key for ``gateway_url``.

    Judged from local facts only, with the rules :class:`GatewayBrokerCredential`
    applies when it is built: the gateway origin it accepts and what its key
    store needs (the ``runtime`` extra, a readable mounted key or a usable key
    directory). No request is sent, no key is generated and nothing is written,
    so it can be asked before the gateway has said anything about this agent.
    """
    try:
        _gateway_origin(gateway_url)  # type: ignore[arg-type]
    except GovernanceConfigurationError:
        return False
    return runtime_key_available()


def provisional_key_headers(capable: object) -> dict[str, str]:
    """The request header a capable client sends; nothing for any other client."""
    return {_PROVISIONAL_KEY_HEADER: _PROVISIONAL_KEY_VALUE} if capable is True else {}


def _endpoint(origin: str, path: str, suffix: str) -> str:
    # Discovery cannot turn the SDK into a bearer forwarding proxy, even to a
    # different path on the same host. Only this protocol's fixed endpoints.
    expected = _PREFIX + suffix
    if path != expected:
        raise _config()
    return origin + expected


def _bounded_string(value: object, limit: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise _config("agent_runtime_response_invalid")
    if any(ord(char) < 33 or ord(char) > 126 for char in value):
        raise _config("agent_runtime_response_invalid")
    return value


def _ttl(value: object, maximum: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _config("agent_runtime_response_invalid")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or result > maximum:
        raise _config("agent_runtime_response_invalid")
    return result


class GatewayBrokerCredential:
    """One gateway/workspace/agent credential with its own durable DPoP key."""

    def __init__(
        self, *, http: httpx.Client, gateway_url: str, virtual_key: str,
        agent_subject: str, broker: RuntimeBrokerInfo,
        accepts_provisional_key: bool = False,
    ) -> None:
        self._origin = _gateway_origin(gateway_url)
        self._enrollment_url = _endpoint(self._origin, broker.enrollment_path, "enroll")
        self._challenge_url = _endpoint(self._origin, broker.challenge_path, "challenge")
        self._session_url = _endpoint(self._origin, broker.session_path, "session")
        if (
            not broker.tenant_id or not broker.workspace_id or not broker.agent_subject
            or not agent_subject or broker.agent_subject != agent_subject
            or not agent_subject.startswith("agent:")
            or broker.runtime_kind not in {"key", "pinned_key", "gcp", "aws", "azure", "kubernetes"}
            or not isinstance(virtual_key, str) or not virtual_key or len(virtual_key) > 8192
            or any(ord(char) < 33 or ord(char) > 126 for char in virtual_key)
        ):
            raise _config()
        self._http = http
        timeout = getattr(http.timeout, "read", None)
        self._exchange_budget = min(30.0, timeout) if isinstance(timeout, (float, int)) and timeout > 0 else 30.0
        self._virtual_key = virtual_key
        self._agent_subject = agent_subject
        self._broker = broker
        self._cache = _CachedToken()
        self._enrolled = False
        self._binding_id = ""
        self._runtime_kind = broker.runtime_kind
        self._runtime_audience = broker.runtime_audience
        self._assertion = None
        # Stated on enrolment only, and only when the owning engine found this
        # process able to hold a runtime key. Challenge and session never
        # carry it: a key that reaches them is already enrolled.
        self._enrollment_headers = provisional_key_headers(accepts_provisional_key)
        self._key = load_runtime_key(
            self._origin, broker.tenant_id, broker.workspace_id, broker.agent_subject,
        )

    @property
    def provider_type(self) -> str:
        return "gateway_broker"

    @property
    def key_thumbprint(self) -> str:
        return self._key.thumbprint

    @property
    def enrolled(self) -> bool:
        """Whether the gateway has accepted this key as an approved binding."""
        return self._enrolled

    def get_token(self) -> str:
        return self._cache.get_or_refresh(self._refresh)

    def proof_headers(self, method: str, uri: str, access_token: str) -> dict[str, str]:
        # This method is called on the SAME credential instance that supplied
        # the token. Never use the process-wide ephemeral issuer DPoP key.
        # Match httpx's actual wire URI normalization (including default port
        # removal), not a differently serialized caller string.
        uri = str(httpx.URL(uri))
        target = urlsplit(uri)
        if _gateway_origin(urlunsplit((target.scheme, target.netloc, "", "", ""))) != self._origin:
            raise _config("agent_workload_destination_invalid")
        if not _SESSION_TOKEN.fullmatch(access_token):
            raise _config("agent_runtime_response_invalid")
        return {"DPoP": self._key.proof(method, uri, access_token)}

    def _post(
        self, url: str, body: dict, *, ath: str, nonce: str | None = None, deadline: float,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GatewayUnavailable(code="agent_runtime_exchange_failed")
        headers = {
            **(extra_headers or {}),
            "Authorization": f"Bearer {self._virtual_key}",
            "X-Agent-Subject": self._agent_subject,
            "Content-Type": "application/json",
            "DPoP": self._key.proof("POST", url, ath, nonce=nonce),
        }
        try:
            response = self._http.post(
                url, json=body, headers=headers, timeout=min(15, remaining), follow_redirects=False,
            )
        except Exception:
            raise GatewayUnavailable(code="agent_runtime_exchange_failed") from None
        if response.status_code < 200 or 300 <= response.status_code < 400:
            raise _config("agent_runtime_configuration_error")
        if response.status_code == 429:
            try:
                retry_after = float(response.headers.get("Retry-After", "5"))
                if not math.isfinite(retry_after) or retry_after < 0:
                    retry_after = 5.0
            except (TypeError, ValueError):
                retry_after = 5.0
            raise _CapacityError(retry_after)
        if len(response.content) > 65536:
            raise _config("agent_runtime_response_invalid")
        try:
            data = response.json()
        except Exception:
            raise _config("agent_runtime_response_invalid") from None
        if not isinstance(data, dict):
            raise _config("agent_runtime_response_invalid")
        if response.status_code >= 400:
            code = data.get("code") or data.get("error")
            if isinstance(code, dict):
                code = code.get("code")
            if not isinstance(code, str) or not _ERROR_CODE.fullmatch(code):
                code = "agent_runtime_exchange_failed"
            error = GovernanceConfigurationError(
                framework="gateway-broker", code=code, status_code=response.status_code,
            )
            # Which of the three fixed endpoints refused. Read by the engine
            # only: a provisional key refused at enrolment or challenge makes
            # it ask for the agent's credential settings once more. Never part
            # of the public error text.
            error.runtime_endpoint = self._endpoint_name(url)
            raise error
        return data

    def _endpoint_name(self, url: str) -> str:
        if url == self._enrollment_url:
            return "enroll"
        if url == self._challenge_url:
            return "challenge"
        if url == self._session_url:
            return "session"
        return ""

    def _enroll(self, deadline: float) -> None:
        result = self._post(
            self._enrollment_url, {"public_jwk": self._key.public_jwk}, ath=self._virtual_key,
            deadline=deadline, extra_headers=self._enrollment_headers,
        )
        if result.get("key_thumbprint") != self._key.thumbprint:
            raise _config("agent_runtime_response_invalid")
        self._binding_id = _bounded_string(result.get("binding_id"), 256)
        if self._broker.binding_id and self._binding_id != self._broker.binding_id:
            raise _config("agent_runtime_response_invalid")
        if result.get("status") == "pending":
            # The payload carries only locally computed or bounded identifiers:
            # the RFC 7638 thumbprint, the binding id and, for a provisional
            # key, the registration the gateway bound it to. The engine turns
            # them into the single developer-facing pending line.
            registration_id = result.get("registration_id")
            if (
                not isinstance(registration_id, str)
                or not 1 <= len(registration_id) <= _REGISTRATION_ID_LIMIT
                or any(ord(char) < 33 or ord(char) > 126 for char in registration_id)
            ):
                registration_id = ""
            raise GovernanceConfigurationError(
                framework="gateway-broker", code="agent_runtime_enrollment_pending",
                payload={
                    "key_thumbprint": self._key.thumbprint,
                    "binding_id": self._binding_id,
                    "registration_id": registration_id,
                    "provisional": result.get("provisional") is True,
                },
            )
        if result.get("status") != "approved" or result.get("enabled") is not True:
            raise _config("agent_runtime_binding_disabled")
        self._runtime_configuration(result)
        self._enrolled = True

    def _runtime_configuration(self, result: dict) -> None:
        from .runtime_assertion import RuntimeAssertion

        # Enrollment identifies this particular pinned key. Its approved
        # cloud constraints can therefore be more specific than pre-enrollment
        # agent discovery. A native cloud kind still requires matching local
        # opt-in; the response can never select an ambient credential by itself.
        kind = result.get("runtime_kind", self._runtime_kind)
        audience = result.get("runtime_audience", self._runtime_audience)
        if not isinstance(kind, str) or not isinstance(audience, str):
            raise _config("agent_runtime_response_invalid")
        assertion = RuntimeAssertion(kind, audience)
        if self._assertion is not None and (
            assertion.kind != self._assertion.kind or assertion.audience != self._assertion.audience
        ):
            raise _config()
        self._runtime_kind, self._runtime_audience = kind, audience
        self._assertion = assertion

    def _refresh(self) -> tuple[str, float]:
        deadline = time.monotonic() + self._exchange_budget
        for attempt in range(3):
            try:
                return self._new_session(deadline)
            except _CapacityError as exc:
                delay = exc.retry_after + random.uniform(0, 0.25)
                if attempt == 2 or time.monotonic() + delay >= deadline:
                    raise
                _sleep(delay)
        raise GatewayUnavailable(code="agent_runtime_exchange_failed")  # pragma: no cover

    def _new_session(self, deadline: float) -> tuple[str, float]:
        if not self._enrolled:
            self._enroll(deadline)
        result = self._post(
            self._challenge_url, {"key_thumbprint": self._key.thumbprint}, ath=self._virtual_key,
            deadline=deadline,
        )
        if result.get("binding_id") != self._binding_id:
            raise _config("agent_runtime_response_invalid")
        self._runtime_configuration(result)
        nonce = _bounded_string(result.get("nonce"))
        _ttl(result.get("expires_in"), 600)
        body = {"binding_id": self._binding_id, "nonce": nonce}
        assert self._assertion is not None
        assertion = self._assertion.get_token()
        if assertion is not None:
            body["runtime_assertion"] = assertion
        result = self._post(self._session_url, body, ath=nonce, nonce=nonce, deadline=deadline)
        token = result.get("access_token")
        if (
            result.get("binding_id") != self._binding_id
            or result.get("token_type") != "DPoP"
            or not isinstance(result.get("cnf"), dict)
            or result["cnf"].get("jkt") != self._key.thumbprint
            or not isinstance(token, str) or not _SESSION_TOKEN.fullmatch(token)
        ):
            raise _config("agent_runtime_response_invalid")
        return token, _ttl(result.get("expires_in"), 300)
