"""The provisional runtime key is asked for only by a client that can hold one.

A gateway offers a pending registration its provisional runtime key to a client
that sends ``X-DeepintShield-Runtime-Key: provisional``. SDK 3.2.0 sends that
header on credential discovery and on enrolment, and nowhere else, when this
process can hold and use a runtime key: the ``runtime`` extra imports, the key
store is usable and the gateway address is one the broker credential accepts.
A client that cannot keeps the earlier flow and its plain pending line.

When a provisional offer arrives anyway and cannot be used, the registration is
still simply pending: the sighting is filed with the virtual key alone and the
developer reads ``Registration pending. Reference <id>. Contact your
administrator.`` with no runtime-key clause. The same failures keep stopping a
declared or approved identity.
"""
from __future__ import annotations

import base64
import builtins
import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from deepintshield.agentic import engine as engine_module
from deepintshield.agentic import gate, registry
from deepintshield.agentic.credentials import gateway_broker, runtime_key
from deepintshield.agentic.credentials.gateway_broker import (
    GatewayBrokerCredential,
    provisional_key_headers,
    runtime_key_capable,
)
from deepintshield.agentic.errors import GovernanceConfigurationError, public_agentic_error
from deepintshield.agentic.types import DelegationContext, RuntimeBrokerInfo

ORIGIN = "https://gateway.example"
HEADER = "X-DeepintShield-Runtime-Key"
CREDENTIAL_INFO = "/api/agentic-new/credential-info"
LEGACY_CREDENTIAL_INFO = "/api/agentic-security/vk-credential-info"
DISCOVER = "/api/agentic-new/registry/discover"
DECIDE = "/api/agentic-new/decide"
PREFIX = "/api/agentic-new/runtime-auth/"
ENROLL, CHALLENGE, SESSION = PREFIX + "enroll", PREFIX + "challenge", PREFIX + "session"
AGENT = "agent:math-agent"
REGISTRATION_ID = "agnew-rr-0123456789abcdef"
SESSION_TOKEN = "dsrs_" + "a" * 43
PLAIN_PENDING = (
    f"agent_registration_pending: Registration pending. Reference {REGISTRATION_ID}. "
    "Contact your administrator."
)
MANIFEST = {
    "network": {"key": "math-agent", "name": "math-agent"},
    "agents": [{"key": "math-agent", "name": "math-agent"}],
    "tools": [],
    "edges": [],
}
Seen = namedtuple("Seen", "method path runtime_key agent_token")

try:
    from cryptography.hazmat.primitives.asymmetric import ec as _ec  # noqa: F401

    _RUNTIME_EXTRA = True
except ImportError:  # the SDK installed without its ``runtime`` extra
    _RUNTIME_EXTRA = False
# Tests that need a process able to hold a runtime key. The others also run,
# unchanged, where the extra is really absent.
needs_runtime_extra = pytest.mark.skipif(not _RUNTIME_EXTRA, reason="needs deepintshield[runtime]")


@pytest.fixture(autouse=True)
def isolated_process_state(tmp_path, monkeypatch):
    """Once-per-process guards and runtime-key storage never leak between tests."""
    monkeypatch.setattr(registry, "_enrolment_logged", set())
    monkeypatch.setattr(engine_module, "_credential_failure_logged", set())
    for name in list(os.environ):
        if name.startswith("DEEPINTSHIELD_RUNTIME_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("DEEPINTSHIELD_REGISTERED_BY", raising=False)
    monkeypatch.delenv("DEEPINTSHIELD_REQUESTER", raising=False)
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KEY_DIR", str(tmp_path / "keys"))
    monkeypatch.setattr(runtime_key, "_secure_keyring", lambda: None)


def _json(request: httpx.Request) -> dict:
    return json.loads(request.content or b"{}")


def _jwk_thumbprint(jwk: dict) -> str:
    raw = json.dumps(jwk, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()


def _broker(**changes) -> dict:
    return {
        "enrollment_path": ENROLL, "challenge_path": CHALLENGE, "session_path": SESSION,
        "tenant_id": "platform-tenant", "workspace_id": "workspace-one",
        "agent_subject": AGENT, "runtime_kind": "key", **changes,
    }


def _key_files(tmp_path: Path) -> list[str]:
    return sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))


class Gateway:
    """A gateway holding one pending registration that nobody has described yet.

    ``asks_first`` models the gateway that reads the request header: the
    provisional block and a provisional enrolment go only to a client that
    sent it, and every other client gets the 409 it always got. With
    ``asks_first=False`` it is an older gateway that makes the offer to
    everybody. ``block_changes`` (``None`` removes a key) and ``broker_changes``
    reshape the credential-info document; ``answers`` replaces the answer of a
    path altogether.
    """

    def __init__(self, *, asks_first: bool = True, notify: str = "") -> None:
        self.asks_first = asks_first
        self.notify = notify
        self.block_changes: dict = {}
        self.broker_changes: dict = {}
        self.answers: dict = {}
        self.seen: list[Seen] = []
        self.sightings: list[dict] = []
        self.jkt = ""

    def on(self, path: str) -> list[Seen]:
        return [seen for seen in self.seen if seen.path == path]

    def paths(self) -> list[str]:
        return [seen.path for seen in self.seen]

    def registration(self, *, echo: bool = True) -> dict:
        row = {"id": REGISTRATION_ID, "agent_subject": AGENT, "status": "pending"}
        if echo and self.notify:
            row["requester_email"] = self.notify
        return row

    def block(self) -> dict:
        document = {
            "runtime_auth_mode": "gateway_broker",
            "runtime_broker": _broker(**self.broker_changes),
            "provider_id": "idp-entra", "provider_type": "entra_agent_id",
            "tenant_id": "directory-tenant", "blueprint_client_id": "blueprint-client",
            "authority": "https://login.microsoftonline.com/directory-tenant/v2.0",
            "gateway_audience": "api://deepintshield", "scopes": ["tools:invoke"],
            "require_proof_of_possession": True, "agent_subject": AGENT,
            "agent_configured": True, "credential_bootstrap": True, "provisional": True,
            "registration_id": REGISTRATION_ID, "registration_state": "pending",
            **self.block_changes,
        }
        return {key: value for key, value in document.items() if value is not None}

    def pending(self) -> httpx.Response:
        body = {
            "error": {"code": "agent_registration_pending", "message": "PRIVATE: open /workspace/private"},
            "registration": self.registration(echo=False),
            "action": {"path": "/workspace/private", "audience": "administrator"},
            "reference": REGISTRATION_ID,
            "contact_administrator": True,
        }
        if self.notify:
            body["notify_email"] = self.notify
        return httpx.Response(409, json=body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        asked = request.headers.get(HEADER, "").strip().lower() == "provisional"
        self.seen.append(
            Seen(request.method, path, request.headers.get(HEADER), request.headers.get("X-Agent-Token", ""))
        )
        if path in self.answers:
            return self.answers[path](request)
        if path == CREDENTIAL_INFO:
            if self.asks_first and not asked and self.block().get("provisional") is True:
                return self.pending()
            return httpx.Response(200, headers={"Cache-Control": "no-store"}, json=self.block())
        if path == ENROLL:
            if self.asks_first and not asked:
                return httpx.Response(401, json={"error": {"code": "agent_runtime_authentication_failed"}})
            self.jkt = _jwk_thumbprint(_json(request)["public_jwk"])
            return httpx.Response(200, json={
                "binding_id": "binding-provisional", "key_thumbprint": self.jkt, "status": "pending",
                "enabled": False, "provisional": True, "registration_id": REGISTRATION_ID,
            })
        if path == DISCOVER:
            assert "X-Agent-Token" not in request.headers, "a pending agent has no workload proof"
            self.sightings.append(_json(request))
            return httpx.Response(202, json={
                "network_id": "network:math", "registration_state": "pending",
                "blueprint_scan_status": "complete", "registration": self.registration(),
            })
        return httpx.Response(404)


def _engine(shield_factory, gateway, **client_kwargs):
    client_kwargs.setdefault("base_url", ORIGIN)
    return shield_factory(gateway, agent_name="math-agent", **client_kwargs).agentic.engine


def _gated_call(engine) -> GovernanceConfigurationError:
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path=DECIDE)
    return caught.value


# ── how a process comes to be unable to hold or use a runtime key ─────────


def _without_runtime_extra(monkeypatch, tmp_path, gateway) -> dict:
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("PRIVATE: no module named cryptography")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    return {}


def _with_unaccepted_gateway_address(monkeypatch, tmp_path, gateway) -> dict:
    # Plain HTTP to a host that is not this machine, as inside a Compose network.
    return {"base_url": "http://gateway.internal:8080"}


def _with_unusable_key_directory(monkeypatch, tmp_path, gateway) -> dict:
    (tmp_path / "keys").write_text("not a directory")
    return {}


def _with_unaccepted_broker_settings(monkeypatch, tmp_path, gateway) -> dict:
    gateway.broker_changes = {"enrollment_path": "https://other.example/enroll"}
    return {}


def _with_corrupt_stored_key(monkeypatch, tmp_path, gateway) -> dict:
    # The directory is fine; only loading shows that this scope's key is not.
    scope = hashlib.sha256(json.dumps(
        [ORIGIN, "platform-tenant", "workspace-one", AGENT], separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()
    directory = tmp_path / "keys"
    directory.mkdir(mode=0o700)
    descriptor = directory / (scope + ".json")
    descriptor.write_bytes(b"not json")
    descriptor.chmod(0o600)
    return {}


def _redirect(request: httpx.Request) -> httpx.Response:
    return httpx.Response(307, headers={"Location": "https://elsewhere.example" + request.url.path})


def _with_redirected_enrolment(monkeypatch, tmp_path, gateway) -> dict:
    gateway.answers[ENROLL] = _redirect
    return {}


Unusable = namedtuple("Unusable", "code capable arrange")
# ``capable`` is what the process concludes about itself beforehand. An offer
# reaches a client that is not capable only from a gateway that makes it to
# every client; a capable one can still meet a key it cannot build or enrol.
UNUSABLE = {
    "runtime_extra_missing": Unusable("credential_dependency_missing", False, _without_runtime_extra),
    "gateway_address_not_accepted": Unusable(
        "agent_runtime_configuration_error", False, _with_unaccepted_gateway_address,
    ),
    "key_directory_unusable": Unusable("agent_runtime_key_unavailable", False, _with_unusable_key_directory),
    "broker_settings_not_accepted": Unusable(
        "agent_runtime_configuration_error", True, _with_unaccepted_broker_settings,
    ),
    "stored_key_corrupt": Unusable("agent_runtime_key_unavailable", True, _with_corrupt_stored_key),
    "enrolment_redirected": Unusable("agent_runtime_configuration_error", True, _with_redirected_enrolment),
}
NOT_CAPABLE = [name for name, case in UNUSABLE.items() if not case.capable]


def _reasons(*, without_the_extra) -> list:
    """Every way a key is unusable; all but ``without_the_extra`` need the extra."""
    return [
        pytest.param(name, id=name, marks=() if name in without_the_extra else needs_runtime_extra)
        for name in UNUSABLE
    ]


# ── the capability ────────────────────────────────────────────────────────


@needs_runtime_extra
def test_a_capable_client_asks_for_the_provisional_key_on_credential_discovery(shield_factory, tmp_path):
    gateway = Gateway()
    engine = _engine(shield_factory, gateway)

    # Nothing is decided, asked or written while the client is only built.
    assert engine._runtime_key_capability is None
    assert gateway.seen == []
    assert _key_files(tmp_path) == []

    info = engine.credential_info

    assert engine._runtime_key_capable() is True
    assert gateway.seen == [Seen("GET", CREDENTIAL_INFO, "provisional", "")]
    assert (info.provisional, info.registration_state, info.registration_id) == (True, "pending", REGISTRATION_ID)
    # The lookup itself still creates no key.
    assert _key_files(tmp_path) == []


@pytest.mark.parametrize("reason", NOT_CAPABLE)
def test_a_client_that_cannot_hold_a_runtime_key_never_sends_the_header(
    shield_factory, monkeypatch, tmp_path, caplog, reason,
):
    gateway = Gateway(notify="dev@corp.com")
    client_kwargs = UNUSABLE[reason].arrange(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway, registered_by="dev@corp.com", **client_kwargs)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # Run 1 exactly as before this release: sighting, then the pending answer.
    gate.preflight(engine, "read_account")
    error = _gated_call(engine)

    assert engine._runtime_key_capable() is False
    assert all(seen.runtime_key is None for seen in gateway.seen)
    assert ENROLL not in gateway.paths()
    assert len(gateway.sightings) == 1
    pending = f"{PLAIN_PENDING} You will be notified at dev@corp.com."
    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == pending
    assert caplog.messages == [f"DeepintShield: {pending}"]


@needs_runtime_extra
def test_the_header_goes_to_the_canonical_route_only(shield_factory):
    seen: list[Seen] = []

    def older_gateway(request: httpx.Request) -> httpx.Response:
        seen.append(Seen(request.method, request.url.path, request.headers.get(HEADER), ""))
        if request.url.path == LEGACY_CREDENTIAL_INFO:
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
        return httpx.Response(404)

    engine = _engine(shield_factory, older_gateway)

    assert engine.credential_info.agent_configured is False
    assert engine._runtime_key_capable() is True
    assert seen == [
        Seen("GET", CREDENTIAL_INFO, "provisional", ""),
        Seen("GET", LEGACY_CREDENTIAL_INFO, None, ""),
    ]
    assert engine._canonical_credential_info_available is False


def _approved_broker(seen: list[Seen], **block_changes):
    """A declared, approved agent with gateway runtime authentication."""
    jkt = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        seen.append(Seen(request.method, path, request.headers.get(HEADER), request.headers.get("X-Agent-Token", "")))
        if path == CREDENTIAL_INFO:
            return httpx.Response(200, json={
                "runtime_auth_mode": "gateway_broker", "runtime_broker": _broker(),
                "provider_type": "entra_agent_id", "agent_configured": True, "agent_subject": AGENT,
                "require_proof_of_possession": True, **block_changes,
            })
        if path == ENROLL:
            jkt["value"] = _jwk_thumbprint(_json(request)["public_jwk"])
            return httpx.Response(200, json={
                "binding_id": "binding-one", "key_thumbprint": jkt["value"], "status": "approved", "enabled": True,
            })
        if path == CHALLENGE:
            return httpx.Response(200, json={"binding_id": "binding-one", "nonce": "one-use-nonce", "expires_in": 60})
        if path == SESSION:
            return httpx.Response(200, json={
                "binding_id": "binding-one", "access_token": SESSION_TOKEN, "token_type": "DPoP",
                "expires_in": 300, "cnf": {"jkt": jkt["value"]},
            })
        if path == DISCOVER:
            return httpx.Response(200, json={"network_id": "network:math", "blueprint_scan_status": "complete"})
        if path == DECIDE:
            return httpx.Response(200, json={
                "verdict": "ALLOW", "allow": True, "proceed": True, "mode": "enforce", "would_block": False,
            })
        return httpx.Response(404)

    return handler


@needs_runtime_extra
def test_the_header_is_sent_on_enrolment_and_never_on_challenge_or_session(shield_factory):
    seen: list[Seen] = []
    engine = _engine(shield_factory, _approved_broker(seen))

    headers = engine._headers(include_agent_token=True, path=DECIDE)
    registry.discover(engine, manifest=dict(MANIFEST), sync=True)
    decision = engine.decide(DelegationContext(tool="read_account", args_digest="digest"))

    assert headers["X-Agent-Token"] == SESSION_TOKEN
    assert HEADER not in headers
    assert decision.proceed is True
    assert [(seen_.path, seen_.runtime_key) for seen_ in seen] == [
        (CREDENTIAL_INFO, "provisional"),
        (ENROLL, "provisional"),
        (CHALLENGE, None),
        (SESSION, None),
        (DISCOVER, None),
        (DECIDE, None),
    ]
    # A later session is a new challenge, not a new enrolment, and still bare.
    engine.agent_credential._cache._refresh_at = 0
    engine._headers(include_agent_token=True, path=DECIDE)
    assert [(seen_.path, seen_.runtime_key) for seen_ in seen[6:]] == [(CHALLENGE, None), (SESSION, None)]
    # Nor does any other request of the client carry it.
    engine._parent().agentic.status()
    assert seen[-1].path == "/api/agentic-new/registry/self"
    assert {seen_.path for seen_ in seen if seen_.runtime_key is not None} == {CREDENTIAL_INFO, ENROLL}


@needs_runtime_extra
def test_the_header_is_absent_from_the_requests_the_shared_fixture_answers_itself(shield_factory):
    # The runtime-config lookup of every governed call never reaches a test
    # gateway: ``shield_factory`` answers it first. Watch it here, together
    # with the connection headers of transparent LLM traffic.
    seen: list[Seen] = []
    settings: list[Seen] = []

    def runtime_config(request: httpx.Request) -> httpx.Response:
        settings.append(Seen(request.method, request.url.path, request.headers.get(HEADER), ""))
        return httpx.Response(200, json={
            "security_enabled": True, "available": True, "plan": "business", "reason": "enabled",
        })

    shield = shield_factory(
        _approved_broker(seen), runtime_config_handler=runtime_config, base_url=ORIGIN, agent_name="math-agent",
    )
    engine = shield.agentic.engine

    decision = gate.resolve(engine, "read_account", (), {})

    assert decision.proceed is True
    assert engine._runtime_key_capable() is True
    assert settings and all(seen_.path.endswith("/agentic-new/runtime-config") for seen_ in settings)
    assert all(seen_.runtime_key is None for seen_ in settings)
    assert {seen_.path for seen_ in seen if seen_.runtime_key is not None} == {CREDENTIAL_INFO, ENROLL}
    assert DECIDE in {seen_.path for seen_ in seen}
    assert HEADER.lower() not in {name.lower() for name in shield.create_headers()}


@needs_runtime_extra
@pytest.mark.parametrize("accepts", [None, False, True])
def test_a_credential_states_the_header_only_when_its_engine_found_it_capable(accepts):
    seen: list[Seen] = []
    kwargs = {} if accepts is None else {"accepts_provisional_key": accepts}
    credential = GatewayBrokerCredential(
        http=httpx.Client(transport=httpx.MockTransport(_approved_broker(seen))),
        gateway_url=ORIGIN, virtual_key="sk-ds-test", agent_subject=AGENT,
        broker=RuntimeBrokerInfo(**_broker()), **kwargs,
    )

    assert credential.enrolled is False
    assert credential.get_token() == SESSION_TOKEN
    assert credential.enrolled is True
    assert [(seen_.path, seen_.runtime_key) for seen_ in seen] == [
        (ENROLL, "provisional" if accepts is True else None), (CHALLENGE, None), (SESSION, None),
    ]


@needs_runtime_extra
def test_the_header_cannot_replace_a_fixed_enrolment_header():
    # Whatever is added on enrolment, the caller's key and selector stay what
    # the credential holds.
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        return httpx.Response(307, headers={"Location": "https://elsewhere.example"})

    credential = GatewayBrokerCredential(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        gateway_url=ORIGIN, virtual_key="sk-ds-test", agent_subject=AGENT,
        broker=RuntimeBrokerInfo(**_broker()), accepts_provisional_key=True,
    )
    credential._enrollment_headers = {"Authorization": "Bearer other", "X-Agent-Subject": "agent:other", HEADER: "provisional"}

    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        credential.get_token()

    assert seen[0]["authorization"] == "Bearer sk-ds-test"
    assert seen[0]["x-agent-subject"] == AGENT
    assert seen[0][HEADER.lower()] == "provisional"


@needs_runtime_extra
def test_the_capability_is_decided_once_and_without_side_effects(shield_factory, monkeypatch, tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ec

    gateway = Gateway(asks_first=True)
    engine = _engine(shield_factory, gateway)
    other = _engine(shield_factory, Gateway())
    asked: list[object] = []
    real = gateway_broker.runtime_key_capable

    def counted(gateway_url):
        asked.append(gateway_url)
        return real(gateway_url)

    def forbidden(*args, **kwargs):
        raise AssertionError("the capability check must stay local")

    monkeypatch.setattr(gateway_broker, "runtime_key_capable", counted)
    with monkeypatch.context() as local:
        local.setattr(socket, "socket", forbidden)
        local.setattr(socket, "create_connection", forbidden)
        local.setattr(socket, "getaddrinfo", forbidden)
        local.setattr(ec, "generate_private_key", forbidden)
        local.setattr(runtime_key, "_write_descriptor", forbidden)
        local.setattr(Path, "mkdir", forbidden)

        assert engine._runtime_key_capable() is True

    # No request, no key, no file: not even the key directory was created.
    assert gateway.seen == []
    assert _key_files(tmp_path) == []
    assert asked == [ORIGIN]

    # Decided once for the engine's lifetime, whatever changes afterwards.
    (tmp_path / "keys").write_text("not a directory")
    assert real(ORIGIN) is False
    for _ in range(3):
        assert engine._runtime_key_capable() is True
        assert engine._provisional_key_headers() == {HEADER: "provisional"}
        engine._clear_credential_failure_cache()
        with pytest.raises(GovernanceConfigurationError):
            engine._headers(include_agent_token=True, path=DECIDE)
    assert asked == [ORIGIN]
    assert {seen.runtime_key for seen in gateway.on(CREDENTIAL_INFO)} == {"provisional"}

    # Another engine decides for itself, also once.
    assert other._runtime_key_capable() is False
    assert other._runtime_key_capable() is False
    assert asked == [ORIGIN, ORIGIN]


@needs_runtime_extra
def test_concurrent_first_lookups_decide_the_capability_once(shield_factory, monkeypatch):
    engine = _engine(shield_factory, Gateway())
    asked: list[object] = []
    real = gateway_broker.runtime_key_capable
    barrier = threading.Barrier(8)

    def slow(gateway_url):
        asked.append(gateway_url)
        time.sleep(0.05)
        return real(gateway_url)

    def first_use(_index):
        barrier.wait(timeout=5)
        return engine._runtime_key_capable()

    monkeypatch.setattr(gateway_broker, "runtime_key_capable", slow)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(first_use, range(8))) == [True] * 8
    assert asked == [ORIGIN]


def test_a_capability_that_cannot_be_judged_is_no_capability(shield_factory, monkeypatch):
    gateway = Gateway()
    engine = _engine(shield_factory, gateway)

    def broken(gateway_url):
        raise OSError("PRIVATE: key store path")

    monkeypatch.setattr(gateway_broker, "runtime_key_capable", broken)

    # The lookup is neither refused nor delayed by it; the client just does not ask.
    error = _gated_call(engine)

    assert engine._runtime_key_capable() is False
    assert error.code == "agent_registration_pending"
    assert gateway.seen == [Seen("GET", CREDENTIAL_INFO, None, "")]


@pytest.mark.parametrize("answer", [1, "provisional", object()], ids=["one", "text", "object"])
def test_only_an_exact_answer_of_the_local_judgement_is_a_capability(shield_factory, monkeypatch, answer):
    gateway = Gateway()
    engine = _engine(shield_factory, gateway)
    monkeypatch.setattr(gateway_broker, "runtime_key_capable", lambda gateway_url: answer)

    error = _gated_call(engine)

    assert engine._runtime_key_capable() is False
    assert engine._provisional_key_headers() == {}
    assert error.code == "agent_registration_pending"
    assert gateway.seen == [Seen("GET", CREDENTIAL_INFO, None, "")]


def _with_a_dependency_that_panics_on_import(monkeypatch) -> None:
    # A broken native build of the optional dependency: pyo3 reports it with
    # ``pyo3_runtime.PanicException``, which derives from BaseException.
    class PanicException(BaseException):
        pass

    real_import = builtins.__import__

    def panicking(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cryptography" or name.startswith("cryptography."):
            raise PanicException("PRIVATE: Python API call failed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", panicking)


def test_a_dependency_that_panics_on_import_is_no_capability(shield_factory, monkeypatch, caplog):
    gateway = Gateway(notify="dev@corp.com")
    _with_a_dependency_that_panics_on_import(monkeypatch)
    engine = _engine(shield_factory, gateway, registered_by="dev@corp.com")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # The lookup that first meets the broken build is neither failed nor
    # changed by it: run 1 ends as it did before the capability existed.
    gate.preflight(engine, "read_account")
    error = _gated_call(engine)

    pending = f"{PLAIN_PENDING} You will be notified at dev@corp.com."
    assert engine._runtime_key_capable() is False
    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == pending
    assert caplog.messages == [f"DeepintShield: {pending}"]
    assert all(seen.runtime_key is None for seen in gateway.seen)
    assert len(gateway.sightings) == 1


def test_a_dependency_that_panics_on_import_leaves_virtual_key_traffic_alone(shield_factory, monkeypatch):
    seen: list[Seen] = []

    def vk_trusted(request: httpx.Request) -> httpx.Response:
        seen.append(Seen(request.method, request.url.path, request.headers.get(HEADER), ""))
        if request.url.path == CREDENTIAL_INFO:
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False, "agent_subject": AGENT})
        if request.url.path == DECIDE:
            return httpx.Response(200, json={
                "verdict": "ALLOW", "allow": True, "proceed": True, "mode": "enforce", "would_block": False,
            })
        return httpx.Response(404)

    _with_a_dependency_that_panics_on_import(monkeypatch)
    engine = _engine(shield_factory, vk_trusted)

    decision = gate.resolve(engine, "read_account", (), {})

    assert decision.proceed is True
    assert engine._runtime_key_capable() is False
    assert engine.agent_token(fail_closed=True) is None
    assert (CREDENTIAL_INFO, None) in {(seen_.path, seen_.runtime_key) for seen_ in seen}
    assert all(seen_.runtime_key is None for seen_ in seen)


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_the_capability_check_never_absorbs_an_interrupt(shield_factory, monkeypatch, signal):
    gateway = Gateway()
    engine = _engine(shield_factory, gateway)
    real = gateway_broker.runtime_key_capable

    def interrupted(gateway_url):
        raise signal()

    monkeypatch.setattr(gateway_broker, "runtime_key_capable", interrupted)
    with pytest.raises(signal):
        engine._runtime_key_capable()
    with pytest.raises(signal):
        engine._headers(include_agent_token=True, path=DECIDE)

    # Nothing was decided or sent, so the next use judges again.
    assert engine._runtime_key_capability is None
    assert gateway.seen == []
    monkeypatch.setattr(gateway_broker, "runtime_key_capable", real)
    decided = engine._runtime_key_capable()
    assert isinstance(decided, bool) and engine._runtime_key_capability is decided


@needs_runtime_extra
def test_an_engine_found_not_capable_states_nothing_even_if_a_key_can_be_built_later(
    shield_factory, tmp_path,
):
    seen: list[Seen] = []
    blocker = tmp_path / "keys"
    blocker.write_text("not a directory")
    # A declared identity: the gateway answers it the same with or without the header.
    engine = _engine(shield_factory, _approved_broker(seen, credential_bootstrap=True))

    assert engine._runtime_key_capable() is False
    blocker.unlink()
    headers = engine._headers(include_agent_token=True, path=DECIDE)

    assert headers["X-Agent-Token"] == SESSION_TOKEN
    assert [(seen_.path, seen_.runtime_key) for seen_ in seen] == [
        (CREDENTIAL_INFO, None), (ENROLL, None), (CHALLENGE, None), (SESSION, None),
    ]


@needs_runtime_extra
def test_a_mounted_key_must_be_readable_and_is_only_read(monkeypatch, tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ec
    from deepintshield.agentic.dpop import DPoPKey

    def load() -> str:
        return runtime_key.load_runtime_key(ORIGIN, "platform-tenant", "workspace-one", AGENT).thumbprint

    mounted = tmp_path / "mounted.pem"
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KEY_FILE", str(mounted))
    # Selected but absent: the loader refuses, and never falls back to the directory.
    assert runtime_key.runtime_key_available() is False
    assert runtime_key_capable(ORIGIN) is False
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        load()

    key = DPoPKey()
    mounted.write_bytes(runtime_key._private_bytes(key))
    mounted.chmod(0o400)
    before = mounted.read_bytes()
    monkeypatch.setattr(ec, "generate_private_key", lambda *args, **kwargs: pytest.fail("no key is generated"))

    assert runtime_key.runtime_key_available() is True
    assert runtime_key_capable(ORIGIN) is True
    assert load() == key.thumbprint
    assert mounted.read_bytes() == before
    # The key directory plays no part beside a mounted key.
    assert _key_files(tmp_path) == ["mounted.pem"]

    if os.name != "nt":
        mounted.chmod(0o644)
        assert runtime_key.runtime_key_available() is False
        with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
            load()


@needs_runtime_extra
@pytest.mark.parametrize(
    "gateway_url",
    [
        "https://gateway.example", "https://gateway.example/", "https://gateway.example:8443",
        "http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080",
        "http://gateway.internal:8080", "http://10.0.0.5:8080", "http://deepintshield-server:8080",
        "https://gateway.example/prefix", "https://someone@gateway.example", "ftp://gateway.example", "",
    ],
)
def test_the_capability_and_the_credential_share_one_gateway_address_rule(gateway_url):
    def build() -> bool:
        try:
            GatewayBrokerCredential(
                http=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404))),
                gateway_url=gateway_url, virtual_key="sk-ds-test", agent_subject=AGENT,
                broker=RuntimeBrokerInfo(**_broker()),
            )
        except GovernanceConfigurationError as error:
            assert error.code == "agent_runtime_configuration_error"
            return False
        return True

    # Asked first, so the answer is given before any key exists.
    capable = runtime_key_capable(gateway_url)
    assert capable is build()


@needs_runtime_extra
@pytest.mark.parametrize(
    "state",
    [
        "missing", "missing_nested", "private", "shared", "file", "below_a_file", "symlink",
        "read_only", "missing_below_read_only",
    ],
)
def test_the_key_store_check_agrees_with_the_loader_and_creates_nothing(monkeypatch, tmp_path, state):
    if os.name == "nt" and state in {"shared", "symlink", "read_only", "missing_below_read_only"}:
        pytest.skip("POSIX ownership, mode and symlink rules")
    if state in {"read_only", "missing_below_read_only"} and os.geteuid() == 0:
        pytest.skip("a privileged user writes to a read-only directory")
    directory = tmp_path / "keys"
    locked = None
    if state == "missing_nested":
        directory = tmp_path / "a" / "b" / "keys"
    elif state == "private":
        directory.mkdir(mode=0o700)
    elif state == "shared":
        directory.mkdir()
        directory.chmod(0o755)
    elif state == "file":
        directory.write_text("not a directory")
    elif state == "below_a_file":
        (tmp_path / "file").write_text("not a directory")
        directory = tmp_path / "file" / "keys"
    elif state == "symlink":
        (tmp_path / "real").mkdir(mode=0o700)
        directory.symlink_to(tmp_path / "real")
    elif state == "read_only":
        # Private and owned, but no lock or descriptor can be created in it.
        directory.mkdir(mode=0o700)
        locked = directory
    elif state == "missing_below_read_only":
        locked = tmp_path / "parent"
        locked.mkdir(mode=0o700)
        directory = locked / "keys"
    if locked is not None:
        locked.chmod(0o500)
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KEY_DIR", str(directory))
    before = _key_files(tmp_path)

    try:
        available = runtime_key.runtime_key_available()

        assert _key_files(tmp_path) == before
        assert available is (state in {"missing", "missing_nested", "private"})
        try:
            runtime_key.load_runtime_key(ORIGIN, "platform-tenant", "workspace-one", AGENT)
            loaded = True
        except GovernanceConfigurationError as error:
            assert error.code == "agent_runtime_key_unavailable"
            loaded = False
        assert loaded is available
    finally:
        if locked is not None:
            locked.chmod(0o700)


@needs_runtime_extra
def test_the_key_store_check_needs_a_keyring_where_a_key_file_is_not_allowed(monkeypatch):
    # Windows keeps a key only in the native credential store.
    monkeypatch.setattr(runtime_key, "_file_storage_supported", lambda: False)
    assert runtime_key.runtime_key_available() is False
    monkeypatch.setattr(runtime_key, "_secure_keyring", lambda: object())
    assert runtime_key.runtime_key_available() is True


def test_only_the_exact_capability_states_the_header():
    assert provisional_key_headers(True) == {HEADER: "provisional"}
    for other in (False, None, 1, "true", "provisional"):
        assert provisional_key_headers(other) == {}


# ── an offer that cannot be used is the plain pending state ───────────────


@pytest.mark.parametrize("notify", ["", "dev@corp.com"])
@pytest.mark.parametrize("reason", _reasons(without_the_extra=NOT_CAPABLE))
def test_an_unusable_provisional_offer_falls_back_to_the_plain_pending_flow(
    shield_factory, monkeypatch, tmp_path, caplog, reason, notify,
):
    case = UNUSABLE[reason]
    # An offer reaches a client that did not ask only from a gateway that
    # makes it to every client.
    gateway = Gateway(asks_first=case.capable, notify=notify)
    client_kwargs = case.arrange(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway, registered_by="dev@corp.com", **client_kwargs)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    filed = registry.discover(engine, manifest=dict(MANIFEST), sync=True)
    error = _gated_call(engine)

    assert engine._runtime_key_capable() is case.capable
    assert {seen.runtime_key for seen in gateway.on(CREDENTIAL_INFO)} == {"provisional" if case.capable else None}
    # The sighting was filed with the virtual key alone.
    assert "error" not in filed
    assert filed["registration_state"] == "pending"
    assert [(seen.method, seen.agent_token) for seen in gateway.on(DISCOVER)] == [("POST", "")]
    assert gateway.sightings[0]["registered_by_email"] == "dev@corp.com"
    # The developer reads the ordinary pending line: no runtime key is named,
    # and the address only when the gateway confirmed it.
    pending = PLAIN_PENDING + (f" You will be notified at {notify}." if notify else "")
    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == pending
    assert "SHA256" not in str(public_agentic_error(error))
    assert error.payload["registration_context"] == {
        "registration_id": REGISTRATION_ID, "key_thumbprint": "", "notify_email": notify, "state": "pending",
    }
    assert caplog.messages == [f"DeepintShield: {pending}"]
    assert case.code not in caplog.text
    # Nothing of the offer is kept: the next lookup asks the gateway again.
    assert engine._cred_info is None
    assert engine._agent_credential is None
    assert engine._registration_known_pending() is True
    if reason == "enrolment_redirected":
        assert [seen.runtime_key for seen in gateway.on(ENROLL)] == ["provisional", "provisional"]
    else:
        assert ENROLL not in gateway.paths()
    assert CHALLENGE not in gateway.paths() and SESSION not in gateway.paths()


@pytest.mark.parametrize("state", ["Pending", " PENDING ", "pending\n"], ids=["capitalised", "padded", "newline"])
def test_the_offer_is_recognised_whatever_the_case_of_its_state(shield_factory, monkeypatch, tmp_path, state):
    gateway = Gateway(asks_first=False)
    gateway.block_changes = {"registration_state": state}
    _without_runtime_extra(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway)

    filed = registry.discover(engine, manifest=dict(MANIFEST), sync=True)
    error = _gated_call(engine)

    assert filed["registration_state"] == "pending"
    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == PLAIN_PENDING
    assert len(gateway.sightings) == 1


@pytest.mark.parametrize("notify", ["", "dev@corp.com"])
@pytest.mark.parametrize(
    "reason",
    [name for name in NOT_CAPABLE if name != "key_directory_unusable"]
    + [pytest.param("key_directory_unusable", marks=needs_runtime_extra)],
)
def test_an_unusable_offer_leaves_the_client_where_a_pending_answer_does(
    shield_factory, monkeypatch, tmp_path, caplog, reason, notify,
):
    # One client that cannot hold a runtime key, the same steps twice: against
    # a gateway that answers its lookup with 409, and against one that makes
    # the offer to every client. Requests, outcomes, the printed line and what
    # the engine keeps must not tell the two apart.
    monkeypatch.setattr(engine_module, "_CREDENTIAL_FAILURE_TTL_SECONDS", 60.0)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    def run_one(asks_first: bool) -> dict:
        registry._enrolment_logged.clear()
        engine_module._credential_failure_logged.clear()
        caplog.clear()
        gateway = Gateway(asks_first=asks_first, notify=notify)
        client_kwargs = UNUSABLE[reason].arrange(monkeypatch, tmp_path, gateway)
        engine = _engine(shield_factory, gateway, registered_by="dev@corp.com", **client_kwargs)
        outcomes: list = []
        gate.preflight(engine, "read_account")
        for _ in range(2):
            error = _gated_call(engine)
            outcomes.append((error.code, str(public_agentic_error(error)), error.payload.get("registration_context")))
        outcomes.append(registry._post_discover(engine, dict(MANIFEST), timeout=1.0).get("registration_state"))
        outcomes.append(engine.agent_token())
        outcomes.append(_gated_call(engine).code)
        return {
            "outcomes": outcomes,
            "lines": list(caplog.messages),
            "requests": [tuple(seen) for seen in gateway.seen],
            "sightings": gateway.sightings,
            "kept": (
                engine._cred_info, engine._agent_credential, engine._cred_info_failure_code,
                engine._runtime_key_thumbprint, engine._registration_known_pending(),
                engine._registration_context(), engine._runtime_key_capable(),
            ),
        }

    answered, offered = run_one(asks_first=True), run_one(asks_first=False)

    pending = PLAIN_PENDING + (f" You will be notified at {notify}." if notify else "")
    assert answered["outcomes"][0][:2] == ("agent_registration_pending", pending)
    assert answered["lines"] == [f"DeepintShield: {pending}"]
    assert answered["kept"][:3] == (None, None, "agent_registration_pending")
    assert offered == answered


@pytest.mark.parametrize("reason", _reasons(without_the_extra=NOT_CAPABLE))
def test_the_fallback_serves_the_first_run_of_a_new_agent(shield_factory, monkeypatch, tmp_path, caplog, reason):
    case = UNUSABLE[reason]
    gateway = Gateway(asks_first=case.capable, notify="dev@corp.com")
    client_kwargs = case.arrange(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway, registered_by="dev@corp.com", **client_kwargs)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # The first governed call of a never-seen agent: proof-less capture first.
    gate.preflight(engine, "read_account")
    error = _gated_call(engine)
    again = _gated_call(engine)

    pending = f"{PLAIN_PENDING} You will be notified at dev@corp.com."
    assert len(gateway.sightings) == 1
    assert str(public_agentic_error(error)) == pending
    assert str(public_agentic_error(again)) == pending
    assert caplog.messages == [f"DeepintShield: {pending}"]
    assert gateway.paths()[0] == DISCOVER


def test_transparent_traffic_goes_on_without_a_token_while_the_offer_is_unusable(
    shield_factory, monkeypatch, tmp_path, caplog,
):
    gateway = Gateway(asks_first=False)
    _without_runtime_extra(monkeypatch, tmp_path, gateway)
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    engine = shield.agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # As with a gateway that answers the pending lookup itself.
    assert engine.agent_token() is None
    assert engine.agent_token() is None
    assert "X-Agent-Token" not in shield.create_headers(identity=True)

    assert caplog.messages == [f"DeepintShield: {PLAIN_PENDING}"]
    assert set(gateway.paths()) == {CREDENTIAL_INFO}


@needs_runtime_extra
def test_transparent_traffic_goes_on_when_the_provisional_key_cannot_be_enrolled(shield_factory, monkeypatch, tmp_path, caplog):
    gateway = Gateway()
    _with_redirected_enrolment(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    assert engine.agent_token() is None

    assert caplog.messages == [f"DeepintShield: {PLAIN_PENDING}"]
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL]
    assert engine._agent_credential is None


# ── the same failures still stop every other identity ─────────────────────


NOT_AN_OFFER = {
    # An administrator declared the identity of a registration still pending.
    "declared_pending": {"provisional": None, "registration_id": None, "registration_state": None},
    # An approved agent.
    "approved": {
        "provisional": None, "registration_id": None, "registration_state": None, "credential_bootstrap": None,
    },
    "explicitly_not_provisional": {"provisional": False},
    # Both statements are needed: a provisional key of a pending registration.
    "provisional_but_not_pending": {"registration_state": "approved"},
    "provisional_without_a_state": {"registration_state": None},
}


@pytest.mark.parametrize("block", list(NOT_AN_OFFER))
@pytest.mark.parametrize(
    "reason",
    # Without the extra its absence is the first thing a key directory meets.
    _reasons(without_the_extra=("runtime_extra_missing", "gateway_address_not_accepted")),
)
def test_the_same_failures_still_stop_a_declared_or_approved_identity(
    shield_factory, monkeypatch, tmp_path, caplog, reason, block,
):
    case = UNUSABLE[reason]
    gateway = Gateway(asks_first=False)
    gateway.block_changes = dict(NOT_AN_OFFER[block])
    client_kwargs = case.arrange(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway, **client_kwargs)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    error = _gated_call(engine)
    report = registry._post_discover(engine, dict(MANIFEST), timeout=1.0)

    assert error.code == case.code
    assert "registration_context" not in error.payload
    assert report == {"dispatched": True, "error": case.code}
    assert gateway.sightings == []
    assert DISCOVER not in gateway.paths()
    assert "Registration pending" not in caplog.text
    # The block is kept; it was not read as a pending answer.
    assert engine._cred_info is not None
    assert engine._cred_info_failure_code == ""


@pytest.mark.parametrize("block", ["declared_pending", "approved"])
def test_transparent_traffic_stays_fail_closed_for_a_declared_or_approved_broker_identity(
    shield_factory, monkeypatch, tmp_path, block,
):
    gateway = Gateway(asks_first=False)
    gateway.block_changes = dict(NOT_AN_OFFER[block])
    _without_runtime_extra(monkeypatch, tmp_path, gateway)
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")

    with pytest.raises(GovernanceConfigurationError) as caught:
        shield._agent_token()
    assert caught.value.code == "credential_dependency_missing"


@needs_runtime_extra
@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        # An authentication refusal is the one enrolment failure that makes
        # the engine ask once more (see the re-read tests below).
        (409, {"error": {"code": "agent_runtime_conflict"}}, "agent_runtime_conflict"),
        (503, {"error": {"code": "agent_runtime_unavailable"}}, "agent_runtime_unavailable"),
        (200, {"binding_id": "binding-provisional", "key_thumbprint": "another-key", "status": "pending"},
         "agent_runtime_response_invalid"),
    ],
)
def test_other_enrolment_failures_of_a_provisional_key_are_not_read_as_pending(
    shield_factory, caplog, status, body, code,
):
    gateway = Gateway()
    gateway.answers[ENROLL] = lambda request: httpx.Response(status, json=body)
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    error = _gated_call(engine)

    assert error.code == code
    assert "Registration pending" not in caplog.text
    assert engine._cred_info is not None and engine._agent_credential is not None
    # Neither read as pending nor asked again.
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL]


# ── a provisional key the gateway refuses: one re-read ────────────────────

REFUSED = {"error": {"code": "agent_runtime_authentication_failed"}}
APPROVED_VK_TRUSTED = {"agent_subject": AGENT, "agent_configured": False}


def _refuse(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json=REFUSED)


def _approved_key(gateway: Gateway):
    def approved(request: httpx.Request) -> httpx.Response:
        gateway.jkt = _jwk_thumbprint(_json(request)["public_jwk"])
        return httpx.Response(200, json={
            "binding_id": "binding-one", "key_thumbprint": gateway.jkt, "status": "approved", "enabled": True,
        })

    return approved


def _answers(*responses):
    """A path answered in turn by ``responses``, the last one repeated."""
    remaining = list(responses)

    def answer(request: httpx.Request) -> httpx.Response:
        response = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return response(request) if callable(response) else httpx.Response(200, json=response)

    return answer


@needs_runtime_extra
@pytest.mark.parametrize("endpoint", ["enroll", "challenge"])
def test_a_provisional_key_refused_by_enrolment_or_challenge_is_reread_once(shield_factory, caplog, endpoint):
    # The administrator approved the agent as VK-trusted after this process
    # read the offer: the gateway no longer accepts the provisional key, and
    # credential-info now answers with the agent subject alone.
    gateway = Gateway()
    gateway.answers[CREDENTIAL_INFO] = _answers(lambda request: httpx.Response(200, json=gateway.block()), APPROVED_VK_TRUSTED)
    if endpoint == "enroll":
        gateway.answers[ENROLL] = _refuse
    else:
        gateway.answers[ENROLL] = _approved_key(gateway)
        gateway.answers[CHALLENGE] = _refuse
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    headers = engine._headers(include_agent_token=True, path=DECIDE)

    # The call goes on with the virtual key and the agent selector.
    assert "X-Agent-Token" not in headers and headers["X-Agent-Subject"] == AGENT
    expected = [CREDENTIAL_INFO, ENROLL] + ([CHALLENGE] if endpoint == "challenge" else []) + [CREDENTIAL_INFO]
    assert gateway.paths() == expected
    # The new answer replaced the cached offer; nothing of the offer is kept.
    assert engine._cred_info.provisional is False and engine._cred_info.agent_configured is False
    assert engine._agent_credential is None and engine._provisional_credential is None
    assert engine._runtime_key_thumbprint == ""
    assert caplog.messages == []
    # Later calls use the cached new answer: no request at all.
    assert engine.agent_token(fail_closed=True) is None
    assert engine.agent_token() is None
    assert gateway.paths() == expected


@needs_runtime_extra
def test_a_repeated_provisional_offer_after_a_refusal_reads_as_plain_pending(shield_factory, caplog):
    # The gateway refuses the key yet repeats the same offer: the registration
    # is pending, the key cannot be cited, and nothing is tried a second time.
    gateway = Gateway(notify="dev@corp.com")
    gateway.answers[ENROLL] = _refuse
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    error = _gated_call(engine)

    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == PLAIN_PENDING
    assert "SHA256" not in str(public_agentic_error(error))
    assert caplog.messages == [f"DeepintShield: {PLAIN_PENDING}"]
    # Exactly one re-read and one enrolment.
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL, CREDENTIAL_INFO]
    assert engine._cred_info is None and engine._agent_credential is None
    assert engine._cred_info_failure_code == "agent_registration_pending"
    assert engine._registration_known_pending() is True


@needs_runtime_extra
def test_after_a_refusal_the_call_continues_with_a_plainly_pending_answer(shield_factory, caplog):
    gateway = Gateway(notify="dev@corp.com")
    gateway.answers[CREDENTIAL_INFO] = _answers(lambda request: httpx.Response(200, json=gateway.block()), lambda request: gateway.pending())
    gateway.answers[ENROLL] = _refuse
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    error = _gated_call(engine)

    pending = f"{PLAIN_PENDING} You will be notified at dev@corp.com."
    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == pending
    assert caplog.messages == [f"DeepintShield: {pending}"]
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL, CREDENTIAL_INFO]


@needs_runtime_extra
def test_after_a_refusal_the_call_continues_with_a_declared_identity(shield_factory, caplog):
    # An administrator declared the identity meanwhile: the new answer is a
    # declared block, its key is enrolled through it and waits for
    # authorization, which stops the call as it always did.
    gateway = Gateway()
    declared = {**gateway.block(), "provisional": False}
    declared.pop("registration_id")
    declared.pop("registration_state")
    gateway.answers[CREDENTIAL_INFO] = _answers(lambda request: httpx.Response(200, json=gateway.block()), declared)
    enrolments = []

    def enroll(request: httpx.Request) -> httpx.Response:
        enrolments.append(request.headers.get(HEADER))
        if len(enrolments) == 1:
            return _refuse(request)
        gateway.jkt = _jwk_thumbprint(_json(request)["public_jwk"])
        return httpx.Response(200, json={
            "binding_id": "binding-declared", "key_thumbprint": gateway.jkt, "status": "pending",
            "enabled": False, "provisional": False, "registration_id": "",
        })

    gateway.answers[ENROLL] = enroll
    engine = _engine(shield_factory, gateway)

    error = _gated_call(engine)

    assert error.code == "agent_runtime_enrollment_pending"
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL, CREDENTIAL_INFO, ENROLL]
    assert engine._cred_info.provisional is False
    assert engine._provisional_credential is None
    assert isinstance(engine._agent_credential, GatewayBrokerCredential)


@needs_runtime_extra
@pytest.mark.parametrize(
    "failure",
    ["session_refused", "capacity", "conflict", "unavailable", "outage", "enrolment_pending"],
)
def test_only_an_authentication_refusal_at_enrolment_or_challenge_is_reread(shield_factory, monkeypatch, failure):
    gateway = Gateway()
    monkeypatch.setattr(gateway_broker, "_sleep", lambda seconds: None)
    expected_code = {
        "session_refused": "agent_runtime_authentication_failed",
        "capacity": "agent_runtime_capacity_exceeded",
        "conflict": "agent_runtime_conflict",
        "unavailable": "agent_runtime_unavailable",
        "outage": "agent_runtime_exchange_failed",
        "enrolment_pending": "agent_runtime_enrollment_pending",
    }[failure]
    if failure == "session_refused":
        gateway.answers[ENROLL] = _approved_key(gateway)
        gateway.answers[CHALLENGE] = lambda request: httpx.Response(200, json={
            "binding_id": "binding-one", "nonce": "nonce-one", "expires_in": 60,
        })
        gateway.answers[SESSION] = _refuse
    elif failure == "capacity":
        gateway.answers[ENROLL] = lambda request: httpx.Response(429, headers={"Retry-After": "0"}, json={})
    elif failure == "conflict":
        gateway.answers[ENROLL] = lambda request: httpx.Response(409, json={"error": {"code": "agent_runtime_conflict"}})
    elif failure == "unavailable":
        gateway.answers[ENROLL] = lambda request: httpx.Response(503, json={"error": {"code": "agent_runtime_unavailable"}})
    elif failure == "outage":
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        gateway.answers[ENROLL] = unreachable
    engine = _engine(shield_factory, gateway)

    with pytest.raises(Exception) as caught:
        engine._headers(include_agent_token=True, path=DECIDE)

    assert getattr(caught.value, "code", "") == expected_code
    # credential-info was read once: no failure but the refusal asks again.
    assert gateway.paths().count(CREDENTIAL_INFO) == 1
    assert engine._cred_info is not None and engine._cred_info.provisional is True


@needs_runtime_extra
def test_a_refusal_of_a_key_built_for_a_declared_identity_is_not_reread(shield_factory):
    gateway = Gateway()
    gateway.block_changes = dict(NOT_AN_OFFER["declared_pending"])
    gateway.answers[ENROLL] = _refuse
    engine = _engine(shield_factory, gateway)

    assert _gated_call(engine).code == "agent_runtime_authentication_failed"
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL]


def test_a_refusal_of_a_supplied_credential_is_not_reread(shield_factory):
    gateway = Gateway(asks_first=False)

    class Supplied:
        provider_type = "gateway_broker"
        enrolled = False

        def get_token(self):
            error = GovernanceConfigurationError(code="agent_runtime_authentication_failed")
            error.runtime_endpoint = "enroll"
            raise error

    engine = _engine(shield_factory, gateway)
    assert engine.credential_info.provisional is True
    engine.set_agent_credential(Supplied())

    assert _gated_call(engine).code == "agent_runtime_authentication_failed"
    assert gateway.paths() == [CREDENTIAL_INFO]


@needs_runtime_extra
def test_transparent_traffic_after_a_refused_provisional_key_rereads_once(shield_factory, caplog):
    gateway = Gateway()
    gateway.answers[CREDENTIAL_INFO] = _answers(lambda request: httpx.Response(200, json=gateway.block()), APPROVED_VK_TRUSTED)
    gateway.answers[ENROLL] = _refuse
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    assert shield._agent_token() is None
    assert "X-Agent-Token" not in shield.create_headers(identity=True)

    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL, CREDENTIAL_INFO]
    assert caplog.messages == []


@needs_runtime_extra
def test_a_concurrent_call_that_met_the_same_refusal_reads_the_answer_left_behind(shield_factory):
    gateway = Gateway()
    gateway.answers[CREDENTIAL_INFO] = _answers(lambda request: httpx.Response(200, json=gateway.block()), APPROVED_VK_TRUSTED)
    engine = _engine(shield_factory, gateway)
    built = engine.agent_credential
    assert built is engine._provisional_credential
    refusal = GovernanceConfigurationError(code="agent_runtime_authentication_failed")
    refusal.runtime_endpoint = "enroll"

    # The first call sets the offer aside and reads the gateway again.
    assert engine._set_aside_refused_provisional_key(built, refusal) is True
    engine._reread_after_refused_provisional_key()
    # A call that still held the same credential when the gateway refused it
    # reads the answer the first one left: no further request.
    assert engine._set_aside_refused_provisional_key(built, refusal) is True
    engine._reread_after_refused_provisional_key()
    assert gateway.paths() == [CREDENTIAL_INFO, CREDENTIAL_INFO]
    assert engine.agent_token(fail_closed=True) is None
    # A credential that was never the offer, or one the application hands
    # back, is not read this way.
    other = GatewayBrokerCredential(
        http=engine._http, gateway_url=ORIGIN, virtual_key="sk-ds-test", agent_subject=AGENT,
        broker=RuntimeBrokerInfo(**_broker()),
    )
    assert engine._set_aside_refused_provisional_key(other, refusal) is False
    engine.set_agent_credential(built)
    assert engine._set_aside_refused_provisional_key(built, refusal) is False


@needs_runtime_extra
def test_concurrent_transparent_calls_meeting_one_refusal_both_go_on(shield_factory):
    gateway = Gateway()
    gateway.answers[CREDENTIAL_INFO] = _answers(lambda request: httpx.Response(200, json=gateway.block()), APPROVED_VK_TRUSTED)
    gateway.answers[ENROLL] = _refuse
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    engine = shield.agentic.engine
    assert isinstance(engine.agent_credential, GatewayBrokerCredential)
    together = threading.Barrier(2)

    def transparent(_index):
        together.wait(10)
        return shield._agent_token()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(transparent, range(2)))

    assert results == [None, None]
    assert gateway.paths().count(CREDENTIAL_INFO) == 2
    assert engine._cred_info.provisional is False


@needs_runtime_extra
def test_transparent_traffic_after_a_refusal_and_a_repeated_offer_goes_on_without_a_token(shield_factory, caplog):
    gateway = Gateway()
    gateway.answers[ENROLL] = _refuse
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # As with a gateway that answers the pending lookup itself.
    assert shield._agent_token() is None

    assert caplog.messages == [f"DeepintShield: {PLAIN_PENDING}"]
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL, CREDENTIAL_INFO]


@needs_runtime_extra
def test_a_transport_failure_while_enrolling_is_not_read_as_pending(shield_factory):
    gateway = Gateway()

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    gateway.answers[ENROLL] = unreachable
    engine = _engine(shield_factory, gateway)

    with pytest.raises(Exception) as caught:
        engine._headers(include_agent_token=True, path=DECIDE)
    assert getattr(caught.value, "code", "") == "agent_runtime_exchange_failed"
    assert gateway.sightings == []


@needs_runtime_extra
def test_a_key_the_gateway_accepted_is_never_read_as_pending(shield_factory, caplog):
    # The block still says provisional, but the key has been authorized since:
    # what fails after enrolment belongs to an approved binding.
    gateway = Gateway()

    def approved(request: httpx.Request) -> httpx.Response:
        gateway.jkt = _jwk_thumbprint(_json(request)["public_jwk"])
        return httpx.Response(200, json={
            "binding_id": "binding-one", "key_thumbprint": gateway.jkt, "status": "approved", "enabled": True,
        })

    gateway.answers[ENROLL] = approved
    gateway.answers[CHALLENGE] = _redirect
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    error = _gated_call(engine)

    assert error.code == "agent_runtime_configuration_error"
    assert engine.agent_credential.enrolled is True
    assert "Registration pending" not in caplog.text
    assert [(seen.path, seen.runtime_key) for seen in gateway.seen] == [
        (CREDENTIAL_INFO, "provisional"), (ENROLL, "provisional"), (CHALLENGE, None),
    ]


@needs_runtime_extra
def test_a_credential_built_for_a_declared_identity_is_never_read_as_pending(shield_factory, monkeypatch, tmp_path):
    # Built while the identity was declared. If the gateway's answer later
    # turns into a provisional offer, the credential in hand is still the
    # declared identity's, and its failure is not a registration state.
    gateway = Gateway()
    gateway.block_changes = dict(NOT_AN_OFFER["declared_pending"])
    engine = _engine(shield_factory, gateway)
    declared = engine.agent_credential
    assert isinstance(declared, GatewayBrokerCredential)

    gateway.block_changes = {}
    engine._cred_info_fetched_at = 0.0
    assert engine.credential_info.provisional is True
    assert engine.agent_credential is declared
    _with_redirected_enrolment(monkeypatch, tmp_path, gateway)

    assert _gated_call(engine).code == "agent_runtime_configuration_error"
    assert engine._agent_credential is declared
    assert engine._cred_info is not None


def test_a_supplied_credential_is_never_read_as_pending(shield_factory):
    gateway = Gateway(asks_first=False)

    class Supplied:
        provider_type = "static"
        enrolled = False

        def get_token(self):
            raise GovernanceConfigurationError(code="agent_runtime_key_unavailable")

    engine = _engine(shield_factory, gateway)
    assert engine.credential_info.provisional is True
    engine.set_agent_credential(Supplied())

    assert _gated_call(engine).code == "agent_runtime_key_unavailable"
    assert engine._cred_info is not None


@needs_runtime_extra
def test_a_credential_the_application_supplies_again_is_no_longer_the_offer(shield_factory, monkeypatch, tmp_path):
    # Built from the offer, then handed back through the override: from then
    # on it is the application's choice, and its failures are its own.
    gateway = Gateway()
    engine = _engine(shield_factory, gateway)
    assert _gated_call(engine).code == "agent_runtime_enrollment_pending"
    built = engine.agent_credential
    assert isinstance(built, GatewayBrokerCredential) and built.enrolled is False

    engine.set_agent_credential(built)
    _with_redirected_enrolment(monkeypatch, tmp_path, gateway)

    assert _gated_call(engine).code == "agent_runtime_configuration_error"
    assert engine._agent_credential is built
    assert engine._cred_info is not None


def test_after_the_fallback_the_next_lookup_sees_the_administrators_decision(shield_factory, monkeypatch, tmp_path):
    gateway = Gateway(asks_first=False)
    _with_unusable_key_directory(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway)
    # Hold the short negative cache open so the count below does not depend on timing.
    monkeypatch.setattr(engine_module, "_CREDENTIAL_FAILURE_TTL_SECONDS", 60.0)

    assert _gated_call(engine).code == "agent_registration_pending"
    # Within the short negative cache the answer is repeated without a request,
    # exactly as after a pending answer from the gateway.
    assert _gated_call(engine).code == "agent_registration_pending"
    assert gateway.paths() == [CREDENTIAL_INFO]

    # Approved as VK-trusted: there is no workload credential to present.
    gateway.answers[CREDENTIAL_INFO] = lambda request: httpx.Response(
        200, json={"provider_type": "", "agent_configured": False, "agent_subject": AGENT},
    )
    engine._clear_credential_failure_cache()

    assert engine.agent_token(fail_closed=True) is None
    assert engine._headers(include_agent_token=True, path=DECIDE).get("X-Agent-Token") is None
    assert gateway.paths() == [CREDENTIAL_INFO, CREDENTIAL_INFO]


@needs_runtime_extra
def test_the_fallback_forgets_a_runtime_key_this_process_can_no_longer_cite(shield_factory, monkeypatch, tmp_path):
    # Enrolled and pending first; then the same key cannot be enrolled again.
    gateway = Gateway(notify="dev@corp.com")
    engine = _engine(shield_factory, gateway)

    first = _gated_call(engine)
    assert first.code == "agent_runtime_enrollment_pending"
    assert f"runtime key SHA256:{gateway.jkt}" in str(public_agentic_error(first))
    assert engine._runtime_key_thumbprint == gateway.jkt

    _with_redirected_enrolment(monkeypatch, tmp_path, gateway)
    second = _gated_call(engine)

    assert second.code == "agent_registration_pending"
    assert str(public_agentic_error(second)) == PLAIN_PENDING
    assert engine._runtime_key_thumbprint == ""
    assert engine._registration_context()["key_thumbprint"] == ""


# ── run 1, end to end ─────────────────────────────────────────────────────


@needs_runtime_extra
def test_first_run_with_the_runtime_extra_cites_the_enrolled_runtime_key(shield_factory, tmp_path, caplog):
    gateway = Gateway(notify="dev@corp.com")
    engine = _engine(shield_factory, gateway, registered_by="dev@corp.com")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    gate.preflight(engine, "read_account")
    error = _gated_call(engine)

    message = str(public_agentic_error(error))
    assert error.code == "agent_runtime_enrollment_pending"
    assert re.fullmatch(
        r"agent_runtime_enrollment_pending: Registration pending\. "
        rf"Reference {REGISTRATION_ID} · runtime key SHA256:[A-Za-z0-9_-]{{43}}\. "
        r"Contact your administrator\. You will be notified at dev@corp\.com\.",
        message,
    ), message
    assert f"runtime key SHA256:{gateway.jkt}." in message
    assert caplog.messages == [f"DeepintShield: {message}"]
    # The header on the lookup and on each enrolment, and on nothing else.
    assert [(seen.path, seen.runtime_key) for seen in gateway.seen] == [
        (DISCOVER, None), (CREDENTIAL_INFO, "provisional"), (ENROLL, "provisional"), (ENROLL, "provisional"),
    ]
    assert gateway.sightings[0]["registered_by_email"] == "dev@corp.com"
    # The key is durable: one descriptor, reused by the next run.
    assert len(list((tmp_path / "keys").glob("*.json"))) == 1
    assert engine.agent_credential.key_thumbprint == gateway.jkt


def test_first_run_without_the_runtime_extra_stays_on_the_earlier_flow(shield_factory, monkeypatch, tmp_path, caplog):
    gateway = Gateway(notify="dev@corp.com")
    _without_runtime_extra(monkeypatch, tmp_path, gateway)
    engine = _engine(shield_factory, gateway, registered_by="dev@corp.com")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    gate.preflight(engine, "read_account")
    error = _gated_call(engine)

    pending = f"{PLAIN_PENDING} You will be notified at dev@corp.com."
    assert error.code == "agent_registration_pending"
    assert str(public_agentic_error(error)) == pending
    assert caplog.messages == [f"DeepintShield: {pending}"]
    assert set(gateway.paths()) == {DISCOVER, CREDENTIAL_INFO}
    assert all(seen.runtime_key is None for seen in gateway.seen)
    assert _key_files(tmp_path) == []


def _run_new_langgraph_agent(shield_factory, gateway) -> tuple[RuntimeError, int]:
    """Run 1 of a real LangGraph application; returns its error and the nodes executed."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    gateway.answers["/api/agentic-security/blueprints"] = lambda request: httpx.Response(
        200, json={"blueprint_id": "blueprint-test"},
    )
    executed = 0
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com")
    shield.agentic.enforce()

    class State(TypedDict):
        value: int

    def node(state: State) -> State:
        nonlocal executed
        executed += 1
        return state

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    app = graph.compile()

    with pytest.raises(RuntimeError) as caught:
        app.invoke({"value": 1})
    return caught.value, executed


@needs_runtime_extra
def test_first_langgraph_run_with_the_runtime_extra_cites_the_enrolled_runtime_key(shield_factory):
    pytest.importorskip("langgraph")
    gateway = Gateway(notify="dev@corp.com")

    error, executed = _run_new_langgraph_agent(shield_factory, gateway)

    assert executed == 0
    assert error._deepintshield_error_code == "agent_runtime_enrollment_pending"
    assert re.fullmatch(
        r"agent_runtime_enrollment_pending: Registration pending\. "
        rf"Reference {REGISTRATION_ID} · runtime key SHA256:[A-Za-z0-9_-]{{43}}\. "
        r"Contact your administrator\. You will be notified at dev@corp\.com\.",
        str(error),
    ), str(error)
    assert f"runtime key SHA256:{gateway.jkt}." in str(error)
    assert {seen.runtime_key for seen in gateway.on(CREDENTIAL_INFO)} == {"provisional"}
    assert {seen.runtime_key for seen in gateway.on(ENROLL)} == {"provisional"}
    assert all(seen.runtime_key is None for seen in gateway.seen if seen.path not in (CREDENTIAL_INFO, ENROLL))
    assert gateway.sightings and all(seen.agent_token == "" for seen in gateway.on(DISCOVER))


@pytest.mark.parametrize("asks_first", [True, False])
def test_first_langgraph_run_without_the_runtime_extra_reads_the_plain_pending_line(
    shield_factory, monkeypatch, tmp_path, asks_first,
):
    pytest.importorskip("langgraph")
    # Whether the gateway waits to be asked or makes its offer to every client,
    # a client without the extra ends run 1 the way it always did.
    gateway = Gateway(asks_first=asks_first, notify="dev@corp.com")
    _without_runtime_extra(monkeypatch, tmp_path, gateway)

    error, executed = _run_new_langgraph_agent(shield_factory, gateway)

    assert executed == 0
    assert error._deepintshield_error_code == "agent_registration_pending"
    assert str(error) == f"{PLAIN_PENDING} You will be notified at dev@corp.com."
    assert all(seen.runtime_key is None for seen in gateway.seen)
    assert ENROLL not in gateway.paths()
    assert gateway.sightings and all(seen.agent_token == "" for seen in gateway.on(DISCOVER))
    assert _key_files(tmp_path) == []
