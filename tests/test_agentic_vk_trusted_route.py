"""A plain (VK-trusted) agent works in two runs, with or without a run-one key.

Run 1 of a new agent is pending: the gateway offers a client that can hold one
a provisional runtime key, which the SDK enrols, and the developer reads the
pending line. An administrator approves the agent as VK-trusted (no identity
provider); the gateway retires the unused key. Run 2 executes on the virtual
key alone.

Two gateway statements keep that true:

* ``credential_optional`` on the credential-info answer of an approved agent
  without a provider that still has an identity declared under a provider with
  gateway exchange (the documented "VK-trusted now, verified identity later").
  A failure to build that credential or obtain its token does not stop the
  call: no token is sent, one line is logged per process, and the failure is
  not retried for a minute. Only the flag does this; without it the call
  stops exactly as before.
* ``agent_runtime_authentication_failed`` from the enrolment or challenge
  endpoint for a key built from a provisional offer: the offer is set aside
  and credential-info is asked once more, so a process that kept running past
  the approval is not stuck on its hour-long cached offer.
"""
from __future__ import annotations

import base64
import builtins
import hashlib
import json
import logging
import os
from collections import namedtuple

import httpx
import pytest

from deepintshield.agentic import engine as engine_module
from deepintshield.agentic import gate, registry
from deepintshield.agentic.credentials import runtime_key
from deepintshield.agentic.credentials.gateway_broker import GatewayBrokerCredential
from deepintshield.agentic.errors import GovernanceConfigurationError, public_agentic_error
from deepintshield.agentic.types import DelegationContext, VKCredentialInfo

ORIGIN = "https://gateway.example"
HEADER = "X-DeepintShield-Runtime-Key"
CREDENTIAL_INFO = "/api/agentic-new/credential-info"
DISCOVER = "/api/agentic-new/registry/discover"
DECIDE = "/api/agentic-new/decide"
BLUEPRINTS = "/api/agentic-security/blueprints"
PREFIX = "/api/agentic-new/runtime-auth/"
ENROLL, CHALLENGE, SESSION = PREFIX + "enroll", PREFIX + "challenge", PREFIX + "session"
AGENT = "agent:math-agent"
CHILD = "11111111-2222-4333-8444-555555555555"
REGISTRATION_ID = "agnew-rr-0123456789abcdef"
SESSION_TOKEN = "dsrs_" + "b" * 43
RUNS_ON_ITS_KEY = "DeepintShield: this agent runs on its virtual key; its declared identity is not active yet"
MANIFEST = {
    "network": {"key": "math-agent", "name": "math-agent"},
    "agents": [{"key": "math-agent", "name": "math-agent"}],
    "tools": [],
    "edges": [],
}
Seen = namedtuple("Seen", "path runtime_key agent_token")

try:
    from cryptography.hazmat.primitives.asymmetric import ec as _ec  # noqa: F401

    _RUNTIME_EXTRA = True
except ImportError:  # the SDK installed without its ``runtime`` extra
    _RUNTIME_EXTRA = False
needs_runtime_extra = pytest.mark.skipif(not _RUNTIME_EXTRA, reason="needs deepintshield[runtime]")


@pytest.fixture(autouse=True)
def isolated_process_state(tmp_path, monkeypatch):
    """Once-per-process guards and runtime-key storage never leak between tests."""
    monkeypatch.setattr(registry, "_enrolment_logged", set())
    monkeypatch.setattr(engine_module, "_credential_failure_logged", set())
    monkeypatch.setattr(engine_module, "_optional_credential_logged", set())
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


def _without_runtime_extra(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("PRIVATE: no module named cryptography")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


class RouteGateway:
    """One agent's registration as the gateway sees it, from run 1 past its approval.

    ``state`` is ``pending`` until :meth:`approve_vk_trusted`. The workspace has
    an Entra provider with gateway exchange (``provisional_provider``), so a
    client that asks is offered a provisional key while nothing is declared.
    ``declared`` models an identity declared under that provider;
    ``optional_flag`` whether the approved, declared answer carries
    ``credential_optional`` (False is a gateway that predates it);
    ``key_authorized`` whether an administrator authorized the enrolled key.
    """

    def __init__(self, *, provisional_provider: bool = True, notify: str = "dev@corp.com") -> None:
        self.state = "pending"
        self.provisional_provider = provisional_provider
        self.notify = notify
        self.declared = False
        self.optional_flag = True
        self.key_authorized = False
        self.provisional_keys: set[str] = set()
        self.declared_keys: dict[str, str] = {}
        self.challenged = ""
        self.seen: list[Seen] = []
        self.sightings: list[dict] = []
        self.decisions: list[str] = []

    # ── what the administrator does ──────────────────────────────────────

    def approve_vk_trusted(self) -> None:
        self.state = "approved"
        # The approving transaction retires the unused run-one key.
        self.provisional_keys.clear()

    # ── what the developer's process sees ────────────────────────────────

    def paths(self) -> list[str]:
        return [seen.path for seen in self.seen]

    def tokens_sent(self) -> list[str]:
        return [seen.agent_token for seen in self.seen if seen.agent_token]

    def _registration(self) -> dict:
        row = {"id": REGISTRATION_ID, "agent_subject": AGENT, "status": "pending"}
        if self.notify:
            row["requester_email"] = self.notify
        return row

    def broker_block(self, **fields) -> dict:
        return {
            "runtime_auth_mode": "gateway_broker",
            "runtime_broker": {
                "enrollment_path": ENROLL, "challenge_path": CHALLENGE, "session_path": SESSION,
                "tenant_id": "platform-tenant", "workspace_id": "workspace-one",
                "agent_subject": AGENT, "runtime_kind": "key",
            },
            "provider_id": "idp-entra", "provider_type": "entra_agent_id",
            "tenant_id": "directory-tenant", "blueprint_client_id": "blueprint-client",
            "authority": "https://login.microsoftonline.com/directory-tenant/v2.0",
            "issuer": "https://login.microsoftonline.com/directory-tenant/v2.0",
            "gateway_audience": "api://deepintshield", "scopes": ["tools:invoke"],
            "require_proof_of_possession": False, "allow_cross_tenant": False,
            "agent_subject": AGENT, "agent_configured": True, "credential_bootstrap": True,
            **fields,
        }

    def credential_info(self, asked: bool) -> httpx.Response:
        if self.state == "pending":
            if self.declared:
                return httpx.Response(200, json=self.broker_block(agent_identity_client_id=CHILD))
            if self.provisional_provider and asked:
                return httpx.Response(200, json=self.broker_block(
                    provisional=True, registration_id=REGISTRATION_ID, registration_state="pending",
                ))
            body = {
                "error": {"code": "agent_registration_pending", "message": "PRIVATE prose"},
                "registration": {"id": REGISTRATION_ID, "agent_subject": AGENT, "status": "pending"},
                "reference": REGISTRATION_ID,
            }
            if self.notify:
                body["notify_email"] = self.notify
            return httpx.Response(409, json=body)
        if self.declared:
            fields: dict = {"agent_identity_client_id": CHILD}
            if self.optional_flag:
                fields.update(credential_optional=True, registration_state="approved")
            return httpx.Response(200, json=self.broker_block(**fields))
        # Approved as VK-trusted: the agent subject, and nothing to present.
        return httpx.Response(200, json={
            "agent_subject": AGENT, "require_proof_of_possession": False,
            "allow_cross_tenant": False, "agent_configured": False,
        })

    def enroll(self, request: httpx.Request, asked: bool) -> httpx.Response:
        jkt = _jwk_thumbprint(_json(request)["public_jwk"])
        if self.declared:
            status = "approved" if self.key_authorized else "pending"
            self.declared_keys[jkt] = status
            return httpx.Response(200, json={
                "binding_id": "binding-declared", "key_thumbprint": jkt, "status": status,
                "enabled": status == "approved", "runtime_kind": "key", "runtime_audience": "",
                "registration_id": "", "provisional": False,
            })
        if self.state == "pending" and self.provisional_provider and asked:
            self.provisional_keys.add(jkt)
            return httpx.Response(200, json={
                "binding_id": "binding-provisional", "key_thumbprint": jkt, "status": "pending",
                "enabled": False, "runtime_kind": "key", "runtime_audience": "",
                "registration_id": REGISTRATION_ID, "provisional": True,
            })
        # Nothing to enrol under: approved without a provider, or not asked.
        return httpx.Response(401, json={"error": {"code": "agent_runtime_authentication_failed"}})

    def challenge(self, request: httpx.Request) -> httpx.Response:
        jkt = _json(request).get("key_thumbprint", "")
        if self.declared and self.declared_keys.get(jkt) == "approved":
            self.challenged = jkt
            return httpx.Response(200, json={"binding_id": "binding-declared", "nonce": "nonce-one", "expires_in": 60})
        return httpx.Response(401, json={"error": {"code": "agent_runtime_authentication_failed"}})

    def session(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "binding_id": "binding-declared", "token_type": "DPoP", "cnf": {"jkt": self.challenged},
            "access_token": SESSION_TOKEN, "expires_in": 300,
        })

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        asked = request.headers.get(HEADER, "").strip().lower() == "provisional"
        token = request.headers.get("X-Agent-Token", "")
        self.seen.append(Seen(path, request.headers.get(HEADER), token))
        if path == CREDENTIAL_INFO:
            return self.credential_info(asked)
        if path == ENROLL:
            return self.enroll(request, asked)
        if path == CHALLENGE:
            return self.challenge(request)
        if path == SESSION:
            return self.session(request)
        if path == DISCOVER:
            self.sightings.append({"agent_token": token, **_json(request)})
            if self.state == "pending":
                assert not token, "a pending agent has no workload proof"
                return httpx.Response(202, json={
                    "network_id": "network:math", "registration_state": "pending",
                    "blueprint_scan_status": "complete", "registration": self._registration(),
                })
            return httpx.Response(200, json={"network_id": "network:math", "blueprint_scan_status": "complete"})
        if path == DECIDE:
            self.decisions.append(token)
            if self.state != "approved":
                return httpx.Response(403, json={"error": {"code": "agent_registration_pending"}})
            return httpx.Response(200, json={
                "verdict": "ALLOW", "allow": True, "proceed": True, "mode": "enforce",
                "decision_id": f"decision-{len(self.decisions)}",
            })
        if path == BLUEPRINTS:
            return httpx.Response(200, json={"blueprint_id": "blueprint-test"})
        return httpx.Response(404)


def _engine(shield_factory, gateway, **client_kwargs):
    client_kwargs.setdefault("base_url", ORIGIN)
    client_kwargs.setdefault("registered_by", "dev@corp.com")
    return shield_factory(gateway, agent_name="math-agent", **client_kwargs).agentic.engine


def _decide(engine):
    return engine.decide(DelegationContext(tool="add_two", args_digest="sha256:" + "0" * 64))


def _pending_call(engine) -> GovernanceConfigurationError:
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path=DECIDE)
    return caught.value


def _run_langgraph(shield_factory, gateway) -> tuple[object, int]:
    """One run of a real two-node LangGraph application; returns (result or error, nodes executed)."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    executed = 0
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com")
    shield.agentic.enforce()

    class State(TypedDict):
        number: int

    def add_two(state: State) -> State:
        nonlocal executed
        executed += 1
        return {"number": state["number"] + 2}

    def multiply_by_three(state: State) -> State:
        nonlocal executed
        executed += 1
        return {"number": state["number"] * 3}

    graph = StateGraph(State)
    graph.add_node("add_two", add_two)
    graph.add_node("multiply_by_three", multiply_by_three)
    graph.add_edge(START, "add_two")
    graph.add_edge("add_two", "multiply_by_three")
    graph.add_edge("multiply_by_three", END)
    try:
        app = graph.compile()
        try:
            return app.invoke({"number": 5}), executed
        except RuntimeError as error:
            return error, executed
    finally:
        # The process ends: the next run is a new client, as on a new start.
        shield.close()


# ── run 1, the approval, run 2 ────────────────────────────────────────────


@pytest.mark.parametrize("runtime_extra", [pytest.param(True, marks=needs_runtime_extra), False], ids=["with_key", "without_key"])
def test_vk_trusted_route_pending_then_approved_runs(shield_factory, monkeypatch, caplog, runtime_extra):
    if not runtime_extra:
        _without_runtime_extra(monkeypatch)
    gateway = RouteGateway()
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # Run 1: a new agent, pending.
    run_one = _engine(shield_factory, gateway)
    gate.preflight(run_one, "add_two")
    error = _pending_call(run_one)
    assert error.code == ("agent_runtime_enrollment_pending" if runtime_extra else "agent_registration_pending")
    line = str(public_agentic_error(error))
    assert f"Reference {REGISTRATION_ID}" in line and "You will be notified at dev@corp.com." in line
    assert ("runtime key SHA256:" in line) is runtime_extra
    assert len(gateway.provisional_keys) == (1 if runtime_extra else 0)

    # The sitting: approved as VK-trusted; the run-one key is retired.
    gateway.approve_vk_trusted()
    run_one_requests = len(gateway.seen)
    caplog.clear()

    # Run 2: a new process.
    run_two = _engine(shield_factory, gateway)
    report = registry._post_discover(run_two, dict(MANIFEST), timeout=1.0)
    decision = _decide(run_two)

    assert "error" not in report
    assert decision.verdict.value == "ALLOW"
    run_two_seen = gateway.seen[run_one_requests:]
    # The whole run goes out on the virtual key: no token, no enrolment.
    assert all(not seen.agent_token for seen in run_two_seen)
    assert ENROLL not in [seen.path for seen in run_two_seen]
    assert [seen.path for seen in run_two_seen].count(CREDENTIAL_INFO) == 1
    assert "Registration pending" not in caplog.text
    assert caplog.messages == []


@pytest.mark.parametrize("runtime_extra", [pytest.param(True, marks=needs_runtime_extra), False], ids=["with_key", "without_key"])
def test_vk_trusted_langgraph_agent_executes_whole_on_its_second_run(shield_factory, monkeypatch, runtime_extra):
    pytest.importorskip("langgraph")
    if not runtime_extra:
        _without_runtime_extra(monkeypatch)
    gateway = RouteGateway()

    error, executed = _run_langgraph(shield_factory, gateway)
    assert executed == 0
    assert isinstance(error, RuntimeError)
    assert "Registration pending" in str(error)

    gateway.approve_vk_trusted()
    start = len(gateway.seen)
    result, executed = _run_langgraph(shield_factory, gateway)

    assert result == {"number": 21}
    assert executed == 2
    assert all(not seen.agent_token for seen in gateway.seen[start:])
    assert ENROLL not in [seen.path for seen in gateway.seen[start:]]
    assert all(token == "" for token in gateway.decisions)


@needs_runtime_extra
def test_vk_trusted_langgraph_agent_with_a_declared_identity_executes_on_its_key(shield_factory, caplog):
    pytest.importorskip("langgraph")
    gateway = RouteGateway()
    gateway.state, gateway.declared = "approved", True
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    result, executed = _run_langgraph(shield_factory, gateway)

    assert result == {"number": 21}
    assert executed == 2
    assert gateway.tokens_sent() == []
    (jkt,) = gateway.declared_keys
    assert caplog.messages == [f"{RUNS_ON_ITS_KEY} (runtime key SHA256:{jkt})."]
    assert gateway.paths().count(ENROLL) == 1


@needs_runtime_extra
def test_one_process_runs_the_same_graph_before_and_after_the_approval(shield_factory, caplog):
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    gateway = RouteGateway()
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com")
    shield.agentic.enforce()
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    class State(TypedDict):
        number: int

    graph = StateGraph(State)
    graph.add_node("add_two", lambda state: {"number": state["number"] + 2})
    graph.add_node("multiply_by_three", lambda state: {"number": state["number"] * 3})
    graph.add_edge(START, "add_two")
    graph.add_edge("add_two", "multiply_by_three")
    graph.add_edge("multiply_by_three", END)
    try:
        app = graph.compile()
        with pytest.raises(RuntimeError, match="Registration pending"):
            app.invoke({"number": 5})
        assert shield.agentic.engine._cred_info.provisional is True

        gateway.approve_vk_trusted()
        before = len(gateway.seen)

        assert app.invoke({"number": 5}) == {"number": 21}
        assert app.invoke({"number": 1}) == {"number": 9}
    finally:
        shield.close()

    after = [seen.path for seen in gateway.seen[before:]]
    assert after.count(ENROLL) == 1 and after.count(CREDENTIAL_INFO) == 1
    assert all(not seen.agent_token for seen in gateway.seen[before:])
    assert shield.agentic.engine._cred_info.provisional is False


@needs_runtime_extra
def test_run_two_with_a_leftover_provisional_offer_rereads_and_runs(shield_factory, caplog):
    # The approval lands between run 2's credential lookup and its enrolment:
    # the offer it read is a leftover, and the gateway refuses its key.
    gateway = RouteGateway()
    original = gateway.credential_info

    def offer_then_approve(asked: bool) -> httpx.Response:
        response = original(asked)
        if gateway.state == "pending":
            gateway.approve_vk_trusted()
        return response

    gateway.credential_info = offer_then_approve
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    headers = engine._headers(include_agent_token=True, path=DECIDE)
    decision = _decide(engine)

    assert "X-Agent-Token" not in headers
    assert decision.verdict.value == "ALLOW"
    assert gateway.paths()[:3] == [CREDENTIAL_INFO, ENROLL, CREDENTIAL_INFO]
    assert gateway.paths().count(CREDENTIAL_INFO) == 2
    assert gateway.paths().count(ENROLL) == 1
    assert engine._cred_info.provisional is False
    assert caplog.messages == []


@needs_runtime_extra
def test_refused_provisional_key_is_reread_and_the_approved_agent_runs(shield_factory, caplog):
    # One process that keeps running: attempt, approval, attempt again. The
    # provisional offer sits in the hour-long credential-info cache.
    gateway = RouteGateway()
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    gate.preflight(engine, "add_two")
    first = _pending_call(engine)
    assert first.code == "agent_runtime_enrollment_pending"
    assert engine._cred_info.provisional is True
    assert isinstance(engine._agent_credential, GatewayBrokerCredential)

    gateway.approve_vk_trusted()
    before = len(gateway.seen)

    headers = engine._headers(include_agent_token=True, path=DECIDE)
    decision = _decide(engine)

    assert "X-Agent-Token" not in headers
    assert decision.verdict.value == "ALLOW"
    after_approval = [seen.path for seen in gateway.seen[before:]]
    # The refused key, the one re-read, then the decision on the virtual key.
    assert after_approval[:2] == [ENROLL, CREDENTIAL_INFO]
    assert after_approval.count(CREDENTIAL_INFO) == 1 and after_approval.count(ENROLL) == 1
    # The re-read replaced the cached offer.
    assert engine._cred_info.provisional is False and engine._cred_info.agent_configured is False
    assert engine._agent_credential is None
    # Later calls ask nothing more of the gateway than the decision itself.
    settled = len(gateway.seen)
    _decide(engine)
    assert [seen.path for seen in gateway.seen[settled:]] == [DECIDE]
    assert all(not seen.agent_token for seen in gateway.seen[before:])
    # Only run 1's pending line was printed.
    assert len(caplog.messages) == 1 and "Registration pending" in caplog.messages[0]


# ── an approved agent whose declared identity is not active yet ───────────


def _declared_and_approved(gateway: RouteGateway, *, optional: bool = True) -> RouteGateway:
    gateway.state = "approved"
    gateway.declared = True
    gateway.optional_flag = optional
    return gateway


@needs_runtime_extra
def test_vk_trusted_with_declared_identity_runs_on_the_virtual_key(shield_factory, caplog):
    # The key is enrolled and waits for authorization: the call goes on.
    gateway = _declared_and_approved(RouteGateway())
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    headers = engine._headers(include_agent_token=True, path=DECIDE)
    decision = _decide(engine)
    report = registry._post_discover(engine, dict(MANIFEST), timeout=1.0)

    assert "X-Agent-Token" not in headers and headers["X-Agent-Subject"] == AGENT
    assert decision.verdict.value == "ALLOW"
    assert "error" not in report
    assert gateway.tokens_sent() == []
    (jkt,) = gateway.declared_keys
    assert caplog.messages == [f"{RUNS_ON_ITS_KEY} (runtime key SHA256:{jkt})."]
    assert "Registration pending" not in caplog.text
    assert engine._registration_known_pending() is False
    # Not retried on every call: one enrolment for all of the above.
    assert gateway.paths().count(ENROLL) == 1
    assert gateway.paths().count(CREDENTIAL_INFO) == 1


@needs_runtime_extra
def test_an_optional_credential_is_tried_again_after_a_minute_and_used_once_it_works(shield_factory, caplog):
    gateway = _declared_and_approved(RouteGateway())
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    assert engine.agent_token(fail_closed=True) is None
    assert engine.agent_token(fail_closed=True) is None
    assert gateway.paths().count(ENROLL) == 1

    # A minute later the key is tried again; still pending, still no token.
    engine._optional_credential_failed_at -= engine_module._OPTIONAL_CREDENTIAL_RETRY_SECONDS + 1
    assert engine.agent_token(fail_closed=True) is None
    assert gateway.paths().count(ENROLL) == 2

    # The administrator authorizes the key: once the minute is over, the
    # declared identity is presented with its proof.
    gateway.key_authorized = True
    engine._optional_credential_failed_at -= engine_module._OPTIONAL_CREDENTIAL_RETRY_SECONDS + 1
    headers = engine._headers(include_agent_token=True, path=DECIDE)

    assert headers["X-Agent-Token"] == SESSION_TOKEN
    assert "DPoP" in headers
    assert gateway.paths()[-3:] == [ENROLL, CHALLENGE, SESSION]
    # One line for the whole process.
    assert len(caplog.messages) == 1 and caplog.messages[0].startswith(RUNS_ON_ITS_KEY)


@pytest.mark.parametrize(
    "unusable",
    [
        "runtime_extra_missing",
        pytest.param("gateway_address_not_accepted", marks=needs_runtime_extra),
        pytest.param("key_directory_unusable", marks=needs_runtime_extra),
    ],
)
def test_an_optional_credential_that_cannot_be_built_leaves_the_call_on_the_virtual_key(
    shield_factory, monkeypatch, tmp_path, caplog, unusable,
):
    gateway = _declared_and_approved(RouteGateway())
    client_kwargs = {}
    if unusable == "runtime_extra_missing":
        _without_runtime_extra(monkeypatch)
    elif unusable == "gateway_address_not_accepted":
        client_kwargs["base_url"] = "http://gateway.internal:8080"
    else:
        (tmp_path / "keys").write_text("not a directory")
    engine = _engine(shield_factory, gateway, **client_kwargs)
    builds = []
    real_build = engine._build_credential_for

    def counted(info):
        builds.append(info)
        return real_build(info)

    monkeypatch.setattr(engine, "_build_credential_for", counted)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    for _ in range(3):
        headers = engine._headers(include_agent_token=True, path=DECIDE)
        assert "X-Agent-Token" not in headers
    assert _decide(engine).verdict.value == "ALLOW"
    assert engine.agent_token() is None

    # Built once; the failure is remembered, not repeated per call.
    assert len(builds) == 1
    assert caplog.messages == [f"{RUNS_ON_ITS_KEY}."]
    assert ENROLL not in gateway.paths()
    assert gateway.tokens_sent() == []

    engine._optional_credential_failed_at -= engine_module._OPTIONAL_CREDENTIAL_RETRY_SECONDS + 1
    assert engine.agent_token(fail_closed=True) is None
    assert len(builds) == 2
    assert len(caplog.messages) == 1


def test_transparent_traffic_goes_on_without_a_token_for_an_optional_credential(shield_factory, monkeypatch, caplog):
    _without_runtime_extra(monkeypatch)
    gateway = _declared_and_approved(RouteGateway())
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    assert shield.agentic.engine.agent_token() is None
    assert shield._agent_token() is None
    assert "X-Agent-Token" not in shield.create_headers(identity=True)
    assert caplog.messages == [f"{RUNS_ON_ITS_KEY}."]


@needs_runtime_extra
def test_transparent_traffic_goes_on_while_the_optional_key_waits_for_authorization(shield_factory, caplog):
    gateway = _declared_and_approved(RouteGateway())
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    assert shield._agent_token() is None
    assert shield._agent_token() is None

    (jkt,) = gateway.declared_keys
    assert caplog.messages == [f"{RUNS_ON_ITS_KEY} (runtime key SHA256:{jkt})."]
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL]


@pytest.mark.parametrize("optional", [True, False])
def test_transparent_traffic_after_an_optional_answer_expires_is_best_effort(shield_factory, monkeypatch, optional):
    # The hour-long answer expired and the gateway cannot be reached to renew
    # it: an optional credential is not worth stopping transparent traffic
    # for; a mandatory broker identity still is.
    _without_runtime_extra(monkeypatch)
    gateway = _declared_and_approved(RouteGateway(), optional=optional)
    shield = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")
    engine = shield.agentic.engine
    assert engine.credential_info.credential_optional is optional
    engine._cred_info_fetched_at = 0.0

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    gateway.credential_info = lambda asked: unreachable(None)
    if optional:
        assert shield._agent_token() is None
    else:
        with pytest.raises(Exception) as caught:
            shield._agent_token()
        assert getattr(caught.value, "code", "") == "gateway_unavailable"


# ── without the flag, everything is as before ─────────────────────────────


def test_without_the_flag_a_credential_that_cannot_be_built_stops_the_call(shield_factory, monkeypatch, caplog):
    _without_runtime_extra(monkeypatch)
    gateway = _declared_and_approved(RouteGateway(), optional=False)
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    assert _pending_call(engine).code == "credential_dependency_missing"
    with pytest.raises(GovernanceConfigurationError) as caught:
        shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent")._agent_token()
    assert caught.value.code == "credential_dependency_missing"
    assert RUNS_ON_ITS_KEY not in caplog.text
    assert gateway.decisions == []


@needs_runtime_extra
def test_without_the_flag_a_key_waiting_for_authorization_stops_the_call_as_pending(shield_factory, caplog):
    # A gateway that predates the flag: the SDK reads the bootstrap block as a
    # pending registration, exactly as it did before this release.
    gateway = _declared_and_approved(RouteGateway(), optional=False)
    engine = _engine(shield_factory, gateway)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    error = _pending_call(engine)

    assert error.code == "agent_runtime_enrollment_pending"
    assert engine._registration_known_pending() is True
    assert RUNS_ON_ITS_KEY not in caplog.text
    assert gateway.decisions == []


@needs_runtime_extra
def test_an_agent_bound_to_its_provider_must_still_present_its_token(shield_factory):
    # Approved and bound to the provider (no ``credential_optional``): a key
    # that cannot get a session stops the call; nothing goes out without proof.
    gateway = _declared_and_approved(RouteGateway(), optional=False)
    gateway.key_authorized = True
    gateway.session = lambda request: httpx.Response(401, json={"error": {"code": "agent_runtime_authentication_failed"}})
    engine = _engine(shield_factory, gateway)

    assert _pending_call(engine).code == "agent_runtime_authentication_failed"
    assert gateway.decisions == []
    assert gateway.paths() == [CREDENTIAL_INFO, ENROLL, CHALLENGE, SESSION]


def test_a_supplied_credential_is_never_made_optional(shield_factory):
    gateway = _declared_and_approved(RouteGateway())

    class Supplied:
        provider_type = "static"

        def get_token(self):
            raise GovernanceConfigurationError(code="credential_exchange_failed")

    engine = _engine(shield_factory, gateway)
    assert engine.credential_info.credential_optional is True
    engine.set_agent_credential(Supplied())

    assert _pending_call(engine).code == "credential_exchange_failed"
    assert gateway.decisions == []


@pytest.mark.parametrize(
    ("fields", "optional"),
    [
        ({}, False),
        ({"credential_optional": False}, False),
        ({"credential_optional": True}, True),
        ({"credential_optional": True, "credential_bootstrap": True, "registration_state": "approved"}, True),
    ],
)
def test_only_the_server_sent_flag_makes_a_credential_optional(fields, optional):
    info = VKCredentialInfo.model_validate({"agent_configured": True, "credential_bootstrap": True, **fields})
    assert engine_module.AgenticEngine._credential_optional(info) is optional
    # An older SDK's view of the same answer: the field is simply ignored.
    assert VKCredentialInfo.model_validate({}).credential_optional is False


def test_a_pending_state_beside_the_flag_is_still_pending(shield_factory):
    gateway = RouteGateway()
    engine = _engine(shield_factory, gateway)
    engine._cred_info = VKCredentialInfo.model_validate({
        "agent_configured": True, "credential_bootstrap": True, "credential_optional": True,
        "registration_state": "pending",
    })
    assert engine._registration_known_pending() is True
    engine._cred_info = VKCredentialInfo.model_validate({
        "agent_configured": True, "credential_bootstrap": True, "credential_optional": True,
        "registration_state": "approved",
    })
    assert engine._registration_known_pending() is False
    engine._cred_info = VKCredentialInfo.model_validate({"agent_configured": True, "credential_bootstrap": True})
    assert engine._registration_known_pending() is True
