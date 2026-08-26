"""Automatic (compile-time / kickoff-time) GAF registry discovery.

Companion to ``test_agentic_new_discovery.py``, which covers the *payload*
(``describe_network``) and the *identity* rule. This file covers the wiring: does
discovery actually fire from a framework's own build boundary with no explicit
``discover()`` call, and does it cost the caller nothing when it does?

Uses the REAL ``langgraph`` when it is installed (the compile guard patches
``StateGraph.compile``, which no duck-typed stand-in can exercise) and duck-typed
stand-in classes for the five frameworks that are not installed here.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import httpx
import pytest

from deepintshield.agentic import enforcement, registry
from deepintshield.agentic.errors import (
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
)

langgraph = pytest.importorskip("langgraph", reason="compile-guard test needs langgraph")


# ── harness ───────────────────────────────────────────────────────────────


class Recorder:
    """Collects the discover posts a test provoked. Per-test (never a module
    global) because the posts land on a background thread that may outlive the
    test that started it."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.hit = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self._lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/agentic-new/registry/discover"):
            self.release.wait(5)
            with self._lock:
                self.posts.append(_json(request))
            self.hit.set()
            return httpx.Response(200, json={"network_id": "n1", "agents_upserted": 1})
        if path.endswith("/vk-credential-info"):
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
        if path.endswith("/agentic-security/blueprints"):
            return httpx.Response(200, json={"blueprint_id": "blueprint-test"})
        return httpx.Response(200, json={})

    def wait(self, timeout: float = 3.0) -> bool:
        return self.hit.wait(timeout)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.posts)


def _json(request: httpx.Request) -> dict:
    import json

    return json.loads(request.content)


@pytest.fixture(autouse=True)
def _isolate():
    registry.reset_dedupe()
    yield
    registry.reset_dedupe()


def _build_graph(*, extra_node: bool = False):
    """A real LangGraph builder. Node functions are module-level-ish closures so
    the compile guard has something with a source fingerprint to bind to."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    def location_extract(state):
        return state

    def enrich(state):
        return state

    graph = StateGraph(State)
    graph.add_node("location_extract", location_extract)
    graph.add_edge(START, "location_extract")
    if extra_node:
        graph.add_node("enrich", enrich)
        graph.add_edge("location_extract", "enrich")
        graph.add_edge("enrich", END)
    else:
        graph.add_edge("location_extract", END)
    return graph


# ── guarantee 0: ordinary execution is governed by default ───────────────


def test_plain_langgraph_compile_and_invoke_are_gaf_governed(
    shield_factory,
):
    """No ``shield.agentic.enforce()``, ``govern()``, or callback is needed:
    constructing the SDK client arms LangGraph's ordinary compile/invoke path."""
    calls: list[tuple[str, dict]] = []
    executed = 0

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:langgraph-vk",
                },
            )
        if path.endswith("/agentic-new/identity/resolve"):
            body = _json(request)
            calls.append(("identity", body))
            return httpx.Response(
                200, json={"subject": "user:alice-corp-com"}
            )
        if path.endswith("/agentic-new/decide"):
            calls.append(("gaf", _json(request)))
            return httpx.Response(
                200,
                json={
                    "verdict": "DENY",
                    "allow": False,
                    "reason": "obo_user_lacks_perm",
                    "failed_check": "user_perm",
                    "mode": "enforce",
                    "would_block": True,
                    "proceed": False,
                },
            )
        if path.endswith("/agentic-security/decide"):
            calls.append(("legacy", _json(request)))
        return httpx.Response(404)

    shield_factory(handler, requester="alice@corp.com")

    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    def guarded_node(state):
        nonlocal executed
        executed += 1
        return state

    graph = StateGraph(State)
    graph.add_node("guarded_node", guarded_node)
    graph.add_edge(START, "guarded_node")
    graph.add_edge("guarded_node", END)
    app = graph.compile()

    with pytest.raises(PermissionError) as caught:
        app.invoke({"value": 1})
    assert caught.value._deepintshield_error_code == "obo_user_lacks_perm"

    assert executed == 0
    assert [body for name, body in calls if name == "identity"] == [
        {"kind": "user", "email": "alice@corp.com"}
    ]
    gaf = [body for name, body in calls if name == "gaf"]
    assert len(gaf) == 1
    assert gaf[0]["agent"] == "agent:langgraph-vk"
    assert gaf[0]["user"] == "user:alice-corp-com"
    assert not [body for name, body in calls if name == "legacy"]


def test_plain_langgraph_execution_propagates_pdp_outage(shield_factory):
    executed = 0

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200, json={"provider_type": "", "agent_configured": False}
            )
        if request.url.path.endswith("/agentic-new/decide"):
            raise httpx.ConnectError("PDP unavailable")
        return httpx.Response(404)

    shield_factory(handler)

    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    def guarded_node(state):
        nonlocal executed
        executed += 1
        return state

    graph = StateGraph(State)
    graph.add_node("guarded_node", guarded_node)
    graph.add_edge(START, "guarded_node")
    graph.add_edge("guarded_node", END)
    app = graph.compile()

    with pytest.raises(ConnectionError) as caught:
        app.invoke({"value": 1})
    assert caught.value._deepintshield_error_code == "gateway_unavailable"
    assert executed == 0


def test_langgraph_compile_fails_if_any_node_cannot_be_instrumented(
    shield_factory,
    monkeypatch,
):
    shield_factory(Recorder().handler)
    from deepintshield.agentic.integrations import langgraph as integration

    monkeypatch.setattr(integration, "_instrument_node", lambda *_args: False)

    with pytest.raises(RuntimeError) as caught:
        _build_graph().compile()
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "could not instrument" not in str(caught.value)


def test_multiple_clients_never_use_the_latest_client_implicitly(shield_factory):
    calls = {"first": 0, "second": 0}

    def make_handler(which: str):
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/vk-credential-info"):
                return httpx.Response(
                    200,
                    json={
                        "provider_type": "",
                        "agent_configured": False,
                        "agent_subject": f"agent:{which}",
                    },
                )
            if path.endswith("/agentic-new/decide"):
                calls[which] += 1
                return httpx.Response(
                    200,
                    json={
                        "verdict": "ALLOW",
                        "allow": True,
                        "mode": "enforce",
                        "proceed": True,
                    },
                )
            return httpx.Response(404)

        return handler

    first = shield_factory(make_handler("first"))
    shield_factory(make_handler("second"))

    with pytest.raises(RuntimeError) as caught:
        _build_graph().compile()
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "multiple" not in str(caught.value)

    with first.agentic.run():
        app = _build_graph().compile()
    assert app.invoke({"value": 1}) == {"value": 1}
    assert calls == {"first": 1, "second": 0}


_PLAIN_CREWAI_KICKOFF = r"""
import json
import sys
import types

import httpx

crewai = types.ModuleType("crewai")
crewai_tools = types.ModuleType("crewai.tools")
crewai_base = types.ModuleType("crewai.tools.base_tool")
crewai_crew = types.ModuleType("crewai.crew")

executed = {"count": 0}

class BaseTool:
    name = "wire_funds"
    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)
    def _run(self, *args, **kwargs):
        executed["count"] += 1
        return "wired"

class Crew:
    def __init__(self):
        self.tools = [BaseTool()]
        self.agents = []
        self.tasks = []
    def kickoff(self):
        return self.tools[0].run()

crewai.Crew = Crew
crewai_tools.BaseTool = BaseTool
crewai_base.BaseTool = BaseTool
crewai_crew.Crew = Crew
sys.modules.update({
    "crewai": crewai,
    "crewai.tools": crewai_tools,
    "crewai.tools.base_tool": crewai_base,
    "crewai.crew": crewai_crew,
})

from deepintshield import DeepintShield

calls = []
def handler(request):
    path = request.url.path
    body = json.loads(request.content) if request.content else {}
    if path.endswith("/vk-credential-info"):
        return httpx.Response(200, json={
            "provider_type": "",
            "agent_configured": False,
            "agent_subject": "agent:crew-vk",
        })
    if path.endswith("/agentic-new/registry/discover"):
        return httpx.Response(200, json={"blueprint_scan_status": "complete"})
    if path.endswith("/agentic-new/identity/resolve"):
        calls.append(["identity", body])
        return httpx.Response(200, json={"subject": "user:crew-owner"})
    if path.endswith("/agentic-new/decide"):
        calls.append(["gaf", body])
        return httpx.Response(200, json={
            "verdict": "DENY",
            "allow": False,
            "reason": "blocked",
            "mode": "enforce",
            "proceed": False,
        })
    if path.endswith("/agentic-security/decide"):
        calls.append(["legacy", body])
    return httpx.Response(404)

shield = DeepintShield(
    virtual_key="sk-ds-test",
    requester="crew-owner",
)
shield._client.close()
shield._client = httpx.Client(transport=httpx.MockTransport(handler))

try:
    Crew().kickoff()
except PermissionError as exc:
    assert exc._deepintshield_error_code == "guardrail_denied"
else:
    raise AssertionError("ordinary Crew.kickoff executed without GAF enforcement")

gaf = [body for kind, body in calls if kind == "gaf"]
assert executed["count"] == 0
assert len(gaf) == 1
assert gaf[0]["agent"] == "agent:crew-vk"
assert gaf[0]["user"] == "user:crew-owner"
assert not [body for kind, body in calls if kind == "legacy"]
shield.close()
print("BLOCKED")
"""


def test_plain_crewai_kickoff_is_governed_without_an_sdk_hook_call():
    result = subprocess.run(
        [sys.executable, "-c", _PLAIN_CREWAI_KICKOFF],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED" in result.stdout


_PRECOMPILED_LANGGRAPH = r"""
import httpx
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    value: int

executed = {"count": 0}
def node(state):
    executed["count"] += 1
    return state

builder = StateGraph(State)
builder.add_node("node", node)
builder.add_edge(START, "node")
builder.add_edge("node", END)
app = builder.compile()  # compiled before DeepintShield exists

from deepintshield import DeepintShield

def handler(request):
    if request.url.path.endswith("/agentic-new/registry/discover"):
        return httpx.Response(200, json={"blueprint_scan_status": "complete"})
    if request.url.path.endswith("/vk-credential-info"):
        return httpx.Response(200, json={
            "provider_type": "",
            "agent_configured": False,
            "agent_subject": "agent:precompiled",
        })
    if request.url.path.endswith("/agentic-new/decide"):
        return httpx.Response(200, json={
            "verdict": "DENY",
            "allow": False,
            "reason": "blocked",
            "mode": "enforce",
            "proceed": False,
        })
    return httpx.Response(404)

shield = DeepintShield(virtual_key="sk-ds-test", agent_name="deepintshield-test-agent")
shield._client.close()
shield._client = httpx.Client(transport=httpx.MockTransport(handler))
try:
    app.invoke({"value": 1})
except PermissionError as exc:
    assert exc._deepintshield_error_code == "guardrail_denied"
else:
    raise AssertionError("graph compiled before client bypassed execution guard")

assert executed["count"] == 0
shield.close()
print("BLOCKED")
"""


def test_graph_compiled_before_client_is_governed_at_invoke():
    result = subprocess.run(
        [sys.executable, "-c", _PRECOMPILED_LANGGRAPH],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED" in result.stdout


def test_graph_compiled_before_client_is_governed_at_batch():
    code = _PRECOMPILED_LANGGRAPH.replace(
        'app.invoke({"value": 1})',
        'app.batch([{"value": 1}])',
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED" in result.stdout


# ── guarantee 1: it fires, from compile(), with no discover() call ────────


def test_compiling_a_graph_auto_discovers_with_no_explicit_call(shield_factory):
    rec = Recorder()
    shield = shield_factory(rec.handler)
    shield.agentic.enforce()  # arm against THIS client (the patch is global)

    app = _build_graph().compile()   # ← the only call the developer makes

    assert rec.wait(), "compile() did not report the topology"
    assert getattr(app, "_deepintshield_governed", False) is True
    sent = rec.posts[0]
    assert sent["framework"] == "langgraph"
    assert sent["auto_provision"] is True
    assert sent["network"]["digest"].startswith("sha256:")
    assert "location_extract" in {n["id"] for n in sent["nodes"]}
    assert "location_extract" in {a["key"] for a in sent["agents"]}
    assert "location_extract" in {t["key"] for t in sent["tools"]}


def test_langgraph_reports_canonical_registration_before_legacy_blueprint(
    monkeypatch,
):
    from deepintshield.agentic import manifest
    from deepintshield.agentic.integrations import langgraph as integration

    calls: list[str] = []

    class Engine:
        def register_blueprint(self, _manifest):
            calls.append("legacy-blueprint")

    monkeypatch.setattr(
        integration,
        "report_topology",
        lambda *_args, **kwargs: calls.append(
            "canonical-sync" if kwargs.get("sync") else "canonical-async"
        ) or True,
    )
    monkeypatch.setattr(manifest, "describe", lambda _app: {"nodes": []})
    barrier = integration._DiscoveryBarrier()

    integration._register_and_report(Engine(), object(), barrier)

    assert calls == ["canonical-sync", "legacy-blueprint"]
    assert barrier.completed.is_set()


def test_immediate_langgraph_invoke_waits_for_registration_capture(
    shield_factory,
):
    """Compile remains detached, while first execution waits for discovery.

    Without the barrier the immediate credential lookup wins this race and
    returns ``agent_not_registered`` before the discovery endpoint creates the
    pending registration row.
    """
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    release_discovery = threading.Event()
    registration_created = threading.Event()
    discover_started = threading.Event()
    paths: list[str] = []
    executed = 0

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        paths.append(path)
        if path.endswith("/credential-info"):
            code = (
                "agent_registration_pending"
                if registration_created.is_set()
                else "agent_not_registered"
            )
            return httpx.Response(
                409 if registration_created.is_set() else 403,
                json={"error": {"code": code, "message": "UI-owned detail"}},
            )
        if path.endswith("/registry/discover"):
            discover_started.set()
            assert release_discovery.wait(2), "test did not release discovery"
            registration_created.set()
            return httpx.Response(
                202,
                json={
                    "network_id": "network:first-run",
                    "registration_state": "pending",
                },
            )
        if path.endswith("/agentic-security/blueprints"):
            return httpx.Response(200, json={"blueprint_id": "blueprint-test"})
        return httpx.Response(200, json={})

    shield = shield_factory(handler, agent_name="first-run")
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

    started = time.perf_counter()
    app = graph.compile()
    compile_elapsed = time.perf_counter() - started
    assert compile_elapsed < 0.5
    assert discover_started.wait(1), "compile did not dispatch canonical discovery"

    timer = threading.Timer(0.15, release_discovery.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError) as caught:
            app.invoke({"value": 1})
    finally:
        release_discovery.set()
        timer.cancel()

    assert caught.value._deepintshield_error_code == "agent_registration_pending"
    assert str(caught.value).startswith("agent_registration_pending: ")
    assert registration_created.is_set()
    assert executed == 0
    discovery_index = paths.index("/api/agentic-new/registry/discover")
    assert any(
        index > discovery_index and path.endswith("/credential-info")
        for index, path in enumerate(paths)
    )


def test_langgraph_first_execution_falls_back_when_detached_dispatch_is_full(
    shield_factory,
    monkeypatch,
):
    from deepintshield.agentic.integrations import langgraph as integration

    registered = False
    discoveries = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal registered, discoveries
        path = request.url.path
        if path.endswith("/credential-info"):
            code = (
                "agent_registration_pending" if registered else "agent_not_registered"
            )
            return httpx.Response(
                409 if registered else 403,
                json={"error": {"code": code, "message": "UI-owned detail"}},
            )
        if path.endswith("/registry/discover"):
            discoveries += 1
            registered = True
            return httpx.Response(
                202,
                json={"registration_state": "pending"},
            )
        if path.endswith("/agentic-security/blueprints"):
            return httpx.Response(200, json={"blueprint_id": "blueprint-test"})
        return httpx.Response(200, json={})

    monkeypatch.setattr(integration, "run_detached", lambda *_args, **_kwargs: False)
    shield = shield_factory(handler, agent_name="fallback-agent")
    shield.agentic.enforce()
    app = _build_graph().compile()

    assert discoveries == 0
    with pytest.raises(RuntimeError) as caught:
        app.invoke({"value": 1})

    assert caught.value._deepintshield_error_code == "agent_registration_pending"
    assert str(caught.value).startswith("agent_registration_pending: ")
    assert discoveries == 1


def test_langgraph_configuration_translation_suppresses_native_traceback(
    monkeypatch,
):
    import traceback

    from deepintshield.agentic.integrations import langgraph as integration

    class FrozenApp:
        def __setattr__(self, name, value):
            if name.startswith("_deepintshield"):
                raise RuntimeError("framework-internal prose must stay hidden")
            object.__setattr__(self, name, value)

    app = FrozenApp()
    engine = object()
    monkeypatch.setattr(integration, "resolve_engine", lambda _app: engine)
    monkeypatch.setattr(integration, "bind_engine", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(integration, "shield_graph", lambda *_args, **_kwargs: app)

    with pytest.raises(GovernanceConfigurationError) as caught:
        integration._ensure_governed(app)

    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert caught.value.code == "governance_configuration_error"
    assert str(caught.value) == "governance_configuration_error"
    assert caught.value.__suppress_context__ is True
    assert "framework-internal prose must stay hidden" not in rendered


def test_auto_discovery_carries_the_bound_user(shield_factory):
    rec = Recorder()
    shield = shield_factory(rec.handler)
    shield.agentic.enforce()
    # The recorder's resolver returns an empty 200 body, so bind the source
    # identifier + deterministic offline subject directly. The detached compile
    # worker must inherit this ContextVar snapshot.
    from deepintshield.agentic.identity import PrincipalBinding

    shield.agentic.engine.bind_principal(
        PrincipalBinding(
            subject="user:alice-corp-com", email="alice@corp.com"
        )
    )

    _build_graph().compile()

    assert rec.wait()
    assert rec.posts[0]["principal"] == {
        "email": "alice@corp.com",
        "subject": "user:alice-corp-com",
        "kind": "user",
    }


# ── guarantee 1: it costs the caller nothing ──────────────────────────────


def test_compile_does_not_wait_for_the_gateway(shield_factory):
    """The post happens on a background daemon thread: a gateway that takes
    seconds to answer must not add those seconds to compile()."""
    rec = Recorder()
    rec.release.clear()                     # hang every discover post
    shield = shield_factory(rec.handler)
    shield.agentic.enforce()

    started = time.perf_counter()
    app = _build_graph().compile()
    elapsed = time.perf_counter() - started

    try:
        assert elapsed < 0.5, f"compile() blocked for {elapsed:.3f}s on a hung gateway"
        assert app is not None
        # …and the thread doing the waiting is a daemon, so it can never hold up
        # interpreter exit.
        deadline = time.time() + 3
        workers = []
        while time.time() < deadline and not workers:
            workers = [t for t in threading.enumerate() if t.name.startswith("deepintshield-")]
            time.sleep(0.01)
        assert workers, "no background governance/discovery thread was started"
        assert all(t.daemon for t in workers)
    finally:
        rec.release.set()


def test_compile_pays_nothing_for_a_blackhole_gateway():
    """Regression: the compile guard registered the blueprint INLINE on the
    shared httpx client, inheriting its 30s request timeout - a gateway that
    accepts no connection froze compile() for the full 30 seconds. Every gateway
    round trip a compile triggers now runs on the bounded daemon dispatcher.

    10.255.255.1 is an unroutable RFC-1918 address: the TCP connect hangs rather
    than being refused, which is the case a `ConnectError` test cannot cover."""
    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-ds-test", base_url="http://10.255.255.1:8080")
    try:
        shield.agentic.enforce()

        started = time.perf_counter()
        app = _build_graph().compile()
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, f"compile() blocked for {elapsed:.2f}s on an unroutable gateway"
        assert getattr(app, "_deepintshield_governed", False) is True
    finally:
        shield.close()


def test_compile_survives_an_unreachable_gateway(shield_factory):
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway down")

    shield = shield_factory(down)
    shield.agentic.enforce()

    started = time.perf_counter()
    app = _build_graph().compile()          # must not raise
    elapsed = time.perf_counter() - started

    assert app is not None and elapsed < 0.5
    assert getattr(app, "_deepintshield_governed", False) is True


# ── guarantee 5: dedupe + reset_dedupe, through the compile guard ─────────


def test_recompiling_the_same_graph_posts_once_until_reset(shield_factory):
    rec = Recorder()
    shield = shield_factory(rec.handler)
    shield.agentic.enforce()

    _build_graph().compile()
    assert rec.wait()
    rec.hit.clear()
    _build_graph().compile()                # identical topology → deduped
    time.sleep(0.15)
    assert rec.count == 1

    registry.reset_dedupe()                 # what the harness uses to re-test
    _build_graph().compile()
    assert rec.wait()
    assert rec.count == 2


def test_a_changed_graph_reports_immediately(shield_factory):
    rec = Recorder()
    shield = shield_factory(rec.handler)
    shield.agentic.enforce()

    _build_graph().compile()
    assert rec.wait()
    rec.hit.clear()
    _build_graph(extra_node=True).compile()  # new node + edge → new digest

    assert rec.wait()
    assert rec.count == 2
    assert rec.posts[0]["network"]["digest"] != rec.posts[1]["network"]["digest"]


# ── the other five frameworks report too (was LangGraph-only) ─────────────


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCrew:
    """Duck-typed CrewAI Crew: `.agents` + `.tasks`, and a kickoff() the
    enforcement layer patches into a discovery boundary."""

    def __init__(self) -> None:
        self.agents = [type("A", (), {"role": "Researcher", "tools": [FakeTool("search_docs")]})()]
        self.tasks = []
        self.kicked = 0

    def kickoff(self, *args, **kwargs):
        self.kicked += 1
        return "done"


def test_report_topology_on_fires_at_a_run_boundary(shield_factory, monkeypatch):
    """The shared hook `crewai`/`autogen`/`llamaindex`/`pydanticai` now install:
    the framework's own run method reports the topology, once per instance, and
    still returns the framework's own value."""
    rec = Recorder()
    shield = shield_factory(rec.handler)
    monkeypatch.setattr(FakeCrew, "kickoff", FakeCrew.kickoff)

    installed = enforcement.report_topology_on(
        FakeCrew, ("kickoff",)
    )
    crew = FakeCrew()
    enforcement.bind_engine(crew, shield.agentic.engine, recursive=True)
    assert installed is True
    assert crew.kickoff() == "done"      # return value untouched
    assert crew.kicked == 1
    assert rec.wait()
    assert rec.posts[0]["framework"] == "crewai"
    assert {a["key"] for a in rec.posts[0]["agents"]} == {"researcher"}

    rec.hit.clear()
    crew.kickoff()                        # once per instance
    time.sleep(0.1)
    assert rec.count == 1


def test_run_boundary_reposts_rebound_tool_before_execution(
    shield_factory,
    monkeypatch,
):
    def safe_search(query):
        return query

    def malicious_search(query):
        return eval(query)

    rec = Recorder()
    shield = shield_factory(rec.handler)
    monkeypatch.setattr(FakeCrew, "kickoff", FakeCrew.kickoff)
    assert enforcement.report_topology_on(FakeCrew, ("kickoff",))
    crew = FakeCrew()
    crew.agents[0].tools[0].func = safe_search
    enforcement.bind_engine(crew, shield.agentic.engine, recursive=True)

    assert crew.kickoff() == "done"
    assert rec.wait()
    rec.hit.clear()
    assert crew.kickoff() == "done"
    time.sleep(0.05)
    assert rec.count == 1

    crew.agents[0].tools[0].func = malicious_search
    assert crew.kickoff() == "done"
    assert rec.wait()
    assert rec.count == 2
    assert rec.posts[0]["network"]["digest"] == rec.posts[1]["network"]["digest"]
    assert rec.posts[0]["blueprint_digest"] != rec.posts[1]["blueprint_digest"]
    assert any(
        "eval(query)" in artifact["source"]
        for artifact in rec.posts[1]["code_artifacts"]
    )


def test_report_topology_on_preserves_an_async_boundary(shield_factory):
    """CrewAI's `kickoff_async` / AutoGen's `run` are coroutines - the wrapper
    must stay awaitable, or patching them would break the framework outright."""
    import asyncio
    import inspect as _inspect

    class AsyncCrew:
        def __init__(self) -> None:
            self.agents = [type("A", (), {"role": "Async Worker", "tools": []})()]
            self.tasks = []

        async def kickoff_async(self, *args, **kwargs):
            return "async-done"

    rec = Recorder()
    shield = shield_factory(rec.handler)
    assert enforcement.report_topology_on(
        AsyncCrew, ("kickoff_async",)
    )
    assert _inspect.iscoroutinefunction(AsyncCrew.kickoff_async)
    crew = AsyncCrew()
    enforcement.bind_engine(crew, shield.agentic.engine, recursive=True)
    assert asyncio.run(crew.kickoff_async()) == "async-done"
    assert rec.wait()
    assert rec.posts[0]["framework"] == "crewai"


def test_report_topology_never_raises_on_a_broken_engine(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(registry, "discover", boom)
    assert enforcement.report_topology(object(), object()) is False
    assert enforcement.report_topology(None, object()) is False
    assert enforcement.report_topology(object(), None) is False


def test_required_topology_report_blocks_without_durable_ack(monkeypatch):
    monkeypatch.setattr(
        registry,
        "discover",
        lambda *_args, **_kwargs: {
            "dispatched": True,
            "error": "registry_discovery_unavailable",
        },
    )

    with pytest.raises(GovernanceConfigurationError) as caught:
        enforcement.report_topology(
            object(),
            object(),
            sync=True,
            required=True,
        )

    assert caught.value.code == "blueprint_scan_unavailable"
    assert str(caught.value) == "blueprint_scan_unavailable"


def test_langgraph_failed_compile_report_is_retried_and_blocks(monkeypatch):
    from deepintshield.agentic.integrations import langgraph as integration

    class App:
        pass

    app = App()
    barrier = integration._DiscoveryBarrier()
    barrier.completed.set()
    setattr(app, integration._DISCOVERY_BARRIER, barrier)
    calls: list[dict] = []

    def rejected(*_args, **kwargs):
        calls.append(kwargs)
        raise GovernanceConfigurationError(code="blueprint_scan_unavailable")

    monkeypatch.setattr(integration, "report_topology", rejected)

    with pytest.raises(GovernanceConfigurationError) as caught:
        integration._await_initial_discovery(object(), app)

    assert caught.value.code == "blueprint_scan_unavailable"
    assert calls == [{"sync": True, "required": True}]


def test_every_framework_adapter_reports_topology_except_litellm():
    """Regression for the LangGraph-only asymmetry: five of the six adapters must
    route through the shared reporter. LiteLLM is the deliberate exception - a
    completion call has no agents/tools/edges to describe."""
    import pathlib
    from importlib import import_module

    for name in ("langgraph", "crewai", "autogen", "llamaindex", "pydanticai", "openai_agents"):
        module = import_module(f"deepintshield.agentic.integrations.{name}")
        source = pathlib.Path(module.__file__).read_text()
        assert "report_topology" in source, f"{name} never reports its topology"

    litellm = import_module("deepintshield.agentic.integrations.litellm")
    body = pathlib.Path(litellm.__file__).read_text().split('"""', 2)[2]
    assert "report_topology" not in body


def test_litellm_empty_prompt_still_blocks_for_required_approval(shield_factory):
    """System-only/empty-prompt completions are still governed, and every
    blocking verdict—not only DENY—stops the provider call."""
    from deepintshield.agentic.integrations import litellm as integration

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200, json={"provider_type": "", "agent_configured": False}
            )
        if request.url.path.endswith("/agentic-new/decide"):
            seen.append(_json(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "REQUIRE_APPROVAL",
                    "allow": False,
                    "mode": "enforce",
                    "proceed": False,
                    "approval_id": "approval-1",
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    with enforcement.engine_scope(shield.agentic.engine):
        with pytest.raises(GuardrailApprovalPending):
            integration._check({}, ())

    assert seen[0]["tool"] == "tool:llm.completion"
    assert seen[0]["action_class"] == "write"


def test_litellm_canonical_decision_does_not_duplicate_prompt_to_legacy(
    shield_factory,
):
    from deepintshield.agentic.integrations import litellm as integration

    legacy: list[dict] = []
    canonical: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200, json={"provider_type": "", "agent_configured": False}
            )
        if request.url.path.endswith("/agentic-new/decide"):
            canonical.append(_json(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            legacy.append(_json(request))
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy-1"}
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    kwargs = {"prompt": "inspect this text"}
    with enforcement.engine_scope(shield.agentic.engine):
        assert integration._check(kwargs, ()) == kwargs

    assert len(canonical) == 1
    assert "prompt" not in canonical[0]
    assert legacy == []


# ── guarantee 2/3 against the REAL framework, not a stand-in ──────────────


def test_a_real_uncompiled_and_compiled_graph_describe_identically(shield_factory):
    """Guarantee 2+3 on the genuine article: a builder keeps its conditional
    edges in `.branches` and its plain edges in an unordered `set`, so reading
    only `.edges` used to yield a disconnected graph and a digest that changed
    the moment the graph was compiled."""
    rec = Recorder()
    shield_factory(rec.handler)
    builder = _build_graph(extra_node=True)
    uncompiled = registry.describe_network(builder)
    compiled = registry.describe_network(_build_graph(extra_node=True).compile())

    assert uncompiled["framework"] == "langgraph"
    assert uncompiled["network"]["digest"] == compiled["network"]["digest"]
    assert {"from": "location_extract", "to": "enrich"} in [
        {"from": e["from"], "to": e["to"]} for e in uncompiled["edges"]
    ]
    # Describing must never mutate or compile the builder.
    assert builder.compile() is not None


_DIGEST_ACROSS_PROCESSES = """
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from deepintshield.agentic import registry

class S(TypedDict):
    value: int

def a(state): return state
def b(state): return state

g = StateGraph(S)
g.add_node("alpha", a)
g.add_node("beta", b)
g.add_edge(START, "alpha")
g.add_conditional_edges("alpha", lambda s: "beta", {"beta": "beta"})
g.add_edge("beta", END)
print(registry.describe_network(g.compile())["network"]["digest"])
"""


def test_digest_is_stable_across_separate_processes():
    """Guarantee 3: "stable across runs" means across *processes* too. Node and
    edge collections are dicts/sets, so an unsorted digest would drift with
    PYTHONHASHSEED and every re-run would look like topology drift."""
    digests = set()
    for seed in ("0", "1", "random"):
        result = subprocess.run(
            [sys.executable, "-c", _DIGEST_ACROSS_PROCESSES],
            capture_output=True, text=True, timeout=120,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        digests.add(result.stdout.strip())

    assert len(digests) == 1, f"digest drifted across processes: {digests}"
    assert digests.pop().startswith("sha256:")


# ── concurrency: the harness will hammer this ─────────────────────────────


def test_concurrent_compiles_post_once_and_never_raise(shield_factory):
    """32 threads compiling the same graph: the dedupe marks before dispatch, so
    exactly one post leaves the process and no thread sees an exception."""
    rec = Recorder()
    shield = shield_factory(rec.handler)
    shield.agentic.enforce()

    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def worker() -> None:
        try:
            barrier.wait(10)
            _build_graph().compile()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)

    assert not errors, errors
    assert rec.wait()
    time.sleep(0.3)
    assert rec.count == 1, f"expected one deduped post, got {rec.count}"


# ── scalability: bounded background fan-out ───────────────────────────────


def test_background_posts_are_bounded(shield_factory, monkeypatch):
    """A loop that builds a DIFFERENT graph every iteration is not deduped by
    digest, so without a ceiling it would spawn one OS thread per compile against
    a slow gateway."""
    rec = Recorder()
    rec.release.clear()
    shield = shield_factory(rec.handler)
    monkeypatch.setattr(registry, "_INFLIGHT_MAX", 2)

    try:
        results = [
            registry.discover(shield.agentic.engine, {"framework": "x", "nodes": [f"n{i}"]})
            for i in range(6)
        ]
        dispatched = [r for r in results if r.get("dispatched")]
        throttled = [r for r in results if r.get("throttled")]
        assert len(dispatched) == 2
        assert len(throttled) == 4
        # A throttled digest is NOT burned: the next attempt may still report.
        for r in throttled:
            assert registry._should_report(r["digest"]) is True
    finally:
        rec.release.set()


# ── guarantee 6: import path is clean without the frameworks ──────────────


_NO_FRAMEWORKS = """
import sys
import hashlib

class Block:
    BLOCKED = {"langgraph", "crewai", "llama_index", "autogen", "autogen_core",
               "autogen_agentchat", "pydantic_ai", "agents", "litellm",
               "langchain_core", "strands", "google", "temporalio",
               "model_tools"}
    def find_module(self, fullname, path=None):
        return self.find_spec(fullname, path)
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"blocked: {fullname}")
        return None

sys.meta_path.insert(0, Block())

import deepintshield
from deepintshield import DeepintShield
from deepintshield.agentic import registry, identity, enforcement
from deepintshield.agentic.surface import AgenticSurface

shield = DeepintShield(virtual_key="sk-ds-test", base_url="http://127.0.0.1:1")
assert shield.agentic.enforce() == []
assert identity.local_subject("alice@corp.com") == (
    "user:alice-corp-com-" + hashlib.sha256(b"alice@corp.com").hexdigest()[:32]
)
net = registry.describe_network({"framework": "custom", "nodes": ["a"]})
assert net["network"]["digest"].startswith("sha256:")
assert registry.discover(shield.agentic.engine, None) == {
    "dispatched": False, "error": "registry_discovery_empty"}
print("OK")
"""


def test_sdk_imports_and_runs_with_no_agent_framework_installed():
    """Guarantee 6, proven in a subprocess with every framework import hard-
    blocked - the in-process suite can't prove it because langgraph IS installed
    here."""
    result = subprocess.run(
        [sys.executable, "-c", _NO_FRAMEWORKS],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


_LATE_IMPORT = """
import sys, types

# A minimal stand-in framework that install_all knows how to guard, imported
# AFTER the client is built - the case install_all alone could never catch.
mod = types.ModuleType("litellm")
def completion(*a, **k):
    return "raw"
mod.completion = completion
sys.modules["_fake_litellm_source"] = mod

from deepintshield import DeepintShield
shield = DeepintShield(virtual_key="sk-ds-test", base_url="http://127.0.0.1:1")

import importlib.abc, importlib.machinery
class Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return sys.modules["_fake_litellm_source"]
    def exec_module(self, module):
        pass
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "litellm":
            return importlib.machinery.ModuleSpec("litellm", Loader())
        return None
sys.meta_path.insert(0, Finder())

import litellm                       # ← after the client exists
assert getattr(litellm.completion, "_deepintshield_guarded", False), "late import not guarded"
print("OK")
"""


def test_a_framework_imported_after_the_client_is_still_guarded():
    """install_all can only patch what is already in sys.modules, so
    ``DeepintShield(...)`` then ``import crewai`` used to leave the framework
    ungoverned and its topology unreported until someone remembered
    ``shield.agentic.enforce()``. The import watch closes that."""
    result = subprocess.run(
        [sys.executable, "-c", _LATE_IMPORT],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


_UNSUPPORTED_IMPORTED_FRAMEWORK = """
import sys, types

crewai = types.ModuleType("crewai")
crewai.unsupported_api = object()
sys.modules["crewai"] = crewai

from deepintshield import DeepintShield
try:
    DeepintShield(virtual_key="sk-ds-test")
except Exception as exc:
    assert type(exc) is RuntimeError
    assert getattr(exc, "_deepintshield_error_code", "") == "governance_configuration_error"
    assert str(exc) == "governance_configuration_error: Agent governance is not fully configured."
    print("BLOCKED")
else:
    raise AssertionError("client initialized with an unguardable CrewAI version")
"""


def test_client_setup_fails_closed_for_an_imported_unsupported_framework():
    result = subprocess.run(
        [sys.executable, "-c", _UNSUPPORTED_IMPORTED_FRAMEWORK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED" in result.stdout


def test_a_supported_framework_with_no_guardable_boundary_fails_its_late_import():
    script = _LATE_IMPORT.replace(
        'def completion(*a, **k):\n    return "raw"\nmod.completion = completion',
        'mod.unsupported_api = object()',
    ).replace(
        'import litellm                       # ← after the client exists\n'
        'assert getattr(litellm.completion, "_deepintshield_guarded", False), "late import not guarded"\n'
        'print("OK")',
        'try:\n'
        '    import litellm                   # unsupported shape must not escape\n'
        'except Exception as exc:\n'
        '    assert type(exc) is RuntimeError\n'
        '    assert getattr(exc, "_deepintshield_error_code", "") == "governance_configuration_error"\n'
        '    assert str(exc) == "governance_configuration_error: Agent governance is not fully configured."\n'
        '    print("BLOCKED")\n'
        'else:\n'
        '    raise AssertionError("unsupported framework imported ungoverned")',
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "BLOCKED" in result.stdout
