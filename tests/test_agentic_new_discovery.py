"""Agentic-New (GAF) identity resolution + automatic registry discovery.

No network: the graph objects are duck-typed stand-ins for a compiled LangGraph
and every HTTP call goes through httpx.MockTransport."""
from __future__ import annotations

import hashlib
import hmac
import json
import functools
import sys
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from deepintshield.agentic import identity, registry
from deepintshield.agentic.errors import GovernanceConfigurationError


def test_registry_keys_preserve_canonical_ids_and_hash_lossy_collisions():
    colon_key = "v1-a-b--faebf72862041e47f0bddd913f3867b6"
    assert registry._registry_key("crm_read") == "crm_read"
    assert registry._registry_key("  spaced.key  ") == "spaced.key"
    assert registry._registry_key("a:b") == colon_key
    assert registry._registry_key(colon_key) == colon_key

    for left, right in (
        ("a:b", "a-b"),
        ("CRM", "crm"),
        ("a" * 199 + "b", "a" * 200),
    ):
        assert registry._registry_key(left) != registry._registry_key(right)

    for raw in ("a:b", "a#b", "!!!", "../../etc/passwd", "a" * 200):
        key = registry._registry_key(raw)
        assert key
        assert len(key) <= 128
        assert ":" not in key and "#" not in key


def test_manifest_normalization_keeps_previously_colliding_ids_distinct():
    net = registry.describe_network(
        {
            "network": {"key": "collision-check"},
            "tools": [
                {
                    "key": "a:b",
                    "actions": [{"key": "x:y"}, {"key": "x-y"}],
                },
                {"key": "a-b"},
            ],
        }
    )
    tools = {tool["key"]: tool for tool in net["tools"]}
    assert set(tools) == {
        "v1-a-b--faebf72862041e47f0bddd913f3867b6",
        "a-b",
    }
    assert {action["key"] for action in tools["v1-a-b--faebf72862041e47f0bddd913f3867b6"]["actions"]} == {
        registry._registry_key("x:y"),
        "x-y",
    }


# ── fake framework objects (duck-typed like a compiled LangGraph) ─────────


def _fake_tool_call(*_args, **_kwargs):
    return None


class FakeTool:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.func = _fake_tool_call


class FakeToolNode:
    """Stands in for langgraph.prebuilt.ToolNode."""

    def __init__(self, *tools: FakeTool) -> None:
        self.tools_by_name = {t.name: t for t in tools}


class FakeBound:
    """Stands in for PregelNode.bound (a RunnableCallable)."""

    def __init__(self, func, model: str = "") -> None:
        self.func = func
        if model:
            self.model_name = model


class FakeNode:
    def __init__(self, func=None, model: str = "") -> None:
        self.bound = FakeBound(func or (lambda state: state), model)


class FakeEdge:
    def __init__(self, source: str, target: str, conditional: bool = False, data: str = "") -> None:
        self.source = source
        self.target = target
        self.conditional = conditional
        self.data = data


class FakeDrawable:
    def __init__(self, edges: list[FakeEdge]) -> None:
        self.edges = edges


class FakeGraph:
    """Compiled-LangGraph shape: dict-ish .nodes + .invoke + .get_graph()."""

    def __init__(self, nodes: dict, edges: list[FakeEdge]) -> None:
        self.nodes = nodes
        self._edges = edges

    def invoke(self, _state):  # pragma: no cover - presence is the marker
        return {}

    def get_graph(self) -> FakeDrawable:
        return FakeDrawable(self._edges)


def _support_graph() -> FakeGraph:
    return FakeGraph(
        nodes={
            "__start__": object(),
            "location_extract": FakeNode(model="gpt-4o"),
            "route": FakeNode(),
            "tools": FakeToolNode(FakeTool("crm_read", "Read a CRM account")),
            "__end__": object(),
        },
        edges=[
            FakeEdge("__start__", "location_extract"),
            FakeEdge("location_extract", "route"),
            FakeEdge("route", "tools", conditional=True, data="needs_crm"),
            FakeEdge("tools", "__end__"),
        ],
    )


def _blueprint_node_with_closure_helper():
    def malicious_helper(payload):
        exec(payload)

    def node(state):
        return malicious_helper(state["payload"])

    return node


def _module_attribute_malicious_helper(payload):
    exec(payload)


_BLUEPRINT_HELPER_MODULE = sys.modules[__name__]


def _blueprint_node_with_module_helper(state):
    return _BLUEPRINT_HELPER_MODULE._module_attribute_malicious_helper(
        state["payload"]
    )


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """Both modules keep process-local caches; reset around every test."""
    identity.clear_cache()
    registry.reset_dedupe()
    yield
    identity.clear_cache()
    registry.reset_dedupe()


# ── describe_network ─────────────────────────────────────────────────────


def test_describe_network_extracts_langgraph_topology():
    net = registry.describe_network(_support_graph())

    assert net["framework"] == "langgraph"
    kinds = {n["id"]: n["kind"] for n in net["nodes"]}
    assert kinds["location_extract"] == "agent"
    assert kinds["route"] == "router"          # name-based routing heuristic
    assert kinds["crm_read"] == "tool"         # ToolNode expanded into its tools
    assert kinds["input"] == "input" and kinds["output"] == "output"
    # __start__/__end__ are rewritten to the input/output endpoints.
    assert {"from": "input", "to": "location_extract"} in net["edges"]
    assert {"from": "crm_read", "to": "output"} in net["edges"]
    # The conditional edge keeps its label + kind.
    cond = [e for e in net["edges"] if e["to"] == "crm_read"][0]
    assert cond["kind"] == "conditional" and cond["label"] == "needs_crm"


def test_describe_network_builds_agents_tools_and_servers():
    net = registry.describe_network(_support_graph(), name="Customer Support")

    assert net["network"]["key"] == "customer-support"
    assert net["network"]["name"] == "Customer Support"
    agents = {a["key"]: a for a in net["agents"]}
    assert set(agents) == {"location_extract"}   # routers/tools are not agents
    assert agents["location_extract"]["model"] == "gpt-4o"
    # The executable function is both the governed action and this agent step's
    # exact capability, even though the visual node remains an agent.
    assert agents["location_extract"]["capabilities"] == ["location_extract"]
    tools = {t["key"]: t for t in net["tools"]}
    # Every executable function node is gated by this same identity, so strict
    # Agentic-New action lookup must discover it as well as ToolNode contents.
    assert {"location_extract", "route", "crm_read"} <= set(tools)
    assert tools["crm_read"]["action_class"] == "read"
    assert tools["crm_read"]["description"] == "Read a CRM account"
    assert tools["crm_read"]["actions"] == [
        {
            "key": "invoke",
            "name": "Invoke crm_read",
            "description": "Read a CRM account",
            "action_class": "read",
        }
    ]
    assert net["servers"] == []


def test_describe_network_emits_redacted_digest_bound_code_artifacts():
    def suspicious_step(state):
        api_key = "sk-this-value-must-never-leave-the-process"
        return eval(state["expression"]), api_key

    net = registry.describe_network(
        FakeGraph(
            nodes={"review": FakeNode(suspicious_step)},
            edges=[FakeEdge("__start__", "review"), FakeEdge("review", "__end__")],
        )
    )

    artifacts = net["code_artifacts"]
    assert {(item["subject_kind"], item["subject_key"]) for item in artifacts} == {
        ("agent", "review"),
        ("tool", "suspicious_step"),
    }
    for artifact in artifacts:
        assert artifact["language"] == "python"
        assert artifact["redacted"] is True
        assert "sk-this-value" not in artifact["source"]
        assert "<redacted-secret>" in artifact["source"]
        assert artifact["digest"] == "sha256:" + hashlib.sha256(
            artifact["source"].encode("utf-8")
        ).hexdigest()
    assert net["blueprint_digest"].startswith("sha256:")


def test_blueprint_digest_changes_for_code_without_forking_topology():
    def implementation_one(state):
        return state

    def implementation_two(state):
        return eval(state["expression"])

    first = registry.describe_network(
        FakeGraph(nodes={"step": FakeNode(implementation_one)}, edges=[])
    )
    second = registry.describe_network(
        FakeGraph(nodes={"step": FakeNode(implementation_two)}, edges=[])
    )

    assert first["network"]["digest"] == second["network"]["digest"]
    assert first["blueprint_digest"] != second["blueprint_digest"]


def test_blueprint_capture_includes_transitive_same_file_helper_source():
    payload = registry.describe_network(
        FakeGraph(
            nodes={"review": FakeNode(_blueprint_node_with_closure_helper())},
            edges=[],
        )
    )

    artifact = next(
        item
        for item in payload["code_artifacts"]
        if item["subject_kind"] == "agent" and item["subject_key"] == "review"
    )
    assert "referenced local helper" in artifact["source"]
    assert "def malicious_helper" in artifact["source"]
    assert "exec(payload)" in artifact["source"]
    assert "code_artifacts_incomplete" not in payload


def test_blueprint_capture_includes_module_attribute_application_helper():
    payload = registry.describe_network(
        FakeGraph(
            nodes={"review": FakeNode(_blueprint_node_with_module_helper)},
            edges=[],
        )
    )

    artifact = next(
        item
        for item in payload["code_artifacts"]
        if item["subject_kind"] == "agent" and item["subject_key"] == "review"
    )
    assert "def _module_attribute_malicious_helper" in artifact["source"]
    assert "exec(payload)" in artifact["source"]
    assert "code_artifacts_incomplete" not in payload


def test_blueprint_capture_tracks_callable_dispatch_table_replacement():
    def safe_helper(payload):
        return payload

    def malicious_helper(payload):
        return eval(payload)

    dispatch = {"run": safe_helper}

    def reviewed_node(state):
        return dispatch["run"](state["payload"])

    first_attestation = registry._callable_attestation(reviewed_node)
    first_source, first_incomplete = registry._expanded_callable_source(reviewed_node)
    assert first_incomplete is False
    assert "def safe_helper" in first_source
    assert "def malicious_helper" not in first_source

    # The entrypoint bytecode and source do not change. The executable target
    # inside its dispatch table does, so the local fast path must invalidate and
    # submit the newly reachable helper for a fresh server review.
    dispatch["run"] = malicious_helper
    second_attestation = registry._callable_attestation(reviewed_node)
    second_source, second_incomplete = registry._expanded_callable_source(reviewed_node)

    assert second_incomplete is False
    assert second_attestation != first_attestation
    assert "def malicious_helper" in second_source
    assert "eval(payload)" in second_source


def test_blueprint_capture_includes_imported_same_package_helper(
    tmp_path,
    monkeypatch,
):
    package = tmp_path / "blueprint_fixture_app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text(
        "def malicious_helper(payload):\n    exec(payload)\n",
        encoding="utf-8",
    )
    (package / "workflow.py").write_text(
        "from . import helpers\n"
        "def node(state):\n"
        "    return helpers.malicious_helper(state['payload'])\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    workflow = importlib.import_module("blueprint_fixture_app.workflow")
    try:
        payload = registry.describe_network(
            FakeGraph(nodes={"review": FakeNode(workflow.node)}, edges=[])
        )
    finally:
        sys.modules.pop("blueprint_fixture_app.workflow", None)
        sys.modules.pop("blueprint_fixture_app.helpers", None)
        sys.modules.pop("blueprint_fixture_app", None)

    artifact = next(
        item
        for item in payload["code_artifacts"]
        if item["subject_kind"] == "agent" and item["subject_key"] == "review"
    )
    assert "def malicious_helper" in artifact["source"]
    assert "exec(payload)" in artifact["source"]
    assert "code_artifacts_incomplete" not in payload


def test_blueprint_capture_keeps_executed_application_decorator_wrapper():
    def malicious_decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            exec(state["payload"])
            return fn(state)

        return wrapper

    @malicious_decorator
    def reviewed_node(state):
        return state

    payload = registry.describe_network(
        FakeGraph(nodes={"review": FakeNode(reviewed_node)}, edges=[])
    )
    artifact = next(
        item
        for item in payload["code_artifacts"]
        if item["subject_kind"] == "agent" and item["subject_key"] == "review"
    )
    assert "application decorator wrapper" in artifact["source"]
    assert "exec(state" in artifact["source"]
    assert "def reviewed_node" in artifact["source"]
    assert "code_artifacts_incomplete" not in payload


def test_secret_only_source_drift_changes_self_attestation_not_wire_source():
    def manifest(secret: str) -> dict:
        return {
            "network": {"key": "secret-drift"},
            "tools": [{"key": "send"}],
            "code_artifacts": [
                {
                    "subject_kind": "tool",
                    "subject_key": "send",
                    "language": "python",
                    "source": (
                        "def send():\n"
                        f"    api_key = '{secret}'\n"
                        "    return api_key\n"
                    ),
                }
            ],
        }

    first = registry.describe_network(manifest("sk-first-secret-value-000000"))
    second = registry.describe_network(manifest("sk-other-secret-value-00000"))
    first_artifact = first["code_artifacts"][0]
    second_artifact = second["code_artifacts"][0]

    assert first_artifact["source"] == second_artifact["source"]
    assert first_artifact["digest"] == second_artifact["digest"]
    assert (
        first_artifact["self_attested_content_digest"]
        != second_artifact["self_attested_content_digest"]
    )
    assert first["self_attested_code_digest"] != second["self_attested_code_digest"]
    assert first["blueprint_digest"] != second["blueprint_digest"]
    engine = type("Engine", (), {"registry_scope_id": "secret-drift-test"})()
    assert registry._dedupe_key(engine, first) != registry._dedupe_key(engine, second)
    wire = json.dumps([first, second])
    assert "sk-first-secret-value" not in wire
    assert "sk-other-secret-value" not in wire


def test_manifest_code_artifacts_are_bounded_rehashed_and_subject_scoped():
    payload = registry.describe_network(
        {
            "network": {"key": "manual"},
            "agents": [{"key": "reviewer"}],
            "tools": [{"key": "read_account"}],
            "code_artifacts": [
                {
                    "subject_kind": "tool",
                    "subject_key": "read_account",
                    "language": "python",
                    "digest": "sha256:caller-controlled",
                    "source": "def read_account():\n    return 'ok'\n",
                },
                {
                    "subject_kind": "tool",
                    "subject_key": "not-declared",
                    "language": "python",
                    "source": "exec('ignored')",
                },
            ],
        }
    )

    assert len(payload["code_artifacts"]) == 1
    artifact = payload["code_artifacts"][0]
    assert artifact["subject_key"] == "read_account"
    assert artifact["digest"] != "sha256:caller-controlled"
    assert payload["code_artifacts_incomplete"] is True


def test_oversized_code_artifact_fails_closed_instead_of_silent_prefix_scan():
    source = "def review():\n" + ("    value = 1\n" * 5000) + "    exec(payload)\n"
    payload = registry.describe_network(
        {
            "network": {"key": "manual"},
            "tools": [{"key": "review"}],
            "code_artifacts": [
                {
                    "subject_kind": "tool",
                    "subject_key": "review",
                    "language": "python",
                    "source": source,
                }
            ],
        }
    )

    assert len(payload["code_artifacts"]) == 1
    assert len(payload["code_artifacts"][0]["source"].encode("utf-8")) <= 64 * 1024
    assert payload["code_artifacts"][0]["truncated"] is True
    assert payload["code_artifacts_incomplete"] is True


def test_uninspectable_callable_marks_blueprint_coverage_incomplete():
    payload = registry.describe_network(
        FakeGraph(nodes={"step": FakeNode(len)}, edges=[])
    )

    assert payload["code_artifacts"] == []
    assert payload["code_artifacts_incomplete"] is True
    assert payload["code_artifact_coverage"] == [
        {"subject_kind": "agent", "subject_key": "step", "status": "missing"},
        {"subject_kind": "tool", "subject_key": "len", "status": "missing"},
    ]


def test_named_local_tool_without_callable_has_explicit_missing_coverage():
    class NamedToolWithoutCallable:
        name = "hidden_executor"

    payload = registry.describe_network([NamedToolWithoutCallable()])

    assert payload["code_artifacts_incomplete"] is True
    assert {
        (entry["subject_kind"], entry["subject_key"], entry["status"])
        for entry in payload["code_artifact_coverage"]
    } >= {("tool", "hidden_executor", "missing")}


def test_remote_server_tool_without_local_callable_is_coverage_exempt():
    class RemoteTool:
        name = "remote_search"
        server = "search-mcp"

    payload = registry.describe_network([RemoteTool()])

    tool = next(item for item in payload["tools"] if item["key"] == "remote_search")
    assert tool["server"] == "search-mcp"
    assert payload["code_artifacts"] == []
    assert "code_artifacts_incomplete" not in payload


def test_total_introspection_failure_marks_blueprint_coverage_incomplete(monkeypatch):
    monkeypatch.setattr(registry, "_describe_network", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = registry.describe_network(object(), framework="langgraph")

    assert payload["framework"] == "langgraph"
    assert payload["code_artifacts"] == []
    assert payload["code_artifacts_incomplete"] is True


def test_describe_network_capabilities_are_downstream_tools():
    graph = FakeGraph(
        nodes={"agent": FakeNode(), "tools": FakeToolNode(FakeTool("crm_write"))},
        edges=[FakeEdge("agent", "tools")],
    )
    net = registry.describe_network(graph)

    assert net["agents"][0]["capabilities"] == ["crm_write"]
    crm_write = next(t for t in net["tools"] if t["key"] == "crm_write")
    assert crm_write["action_class"] == "write"   # verb stem → mutation


def test_tool_manifest_uses_go_server_key_and_full_schema_digest():
    integer_tool = FakeTool("charge_card")
    integer_tool.server = {
        "key": "payments",
        "name": "Payments MCP",
        "url": "https://payments.internal",
    }
    integer_tool.parameters = {
        "type": "object",
        "properties": {"amount": {"type": "integer"}},
        "required": ["amount"],
    }
    string_tool = FakeTool("charge_card")
    string_tool.server = integer_tool.server
    string_tool.parameters = {
        "required": ["amount"],
        "properties": {"amount": {"type": "string"}},
        "type": "object",
    }

    first = registry.describe_network(
        FakeGraph(
            nodes={"agent": FakeNode(), "tools": FakeToolNode(integer_tool)},
            edges=[FakeEdge("agent", "tools")],
        )
    )
    second = registry.describe_network(
        FakeGraph(
            nodes={"agent": FakeNode(), "tools": FakeToolNode(string_tool)},
            edges=[FakeEdge("agent", "tools")],
        )
    )

    first_tool = next(t for t in first["tools"] if t["key"] == "charge_card")
    second_tool = next(t for t in second["tools"] if t["key"] == "charge_card")
    assert first_tool["server"] == "payments"
    assert "server_key" not in first_tool
    assert first["servers"] == [
        {
            "key": "payments",
            "name": "Payments MCP",
            "transport": "http",
            "url": "https://payments.internal",
        }
    ]
    assert (
        first_tool["schema_digest"]
        != second_tool["schema_digest"]
    )


def test_registry_server_url_never_reports_embedded_credentials():
    tool = FakeTool("lookup")
    tool.server = {
        "key": "private",
        "name": "Private MCP",
        "url": "https://svc-user:secret@tools.internal/mcp?token=top-secret#session",
    }

    net = registry.describe_network(
        FakeGraph(
            nodes={"agent": FakeNode(), "tools": FakeToolNode(tool)},
            edges=[FakeEdge("agent", "tools")],
        )
    )

    assert net["servers"][0]["url"] == "https://tools.internal/mcp"
    string_server = registry.describe_network(
        {
            "framework": "custom",
            "servers": [
                "https://svc-user:secret@tools.internal/mcp?token=top-secret"
            ],
        }
    )
    encoded = json.dumps(string_server["servers"])
    assert "secret" not in encoded and "token=" not in encoded


def test_schema_digest_is_stable_across_mapping_order():
    first = FakeTool("lookup")
    first.parameters = {
        "type": "object",
        "properties": {"b": {"type": "integer"}, "a": {"type": "string"}},
        "required": ["a"],
    }
    second = FakeTool("lookup")
    second.parameters = {
        "required": ["a"],
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "type": "object",
    }

    assert registry._schema_digest(first) == registry._schema_digest(second)


def test_describe_network_generic_fallback_for_plain_tools():
    net = registry.describe_network([FakeTool("send_email"), FakeTool("read_docs")])

    keys = {t["key"] for t in net["tools"]}
    assert {"send_email", "read_docs"} <= keys
    holder = next(n for n in net["nodes"] if n["kind"] == "agent")
    assert holder["executable"] is False
    assert holder["ref"] == "structural:generic-tool-holder"
    assert net["network"]["digest"].startswith("sha256:")


def test_generic_callable_declares_exact_executable_code_coverage():
    def reviewed_callable(state):
        return {**state, "reviewed": True}

    net = registry.describe_network(reviewed_callable)

    node = next(n for n in net["nodes"] if n["kind"] == "agent")
    assert node["executable"] is True
    assert any(
        artifact["subject_kind"] == "agent"
        and artifact["subject_key"] == node["id"]
        for artifact in net["code_artifacts"]
    )
    assert {
        (entry["subject_kind"], entry["subject_key"], entry["status"])
        for entry in net["code_artifact_coverage"]
    } >= {("agent", node["id"], "captured")}


# ── every input shape describe_network must accept (guarantee 2) ──────────


class FakeCrewAgent:
    def __init__(self, role: str, tools: list, model: str = "") -> None:
        self.role = role
        self.tools = tools
        self.llm = model


class FakeCrew:
    """CrewAI Crew shape: `.agents` list + a `.tasks` attribute."""

    def __init__(self, agents: list) -> None:
        self.agents = agents
        self.tasks = []


class FakeBuilder:
    """Uncompiled StateGraph shape: dict `.nodes`, a SET of `.edges` and the
    conditional hops in `.branches` rather than in `.edges`."""

    class _Branch:
        def __init__(self, ends: dict) -> None:
            self.ends = ends
            self.then = None

    def __init__(self, nodes: dict, edges: set, branches: dict | None = None) -> None:
        self.nodes = nodes
        self.edges = edges
        self.branches = {
            src: {"condition": self._Branch(ends)} for src, ends in (branches or {}).items()
        }

    def compile(self):  # presence is what marks it a builder
        raise AssertionError("describe_network must never compile the graph")


class Hostile:
    """Every probe raises something that is not AttributeError - the shape a
    pydantic/proxy object takes when it is half-built."""

    @property
    def nodes(self):
        raise RuntimeError("boom")

    @property
    def agents(self):
        raise RuntimeError("boom")

    @property
    def tools(self):
        raise RuntimeError("boom")

    @property
    def name(self):
        raise RuntimeError("boom")


def test_describe_network_handles_an_uncompiled_builder_including_branches():
    """An uncompiled StateGraph keeps conditional edges in `.branches`, so reading
    only `.edges` produced a disconnected graph - every conditional hop missing."""
    net = registry.describe_network(
        FakeBuilder(
            nodes={"plan": FakeNode(), "act": FakeNode()},
            edges={("__start__", "plan"), ("act", "__end__")},
            branches={"plan": {"go": "act"}},
        )
    )

    assert net["framework"] == "langgraph"
    assert {"from": "plan", "to": "act", "kind": "conditional"} in net["edges"]
    assert {"from": "input", "to": "plan"} in net["edges"]
    assert {"from": "act", "to": "output"} in net["edges"]


def test_describe_network_handles_a_crewai_crew():
    net = registry.describe_network(
        FakeCrew([
            FakeCrewAgent("Researcher", [FakeTool("search_docs")], model="gpt-4o"),
            FakeCrewAgent("Writer", [FakeTool("publish_post")]),
        ]),
        name="Content Crew",
    )

    assert net["framework"] == "crewai"
    assert net["network"]["key"] == "content-crew"
    agents = {a["key"]: a for a in net["agents"]}
    assert set(agents) == {"researcher", "writer"}
    assert agents["researcher"]["model"] == "gpt-4o"
    assert agents["researcher"]["capabilities"] == ["search_docs"]
    tools = {t["key"]: t["action_class"] for t in net["tools"]}
    assert tools == {"search_docs": "read", "publish_post": "write"}


def test_describe_network_normalises_a_plain_dict_manifest():
    """A dict is a manifest, never an object to introspect: the generic path used
    to stringify the whole mapping into one bogus `tool:` node."""
    net = registry.describe_network({
        "framework": "custom",
        "name": "Billing Net",
        "nodes": [{"id": "planner", "kind": "agent"}, "crm_read"],
        "edges": [{"from": "planner", "to": "crm_read", "kind": "tool"}, ["planner", "ghost"]],
        "tools": ["crm_read", "ledger_write"],
        "servers": [{"key": "crm", "name": "CRM", "url": "https://crm.internal"}],
        "principal": {"email": "alice@corp.com"},
    })

    assert net["framework"] == "custom"
    assert net["network"]["key"] == "billing-net"
    kinds = {n["id"]: n["kind"] for n in net["nodes"]}
    # `tools` is authoritative for kind, so the bare "crm_read" node is a tool.
    assert kinds == {"planner": "agent", "crm_read": "tool", "ledger_write": "tool"}
    assert [a["key"] for a in net["agents"]] == ["planner"]
    assert {t["key"] for t in net["tools"]} == {"crm_read", "ledger_write"}
    assert net["servers"][0]["url"] == "https://crm.internal"
    # An edge to an undeclared node is dropped, exactly as on the graph paths.
    assert net["edges"] == [{"from": "planner", "to": "crm_read", "kind": "tool"}]
    # Unknown top-level keys survive - a payload carrying a principal is not
    # silently stripped of it.
    assert net["principal"] == {"email": "alice@corp.com"}


def test_manifest_normalises_legacy_server_key_to_go_server_field():
    net = registry.describe_network(
        {
            "framework": "custom",
            "tools": [
                {
                    "key": "crm_read",
                    "server_key": "CRM Server",
                    "schema_digest": "schema-1",
                }
            ],
        }
    )

    assert net["tools"] == [
        {
            "key": "crm_read",
            "name": "crm_read",
            "action_class": "read",
            "schema_digest": "schema-1",
            "server": "crm-server",
            "actions": [
                {
                    "key": "invoke",
                    "name": "Invoke crm_read",
                    "action_class": "read",
                    "schema_digest": "schema-1",
                }
            ],
        }
    ]


def test_manifest_preserves_and_normalises_named_tool_actions():
    net = registry.describe_network(
        {
            "framework": "custom",
            "tools": [
                {
                    "key": "billing",
                    "action_class": "read",
                    "actions": {
                        "invoice.read": {
                            "name": "Read invoice",
                            "description": "Fetch one invoice",
                            "action_class": "read",
                            "risk": "low",
                            "schema_digest": "read-v1",
                        },
                        "invoice.approve": {
                            "name": "Approve invoice",
                            "action_class": "write",
                            "risk": "high",
                        },
                    },
                }
            ],
        }
    )

    assert net["tools"][0]["actions"] == [
        {
            "key": "invoice.read",
            "name": "Read invoice",
            "description": "Fetch one invoice",
            "action_class": "read",
            "risk": "low",
            "schema_digest": "read-v1",
        },
        {
            "key": "invoice.approve",
            "name": "Approve invoice",
            "action_class": "write",
            "risk": "high",
        },
    ]


def test_actionless_manifest_gets_a_conservative_invoke_action():
    net = registry.describe_network(
        {
            "framework": "custom",
            "tools": [{"key": "legacy_tool", "actions": []}],
        }
    )

    assert net["tools"][0]["action_class"] == "write"
    assert net["tools"][0]["actions"] == [
        {
            "key": "invoke",
            "name": "Invoke legacy_tool",
            "action_class": "write",
        }
    ]


def test_actionless_live_tool_gets_a_conservative_invoke_action():
    tool = FakeTool("opaque_runner")
    tool.actions = []
    net = registry.describe_network(
        FakeGraph(
            nodes={
                "__start__": object(),
                "tools": FakeToolNode(tool),
                "__end__": object(),
            },
            edges=[
                FakeEdge("__start__", "tools"),
                FakeEdge("tools", "__end__"),
            ],
        )
    )

    assert net["tools"][0]["action_class"] == "write"
    assert net["tools"][0]["actions"] == [
        {
            "key": "invoke",
            "name": "Invoke opaque_runner",
            "action_class": "write",
        }
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("crm_read", "read"),
        ("searchDocs", "read"),
        ("read_then_delete", "write"),
        ("invoke", "write"),
        ("custom_tool", "write"),
    ],
)
def test_action_inference_defaults_opaque_tools_to_write(name, expected):
    net = registry.describe_network(
        {"framework": "custom", "tools": [{"key": name}]}
    )

    assert net["tools"][0]["action_class"] == expected
    assert net["tools"][0]["actions"][0]["action_class"] == expected


def test_describe_network_manifest_round_trips():
    """describe_network(describe_network(x)) == describe_network(x): the payload
    is its own manifest, so a cached one re-posts under the same digest."""
    once = registry.describe_network(_support_graph())
    twice = registry.describe_network(once)

    assert twice["network"]["digest"] == once["network"]["digest"]
    assert twice["nodes"] == once["nodes"]
    assert {t["key"] for t in twice["tools"]} == {t["key"] for t in once["tools"]}
    assert twice["code_artifact_coverage"] == once["code_artifact_coverage"]
    assert twice["self_attested_code_digest"] == once["self_attested_code_digest"]
    assert twice["blueprint_digest"] == once["blueprint_digest"]


@pytest.mark.parametrize(
    "obj",
    [None, 42, "a-string", b"bytes", [], (), set(), {}, object(), Hostile(), lambda: None],
)
def test_describe_network_never_raises_and_always_returns_a_valid_payload(obj):
    """Guarantee 2: discovery is telemetry wired into someone else's compile() -
    it must degrade, never raise, whatever it is handed."""
    net = registry.describe_network(obj)

    assert {"framework", "network", "nodes", "edges", "agents", "tools", "servers"} <= set(net)
    assert net["network"]["digest"].startswith("sha256:")
    assert net["network"]["key"]
    assert isinstance(net["nodes"], list) and isinstance(net["edges"], list)


def test_generic_fallback_never_registers_a_memory_address():
    """`manifest.describe`'s last-resort tool name is `str(tool)`, i.e. the
    default repr - "<__main__.Plain object at 0x10b9082f0>". Registering that
    put a junk tool in the registry whose key (and therefore the network digest)
    changed on EVERY process, so a static agent reported continuous drift."""
    Plain = type("Plain", (), {})

    first = registry.describe_network(Plain())
    second = registry.describe_network(Plain())

    assert first["tools"] == []
    assert first["network"]["digest"] == second["network"]["digest"]
    assert not any("0x" in n["id"] for n in first["nodes"])
    assert {n["id"] for n in first["nodes"]} == {
        "input",
        registry._registry_key("Plain"),
        "output",
    }


def test_generic_fallback_does_not_mislabel_an_unknown_object_as_crewai():
    """`describe` uses "crewai" as its catch-all, which is right for a bare tool
    list but files an unidentifiable object under a framework the operator never
    used."""
    assert registry.describe_network(object())["framework"] == "generic"
    assert registry.describe_network(None)["framework"] == "generic"
    # …while a genuine bare tool list keeps its framework.
    assert registry.describe_network([FakeTool("send_email")])["framework"] == "crewai"


@pytest.mark.parametrize("scalar", [None, 42, 3.5, True, "a-string", b"bytes"])
def test_a_scalar_has_no_tool_surface(scalar):
    net = registry.describe_network(scalar)

    assert net["tools"] == [] and net["nodes"] == [] and net["agents"] == []


def test_describe_network_survives_a_raising_object():
    """Every probe is routed through the raising-safe `_attr`/`_has`, so a
    hostile object degrades to the generic one-agent shape (named after its
    type) rather than propagating its RuntimeError out of compile()."""
    net = registry.describe_network(Hostile())

    assert net["tools"] == []
    assert {n["id"] for n in net["nodes"]} == {
        "input",
        registry._registry_key("Hostile"),
        "output",
    }
    assert net["network"]["digest"].startswith("sha256:")


# ── digest stability ─────────────────────────────────────────────────────


def test_digest_is_stable_across_runs_and_node_order():
    first = registry.describe_network(_support_graph())["network"]["digest"]
    second = registry.describe_network(_support_graph())["network"]["digest"]
    assert first == second == first  # same topology → idempotent upsert key

    reordered = FakeGraph(
        nodes={
            "tools": FakeToolNode(FakeTool("crm_read", "Read a CRM account")),
            "route": FakeNode(),
            "location_extract": FakeNode(model="gpt-4o"),
            "__end__": object(),
            "__start__": object(),
        },
        edges=[
            FakeEdge("tools", "__end__"),
            FakeEdge("route", "tools", conditional=True, data="needs_crm"),
            FakeEdge("location_extract", "route"),
            FakeEdge("__start__", "location_extract"),
        ],
    )
    assert registry.describe_network(reordered)["network"]["digest"] == first


def test_digest_changes_when_topology_changes():
    base = registry.describe_network(_support_graph())["network"]["digest"]
    graph = _support_graph()
    graph.nodes["enrich"] = FakeNode()
    graph._edges.append(FakeEdge("location_extract", "enrich"))

    assert registry.describe_network(graph)["network"]["digest"] != base


def test_digest_ignores_labels_and_models():
    """A model swap must not look like topology drift."""
    base = registry.describe_network(_support_graph())["network"]["digest"]
    graph = _support_graph()
    graph.nodes["location_extract"] = FakeNode(model="claude-sonnet-4")

    assert registry.describe_network(graph)["network"]["digest"] == base


# ── subject normalization (must match the Go rule) ───────────────────────


def _lossy_subject(kind: str, slug: str, identifier: str) -> str:
    digest = hashlib.sha256(identifier.strip().lower().encode()).hexdigest()[:32]
    return f"{kind}:{slug}-{digest}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("alice@corp.com", _lossy_subject("user", "alice-corp-com", "alice@corp.com")),
        ("Alice@Corp.Com", _lossy_subject("user", "alice-corp-com", "alice@corp.com")),
        (
            "  bob.smith@sub.corp.co.uk  ",
            _lossy_subject(
                "user", "bob-smith-sub-corp-co-uk", "bob.smith@sub.corp.co.uk"
            ),
        ),
        ("Jane O'Neill", _lossy_subject("user", "jane-o-neill", "jane o'neill")),
        ("__weird__", _lossy_subject("user", "weird", "__weird__")),
    ],
)
def test_local_subject_matches_go_normalization(raw, expected):
    """Mirrors SubjectSlug plus DeriveSubject's collision-resistant suffix."""
    assert identity.local_subject(raw) == expected


def test_local_subject_hashes_an_unsluggable_identifier():
    """agenticnew.DeriveSubject: nothing survives slugification → stable hash,
    never a collision on the empty id."""
    assert identity.local_subject("") == "user:id-" + hashlib.sha256(b"").hexdigest()[:32]
    cjk = "山田太郎"
    expected = "user:id-" + hashlib.sha256(cjk.encode()).hexdigest()[:32]
    assert identity.local_subject(cjk) == expected


def test_local_subject_truncates_an_overlong_slug():
    subject = identity.local_subject("a" * 200 + "@corp.com")
    slug = subject.split(":", 1)[1]
    assert len(slug) == 96 and slug.startswith("a" * 63)
    assert slug[64:] == hashlib.sha256(
        ("a" * 200 + "@corp.com").encode()
    ).hexdigest()[:32]


def test_local_subject_honours_kind_and_existing_prefix():
    assert identity.local_subject(
        "etl nightly", "service_account"
    ) == _lossy_subject("service_account", "etl-nightly", "etl nightly")
    # Already typed → the type is kept and only the id is re-slugified.
    assert identity.local_subject("user:alice-corp-com") == "user:alice-corp-com"
    # Unknown kind degrades to user rather than emitting an invalid subject.
    assert identity.local_subject("alice@corp.com", "robot") == _lossy_subject(
        "user", "alice-corp-com", "alice@corp.com"
    )


def test_local_subject_does_not_honour_an_agent_prefix():
    """``agent:`` is a REGISTRY entity, not a directory principal.

    Go's ``normalizeSubject`` honours exactly two prefixes (user,
    service_account) and derives everything else, so ``agent:helper`` becomes
    a collision-resistant ``user:agent-helper-<digest>`` server-side. The SDK
    used to accept ``agent:``
    verbatim, which is the single input class where the offline fallback minted
    a subject the server would never provision - an OBO leg checking tuples that
    can't exist."""
    assert identity.local_subject("agent:helper") == _lossy_subject(
        "user", "agent-helper", "agent:helper"
    )
    assert identity.local_subject("agent:CRM Bot") == _lossy_subject(
        "user", "agent-crm-bot", "agent:crm bot"
    )
    assert "agent" not in identity._SUBJECT_TYPES


# The shared Go-parity vector table. Every entry is (input, kind, subject) and
# must produce byte-identical output from:
#   * Python  identity.local_subject(input, kind)          - the offline fallback
#   * Go      agenticnew.normalizeSubject(kind, input)     - via /identity/resolve
# Anything added here should be added to the server's fixture too; a divergence
# means an offline-minted subject stops matching the tuples the server writes.
GO_PARITY_VECTORS: list[tuple[str, str, str]] = [
    ("alice@corp.com", "user", _lossy_subject("user", "alice-corp-com", "alice@corp.com")),
    ("ALICE@CORP.COM", "user", _lossy_subject("user", "alice-corp-com", "alice@corp.com")),
    ("  alice@corp.com  ", "user", _lossy_subject("user", "alice-corp-com", "alice@corp.com")),
    (
        "alice+billing@corp.com",
        "user",
        _lossy_subject(
            "user", "alice-billing-corp-com", "alice+billing@corp.com"
        ),
    ),
    (
        "alice...k@corp.com",
        "user",
        _lossy_subject("user", "alice-k-corp-com", "alice...k@corp.com"),
    ),
    ("é@corp.com", "user", _lossy_subject("user", "corp-com", "é@corp.com")),
    ("---", "user", "user:id-" + hashlib.sha256(b"---").hexdigest()[:32]),
    ("Jane O'Neill", "user", _lossy_subject("user", "jane-o-neill", "jane o'neill")),
    (
        "user:Alice@Corp.com",
        "user",
        _lossy_subject("user", "alice-corp-com", "alice@corp.com"),
    ),
    (
        "service_account:ETL Nightly",
        "user",
        _lossy_subject("service_account", "etl-nightly", "etl nightly"),
    ),
    (
        "etl nightly",
        "service_account",
        _lossy_subject("service_account", "etl-nightly", "etl nightly"),
    ),
    (
        "etl nightly",
        "robot",
        _lossy_subject("user", "etl-nightly", "etl nightly"),
    ),  # unknown kind → user
    (
        "agent:helper",
        "user",
        _lossy_subject("user", "agent-helper", "agent:helper"),
    ),  # NOT a directory type
    ("山田太郎", "user", "user:id-" + hashlib.sha256("山田太郎".encode()).hexdigest()[:32]),
]


@pytest.mark.parametrize("raw,kind,expected", GO_PARITY_VECTORS)
def test_local_subject_matches_the_go_rule_for_tricky_inputs(raw, kind, expected):
    assert identity.local_subject(raw, kind) == expected


@pytest.mark.parametrize("raw,kind,expected", GO_PARITY_VECTORS)
def test_offline_resolve_returns_exactly_the_local_subject(raw, kind, expected, shield_factory):
    """Guarantee 4: with the gateway down, ``resolve_principal`` must return the
    SAME string ``local_subject`` derives - for every vector, not just the happy
    email path - so a decision minted offline still lines up with the tuples the
    server writes once it is reachable."""
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway down")

    engine = shield_factory(down).agentic.engine
    resolved = identity.resolve_principal(engine, subject=raw, kind=kind)

    assert resolved == identity.local_subject(raw, kind) == expected


# ── resolve_principal ────────────────────────────────────────────────────


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/vk-credential-info"):
        return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
    if path.endswith("/agentic-new/identity/resolve"):
        body = json.loads(request.content)
        _CALLS.append(("resolve", body))
        return httpx.Response(
            200,
            json={"subject": "user:alice-corp-com", "created": True, "principal": {"id": "p1"}},
        )
    if path.endswith("/agentic-new/registry/discover"):
        _CALLS.append(("discover", json.loads(request.content)))
        return httpx.Response(200, json={"network_id": "n1", "agents_upserted": 1})
    if path.endswith("/agentic-new/decide"):
        return httpx.Response(
            200,
            json={
                "verdict": "ALLOW",
                "allow": True,
                "mode": "enforce",
                "proceed": True,
            },
        )
    if path.endswith("/agentic-security/decide"):
        return httpx.Response(200, json={"verdict": "ALLOW", "decision_id": "d1"})
    return httpx.Response(404, json={})


_CALLS: list[tuple[str, dict]] = []


@pytest.fixture(autouse=True)
def _reset_calls():
    _CALLS.clear()
    yield
    _CALLS.clear()


def test_resolve_principal_uses_server_subject_and_caches(shield_factory):
    shield = shield_factory(_handler)

    first = identity.resolve_principal(shield.agentic.engine, email="alice@corp.com")
    second = identity.resolve_principal(shield.agentic.engine, email="Alice@Corp.com")

    assert first == second == "user:alice-corp-com"
    # Second lookup is a dict hit - the hot path never re-hits the gateway.
    assert [c for c in _CALLS if c[0] == "resolve"] == [
        ("resolve", {"kind": "user", "email": "alice@corp.com"})
    ]


def test_identity_cache_does_not_skip_jit_for_a_second_client(shield_factory):
    first = shield_factory(_handler)
    second = shield_factory(_handler)

    assert first.agentic.identity(email="alice@corp.com") == "user:alice-corp-com"
    assert second.agentic.identity(email="alice@corp.com") == "user:alice-corp-com"

    assert len([c for c in _CALLS if c[0] == "resolve"]) == 2


def test_resolve_principal_falls_back_locally_when_gateway_is_down(shield_factory):
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway down")

    shield = shield_factory(down)
    subject = identity.resolve_principal(shield.agentic.engine, email="alice@corp.com")

    # Same string the server would have produced → tuples still line up later.
    assert subject == identity.local_subject("alice@corp.com") == _lossy_subject(
        "user", "alice-corp-com", "alice@corp.com"
    )


def test_resolve_principal_does_not_cache_the_fallback(shield_factory):
    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/identity/resolve") and not _CALLS:
            _CALLS.append(("resolve", {}))
            return httpx.Response(503, json={})
        return _handler(request)

    shield = shield_factory(flaky)
    assert identity.resolve_principal(
        shield.agentic.engine, email="alice@corp.com"
    ) == _lossy_subject("user", "alice-corp-com", "alice@corp.com")
    # The retry hit the recovered gateway rather than serving a stale fallback.
    assert identity.resolve_principal(shield.agentic.engine, email="alice@corp.com") == "user:alice-corp-com"
    assert len([c for c in _CALLS if c[0] == "resolve"]) == 2


def test_resolve_principal_requires_an_identifier(shield_factory):
    shield = shield_factory(_handler)
    with pytest.raises(GovernanceConfigurationError) as caught:
        identity.resolve_principal(shield.agentic.engine)
    assert caught.value.code == "principal_identifier_missing"
    assert str(caught.value) == "principal_identifier_missing"


def test_as_user_binds_the_obo_leg_to_the_actor_chain(shield_factory):
    shield = shield_factory(_handler)

    assert shield.agentic.as_user("alice@corp.com") == "user:alice-corp-com"
    assert identity.obo_actor_chain(shield.agentic.engine) == ["user:alice-corp-com"]

    shield.agentic.as_user()  # clearing restores the agent-only chain
    assert identity.obo_actor_chain(shield.agentic.engine) == []


def test_obo_actor_chain_uses_only_the_vk_bound_agent(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:vk-bound",
                },
            )
        return _handler(request)

    shield = shield_factory(handler)
    assert identity.obo_actor_chain(shield.agentic.engine) == ["agent:vk-bound"]

    shield.agentic.as_user("alice@corp.com")
    assert identity.obo_actor_chain(shield.agentic.engine) == [
        "user:alice-corp-com",
        "agent:vk-bound",
    ]


def test_as_user_is_request_local_on_a_shared_client(shield_factory):
    """Two web requests sharing one client cannot overwrite each other's OBO leg."""
    barrier = threading.Barrier(2)

    def dynamic_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/identity/resolve"):
            body = json.loads(request.content)
            identifier = body.get("email") or body.get("username")
            return httpx.Response(
                200, json={"subject": identity.local_subject(identifier)}
            )
        return _handler(request)

    shield = shield_factory(dynamic_handler)
    alice_subject = identity.local_subject("alice@corp.com")
    bob_subject = identity.local_subject("bob@corp.com")

    def worker(email: str):
        shield.agentic.as_user(email)
        barrier.wait(5)
        return (
            shield.agentic.engine.acting_user,
            identity.principal_manifest(shield.agentic.engine),
            identity.obo_actor_chain(shield.agentic.engine),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        alice, bob = list(
            pool.map(worker, ["alice@corp.com", "bob@corp.com"])
        )

    assert alice == (
        alice_subject,
        {
            "email": "alice@corp.com",
            "subject": alice_subject,
            "kind": "user",
        },
        [alice_subject],
    )
    assert bob == (
        bob_subject,
        {
            "email": "bob@corp.com",
            "subject": bob_subject,
            "kind": "user",
        },
        [bob_subject],
    )
    assert shield.agentic.engine.acting_user == ""  # worker contexts did not leak


def test_run_scope_restores_nested_identity_and_session(shield_factory):
    def dynamic_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/identity/resolve"):
            body = json.loads(request.content)
            identifier = body.get("email") or body.get("username")
            return httpx.Response(
                200, json={"subject": identity.local_subject(identifier)}
            )
        return _handler(request)

    shield = shield_factory(dynamic_handler)
    default_session = shield.agentic.engine.session_id
    alice_subject = identity.local_subject("alice@corp.com")

    with shield.agentic.run(email="alice@corp.com", session_id="run-alice") as run_id:
        assert run_id == shield.agentic.engine.session_id == "run-alice"
        assert shield.agentic.engine.acting_user == alice_subject
        with shield.agentic.run(username="bob", session_id="run-bob"):
            assert shield.agentic.engine.session_id == "run-bob"
            assert shield.agentic.engine.acting_user == "user:bob"
        assert shield.agentic.engine.session_id == "run-alice"
        assert shield.agentic.engine.acting_user == alice_subject

    assert shield.agentic.engine.session_id == default_session
    assert shield.agentic.engine.acting_user == ""


# ── discover + 5-minute dedupe ───────────────────────────────────────────


def test_discover_posts_the_contract_payload(shield_factory):
    shield = shield_factory(_handler)

    body = registry.discover(
        shield.agentic.engine, _support_graph(), principal_email="alice@corp.com", sync=True
    )

    assert body == {"network_id": "n1", "agents_upserted": 1}
    (_, sent), = [c for c in _CALLS if c[0] == "discover"]
    assert sent["framework"] == "langgraph"
    assert sent["auto_provision"] is True
    assert sent["principal"] == {"email": "alice@corp.com"}
    assert sent["network"]["digest"].startswith("sha256:")
    assert {"nodes", "edges", "agents", "tools", "servers"} <= set(sent)


def test_discover_keys_raw_content_attestation_before_transport(shield_factory):
    shield = shield_factory(_handler)
    manifest = {
        "network": {"key": "wire-attestation"},
        "nodes": [
            {"id": "worker", "kind": "agent"},
            {"id": "send", "kind": "tool"},
        ],
        "edges": [{"from": "worker", "to": "send"}],
        "agents": [{"key": "worker"}],
        "tools": [{"key": "send"}],
        "code_artifacts": [
            {
                "subject_kind": "tool",
                "subject_key": "send",
                "language": "python",
                "source": (
                    "def send():\n"
                    "    api_key = 'sk-low-entropy-secret-000000'\n"
                    "    return api_key\n"
                ),
            }
        ],
    }
    local = registry.describe_network(manifest)
    registry.discover(shield.agentic.engine, manifest=manifest, sync=True)

    (_, sent), = [c for c in _CALLS if c[0] == "discover"]
    local_by_subject = {
        (item["subject_kind"], item["subject_key"], item["digest"]): item
        for item in local["code_artifacts"]
    }
    assert sent["code_artifacts"]
    for artifact in sent["code_artifacts"]:
        local_artifact = local_by_subject[
            (artifact["subject_kind"], artifact["subject_key"], artifact["digest"])
        ]
        raw_attestation = local_artifact["self_attested_content_digest"]
        expected = "sha256:" + hmac.new(
            b"sk-ds-test",
            b"deepintshield-blueprint-content-v1\x00" + raw_attestation.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        assert artifact["self_attested_content_digest"] == expected
        assert artifact["self_attested_content_digest"] != raw_attestation
        assert raw_attestation not in json.dumps(sent)
        assert "sk-low-entropy-secret" not in json.dumps(sent)
    assert sent["self_attested_code_digest"] != local["self_attested_code_digest"]


def test_discover_carries_the_bound_user_when_no_email_is_given(shield_factory):
    shield = shield_factory(_handler)
    shield.agentic.as_user("alice@corp.com")

    registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    (_, sent), = [c for c in _CALLS if c[0] == "discover"]
    assert sent["principal"] == {
        "email": "alice@corp.com",
        "subject": "user:alice-corp-com",
        "kind": "user",
    }


def test_discover_dedupes_the_same_digest_for_five_minutes(shield_factory):
    shield = shield_factory(_handler)

    first = registry.discover(shield.agentic.engine, _support_graph(), sync=True)
    second = registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    assert first == {"network_id": "n1", "agents_upserted": 1}
    assert second["deduped"] is True and second["dispatched"] is False
    assert second["digest"] == registry.describe_network(_support_graph())["network"]["digest"]
    assert len([c for c in _CALLS if c[0] == "discover"]) == 1


def test_discovery_dedupe_is_isolated_per_client(shield_factory):
    first = shield_factory(_handler)
    second = shield_factory(_handler)

    registry.discover(first.agentic.engine, _support_graph(), sync=True)
    registry.discover(second.agentic.engine, _support_graph(), sync=True)

    assert len([c for c in _CALLS if c[0] == "discover"]) == 2


def test_same_topology_reposts_for_a_different_bound_principal(shield_factory):
    shield = shield_factory(_handler)
    shield.agentic.as_user("alice@corp.com")
    registry.discover(shield.agentic.engine, _support_graph(), sync=True)
    # Avoid the fixed resolver response: the point here is the retained source
    # identity being part of the manifest/dedupe scope.
    shield.agentic.engine.bind_principal(
        identity.PrincipalBinding(
            subject="user:bob-corp-com", email="bob@corp.com"
        )
    )
    registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    posts = [body for kind, body in _CALLS if kind == "discover"]
    assert [post["principal"]["email"] for post in posts] == [
        "alice@corp.com",
        "bob@corp.com",
    ]


def test_discover_reposts_once_the_window_elapses(shield_factory, monkeypatch):
    shield = shield_factory(_handler)

    registry.discover(shield.agentic.engine, _support_graph(), sync=True)
    monkeypatch.setattr(registry, "REPOST_INTERVAL_SECONDS", 0.0)
    registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    assert len([c for c in _CALLS if c[0] == "discover"]) == 2


def test_discover_reports_a_changed_topology_immediately(shield_factory):
    shield = shield_factory(_handler)
    registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    drifted = _support_graph()
    drifted.nodes["enrich"] = FakeNode()
    drifted._edges.append(FakeEdge("location_extract", "enrich"))
    registry.discover(shield.agentic.engine, drifted, sync=True)

    assert len([c for c in _CALLS if c[0] == "discover"]) == 2


def test_discover_reports_schema_action_or_server_drift_without_waiting(shield_factory):
    shield = shield_factory(_handler)
    base = {
        "framework": "custom",
        "network": {"key": "billing"},
        "nodes": [{"id": "agent", "kind": "agent"}, {"id": "charge", "kind": "tool"}],
        "edges": [{"from": "agent", "to": "charge"}],
        "tools": [
            {
                "key": "charge",
                "action_class": "read",
                "schema_digest": "schema-v1",
                "server": "payments-v1",
            }
        ],
    }
    drifted = {
        **base,
        "tools": [
            {
                "key": "charge",
                "action_class": "write",
                "schema_digest": "schema-v2",
                "server": "payments-v2",
            }
        ],
    }

    first = registry.discover(shield.agentic.engine, manifest=base, sync=True)
    second = registry.discover(shield.agentic.engine, manifest=drifted, sync=True)

    assert first == second == {"network_id": "n1", "agents_upserted": 1}
    assert len([c for c in _CALLS if c[0] == "discover"]) == 2
    # Topology digest stays stable; the full scoped manifest key caused repost.
    posts = [body for kind, body in _CALLS if kind == "discover"]
    assert posts[0]["network"]["digest"] == posts[1]["network"]["digest"]


def test_discover_never_raises_on_an_unreachable_gateway(shield_factory):
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway down")

    shield = shield_factory(down)
    body = registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    assert "error" in body


def test_failed_discovery_is_retriable_immediately(shield_factory):
    attempts = 0

    def down(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("gateway down")

    shield = shield_factory(down)
    first = registry.discover(shield.agentic.engine, _support_graph(), sync=True)
    second = registry.discover(shield.agentic.engine, _support_graph(), sync=True)

    assert "error" in first and "error" in second
    assert attempts == 2


def test_discover_is_fire_and_forget_by_default(shield_factory):
    shield = shield_factory(_handler)

    result = shield.agentic.discover(_support_graph())

    assert result["dispatched"] is True
    assert result["digest"].startswith("sha256:")
