"""Source capture against installed SDKs, without external model/gateway calls.

Install ``deepintshield[all,dev]`` to run every case. Each optional framework is
skipped independently in a minimal SDK environment. These tests use real tool
objects so an SDK changing its callable storage cannot pass through a mock.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import sys
import textwrap
import types

import httpx
import pytest

from deepintshield.agentic import registry
from deepintshield.agentic.integrations._common import wrap_callable
from deepintshield.agentic.manifest import describe


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked_connect(*args, **kwargs):
        raise AssertionError("Source-capture tests must not make network calls")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_connect)


def local_increment(value: int) -> int:
    return value + 2


def add_two(value: int) -> int:
    """Add two to the supplied integer."""
    return local_increment(value)


def math_node(state: dict) -> dict:
    return {"value": local_increment(state["value"])}


def before_tool_callback(*args, **kwargs):
    return None


def select_math_agent(messages):
    return "math_agent"


def strands_callback(**kwargs):
    return None


def openai_instructions(context, agent):
    return "Use the arithmetic tools."


async def openai_guardrail(context, agent, value):
    return None


async def openai_handoff(context):
    return None


def multiply_three(value: int) -> int:
    """Multiply the supplied integer by three."""
    return value * 3


def pydantic_instructions(context):
    return "Use the arithmetic tools."


def pydantic_system_prompt(context):
    return "Return one integer."


def pydantic_output_validator(context, value):
    return value


async def pydantic_prepare(context, tool_definition):
    return tool_definition


_FRAMEWORKS = (
    "langchain",
    "openai_agents",
    "pydanticai",
    "google_adk",
    "llamaindex",
    "autogen",
    "strands",
    "temporal",
)


def _real_tool(framework, fn):
    if framework == "langchain":
        tools = pytest.importorskip("langchain_core.tools")
        return tools.StructuredTool.from_function(fn)
    if framework == "openai_agents":
        agents = pytest.importorskip("agents")
        return agents.function_tool(fn)
    if framework == "pydanticai":
        tools = pytest.importorskip("pydantic_ai.tools")
        return tools.Tool(fn)
    if framework == "google_adk":
        tools = pytest.importorskip("google.adk.tools")
        return tools.FunctionTool(func=fn)
    if framework == "llamaindex":
        tools = pytest.importorskip("llama_index.core.tools")
        return tools.FunctionTool.from_defaults(fn=fn)
    if framework == "autogen":
        tools = pytest.importorskip("autogen_core.tools")
        return tools.FunctionTool(fn, description="Add two")
    if framework == "strands":
        strands = pytest.importorskip("strands")
        return strands.tool(fn)
    if framework == "temporal":
        activity = pytest.importorskip("temporalio.activity")
        # Temporal attaches its definition to the function. Give each case an
        # independent function so the original/governed cases cannot share it.
        copied = types.FunctionType(
            fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
        )
        copied.__annotations__ = dict(fn.__annotations__)
        copied.__dict__.update(fn.__dict__)
        return activity.defn(copied)
    raise AssertionError(framework)


def _assert_tool_capture(payload, name="add_two"):
    assert payload.get("code_artifacts_incomplete") is not True, payload.get(
        "code_artifact_coverage"
    )
    assert any(tool["key"] == name for tool in payload["tools"])
    artifacts = [
        artifact
        for artifact in payload["code_artifacts"]
        if artifact["subject_kind"] == "tool" and artifact["subject_key"] == name
    ]
    assert artifacts
    source = "\n".join(artifact["source"] for artifact in artifacts)
    assert "def add_two" in source
    assert "def local_increment" in source
    assert "return value + 2" in source


@pytest.mark.parametrize("framework", _FRAMEWORKS)
@pytest.mark.parametrize("governed", (False, True), ids=("original", "governed"))
def test_real_framework_tool_has_complete_application_source(
    framework, governed, monkeypatch
):
    # Reproduce an app whose source tree also contains an editable SDK checkout.
    # SDK instrumentation must not pull its internal policy machinery into the
    # local function bundle, but the application's helper must remain covered.
    project_root = str(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(registry, "_application_source_root", lambda _: project_root)
    fn = wrap_callable(object(), "add_two", add_two) if governed else add_two
    tool = _real_tool(framework, fn)

    _assert_tool_capture(registry.describe_network([tool]))
    legacy = describe([tool]).to_dict()
    assert legacy["tools"] == ["add_two"]
    assert "def add_two" in legacy["tool_sources"]["add_two"]
    assert "def local_increment" in legacy["tool_sources"]["add_two"]


def test_real_langgraph_nodes_capture_after_instrumentation(monkeypatch, shield_factory):
    graph = pytest.importorskip("langgraph.graph")
    from deepintshield.agentic.enforcement import engine_scope
    from deepintshield.agentic.integrations import langgraph

    project_root = str(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(registry, "_application_source_root", lambda _: project_root)
    # Other integration tests may have installed the process-wide compile
    # guard. Exercise it with a real, explicitly scoped engine and an in-memory
    # registration acknowledgement; do not unwrap or disable that boundary.
    shield = shield_factory(
        lambda request: httpx.Response(
            200,
            json={"blueprint_id": "math-test", "blueprint_scan_status": "complete"},
        )
    )
    engine = shield.agentic.engine
    builder = graph.StateGraph(dict)
    builder.add_node(
        "math_node", langgraph._wrap_fn(engine, "math_node", math_node, is_async=False)
    )
    builder.add_edge(graph.START, "math_node")
    builder.add_edge("math_node", graph.END)
    with engine_scope(engine):
        app = builder.compile()
        # Finish any compile-time mocked discovery before the fixture closes.
        langgraph._await_initial_discovery(engine, app)

    payload = registry.describe_network(app)
    assert payload["framework"] == "langgraph"
    assert payload.get("code_artifacts_incomplete") is not True
    assert payload["code_artifacts"]
    source = "\n".join(item["source"] for item in payload["code_artifacts"])
    assert "def math_node" in source
    assert "def local_increment" in source


def test_real_openai_agent_captures_function_tool():
    agents = pytest.importorskip("agents")
    agent = agents.Agent(name="math_agent", tools=[agents.function_tool(add_two)])
    _assert_tool_capture(registry.describe_network(agent))


def test_real_openai_agent_captures_callbacks_and_handoff_agent():
    agents = pytest.importorskip("agents")
    child = agents.Agent(name="child_math_agent", tools=[agents.function_tool(multiply_three)])
    agent = agents.Agent(
        name="math_agent",
        tools=[agents.function_tool(add_two)],
        instructions=openai_instructions,
        input_guardrails=[agents.input_guardrail(openai_guardrail)],
        output_guardrails=[agents.output_guardrail(openai_guardrail)],
        handoffs=[agents.handoff(child, on_handoff=openai_handoff)],
    )
    payload = registry.describe_network(agent)
    _assert_tool_capture(payload)
    source = "\n".join(item["source"] for item in payload["code_artifacts"])
    for callback in ("openai_instructions", "openai_guardrail", "openai_handoff", "multiply_three"):
        assert f"def {callback}" in source


def test_real_pydantic_agent_captures_internal_function_registry():
    pydantic_ai = pytest.importorskip("pydantic_ai")
    agent = pydantic_ai.Agent(name="math_agent", tools=[add_two])
    _assert_tool_capture(registry.describe_network(agent))


def test_real_pydantic_agent_captures_prompt_validator_and_tool_callbacks():
    pydantic_ai = pytest.importorskip("pydantic_ai")
    agent = pydantic_ai.Agent(
        name="math_agent",
        instructions=pydantic_instructions,
        tools=[pydantic_ai.Tool(add_two, prepare=pydantic_prepare)],
    )
    agent.system_prompt(pydantic_system_prompt)
    agent.output_validator(pydantic_output_validator)
    payload = registry.describe_network(agent)
    _assert_tool_capture(payload)
    source = "\n".join(item["source"] for item in payload["code_artifacts"])
    for callback in (
        "pydantic_instructions", "pydantic_system_prompt",
        "pydantic_output_validator", "pydantic_prepare",
    ):
        assert f"def {callback}" in source


def test_real_google_adk_captures_tools_and_callback():
    agents = pytest.importorskip("google.adk.agents")
    tools = pytest.importorskip("google.adk.tools")
    agent = agents.Agent(
        name="math_agent",
        model="gemini-2.5-flash",
        tools=[tools.FunctionTool(func=add_two)],
        before_tool_callback=before_tool_callback,
    )
    payload = registry.describe_network(agent)
    _assert_tool_capture(payload)
    assert any(
        item["subject_kind"] == "agent" and "def before_tool_callback" in item["source"]
        for item in payload["code_artifacts"]
    )


def test_real_llamaindex_agent_captures_function_tool():
    agents = pytest.importorskip("llama_index.core.agent.workflow")
    llms = pytest.importorskip("llama_index.llms.openai")
    llm = llms.OpenAI(model="gpt-4o-mini", api_key="test-no-network")
    agent = agents.FunctionAgent(name="math_agent", tools=[add_two], llm=llm)
    _assert_tool_capture(registry.describe_network(agent))


def test_real_autogen_team_captures_tools_and_selector():
    agents = pytest.importorskip("autogen_agentchat.agents")
    teams = pytest.importorskip("autogen_agentchat.teams")
    models = pytest.importorskip("autogen_ext.models.openai")
    client = models.OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="test-no-network")
    agent = agents.AssistantAgent("math_agent", model_client=client, tools=[add_two])
    second = agents.AssistantAgent("second_agent", model_client=client)
    team = teams.SelectorGroupChat(
        [agent, second], model_client=client, selector_func=select_math_agent
    )
    payload = registry.describe_network(team)
    _assert_tool_capture(payload)
    assert any("def select_math_agent" in item["source"] for item in payload["code_artifacts"])


def test_real_strands_agent_captures_tools_and_callback():
    strands = pytest.importorskip("strands")
    models = pytest.importorskip("strands.models.openai")
    model = models.OpenAIModel(model_id="gpt-4o-mini", client_args={"api_key": "test-no-network"})
    agent = strands.Agent(
        name="math_agent",
        model=model,
        tools=[strands.tool(add_two)],
        callback_handler=strands_callback,
    )
    payload = registry.describe_network(agent)
    _assert_tool_capture(payload)
    assert any("def strands_callback" in item["source"] for item in payload["code_artifacts"])


def test_real_temporal_workflow_captures_registered_method_helpers(tmp_path, monkeypatch):
    worker_module = pytest.importorskip("temporalio.worker")
    source_path = tmp_path / "saved_math_workflow.py"
    source_path.write_text(
        textwrap.dedent(
            """\
            from temporalio import workflow

            def local_increment(value: int) -> int:
                return value + 2

            @workflow.defn
            class MathWorkflow:
                @workflow.run
                async def run(self, value: int) -> int:
                    return local_increment(value)
            """
        )
    )
    spec = importlib.util.spec_from_file_location("saved_math_workflow", source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Use the installed Worker class and actual decorated workflow definition,
    # supplying just its registration config: creating a live worker requires a
    # Temporal service and is outside this source-introspection test's scope.
    worker = object.__new__(worker_module.Worker)
    worker._config = {
        "task_queue": "math-test",
        "workflows": [module.MathWorkflow],
        "activities": [],
    }
    payload = registry.describe_network(worker)
    assert payload["framework"] == "temporal"
    assert payload.get("code_artifacts_incomplete") is not True
    source = "\n".join(item["source"] for item in payload["code_artifacts"])
    assert "class MathWorkflow" in source
    assert "def local_increment" in source
    assert "return value + 2" in source
