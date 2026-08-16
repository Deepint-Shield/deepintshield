from __future__ import annotations

import asyncio
import contextlib
import sys
import types

import pytest

from deepintshield.agentic.errors import GatewayUnavailable
from deepintshield.agentic.types import Decision, Verdict


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_current_pydanticai_shape(monkeypatch):
    class ToolManager:
        def __init__(self, engine):
            self._deepintshield_engine = engine
            self.executed = 0
            self.received = None
            self.toolset = types.SimpleNamespace(tools={})

        async def execute_tool_call(self, validated, **_kwargs):
            self.executed += 1
            self.received = validated
            return validated.validated_args

    package = _module("pydantic_ai")
    package.__path__ = []
    tool_manager = _module(
        "pydantic_ai.tool_manager",
        ToolManager=ToolManager,
    )
    monkeypatch.setitem(sys.modules, "pydantic_ai", package)
    monkeypatch.setitem(sys.modules, "pydantic_ai.tool_manager", tool_manager)
    return ToolManager


def _validated_call(**tool_args):
    toolset = types.SimpleNamespace(tools={})
    tool = types.SimpleNamespace(
        tool_def=types.SimpleNamespace(name="send_email", kind="function"),
        toolset=toolset,
    )
    return types.SimpleNamespace(
        args_valid=True,
        call=types.SimpleNamespace(tool_name="send_email"),
        tool=tool,
        validated_args=tool_args,
    )


@pytest.fixture
def no_execution_telemetry(monkeypatch):
    from deepintshield.agentic import execution

    @contextlib.contextmanager
    def scope(*_args, **_kwargs):
        yield types.SimpleNamespace()

    monkeypatch.setattr(execution, "current_execution", lambda *_args: None)
    monkeypatch.setattr(execution, "execution_scope", scope)


def test_current_pydanticai_tool_manager_masks_at_final_dispatch(
    monkeypatch,
    no_execution_telemetry,
):
    from deepintshield.agentic.integrations import pydanticai

    manager_cls = _install_current_pydanticai_shape(monkeypatch)
    engine = object()
    manager = manager_cls(engine)
    original = _validated_call(email="alice@example.com", subject="Hello")

    monkeypatch.setattr(
        pydanticai,
        "resolve",
        lambda *_args, **_kwargs: Decision(
            verdict=Verdict.MASK,
            decision_id="decision-1",
            obligations=["mask:pii"],
        ),
    )

    assert pydanticai.enforce() is True
    result = asyncio.run(manager.execute_tool_call(original))

    assert result == {"email": "***", "subject": "Hello"}
    assert manager.executed == 1
    assert manager.received is not original
    assert original.validated_args["email"] == "alice@example.com"


def test_current_pydanticai_tool_manager_fails_closed_before_execution(
    monkeypatch,
    no_execution_telemetry,
):
    from deepintshield.agentic.integrations import pydanticai

    manager_cls = _install_current_pydanticai_shape(monkeypatch)
    manager = manager_cls(object())

    def unavailable(*_args, **_kwargs):
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(pydanticai, "resolve", unavailable)

    assert pydanticai.enforce() is True
    with pytest.raises(ConnectionError) as caught:
        asyncio.run(manager.execute_tool_call(_validated_call(amount=10)))

    assert caught.value._deepintshield_error_code == "gateway_unavailable"
    assert manager.executed == 0


def test_explicit_pydanticai_helper_does_not_double_wrap_current_boundary(
    monkeypatch,
):
    from deepintshield.agentic.integrations import pydanticai

    _install_current_pydanticai_shape(monkeypatch)
    assert pydanticai.enforce() is True

    def send_email(email: str):
        return email

    registered = types.SimpleNamespace(function=send_email, name="send_email")
    agent = types.SimpleNamespace(
        _function_toolset=types.SimpleNamespace(tools={"send_email": registered})
    )
    engine = object()

    assert pydanticai.shield_agent(agent, engine=engine) is agent
    assert registered.function is send_email
    assert agent._deepintshield_engine is engine
    assert registered._deepintshield_engine is engine


def test_google_adk_runner_discovers_nested_agents_and_tools():
    from deepintshield.agentic.registry import describe_network

    class Agent:
        def __init__(self, name, *, tools=(), sub_agents=(), model="gemini"):
            self.name = name
            self.tools = list(tools)
            self.sub_agents = list(sub_agents)
            self.model = model

    class Runner:
        def __init__(self, agent):
            self.agent = agent

    Agent.__module__ = "google.adk.agents"
    Runner.__module__ = "google.adk.runners"
    search = types.SimpleNamespace(name="search_web")
    refund = types.SimpleNamespace(name="issue_refund")
    specialist = Agent("refund_specialist", tools=[refund])
    runner = Runner(
        Agent("support_router", tools=[search], sub_agents=[specialist])
    )

    manifest = describe_network(runner)

    assert manifest["framework"] == "google_adk"
    assert {agent["key"] for agent in manifest["agents"]} == {
        "support_router",
        "refund_specialist",
    }
    assert {tool["key"] for tool in manifest["tools"]} == {
        "search_web",
        "issue_refund",
    }
    assert {
        (edge["from"], edge["to"])
        for edge in manifest["edges"]
    } >= {
        ("input", "support_router"),
        ("support_router", "refund_specialist"),
        ("support_router", "search_web"),
        ("refund_specialist", "issue_refund"),
    }


def test_strands_tool_registry_uses_governed_tool_names():
    from deepintshield.agentic.registry import describe_network

    class Agent:
        name = "payments_agent"
        model = "bedrock/claude"

        def __init__(self):
            self.tool_registry = types.SimpleNamespace(
                registry={
                    "wire_funds": types.SimpleNamespace(
                        tool_name="wire_funds",
                        description="Transfer funds",
                    )
                }
            )

    Agent.__module__ = "strands.agent.agent"

    manifest = describe_network(Agent())

    assert manifest["framework"] == "strands"
    assert [tool["key"] for tool in manifest["tools"]] == ["wire_funds"]
    assert manifest["agents"][0]["capabilities"] == ["wire_funds"]


def test_temporal_worker_discovers_workflows_and_activity_surface():
    from deepintshield.agentic.registry import describe_network

    class ReconcilePayments:
        pass

    def charge_card():
        pass

    def issue_refund():
        pass

    class Worker:
        def __init__(self):
            self._config = {
                "task_queue": "payments",
                "workflows": [ReconcilePayments],
                "activities": [charge_card, issue_refund],
            }

    Worker.__module__ = "temporalio.worker._worker"

    manifest = describe_network(Worker())

    assert manifest["framework"] == "temporal"
    assert len(manifest["agents"]) == 1
    assert manifest["agents"][0]["name"] == "ReconcilePayments"
    workflow_key = manifest["agents"][0]["key"]
    workflow_node = next(node for node in manifest["nodes"] if node["id"] == workflow_key)
    assert workflow_node["executable"] is True
    assert any(
        artifact["subject_kind"] == "agent"
        and artifact["subject_key"] == workflow_key
        for artifact in manifest["code_artifacts"]
    )
    assert {tool["key"] for tool in manifest["tools"]} == {
        "charge_card",
        "issue_refund",
    }
    assert set(manifest["agents"][0]["capabilities"]) == {
        "charge_card",
        "issue_refund",
    }


def test_temporal_worker_without_workflows_is_explicitly_structural():
    from deepintshield.agentic.registry import describe_network

    def heartbeat():
        return "ok"

    class Worker:
        def __init__(self):
            self._config = {
                "task_queue": "monitoring",
                "workflows": [],
                "activities": [heartbeat],
            }

    Worker.__module__ = "temporalio.worker._worker"
    manifest = describe_network(Worker())

    worker_node = next(node for node in manifest["nodes"] if node["kind"] == "agent")
    assert worker_node["executable"] is False
    assert worker_node["ref"] == "structural:temporal-worker"
    assert not any(
        artifact["subject_kind"] == "agent"
        for artifact in manifest["code_artifacts"]
    )
