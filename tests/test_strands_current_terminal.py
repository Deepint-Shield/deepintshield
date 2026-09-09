"""Exercise the installed Strands terminal, including fail-closed decisions."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from deepintshield.agentic import enforcement
from deepintshield.agentic.errors import GatewayUnavailable, GuardrailDenied
from deepintshield.agentic.integrations import strands


@pytest.fixture
def terminal(monkeypatch):
    executor = pytest.importorskip("strands.tools.executors._executor")
    stages = pytest.importorskip("strands._middleware.stages")
    if not hasattr(executor, "_make_execute_tool_terminal"):
        pytest.skip("installed Strands uses the legacy executor boundary")
    original = executor._make_execute_tool_terminal
    monkeypatch.setattr(executor, "_make_execute_tool_terminal", original)
    monkeypatch.setattr(enforcement, "report_topology_for", lambda *_: None)
    monkeypatch.setattr(enforcement, "resolve_engine", lambda *_: object())
    monkeypatch.setattr(enforcement, "bind_engine", lambda *a, **k: None)
    order = []
    monkeypatch.setattr(enforcement, "ensure_topology_reported", lambda *a, **k: order.append("topology"))
    assert strands.enforce()
    assert strands._automatic_guard_active()
    # The retained explicit helper uses the installed public hook event name.
    assert strands.hook_provider(object())._deepintshield_hook
    invoked = []

    class Tool:
        name = "final_tool"

        async def stream(self, tool_use, invocation_state, **kwargs):
            order.append("execute")
            invoked.append((tool_use["input"], invocation_state, kwargs))
            yield {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": "done"}]}

    ctx = stages.ExecuteToolContext(
        agent=object(), tool=Tool(),
        tool_use={"name": "final_tool", "toolUseId": "tool-1", "input": {"email": "user@example.test"}},
        invocation_state={"caller": "test"}, cancel_signal=threading.Event(), _interrupt_state=None,
    )
    return executor._make_execute_tool_terminal({"native_option": True}), ctx, invoked, order


async def collect(terminal, ctx):
    return [event async for event in terminal(ctx)]


@pytest.mark.parametrize("mask", [False, True])
def test_native_terminal_authorizes_final_selection_and_masks(monkeypatch, terminal, mask):
    native, ctx, invoked, order = terminal
    selected = ctx.tool

    def allowed(engine, name, args, kwargs, **options):
        order.append("authorize")
        assert name == "final_tool"
        assert kwargs == {"email": "user@example.test"}
        assert options["tool_callable"] is selected
        return SimpleNamespace(obligations=["mask:pii"] if mask else [])

    monkeypatch.setattr(strands, "resolve", allowed)
    assert asyncio.run(collect(native, ctx))
    assert order == ["topology", "authorize", "execute"]
    assert invoked == [({"email": "***" if mask else "user@example.test"}, {"caller": "test"}, {"native_option": True})]


@pytest.mark.parametrize("error,expected", [(GuardrailDenied(reason="blocked"), PermissionError), (GatewayUnavailable(reason="offline"), ConnectionError)])
def test_native_terminal_denial_and_pdp_failure_prevent_execution(monkeypatch, terminal, error, expected):
    native, ctx, invoked, order = terminal

    def blocked(*args, **kwargs):
        order.append("authorize")
        raise error

    monkeypatch.setattr(strands, "resolve", blocked)
    with pytest.raises(expected):
        asyncio.run(collect(native, ctx))
    assert invoked == []
    assert order == ["topology", "authorize"]


def test_native_terminal_unknown_context_fails_closed(terminal):
    native, ctx, invoked, _ = terminal
    with pytest.raises(RuntimeError, match="governance_configuration_error"):
        asyncio.run(collect(native, SimpleNamespace(**ctx.__dict__)))
    assert invoked == []
