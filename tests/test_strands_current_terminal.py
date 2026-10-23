"""Exercise the installed Strands terminal, including fail-closed decisions."""

import asyncio
import inspect
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


def _background_terminal_factory():
    from strands.tools.executors import _executor

    factory = _executor._make_execute_tool_terminal
    if "tool_context" not in inspect.signature(factory).parameters:
        pytest.skip("installed Strands predates background-tool terminal options")
    return factory


def test_native_terminal_preserves_background_context_and_tool_guard(monkeypatch, terminal):
    _, ctx, invoked, order = terminal
    factory = _background_terminal_factory()
    tool_context = object()
    validated = []

    def allowed(*args, **kwargs):
        order.append("authorize")
        return SimpleNamespace(obligations=[])

    def tool_guard(tool):
        order.append("native-guard")
        validated.append(tool)

    monkeypatch.setattr(strands, "resolve", allowed)
    native = factory(
        {"native_option": True}, tool_context=tool_context, tool_guard=tool_guard,
    )
    assert asyncio.run(collect(native, ctx))
    assert validated == [ctx.tool]
    assert order == ["topology", "authorize", "native-guard", "execute"]
    assert invoked == [(
        {"email": "user@example.test"},
        {"caller": "test"},
        {"native_option": True, "_tool_context": tool_context},
    )]


def test_native_terminal_preserves_native_tool_guard_rejection(monkeypatch, terminal):
    _, ctx, invoked, order = terminal
    factory = _background_terminal_factory()
    monkeypatch.setattr(strands, "resolve", lambda *a, **k: SimpleNamespace(obligations=[]))

    def tool_guard(tool):
        assert tool is ctx.tool
        raise ValueError("native tool rejected")

    native = factory({}, tool_context=object(), tool_guard=tool_guard)
    with pytest.raises(ValueError, match="native tool rejected"):
        asyncio.run(collect(native, ctx))
    assert invoked == []
    assert order == ["topology"]


@pytest.mark.parametrize("positional", [False, True])
def test_native_terminal_preset_rejection_does_not_authorize_or_execute(monkeypatch, terminal, positional):
    _, ctx, invoked, order = terminal
    factory = _background_terminal_factory()
    monkeypatch.setattr(strands, "resolve", lambda *a, **k: pytest.fail("rejected tool must not authorize"))
    result = {
        "toolUseId": "tool-1", "status": "error",
        "content": [{"text": "Native tool validation rejected the call"}],
    }
    native = factory({}, result) if positional else factory({}, preset_result=result)
    events = asyncio.run(collect(native, ctx))
    assert len(events) == 1
    assert events[0].tool_result == result
    assert invoked == []
    assert order == []


def test_unknown_terminal_factory_shape_is_rejected(monkeypatch, terminal):
    from strands.tools.executors import _executor

    def unsupported_factory(extra_kwargs, unknown_option=None):
        async def terminal(ctx):
            yield None
        return terminal

    monkeypatch.setattr(_executor, "_make_execute_tool_terminal", unsupported_factory)
    assert strands._enforce_middleware_terminal() is False
    assert _executor._make_execute_tool_terminal is unsupported_factory


def test_original_single_parameter_terminal_remains_supported(monkeypatch, terminal):
    from strands.tools.executors import _executor

    _, ctx, invoked, order = terminal
    original = inspect.unwrap(_executor._make_execute_tool_terminal)

    def legacy_factory(extra_kwargs):
        return original(extra_kwargs)

    monkeypatch.setattr(_executor, "_make_execute_tool_terminal", legacy_factory)
    monkeypatch.setattr(strands, "resolve", lambda *a, **k: SimpleNamespace(obligations=[]))
    assert strands._enforce_middleware_terminal() is True
    native = _executor._make_execute_tool_terminal({"legacy_option": True})
    assert asyncio.run(collect(native, ctx))
    assert order == ["topology", "execute"]
    assert invoked == [(
        {"email": "user@example.test"}, {"caller": "test"}, {"legacy_option": True},
    )]
