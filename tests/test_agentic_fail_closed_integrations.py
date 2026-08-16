from __future__ import annotations

import asyncio
import json
import sys
import traceback
import types

import pytest

from deepintshield.agentic.errors import (
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
    public_agentic_error_details,
)


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _assert_marked(error: BaseException, code: str) -> None:
    details = public_agentic_error_details(code)
    assert getattr(error, "_deepintshield_error_code") == code
    assert str(error) == f"{code}: {details['message']}"
    assert getattr(error, "_deepintshield_error_action") == details["action"]


def _adk_block(code: str) -> dict[str, str]:
    details = public_agentic_error_details(code)
    return {
        "error": code,
        "status": "blocked",
        "message": details["message"],
        "action": details["action"],
        "dashboard_path": details["dashboard_path"],
    }


def _hermes_directive(code: str) -> dict[str, str]:
    details = public_agentic_error_details(code)
    return {
        "action": "block",
        "code": code,
        "message": details["message"],
        "remediation": details["action"],
        "dashboard_path": details["dashboard_path"],
    }


def test_google_adk_plugin_propagates_pdp_outage(monkeypatch):
    from deepintshield.agentic.integrations import google_adk

    class BasePlugin:
        def __init__(self, name: str):
            self.name = name

    google = _module("google")
    google.__path__ = []
    adk = _module("google.adk")
    adk.__path__ = []
    plugins = _module("google.adk.plugins")
    plugins.__path__ = []
    base = _module("google.adk.plugins.base_plugin", BasePlugin=BasePlugin)
    for name, module in (
        ("google", google),
        ("google.adk", adk),
        ("google.adk.plugins", plugins),
        ("google.adk.plugins.base_plugin", base),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def unavailable(*_args, **_kwargs):
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(google_adk, "resolve", unavailable)
    callback = google_adk.plugin(object()).before_tool_callback

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(callback(tool=lambda: None, tool_args={"id": 1}))
    _assert_marked(caught.value, "gateway_unavailable")


def test_litellm_public_boundaries_use_builtin_gateway_error(monkeypatch):
    from deepintshield.agentic.integrations import litellm as integration

    provider_calls = 0

    def completion(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return "ran"

    async def acompletion(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return "ran"

    module = _module(
        "litellm",
        completion=completion,
        acompletion=acompletion,
    )
    monkeypatch.setitem(sys.modules, "litellm", module)

    def unavailable(*_args, **_kwargs):
        raise GatewayUnavailable(
            reason="PRIVATE gateway transport response",
            code="invalid_gateway_response",
            payload={"credential": "sk-private"},
        )

    monkeypatch.setattr(integration, "_check", unavailable)
    assert integration.enforce() is True

    with pytest.raises(ConnectionError) as synchronous:
        module.completion(messages=[])
    with pytest.raises(ConnectionError) as asynchronous:
        asyncio.run(module.acompletion(messages=[]))

    for caught in (synchronous, asynchronous):
        _assert_marked(caught.value, "invalid_gateway_response")
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None
        assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
        assert "sk-private" not in "".join(
            traceback.format_exception(caught.value)
        )
    assert provider_calls == 0


def test_explicit_factories_normalize_incompatible_framework_versions(
    monkeypatch,
):
    from deepintshield.agentic.integrations import (
        google_adk,
        langchain,
        strands,
        temporal,
    )

    class RequiresArgument:
        def __init__(self, required):
            self.required = required

    class BasePlugin:
        def __init__(self, *, required):
            self.required = required

    class HookRegistry:
        pass

    class BeforeToolInvocationEvent:
        pass

    class ActivityInboundInterceptor:
        def __init__(self, next):
            self.next = next

    google = _module("google")
    google.__path__ = []
    adk = _module("google.adk")
    adk.__path__ = []
    plugins = _module("google.adk.plugins")
    plugins.__path__ = []
    temporalio = _module("temporalio")
    temporalio.__path__ = []
    langchain_core = _module("langchain_core")
    langchain_core.__path__ = []
    modules = {
        "google": google,
        "google.adk": adk,
        "google.adk.plugins": plugins,
        "google.adk.plugins.base_plugin": _module(
            "google.adk.plugins.base_plugin", BasePlugin=BasePlugin
        ),
        "strands": _module("strands"),
        "strands.hooks": _module(
            "strands.hooks",
            HookProvider=RequiresArgument,
            HookRegistry=HookRegistry,
        ),
        "strands.experimental": _module("strands.experimental"),
        "strands.experimental.hooks": _module(
            "strands.experimental.hooks",
            BeforeToolInvocationEvent=BeforeToolInvocationEvent,
        ),
        "temporalio": temporalio,
        "temporalio.exceptions": _module(
            "temporalio.exceptions", ApplicationError=Exception
        ),
        "temporalio.worker": _module(
            "temporalio.worker",
            ActivityInboundInterceptor=ActivityInboundInterceptor,
            Interceptor=RequiresArgument,
        ),
        "langchain_core": langchain_core,
        "langchain_core.callbacks": _module(
            "langchain_core.callbacks", BaseCallbackHandler=RequiresArgument
        ),
    }
    for name, module in modules.items():
        if name in {"strands", "strands.experimental"}:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(langchain, "_HANDLER_CLASS", None)

    for invoke in (
        lambda: google_adk.plugin(object()),
        lambda: strands.hook_provider(object()),
        lambda: temporal.interceptor(object()),
        lambda: langchain.make_handler(object()),
    ):
        with pytest.raises(RuntimeError) as caught:
            invoke()
        _assert_marked(caught.value, "framework_integration_unsupported")
        assert caught.value.__context__ is None
        assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("error", "expected_code"),
    (
        (
            GuardrailDenied(
                reason="secret policy explanation",
                decision_id="decision-secret",
                tool="wire_funds",
                code="agent_access_denied",
            ),
            "agent_access_denied",
        ),
        (
            GuardrailApprovalPending(
                decision_id="decision-secret",
                reason="secret approval explanation",
                code="agent_approval_pending",
            ),
            "agent_approval_pending",
        ),
    ),
)
def test_google_adk_plugin_short_circuit_exposes_safe_details(
    monkeypatch, error, expected_code
):
    from deepintshield.agentic.integrations import google_adk

    class BasePlugin:
        def __init__(self, name: str):
            self.name = name

    google = _module("google")
    google.__path__ = []
    adk = _module("google.adk")
    adk.__path__ = []
    plugins = _module("google.adk.plugins")
    plugins.__path__ = []
    base = _module("google.adk.plugins.base_plugin", BasePlugin=BasePlugin)
    for name, module in (
        ("google", google),
        ("google.adk", adk),
        ("google.adk.plugins", plugins),
        ("google.adk.plugins.base_plugin", base),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def blocked(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(google_adk, "resolve", blocked)
    callback = google_adk.plugin(object()).before_tool_callback

    result = asyncio.run(
        callback(
            tool=types.SimpleNamespace(name="wire_funds", func=lambda: None),
            tool_args={"amount": 10},
        )
    )

    assert result == _adk_block(expected_code)


def test_hermes_blocks_pdp_outage_and_rejects_missing_hook_boundary(monkeypatch):
    from deepintshield.agentic.integrations import hermes

    def unavailable(*_args, **_kwargs):
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(hermes, "resolve", unavailable)
    result = hermes._pre_tool_call(object(), "wire_funds", {"amount": 10})

    assert result == _hermes_directive("gateway_unavailable")

    with pytest.raises(RuntimeError) as caught:
        hermes.install(object(), object())
    _assert_marked(caught.value, "governance_configuration_error")


def test_hermes_configuration_translation_suppresses_native_hook_error():
    from deepintshield.agentic.integrations import hermes

    class BrokenContext:
        def register_hook(self, _name, _callback):
            raise RuntimeError("native-hook-secret")

    with pytest.raises(RuntimeError) as caught:
        hermes.install(BrokenContext(), object())

    _assert_marked(caught.value, "governance_configuration_error")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert "native-hook-secret" not in "".join(
        traceback.format_exception(caught.value)
    )


@pytest.mark.parametrize(
    ("error", "expected_code"),
    (
        (
            GuardrailDenied(
                reason="secret policy explanation",
                decision_id="decision-secret",
                tool="wire_funds",
                code="agent_access_denied",
            ),
            "agent_access_denied",
        ),
        (
            GuardrailApprovalPending(
                decision_id="decision-secret",
                reason="secret approval explanation",
                code="agent_approval_pending",
            ),
            "agent_approval_pending",
        ),
        (RuntimeError("secret infrastructure explanation"), "agentic_error"),
    ),
)
def test_hermes_block_directive_exposes_safe_details(
    monkeypatch, error, expected_code
):
    from deepintshield.agentic.integrations import hermes

    def blocked(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(hermes, "resolve", blocked)

    assert hermes._pre_tool_call(
        object(), "wire_funds", {"amount": 10}
    ) == _hermes_directive(expected_code)


def test_hermes_unsupported_mask_exposes_safe_details(monkeypatch):
    from deepintshield.agentic.integrations import hermes
    from deepintshield.agentic.types import Verdict

    monkeypatch.setattr(
        hermes,
        "resolve",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            verdict=Verdict.MASK,
            obligations=["mask:pii"],
        ),
    )

    assert hermes._pre_tool_call(
        object(), "wire_funds", {"account": "secret"}
    ) == _hermes_directive("mask_obligation_unsupported")


def test_strands_hook_propagates_pdp_outage(monkeypatch):
    from deepintshield.agentic.integrations import strands

    class HookProvider:
        pass

    class HookRegistry:
        def __init__(self):
            self.callback = None

        def add_callback(self, _event_type, callback):
            self.callback = callback

    class BeforeToolInvocationEvent:
        pass

    strands_pkg = _module("strands")
    strands_pkg.__path__ = []
    hooks = _module(
        "strands.hooks",
        HookProvider=HookProvider,
        HookRegistry=HookRegistry,
    )
    experimental = _module("strands.experimental")
    experimental.__path__ = []
    experimental_hooks = _module(
        "strands.experimental.hooks",
        BeforeToolInvocationEvent=BeforeToolInvocationEvent,
    )
    for name, module in (
        ("strands", strands_pkg),
        ("strands.hooks", hooks),
        ("strands.experimental", experimental),
        ("strands.experimental.hooks", experimental_hooks),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def unavailable(*_args, **_kwargs):
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(strands, "resolve", unavailable)
    provider = strands.hook_provider(object())
    registry = HookRegistry()
    provider.register_hooks(registry)
    event = types.SimpleNamespace(
        tool_use={"name": "wire_funds", "input": {"amount": 10}},
        selected_tool=types.SimpleNamespace(func=lambda: None),
    )

    with pytest.raises(ConnectionError) as caught:
        registry.callback(event)
    _assert_marked(caught.value, "gateway_unavailable")
    assert event.selected_tool is not None


@pytest.mark.parametrize(
    ("error", "expected_type", "expected_code"),
    (
        (
            GuardrailDenied(
                reason="secret policy explanation",
                decision_id="decision-secret",
                tool="wire_funds",
                code="agent_access_denied",
            ),
            PermissionError,
            "agent_access_denied",
        ),
        (
            GuardrailApprovalPending(
                decision_id="decision-secret",
                reason="secret approval explanation",
                code="agent_approval_pending",
            ),
            RuntimeError,
            "agent_approval_pending",
        ),
    ),
)
def test_strands_hook_uses_marked_builtin_error(
    monkeypatch, error, expected_type, expected_code
):
    from deepintshield.agentic.integrations import strands

    class HookProvider:
        pass

    class HookRegistry:
        def __init__(self):
            self.callback = None

        def add_callback(self, _event_type, callback):
            self.callback = callback

    class BeforeToolInvocationEvent:
        pass

    strands_pkg = _module("strands")
    strands_pkg.__path__ = []
    hooks = _module(
        "strands.hooks",
        HookProvider=HookProvider,
        HookRegistry=HookRegistry,
    )
    experimental = _module("strands.experimental")
    experimental.__path__ = []
    experimental_hooks = _module(
        "strands.experimental.hooks",
        BeforeToolInvocationEvent=BeforeToolInvocationEvent,
    )
    for name, module in (
        ("strands", strands_pkg),
        ("strands.hooks", hooks),
        ("strands.experimental", experimental),
        ("strands.experimental.hooks", experimental_hooks),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def blocked(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(strands, "resolve", blocked)
    provider = strands.hook_provider(object())
    registry = HookRegistry()
    provider.register_hooks(registry)
    event = types.SimpleNamespace(
        tool_use={"name": "wire_funds", "input": {"amount": 10}},
        selected_tool=types.SimpleNamespace(func=lambda: None),
    )

    with pytest.raises(expected_type) as caught:
        registry.callback(event)

    _assert_marked(caught.value, expected_code)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.__suppress_context__ is True
    assert event.selected_tool is None


def test_temporal_interceptor_propagates_pdp_outage_before_activity(monkeypatch):
    from deepintshield.agentic.integrations import temporal

    class ApplicationError(Exception):
        def __init__(self, message: str, **kwargs):
            super().__init__(message)
            self.kwargs = kwargs

    class ActivityInboundInterceptor:
        def __init__(self, next):
            self.next = next

        async def execute_activity(self, input):
            return await self.next.execute_activity(input)

    class Interceptor:
        pass

    temporalio = _module("temporalio")
    temporalio.__path__ = []
    exceptions = _module(
        "temporalio.exceptions",
        ApplicationError=ApplicationError,
    )
    worker = _module(
        "temporalio.worker",
        ActivityInboundInterceptor=ActivityInboundInterceptor,
        Interceptor=Interceptor,
    )
    for name, module in (
        ("temporalio", temporalio),
        ("temporalio.exceptions", exceptions),
        ("temporalio.worker", worker),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def unavailable(*_args, **_kwargs):
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(temporal, "resolve", unavailable)
    _, inbound_cls = temporal._make_interceptor_classes(lambda: object())
    executed = 0

    class Next:
        async def execute_activity(self, _input):
            nonlocal executed
            executed += 1
            return "ran"

    input_value = types.SimpleNamespace(fn=lambda: None, args=())
    inbound = inbound_cls(Next())

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(inbound.execute_activity(input_value))
    _assert_marked(caught.value, "gateway_unavailable")
    assert executed == 0


@pytest.mark.parametrize(
    ("error", "expected_code"),
    (
        (
            GuardrailDenied(
                reason="secret policy explanation",
                decision_id="decision-secret",
                tool="wire_funds",
                code="agent_access_denied",
            ),
            "agent_access_denied",
        ),
        (
            GuardrailApprovalPending(
                decision_id="decision-secret",
                reason="secret approval explanation",
                code="agent_approval_pending",
            ),
            "agent_approval_pending",
        ),
    ),
)
def test_temporal_application_error_exposes_safe_code_and_message(
    monkeypatch, error, expected_code
):
    from deepintshield.agentic import execution
    from deepintshield.agentic.integrations import temporal

    class ApplicationError(Exception):
        def __init__(self, message: str, **kwargs):
            super().__init__(message)
            self.kwargs = kwargs

    class ActivityInboundInterceptor:
        def __init__(self, next):
            self.next = next

        async def execute_activity(self, input):
            return await self.next.execute_activity(input)

    class Interceptor:
        pass

    temporalio = _module("temporalio")
    temporalio.__path__ = []
    exceptions = _module(
        "temporalio.exceptions",
        ApplicationError=ApplicationError,
    )
    worker = _module(
        "temporalio.worker",
        ActivityInboundInterceptor=ActivityInboundInterceptor,
        Interceptor=Interceptor,
    )
    for name, module in (
        ("temporalio", temporalio),
        ("temporalio.exceptions", exceptions),
        ("temporalio.worker", worker),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def blocked(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(temporal, "resolve", blocked)
    monkeypatch.setattr(execution, "_dispatch", lambda *_args: False)
    _, inbound_cls = temporal._make_interceptor_classes(lambda: object())
    executed = 0

    class Next:
        async def execute_activity(self, _input):
            nonlocal executed
            executed += 1
            return "ran"

    input_value = types.SimpleNamespace(fn=lambda: None, args=())
    inbound = inbound_cls(Next())

    with pytest.raises(ApplicationError) as caught:
        asyncio.run(inbound.execute_activity(input_value))

    details = public_agentic_error_details(expected_code)
    assert str(caught.value) == f"{expected_code}: {details['message']}"
    assert caught.value.kwargs == {
        "type": expected_code,
        "non_retryable": True,
    }
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.__suppress_context__ is True
    assert executed == 0


def test_strands_automatic_boundary_uses_final_hook_result_and_fails_closed(
    monkeypatch,
):
    from deepintshield.agentic import enforcement
    from deepintshield.agentic.integrations import strands

    final_event = types.SimpleNamespace(
        tool_use={"name": "wire_funds", "input": {"amount": 10}},
        selected_tool=types.SimpleNamespace(func=lambda: None),
        cancel_tool=False,
    )
    hook_result = [(final_event, [])]

    class ToolExecutor:
        @staticmethod
        async def _invoke_before_tool_call_hook(*_args):
            # Represents all user hooks having completed, including tool/arg
            # rewrites. The automatic guard must authorize this final value.
            return hook_result[0]

    strands_pkg = _module("strands")
    strands_pkg.__path__ = []
    tools = _module("strands.tools")
    tools.__path__ = []
    executors = _module("strands.tools.executors")
    executors.__path__ = []
    executor = _module(
        "strands.tools.executors._executor",
        ToolExecutor=ToolExecutor,
    )
    for name, module in (
        ("strands", strands_pkg),
        ("strands.tools", tools),
        ("strands.tools.executors", executors),
        ("strands.tools.executors._executor", executor),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    engine = object()
    agent = object()
    topology_reports = []
    boundary_order = []
    monkeypatch.setattr(enforcement, "resolve_engine", lambda _owner=None: engine)
    monkeypatch.setattr(enforcement, "bind_engine", lambda *_args, **_kwargs: True)

    def report_topology(*args, **kwargs):
        boundary_order.append("report")
        topology_reports.append((args, kwargs))
        return True

    monkeypatch.setattr(enforcement, "report_topology", report_topology)

    def allowed(*_args, **_kwargs):
        boundary_order.append("authorize")
        return types.SimpleNamespace(obligations=[])

    monkeypatch.setattr(strands, "resolve", allowed)
    assert strands.enforce() is True

    result = asyncio.run(
        ToolExecutor._invoke_before_tool_call_hook(
            agent,
            final_event.selected_tool,
            final_event.tool_use,
            {},
        )
    )
    assert result == (final_event, [])
    assert topology_reports == [((engine, agent), {"sync": True, "required": True})]
    assert boundary_order == ["report", "authorize"]

    def unavailable(*_args, **_kwargs):
        boundary_order.append("authorize")
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(strands, "resolve", unavailable)
    boundary_order.clear()

    with pytest.raises(ConnectionError) as caught:
        asyncio.run(
            ToolExecutor._invoke_before_tool_call_hook(
                agent,
                final_event.selected_tool,
                final_event.tool_use,
                {},
            )
        )
    _assert_marked(caught.value, "gateway_unavailable")
    assert boundary_order == ["report", "authorize"]

    class BrokenResult:
        def __iter__(self):
            raise RuntimeError("native-hook-result-secret")

    hook_result[0] = BrokenResult()
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            ToolExecutor._invoke_before_tool_call_hook(
                agent,
                final_event.selected_tool,
                final_event.tool_use,
                {},
            )
        )

    _assert_marked(caught.value, "governance_configuration_error")
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert "native-hook-result-secret" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_google_adk_automatic_final_dispatch_blocks_before_tool(monkeypatch):
    from deepintshield.agentic import enforcement
    from deepintshield.agentic.integrations import google_adk

    executed = 0

    async def call_tool(tool, args, tool_context):
        nonlocal executed
        executed += 1
        return await tool.run_async(args=args, tool_context=tool_context)

    google = _module("google")
    google.__path__ = []
    adk = _module("google.adk")
    adk.__path__ = []
    flows = _module("google.adk.flows")
    flows.__path__ = []
    llm_flows = _module("google.adk.flows.llm_flows")
    llm_flows.__path__ = []
    functions = _module("google.adk.flows.llm_flows.functions")
    setattr(functions, "__call_tool_async", call_tool)
    llm_flows.functions = functions
    for name, module in (
        ("google", google),
        ("google.adk", adk),
        ("google.adk.flows", flows),
        ("google.adk.flows.llm_flows", llm_flows),
        ("google.adk.flows.llm_flows.functions", functions),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    engine = object()
    topology_reports = []
    boundary_order = []
    monkeypatch.setattr(enforcement, "resolve_engine", lambda _owner=None: engine)
    monkeypatch.setattr(enforcement, "bind_engine", lambda *_args, **_kwargs: True)

    def report_topology(*args, **kwargs):
        boundary_order.append("report")
        topology_reports.append((args, kwargs))
        return True

    monkeypatch.setattr(enforcement, "report_topology", report_topology)

    def allowed(*_args, **_kwargs):
        boundary_order.append("authorize")
        return types.SimpleNamespace(obligations=[])

    monkeypatch.setattr(google_adk, "resolve", allowed)
    assert google_adk.enforce() is True

    async def run_async(**_kwargs):
        return "ran"

    tool = types.SimpleNamespace(
        name="wire_funds",
        func=lambda: None,
        run_async=run_async,
    )
    result = asyncio.run(
        functions.__call_tool_async(
            tool,
            {"amount": 10},
            object(),
        )
    )
    assert result == "ran"
    assert topology_reports == [((engine, tool), {"sync": True, "required": True})]
    assert boundary_order == ["report", "authorize"]
    assert executed == 1

    def unavailable(*_args, **_kwargs):
        boundary_order.append("authorize")
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(google_adk, "resolve", unavailable)
    boundary_order.clear()
    with pytest.raises(ConnectionError) as caught:
        asyncio.run(
            functions.__call_tool_async(
                tool,
                {"amount": 10},
                object(),
            )
        )
    _assert_marked(caught.value, "gateway_unavailable")
    assert boundary_order == ["authorize"]
    assert executed == 1


def test_temporal_automatic_worker_injects_one_interceptor(monkeypatch):
    from deepintshield.agentic import enforcement
    from deepintshield.agentic.integrations import temporal

    class ApplicationError(Exception):
        pass

    class ActivityInboundInterceptor:
        def __init__(self, next):
            self.next = next

        async def execute_activity(self, input):
            return await self.next.execute_activity(input)

    class Interceptor:
        pass

    class Worker:
        def __init__(self, client, *, task_queue, interceptors=()):
            self.client = client
            self.task_queue = task_queue
            self.interceptors = list(interceptors)

    temporalio = _module("temporalio")
    temporalio.__path__ = []
    exceptions = _module(
        "temporalio.exceptions",
        ApplicationError=ApplicationError,
    )
    worker = _module(
        "temporalio.worker",
        ActivityInboundInterceptor=ActivityInboundInterceptor,
        Interceptor=Interceptor,
        Worker=Worker,
    )
    for name, module in (
        ("temporalio", temporalio),
        ("temporalio.exceptions", exceptions),
        ("temporalio.worker", worker),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    engine = object()
    topology_reports = []
    boundary_order = []
    monkeypatch.setattr(enforcement, "resolve_engine", lambda _owner=None: engine)
    monkeypatch.setattr(enforcement, "bind_engine", lambda *_args, **_kwargs: True)

    def report_topology(*args, **kwargs):
        boundary_order.append("report")
        topology_reports.append((args, kwargs))
        return True

    monkeypatch.setattr(enforcement, "report_topology", report_topology)
    assert temporal.enforce() is True

    instance = Worker(object(), task_queue="agent-tools")
    assert len(instance.interceptors) == 1
    assert instance.interceptors[0]._deepintshield_interceptor is True
    assert topology_reports == [((engine, instance), {"sync": True, "required": True})]
    assert boundary_order == ["report"]


def test_hermes_automatic_dispatcher_converts_outage_to_block(monkeypatch):
    from deepintshield.agentic import enforcement
    from deepintshield.agentic.integrations import hermes

    executed = 0

    def handle_function_call(_name, _args):
        nonlocal executed
        executed += 1
        return '{"ok": true}'

    model_tools = _module(
        "model_tools",
        handle_function_call=handle_function_call,
    )
    monkeypatch.setitem(sys.modules, "model_tools", model_tools)
    engine = object()
    topology_reports = []
    boundary_order = []
    monkeypatch.setattr(enforcement, "resolve_engine", lambda _owner=None: engine)

    def report_topology(*args, **kwargs):
        boundary_order.append("report")
        topology_reports.append((args, kwargs))
        return True

    monkeypatch.setattr(enforcement, "report_topology", report_topology)

    def allowed(*_args, **_kwargs):
        boundary_order.append("authorize")
        return types.SimpleNamespace(
            verdict="allow",
            obligations=[],
        )

    monkeypatch.setattr(hermes, "resolve", allowed)
    assert hermes.enforce() is True

    assert json.loads(
        model_tools.handle_function_call("wire_funds", {"amount": 10})
    ) == {"ok": True}
    assert topology_reports == [((engine, model_tools), {"sync": True, "required": True})]
    assert boundary_order == ["report", "authorize"]
    assert executed == 1

    def unavailable(*_args, **_kwargs):
        boundary_order.append("authorize")
        raise GatewayUnavailable(reason="PDP down")

    monkeypatch.setattr(hermes, "resolve", unavailable)
    boundary_order.clear()
    result = model_tools.handle_function_call("wire_funds", {"amount": 10})

    assert json.loads(result) == _adk_block("gateway_unavailable")
    assert boundary_order == ["authorize"]
    assert executed == 1
