from __future__ import annotations

import builtins
import runpy
import sys
import traceback
import types
from pathlib import Path

import httpx
import pytest

from deepintshield.agentic.engine import AgenticEngine
from deepintshield.agentic.errors import (
    DeepIntShieldError,
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
    normalize_agentic_error_code,
    public_agentic_error,
    public_agentic_error_boundary,
    public_agentic_error_details,
)
from deepintshield.agentic.execution import _failure_metadata


def _assert_sdk_traceback_has_no_text(error: BaseException, secret: str) -> None:
    traceback_node = error.__traceback__
    while traceback_node is not None:
        frame = traceback_node.tb_frame
        module = str(frame.f_globals.get("__name__", ""))
        if module.startswith("deepintshield.agentic"):
            assert secret not in repr(tuple(frame.f_locals.values()))
        traceback_node = traceback_node.tb_next


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            GovernanceConfigurationError(
                framework="langgraph",
                reason="human configuration detail",
            ),
            "governance_configuration_error",
        ),
        (
            GuardrailApprovalPending(
                decision_id="decision-secret",
                reason="human approval detail",
            ),
            "require_approval",
        ),
        (
            GuardrailDenied(
                reason="human denial detail",
                decision_id="decision-secret",
                code="obo_tool_not_allowed",
            ),
            "obo_tool_not_allowed",
        ),
        (GatewayUnavailable(reason="private transport detail"), "gateway_unavailable"),
    ],
)
def test_agentic_errors_render_only_the_public_code(error, code):
    assert error.code == code
    assert error.message == code
    assert error.args == (code,)
    assert str(error) == code
    assert code in repr(error)
    assert "human" not in repr(error)
    assert "private" not in repr(error)
    assert "decision-secret" not in repr(error)


def test_untrusted_error_text_cannot_become_a_public_code():
    assert (
        normalize_agentic_error_code("Agent registration is pending", "fallback")
        == "fallback"
    )
    assert (
        normalize_agentic_error_code("agent-registration-pending", "fallback")
        == "agent_registration_pending"
    )
    assert (
        normalize_agentic_error_code("invalid prose", "unsafe fallback!")
        == "agentic_error"
    )
    assert normalize_agentic_error_code(True, "fallback") == "fallback"
    assert normalize_agentic_error_code(object(), "") == ""
    assert normalize_agentic_error_code("invalid prose", object()) == "agentic_error"


def test_gateway_error_code_parser_ignores_arbitrary_nested_codes():
    assert AgenticEngine._response_error_code(
        httpx.Response(
            409,
            json={
                "details": {
                    "provider": {
                        "code": "private_third_party_identifier",
                    }
                }
            },
        )
    ) == ""
    assert AgenticEngine._response_error_code(
        httpx.Response(
            409,
            json={
                "error": {"code": "agent_registration_pending"},
                "details": {"code": "private_third_party_identifier"},
            },
        )
    ) == "agent_registration_pending"


def test_native_framework_translation_retains_code_for_run_telemetry():
    error = PermissionError("require_approval")
    error._deepintshield_error_code = "require_approval"

    assert _failure_metadata(error) == (
        "awaiting_approval",
        "require_approval",
        "",
    )


@pytest.mark.parametrize(
    ("internal", "public_type", "code"),
    [
        (GuardrailDenied(reason="private", code="obo_tool_not_allowed"), PermissionError, "obo_tool_not_allowed"),
        (DeepIntShieldError(detail="private", code="guardrail_denied"), PermissionError, "guardrail_denied"),
        (GuardrailApprovalPending(reason="private"), RuntimeError, "require_approval"),
        (GuardrailApprovalPending(reason="private", code="approval_required"), RuntimeError, "approval_required"),
        (GovernanceConfigurationError(reason="private", code="approval_access_denied"), PermissionError, "approval_access_denied"),
        (GatewayUnavailable(reason="private", code="approval_store_unavailable"), ConnectionError, "approval_store_unavailable"),
        (GovernanceConfigurationError(reason="private", code="agent_registration_pending"), RuntimeError, "agent_registration_pending"),
        (GatewayUnavailable(reason="private"), ConnectionError, "gateway_unavailable"),
        (GovernanceConfigurationError(reason="private", code="invalid_gateway_response"), ConnectionError, "invalid_gateway_response"),
    ],
)
def test_public_boundary_uses_marked_builtins_with_only_trusted_details(
    internal, public_type, code
):
    translated = public_agentic_error(internal)

    assert type(translated) is public_type
    assert getattr(translated, "_deepintshield_error_code") == code
    assert str(translated).startswith(f"{code}: ")
    rendered = repr(translated)
    assert "private" not in rendered
    assert "decision" not in rendered.lower() or code == "obo_tool_not_allowed"
    assert public_agentic_error(translated) is translated
    details = public_agentic_error_details(translated)
    assert details["code"] == code
    assert details["message"]
    assert details["action"]
    assert details["dashboard_path"].startswith("/workspace/")


def test_public_boundary_does_not_swallow_a_real_application_runtime_error():
    application_error = RuntimeError("application-bug")

    with pytest.raises(RuntimeError) as caught:
        with public_agentic_error_boundary():
            raise application_error

    assert caught.value is application_error
    assert not hasattr(caught.value, "_deepintshield_error_code")


def test_public_boundary_severs_private_internal_exception_state():
    internal = GovernanceConfigurationError(
        reason="TOP-SECRET gateway reason",
        code="invalid_gateway_response",
        payload={"credential": "sk-private", "response": "private-body"},
    )

    with pytest.raises(ConnectionError) as caught:
        with public_agentic_error_boundary():
            raise internal

    public = caught.value
    assert public.__context__ is None
    assert public.__cause__ is None
    assert public.__suppress_context__ is True
    rendered = "".join(traceback.format_exception(caught.type, public, caught.tb))
    assert "TOP-SECRET" not in rendered
    assert "sk-private" not in rendered
    assert "private-body" not in rendered
    _assert_sdk_traceback_has_no_text(public, "TOP-SECRET")
    _assert_sdk_traceback_has_no_text(public, "sk-private")
    _assert_sdk_traceback_has_no_text(public, "private-body")


def test_unknown_public_code_uses_safe_generic_guidance():
    translated = public_agentic_error(
        GovernanceConfigurationError(
            reason="private server response",
            code="future_agentic_code",
        )
    )

    assert getattr(translated, "_deepintshield_error_code") == "future_agentic_code"
    assert str(translated) == (
        "future_agentic_code: The governed operation stopped safely."
    )
    assert "private server response" not in repr(translated)


def test_repository_cli_formats_invalid_gateway_response_with_safe_details():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "test_agentic_new.py")
    )
    payload = namespace["_safe_error_payload"](
        public_agentic_error(
            DeepIntShieldError(
                detail="raw response containing a secret",
                code="invalid_gateway_response",
            )
        )
    )

    assert payload == {
        "type": "ConnectionError",
        "code": "invalid_gateway_response",
        "message": "The Agentic gateway returned an invalid response.",
        "action": (
            "Check gateway health and retry; the operation was blocked safely."
        ),
        "dashboard_path": "/workspace/agentic-new/overview",
    }
    assert "secret" not in repr(payload)
    assert namespace["_safe_error_payload"](RuntimeError("application-bug")) is None

    forged = RuntimeError("raw application text")
    forged._deepintshield_error_code = "future_agentic_code"
    forged._deepintshield_error_message = "PRIVATE forged message"
    forged._deepintshield_error_action = "PRIVATE forged action"
    assert namespace["_safe_error_payload"](forged) == {
        "type": "RuntimeError",
        "code": "future_agentic_code",
        "message": "The governed operation stopped safely.",
        "action": "Review the recorded Agentic decision before retrying.",
        "dashboard_path": "/workspace/agentic-new/activity?tab=decisions",
    }


@pytest.mark.parametrize(
    ("module_name", "invoke", "framework"),
    [
        (
            "langgraph",
            lambda: __import__(
                "deepintshield.agentic.integrations.langgraph",
                fromlist=["shield_graph"],
            ).shield_graph(object(), engine=object()),
            "langgraph",
        ),
        (
            "langchain_core",
            lambda: __import__(
                "deepintshield.agentic.integrations.langchain",
                fromlist=["make_handler"],
            ).make_handler(object()),
            "langchain",
        ),
    ],
)
def test_missing_framework_dependency_uses_builtin_safe_contract(
    monkeypatch,
    module_name,
    invoke,
    framework,
):
    """Optional framework failures stay machine-readable at public surfaces."""
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name or name.startswith(f"{module_name}."):
            raise ImportError("private dependency loader detail")
        return real_import(name, globals, locals, fromlist, level)

    if module_name == "langchain_core":
        from deepintshield.agentic.integrations import langchain

        monkeypatch.setattr(langchain, "_HANDLER_CLASS", None)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError) as caught:
        invoke()

    rendered = "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert getattr(caught.value, "_deepintshield_error_code") == "framework_dependency_missing"
    assert str(caught.value) == (
        "framework_dependency_missing: "
        "The selected agent framework dependency is missing."
    )
    assert getattr(caught.value, "_deepintshield_error_action")
    assert caught.value.__suppress_context__ is True
    assert "private dependency loader detail" not in rendered


@pytest.mark.parametrize("entrypoint", ["enforce", "enforce_compile"])
def test_langgraph_direct_enforcement_patch_failure_uses_builtin_boundary(
    monkeypatch,
    entrypoint,
):
    """The exported installer is an application boundary, not only its guards."""
    from deepintshield.agentic.integrations import langgraph as integration

    class FrozenCompiledType(type):
        def __setattr__(cls, name, value):
            if name == "invoke":
                raise TypeError("PRIVATE frozen framework detail")
            return super().__setattr__(name, value)

    class CompiledStateGraph(metaclass=FrozenCompiledType):
        def invoke(self, value):
            return value

    class StateGraph:
        def compile(self):
            return CompiledStateGraph()

    langgraph_package = types.ModuleType("langgraph")
    langgraph_package.__path__ = []
    graph_package = types.ModuleType("langgraph.graph")
    graph_package.__path__ = []
    state_module = types.ModuleType("langgraph.graph.state")
    state_module.StateGraph = StateGraph
    state_module.CompiledStateGraph = CompiledStateGraph
    monkeypatch.setitem(sys.modules, "langgraph", langgraph_package)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_package)
    monkeypatch.setitem(sys.modules, "langgraph.graph.state", state_module)

    with pytest.raises(RuntimeError) as caught:
        getattr(integration, entrypoint)()

    assert type(caught.value) is RuntimeError
    assert getattr(caught.value, "_deepintshield_error_code") == (
        "governance_configuration_error"
    )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    assert "PRIVATE frozen framework detail" not in "".join(
        traceback.format_exception(caught.value)
    )


@pytest.mark.parametrize(
    ("integration_name", "factory_name", "blocked_module"),
    [
        ("strands", "hook_provider", "strands.hooks"),
        ("temporal", "interceptor", "temporalio.exceptions"),
        ("google_adk", "plugin", "google.adk.plugins.base_plugin"),
    ],
)
def test_optional_adapter_factories_normalize_missing_dependencies(
    monkeypatch,
    integration_name,
    factory_name,
    blocked_module,
):
    integration = __import__(
        f"deepintshield.agentic.integrations.{integration_name}",
        fromlist=[factory_name],
    )
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == blocked_module or name.startswith(f"{blocked_module}."):
            raise ImportError("PRIVATE dependency resolution failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError) as caught:
        getattr(integration, factory_name)(object())

    assert getattr(caught.value, "_deepintshield_error_code") == (
        "framework_dependency_missing"
    )
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "PRIVATE dependency resolution failure" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_set_default_client_uses_builtin_setup_error_and_preserves_app_bug():
    from deepintshield.agentic import set_default_client

    class NonWeakReferenceable:
        __slots__ = ("agentic",)

        def __init__(self):
            self.agentic = object()

    with pytest.raises(RuntimeError) as caught:
        set_default_client(NonWeakReferenceable())

    assert getattr(caught.value, "_deepintshield_error_code") == (
        "governance_configuration_error"
    )
    assert caught.value.__context__ is None

    application_error = RuntimeError("real application property bug")

    class BrokenProperty:
        @property
        def agentic(self):
            raise application_error

    with pytest.raises(RuntimeError) as unmarked:
        set_default_client(BrokenProperty())
    assert unmarked.value is application_error
    assert not hasattr(unmarked.value, "_deepintshield_error_code")


def test_openai_agents_frozen_tool_uses_builtin_setup_error():
    from deepintshield.agentic.integrations.openai_agents import shield_agent

    class FrozenTool:
        __slots__ = ()
        name = "frozen_tool"

        def on_invoke_tool(self, *_args, **_kwargs):
            return "raw"

    with pytest.raises(RuntimeError) as caught:
        shield_agent(FrozenTool(), engine=object())

    assert getattr(caught.value, "_deepintshield_error_code") == (
        "governance_configuration_error"
    )
    assert caught.value.__context__ is None
    assert "read-only" not in "".join(traceback.format_exception(caught.value))
