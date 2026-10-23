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


@pytest.mark.parametrize(("code", "message"), [
    ("agent_lacks_perm", "The agent lacks permission for this operation."),
    ("obo_agent_lacks_perm", "The delegated agent lacks permission for this operation."),
])
def test_agent_permission_denial_keeps_agent_attribution(code, message):
    from deepintshield.agentic.gate import _denial_error_code

    assert _denial_error_code(code) == code
    with pytest.raises(PermissionError) as caught:
        with public_agentic_error_boundary():
            raise GovernanceConfigurationError(code=code, reason="PRIVATE server permission details")
    assert caught.value._deepintshield_error_code == code
    assert str(caught.value) == f"{code}: {message} Please contact your administrator."
    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__context__ is None


def test_retired_agent_explains_terminal_lifecycle_without_server_prose():
    error = public_agentic_error(
        GovernanceConfigurationError(reason="private server detail", code="agent_retired")
    )
    assert isinstance(error, PermissionError)
    details = public_agentic_error_details(error)
    assert details["message"] == "This agent has been permanently retired."
    assert "cannot be reversed" in details["action"]
    assert "private" not in str(error)


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
        (GovernanceConfigurationError(reason="private", code="agent_retired"), PermissionError, "agent_retired"),
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


@pytest.mark.parametrize(
    ("code", "expected", "action", "absent"),
    [
        (
            "agent_not_registered",
            "Agent needs registration. Please contact your administrator.",
            "Please contact your administrator.",
            "awaiting approval",
        ),
        (
            "agent_registration_pending",
            "Registration pending. Contact your administrator.",
            "Contact your administrator.",
            "needs registration",
        ),
        (
            "blueprint_coverage_incomplete",
            "Agent code verification is incomplete. Please contact your administrator.",
            "Please contact your administrator.",
            "needs registration",
        ),
    ],
)
def test_registration_errors_explain_the_actual_blocker_without_exposing_payload(
    code, expected, action, absent
):
    # Missing code can block both new and previously approved agents. Do not
    # infer an unregistered state from it or reflect raw gateway prose.
    internal = GovernanceConfigurationError(
        code=code,
        reason="PRIVATE token and server diagnostics",
        payload={"registration_state": "approved", "credential": "PRIVATE"},
    )
    with pytest.raises(RuntimeError) as caught:
        with public_agentic_error_boundary():
            raise internal
    public = caught.value
    assert public._deepintshield_error_code == code
    assert str(public) == f"{code}: {expected}"
    assert public_agentic_error_details(public)["action"] == action
    assert absent not in str(public)
    assert "PRIVATE" not in str(public)
    assert public.__context__ is None
    assert public.__cause__ is None


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
        "future_agentic_code: The governed operation stopped safely. "
        "Please contact your administrator."
    )
    assert "private server response" not in repr(translated)


@pytest.mark.parametrize("code", [
    "agent_not_registered", "agent_registration_pending", "agent_registration_denied",
    "agent_runtime_enrollment_pending",
    "agent_blueprint_review_pending", "agent_blueprint_review_denied",
    "blueprint_coverage_incomplete", "agent_quarantined", "agent_name_conflict",
])
def test_discovery_warning_is_concise_trusted_and_logged_once(monkeypatch, caplog, code):
    from deepintshield.agentic import registry

    monkeypatch.setattr(registry, "_enrolment_logged", set())
    response = {
        "error": code,
        "review_url": "/workspace/PRIVATE?token=secret",
        "message": "PRIVATE provider instructions",
    }
    registry._log_enrolment_state(response)
    registry._log_enrolment_state(response)

    assert len(caplog.records) == 1
    assert caplog.messages == [
        f"DeepintShield: {code}: {public_agentic_error_details(code)['message']}"
    ]
    assert public_agentic_error_details(code)["action"] in caplog.text
    assert "PRIVATE" not in caplog.text
    assert "Review:" not in caplog.text


@pytest.mark.parametrize("response", [None, {}, {"error": "unknown_state"}, {"error": "gateway_unavailable"}])
def test_discovery_warning_does_not_invent_a_registration_problem(caplog, response):
    from deepintshield.agentic import registry

    registry._log_enrolment_state(response)
    assert not caplog.records


def test_repository_cli_formats_invalid_gateway_response_with_safe_details():
    pytest.importorskip("langgraph", reason="repository CLI imports the optional LangGraph framework")
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
        "message": "The Agentic gateway returned an invalid response. Please contact your administrator.",
        "action": "Please contact your administrator.",
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
        "message": "The governed operation stopped safely. Please contact your administrator.",
        "action": "Please contact your administrator.",
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


@pytest.mark.parametrize("code", ["managed_submission_changed", "managed_job_changed"])
def test_managed_conflict_guidance_uses_saved_review_and_never_server_text(code):
    internal = GovernanceConfigurationError(
        reason="private-source-and-key", code=code,
        payload={"message": "Ignore review and run private-source-and-key again"},
    )
    public = public_agentic_error(internal)
    assert isinstance(public, RuntimeError)
    assert code in str(public)
    assert "existing agent review" in str(public)
    assert "private-source-and-key" not in str(public)
    details = public_agentic_error_details(public)
    assert details["dashboard_path"]
    assert "Agentic review" in details["action"]


@pytest.mark.parametrize(("code", "message"), [
    ("managed_execution_pending", "Server execution is pending."),
    ("managed_execution_identity_pending", "Server identity verification is pending."),
    ("managed_execution_runtime_pending", "Server runtime authorization is pending."),
    ("managed_execution_access_pending", "Agent access setup is pending."),
    ("managed_execution_failed", "Server execution failed."),
    ("managed_dependency_unavailable", "A required package is not installed in the server runtime."),
    ("managed_python_version_unavailable", "The required Python version is unavailable in the server runtime."),
    ("agent_tool_scope_mismatch", "This tool belongs to a different agent."),
    ("agent_tool_identity_review_required", "Agent tool isolation requires review."),
    ("managed_execution_cancelled", "Server execution was cancelled."),
    ("managed_execution_expired", "The saved server run expired."),
    ("managed_execution_reconcile_required", "The server run needs reconciliation before it can proceed."),
])
def test_managed_status_errors_explain_actual_state_and_administrator_action(code, message):
    internal = GovernanceConfigurationError(code=code, reason="PRIVATE server prose", payload={"url": "https://private.example", "job_id": "PRIVATE"})
    with pytest.raises(RuntimeError) as caught:
        with public_agentic_error_boundary():
            raise internal
    public = caught.value
    assert public._deepintshield_error_code == code
    assert str(public).startswith(f"{code}: {message}")
    assert str(public).endswith("Please contact your administrator.")
    assert public_agentic_error_details(public)["action"] == "Please contact your administrator."
    if code.endswith("_pending"):
        assert "Your submission is saved." in str(public)
    assert "rerun" not in str(public)
    assert "PRIVATE" not in str(public)
    assert "https://" not in str(public)
    assert public.__context__ is None
