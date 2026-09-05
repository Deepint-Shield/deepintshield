from __future__ import annotations

import builtins
import json

import httpx
import pytest

from deepintshield import (
    DeepintShield,
    DeepintShieldBlockedError,
    DeepintShieldError,
    ShieldConfig,
    get_exception_error_code,
)


def test_client_configuration_closed_and_guardrail_errors_have_codes(shield_factory):
    missing_key = DeepintShield()
    with pytest.raises(DeepintShieldError) as missing:
        missing_key.virtual_key_or_raise()
    assert missing.value.code == "virtual_key_missing"
    missing_key.close()

    shield = shield_factory(
        lambda _request: httpx.Response(
            200,
            json={"result": {"decision": "block", "reason": "policy"}},
        )
    )
    with pytest.raises(DeepintShieldBlockedError) as blocked:
        shield.guard(stage="input", input="blocked")
    assert blocked.value.code == "guardrail_blocked"
    assert blocked.value.details == {"stage": "input", "decision": "block"}

    shield.close()
    with pytest.raises(DeepintShieldError) as closed:
        shield.request("GET", "/health")
    assert closed.value.code == "client_closed"


def test_generic_request_preserves_successful_raw_and_array_responses(shield_factory):
    plain = shield_factory(lambda _request: httpx.Response(200, text="ok"))
    array = shield_factory(lambda _request: httpx.Response(200, json=[1, 2, 3]))

    assert plain.request("GET", "/plain") == {"raw": "ok"}
    assert array.request("GET", "/array") == [1, 2, 3]


def test_typed_chat_rejects_a_non_object_success_response(shield_factory):
    shield = shield_factory(lambda _request: httpx.Response(200, json=["bad-shape"]))

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[])

    assert caught.value.code == "invalid_response"


def test_typed_non_json_error_does_not_retain_decoder_context(shield_factory):
    shield = shield_factory(
        lambda _request: httpx.Response(200, text="private-upstream-body")
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[])

    assert caught.value.code == "invalid_response"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "private-upstream-body" not in repr(caught.value.to_dict())


def test_typed_deeply_nested_json_is_a_coded_invalid_response(shield_factory):
    private_body = b"[" * 2_000 + b"0" + b"]" * 2_000
    shield = shield_factory(
        lambda _request: httpx.Response(
            200,
            content=private_body,
            headers={"content-type": "application/json"},
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[])

    assert caught.value.code == "invalid_response"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert private_body.decode() not in repr(caught.value.to_dict())


def test_http_status_and_feature_fallback_codes(shield_factory):
    server = shield_factory(lambda _request: httpx.Response(500, json={"message": "boom"}))
    bad_chat = shield_factory(lambda _request: httpx.Response(400, json={"message": "bad"}))

    with pytest.raises(DeepintShieldError) as server_error:
        server.request("GET", "/failure")
    with pytest.raises(DeepintShieldError) as chat_error:
        bad_chat.chat(model="model", messages=[])

    assert server_error.value.code == "server_error"
    assert server_error.value.retryable is True
    assert chat_error.value.code == "chat_request_failed"


def test_transport_error_keeps_native_httpx_type_and_gains_code(shield_factory):
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    shield = shield_factory(timeout)
    with pytest.raises(httpx.ReadTimeout) as caught:
        shield.request("GET", "/slow")

    assert get_exception_error_code(caught.value) == "transport_timeout"
    assert caught.value.retryable is True
    assert caught.value.details == {"method": "GET", "path": "/slow"}


def test_validation_sites_keep_builtin_types_and_gain_codes(shield_factory, allow_response):
    shield = shield_factory(lambda _request: httpx.Response(200, json=allow_response))

    with pytest.raises(ValueError) as invocation:
        shield.agent.evaluate_tool()
    with pytest.raises(TypeError) as retriever:
        shield.rag.guard_retriever(object())
    with pytest.raises(TypeError) as embedder:
        shield.rag.guard_embedder(object())
    with pytest.raises(ValueError) as framework:
        shield.bind("unknown-framework")

    assert get_exception_error_code(invocation.value) == "agent_invocation_invalid"
    assert get_exception_error_code(retriever.value) == "rag_retriever_unsupported"
    assert get_exception_error_code(embedder.value) == "rag_embedder_unsupported"
    assert get_exception_error_code(framework.value) == "framework_binder_not_found"


def test_agent_invalid_mapping_and_circular_input_keep_coded_value_error(
    shield_factory,
):
    shield = shield_factory(
        lambda _request: pytest.fail("invalid invocation must not reach transport")
    )
    circular: dict[str, object] = {}
    circular["self"] = circular

    with pytest.raises(ValueError) as malformed:
        shield.agent.evaluate_tool({"unexpected": "field"})
    with pytest.raises(ValueError) as circular_input:
        shield.agent.evaluate_tool(name="tool", args=circular)

    assert get_exception_error_code(malformed.value) == "agent_invocation_invalid"
    assert malformed.value.__context__ is None
    assert get_exception_error_code(circular_input.value) == "agent_invocation_invalid"
    assert circular_input.value.__context__ is None


def test_invalid_timeout_environment_keeps_value_error_and_gains_code(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_TIMEOUT", "not-a-number")

    with pytest.raises(ValueError) as caught:
        ShieldConfig.from_env()

    assert get_exception_error_code(caught.value) == "configuration_error"
    assert caught.value.details == {
        "environment_variable": "DEEPINTSHIELD_TIMEOUT"
    }


def test_mcp_validation_execution_and_discovery_errors_have_codes(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/mcp/tool/execute"):
            return httpx.Response(400, json={"message": "bad tool"})
        return httpx.Response(400, json={"message": "no discovery"})

    shield = shield_factory(handler)

    with pytest.raises(DeepintShieldError) as name:
        shield.mcp.call_qualified("unqualified", {})
    with pytest.raises(DeepintShieldError) as arguments:
        shield.mcp.call_qualified("server-tool", "private-not-json")
    with pytest.raises(DeepintShieldError) as execution:
        shield.mcp.call(server="server", tool="tool")
    with pytest.raises(DeepintShieldError) as discovery:
        shield.mcp.list_tools()

    assert name.value.code == "mcp_tool_name_invalid"
    assert arguments.value.code == "mcp_arguments_invalid"
    assert arguments.value.__context__ is None
    assert arguments.value.__cause__ is None
    assert execution.value.code == "mcp_execution_failed"
    assert discovery.value.code == "mcp_discovery_failed"


def test_mcp_decoder_limits_and_encoder_failures_are_coded_without_context(
    shield_factory,
):
    shield = shield_factory(
        lambda _request: pytest.fail("invalid arguments must not reach transport")
    )
    private_integer = "9" * 5_000
    circular: dict[str, object] = {}
    circular["self"] = circular

    with pytest.raises(DeepintShieldError) as decoded:
        shield.mcp.call_qualified(
            "server-tool",
            f'{{"private":{private_integer}}}',
        )
    with pytest.raises(DeepintShieldError) as encoded:
        shield.mcp.call(
            server="server",
            tool="tool",
            arguments=circular,
        )
    with pytest.raises(DeepintShieldError) as nonstandard_decoded:
        shield.mcp.call_qualified("server-tool", '{"value":NaN}')
    with pytest.raises(DeepintShieldError) as nonstandard_encoded:
        shield.mcp.call(
            server="server",
            tool="tool",
            arguments={"value": float("nan")},
        )

    assert decoded.value.code == "mcp_arguments_invalid"
    assert decoded.value.__context__ is None
    assert decoded.value.__cause__ is None
    assert private_integer not in repr(decoded.value.to_dict())
    assert encoded.value.code == "mcp_arguments_invalid"
    assert encoded.value.__context__ is None
    assert encoded.value.__cause__ is None
    assert nonstandard_decoded.value.code == "mcp_arguments_invalid"
    assert nonstandard_encoded.value.code == "mcp_arguments_invalid"


def test_mcp_success_paths_use_central_transport_and_workspace_vk(shield_factory):
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("x-deepintshield-vk")))
        if request.url.path.endswith("/v1/mcp/tool/execute"):
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "done"}]},
            )
        return httpx.Response(
            200,
            json={
                "clients": [
                    {
                        "config": {"name": "server"},
                        "tools": [
                            {
                                "name": "server-tool",
                                "description": "A tool",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                ]
            },
        )

    shield = shield_factory(handler)
    result = shield.mcp.call(server="server", tool="tool", value=1)
    tools = shield.mcp.list_tools()

    assert result.text == "done"
    assert [(tool.server, tool.name) for tool in tools] == [("server", "tool")]
    assert seen == [
        ("/v1/mcp/tool/execute", "sk-ds-test"),
        ("/api/mcp/clients", "sk-ds-test"),
    ]


@pytest.mark.parametrize(
    ("status_code", "gateway_code", "expected_retryable"),
    [
        (403, "mcp_tool_authorization_denied", False),
        (503, "mcp_tool_authorization_unavailable", True),
    ],
)
def test_mcp_preserves_canonical_gaf_error_codes(
    shield_factory,
    status_code,
    gateway_code,
    expected_retryable,
):
    shield = shield_factory(
        lambda _request: httpx.Response(
            status_code,
            json={"error": {"code": gateway_code, "message": "safe failure"}},
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.mcp.call(server="server", tool="tool")

    assert caught.value.code == gateway_code
    assert caught.value.retryable is expected_retryable


def test_mcp_approval_202_is_not_returned_as_a_successful_tool_result(
    shield_factory,
):
    shield = shield_factory(
        lambda _request: httpx.Response(
            202,
            json={
                "error": {
                    "code": "mcp_tool_approval_required",
                    "message": "MCP tool execution requires approval",
                    "param": {
                        "verdict": "REQUIRE_APPROVAL",
                        "decision_id": "decision-1",
                        "approval_id": "approval-1",
                    },
                }
            },
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.mcp.call(server="server", tool="tool")

    assert caught.value.code == "mcp_tool_approval_required"
    assert caught.value.status_code == 202
    assert caught.value.payload["error"]["param"]["approval_id"] == "approval-1"
    assert caught.value.details == {
        "method": "POST",
        "path": "/v1/mcp/tool/execute",
        "status_code": 202,
        "response_code": "mcp_tool_approval_required",
    }


def test_mcp_forwards_delegated_subject_and_workload_proof_only_as_headers(
    shield_factory,
):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["subject"] = request.headers.get("x-mcp-subject-token")
        captured["agent"] = request.headers.get("x-agent-token")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": "done"})

    shield = shield_factory(handler)
    result = shield.mcp.call(
        server="server",
        tool="tool",
        arguments={"value": 1},
        extra_headers={
            "X-MCP-Subject-Token": "caller-subject-token",
            "X-Agent-Token": "signed-workload-proof",
        },
    )

    assert result.text == "done"
    assert captured["subject"] == "caller-subject-token"
    assert captured["agent"] == "signed-workload-proof"
    assert captured["body"]["function"]["arguments"] == '{"value": 1}'
    assert "caller-subject-token" not in repr(captured["body"])


@pytest.mark.parametrize("adapter", ["openai", "anthropic"])
@pytest.mark.parametrize(
    ("status_code", "gateway_code"),
    [
        (403, "mcp_tool_authorization_denied"),
        (503, "mcp_tool_authorization_unavailable"),
        (202, "mcp_tool_approval_required"),
    ],
)
def test_mcp_adapters_do_not_turn_canonical_outcomes_into_model_content(
    shield_factory,
    adapter,
    status_code,
    gateway_code,
):
    shield = shield_factory(
        lambda _request: httpx.Response(
            status_code,
            json={
                "error": {
                    "code": gateway_code,
                    "message": "MCP tool execution stopped safely",
                }
            },
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        if adapter == "openai":
            shield.mcp.run_openai_tool_calls(
                [
                    {
                        "id": "call-1",
                        "function": {"name": "server-tool", "arguments": "{}"},
                    }
                ]
            )
        else:
            shield.mcp.run_anthropic_tool_uses(
                [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "server-tool",
                        "input": {},
                    }
                ]
            )

    assert caught.value.code == gateway_code


@pytest.mark.parametrize("adapter", ["openai", "anthropic"])
def test_mcp_adapters_keep_ordinary_execution_error_compatibility(
    shield_factory,
    adapter,
):
    shield = shield_factory(
        lambda _request: httpx.Response(500, json={"message": "ordinary failure"})
    )

    if adapter == "openai":
        messages = shield.mcp.run_openai_tool_calls(
            [
                {
                    "id": "call-1",
                    "function": {"name": "server-tool", "arguments": "{}"},
                }
            ]
        )
        assert messages[0]["content"].startswith("[MCP execution error]")
    else:
        results = shield.mcp.run_anthropic_tool_uses(
            [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "server-tool",
                    "input": {},
                }
            ]
        )
        assert results[0]["is_error"] is True
        assert results[0]["content"][0]["text"].startswith(
            "[MCP execution error]"
        )


@pytest.mark.parametrize("adapter", ["openai", "anthropic"])
def test_mcp_adapters_forward_request_scoped_delegated_credentials(
    shield_factory,
    adapter,
):
    seen_subject_tokens: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_subject_tokens.append(request.headers.get("x-mcp-subject-token"))
        return httpx.Response(200, json={"content": "done"})

    shield = shield_factory(handler)
    headers = {"X-MCP-Subject-Token": "caller-subject-token"}
    if adapter == "openai":
        shield.mcp.run_openai_tool_calls(
            [
                {
                    "id": "call-1",
                    "function": {"name": "server-tool", "arguments": "{}"},
                }
            ],
            extra_headers=headers,
        )
    else:
        shield.mcp.run_anthropic_tool_uses(
            [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "server-tool",
                    "input": {},
                }
            ],
            extra_headers=headers,
        )

    assert seen_subject_tokens == ["caller-subject-token"]


def test_provider_missing_dependency_keeps_import_error_and_gains_code(
    monkeypatch,
):
    shield = DeepintShield(virtual_key="vk")
    real_import = builtins.__import__

    def missing_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("forced missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_openai)
    try:
        with pytest.raises(ImportError) as caught:
            shield.openai()
    finally:
        shield.close()

    assert get_exception_error_code(caught.value) == "provider_dependency_missing"
    assert caught.value.details == {"component": "openai"}


def test_framework_missing_dependency_and_binder_attribute_keep_builtin_types(
    monkeypatch,
):
    shield = DeepintShield(virtual_key="vk")
    binder = shield.autogen()

    with pytest.raises(AttributeError) as missing_binder:
        binder.not_a_binder

    real_import = builtins.__import__

    def missing_autogen(name, *args, **kwargs):
        if name.startswith("autogen_ext"):
            raise ImportError("forced missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_autogen)
    try:
        with pytest.raises(ImportError) as missing_dependency:
            binder.model_client()
    finally:
        shield.close()

    assert get_exception_error_code(missing_binder.value) == (
        "framework_binder_attribute_missing"
    )
    assert get_exception_error_code(missing_dependency.value) == (
        "framework_dependency_missing"
    )
    assert missing_dependency.value.details == {"component": "autogen"}


def test_mcp_call_sends_the_clients_agent_subject(shield_factory):
    """The brokered MCP path carries the same agent selector as decide, so a
    key bound to several active agents resolves to this client's agent."""
    from deepintshield.agentic.registry import _registry_key

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["subject"] = request.headers.get("x-agent-subject")
        return httpx.Response(200, json={"content": "done"})

    shield = shield_factory(handler, agent_name="GISEC Demo Agent")
    assert shield.mcp.call(server="server", tool="tool").text == "done"
    assert captured["subject"] == f"agent:{_registry_key('GISEC Demo Agent')}"

    # An explicit selector from the caller is never overridden.
    shield.mcp.call(server="server", tool="tool", extra_headers={"X-Agent-Subject": "agent:other"})
    assert captured["subject"] == "agent:other"


def test_mcp_call_without_an_agent_name_adds_no_identity(shield_factory):
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["subject"] = request.headers.get("x-agent-subject")
        captured["token"] = request.headers.get("x-agent-token")
        return httpx.Response(200, json={"content": "done"})

    shield = shield_factory(handler, agent_name="")
    shield.mcp.call(server="server", tool="tool")
    assert captured["subject"] is None
    assert captured["token"] is None
