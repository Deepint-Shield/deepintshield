"""Exercise actual official ClientSession methods through their nested request path."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")
from mcp import types
from mcp.shared.exceptions import McpError
from mcp.shared.session import BaseSession

from deepintshield.errors import DeepintShieldError, ErrorCode


@pytest.fixture
def session():
    from deepintshield.mcp._native import _DeepIntShieldClientSession

    # The transport is never opened: only BaseSession's wire exchange is mocked.
    # ClientSession's request construction/result handling remains real.
    return _DeepIntShieldClientSession(None, None)


def _rpc_error():
    return McpError(
        types.ErrorData(
            code=types.INVALID_PARAMS,
            message="Unknown tool: private-name; token sk-secret",
            data={"access_token": "must-not-escape"},
        )
    )


@pytest.mark.asyncio
async def test_empty_discovery_is_valid_and_unknown_tool_retains_numeric_error(
    session, monkeypatch
):
    methods = []

    async def exchange(self, request, result_type, **kwargs):
        methods.append(request.root.method)
        if request.root.method == "tools/list":
            return result_type.model_validate({"tools": []})
        raise _rpc_error()

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    assert (await session.list_tools()).tools == []
    with pytest.raises(DeepintShieldError) as caught:
        await session.call_tool("DeepWiki-ask_question", {})
    assert methods == ["tools/list", "tools/call"]
    assert caught.value.code == ErrorCode.MCP_EXECUTION_FAILED.value
    assert caught.value.details["operation"] == "call_tool"
    assert caught.value.details["mcp_code"] == types.INVALID_PARAMS
    assert caught.value.details["upstream_type"] == "McpError"
    assert "sk-secret" not in repr(caught.value.to_dict())
    assert "must-not-escape" not in repr(caught.value.to_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected_code", "expected_operation"),
    [
        ("list_tools", ErrorCode.MCP_DISCOVERY_FAILED, "list_tools"),
        ("call_tool", ErrorCode.MCP_EXECUTION_FAILED, "call_tool"),
        ("initialize", ErrorCode.MCP_CONNECTION_FAILED, "connect"),
        ("send_ping", ErrorCode.MCP_PROTOCOL_ERROR, "send_request"),
    ],
)
async def test_real_high_level_methods_use_their_public_error_boundary(
    session, monkeypatch, method, expected_code, expected_operation
):
    async def exchange(self, *args, **kwargs):
        raise _rpc_error()

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    args = ("missing-tool", {}) if method == "call_tool" else ()
    with pytest.raises(DeepintShieldError) as caught:
        await getattr(session, method)(*args)
    assert caught.value.code == expected_code.value
    assert caught.value.details["operation"] == expected_operation
    assert caught.value.details["mcp_code"] == types.INVALID_PARAMS


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["list_tools", "call_tool", "initialize"])
async def test_malformed_remote_result_stays_protocol_error(
    session, monkeypatch, method
):
    async def exchange(self, request, result_type, **kwargs):
        return result_type.model_validate(
            {"tools": "invalid", "content": [{"type": "text"}]}
        )

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    args = ("some-tool", {}) if method == "call_tool" else ()
    with pytest.raises(DeepintShieldError) as caught:
        await getattr(session, method)(*args)
    assert caught.value.code == ErrorCode.MCP_PROTOCOL_ERROR.value
    assert caught.value.details["upstream_type"] == "ValidationError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED,
        ErrorCode.MCP_TOOL_AUTHORIZATION_UNAVAILABLE,
        ErrorCode.MCP_TOOL_APPROVAL_REQUIRED,
    ],
)
async def test_native_call_preserves_canonical_policy_result(session, monkeypatch, code):
    async def exchange(self, request, result_type, **kwargs):
        return result_type.model_validate(
            {
                "isError": True,
                "content": [],
                "_meta": {
                    "code": code.value,
                    "decision_id": "decision-safe",
                    "approval_id": "approval-safe",
                    "access_token": "must-not-escape",
                },
            }
        )

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    with pytest.raises(DeepintShieldError) as caught:
        await session.call_tool("governed-tool", {})
    assert caught.value.code == code.value
    assert caught.value.details["decision_id"] == "decision-safe"
    assert caught.value.details["approval_id"] == "approval-safe"
    assert "must-not-escape" not in repr(caught.value.to_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED,
        ErrorCode.MCP_TOOL_AUTHORIZATION_UNAVAILABLE,
        ErrorCode.MCP_TOOL_APPROVAL_REQUIRED,
    ],
)
async def test_rpc_numeric_envelope_does_not_shadow_canonical_policy_code(
    session, monkeypatch, code
):
    async def exchange(self, request, result_type, **kwargs):
        raise McpError(
            types.ErrorData(
                code=-32000,
                message="private upstream prose",
                data={
                    "code": code.value,
                    "decision_id": "decision-safe",
                    "approval_id": "approval-safe",
                    "access_token": "must-not-escape",
                },
            )
        )

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    with pytest.raises(DeepintShieldError) as caught:
        await session.call_tool("governed-tool", {})
    assert caught.value.code == code.value
    assert caught.value.details["mcp_code"] == -32000
    assert caught.value.details["decision_id"] == "decision-safe"
    assert caught.value.details["approval_id"] == "approval-safe"
    assert "must-not-escape" not in repr(caught.value.to_dict())


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_operation_state(session, monkeypatch):
    async def exchange(self, request, result_type, **kwargs):
        await asyncio.sleep(0)
        raise _rpc_error()

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    failures = await asyncio.gather(
        session.call_tool("missing-tool", {}),
        session.list_tools(),
        return_exceptions=True,
    )
    assert [error.code for error in failures] == [
        ErrorCode.MCP_EXECUTION_FAILED.value,
        ErrorCode.MCP_DISCOVERY_FAILED.value,
    ]
    assert [error.details["operation"] for error in failures] == [
        "call_tool", "list_tools"
    ]


@pytest.mark.asyncio
async def test_native_send_request_preserves_cancellation(session, monkeypatch):
    async def exchange(self, request, result_type, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(BaseSession, "send_request", exchange)
    with pytest.raises(asyncio.CancelledError):
        await session.list_tools()
