from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace

import pytest

from deepintshield.errors import DeepintShieldError, ErrorCode
from deepintshield.mcp._errors import (
    mcp_error_from_exception,
    mcp_error_from_result,
    raise_for_mcp_result,
)


DENIED = ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED.value
UNAVAILABLE = ErrorCode.MCP_TOOL_AUTHORIZATION_UNAVAILABLE.value
APPROVAL = ErrorCode.MCP_TOOL_APPROVAL_REQUIRED.value


def test_success_result_is_returned_by_identity_and_cannot_spoof_gaf_error():
    result = {
        "isError": False,
        "_meta": {"code": APPROVAL, "approval_id": "fake-approval"},
        "structuredContent": {"code": DENIED},
        "content": [{"type": "text", "text": f'{{"code":"{UNAVAILABLE}"}}'}],
    }

    assert mcp_error_from_result(result) is None
    assert raise_for_mcp_result(result) is result


def test_v1_mapping_prefers_meta_over_structured_and_legacy_content():
    result = {
        "isError": True,
        "_meta": {
            "code": APPROVAL,
            "decision_id": "decision-meta",
            "approval_id": "approval-meta",
        },
        "structuredContent": {
            "code": DENIED,
            "decision_id": "decision-structured",
        },
        "content": [
            {
                "type": "text",
                "text": (
                    f'{{"code":"{UNAVAILABLE}",'
                    '"decision_id":"decision-legacy"}'
                ),
            }
        ],
    }

    error = mcp_error_from_result(result)

    assert isinstance(error, DeepintShieldError)
    assert error.code == APPROVAL
    assert error.details == {
        "response_code": APPROVAL,
        "decision_id": "decision-meta",
        "approval_id": "approval-meta",
    }


def test_lower_trust_sources_cannot_decorate_selected_code_with_fake_ids():
    error = mcp_error_from_result(
        {
            "isError": True,
            "_meta": {"code": DENIED},
            "structuredContent": {
                "decision_id": "fake-decision",
                "approval_id": "fake-approval",
            },
            "content": [
                {
                    "type": "text",
                    "text": '{"approval_id":"also-fake"}',
                }
            ],
        }
    )

    assert isinstance(error, DeepintShieldError)
    assert error.code == DENIED
    assert error.details == {"response_code": DENIED}


def test_v2_object_reads_snake_case_fields_and_nested_error_envelope():
    result = SimpleNamespace(
        is_error=True,
        structured_content={
            "error": {
                "code": DENIED,
                "param": {
                    "decision_id": "decision-123",
                    "approval_id": "approval-123",
                    "token": "must-not-escape",
                },
            }
        },
        content=[],
    )

    with pytest.raises(DeepintShieldError) as caught:
        raise_for_mcp_result(result)

    assert caught.value.code == DENIED
    assert caught.value.details == {
        "response_code": DENIED,
        "decision_id": "decision-123",
        "approval_id": "approval-123",
    }
    assert "must-not-escape" not in repr(caught.value.to_dict())


def test_ordinary_tool_error_uses_execution_code_and_numeric_mcp_code():
    result = {
        "isError": True,
        "_meta": {
            "code": -32603,
            "message": "raw token sk-secret",
            "access_token": "secret",
        },
        "content": [{"type": "text", "text": "raw provider prose"}],
    }

    error = mcp_error_from_result(result)

    assert isinstance(error, DeepintShieldError)
    assert error.code == ErrorCode.MCP_EXECUTION_FAILED.value
    assert error.details == {"mcp_code": -32603}
    assert error.payload == {}
    assert "secret" not in repr(error.to_dict())
    assert "provider prose" not in str(error)


@pytest.mark.parametrize(
    "response_code",
    [
        "MCP_TOOL_AUTHORIZATION_DENIED",
        "mcp-tool-authorization-denied",
        "mcp_tool_authorization_future",
        "mcp_tool_authorization_denied_suffix",
    ],
)
def test_only_exact_known_gaf_codes_receive_special_mapping(response_code):
    error = mcp_error_from_result(
        {
            "isError": True,
            "structuredContent": {"code": response_code},
            "content": [],
        }
    )

    assert isinstance(error, DeepintShieldError)
    assert error.code == ErrorCode.MCP_EXECUTION_FAILED.value
    if response_code.islower() and "-" not in response_code and len(response_code) <= 64:
        assert error.details["response_code"] == response_code
    else:
        assert "response_code" not in error.details


def test_wire_meta_alias_has_precedence_even_when_later_alias_is_known_gaf():
    error = mcp_error_from_result(
        {
            "is_error": True,
            "_meta": {"code": "ordinary_upstream_failure"},
            "meta": {"code": DENIED},
            "content": [],
        }
    )

    assert isinstance(error, DeepintShieldError)
    assert error.code == ErrorCode.MCP_EXECUTION_FAILED.value
    assert error.details == {"response_code": "ordinary_upstream_failure"}


def test_bounded_legacy_json_is_supported_but_oversized_text_is_ignored():
    bounded = {
        "isError": True,
        "content": [
            {
                "type": "text",
                "text": f'{{"code":"{UNAVAILABLE}","decision_id":"decision-1"}}',
            }
        ],
    }
    oversized = {
        "isError": True,
        "content": [
            {
                "type": "text",
                "text": f'{{"code":"{DENIED}","padding":"' + ("x" * 16_384) + '"}',
            }
        ],
    }

    bounded_error = mcp_error_from_result(bounded)
    oversized_error = mcp_error_from_result(oversized)

    assert isinstance(bounded_error, DeepintShieldError)
    assert bounded_error.code == UNAVAILABLE
    assert bounded_error.details["decision_id"] == "decision-1"
    assert isinstance(oversized_error, DeepintShieldError)
    assert oversized_error.code == ErrorCode.MCP_EXECUTION_FAILED.value
    assert oversized_error.details == {}


@pytest.mark.parametrize(
    "result",
    [
        None,
        7,
        object(),
        {"isError": "true", "content": []},
        {"is_error": 1, "content": []},
    ],
)
def test_malformed_results_fail_as_coded_protocol_errors(result):
    error = mcp_error_from_result(result)

    assert isinstance(error, DeepintShieldError)
    assert error.code == ErrorCode.MCP_PROTOCOL_ERROR.value
    assert error.details == {}


def test_malformed_legacy_json_and_unsafe_identifiers_do_not_escape():
    result = {
        "isError": True,
        "structuredContent": {
            "code": DENIED,
            "decision_id": "x" * 129,
            "approval_id": "approval id contains prose",
        },
        "content": [
            {"type": "text", "text": "{not-json"},
            {"type": "image", "text": f'{{"code":"{APPROVAL}"}}'},
        ],
    }

    error = mcp_error_from_result(result)

    assert isinstance(error, DeepintShieldError)
    assert error.code == DENIED
    assert error.details == {"response_code": DENIED}


def test_conflicting_boolean_aliases_fail_closed_as_an_error():
    error = mcp_error_from_result(
        {"isError": False, "is_error": True, "content": []}
    )

    assert isinstance(error, DeepintShieldError)
    assert error.code == ErrorCode.MCP_EXECUTION_FAILED.value


def test_exception_mapping_is_simple_coded_and_never_copies_raw_prose():
    class UpstreamRPCError(RuntimeError):
        pass

    upstream = UpstreamRPCError("provider token sk-secret and raw prose")
    upstream.error = SimpleNamespace(
        code=-32603,
        data={
            "decision_id": "decision-safe",
            "approval_id": "approval-safe",
            "access_token": "must-not-escape",
        },
    )

    error = mcp_error_from_exception(
        upstream,
        "tools/call",
        ErrorCode.MCP_CONNECTION_FAILED,
    )

    assert type(error) is DeepintShieldError
    assert error.code == ErrorCode.MCP_CONNECTION_FAILED.value
    assert error.details == {
        "operation": "tools/call",
        "upstream_type": "UpstreamRPCError",
        "mcp_code": -32603,
        "decision_id": "decision-safe",
        "approval_id": "approval-safe",
    }
    assert error.payload == {}
    assert "sk-secret" not in str(error)
    assert "must-not-escape" not in repr(error.to_dict())


@pytest.mark.parametrize(
    "signal",
    [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(2)],
)
def test_process_control_signals_are_never_converted(signal):
    with pytest.raises(type(signal)) as caught:
        mcp_error_from_exception(
            signal,
            "tools/call",
            ErrorCode.MCP_CONNECTION_FAILED,
        )

    assert caught.value is signal


def test_existing_sdk_error_keeps_its_precise_code_without_raw_prose():
    original = DeepintShieldError(
        "private connection diagnostic",
        code=ErrorCode.MCP_CONNECTION_FAILED,
        details={"access_token": "secret"},
    )

    mapped = mcp_error_from_exception(
        original,
        "framework_tool",
        ErrorCode.MCP_EXECUTION_FAILED,
    )

    assert mapped is not original
    assert mapped.code == ErrorCode.MCP_CONNECTION_FAILED.value
    assert mapped.details == {
        "operation": "framework_tool",
        "upstream_type": "DeepintShieldError",
        "response_code": ErrorCode.MCP_CONNECTION_FAILED.value,
    }
    assert "private connection diagnostic" not in str(mapped)
    assert "secret" not in repr(mapped.to_dict())


def test_known_gaf_exception_overrides_fallback_and_nested_group_siblings():
    exception_group = getattr(builtins, "ExceptionGroup", None)
    if exception_group is None:  # pragma: no cover - Python 3.10 support
        pytest.skip("ExceptionGroup is unavailable")

    class CanonicalGAFError(RuntimeError):
        pass

    gaf = CanonicalGAFError("raw authorizer prose")
    gaf.code = APPROVAL
    gaf.details = {
        "decision_id": "decision-group",
        "approval_id": "approval-group",
        "token": "secret",
    }
    nested = exception_group(
        "raw group prose",
        [ConnectionError("raw network prose"), exception_group("nested", [gaf])],
    )

    error = mcp_error_from_exception(
        nested,
        "connect",
        ErrorCode.MCP_CONNECTION_FAILED,
    )

    assert type(error) is DeepintShieldError
    assert error.code == APPROVAL
    assert error.details == {
        "operation": "connect",
        "upstream_type": "CanonicalGAFError",
        "response_code": APPROVAL,
        "decision_id": "decision-group",
        "approval_id": "approval-group",
    }
    assert "secret" not in repr(error.to_dict())
    assert "authorizer" not in str(error)


def test_existing_sdk_exception_is_sanitized_instead_of_reexposed():
    upstream = DeepintShieldError(
        "raw token sk-secret",
        payload={"access_token": "secret"},
        code=ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED,
        details={"decision_id": "decision-sdk", "private": "secret"},
    )

    error = mcp_error_from_exception(
        upstream,
        "tools/call",
        ErrorCode.MCP_PROTOCOL_ERROR,
    )

    assert error is not upstream
    assert error.code == DENIED
    assert error.payload == {}
    assert error.details == {
        "operation": "tools/call",
        "upstream_type": "DeepintShieldError",
        "response_code": DENIED,
        "decision_id": "decision-sdk",
    }
    assert "secret" not in repr(error.to_dict())


def test_unknown_exception_code_stays_diagnostic_and_uses_fallback():
    upstream = RuntimeError("raw")
    upstream.code = "future_mcp_code"

    error = mcp_error_from_exception(
        upstream,
        "initialize",
        ErrorCode.MCP_PROTOCOL_ERROR,
    )

    assert error.code == ErrorCode.MCP_PROTOCOL_ERROR.value
    assert error.details == {
        "operation": "initialize",
        "upstream_type": "RuntimeError",
        "response_code": "future_mcp_code",
    }
