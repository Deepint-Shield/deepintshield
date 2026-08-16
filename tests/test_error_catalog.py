from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from deepintshield import (
    ERROR_CATALOG,
    DeepintShieldError,
    ErrorCategory,
    ErrorCode,
    get_error_definition,
    get_exception_error_code,
    iter_error_definitions,
)
from deepintshield.agentic.errors import (
    GovernanceConfigurationError,
    public_agentic_error,
    public_agentic_error_details,
)


def test_catalog_is_complete_immutable_and_deterministic():
    first = iter_error_definitions()
    second = iter_error_definitions()

    assert first is second
    assert len(first) == len(ErrorCode) == len(ERROR_CATALOG)
    assert tuple(item.code for item in first) == tuple(ERROR_CATALOG)
    assert set(ERROR_CATALOG) == {code.value for code in ErrorCode}
    assert all(item.description for item in first)

    with pytest.raises(TypeError):
        ERROR_CATALOG["new_code"] = first[0]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first[0].description = "changed"  # type: ignore[misc]


def test_catalog_category_filter_and_unknown_category():
    rag = iter_error_definitions(ErrorCategory.RAG)

    assert {item.code for item in rag} == {
        "rag_evaluation_failed",
        "rag_retriever_unsupported",
        "rag_embedder_unsupported",
    }
    assert iter_error_definitions("rag") == rag
    assert iter_error_definitions("not-a-category") == ()


def test_catalog_lookup_reuses_static_definition_across_threads():
    expected = get_error_definition(ErrorCode.RATE_LIMITED)
    assert expected is not None

    def read_many(_: int) -> bool:
        return all(
            get_error_definition("RATE-LIMITED") is expected
            for _ in range(5_000)
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(read_many, range(16)))


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"error": {"code": "FEATURE_LOCKED"}}, "feature_locked"),
        ({"code": "QUOTA_EXCEEDED"}, "quota_exceeded"),
        ({"error_code": "INTERNAL_ERROR"}, "internal_error"),
        ({"error": "RATE-LIMIT-EXCEEDED"}, "rate_limited"),
        ({"error": {"code": "unauthorized"}}, "authentication_failed"),
    ],
)
def test_response_codes_are_normalized_across_gateway_payload_shapes(
    payload, expected
):
    error = DeepintShieldError.from_response(
        400,
        payload,
        fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
    )

    assert error.code == expected
    assert error.description == ERROR_CATALOG[expected].description


def test_response_code_precedence_is_explicit_then_gateway_then_status_then_feature():
    explicit = DeepintShieldError.from_response(
        429,
        {"error": {"code": "FEATURE_LOCKED"}},
        code=ErrorCode.MCP_EXECUTION_FAILED,
        fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
    )
    gateway = DeepintShieldError.from_response(
        429,
        {"error": {"code": "FEATURE_LOCKED"}},
        fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
    )
    status = DeepintShieldError.from_response(
        429,
        {},
        fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
    )
    feature = DeepintShieldError.from_response(
        400,
        {},
        fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
    )
    timeout = DeepintShieldError.from_response(
        408,
        {},
        fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
    )

    assert explicit.code == "mcp_execution_failed"
    assert gateway.code == "feature_locked"
    assert status.code == "rate_limited"
    assert feature.code == "chat_request_failed"
    assert timeout.code == "transport_timeout"


def test_unknown_gateway_code_does_not_enter_stable_catalog():
    error = DeepintShieldError.from_response(
        400,
        {"error": {"code": "FUTURE_GATEWAY_CODE"}},
        fallback_code=ErrorCode.GUARDRAIL_EVALUATION_FAILED,
    )

    assert error.code == "guardrail_evaluation_failed"
    assert error.details["response_code"] == "future_gateway_code"
    assert get_error_definition("future_gateway_code") is None


def test_exception_structured_output_separates_trusted_and_diagnostic_fields():
    error = DeepintShieldError(
        "raw gateway secret",
        status_code=500,
        payload={"secret": "payload-secret"},
        code=ErrorCode.SERVER_ERROR,
        details={"request_id": "diagnostic"},
    )

    structured = error.to_dict()
    assert structured == {
        "code": "server_error",
        "description": "The gateway failed to process the request.",
        "status_code": 500,
        "retryable": True,
        "details": {"request_id": "diagnostic"},
    }
    assert "raw gateway secret" not in repr(structured)
    assert "payload-secret" not in repr(structured)


def test_public_exception_code_accessor_supports_sdk_and_agentic_boundaries():
    sdk_error = DeepintShieldError("bad input", code=ErrorCode.INVALID_ARGUMENT)
    public_agentic = public_agentic_error(
        GovernanceConfigurationError(code="agent_registration_pending")
    )
    ordinary = RuntimeError("application prose")
    forged_prose = RuntimeError("application prose")
    forged_prose._deepintshield_error_code = "not valid prose"
    third_party = RuntimeError("application prose")
    third_party.code = "private_future_code"
    colliding_third_party = RuntimeError("application prose")
    colliding_third_party.code = "rate_limited"

    assert get_exception_error_code(sdk_error) == "invalid_argument"
    assert get_exception_error_code(public_agentic) == "agent_registration_pending"
    assert get_exception_error_code(ordinary) == ""
    assert get_exception_error_code(forged_prose) == ""
    assert get_exception_error_code(third_party) == ""
    assert get_exception_error_code(colliding_third_party) == ""


def test_from_response_keeps_legacy_exception_subclass_constructor_compatible():
    class LegacyDeepintShieldError(DeepintShieldError):
        def __init__(self, message, status_code=None, payload=None):
            super().__init__(message, status_code, payload)

    error = LegacyDeepintShieldError.from_response(
        429,
        {"message": "slow down"},
    )

    assert type(error) is LegacyDeepintShieldError
    assert error.code == "rate_limited"
    assert error.retryable is True
    assert error.details == {"status_code": 429}


def test_native_exception_annotation_preserves_repr_args_and_pickle_contract(
    monkeypatch,
):
    from deepintshield import ShieldConfig

    monkeypatch.setenv("DEEPINTSHIELD_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError) as caught:
        ShieldConfig.from_env()

    error = caught.value
    assert error.args == ("could not convert string to float: 'not-a-number'",)
    assert repr(error) == "ValueError(\"could not convert string to float: 'not-a-number'\")"

    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is ValueError
    assert restored.args == error.args
    assert repr(restored) == repr(error)
    assert get_exception_error_code(restored) == "configuration_error"
    assert restored.details == {
        "environment_variable": "DEEPINTSHIELD_TIMEOUT"
    }


def test_agentic_guidance_comes_from_central_catalog():
    definition = get_error_definition(ErrorCode.VIRTUAL_KEY_MISSING)
    details = public_agentic_error_details("virtual_key_missing")
    typed_details = public_agentic_error_details(
        GovernanceConfigurationError(code="virtual_key_missing")
    )

    assert definition is not None
    assert details == {
        "code": definition.code,
        "message": definition.description,
        "action": definition.action,
        "dashboard_path": definition.dashboard_path,
    }
    assert typed_details == details
