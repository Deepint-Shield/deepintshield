from __future__ import annotations

import pytest

from deepintshield import (
    GuardrailResult,
    NON_BLOCKING_DECISIONS,
    RetrievedChunk,
    ToolInvocation,
)


def test_retrieved_chunk_payload_defaults_offset_end_to_content_length():
    chunk = RetrievedChunk(chunk_id="c", document_id="d", content="hello world")
    payload = chunk.to_payload()
    assert payload["offset_end"] == len("hello world")


def test_retrieved_chunk_respects_explicit_offset_end():
    chunk = RetrievedChunk(
        chunk_id="c", document_id="d", content="hello world", offset_start=5, offset_end=20
    )
    assert chunk.to_payload()["offset_end"] == 20


def test_retrieved_chunk_drops_empty_metadata():
    chunk = RetrievedChunk(chunk_id="c", document_id="d", content="x")
    assert "metadata" not in chunk.to_payload()


def test_tool_invocation_payload_drops_empty_metadata():
    tool = ToolInvocation(tool_name="read", tool_input={"path": "/tmp"})
    payload = tool.to_payload()
    assert payload["tool_name"] == "read"
    assert "metadata" not in payload


def test_guardrail_result_allow_is_non_blocking():
    result = GuardrailResult(decision="allow", stage="input")
    assert result.allowed is True
    assert result.blocked is False


def test_guardrail_result_redact_and_monitor_are_non_blocking():
    for decision in ("redact", "monitor"):
        result = GuardrailResult(decision=decision, stage="input")
        assert result.allowed is True
    assert {"allow", "redact", "monitor"} <= NON_BLOCKING_DECISIONS


def test_guardrail_result_block_is_blocking():
    assert GuardrailResult(decision="block", stage="action").blocked is True
    assert GuardrailResult(decision="deny", stage="action").blocked is True
    assert GuardrailResult(decision="approval_required", stage="action").blocked is True


def test_guardrail_result_from_response_normalizes_decision():
    payload = {"result": {"decision": "BLOCK", "reason": "nope"}}
    result = GuardrailResult.from_response("input", payload)
    assert result.decision == "block"
    assert result.reason == "nope"
    assert result.raw is payload


def test_guardrail_result_from_response_handles_dict_decision():
    payload = {"result": {"decision": {"decision": "allow"}}}
    result = GuardrailResult.from_response("input", payload)
    assert result.decision == "allow"


@pytest.mark.parametrize("payload", [{}, {"result": {}}, {"result": None}, {"result": []}, {"decision": ""}, {"decision": 1}])
def test_guardrail_result_rejects_missing_or_malformed_decision(payload):
    with pytest.raises(ValueError, match="guardrail"):
        GuardrailResult.from_response("input", payload)


def test_guardrail_result_mode_defaults_empty_and_is_captured():
    # Additive, non-breaking: absent mode -> "".
    assert GuardrailResult(decision="allow", stage="input").mode == ""
    assert GuardrailResult.from_response("input", {"result": {"decision": "allow"}}).mode == ""
    # Captured and normalized when the gateway reports it.
    result = GuardrailResult.from_response("output", {"result": {"decision": "block", "mode": "Shadow"}})
    assert result.mode == "shadow"
