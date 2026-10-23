"""Canonical gateway decisions retain their meaning through the public client."""
from __future__ import annotations

import json

import httpx
import pytest

from deepintshield import DeepintShieldBlockedError, GuardrailResult, NON_BLOCKING_DECISIONS


@pytest.mark.parametrize("stage", ["input", "output"])
@pytest.mark.parametrize("method", ["evaluate_guardrail", "guard"])
def test_gateway_redaction_verdict_is_allowed_and_preserves_sanitized_content(shield_factory, stage, method):
    payload = {"result": {
        "decision": "allow_with_redaction", "mode": "sync", "reason": "Sensitive text was redacted",
        "sanitized_" + stage: "Contact [REDACTED]",
        "findings": [{"outcome": "redact"}],
    }}
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    shield = shield_factory(handler)
    result = getattr(shield, method)(stage=stage, persist=False, **{stage: "Contact person@example.com"})
    assert result.decision == "allow_with_redaction"
    assert result.allowed is True and result.blocked is False
    assert result.raw == payload
    assert result.raw["result"]["sanitized_" + stage] == "Contact [REDACTED]"
    assert len(requests) == 1
    assert requests[0].url.path == "/api/guardrails/evaluate"
    assert json.loads(requests[0].content)["persist"] is False


@pytest.mark.parametrize("decision", ["deny", "sandbox", "block", "approval_required", "unknown_decision"])
@pytest.mark.parametrize("mode", ["sync", "shadow"])
def test_guard_preserves_fail_closed_verdicts_and_explicit_nonraising_evaluation(shield_factory, decision, mode):
    payload = {"result": {"decision": decision, "mode": mode, "reason": "Policy requires intervention"}}
    shield = shield_factory(lambda request: httpx.Response(200, json=payload))
    evaluated = shield.evaluate_guardrail(stage="input", input="test", persist=False)
    assert evaluated.blocked is True and evaluated.allowed is False
    assert evaluated.mode == mode
    with pytest.raises(DeepintShieldBlockedError) as failure:
        shield.guard(stage="input", input="test", persist=False)
    assert failure.value.decision == decision
    assert failure.value.code == "guardrail_blocked"
    assert failure.value.payload == payload
    observed = shield.guard(stage="input", input="test", persist=False, raise_on_block=False)
    assert observed.blocked is True and observed.raw == payload


@pytest.mark.parametrize("decision", ["allow", "allow_with_redaction", "redact", "monitor"])
def test_nonblocking_verdicts_include_canonical_redaction_and_legacy_values(decision):
    result = GuardrailResult.from_response("input", {"result": {"decision": "  " + decision.upper() + "  "}})
    assert result.allowed is True and result.blocked is False
    assert decision in NON_BLOCKING_DECISIONS
