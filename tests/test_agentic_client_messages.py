from __future__ import annotations

import logging

import httpx
import pytest

from deepintshield.agentic import engine as engine_module
from deepintshield.agentic.engine import AgenticEngine
from deepintshield.agentic.errors import (
    GovernanceConfigurationError,
    format_registration_message,
)
from deepintshield.errors import get_error_definition


@pytest.mark.parametrize(
    ("phase", "failure", "code", "dependency_hint"),
    [
        (
            "discovery",
            GovernanceConfigurationError(
                code="agent_registration_pending", reason="private provider instructions"
            ),
            "agent_registration_pending",
            False,
        ),
        ("discovery", RuntimeError("private provider instructions"), "credential_configuration_error", False),
        ("discovery", ImportError("private provider instructions"), "credential_dependency_missing", True),
        ("acquisition", RuntimeError("private provider instructions"), "credential_exchange_failed", False),
        ("acquisition", ImportError("private provider instructions"), "credential_dependency_missing", True),
    ],
)
def test_best_effort_credential_warnings_use_trusted_guidance(
    shield_factory, monkeypatch, caplog, phase, failure, code, dependency_hint
):
    requests = []

    def handler(request):
        requests.append(request)
        raise AssertionError("Credential warning must not trigger a request")

    engine = shield_factory(handler).agentic.engine
    # Each code is warned once per process; isolate this parametrization.
    monkeypatch.setattr(engine_module, "_credential_failure_logged", set())

    def fail(*args):
        raise failure

    if phase == "discovery":
        monkeypatch.setattr(AgenticEngine, "agent_credential", property(fail))
    else:
        class BrokenCredential:
            def get_token(self):
                return fail()

        engine.set_agent_credential(BrokenCredential())

    caplog.set_level(logging.WARNING, logger="deepintshield.agentic.engine")
    assert engine.agent_token() is None
    assert requests == []
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert code in messages[0]
    assert "private provider instructions" not in messages[0]
    assert "Agentic >" not in messages[0]
    if dependency_hint:
        assert "Install the configured provider dependency and retry." in messages[0]
    elif code == "agent_registration_pending":
        assert messages[0].endswith("Contact your administrator. (agent_registration_pending)")
        assert "Please contact" not in messages[0]
    else:
        assert messages[0].count("Please contact your administrator.") == 1


@pytest.mark.parametrize(
    ("state", "code"),
    [
        ("pending", "agent_registration_pending"),
        ("approved", "governance_configuration_error"),
        ("denied", "agent_registration_denied"),
        ("quarantined", "agent_quarantined"),
        ("not_registered", "agent_not_registered"),
    ],
)
def test_status_normalizes_guidance_and_preserves_lifecycle_metadata(
    shield_factory, state, code
):
    requests = []
    body = {
        "state": state,
        "reason": "private provider instructions",
        "action": "Complete the review form in the Agentic Registry.",
        "registration_id": "registration:test",
        "agent_subject": "agent:test",
        "review_url": "/workspace/agentic-new/registry?review_registration=test",
    }

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json=body)

    result = shield_factory(handler).agentic.status()

    assert requests == ["/api/agentic-new/registry/self"]
    assert result["state"] == state
    for key in ("registration_id", "agent_subject"):
        assert result[key] == body[key]
    assert "review_url" not in result
    if state == "pending":
        assert result["reason"] == format_registration_message(
            "agent_registration_pending", reference="registration:test"
        )
        assert result["action"] == "Contact your administrator."
    else:
        assert result["reason"] == get_error_definition(code).description
        assert result["action"] == "Please contact your administrator."
    assert "private provider instructions" not in result["reason"]
    assert "Registry" not in result["action"]


def test_status_preserves_future_state_without_claiming_service_failure(shield_factory):
    body = {
        "state": "future_state",
        "registration_id": "registration:test",
        "reason": "private provider instructions",
        "action": "Complete the review form in the Agentic Registry.",
    }
    shield = shield_factory(lambda request: httpx.Response(200, json=body))

    result = shield.agentic.status()

    assert result["state"] == "future_state"
    assert result["registration_id"] == "registration:test"
    assert result["reason"] == (
        "Agent registration status is unknown. Please contact your administrator."
    )
    assert result["action"] == "Please contact your administrator."


def test_status_keeps_live_response_unchanged(shield_factory):
    body = {
        "state": "live",
        "agent_subject": "agent:test",
        "identity_provider_id": "provider:test",
        "reason": "This agent is enrolled and its governance profile is enabled.",
    }
    shield = shield_factory(lambda request: httpx.Response(200, json=body))
    assert shield.agentic.status() == body


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("network", "registry_unavailable"),
        ("invalid_json", "invalid_gateway_response"),
        ("invalid_body", "invalid_gateway_response"),
        ("missing_state", "invalid_gateway_response"),
        ("nonstring_state", "invalid_gateway_response"),
        ("empty_state", "invalid_gateway_response"),
        (401, "authentication_failed"),
        (403, "permission_denied"),
        (500, "registry_unavailable"),
    ],
)
def test_status_failures_remain_unknown_without_exposing_server_prose(
    shield_factory, failure, code
):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if failure == "network":
            raise httpx.ConnectError("private provider instructions", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, content="private provider instructions")
        if failure == "invalid_body":
            return httpx.Response(200, json=["private provider instructions"])
        if failure == "missing_state":
            return httpx.Response(200, json={"reason": "private provider instructions"})
        if failure == "nonstring_state":
            return httpx.Response(200, json={"state": [], "reason": "private provider instructions"})
        if failure == "empty_state":
            return httpx.Response(200, json={"state": "", "reason": "private provider instructions"})
        return httpx.Response(failure, json={"message": "private provider instructions"})

    result = shield_factory(handler).agentic.status()

    assert requests == ["/api/agentic-new/registry/self"]
    assert result["state"] == "unknown"
    assert result["reason"] == get_error_definition(code).description
    assert result["reason"].endswith("Please contact your administrator.")
    assert result["action"] == "Please contact your administrator."
    assert "private provider instructions" not in str(result)
