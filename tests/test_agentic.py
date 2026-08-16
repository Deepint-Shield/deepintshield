from __future__ import annotations

import json

import httpx
import pytest

def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/vk-credential-info"):
        return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
    if path.endswith("/agentic-new/decide"):
        body = json.loads(request.content)
        if body["tool"] == "tool:deny.me":
            return httpx.Response(
                200,
                headers={"x-request-id": "d1"},
                json={
                    "verdict": "DENY",
                    "allow": False,
                    "reason": "nope",
                    "failed_check": "permission",
                    "mode": "enforce",
                    "proceed": False,
                },
            )
        return httpx.Response(
            200,
            headers={"x-request-id": "d3"},
            json={
                "verdict": "ALLOW",
                "allow": True,
                "mode": "enforce",
                "proceed": True,
            },
        )
    if path.endswith("/agentic-security/decide"):
        body = json.loads(request.content)
        tool = body["tool"]
        if tool == "deny.me":
            return httpx.Response(
                200,
                json={"verdict": "DENY", "decision_id": "d1", "reason": "nope", "policy_id": "p1"},
            )
        if tool == "mask.me":
            return httpx.Response(
                200, json={"verdict": "MASK", "decision_id": "d2", "obligations": ["mask:pii"]}
            )
        return httpx.Response(200, json={"verdict": "ALLOW", "decision_id": "d3"})
    return httpx.Response(404, json={})


def test_discovery(shield_factory):
    shield = shield_factory(_handler)
    info = shield.agentic.credential_info
    assert info.agent_configured is False
    assert info.provider_type == ""


def test_discovery_prefers_canonical_agentic_identity_provider(shield_factory):
    calls: list[str] = []
    selectors: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        selectors.append(request.headers.get("X-Agent-Subject", ""))
        if request.url.path.endswith("/agentic-new/credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_id": "idp-entra",
                    "provider_type": "entra_agent_id",
                    "agent_subject": "agent:planner",
                    "tenant_id": "directory-tenant",
                    "authority": "https://login.microsoftonline.com/directory-tenant/v2.0",
                    "blueprint_client_id": "blueprint-client",
                    "gateway_audience": "api://deepintshield",
                    "fic_audience": "api://AzureADTokenExchange",
                    "exchange_endpoint": "https://login.microsoftonline.com/directory-tenant/oauth2/v2.0/token",
                    "scopes": ["api://downstream/.default"],
                    "agent_configured": True,
                },
            )
        if request.url.path.endswith("/vk-credential-info"):
            raise AssertionError("legacy discovery must not run after canonical success")
        return httpx.Response(404)

    shield = shield_factory(handler, agent_name="Invoice Helper")
    info = shield.agentic.credential_info

    assert info.agent_configured is True
    assert info.provider_id == "idp-entra"
    assert info.provider_type == "entra_agent_id"
    assert info.agent_subject == "agent:planner"
    assert info.exchange_endpoint.endswith("/oauth2/v2.0/token")
    assert calls == ["/api/agentic-new/credential-info"]
    assert selectors == [
        "agent:v1-invoice-helper--aa0e8de6daeee155f2bc6dc1797481be"
    ]


def test_discovery_does_not_fallback_after_canonical_auth_failure(shield_factory):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/agentic-new/credential-info"):
            return httpx.Response(403, json={"error": "provider disabled"})
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(200, json={"agent_configured": False})
        return httpx.Response(404)

    shield = shield_factory(handler)
    with pytest.raises(RuntimeError) as caught:
        _ = shield.agentic.credential_info
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "provider disabled" not in str(caught.value)
    assert calls == ["/api/agentic-new/credential-info"]


def test_discovery_returns_code_for_shared_key_selector_error(shield_factory):
    selectors: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/agentic-new/credential-info"):
            selectors.append(request.headers.get("X-Agent-Subject", ""))
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "X-Agent-Subject is required because this virtual "
                            "key is associated with multiple agents"
                        )
                    }
                },
            )
        if request.url.path.endswith("/vk-credential-info"):
            raise AssertionError("canonical selector errors must not downgrade")
        return httpx.Response(404)

    shield = shield_factory(handler, agent_name="")
    with pytest.raises(RuntimeError) as caught:
        _ = shield.agentic.credential_info
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "X-Agent-Subject" not in str(caught.value)
    assert selectors == [""]


def test_decide_returns_code_for_registry_association_error_after_refresh(
    shield_factory,
):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/agentic-new/credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:planner",
                },
            )
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                403,
                json={
                    "error": {
                        "message": (
                            "this virtual key is not associated with the "
                            "requested Agentic agent"
                        )
                    }
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            raise AssertionError("canonical association errors must not downgrade")
        return httpx.Response(404)

    shield = shield_factory(handler, agent_name="planner")
    with pytest.raises(RuntimeError) as caught:
        shield.agentic.decide(tool="crm_read", args={})
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "not associated" not in str(caught.value)

    assert calls == [
        "/api/agentic-new/credential-info",
        "/api/agentic-new/decide",
        "/api/agentic-new/credential-info",
        "/api/agentic-new/decide",
    ]


def test_decide_allow_returns_decision(shield_factory):
    shield = shield_factory(_handler)
    decision = shield.agentic.decide(tool="something", args={"x": 1})
    assert decision.verdict.value == "ALLOW"
    assert decision.decision_id == "d3"


def test_tool_allow_runs_body(shield_factory):
    shield = shield_factory(_handler)

    @shield.agentic.tool("allow.me")
    def double(x: int) -> int:
        return x * 2

    assert double(21) == 42


def test_tool_deny_raises(shield_factory):
    shield = shield_factory(_handler)

    @shield.agentic.tool("deny.me")
    def run() -> str:
        return "ran"

    with pytest.raises(PermissionError) as exc:
        run()
    assert exc.value._deepintshield_error_code == "guardrail_denied"


def test_older_gateway_compatibility_mask_redacts_pii_kwargs(shield_factory):
    def legacy_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/agentic-new/credential-info"):
            return httpx.Response(404)
        if path.endswith("/vk-credential-info"):
            return httpx.Response(
                200, json={"provider_type": "", "agent_configured": False}
            )
        if path.endswith("/agentic-new/decide"):
            return httpx.Response(404)
        if path.endswith("/agentic-security/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "MASK",
                    "decision_id": "d2",
                    "obligations": ["mask:pii"],
                },
            )
        return httpx.Response(404)

    shield = shield_factory(legacy_handler)
    captured: dict = {}

    @shield.agentic.tool("mask.me")
    def report(**kwargs) -> str:
        captured.update(kwargs)
        return "ok"

    report(email="jane@example.com", name="Jane")
    assert captured["email"] == "***"  # masked
    assert captured["name"] == "Jane"  # untouched


def test_public_denial_is_a_marked_builtin_permission_error(shield_factory):
    shield = shield_factory(_handler)

    @shield.agentic.tool("deny.me")
    def run() -> str:
        return "ran"

    with pytest.raises(PermissionError) as caught:
        run()
    assert caught.value._deepintshield_error_code == "guardrail_denied"
