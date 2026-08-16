from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from deepintshield.agentic.engine import _permission_subject
from deepintshield.agentic.errors import (
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
)


def test_gate_blocks_when_fallback_blueprint_capture_is_not_acknowledged(
    monkeypatch,
):
    from deepintshield.agentic import gate, registry

    decided = False

    class Engine:
        virtual_key = "vk"

        def decide(self, _context):
            nonlocal decided
            decided = True
            raise AssertionError("PDP must not run before blueprint acknowledgement")

    monkeypatch.setattr(
        registry,
        "ensure_registration_capture",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(GovernanceConfigurationError) as caught:
        gate.resolve(Engine(), "write_account", (), {})

    assert caught.value.code == "blueprint_scan_unavailable"
    assert decided is False


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content) if request.content else {}


def _base_response(request: httpx.Request, calls: list[tuple[str, dict]]):
    path = request.url.path
    if path.endswith("/vk-credential-info"):
        return httpx.Response(
            200, json={"provider_type": "", "agent_configured": False}
        )
    if path.endswith("/agentic-new/identity/resolve"):
        body = _body(request)
        calls.append(("identity", body))
        identifier = body.get("email") or body.get("username")
        slug = str(identifier).lower().replace("@", "-").replace(".", "-")
        return httpx.Response(200, json={"subject": f"user:{slug}"})
    return None


def test_governed_tool_is_blocked_by_agentic_new_before_legacy(shield_factory):
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, calls)
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            calls.append(("gaf", _body(request)))
            return httpx.Response(
                200,
                json={
                    "verdict": "DENY",
                    "allow": False,
                    "reason": "obo_user_lacks_perm",
                    "failed_check": "user_perm",
                    "checks": [{"name": "user_perm", "allowed": False}],
                    "mode": "enforce",
                    "would_block": True,
                    "proceed": False,
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            calls.append(("legacy", _body(request)))
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy-allow"}
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    shield.agentic.as_user("alice@corp.com")

    @shield.agentic.tool("ledger_write")
    def write_ledger(amount: int) -> str:
        return f"wrote {amount}"

    with pytest.raises(PermissionError) as exc:
        write_ledger(50)

    assert exc.value._deepintshield_error_code == "obo_user_lacks_perm"
    assert not [call for call in calls if call[0] == "legacy"]
    (_, sent), = [call for call in calls if call[0] == "gaf"]
    assert sent == {
        "agent": "",
        "user": "user:alice-corp-com",
        "permission": "holder",
        "object": "permission:v1-ledger-write-write--aa0d77fb375df555cbb7b17f2e55a4e8",
        "tool": "tool:ledger_write",
        "action": "",
        "delegation_id": "",
        "action_class": "write",
        "args_digest": sent["args_digest"],
        "execution_id": sent["execution_id"],
        "session_id": sent["session_id"],
    }
    assert sent["execution_id"] == sent["session_id"]
    assert sent["args_digest"].startswith("sha256:")
    assert len(sent["args_digest"]) == len("sha256:") + 64


def test_direct_decide_uses_safe_builtin_context_error(shield_factory):
    shield = shield_factory(lambda _request: httpx.Response(500))

    with pytest.raises(RuntimeError) as caught:
        shield.agentic.decide()

    assert caught.value._deepintshield_error_code == "agent_decision_context_missing"
    assert str(caught.value).startswith("agent_decision_context_missing: ")


def test_canonical_allow_is_not_replaced_by_hidden_legacy_deny(shield_factory):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            calls.append("gaf")
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "reason": "allowed",
                    "mode": "enforce",
                    "would_block": False,
                    "proceed": True,
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            calls.append("legacy")
            return httpx.Response(
                200,
                json={
                    "verdict": "DENY",
                    "decision_id": "legacy-deny",
                    "reason": "legacy policy",
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("crm_read")
    def read() -> str:
        return "secret"

    assert read() == "secret"
    assert calls == ["gaf"]
    decision = shield.agentic.decide(tool="crm_read", args={})
    assert decision.verdict.value == "ALLOW"
    assert decision.proceed is True
    assert decision.gaf_checked is True
    assert calls == ["gaf", "gaf"]


def test_canonical_allow_never_calls_or_polls_legacy_approval(shield_factory):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "REQUIRE_APPROVAL",
                    "decision_id": "legacy-approval",
                    "reason": "legacy approval",
                    "approvers": ["security"],
                },
            )
        if request.url.path.endswith(
            "/agentic-security/approvals/legacy-approval"
        ):
            return httpx.Response(200, json={"state": "approved"})
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("payment_send")
    def send() -> str:
        return "sent"

    assert send() == "sent"
    assert "/api/agentic-security/decide" not in paths
    assert not any("/approvals/" in path for path in paths)


def test_older_gateway_compatibility_approval_can_still_poll(shield_factory):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(404)
        if request.url.path.endswith("/agentic-security/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "REQUIRE_APPROVAL",
                    "decision_id": "legacy-approval",
                    "reason": "legacy approval",
                    "approvers": ["security"],
                },
            )
        if request.url.path.endswith(
            "/agentic-security/approvals/legacy-approval"
        ):
            return httpx.Response(200, json={"state": "approved"})
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("payment_send")
    def send() -> str:
        return "sent"

    assert send() == "sent"
    assert "/api/agentic-security/decide" in paths
    assert "/api/agentic-security/approvals/legacy-approval" in paths


def test_public_approval_poll_timeout_is_code_only(shield_factory):
    engine = shield_factory(lambda _request: httpx.Response(404)).agentic.engine
    engine._approval_timeout = 0.0

    with pytest.raises(GuardrailApprovalPending) as caught:
        engine.poll_approval("approval-timeout")

    assert caught.value.code == "require_approval"
    assert caught.value.decision_id == "approval-timeout"
    assert str(caught.value) == "require_approval"
    assert caught.value.__cause__ is None


def test_approval_poll_transport_failure_suppresses_third_party_prose(
    shield_factory,
):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private approval host", request=request)

    engine = shield_factory(handler).agentic.engine

    with pytest.raises(GatewayUnavailable) as caught:
        engine.poll_approval("approval-network")

    assert caught.value.code == "gateway_unavailable"
    assert caught.value.reason == "ConnectError"
    assert caught.value.__cause__ is None
    assert "private approval host" not in repr(caught.value)


@pytest.mark.parametrize(
    ("status", "server_code", "exception_type"),
    (
        (503, "approval_store_unavailable", GatewayUnavailable),
        (403, "approval_access_denied", GovernanceConfigurationError),
    ),
)
def test_approval_poll_http_failures_preserve_only_server_codes(
    shield_factory,
    status,
    server_code,
    exception_type,
):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "error": {
                    "code": server_code,
                    "message": "private approval backend prose",
                }
            },
        )

    engine = shield_factory(handler).agentic.engine

    with pytest.raises(exception_type) as caught:
        engine.poll_approval("approval-http")

    assert caught.value.code == server_code
    assert str(caught.value) == server_code
    assert caught.value.__cause__ is None
    assert "private approval backend prose" not in repr(caught.value)


def test_older_gateway_fallback_never_receives_an_invented_default_agent(
    shield_factory,
):
    legacy_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(404)
        if request.url.path.endswith("/agentic-security/decide"):
            legacy_payloads.append(_body(request))
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy"}
            )
        return httpx.Response(404)

    shield_factory(handler).agentic.decide(tool="crm_read", args={})

    assert legacy_payloads[0]["principal"] == ""
    assert legacy_payloads[0]["actor_chain"] == []


def test_old_gateway_uses_explicit_legacy_compat_and_caches_route_absence(
    shield_factory,
):
    calls = {"gaf": 0, "legacy": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            calls["gaf"] += 1
            return httpx.Response(404, json={"error": "not found"})
        if request.url.path.endswith("/agentic-security/decide"):
            calls["legacy"] += 1
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy"}
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("crm_read")
    def read() -> str:
        return "ok"

    assert read() == "ok"
    assert read() == "ok"
    decision = shield.agentic.decide(tool="crm_read", args={})
    assert decision.mode == "legacy-compat"
    assert decision.gaf_checked is False
    assert decision.gaf_verdict == ""
    assert calls == {"gaf": 1, "legacy": 3}


def test_modern_gateway_missing_canonical_decide_fails_closed_without_legacy(
    shield_factory,
):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/agentic-new/credential-info"):
            return httpx.Response(
                200, json={"provider_type": "", "agent_configured": False}
            )
        if request.url.path.endswith("/agentic-new/decide"):
            calls.append("gaf")
            return httpx.Response(404)
        if request.url.path.endswith("/agentic-security/decide"):
            calls.append("legacy")
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy"}
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    with pytest.raises(RuntimeError) as caught:
        shield.agentic.decide(tool="crm_read", args={})
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "canonical decision route is missing" not in str(caught.value)
    assert calls == ["gaf"]


def test_agentic_new_only_gateway_does_not_require_legacy_route(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "reason": "allowed",
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("crm_read")
    def read() -> str:
        return "ok"

    assert read() == "ok"


def test_shadow_gaf_denial_proceeds_but_remains_visible(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "DENY",
                    "allow": False,
                    "reason": "obo_tool_not_allowed",
                    "failed_check": "tool",
                    "mode": "shadow",
                    "would_block": True,
                    "proceed": True,
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy"}
            )
        return httpx.Response(404)

    decision = shield_factory(handler).agentic.decide(tool="crm_read", args={})

    assert decision.verdict.value == "ALLOW"
    assert decision.gaf_checked is True
    assert decision.gaf_verdict == "DENY"
    assert decision.would_block is True
    assert decision.failed_check == "tool"


def test_gaf_proceed_false_cannot_be_bypassed_by_inconsistent_allow_verdict(
    shield_factory,
):
    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": False,
                    "reason": "inconsistent upstream response",
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("dangerous_action")
    def dangerous_action() -> str:
        return "executed"

    with pytest.raises(PermissionError) as caught:
        dangerous_action()
    assert caught.value._deepintshield_error_code == "guardrail_denied"
    assert "inconsistent upstream response" not in str(caught.value)


@pytest.mark.parametrize(
    ("verdict", "exception_type", "expected_code"),
    (
        ("DENY", PermissionError, "guardrail_denied"),
        ("REQUIRE_APPROVAL", RuntimeError, "require_approval"),
    ),
)
def test_blocking_verdict_cannot_emit_success_reason_as_error_code(
    shield_factory,
    verdict,
    exception_type,
    expected_code,
):
    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": verdict,
                    "allow": False,
                    "mode": "enforce",
                    "proceed": False,
                    "reason": "allowed",
                    "approval_id": "approval-safe-code",
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("dangerous_action")
    def dangerous_action() -> str:
        return "executed"

    with pytest.raises(exception_type) as caught:
        dangerous_action()

    assert caught.value._deepintshield_error_code == expected_code
    assert str(caught.value).startswith(expected_code + ": ")


def test_gaf_approval_returns_control_without_polling_admin_endpoint(shield_factory):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                json={
                    "verdict": "REQUIRE_APPROVAL",
                    "allow": False,
                    "reason": "require_approval",
                    "mode": "enforce",
                    "would_block": True,
                    "proceed": False,
                    "approval_id": "approval-1",
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("payment_send")
    def send() -> str:
        return "sent"

    with pytest.raises(RuntimeError) as exc:
        send()
    assert exc.value._deepintshield_error_code == "require_approval"
    assert not any("/approvals/" in path for path in paths)


def test_gaf_approval_without_queue_id_never_enters_legacy_poll(shield_factory):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                200,
                headers={"x-request-id": "request-approval-failed"},
                json={
                    "verdict": "REQUIRE_APPROVAL",
                    "allow": False,
                    "reason": "require_approval",
                    "mode": "enforce",
                    "proceed": False,
                    "approval_id": "",
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    @shield.agentic.tool("payment_send")
    def send() -> str:
        return "sent"

    started = time.perf_counter()
    with pytest.raises(RuntimeError) as exc:
        send()
    assert time.perf_counter() - started < 0.5
    assert exc.value._deepintshield_error_code == "require_approval"
    assert not any("/approvals/" in path for path in paths)


def test_explicit_gaf_resource_contract_overrides_safe_defaults(shield_factory):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            sent.append(_body(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "reason": "allowed",
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    shield.agentic.decide(
        tool="close_request",
        args={"id": "r1"},
        agent="agent:helper",
        user="user:analyst",
        permission="can_fulfill",
        object="request:r1",
        delegation_id="delegation:d1",
        action="close",
        action_class="write",
    )

    assert sent[0] | {"args_digest": "<digest>"} == {
        "agent": "agent:helper",
        "user": "user:analyst",
        "permission": "can_fulfill",
        "object": "request:r1",
        "tool": "tool:close_request",
        "action": "close",
        "delegation_id": "delegation:d1",
        "action_class": "write",
        "args_digest": "<digest>",
        "execution_id": sent[0]["execution_id"],
        "session_id": sent[0]["session_id"],
    }
    assert sent[0]["execution_id"] == sent[0]["session_id"]


def test_opaque_action_defaults_to_write_but_explicit_class_wins(shield_factory):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            sent.append(_body(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    shield.agentic.decide(tool="invoke", args={})
    shield.agentic.decide(tool="invoke", args={}, action_class="read")
    shield.agentic.decide(tool="read_records", args={}, action="delete")

    assert sent[0]["action_class"] == "write"
    assert (
        sent[0]["object"]
        == "permission:v1-invoke-write--0e0ef2d4f70e02261a12869993a4cea1"
    )
    assert sent[1]["action_class"] == "read"
    assert (
        sent[1]["object"]
        == "permission:v1-invoke-read--3982b46ec69ee21987493ac81a5026e7"
    )
    assert sent[2]["action"] == "delete"
    assert sent[2]["action_class"] == "write"
    assert (
        sent[2]["object"]
        == "permission:v1-read-records-write--d3b5e9943f4e1d2141d0a2016c537f57"
    )


def test_canonical_tool_subject_is_not_double_prefixed_in_permission_default(
    shield_factory,
):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            sent.append(_body(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield_factory(handler).agentic.decide(tool="tool:crm_read", args={})

    assert sent[0]["tool"] == "tool:crm_read"
    assert (
        sent[0]["object"]
        == "permission:v1-crm-read-read--6006eac4d1201b30bb01eb8c12648ab3"
    )


def test_permission_subject_matches_go_versioned_collision_resistant_contract():
    expected = "permission:v1-code-read--0a0634a801d3fa80ef56252a11b54a4e"
    assert _permission_subject("code:read") == expected
    assert _permission_subject("  CODE:READ  ") == expected
    assert _permission_subject("code-read") != expected


def test_gaf_uses_the_agent_subject_bound_to_the_virtual_key(shield_factory):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:invoice-helper",
                },
            )
        if request.url.path.endswith("/agentic-new/decide"):
            sent.append(_body(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield_factory(handler).agentic.decide(tool="invoice_read", args={})

    assert sent[0]["agent"] == "agent:invoice-helper"


def test_clients_sharing_one_vk_select_distinct_agent_bindings(shield_factory):
    sent: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        selector = request.headers.get("X-Agent-Subject", "")
        if request.url.path.endswith("/agentic-new/credential-info"):
            sent.append(("credential-info", selector, ""))
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": selector,
                },
            )
        if request.url.path.endswith("/agentic-new/decide"):
            body = _body(request)
            sent.append(("decide", selector, body["agent"]))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    planner = shield_factory(handler, agent_name="planner")
    auditor = shield_factory(handler, agent_name="risk-auditor")

    planner.agentic.decide(tool="invoice_read", args={})
    auditor.agentic.decide(tool="invoice_read", args={})

    assert sent == [
        ("credential-info", "agent:planner", ""),
        ("decide", "agent:planner", "agent:planner"),
        ("credential-info", "agent:risk-auditor", ""),
        ("decide", "agent:risk-auditor", "agent:risk-auditor"),
    ]


def test_vk_agent_subject_discovery_is_single_flight(shield_factory):
    reads = 0
    lock = threading.Lock()
    barrier = threading.Barrier(12)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal reads
        if request.url.path.endswith("/vk-credential-info"):
            with lock:
                reads += 1
            time.sleep(0.02)
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:single-flight",
                },
            )
        return httpx.Response(404)

    engine = shield_factory(handler).agentic.engine

    def get_subject(_index: int) -> str:
        barrier.wait(5)
        return engine.agent_subject

    with ThreadPoolExecutor(max_workers=12) as pool:
        subjects = list(pool.map(get_subject, range(12)))

    assert subjects == ["agent:single-flight"] * 12
    assert reads == 1


def test_configured_agent_token_failure_blocks_before_the_pdp(shield_factory):
    pdp_calls = 0

    class BrokenCredential:
        def get_token(self) -> str:
            raise RuntimeError("token exchange unavailable")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pdp_calls
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "generic_oidc",
                    "agent_configured": True,
                    "agent_subject": "agent:configured",
                },
            )
        if request.url.path.endswith(("/agentic-new/decide", "/agentic-security/decide")):
            pdp_calls += 1
        return httpx.Response(404)

    shield = shield_factory(handler)
    shield.agentic.engine.set_agent_credential(BrokenCredential())

    with pytest.raises(ConnectionError) as caught:
        shield.agentic.decide(tool="invoice_read", args={})
    assert caught.value._deepintshield_error_code == "gateway_unavailable"
    assert "token exchange unavailable" not in repr(caught.value)
    assert pdp_calls == 0


def test_transport_failure_exposes_no_third_party_message_or_cause(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={"provider_type": "", "agent_configured": False},
            )
        raise httpx.ConnectError("private upstream hostname", request=request)

    shield = shield_factory(handler)

    with pytest.raises(ConnectionError) as caught:
        shield.agentic.decide(tool="invoice_read", args={})

    assert caught.value._deepintshield_error_code == "gateway_unavailable"
    assert caught.value.__cause__ is None
    assert "private upstream hostname" not in repr(caught.value)


def test_empty_agent_token_blocks_before_the_pdp(shield_factory):
    pdp_calls = 0

    class EmptyCredential:
        def get_token(self) -> str:
            return ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal pdp_calls
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "generic_oidc",
                    "agent_configured": True,
                    "agent_subject": "agent:configured",
                },
            )
        if request.url.path.endswith(("/agentic-new/decide", "/agentic-security/decide")):
            pdp_calls += 1
        return httpx.Response(404)

    shield = shield_factory(handler)
    shield.agentic.engine.set_agent_credential(EmptyCredential())

    with pytest.raises(RuntimeError) as caught:
        shield.agentic.decide(tool="invoice_read", args={})
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "empty agent token" not in str(caught.value)
    assert pdp_calls == 0


def test_older_gateway_sends_empty_agent_for_server_side_vk_binding(shield_factory):
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            sent.append(_body(request))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield_factory(handler).agentic.decide(tool="invoice_read", args={})

    assert sent[0]["agent"] == ""


def test_mismatched_explicit_agent_is_rejected_after_one_binding_refresh(
    shield_factory,
):
    sent: list[dict] = []
    credential_reads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal credential_reads
        if request.url.path.endswith("/vk-credential-info"):
            credential_reads += 1
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:vk-bound",
                },
            )
        if request.url.path.endswith("/agentic-new/decide"):
            sent.append(_body(request))
            return httpx.Response(
                403,
                json={"error": "agent does not match virtual key binding"},
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    with pytest.raises(RuntimeError) as caught:
        shield.agentic.decide(
            tool="invoice_read",
            args={},
            agent="agent:impersonated",
        )
    assert caught.value._deepintshield_error_code == "governance_configuration_error"

    assert credential_reads == 2
    assert [body["agent"] for body in sent] == [
        "agent:impersonated",
        "agent:impersonated",
    ]


@pytest.mark.parametrize(
    ("requester", "resolve_field", "expected_user"),
    [
        ("alice@corp.com", "email", "user:alice-corp-com"),
        ("alice", "username", "user:alice"),
    ],
)
def test_client_requester_is_lazily_resolved_and_sent_to_gaf(
    shield_factory,
    requester,
    resolve_field,
    expected_user,
):
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, calls)
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            calls.append(("gaf", _body(request)))
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler, requester=requester)
    assert calls == []  # construction is still network-free

    @shield.agentic.tool("invoice_read")
    def read_invoice() -> str:
        return "invoice"

    assert read_invoice() == "invoice"

    identity_calls = [body for name, body in calls if name == "identity"]
    assert identity_calls == [{"kind": "user", resolve_field: requester}]
    gaf_calls = [body for name, body in calls if name == "gaf"]
    assert len(gaf_calls) == 1
    assert gaf_calls[0]["user"] == expected_user


def test_run_identity_overrides_requester_and_restores_it_request_locally(
    shield_factory,
):
    users: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/vk-credential-info"):
            return httpx.Response(
                200, json={"provider_type": "", "agent_configured": False}
            )
        if path.endswith("/agentic-new/identity/resolve"):
            body = _body(request)
            identifier = body.get("email") or body.get("username")
            slug = str(identifier).lower().replace("@", "-").replace(".", "-")
            return httpx.Response(200, json={"subject": f"user:{slug}"})
        if path.endswith("/agentic-new/decide"):
            users.append(_body(request)["user"])
            return httpx.Response(
                200,
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler, requester="alice@corp.com")

    @shield.agentic.tool("invoice_read")
    def read_invoice() -> str:
        return "invoice"

    read_invoice()
    with shield.agentic.run(email="bob@corp.com"):
        read_invoice()
    read_invoice()

    assert users == [
        "user:alice-corp-com",
        "user:bob-corp-com",
        "user:alice-corp-com",
    ]


def test_configured_gaf_outage_fails_closed_instead_of_using_legacy(shield_factory):
    legacy_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal legacy_called
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                500,
                json={
                    "error": {
                        "code": "authz_store_unavailable",
                        "message": "private OpenFGA backend prose",
                    }
                },
            )
        if request.url.path.endswith("/agentic-security/decide"):
            legacy_called = True
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy"}
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    with pytest.raises(ConnectionError) as caught:
        shield.agentic.decide(tool="crm_read", args={})
    assert caught.value._deepintshield_error_code == "authz_store_unavailable"
    assert str(caught.value).startswith("authz_store_unavailable: ")
    assert "private OpenFGA backend prose" not in repr(caught.value)
    assert legacy_called is False


def test_legacy_compatibility_outage_preserves_server_error_code(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/credential-info"):
            return httpx.Response(404)
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(404)
        if request.url.path.endswith("/agentic-security/decide"):
            return httpx.Response(
                503,
                json={
                    "error": {
                        "code": "legacy_pdp_unavailable",
                        "message": "private legacy backend prose",
                    }
                },
            )
        return httpx.Response(404)

    shield = shield_factory(handler)

    with pytest.raises(ConnectionError) as caught:
        shield.agentic.decide(tool="crm_read", args={})

    assert caught.value._deepintshield_error_code == "legacy_pdp_unavailable"
    assert str(caught.value).startswith("legacy_pdp_unavailable: ")
    assert "private legacy backend prose" not in repr(caught.value)


def test_unconfigured_modern_gaf_fails_closed_instead_of_using_legacy(
    shield_factory,
):
    legacy_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal legacy_called
        base = _base_response(request, [])
        if base is not None:
            return base
        if request.url.path.endswith("/agentic-new/decide"):
            return httpx.Response(
                503, json={"error": "OpenFGA is not configured"}
            )
        if request.url.path.endswith("/agentic-security/decide"):
            legacy_called = True
            return httpx.Response(
                200, json={"verdict": "ALLOW", "decision_id": "legacy"}
            )
        return httpx.Response(404)

    shield = shield_factory(handler)
    with pytest.raises(RuntimeError) as caught:
        shield.agentic.decide(tool="crm_read", args={})
    assert caught.value._deepintshield_error_code == "governance_configuration_error"
    assert "canonical Agentic-New PDP is not configured" not in str(caught.value)
    assert legacy_called is False
