from __future__ import annotations

import asyncio
from typing import TypedDict

import httpx
import pytest

from deepintshield.agentic.errors import DeepIntShieldError, GatewayUnavailable
from deepintshield.agentic.runtime import security_scope


class State(TypedDict):
    value: int


def increment(state: State) -> State:
    return {"value": state["value"] + 1}


def runtime_config(plan: str, *, enabled: bool = False) -> dict:
    available = plan in {"business", "enterprise", "enterprise-vpc", "enterprise_vpc"}
    return {
        "security_enabled": enabled,
        "available": available,
        "plan": plan,
        "reason": (
            "enabled" if enabled else "workspace_disabled" if available else "plan_not_included"
        ),
    }


def graph(monkeypatch):
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph
    from deepintshield.agentic.integrations import langgraph as integration

    # Finish compile's discovery attempt deterministically before each test's
    # execution. No real network, tool effects, or detached work survives it.
    def run_now(fn, *args):
        fn(*args)
        return True

    monkeypatch.setattr(integration, "run_detached", run_now)
    builder = StateGraph(State)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile()


@pytest.mark.parametrize("plan", ["community", "developer", "team", "business", "enterprise", "enterprise-vpc", "enterprise_vpc"])
@pytest.mark.parametrize("method", ["invoke", "ainvoke", "stream", "astream"])
def test_disabled_runs_skip_all_agentic_security_calls(
    shield_factory, monkeypatch, caplog, plan, method
):
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/runtime-config"):
            assert request.headers["authorization"] == "Bearer sk-ds-test"
            assert "x-agent-token" not in request.headers
            return httpx.Response(200, json=runtime_config(plan))
        raise AssertionError(f"Unexpected Agentic call: {request.url.path}")

    shield = shield_factory(handler, runtime_config_handler=handler)
    app = graph(monkeypatch)
    calls.clear()

    from deepintshield.agentic import execution
    telemetry: list[object] = []
    monkeypatch.setattr(execution, "_dispatch", lambda *args: telemetry.append(args))

    if method == "invoke":
        assert app.invoke({"value": 1}) == {"value": 2}
    elif method == "ainvoke":
        assert asyncio.run(app.ainvoke({"value": 1})) == {"value": 2}
    elif method == "stream":
        assert list(app.stream({"value": 1}, stream_mode="values"))[-1] == {"value": 2}
    else:
        async def consume():
            return [item async for item in app.astream({"value": 1}, stream_mode="values")]

        assert asyncio.run(consume())[-1] == {"value": 2}

    assert calls == ["/api/agentic-new/runtime-config"]
    assert telemetry == []
    assert "not governed yet" not in caplog.text
    assert shield.agentic.engine.execution_id == ""


@pytest.mark.parametrize("plan", ["business", "enterprise", "enterprise-vpc", "enterprise_vpc"])
def test_enabling_security_is_effective_on_next_run(shield_factory, monkeypatch, plan):
    enabled = False
    governance_calls = []

    def handler(request):
        path = request.url.path
        if path.endswith("/runtime-config"):
            return httpx.Response(200, json=runtime_config(plan, enabled=enabled))
        governance_calls.append(path)
        if path.endswith("/credential-info"):
            return httpx.Response(200, json={"agent_configured": False})
        if path.endswith("/registry/discover"):
            return httpx.Response(409, json={"error": {"code": "agent_blueprint_review_pending"}})
        raise AssertionError(f"Unexpected call: {path}")

    shield_factory(handler, runtime_config_handler=handler)
    app = graph(monkeypatch)
    assert app.invoke({"value": 1}) == {"value": 2}
    assert governance_calls == []

    enabled = True
    with pytest.raises(RuntimeError, match="agent_blueprint_review_pending"):
        app.invoke({"value": 1})
    assert any(path.endswith("/registry/discover") for path in governance_calls)

    # A prior pending approval must not remain a sticky failure after the
    # workspace owner deliberately turns Agentic Security off.
    governance_calls.clear()
    enabled = False
    assert app.invoke({"value": 1}) == {"value": 2}
    assert governance_calls == []


@pytest.mark.parametrize("status", [401, 403, 409, 429, 500, 503])
def test_runtime_config_errors_do_not_enable_passthrough(shield_factory, status):
    def handler(request):
        return httpx.Response(status, json={"error": {"code": "feature_locked"}})

    shield = shield_factory(handler, runtime_config_handler=handler)
    with pytest.raises(DeepIntShieldError):
        shield.agentic.engine.fetch_security_enabled()


@pytest.mark.parametrize("payload", [
    {},
    {"security_enabled": False},
    {**runtime_config("business"), "security_enabled": "false"},
    {**runtime_config("team"), "available": 0},
    {**runtime_config("business"), "reason": "plan_not_included"},
    {**runtime_config("team"), "reason": "workspace_disabled"},
    {**runtime_config("enterprise"), "plan": "unknown"},
    {**runtime_config("team"), "reason": "enabled"},
])
def test_malformed_runtime_config_fails_closed(shield_factory, payload):
    handler = lambda request: httpx.Response(200, json=payload)
    shield = shield_factory(handler, runtime_config_handler=handler)
    with pytest.raises(DeepIntShieldError) as caught:
        shield.agentic.engine.fetch_security_enabled()
    assert caught.value.code == "invalid_gateway_response"


def test_runtime_config_transport_failure_fails_closed(shield_factory):
    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    shield = shield_factory(handler, runtime_config_handler=handler)
    with pytest.raises(GatewayUnavailable):
        shield.agentic.engine.fetch_security_enabled()


@pytest.mark.parametrize("status", [404, 405, 410, 501])
def test_older_gateway_stays_governed(shield_factory, status):
    handler = lambda request: httpx.Response(status)
    shield = shield_factory(handler, runtime_config_handler=handler)
    assert shield.agentic.engine.fetch_security_enabled() is True


def test_older_gateway_spa_fallback_still_enforces_decisions(shield_factory, monkeypatch):
    decisions = []

    def handler(request):
        path = request.url.path
        if path.endswith("/runtime-config"):
            # Older gateways register their SPA catch-all after API routes.
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<!doctype html><html><body>DeepintShield</body></html>",
            )
        if path.endswith("/credential-info"):
            return httpx.Response(200, json={"agent_configured": False})
        if path.endswith("/registry/discover"):
            return httpx.Response(200, json={"blueprint_scan_status": "complete"})
        if path.endswith("/blueprints"):
            return httpx.Response(200, json={"blueprint_id": "approved-blueprint"})
        if path.endswith("/decide"):
            decisions.append(path)
            return httpx.Response(200, json={
                "verdict": "DENY", "allow": False, "proceed": False,
                "reason": "tool_not_allowed", "mode": "enforce",
            })
        raise AssertionError(f"Unexpected call: {path}")

    from deepintshield.agentic import execution
    monkeypatch.setattr(execution, "_dispatch", lambda *args: False)
    shield_factory(handler, runtime_config_handler=handler)
    app = graph(monkeypatch)
    with pytest.raises(PermissionError, match="guardrail_denied"):
        app.invoke({"value": 1})
    assert decisions == ["/api/agentic-new/decide"]


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_html_auth_and_gateway_errors_do_not_enable_passthrough(shield_factory, status):
    handler = lambda request: httpx.Response(
        status, headers={"content-type": "text/html"}, text="<html>Request failed</html>"
    )
    shield = shield_factory(handler, runtime_config_handler=handler)
    with pytest.raises(DeepIntShieldError):
        shield.agentic.engine.fetch_security_enabled()


def test_disabled_runtime_does_not_fetch_agent_credentials(shield_factory):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=runtime_config("team"))

    shield = shield_factory(handler, runtime_config_handler=handler)
    with security_scope(shield.agentic.engine):
        assert shield.agentic.engine.agent_token(fail_closed=True) is None
        assert shield._agent_token() is None
    assert calls == ["/api/agentic-new/runtime-config"]


def test_direct_decide_does_not_fetch_credentials_when_disabled(shield_factory):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=runtime_config("team"))

    shield = shield_factory(handler, runtime_config_handler=handler)
    decision = shield.agentic.decide(tool="increment", args={"value": 1})
    assert decision.mode == "passthrough"
    assert decision.proceed is True
    assert decision.gaf_checked is False
    assert calls == ["/api/agentic-new/runtime-config"]


def test_skipped_async_discovery_does_not_attest_code(shield_factory, monkeypatch):
    import threading
    from deepintshield.agentic import registry

    posted = threading.Event()
    handler = lambda request: httpx.Response(200, json=runtime_config("team"))
    shield = shield_factory(handler, runtime_config_handler=handler)
    engine = shield.agentic.engine
    original = registry._post_and_release

    def completed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            posted.set()

    monkeypatch.setattr(registry, "_post_and_release", completed)
    result = registry.discover(engine, [increment])
    assert result["dispatched"] is True
    assert posted.wait(2)
    assert engine._registration_capture_ready is False
    assert engine._registration_captured_tools == set()


def test_compile_config_outage_is_retried_and_paid_denial_still_enforced(
    shield_factory, monkeypatch
):
    available = False
    decisions = []

    def handler(request):
        path = request.url.path
        if path.endswith("/runtime-config"):
            if not available:
                raise httpx.ConnectError("temporary outage", request=request)
            return httpx.Response(200, json=runtime_config("business", enabled=True))
        if path.endswith("/credential-info"):
            return httpx.Response(200, json={"agent_configured": False})
        if path.endswith("/registry/discover"):
            return httpx.Response(200, json={"blueprint_scan_status": "complete"})
        if path.endswith("/blueprints"):
            return httpx.Response(200, json={"blueprint_id": "approved-blueprint"})
        if path.endswith("/decide"):
            decisions.append(path)
            return httpx.Response(200, json={
                "verdict": "DENY", "allow": False, "proceed": False,
                "reason": "tool_not_allowed", "mode": "enforce",
            })
        raise AssertionError(f"Unexpected call: {path}")

    from deepintshield.agentic import execution
    monkeypatch.setattr(execution, "_dispatch", lambda *args: False)
    shield_factory(handler, runtime_config_handler=handler)
    app = graph(monkeypatch)
    available = True
    with pytest.raises(PermissionError, match="guardrail_denied"):
        app.invoke({"value": 1})
    assert decisions == ["/api/agentic-new/decide"]
