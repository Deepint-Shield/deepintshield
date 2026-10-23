from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from deepintshield.agentic import task_scope
from deepintshield.agentic.errors import GovernanceConfigurationError
from deepintshield.mcp._workload import gateway_task_headers, gateway_workload_hook
from tests.test_agentic_runtime_config import runtime_config


def test_task_mcp_preflights_capability_before_remote_execution(shield_factory):
    calls = []
    shield = shield_factory(lambda request: calls.append(request.url.path) or httpx.Response(200),
        runtime_config_handler=lambda _: httpx.Response(200, json=runtime_config("business", enabled=True)))
    with task_scope("signed-task"):
        with pytest.raises(GovernanceConfigurationError, match="agent_task_security_unavailable"):
            gateway_task_headers(shield, {})
    assert calls == []


def test_mcp_snapshot_contains_only_scoped_receipts_and_requires_supported_runtime(shield_factory):
    shield = shield_factory(lambda _: httpx.Response(200), runtime_config_handler=lambda _: httpx.Response(200,
        json={**runtime_config("business", enabled=True), "task_mandates_supported": True}))
    with task_scope("signed-task", input_receipts=["parent"]):
        headers = gateway_task_headers(shield, {"X-DS-Task-Mandate": "stale", "X-DS-Input-Receipts": '["stale-parent"]'})
    assert headers["X-Deepintshield-Task-Mandate"] == "signed-task"
    assert json.loads(headers["X-Deepintshield-Input-Receipts"]) == ["parent"]
    assert "X-DS-Task-Mandate" not in headers
    assert "X-DS-Input-Receipts" not in headers


@pytest.mark.parametrize("name", ["x-deepintshield-input-receipts", "x-ds-memory-inputs"])
def test_mcp_input_constraints_without_mandate_cannot_reach_an_older_gateway(shield_factory, name):
    shield = shield_factory(lambda _: pytest.fail("incomplete task reached gateway"))
    with pytest.raises(GovernanceConfigurationError, match="agent_task_context_invalid"):
        gateway_task_headers(shield, {name: '["input-reference"]'})


def test_mcp_per_request_hook_renews_token_and_proof_without_changing_native_calls(shield_factory, monkeypatch):
    pytest.importorskip("cryptography")
    from tests.test_agentic_dpop import _ath, _verify

    shield = shield_factory(lambda _: httpx.Response(200))
    tokens = iter(["first-token", "renewed-token"])
    monkeypatch.setattr(shield.agentic.engine, "agent_token", lambda **_: next(tokens))
    hook = gateway_workload_hook(shield, "https://gateway.example/mcp", identity=True, headers={})

    async def run():
        requests = [httpx.Request("POST", "https://gateway.example/mcp") for _ in range(2)]
        for request in requests:
            await hook(request)
        return requests

    first, second = asyncio.run(run())
    assert first.headers["X-Agent-Token"] == "first-token"
    assert second.headers["X-Agent-Token"] == "renewed-token"
    one, two = _verify(first.headers["DPoP"]), _verify(second.headers["DPoP"])
    assert one["ath"] == _ath("first-token") and two["ath"] == _ath("renewed-token")
    assert one["jti"] != two["jti"]


@pytest.mark.parametrize("url", ["https://external.example/mcp", "https://gateway.example/other", "http://gateway.example/mcp"])
def test_mcp_proof_hook_cannot_send_workload_identity_to_another_destination(shield_factory, monkeypatch, url):
    shield = shield_factory(lambda _: httpx.Response(200))
    monkeypatch.setattr(shield.agentic.engine, "agent_token", lambda **_: pytest.fail("credential requested for another destination"))
    hook = gateway_workload_hook(shield, "https://gateway.example/mcp", identity=True, headers={})
    with pytest.raises(GovernanceConfigurationError, match="agent_workload_destination_invalid"):
        asyncio.run(hook(httpx.Request("POST", url)))
