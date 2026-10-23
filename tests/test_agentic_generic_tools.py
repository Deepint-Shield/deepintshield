"""Removing an optional framework keeps explicit tool governance intact."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.parametrize("callable_field", ["func", "_run", "run"])
@pytest.mark.parametrize("allowed", [False, True])
def test_generic_tool_guard_checks_policy_before_executing(
    shield_factory, callable_field, allowed
):
    decisions = []
    executed = []

    def handler(request):
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
        if request.url.path.endswith("/agentic-new/decide"):
            decisions.append(json.loads(request.content))
            return httpx.Response(200, json={
                "verdict": "ALLOW" if allowed else "DENY",
                "allow": allowed,
                "proceed": allowed,
                "mode": "enforce",
            })
        if request.url.path.endswith("/agentic-security/blueprints"):
            return httpx.Response(200, json={"blueprint_id": "generic-tools"})
        return httpx.Response(404)

    def write_ledger(value):
        executed.append(value)
        return value * 2

    shield = shield_factory(handler)
    tool = SimpleNamespace(name="write_ledger", **{callable_field: write_ledger})
    assert shield.agentic.guard(tool) is tool
    call = getattr(tool, callable_field)
    if allowed:
        assert call(21) == 42
        assert executed == [21]
    else:
        with pytest.raises(PermissionError) as caught:
            call(21)
        assert caught.value._deepintshield_error_code == "guardrail_denied"
        assert executed == []
    assert len(decisions) == 1
    assert decisions[0]["tool"] == "tool:write_ledger"


def test_removed_framework_is_not_automatically_imported_or_guarded():
    script = """
import sys, types
from deepintshield import DeepintShield
from deepintshield.agentic import enforcement
assert "crewai" not in enforcement._WATCHED
module = types.ModuleType("crewai")
module.unsupported_api = object()
sys.modules["crewai"] = module
with DeepintShield(virtual_key="sk-ds-test", base_url="http://127.0.0.1:1") as shield:
    assert "crewai" not in shield.agentic.enforce()
assert "deepintshield.agentic.integrations.crewai" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
