"""The native LangChain callback path: shield.agentic.guard() gates every tool
through the PDP with zero per-tool code."""
from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("langchain_core")

from langchain_core.tools import tool  # noqa: E402


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/vk-credential-info"):
        return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
    if path.endswith("/agentic-new/decide"):
        body = json.loads(request.content)
        if body["tool"] == "tool:write_ledger":
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
            headers={"x-request-id": "d2"},
            json={
                "verdict": "ALLOW",
                "allow": True,
                "mode": "enforce",
                "proceed": True,
            },
        )
    if path.endswith("/agentic-security/decide"):
        body = json.loads(request.content)
        if body["tool"] == "write_ledger":
            return httpx.Response(
                200, json={"verdict": "DENY", "decision_id": "d1", "reason": "nope", "policy_id": "p1"}
            )
        return httpx.Response(200, json={"verdict": "ALLOW", "decision_id": "d2"})
    return httpx.Response(404, json={})


@tool
def write_ledger(row: str) -> str:
    """Append a row to the finance ledger."""
    return f"wrote {row}"


@tool
def read_ledger(row: str) -> str:
    """Read a row from the finance ledger."""
    return f"read {row}"


def test_guard_returns_callback_handler(shield_factory):
    shield = shield_factory(_handler)
    guard = shield.agentic.guard()
    # raise_error must be set or LangChain swallows the DENY exception.
    assert getattr(guard, "raise_error", False) is True


def test_guard_denies_tool(shield_factory):
    shield = shield_factory(_handler)
    guard = shield.agentic.guard()
    with pytest.raises(PermissionError) as exc:
        write_ledger.invoke({"row": "x"}, config={"callbacks": [guard]})
    assert exc.value._deepintshield_error_code == "guardrail_denied"
    assert str(exc.value) == (
        "guardrail_denied: Agentic authorization denied this operation."
    )


def test_guard_allows_tool(shield_factory):
    shield = shield_factory(_handler)
    guard = shield.agentic.guard()
    assert read_ledger.invoke({"row": "y"}, config={"callbacks": [guard]}) == "read y"


def test_plain_tool_execution_is_automatically_denied(shield_factory):
    """No callback/decorator is required once a DeepintShield client exists."""
    shield_factory(_handler)
    with pytest.raises(PermissionError) as exc:
        write_ledger.invoke({"row": "x"})
    assert exc.value._deepintshield_error_code == "guardrail_denied"


def test_plain_tool_execution_is_automatically_allowed(shield_factory):
    shield_factory(_handler)
    assert read_ledger.invoke({"row": "y"}) == "read y"
