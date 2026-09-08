"""Agent tool round trips through SDK adapters and the gateway transport."""
from __future__ import annotations

import json
import re

import httpx
import pytest

from deepintshield import DeepintShieldError
from deepintshield.mcp.tool import Tool


def test_anthropic_tool_aliases_round_trip_without_collisions(shield_factory):
    seen = []

    def handler(request):
        seen.append((json.loads(request.content), request.headers))
        return httpx.Response(200, json={"content": "tool completed"})

    shield = shield_factory(handler)
    tools = [
        Tool(server="data-server", name=name)
        for name in ("read", "read.file", "read_file", "read file", "x" * 90 + "a", "x" * 90 + "b")
    ]
    advertised = shield.mcp.to_anthropic(iter(tools))
    aliases = [item["name"] for item in advertised]
    assert len(set(aliases)) == len(tools)
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name) for name in aliases)
    assert aliases[0] == tools[0].qualified_name
    assert [item["name"] for item in shield.mcp.to_anthropic(reversed(tools))] == list(reversed(aliases))
    content = [{"type": "text", "text": "Calling the tools"}] + [
        {"type": "tool_use", "id": f"use-{i}", "name": name, "input": {"query": i}}
        for i, name in enumerate(aliases)
    ]
    results = shield.mcp.run_anthropic_tool_uses(content, extra_headers={"X-MCP-Subject-Token": "test-subject"})
    assert len(results) == len(tools)
    for i, (result, tool, (body, headers)) in enumerate(zip(results, tools, seen)):
        assert body["function"]["name"] == tool.qualified_name
        assert json.loads(body["function"]["arguments"]) == {"query": i}
        assert body["id"] == result["tool_use_id"] == f"use-{i}"
        assert headers["X-MCP-Subject-Token"] == "test-subject"
        assert result["is_error"] is False


@pytest.mark.parametrize("status,code", [(403, "mcp_tool_authorization_denied"), (202, "mcp_tool_approval_required"), (503, "mcp_tool_authorization_unavailable")])
def test_aliased_tool_preserves_authorization_outcomes(shield_factory, status, code):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["function"]["name"])
        return httpx.Response(status, json={"error": {"code": code, "message": "Execution stopped"}})

    shield = shield_factory(handler)
    tool = Tool(server="docs", name="read.file")
    name = shield.mcp.to_anthropic([tool])[0]["name"]
    with pytest.raises(DeepintShieldError) as error:
        shield.mcp.run_anthropic_tool_uses([{"type": "tool_use", "id": "use-1", "name": name, "input": {}}])
    assert error.value.code == code
    assert seen == [tool.qualified_name]


def test_anthropic_alias_map_is_client_local(shield_factory):
    first = shield_factory(lambda _: httpx.Response(200, json={"content": "ok"}))
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["function"]["name"])
        return httpx.Response(200, json={"content": "ok"})

    second = shield_factory(handler)
    name = first.mcp.to_anthropic([Tool(server="docs", name="read.file")])[0]["name"]
    second.mcp.run_anthropic_tool_uses([{"type": "tool_use", "id": "use-1", "name": name, "input": {}}])
    assert seen == [name], "another client's alias must never change this client's execution target"


def test_anthropic_alias_collision_fails_before_mutating_routes(shield_factory):
    shield = shield_factory(lambda _: httpx.Response(200, json={"content": "ok"}))
    original = Tool(server="docs", name="read.file")
    alias = shield.mcp.to_anthropic([original])[0]["name"]
    colliding = Tool.from_qualified(alias)
    with pytest.raises(ValueError, match="collide"):
        shield.mcp.to_anthropic([colliding])
    assert shield.mcp._anthropic_tool_names[alias] == original.qualified_name
