"""OpenAI / LiteLLM adapter for MCP tools."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from ..tool import Tool

if TYPE_CHECKING:
    from ..client import MCPClient


def to_openai(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    """Convert ``Tool`` objects to the OpenAI tools array shape."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {
            "name": tool.qualified_name,
            "description": tool.description or "",
        }
        if tool.schema:
            function["parameters"] = tool.schema
        out.append({"type": "function", "function": function})
    return out


def run_tool_calls(client: "MCPClient", tool_calls: Iterable[Any]) -> list[dict[str, Any]]:
    """Execute each OpenAI tool_call and return the corresponding tool messages.

    Accepts either OpenAI Pydantic objects or plain dicts. Any execution error
    is surfaced as the tool message content so the conversation can continue.
    """
    messages: list[dict[str, Any]] = []
    for call in tool_calls:
        call_id, name, arguments = _extract_call(call)
        try:
            result = client.call_qualified(name, arguments, call_id=call_id)
            content = result.text or "(empty result)"
        except Exception as exc:  # noqa: BLE001 - surface full error to the model
            content = f"[MCP execution error] {exc}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": content,
            }
        )
    return messages


def _extract_call(call: Any) -> tuple[str, str, str]:
    """Pull (id, name, arguments_json) out of an OpenAI tool_call."""
    if isinstance(call, dict):
        call_id = call.get("id", "")
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments") or "{}"
    else:
        call_id = getattr(call, "id", "") or ""
        function = getattr(call, "function", None)
        name = getattr(function, "name", "") or ""
        arguments = getattr(function, "arguments", None) or "{}"
    if not isinstance(arguments, str):
        import json as _json
        arguments = _json.dumps(arguments)
    return call_id, name, arguments
