"""Anthropic Messages API adapter for MCP tools."""
from __future__ import annotations

import re
import hashlib
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..tool import Tool
from ._security import is_mcp_authorization_boundary_error

if TYPE_CHECKING:
    from ..client import MCPClient

# Anthropic tool name pattern: ^[a-zA-Z0-9_-]{1,64}$. Most MCP tools are
# already compliant; sanitize defensively in case a server uses dots or spaces.
_ANTHROPIC_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def to_anthropic(
    tools: Iterable[Tool], *, name_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert ``Tool`` objects to Anthropic's Messages API tools array."""
    out: list[dict[str, Any]] = []
    aliases = dict(name_map or {})
    for tool in tools:
        qualified = tool.qualified_name
        sanitized = _ANTHROPIC_NAME_RE.sub("_", qualified)
        if sanitized != qualified or len(sanitized) > 64:
            # Truncation/replacement alone can send two different tools to
            # the same server name. Keep a stable, provider-valid alias.
            digest = hashlib.sha256(qualified.encode("utf-8")).hexdigest()[:16]
            sanitized = f"{sanitized[:47]}_{digest}"
        if sanitized in aliases and aliases[sanitized] != qualified:
            raise ValueError("Anthropic MCP tool aliases collide")
        aliases[sanitized] = qualified
        out.append(
            {
                "name": sanitized,
                "description": tool.description or "",
                "input_schema": tool.schema or {"type": "object", "properties": {}},
            }
        )
    if name_map is not None:
        name_map.update(aliases)
    return out


def run_tool_uses(
    client: "MCPClient",
    content: Iterable[Any],
    *,
    extra_headers: Mapping[str, str] | None = None,
    name_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Execute every ``tool_use`` block in an assistant content array.

    Returns the list of ``tool_result`` blocks that should make up the next
    user message: ``messages.append({"role":"user","content": <returned>})``.
    Canonical authorization outcomes are raised rather than converted into
    model-visible tool errors.
    """
    results: list[dict[str, Any]] = []
    for block in content:
        block_type, tool_use_id, name, args = _extract_block(block)
        if block_type != "tool_use":
            continue
        try:
            result = client.call_qualified(
                (name_map or {}).get(name, name),
                args or {},
                call_id=tool_use_id,
                extra_headers=extra_headers,
            )
            text = result.text or "(empty result)"
            is_error = result.is_error
        except Exception as exc:  # noqa: BLE001
            if is_mcp_authorization_boundary_error(exc):
                raise
            text = f"[MCP execution error] {exc}"
            is_error = True
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [{"type": "text", "text": text}],
                "is_error": is_error,
            }
        )
    return results


def _extract_block(block: Any) -> tuple[str, str, str, dict[str, Any]]:
    """Pull (type, id, name, input) from an Anthropic content block."""
    if isinstance(block, dict):
        return (
            block.get("type", ""),
            block.get("id", ""),
            block.get("name", ""),
            block.get("input") or {},
        )
    return (
        getattr(block, "type", "") or "",
        getattr(block, "id", "") or "",
        getattr(block, "name", "") or "",
        getattr(block, "input", None) or {},
    )
