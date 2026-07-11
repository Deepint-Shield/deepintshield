"""Generic MCP client built on top of a DeepintShield instance.

Designed to work with *any* MCP server connected to the gateway. There is no
per-server logic anywhere in this module.
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..errors import DeepintShieldError
from .tool import MCPResult, Tool, normalize_result

if TYPE_CHECKING:
    from ..client import DeepintShield


class MCPClient:
    """Generic MCP client.

    Exposes three layers, smallest first:

    1. ``call(server, tool, **args)`` - execute one tool by name.
    2. ``list_tools()`` - discover tools (requires admin auth on the gateway).
    3. ``to_openai`` / ``to_anthropic`` / ``to_langchain`` adapters and their
       matching ``run_*`` dispatch helpers.
    """

    def __init__(self, shield: "DeepintShield") -> None:
        self._shield = shield

    # ─────────────────────────── direct execution ────────────────────────────

    def call(
        self,
        *,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        call_id: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> MCPResult:
        """Execute a single MCP tool.

        Either pass the ``arguments=`` mapping or use ``**kwargs``. Tool name
        is *bare* (no ``<server>-`` prefix); the SDK adds the prefix.
        """
        merged_args = dict(arguments or {})
        merged_args.update(kwargs)
        qualified = f"{server}-{tool}"
        payload = {
            "id": call_id or f"sdk-{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": qualified,
                "arguments": json.dumps(merged_args),
            },
        }
        headers = self._shield.headers(extra_headers)
        response = self._shield._client.post(
            f"{self._shield.base_url}/v1/mcp/tool/execute",
            json=payload,
            headers=headers,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        if response.status_code >= 400:
            raise DeepintShieldError.from_response(response.status_code, body)
        return normalize_result(qualified, body)

    def call_qualified(self, qualified_name: str, arguments: Mapping[str, Any] | str, **kwargs: Any) -> MCPResult:
        """Execute by qualified ``<server>-<tool>`` name. Accepts a JSON
        string for ``arguments`` (matching the OpenAI tool_call shape)."""
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                args_dict = {}
        else:
            args_dict = dict(arguments or {})
        server, _, tool = qualified_name.partition("-")
        if not tool:
            raise DeepintShieldError(f"Tool name '{qualified_name}' is missing a server prefix")
        return self.call(server=server, tool=tool, arguments=args_dict, **kwargs)

    # ─────────────────────────── discovery (optional) ────────────────────────

    def list_tools(
        self,
        *,
        server: str | None = None,
        admin_token: str | None = None,
    ) -> list[Tool]:
        """Discover tools via DeepintShield's admin API.

        Requires ``admin_token`` (Authorization: Bearer ...) since the
        endpoint is not VK-authed today. If discovery is unavailable in your
        environment, supply tool definitions manually instead.
        """
        headers: dict[str, str] = {}
        if admin_token:
            headers["Authorization"] = f"Bearer {admin_token}"
        response = self._shield._client.get(
            f"{self._shield.base_url}/api/mcp/clients",
            headers=self._shield.headers(headers),
        )
        if response.status_code >= 400:
            raise DeepintShieldError.from_response(response.status_code, {"raw": response.text})
        payload = response.json() or {}
        clients = payload.get("clients") or payload.get("data") or payload
        tools: list[Tool] = []
        for entry in clients or []:
            config = entry.get("config", entry) if isinstance(entry, dict) else {}
            client_name = (config.get("name") or entry.get("name") or "") if isinstance(entry, dict) else ""
            if not client_name:
                continue
            if server is not None and client_name != server:
                continue
            for raw_tool in (entry.get("tools") if isinstance(entry, dict) else None) or []:
                tool_name = raw_tool.get("name", "")
                prefix = f"{client_name}-"
                if tool_name.startswith(prefix):
                    tool_name = tool_name[len(prefix):]
                tools.append(
                    Tool(
                        server=client_name,
                        name=tool_name,
                        description=raw_tool.get("description") or "",
                        schema=raw_tool.get("parameters") or raw_tool.get("inputSchema") or {},
                    )
                )
        return tools

    # ─────────────────────────── adapters ────────────────────────────────────

    def to_openai(self, tools: Iterable[Tool]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI / LiteLLM ``tools=`` array shape."""
        from .adapters import openai as _openai
        return _openai.to_openai(tools)

    def run_openai_tool_calls(self, tool_calls: Iterable[Any]) -> list[dict[str, Any]]:
        """Execute each OpenAI ``tool_call`` and return the matching list of
        ``role: tool`` messages to append to the conversation."""
        from .adapters import openai as _openai
        return _openai.run_tool_calls(self, tool_calls)

    def to_anthropic(self, tools: Iterable[Tool]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic Messages API ``tools=`` array shape."""
        from .adapters import anthropic as _anthropic
        return _anthropic.to_anthropic(tools)

    def run_anthropic_tool_uses(self, content: Iterable[Any]) -> list[dict[str, Any]]:
        """Execute each ``tool_use`` block in an assistant message and return
        the corresponding ``tool_result`` blocks for the next user turn."""
        from .adapters import anthropic as _anthropic
        return _anthropic.run_tool_uses(self, content)

    def to_langchain(self, tools: Iterable[Tool]) -> list[Any]:
        """Wrap each MCP tool as a LangChain ``BaseTool`` that can be passed
        to LangChain agents or LangGraph ``ToolNode``s."""
        from .adapters import langchain as _langchain
        return _langchain.to_langchain(self, tools)
