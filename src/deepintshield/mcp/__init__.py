"""Thin MCP gateway integration for DeepIntShield.

The official MCP Python SDK owns protocol negotiation, transport, sessions,
types, and result content. DeepIntShield supplies only the governed ``/mcp``
connection and a stable coded-exception boundary.

Quick start
-----------

    import asyncio
    from deepintshield import DeepintShield, DeepintShieldError

    shield = DeepintShield(
        virtual_key="sk-ds-your-virtual-key",
        base_url="http://localhost:8080",
    )

    async def main():
        try:
            async with shield.mcp.connect() as session:
                result = await session.call_tool(
                    "DeepWiki-ask_question",
                    {"repoName": "facebook/react", "question": "What is Suspense?"},
                )
        except DeepintShieldError as exc:
            print(exc.code)

    asyncio.run(main())

``Tool``, ``ContentPart``, ``MCPResult``, and the custom provider adapters are
deprecated 2.x compatibility shims. New integrations should consume official
MCP types and framework-native MCP adapters; the shims are removed in 3.0.
"""

from .client import MCPClient
from ._errors import mcp_error_from_result, raise_for_mcp_result
from .tool import ContentPart, MCPResult, Tool, normalize_result

__all__ = [
    "ContentPart",
    "MCPClient",
    "MCPResult",
    "Tool",
    "mcp_error_from_result",
    "normalize_result",
    "raise_for_mcp_result",
]
