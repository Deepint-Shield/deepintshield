"""Generic MCP support for DeepintShield.

Works with any MCP server connected to your DeepintShield gateway. No
per-server code anywhere; per-framework adapters are thin format translators.

Quick start
-----------

    from deepintshield import DeepintShield
    from deepintshield.mcp import Tool

    shield = DeepintShield(virtual_key="sk-bf-...", base_url="http://localhost:8080")

    # Direct call - works for any tool on any connected server.
    result = shield.mcp.call(
        server="DeepWiki",
        tool="ask_question",
        repoName="facebook/react",
        question="What is Suspense?",
    )
    print(result.text)

    # OpenAI loop
    tools = [Tool(server="DeepWiki", name="ask_question",
                  description="...", schema={...})]
    openai_tools = shield.mcp.to_openai(tools)
"""

from .client import MCPClient
from .tool import ContentPart, MCPResult, Tool, normalize_result

__all__ = [
    "ContentPart",
    "MCPClient",
    "MCPResult",
    "Tool",
    "normalize_result",
]
