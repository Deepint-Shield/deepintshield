"""Load governed MCP tools with the maintained LangChain adapter."""
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from deepintshield import DeepintShield, DeepintShieldError


shield = DeepintShield.from_env()


async def enforce_deepintshield_result(request, handler):
    """Stop a canonical failed result before LangChain creates model content."""
    try:
        result = await handler(request)
    except Exception as exc:
        shield.mcp.raise_for_error(exc, operation="langchain_tool")
    return shield.mcp.raise_for_result(result)


async def main() -> None:
    try:
        url, headers = shield.mcp.connection()
        client = MultiServerMCPClient(
            {
                "deepintshield": {
                    "transport": "streamable_http",
                    "url": url,
                    "headers": headers,
                }
            },
            tool_interceptors=[enforce_deepintshield_result],
            handle_tool_errors=False,
        )
        try:
            tools = await client.get_tools()
        except Exception as exc:
            shield.mcp.raise_for_error(exc, operation="langchain_discovery")
        # These are native LangChain tools. Pass them unchanged to a LangChain
        # agent or a LangGraph ToolNode in the application that owns the model.
        print([tool.name for tool in tools])
    except DeepintShieldError as exc:
        if exc.code == "mcp_tool_approval_required":
            print("The MCP action is waiting for approval.")
        elif exc.code == "mcp_tool_authorization_denied":
            print("The MCP action was denied by policy.")
        elif exc.code == "mcp_tool_authorization_unavailable":
            print("MCP authorization is temporarily unavailable.")
        else:
            print(f"DeepIntShield error [{exc.code}]: {exc.description}")
    finally:
        shield.close()


if __name__ == "__main__":
    asyncio.run(main())
