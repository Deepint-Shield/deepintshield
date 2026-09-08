"""OpenAI Agents MCP integration through DeepIntShield's governed endpoint.

OpenAI Agents owns the Streamable HTTP session, discovery, tool conversion,
and agent loop. Its public result and failure hooks translate MCP tool-call
failures to stable codes before the SDK can create model-visible tool output.
"""
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

from deepintshield import DeepintShield, DeepintShieldError


shield = DeepintShield.from_env()
model = os.getenv("DEEPINTSHIELD_MODEL", "gpt-4o-mini")


def enforce_deepintshield_result(context):
    """Translate the raw MCP result before OpenAI Agents emits tool output."""
    tool_output = context.tool_output
    if isinstance(tool_output, list):
        content_blocks = tool_output
    elif isinstance(tool_output, str):
        content_blocks = [{"type": "text", "text": tool_output}]
    else:
        content_blocks = [tool_output]
    shield.mcp.raise_for_result(
        {
            "isError": context.is_error,
            "_meta": context.result_meta,
            "structuredContent": context.structured_content,
            "content": content_blocks,
        }
    )
    return None


def enforce_deepintshield_exception(_context, error):
    """Translate an upstream MCP exception instead of formatting model text."""
    shield.mcp.raise_for_error(error, operation="openai_agents_tool")


async def main() -> None:
    model_client = None
    try:
        model_client = shield.bind("openai_agents").apply()
        url, headers = shield.mcp.connection()
        async with MCPServerStreamableHttp(
            name="DeepIntShield",
            params={"url": url, "headers": headers},
            cache_tools_list=True,
            custom_data_extractor=enforce_deepintshield_result,
            failure_error_function=enforce_deepintshield_exception,
        ) as server:
            agent = Agent(
                name="MCP Assistant",
                instructions="Use the governed MCP tools when they help.",
                model=model,
                mcp_servers=[server],
            )
            result = await Runner.run(
                agent,
                "Use DeepWiki to explain Suspense in facebook/react.",
            )
            print(result.final_output)
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
        try:
            if model_client is not None:
                await model_client.close()
        finally:
            shield.close()


if __name__ == "__main__":
    asyncio.run(main())
