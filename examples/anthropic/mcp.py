"""Anthropic tool use with its official MCP helper through DeepIntShield.

Anthropic owns MCP-to-Messages schema and result conversion. Dispatch remains
explicit because the released Messages tool runner converts every tool
exception into model-visible content; direct ``tool.call()`` preserves the
coded DeepIntShield authorization exception.
"""
import asyncio
import os

from anthropic import AsyncAnthropic
from anthropic.lib.tools.mcp import async_mcp_tool
from deepintshield import DeepintShield, DeepintShieldError


shield = DeepintShield.from_env()
model = os.getenv("DEEPINTSHIELD_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")


async def main() -> None:
    try:
        provider_url, provider_headers = shield.connection(provider="anthropic")
        async with AsyncAnthropic(
            base_url=provider_url,
            api_key=shield.api_key(),
            default_headers=provider_headers,
        ) as anthropic:
            # ``session`` is the official mcp.ClientSession, already initialized.
            async with shield.mcp.connect() as session:
                listing = await session.list_tools()
                mcp_tools = [async_mcp_tool(tool, session) for tool in listing.tools]
                tools_by_name = {tool.name: tool for tool in mcp_tools}
                tool_definitions = [tool.to_dict() for tool in mcp_tools]
                messages = [
                    {
                        "role": "user",
                        "content": "Use DeepWiki to explain Suspense in facebook/react.",
                    }
                ]
                response = await anthropic.beta.messages.create(
                    model=model,
                    max_tokens=1024,
                    tools=tool_definitions,
                    messages=messages,
                )

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    content = await tools_by_name[block.name].call(block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        }
                    )

                if not tool_results:
                    for block in response.content:
                        if block.type == "text":
                            print(block.text)
                    return

                messages.append(
                    {
                        "role": "assistant",
                        "content": [block.model_dump() for block in response.content],
                    }
                )
                messages.append({"role": "user", "content": tool_results})
                final = await anthropic.beta.messages.create(
                    model=model,
                    max_tokens=1024,
                    tools=tool_definitions,
                    messages=messages,
                )
                for block in final.content:
                    if block.type == "text":
                        print(block.text)
    except DeepintShieldError as exc:
        if exc.code == "mcp_tool_approval_required":
            print("The MCP action is waiting for approval.")
        elif exc.code == "mcp_tool_authorization_denied":
            print("The MCP action was denied by policy.")
        elif exc.code == "mcp_tool_authorization_unavailable":
            print("MCP authorization is temporarily unavailable.")
        else:
            print(f"DeepIntShield error [{exc.code}]: {exc.description}")


if __name__ == "__main__":
    asyncio.run(main())
