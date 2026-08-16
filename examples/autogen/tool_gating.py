"""Native AutoGen tool execution with automatic Agentic enforcement."""

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
model_client = shield.bind("autogen").model_client("gpt-4o-mini")


async def wire_transfer(amount: float) -> str:
    return f"sent {amount}"


tool = FunctionTool(wire_transfer, description="Wire money to an account.")
agent = AssistantAgent("treasury", model_client=model_client, tools=[tool])


async def main() -> None:
    result = await agent.run(task="Use wire_transfer to send 10.00.")
    print(result.messages[-1].content)
    await model_client.close()


asyncio.run(main())
