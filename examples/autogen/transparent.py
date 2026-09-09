"""AutoGen (AG2) with its model client routed through DeepintShield."""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
model_client = shield.bind("autogen").model_client(os.getenv("DEEPINTSHIELD_MODEL", "openai/gpt-4o-mini"))

agent = AssistantAgent("assistant", model_client=model_client)


async def main() -> None:
    try:
        result = await agent.run(task="Say hello from AutoGen via DeepintShield.")
        print(result.messages[-1].content)
    finally:
        await model_client.close()


asyncio.run(main())
