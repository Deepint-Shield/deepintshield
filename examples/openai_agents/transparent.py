"""Native OpenAI Agents model, Agent and Runner through the gateway."""
import asyncio
import os

from agents import Agent, Runner

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
agent = Agent(
    name="Assistant", instructions="Be concise.",
    model=shield.bind("openai_agents").model(os.getenv("DEEPINTSHIELD_MODEL", "openai/gpt-4o-mini")),
)
result = asyncio.run(Runner.run(agent, "Say hi from the OpenAI Agents SDK via DeepintShield."))
print(result.final_output)
