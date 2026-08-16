"""OpenAI Agents SDK - register the gateway client as the SDK default, then
write completely native Agent/Runner code."""
import asyncio

from agents import Agent, Runner

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
shield.bind("openai_agents").apply()

agent = Agent(name="Assistant", instructions="Be concise.")
result = asyncio.run(Runner.run(agent, "Say hi from the OpenAI Agents SDK via DeepintShield."))
print(result.final_output)
