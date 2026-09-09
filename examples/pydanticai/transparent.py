"""PydanticAI model obtained from the DeepintShield binder. Pass it to a
native pydantic_ai.Agent - the rest of your code is unchanged."""
import os

from pydantic_ai import Agent

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
agent = Agent(
    shield.bind("pydanticai").model(os.getenv("DEEPINTSHIELD_MODEL", "openai/gpt-4o-mini")),
    instructions="Be concise.",
)

print(agent.run_sync("hello from the PydanticAI binder").output)
