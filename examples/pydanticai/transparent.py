"""PydanticAI model obtained from the DeepintShield binder. Pass it to a
native pydantic_ai.Agent - the rest of your code is unchanged."""
from pydantic_ai import Agent

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
agent = Agent(
    shield.bind("pydanticai").model("gpt-4o-mini"),
    instructions="Be concise.",
)

print(agent.run_sync("hello from the PydanticAI binder").output)
