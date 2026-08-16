from pydantic_ai import Agent

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
agent = Agent(
    shield.bind("pydanticai").model("gpt-4o-mini"),
    instructions="Be concise and helpful.",
)

result = agent.run_sync("Say hello from PydanticAI via DeepintShield.")
print(result.output)
