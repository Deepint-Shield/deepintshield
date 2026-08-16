"""PydanticAI's ordinary tool path is gated automatically through the PDP."""
from pydantic_ai import Agent

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
agent = Agent(shield.bind("pydanticai").model("gpt-4o-mini"))


@agent.tool_plain
def charge_card(amount: float) -> str:
    return f"charged {amount}"


print(agent.run_sync("charge 9.99 to the card").output)
