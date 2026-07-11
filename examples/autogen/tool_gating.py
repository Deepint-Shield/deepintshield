"""Gate an AutoGen FunctionTool through the PDP - decide() runs before the
underlying callable executes."""
from autogen_core.tools import FunctionTool

from deepintshield import DeepintShield


shield = DeepintShield.from_env()


async def wire_transfer(amount: float) -> str:
    return f"sent {amount}"


tool = FunctionTool(wire_transfer, description="Wire money to an account.")
shield.agentic.autogen(tool)  # gate the tool's callable in place
print("gated tool:", tool.name)
# Register `tool` with your AssistantAgent(tools=[tool], …) as usual.
