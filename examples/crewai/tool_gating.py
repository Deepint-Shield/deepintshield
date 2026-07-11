"""Gate CrewAI tools through the PDP - decide() runs before each tool body."""
from crewai.tools import tool

from deepintshield import DeepintShield


shield = DeepintShield.from_env()


@tool("write_ledger")
def write_ledger(row: str) -> str:
    """Append a row to the finance ledger."""
    return f"wrote {row}"


# Wrap the tool list; each invocation now passes through decide() first.
gated = shield.agentic.crewai([write_ledger])
print("gated tools:", [t.name for t in gated])
# Hand `gated` to your Agent(tools=gated, …) as usual.
