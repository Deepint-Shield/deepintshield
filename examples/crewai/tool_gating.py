"""Native CrewAI tool execution with automatic Agentic enforcement."""
from crewai import Agent, Crew, Task
from crewai.tools import tool

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
llm = shield.bind("crewai").llm("gpt-4o-mini")


@tool("write_ledger")
def write_ledger(row: str) -> str:
    """Append a row to the finance ledger."""
    return f"wrote {row}"


accountant = Agent(
    role="Accountant",
    goal="Record the requested ledger row",
    backstory="You maintain the finance ledger.",
    llm=llm,
    tools=[write_ledger],
    allow_delegation=False,
)
task = Task(
    description="Use write_ledger to record amount=12.50,currency=USD.",
    expected_output="Confirmation that the row was recorded.",
    agent=accountant,
)

print(Crew(agents=[accountant], tasks=[task]).kickoff())
