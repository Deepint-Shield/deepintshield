"""Google ADK agent governed by the PDP with one app-level plugin.

A single ``BasePlugin`` on the runner gates EVERY agent and tool - the closest
analog to ``govern()``. ``before_tool_callback`` calls the PDP; a DENY returns a
short-circuit dict the model sees instead of the tool output.

    pip install 'deepintshield[google-adk]'   # google-adk
"""
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from deepintshield import DeepintShield

shield = DeepintShield.from_env()


def crm_read(customer_id: str) -> dict:
    """A tool - governed by its function name ``crm_read``."""
    return {"customer_id": customer_id, "name": "Ada"}


agent = Agent(name="assistant", model="gemini-2.0-flash", tools=[crm_read])

runner = InMemoryRunner(
    agent=agent,
    plugins=[shield.agentic.google_adk()],  # ← one line gates every tool app-wide
)
# Drive `runner` as usual; each tool call passes through the PDP first.
