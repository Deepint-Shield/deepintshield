"""Google ADK agent governed automatically at final tool dispatch.

Constructing ``DeepintShield`` arms ADK's normal, live and threaded dispatch
functions. Every agent and tool is governed without adding a plugin to the
runner.

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
runner = InMemoryRunner(agent=agent)
# Drive `runner` as usual; each tool call passes through the PDP first.
