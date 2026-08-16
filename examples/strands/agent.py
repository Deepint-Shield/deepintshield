"""AWS Strands agent governed automatically at final tool dispatch.

Constructing ``DeepintShield`` arms Strands' final ``ToolExecutor`` boundary;
ordinary agent calls are gated without a hook argument or per-tool wrapper.
Point ``OpenAIModel(base_url=...)`` at the gateway for the LLM leg
(cache/guardrails/cost).

    pip install 'deepintshield[strands]'   # strands-agents
"""
from strands import Agent, tool

from deepintshield import DeepintShield

shield = DeepintShield.from_env()


@tool
def crm_read(customer_id: str) -> dict:
    """A tool - governed by its function name ``crm_read``."""
    return {"customer_id": customer_id, "name": "Ada"}


agent = Agent(
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    tools=[crm_read],
)
print(agent("Look up customer 42"))
