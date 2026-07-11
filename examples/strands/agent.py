"""AWS Strands agent governed by the PDP with one hook provider.

Strands' typed hook system exposes ``BeforeToolInvocationEvent``; the
DeepintShield ``HookProvider`` gates every tool invocation there and cancels the
call on a DENY. Point ``OpenAIModel(base_url=...)`` at the gateway for the LLM
leg (cache/guardrails/cost).

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
    hooks=[shield.agentic.strands()],  # ← one line gates every tool invocation
)
print(agent("Look up customer 42"))
