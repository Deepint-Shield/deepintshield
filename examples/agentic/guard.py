"""One line gates every tool an agent calls - no per-tool code, no parameters.

``shield.agentic.guard()`` returns a native LangChain callback handler. LangChain
fires it before each tool runs; the PDP decides ALLOW / DENY / REQUIRE_APPROVAL,
and a block aborts the tool. The tool name comes from the framework and the
tier / policy / identity are all resolved server-side - the application only
attaches the handler.
"""
from langchain_core.tools import tool

from deepintshield import DeepintShield, GuardrailDenied


shield = DeepintShield.from_env()


@tool
def write_ledger(row: str) -> str:
    """Append a row to the finance ledger."""
    return f"wrote {row}"


# Attach the guard once; it covers every tool in the run (agents, chains,
# LangGraph). Here we invoke a single tool directly to show the gating.
guard = shield.agentic.guard()
try:
    print("OK:", write_ledger.invoke({"row": "amount=12.5"}, config={"callbacks": [guard]}))
except GuardrailDenied as exc:
    print(f"DENIED: {exc.reason} (decision_id={exc.decision_id}, policy={exc.policy_id})")

# With a real agent it is the same single line:
#   agent_executor.invoke({"input": "…"}, config={"callbacks": [guard]})
