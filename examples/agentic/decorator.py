"""Gate any function as a tool. The gateway PDP decides before the body runs:
ALLOW runs it, MASK redacts PII kwargs, REQUIRE_APPROVAL blocks for a human,
DENY raises. Only the VK + base_url are configured - identity, policy and the
tool's tier (recovery cost / sensitivity) are resolved server-side from the
Tools & Tiering registry, so the only thing you pass is the tool name.

Prefer ``shield.agentic.guard()`` (see guard.py) for whole agents - this
decorator is the explicit, per-function form."""
from deepintshield import DeepintShield, GuardrailApprovalPending, GuardrailDenied


shield = DeepintShield.from_env()


@shield.agentic.tool("db.write")
def write_ledger(row: dict) -> dict:
    return {"inserted": row}


try:
    print("OK:", write_ledger({"amount": 12.5, "currency": "USD"}))
except GuardrailDenied as exc:
    print(f"DENIED: {exc.reason} (decision_id={exc.decision_id}, policy={exc.policy_id})")
except GuardrailApprovalPending as exc:
    print(f"PENDING human approval (decision_id={exc.decision_id}, approvers={exc.approvers})")
