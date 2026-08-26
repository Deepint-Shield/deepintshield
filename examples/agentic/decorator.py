"""Gate any function as a tool. The gateway PDP decides before the body runs:
ALLOW runs it, MASK runs it with the decision's obligations applied to the
arguments, REQUIRE_APPROVAL blocks for a human, DENY raises.

Redaction covers positional and keyword arguments alike, so the calling
convention cannot opt out of an obligation - see obligations.py for the
handlers (mask:pii, redact:secrets, redact:phi, redact:card-numbers,
redact:bank-accounts, redact:value).

You configure the VK, the base_url, and an agent_name; identity, policy and the
tool's tier (recovery cost / sensitivity) are resolved server-side from the
Tools & Tiering registry, so the only thing you pass here is the tool name.
DEEPINTSHIELD_AGENT_NAME is required for anything agentic and has no default.

Native framework execution is preferred for whole agents; this decorator is the
advanced, explicit per-function form."""
from deepintshield import DeepintShield


shield = DeepintShield.from_env()


@shield.agentic.tool("db.write")
def write_ledger(row: dict) -> dict:
    return {"inserted": row}


print(write_ledger({"amount": 12.5, "currency": "USD"}))
