"""MASK verdicts: the tool runs, but not with the arguments you passed.

A verdict can ALLOW a call *and* attach an obligation. The SDK applies the
obligations it can carry out locally before the function body sees its
arguments, so the tool never handles the protected value in the clear.

The point to notice is that redaction covers POSITIONAL arguments too. Masking
only kwargs made the obligation optional in practice, because the caller picks
the calling convention: send(email) would have leaked what send(email=...)
redacted. Both are protected identically now.

The argument digest sent to the PDP is computed from the original call, so
redaction never invalidates an approval pinned to that digest. An obligation
the SDK does not recognise is a local no-op and stays the gateway's to enforce.

This file shows the handlers directly - no gateway needed - so you can see what
each obligation does before wiring one into a policy."""
from deepintshield.agentic.obligations import apply_obligations

# mask:pii matches field names EXACTLY - "email", not "customer_email".
CALL = {
    "email": "ada@example.com",
    "api_key": "sk-live-9f2c41d7a8b3e5061c4d",
    "note": "rotate before Friday",
    "card": "4111 1111 1111 1111",
    "order_ref": "4400000000000000124",   # long, but not a card - left intact
    "iban": "GB33BUKB20201555555555",
    "diagnosis": "E11.9",
    "value": "hunter2",
}

for obligation in (
    "mask:pii",
    "redact:secrets",
    "redact:phi",
    "redact:card-numbers",
    "redact:bank-accounts",
    "redact:value",
):
    changed = {
        k: v for k, v in apply_obligations(CALL, [obligation]).items()
        if v != CALL[k]
    }
    print(f"{obligation:<22} -> {changed or '(nothing matched)'}")

print("\nAll together:")
print(" ", apply_obligations(CALL, ["mask:pii", "redact:secrets", "redact:card-numbers"]))
