"""Startup healthcheck: refuse to serve traffic this workload cannot get past.

An agent that is pending, denied or quarantined is not governed yet, and every
governed call will be denied. Discovering that one authorization failure at a
time - possibly minutes into a deployment - is the problem this closes.

status() reads the workload's own enrolment state from the gateway. It never
raises and never blocks a governed call: it is a read for your own healthcheck,
and it reports only this key's lifecycle state.

Requires DEEPINTSHIELD_AGENT_NAME - the name IS the identity being looked up."""
import sys

from deepintshield import DeepintShield


shield = DeepintShield.from_env()
state = shield.agentic.status()

print("state         :", state.get("state"))
print("agent_subject :", state.get("agent_subject") or "-")
print("reason        :", state.get("reason") or "-")
print("action        :", state.get("action") or "-")
print("review_url    :", state.get("review_url") or "-")

# live            - enrolled and enabled; governed calls are decided normally
# pending         - waiting for an operator in Work Queue
# approved        - approved, but the governance profile is not enabled yet
# denied          - enrol under a different agent name, or ask for a supersede
# quarantined     - code changed; review the blueprint scan to restore it
# not_registered  - no registration on this virtual key yet
# unknown         - the gateway could not be reached; `reason` has the detail
if state.get("state") != "live":
    print("\nThis workload is not governed yet; governed calls will be denied.")
    sys.exit(1)
