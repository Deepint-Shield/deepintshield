"""Direct PDP probe - fetch the raw verdict without raising. Useful for custom
control flow or diagnostics."""
from deepintshield import DeepintShield


shield = DeepintShield.from_env()

decision = shield.agentic.decide(tool="finance.report", args={"month": "2026-05"})
print("verdict     :", decision.verdict.value)
print("reason      :", decision.reason or "-")
print("obligations :", decision.obligations or "-")
print("cache_hit   :", decision.cache_hit)
print("decision_id :", decision.decision_id)
