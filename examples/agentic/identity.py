"""Inspect the agent-identity binding the gateway resolved for this VK.

No Entra / ZeroID / OIDC GUIDs appear in code - the SDK discovers the binding
from /api/agentic-security/vk-credential-info and, on Azure compute, detects
the Managed Identity automatically. This is purely for ops visibility."""
from deepintshield import DeepintShield


shield = DeepintShield.from_env()
info = shield.agentic.credential_info

print("provider_type    :", info.provider_type or "(none)")
print("agent_configured :", info.agent_configured)
print("tenant_id        :", info.tenant_id or "(none)")
print("blueprint_client :", info.blueprint_client_id or "(none)")
print("authority        :", info.authority or "(none)")
print("scopes           :", info.scopes or "(none)")
