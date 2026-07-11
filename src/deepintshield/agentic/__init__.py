"""DeepintShield agentic (PDP) layer.

The tool-enforcement half of the SDK: ``decide()`` runs before a gated tool
executes and the verdict (ALLOW / DENY / MASK / REQUIRE_APPROVAL) maps to a
return value or an exception. All Entra / identity / policy machinery is
auto-discovered from the gateway - the user passes only ``virtual_key`` and
``base_url`` to :class:`~deepintshield.client.DeepintShield`.

Typically consumed via ``shield.agentic`` rather than imported directly::

    shield = DeepintShield(virtual_key="sk-bf-…", base_url="https://gw…")

    @shield.agentic.tool("db.write", recovery_cost="high")
    def write_ledger(row: dict) -> dict:
        ...

    shield.agentic.langgraph(compiled_graph)   # wrap a whole framework graph
"""

from .credentials import (
    AgentCredential,
    EntraAgentCredential,
    OIDCCredential,
    StaticAgentCredential,
    ZeroIDCredential,
)
from .decorators import set_default_client, shield_tool
from .engine import AgenticEngine
from .errors import (
    DeepIntShieldError,
    GatewayUnavailable,
    GuardrailApprovalPending,
    GuardrailDenied,
    GuardrailMasked,
)
from .surface import AgenticSurface
from .types import ContextBag, Decision, DelegationContext, Verdict, VKCredentialInfo

__all__ = [
    "AgenticSurface",
    "AgenticEngine",
    "shield_tool",
    "set_default_client",
    "AgentCredential",
    "EntraAgentCredential",
    "ZeroIDCredential",
    "OIDCCredential",
    "StaticAgentCredential",
    "DeepIntShieldError",
    "GuardrailDenied",
    "GuardrailApprovalPending",
    "GuardrailMasked",
    "GatewayUnavailable",
    "Decision",
    "DelegationContext",
    "ContextBag",
    "Verdict",
    "VKCredentialInfo",
]
