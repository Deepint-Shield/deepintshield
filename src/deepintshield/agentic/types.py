"""Wire-shape types for the agentic (PDP) layer - the JSON contracts the
SDK exchanges with the gateway's agentic-security endpoints.

Kept Pydantic-only so the SDK doesn't pull in the framework's Go types or
any heavy schema lib.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Closed set of decisions the PDP returns."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    MASK = "MASK"


class ContextBag(BaseModel):
    """ABAC inputs at decide-time."""

    rag_provenance: str = ""
    cost_used: float = 0.0
    recovery_cost: str = ""
    # "src:<sha256[:16]>" of the tool's implementation - binds the decision to
    # the actual code (server folds it into the cache key + feeds the code-threat
    # scan). Empty for callers that don't supply it (zero impact).
    tool_fingerprint: str = ""
    # OWASP-gap ABAC signals (optional; all default off so they never perturb the
    # cache). memory_integrity (T1) - agent memory failed validation;
    # hallucination_risk (T5) - 0..1 faithfulness risk; goal_drift (T7) - behaviour
    # diverged from the declared goal; comm_integrity (T12) - inter-agent message
    # failed auth. (delegation_depth (T14) is computed server-side from the chain.)
    memory_integrity: bool = False
    hallucination_risk: float = 0.0
    goal_drift: bool = False
    comm_integrity: bool = False
    # output_manipulation (T15) - the agent's response tripped the output guardrail
    # (injected link / fraudulent instruction). (approval_pressure (T10) is computed
    # server-side from the workspace's approval volume.)
    output_manipulation: bool = False


class DelegationContext(BaseModel):
    """Normalised input to the PDP.

    Mirrors the Go-side ``agentic.DelegationContext`` 1:1. Field names use
    snake_case to match the JSON wire format; the SDK never produces or
    consumes anything else.
    """

    principal: str = ""
    actor_chain: List[str] = Field(default_factory=list)
    identity_type: str = ""
    scope: List[str] = Field(default_factory=list)
    tenant: str = ""
    workspace: str = ""
    virtual_key: str = ""
    # Optional run/session id. When set (the SDK stamps a per-process one), the
    # gateway groups every step of the run into one Agent Execution; a new
    # session id starts a new execution. Observability grouping only.
    session_id: str = ""
    # Optional agent prompt/instruction, supplied for SCAN ONLY. The gateway runs
    # it through the prompt guardrail (injection / PII) at the PDP boundary and
    # then discards it - it is never stored (zero-data-retention). Omit it and the
    # decision is unchanged; supply it to have a malicious instruction that drives
    # this tool call caught inline (sets prompt_injection / prompt_pii signals).
    prompt: str = ""
    allowed_tools: List[str] = Field(default_factory=list)
    cross_tenant: bool = False
    tool: str
    args_digest: str
    provider_id: str = ""
    policy_version: int = 0
    context: ContextBag = Field(default_factory=ContextBag)


class Decision(BaseModel):
    """PDP output. Forwarded by the SDK to the caller, mostly via exception
    classes (see deepintshield.agentic.errors)."""

    verdict: Verdict
    reason: str = ""
    approvers: List[str] = Field(default_factory=list)
    obligations: List[str] = Field(default_factory=list)
    policy_id: str = ""
    decision_id: str
    mode: str = ""
    cache_hit: bool = False
    latency_us: int = 0


class VKCredentialInfo(BaseModel):
    """Public discovery info the SDK fetches once from
    GET /api/agentic-security/vk-credential-info.

    Contains NO secrets - only the OIDC discovery values the SDK needs to
    build the right AgentCredential implementation for whichever identity
    provider the VK is bound to.
    """

    provider_type: str = ""
    tenant_id: str = ""
    blueprint_client_id: str = ""
    authority: str = ""
    gateway_audience: str = ""
    scopes: List[str] = Field(default_factory=list)
    fic_audience: str = "api://AzureADTokenExchange"
    exchange_endpoint: str = ""
    allow_cross_tenant: bool = False
    agent_configured: bool = False


__all__ = ["Verdict", "Decision", "DelegationContext", "ContextBag", "VKCredentialInfo"]
