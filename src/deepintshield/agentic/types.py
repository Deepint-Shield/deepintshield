"""Wire-shape types for the agentic (PDP) layer - the JSON contracts the
SDK exchanges with the gateway's agentic-security endpoints.

Kept Pydantic-only so the SDK doesn't pull in the framework's Go types or
any heavy schema lib.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Normalised input shared by the legacy PDP and Agentic-New GAF.

    The legacy-only fields are omitted from the GAF request and the GAF-only
    fields are omitted from the legacy request. Field names use snake_case to
    match both JSON wire contracts.
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
    # Agentic-New / OpenFGA inputs.  They are optional so existing
    # agentic-security callers remain wire-compatible.  The SDK derives
    # fail-closed defaults for governed tool calls:
    #   agent=<the selected Registry principal>, or empty for server selection
    #   permission=holder
    #   object=permission:v1-<tool>-<read|write>--<128-bit digest>
    #   tool=tool:<tool>
    agent: str = ""
    user: str = ""
    permission: str = ""
    object: str = ""
    delegation_id: str = ""
    # Named operation inside ``tool``. Empty deliberately preserves old
    # one-operation tools; when discovery registered exactly one action, the
    # server resolves it without trusting client-supplied impact metadata.
    action: str = ""
    action_class: str = ""


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
    # Canonical-source metadata. Whenever ``gaf_checked`` is true, ``verdict``
    # is derived solely from this same GAF response; no hidden PDP can replace
    # it. ``gaf_checked`` is false only for the explicitly marked
    # ``legacy-compat`` result returned by a genuinely older gateway.
    # ``gaf_verdict`` retains the raw shadow-mode answer while ``verdict`` uses
    # the canonical ``proceed`` bit.
    gaf_checked: bool = False
    gaf_verdict: str = ""
    failed_check: str = ""
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    proceed: Optional[bool] = None
    would_block: bool = False
    approval_id: str = ""


class GAFDecision(BaseModel):
    """Response from ``POST /api/agentic-new/decide``."""

    verdict: Verdict
    decision_id: str = ""
    allow: bool = False
    reason: str = ""
    failed_check: str = ""
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    latency_us: int = 0
    mode: str = ""
    would_block: bool = False
    proceed: Optional[bool] = None
    approval_id: str = ""


class VKCredentialInfo(BaseModel):
    """Public discovery info the SDK fetches once from the canonical
    ``GET /api/agentic-new/credential-info`` route (with the legacy
    ``/api/agentic-security/vk-credential-info`` route as an older-gateway
    fallback).

    Contains NO secrets - only the OIDC discovery values the SDK needs to
    build the right AgentCredential implementation for the identity provider
    on the server-selected Registry profile.
    """

    provider_id: str = ""
    provider_type: str = ""
    tenant_id: str = ""
    blueprint_client_id: str = ""
    agent_identity_client_id: str = ""
    authority: str = ""
    gateway_audience: str = ""
    scopes: List[str] = Field(default_factory=list)
    fic_audience: str = "api://AzureADTokenExchange"
    exchange_endpoint: str = ""
    allow_cross_tenant: bool = False
    agent_configured: bool = False
    # Canonical Agentic-New principal selected from the authenticated virtual
    # key's Registry associations. Empty on older gateways; in that case the SDK
    # omits its claim and lets the data plane select from authenticated context.
    agent_subject: str = ""


__all__ = ["Verdict", "Decision", "DelegationContext", "ContextBag", "VKCredentialInfo"]
