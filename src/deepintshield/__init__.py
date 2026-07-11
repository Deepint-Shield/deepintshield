"""
DeepintShield - unified Python SDK.

Quick start
-----------

    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-...")
    openai_client = shield.openai()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )

Traffic defaults to ``https://app.deepintshield.com``. Override per-call with
``DeepintShield(base_url=...)`` or via the ``DEEPINTSHIELD_BASE_URL``
environment variable when using ``DeepintShield.from_env()``.
"""

from .agentic import (
    AgenticSurface,
    ContextBag,
    Decision,
    DelegationContext,
    GatewayUnavailable,
    GuardrailApprovalPending,
    GuardrailDenied,
    GuardrailMasked,
    Verdict,
    VKCredentialInfo,
    set_default_client,
    shield_tool,
)
from .client import DeepintShield
from .config import DEFAULT_BASE_URL, ShieldConfig
from .errors import DeepintShieldBlockedError, DeepintShieldError
from .mcp import ContentPart, MCPClient, MCPResult, Tool
from .rag import allowed_chunk_ids, build_chunk, filter_chunks
from .types import (
    NON_BLOCKING_DECISIONS,
    GuardrailDecision,
    GuardrailResult,
    GuardrailStage,
    RetrievedChunk,
    ToolInvocation,
)
from .version import __version__

# Backwards-compatible alias.
DeepintShieldClient = DeepintShield

__all__ = [
    "__version__",
    "DEFAULT_BASE_URL",
    "ContentPart",
    "DeepintShield",
    "DeepintShieldClient",
    "DeepintShieldBlockedError",
    "DeepintShieldError",
    "GuardrailDecision",
    "GuardrailResult",
    "GuardrailStage",
    "MCPClient",
    "MCPResult",
    "NON_BLOCKING_DECISIONS",
    "RetrievedChunk",
    "ShieldConfig",
    "Tool",
    "ToolInvocation",
    "allowed_chunk_ids",
    "build_chunk",
    "filter_chunks",
    # ── agentic (PDP) layer ──
    "AgenticSurface",
    "shield_tool",
    "set_default_client",
    "Verdict",
    "Decision",
    "DelegationContext",
    "ContextBag",
    "VKCredentialInfo",
    "GuardrailDenied",
    "GuardrailApprovalPending",
    "GuardrailMasked",
    "GatewayUnavailable",
]
