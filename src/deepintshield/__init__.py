"""
DeepintShield - unified Python SDK.

Quick start
-----------

    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-ds-your-virtual-key")
    openai_client = shield.openai()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )

Traffic defaults to ``https://app.deepintshield.com``. Override per-call with
``DeepintShield(base_url=...)`` or via the ``DEEPINTSHIELD_BASE_URL``
environment variable when using ``DeepintShield.from_env()``. The former
``DEEPINTSHIELD_GATEWAY_URL`` name remains a fallback alias.
"""

from .agentic import (
    AgenticSurface,
    ContextBag,
    Decision,
    DelegationContext,
    GAFDecision,
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
    GuardrailMasked,
    PrincipalBinding,
    Verdict,
    VKCredentialInfo,
    set_default_client,
    shield_tool,
)
from .client import DeepintShield
from .config import DEFAULT_BASE_URL, ShieldConfig
from .errors import (
    ERROR_CATALOG,
    DeepintShieldBlockedError,
    DeepintShieldError,
    ErrorCategory,
    ErrorCode,
    ErrorDefinition,
    get_error_definition,
    get_exception_error_code,
    iter_error_definitions,
)
from .mcp import ContentPart, MCPClient, MCPResult, Tool
from .rag import allowed_chunk_ids, build_chunk, filter_chunks
from .streaming import ChatCompletionStream
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
    "ChatCompletionStream",
    "DeepintShield",
    "DeepintShieldClient",
    "DeepintShieldBlockedError",
    "DeepintShieldError",
    "ERROR_CATALOG",
    "ErrorCategory",
    "ErrorCode",
    "ErrorDefinition",
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
    "get_error_definition",
    "get_exception_error_code",
    "iter_error_definitions",
    # ── agentic (PDP) layer ──
    "AgenticSurface",
    "shield_tool",
    "set_default_client",
    "Verdict",
    "Decision",
    "GAFDecision",
    "DelegationContext",
    "PrincipalBinding",
    "ContextBag",
    "VKCredentialInfo",
    "GuardrailDenied",
    "GuardrailApprovalPending",
    "GuardrailMasked",
    "GatewayUnavailable",
    "GovernanceConfigurationError",
]
