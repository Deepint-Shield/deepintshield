"""DeepintShield agentic (PDP) layer.

The tool-enforcement half of the SDK: ``decide()`` runs before a gated tool
executes and the verdict (ALLOW / DENY / MASK / REQUIRE_APPROVAL) maps to a
return value or an exception. All Entra / identity / policy machinery is
auto-discovered from the gateway - the user passes only ``virtual_key`` and
``base_url`` to :class:`~deepintshield.client.DeepintShield`.

Ordinary framework applications only create one client; DeepIntShield installs
the supported framework's native compile/run/tool guards automatically::

    shield = DeepintShield.from_env()

    builder = StateGraph(State)   # native LangGraph code
    ...
    app = builder.compile()
    app.invoke(input_state)

``shield.agentic`` and the native framework adapters are the supported
application boundary. They expose marked standard Python exceptions with
trusted guidance. ``AgenticEngine``, direct identity/credential primitives,
and the typed Agentic exception exports are retained as low-level SDK control
primitives and for backward compatibility; new application code should not
call or catch those primitives directly.
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
from .execution import (
    ExecutionFrame,
    current_execution_id,
    execution_scope,
    mark_execution_outcome,
)
from .errors import (
    DeepIntShieldError,
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
    GuardrailMasked,
)
from .identity import PrincipalBinding, local_subject, resolve_principal
from .registry import describe_network, discover
from .surface import AgenticSurface
from .types import (
    ContextBag,
    Decision,
    DelegationContext,
    GAFDecision,
    Verdict,
    VKCredentialInfo,
)

__all__ = [
    "AgenticSurface",
    "AgenticEngine",
    "ExecutionFrame",
    "execution_scope",
    "current_execution_id",
    "mark_execution_outcome",
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
    "GovernanceConfigurationError",
    "Decision",
    "GAFDecision",
    "DelegationContext",
    "ContextBag",
    "Verdict",
    "VKCredentialInfo",
    # GAF directory + registry (agentic-new)
    "resolve_principal",
    "PrincipalBinding",
    "local_subject",
    "describe_network",
    "discover",
]
