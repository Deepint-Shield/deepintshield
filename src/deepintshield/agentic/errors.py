"""Agentic-layer exception hierarchy.

Each PDP verdict maps to a distinct exception so the caller can pattern-
match the kind of failure without inspecting strings. Audit metadata
(decision_id, policy_id, reason, obligations) is preserved on the exception
so the caller can log it without re-fetching from the audit endpoint.

All agentic errors subclass the unified ``DeepintShieldError`` so a single
``except DeepintShieldError`` catches both LLM-stage guardrail failures and
PDP rejections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from ..errors import DeepintShieldError


class DeepIntShieldError(DeepintShieldError):
    """Base for agentic-layer errors. Kept as a distinct name for backwards
    compatibility with code migrated from ``deepintshield_agents`` while
    inheriting from the package-wide :class:`DeepintShieldError`."""


@dataclass
class GuardrailDenied(DeepIntShieldError):
    """Raised when the PDP returns DENY. The call did not execute.

    Always a hard failure - do NOT retry without a policy change. The audit
    row is already persisted server-side.
    """

    reason: str = ""
    decision_id: str = ""
    policy_id: str = ""
    tool: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"DENY ({self.reason}; policy={self.policy_id}; decision={self.decision_id})"


@dataclass
class GuardrailApprovalPending(DeepIntShieldError):
    """Raised when the PDP returns REQUIRE_APPROVAL and the SDK's poll
    timeout expires before a human decides.

    The decision_id is durable - the caller can poll
    GET /api/agentic-security/approvals later, or be notified via the
    platform's webhook integration.
    """

    decision_id: str = ""
    approvers: List[str] = field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return f"REQUIRE_APPROVAL (decision={self.decision_id}; approvers={self.approvers})"


@dataclass
class GuardrailMasked(DeepIntShieldError):
    """Raised informationally when the PDP returns MASK + obligations the SDK
    does not know how to apply locally.

    Default behaviour: known obligations (``mask:pii``) are applied
    transparently to the arguments before the tool runs. Anything unknown
    surfaces here so the caller can decide.
    """

    obligations: List[str] = field(default_factory=list)
    value: Any = None  # the tool's return value (if it ran)


@dataclass
class GatewayUnavailable(DeepIntShieldError):
    """Raised when the gateway is unreachable. Distinct from a DENY -
    indicates infrastructure trouble, not a policy verdict."""

    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return f"GATEWAY_UNAVAILABLE ({self.reason})"


__all__ = [
    "DeepIntShieldError",
    "GuardrailDenied",
    "GuardrailApprovalPending",
    "GuardrailMasked",
    "GatewayUnavailable",
]
