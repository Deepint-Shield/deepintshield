"""The single decision-and-enforce core that every framework adapter and
the ``@shield.agentic.tool`` decorator funnel through.

Contract:
    * Build a DelegationContext from the args, compute the SHA-256 digest of
      the canonicalised args, and call ``engine.decide()``.
    * On ALLOW: return the (unchanged) kwargs.
    * On MASK: apply obligations to the kwargs and return them.
    * On canonical REQUIRE_APPROVAL: return control immediately with the GAF
      approval id. Polling exists only for an explicitly marked older-gateway
      compatibility decision.
    * On DENY: raise ``GuardrailDenied`` - the body must not run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import (
    AGENT_ACCESS_DENIED,
    AGENT_APPROVAL_PENDING,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
    normalize_agentic_error_code,
)
from .identity import obo_actor_chain
from .obligations import apply_obligations, digest
from .types import ContextBag, Decision, DelegationContext, Verdict

if TYPE_CHECKING:
    from .engine import AgenticEngine


_DENIAL_ERROR_CODES = frozenset(
    {
        "obo_delegation_required",
        "obo_no_acts_for",
        "obo_user_mismatch",
        "obo_scope_mismatch",
        "obo_user_lacks_perm",
        "obo_tool_not_allowed",
        "obo_action_not_allowed",
        "context_deny",
        "authz_engine_error",
        "no_store",
    }
)


def _denial_error_code(reason: str) -> str:
    candidate = normalize_agentic_error_code(reason, "")
    return candidate if candidate in _DENIAL_ERROR_CODES else AGENT_ACCESS_DENIED


def preflight(
    engine: "AgenticEngine",
    tool_name: str,
    tool_callable: Any = None,
) -> None:
    """Durably attest code before execution telemetry or credential lookup."""
    from .registry import ensure_registration_capture

    if not ensure_registration_capture(
        engine,
        tool_name,
        tool_callable=tool_callable,
    ):
        capture_code = normalize_agentic_error_code(
            getattr(engine, "_registration_capture_error_code", ""),
            "",
        )
        if capture_code != "agent_registration_quota_exceeded":
            capture_code = "blueprint_scan_unavailable"
        raise GovernanceConfigurationError(code=capture_code) from None


def resolve(
    engine: "AgenticEngine",
    tool_name: str,
    args: tuple,
    kwargs: dict,
    *,
    recovery_cost: str = "",
    rag_provenance: str = "",
    tool_fingerprint: str = "",
    tool_callable: Any = None,
    agent: str = "",
    permission: str = "",
    object: str = "",
    delegation_id: str = "",
    action: str = "",
    action_class: str = "",
    prompt: str = "",
) -> Decision:
    """Run a PDP decision for ``tool_name`` and raise on any blocking verdict.

    Returns the resolved :class:`Decision` (verdict ALLOW or MASK, with any
    obligations) so the caller can apply obligations to whatever argument
    shape it has. Raises :class:`GuardrailDenied` on DENY / denied approval
    and :class:`GuardrailApprovalPending` on approval timeout.
    """
    # Idempotent O(1)/local check when the caller already ran preflight before
    # entering its execution scope; direct adapter calls remain protected too.
    preflight(engine, tool_name, tool_callable)
    dc = DelegationContext(
        tool=tool_name,
        args_digest=digest(args, kwargs),
        virtual_key=engine.virtual_key,
        # Bind only to the agent identity authenticated for this virtual key.
        # Older gateways advertise no public subject, so send an empty principal
        # and let the data plane derive it from the VK; never invent a shared
        # ``agent:sdk`` subject.
        # ``shield.agentic.as_user(…)`` prepends the human's subject to the
        # chain so the on-behalf-of leg travels with every gated tool call.
        principal=engine.agent_subject,
        actor_chain=obo_actor_chain(engine),
        identity_type="application",
        context=ContextBag(
            recovery_cost=recovery_cost,
            rag_provenance=rag_provenance,
            tool_fingerprint=tool_fingerprint,
        ),
        agent=agent,
        permission=permission,
        object=object,
        delegation_id=delegation_id,
        action=action,
        action_class=action_class,
        prompt=prompt,
    )
    decision = engine.decide(dc)

    if decision.verdict == Verdict.DENY:
        raise GuardrailDenied(
            reason=decision.reason,
            decision_id=decision.decision_id,
            policy_id=decision.policy_id,
            tool=tool_name,
            code=_denial_error_code(decision.reason),
        )
    if decision.verdict == Verdict.REQUIRE_APPROVAL:
        if decision.gaf_checked:
            # Agentic-New's approval queue is an admin/session surface; there is
            # intentionally no VK-safe status-by-id endpoint to poll here. Return
            # control immediately so the caller can persist/resume the run. If
            # queue persistence failed and no approval id was returned, retain
            # the request id instead of falling into the legacy 5-minute poll.
            raise GuardrailApprovalPending(
                decision_id=decision.approval_id or decision.decision_id,
                approvers=list(decision.approvers),
                reason=decision.reason,
                code=AGENT_APPROVAL_PENDING,
            )
        try:
            resolved = engine.poll_approval(decision.decision_id)
        except GuardrailApprovalPending as error:
            raise GuardrailApprovalPending(
                decision_id=error.decision_id or decision.decision_id,
                approvers=list(decision.approvers),
                reason=decision.reason,
                code=error.code,
            ) from None
        except TimeoutError:
            # Compatibility for custom/older engine implementations that still
            # use the pre-code-contract timeout signal.
            raise GuardrailApprovalPending(
                decision_id=decision.decision_id,
                approvers=list(decision.approvers),
                reason=decision.reason,
                code=AGENT_APPROVAL_PENDING,
            ) from None
        if resolved.verdict == Verdict.DENY:
            raise GuardrailDenied(
                reason="approval denied",
                decision_id=decision.decision_id,
                policy_id=decision.policy_id,
                tool=tool_name,
                code=AGENT_ACCESS_DENIED,
            )
    return decision


def enforce(
    engine: "AgenticEngine",
    tool_name: str,
    args: tuple,
    kwargs: dict,
    *,
    recovery_cost: str = "",
    rag_provenance: str = "",
    tool_fingerprint: str = "",
    tool_callable: Any = None,
    agent: str = "",
    permission: str = "",
    object: str = "",
    delegation_id: str = "",
    action: str = "",
    action_class: str = "",
    prompt: str = "",
) -> dict[str, Any]:
    """``resolve`` + apply MASK obligations to ``kwargs``.

    Returns the (possibly masked) kwargs to forward to the tool body.
    """
    decision = resolve(
        engine,
        tool_name,
        args,
        kwargs,
        recovery_cost=recovery_cost,
        rag_provenance=rag_provenance,
        tool_fingerprint=tool_fingerprint,
        tool_callable=tool_callable,
        agent=agent,
        permission=permission,
        object=object,
        delegation_id=delegation_id,
        action=action,
        action_class=action_class,
        prompt=prompt,
    )
    return apply_obligations(kwargs, decision.obligations)


__all__ = ["preflight", "resolve", "enforce"]
