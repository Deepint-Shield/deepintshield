"""The single decision-and-enforce core that every framework adapter and
the ``@shield.agentic.tool`` decorator funnel through.

Contract:
    * Build a DelegationContext from the args, compute the SHA-256 digest of
      the canonicalised args, and call ``engine.decide()``.
    * On ALLOW: return the (unchanged) kwargs.
    * On MASK: apply obligations to the kwargs and return them.
    * On REQUIRE_APPROVAL: poll the approval endpoint; continue on approval,
      raise on denial/timeout.
    * On DENY: raise ``GuardrailDenied`` - the body must not run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import GuardrailApprovalPending, GuardrailDenied
from .obligations import apply_obligations, digest
from .types import ContextBag, Decision, DelegationContext, Verdict

if TYPE_CHECKING:
    from .engine import AgenticEngine


def resolve(
    engine: "AgenticEngine",
    tool_name: str,
    args: tuple,
    kwargs: dict,
    *,
    recovery_cost: str = "",
    rag_provenance: str = "",
    tool_fingerprint: str = "",
) -> Decision:
    """Run a PDP decision for ``tool_name`` and raise on any blocking verdict.

    Returns the resolved :class:`Decision` (verdict ALLOW or MASK, with any
    obligations) so the caller can apply obligations to whatever argument
    shape it has. Raises :class:`GuardrailDenied` on DENY / denied approval
    and :class:`GuardrailApprovalPending` on approval timeout.
    """
    dc = DelegationContext(
        tool=tool_name,
        args_digest=digest(args, kwargs),
        virtual_key=engine.virtual_key,
        # PDP subject matchers are authored against an agent role
        # (any_role: ["agent", ...]); the decorator/adapter caller doesn't
        # spell out an identity. Carry a default agent principal + actor_chain
        # so role-scoped policies match the decorator path the same way they
        # match an explicitly-populated DelegationContext. The server only
        # synthesises this for agent-bound VKs; LLM-only VKs need it from here.
        principal="agent:sdk",
        actor_chain=["agent:sdk"],
        identity_type="application",
        context=ContextBag(
            recovery_cost=recovery_cost,
            rag_provenance=rag_provenance,
            tool_fingerprint=tool_fingerprint,
        ),
    )
    decision = engine.decide(dc)

    if decision.verdict == Verdict.DENY:
        raise GuardrailDenied(
            reason=decision.reason,
            decision_id=decision.decision_id,
            policy_id=decision.policy_id,
            tool=tool_name,
        )
    if decision.verdict == Verdict.REQUIRE_APPROVAL:
        try:
            resolved = engine.poll_approval(decision.decision_id)
        except TimeoutError:
            raise GuardrailApprovalPending(
                decision_id=decision.decision_id,
                approvers=list(decision.approvers),
                reason=decision.reason,
            )
        if resolved.verdict == Verdict.DENY:
            raise GuardrailDenied(
                reason="approval denied",
                decision_id=decision.decision_id,
                policy_id=decision.policy_id,
                tool=tool_name,
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
    )
    return apply_obligations(kwargs, decision.obligations)


__all__ = ["resolve", "enforce"]
