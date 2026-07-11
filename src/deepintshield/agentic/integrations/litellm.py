"""LiteLLM enforcement - gate every ``litellm.completion`` / ``acompletion`` call
through the PDP's prompt guardrail (injection / PII) before the model is called.

LiteLLM is an LLM SDK, not a tool framework, so "enforce in the SDK" means the
prompt boundary: we run the last user message through the agent-prompt guardrail
(``decide(tool="llm.completion", prompt=…)``) and block on a DENY verdict. The
prompt is scan-only and never stored (zero-data-retention). Fail-OPEN on infra
(a gateway hiccup must not break completions) but fail-CLOSED on a verdict.

Cooperative defense-in-depth - the gateway stays the authoritative boundary.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

log = logging.getLogger(__name__)

_LLM_TOOL = "llm.completion"


def _last_user_prompt(kwargs: dict, args: tuple) -> str:
    msgs = kwargs.get("messages")
    if msgs is None and args:
        msgs = next((a for a in args if isinstance(a, list)), None)
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # multimodal content blocks
                return " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return ""


def _check(get_engine: Any, kwargs: dict, args: tuple) -> None:
    """Run the prompt guardrail; raise GuardrailDenied on a blocking verdict.
    Fail-open on any infrastructure error (never break a completion)."""
    from ..errors import GuardrailDenied
    from ..types import Verdict

    prompt = _last_user_prompt(kwargs, args)
    if not prompt:
        return
    try:
        from ..obligations import digest
        from ..types import ContextBag, DelegationContext

        engine = get_engine()
        dc = DelegationContext(
            tool=_LLM_TOOL, args_digest=digest((), {}), virtual_key=engine.virtual_key,
            prompt=prompt, principal="agent:sdk", actor_chain=["agent:sdk"],
            identity_type="application", context=ContextBag(),
        )
        decision = engine.decide(dc)
    except Exception:
        return  # infra → fail-open
    if decision is not None and decision.verdict == Verdict.DENY:
        raise GuardrailDenied(
            reason=decision.reason, decision_id=decision.decision_id,
            policy_id=decision.policy_id, tool=_LLM_TOOL,
        )


def enforce(get_engine: Any) -> bool:
    """Patch ``litellm.completion`` + ``acompletion`` so every call is gated by the
    prompt guardrail. Idempotent + fail-open. Returns True if installed."""
    try:
        import litellm
    except Exception:
        return False

    installed = False

    orig = getattr(litellm, "completion", None)
    if callable(orig) and not getattr(orig, "_deepintshield_guarded", False):
        @functools.wraps(orig)
        def guarded_completion(*args: Any, **kwargs: Any) -> Any:
            _check(get_engine, kwargs, args)
            return orig(*args, **kwargs)

        guarded_completion._deepintshield_guarded = True  # type: ignore[attr-defined]
        litellm.completion = guarded_completion
        installed = True

    aorig = getattr(litellm, "acompletion", None)
    if callable(aorig) and not getattr(aorig, "_deepintshield_guarded", False):
        @functools.wraps(aorig)
        async def guarded_acompletion(*args: Any, **kwargs: Any) -> Any:
            _check(get_engine, kwargs, args)
            return await aorig(*args, **kwargs)

        guarded_acompletion._deepintshield_guarded = True  # type: ignore[attr-defined]
        litellm.acompletion = guarded_acompletion
        installed = True

    return installed


__all__ = ["enforce"]
