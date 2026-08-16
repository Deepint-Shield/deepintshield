"""LiteLLM enforcement - gate every ``litellm.completion`` / ``acompletion`` call
through canonical Agentic authorization before the model is called.

LiteLLM is an LLM SDK, not a tool framework, so "enforce in the SDK" means the
completion boundary: ``decide(tool="llm.completion")`` must allow before the
call runs. Raw prompt text is never sent to canonical GAF (zero-data-retention).
It remains available only to the explicitly marked older-gateway compatibility
PDP, where it is scan-only and never stored. PDP or client resolution failures
are fail-closed, exactly like tool execution.

Cooperative defense-in-depth - the gateway stays the authoritative boundary.

No GAF registry discovery here, deliberately: LiteLLM has no agents, tools or
edges to describe - there is no topology, only a completion call - so unlike the
other five adapters it does not call ``enforcement.report_topology``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ..errors import public_agentic_boundary

log = logging.getLogger(__name__)

_LLM_TOOL = "llm.completion"


def _last_user_prompt(kwargs: dict, args: tuple) -> str:
    raw_prompt = kwargs.get("prompt")
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list):
        return "\n".join(str(item) for item in raw_prompt)
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


def _check(kwargs: dict, args: tuple) -> dict:
    """Run authorization before every completion call."""
    prompt = _last_user_prompt(kwargs, args)
    from ..enforcement import resolve_engine
    from ..gate import enforce as gate_enforce

    engine = resolve_engine()
    # Even a completion with no user-text field must pass authorization. An
    # early return here used to make ``prompt=""`` / system-only calls a policy
    # bypass, and handling only DENY let REQUIRE_APPROVAL execute. The shared
    # gate maps every blocking verdict consistently. Prompt text remains solely
    # in the older-gateway compatibility payload; canonical GAF receives neither
    # the raw prompt nor another hidden PDP verdict.
    return gate_enforce(
        engine,
        _LLM_TOOL,
        args,
        kwargs,
        prompt=prompt,
    )


def enforce() -> bool:
    """Patch ``litellm.completion`` + ``acompletion`` so every call is gated.

    Idempotent and fail-closed. Returns True if installed.
    """
    try:
        import litellm
    except Exception:
        return False

    installed = False

    orig = getattr(litellm, "completion", None)
    if callable(orig) and not getattr(orig, "_deepintshield_guarded", False):
        @functools.wraps(orig)
        @public_agentic_boundary
        def guarded_completion(*args: Any, **kwargs: Any) -> Any:
            kwargs = _check(kwargs, args)
            return orig(*args, **kwargs)

        guarded_completion._deepintshield_guarded = True  # type: ignore[attr-defined]
        litellm.completion = guarded_completion
        installed = True

    aorig = getattr(litellm, "acompletion", None)
    if callable(aorig) and not getattr(aorig, "_deepintshield_guarded", False):
        @functools.wraps(aorig)
        @public_agentic_boundary
        async def guarded_acompletion(*args: Any, **kwargs: Any) -> Any:
            kwargs = _check(kwargs, args)
            return await aorig(*args, **kwargs)

        guarded_acompletion._deepintshield_guarded = True  # type: ignore[attr-defined]
        litellm.acompletion = guarded_acompletion
        installed = True

    return installed


__all__ = ["enforce"]
