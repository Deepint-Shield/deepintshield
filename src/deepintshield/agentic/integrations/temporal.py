"""Temporal enforcement - gate every activity through the PDP.

Temporal has no policy engine of its own ("the workflow *is* the gate"), yet it
runs durable agents (OpenAI Agents SDK via ``temporalio.contrib.openai_agents``,
Codex/Replit-style loops) in production. The activity boundary is the natural
PEP: an ``ActivityInboundInterceptor`` runs OUTSIDE the deterministic workflow
sandbox, so network I/O to ``/decide`` is allowed there.

Minimal-code surface - one interceptor:
    from deepintshield import DeepintShield
    shield = DeepintShield(virtual_key=...)
    worker = Worker(client, task_queue="q", activities=[...],
                    interceptors=[shield.agentic.temporal()])

Everything else stays native: durability, retries, replay, event-history audit.
The LLM leg is a separate one-liner - point the model activity's client at the
gateway (``ds.openai()`` base_url swap) to light up cache/guardrails/cost.

Design mirrors the other adapters: fail-OPEN on an infrastructure error (a
gateway hiccup must never break a durable workflow) but fail-CLOSED on a verdict
- a DENY raises a NON-retryable ``ApplicationError`` so Temporal doesn't spin the
activity forever on a policy denial.
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import GuardrailApprovalPending, GuardrailDenied
from ..gate import resolve
from ._common import source_fingerprint

log = logging.getLogger(__name__)


def _activity_name(input: Any) -> str:
    """The governed tool identity for an activity = the activity function's own
    name, so policy follows the implementation, not the registration label."""
    fn = getattr(input, "fn", None)
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name and name != "<lambda>":
        return name
    # Fall back to the runtime activity type (set at execution time).
    try:
        from temporalio import activity

        info = activity.info()
        if getattr(info, "activity_type", ""):
            return info.activity_type
    except Exception:
        pass
    return getattr(fn, "__qualname__", "activity")


def _make_interceptor_classes(get_engine: Any) -> tuple[Any, Any]:
    """Build the (Interceptor, ActivityInboundInterceptor) pair bound to
    ``get_engine``. Imported lazily so ``temporalio`` is only required when the
    integration is actually used."""
    from temporalio.exceptions import ApplicationError
    from temporalio.worker import ActivityInboundInterceptor, Interceptor

    class _DISActivityInbound(ActivityInboundInterceptor):
        async def execute_activity(self, input: Any) -> Any:  # noqa: ANN401
            engine = get_engine()
            name = _activity_name(input)
            args = tuple(getattr(input, "args", ()) or ())
            try:
                fp = source_fingerprint(getattr(input, "fn", None))
            except Exception:
                fp = ""
            try:
                resolve(engine, name, args, {}, tool_fingerprint=fp)
            except GuardrailDenied as d:
                # Non-retryable: a policy DENY is deterministic, retrying is waste.
                raise ApplicationError(
                    f"deepintshield denied activity {name!r}: {d}",
                    type="DeepintShieldDenied",
                    non_retryable=True,
                ) from d
            except GuardrailApprovalPending as p:
                raise ApplicationError(
                    f"deepintshield approval pending for {name!r}: {p}",
                    type="DeepintShieldApprovalPending",
                    non_retryable=True,
                ) from p
            except Exception:  # infra hiccup → fail-open, never break the workflow
                log.debug("deepintshield temporal PEP fail-open for %s", name, exc_info=True)
            return await super().execute_activity(input)

    class _DISInterceptor(Interceptor):
        def intercept_activity(self, next: Any) -> Any:  # noqa: ANN401
            return _DISActivityInbound(next)

    return _DISInterceptor, _DISActivityInbound


def interceptor(engine: Any) -> Any:
    """Return a Temporal ``Interceptor`` that gates every activity through the
    PDP. Pass it to ``Worker(..., interceptors=[interceptor])``."""
    cls, _ = _make_interceptor_classes(lambda: engine)
    return cls()


def enforce(get_engine: Any) -> bool:
    """Auto-enforcement entry point (``install_all``). Temporal has no global
    build/execute boundary to monkey-patch safely - a Worker is constructed
    explicitly and takes its interceptors as a constructor argument - so there is
    nothing to patch at import time. Returns False (nothing installed); use
    ``shield.agentic.temporal()`` to attach the interceptor to your Worker."""
    return False


__all__ = ["interceptor", "enforce"]
