"""Authenticated runtime configuration, scoped to one native execution.

Disabled Agentic Security is an explicit gateway capability, never an error
fallback. No disabled state survives a top-level run, so changing plans or
enabling workspace security takes effect on the next invocation.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from .types import Decision, Verdict
from .task_context import require_task_security

_states: contextvars.ContextVar[tuple[tuple[Any, bool], ...]] = (
    contextvars.ContextVar("deepintshield_agentic_runtime", default=())
)


def scoped_security_enabled(engine: Any) -> bool | None:
    for owner, enabled in reversed(_states.get()):
        if owner is engine:
            return enabled
    return None


def security_enabled(engine: Any) -> bool:
    enabled = scoped_security_enabled(engine)
    if enabled is not None:
        return require_task_security(enabled)
    # Older/custom engines retain their existing fail-closed behavior. Check
    # the class method so dynamic mock/proxy attributes cannot imply opt-out.
    fetch = getattr(type(engine), "fetch_security_enabled", None)
    return require_task_security(fetch(engine) if callable(fetch) else True)


@contextmanager
def security_scope(engine: Any) -> Iterator[bool]:
    for owner, enabled in reversed(_states.get()):
        if owner is engine:
            require_task_security(enabled)
            yield enabled
            return
    enabled = security_enabled(engine)
    token = _states.set((*_states.get(), (engine, enabled)))
    try:
        yield enabled
    finally:
        _states.reset(token)


def passthrough_decision() -> Decision:
    """Keep the adapter contract without claiming a security authorization."""
    return Decision(
        verdict=Verdict.ALLOW,
        decision_id="",
        mode="passthrough",
        reason="agentic_security_disabled",
        proceed=True,
        gaf_checked=False,
    )
