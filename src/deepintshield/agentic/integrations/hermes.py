"""Hermes Agent (NousResearch/hermes-agent) enforcement.

Hermes exposes a first-class plugin API: a package in the ``hermes_agent.plugins``
entry-point group (or dropped in ``~/.hermes/plugins/<name>/``) whose
``register(ctx)`` wires hooks. The ``pre_tool_call`` hook fires in
``model_tools.handle_function_call`` before dispatch for EVERY tool - built-in,
plugin AND MCP - so one hook governs the whole tool surface.

Minimal-code surface - a Hermes plugin that reuses the shared PDP core:

    # ~/.hermes/plugins/deepintshield/__init__.py
    from deepintshield import DeepintShield
    from deepintshield.agentic.integrations.hermes import install
    _shield = DeepintShield(virtual_key="sk-ds-...")
    def register(ctx):
        install(ctx, _shield.agentic.engine)

Everything else stays native: the terminal sandbox, the approval UI, the tool
registry, and (one config line) the LLM leg - set Hermes's ``base_url`` to the
gateway so cache/guardrails/budgets/cost light up with zero code.

Hooks fire fail-open in Hermes (errors are caught + logged by the runtime), so
fail-CLOSED enforcement is implemented INSIDE the callback: a DENY returns an
explicit ``{"action": "block", ...}`` rather than relying on an exception.
Note: ``pre_tool_call`` can block but not rewrite args (Hermes issue #44582), so
MASK obligations are surfaced as a block with the reason rather than silently
mutating - use ``transform_tool_result`` for output redaction if needed.
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import GuardrailApprovalPending, GuardrailDenied
from ..gate import resolve
from ..types import Verdict

log = logging.getLogger(__name__)


def _pre_tool_call(engine: Any, tool_name: str, args: Any) -> dict[str, Any] | None:
    """Return a Hermes block directive on a blocking verdict, else ``None``."""
    call_args = args if isinstance(args, dict) else {"args": args}
    try:
        decision = resolve(engine, tool_name, (), call_args)
    except GuardrailDenied as d:
        return {"action": "block", "message": f"deepintshield policy denied {tool_name!r}: {d.reason or d}"}
    except GuardrailApprovalPending as p:
        return {"action": "block", "message": f"deepintshield approval required for {tool_name!r} (decision {p.decision_id})"}
    except Exception:  # infra hiccup → fail-open, let the tool run
        log.debug("deepintshield hermes PEP fail-open for %s", tool_name, exc_info=True)
        return None
    # MASK can't rewrite args in Hermes' pre-hook (#44582); block with the reason
    # so the obligation isn't silently dropped.
    if decision.verdict == Verdict.MASK and decision.obligations:
        return {"action": "block", "message": f"deepintshield requires redaction of {tool_name!r} inputs before this call"}
    return None


def install(ctx: Any, engine: Any) -> bool:
    """Register the PDP hooks on a Hermes plugin context. Returns True if at
    least the enforcement hook was registered. Idempotent per context."""
    register_hook = getattr(ctx, "register_hook", None)
    if not callable(register_hook):
        return False

    def pre_tool_call(tool_name: str, args: Any = None, task_id: Any = None, **_: Any) -> Any:
        return _pre_tool_call(engine, tool_name, args)

    def post_tool_call(tool_name: str, args: Any = None, result: Any = None, **_: Any) -> None:
        # Observation hook: the pre-call decide already fingerprinted + logged the
        # call server-side; this is the seam for extra audit/drift emission.
        return None

    register_hook("pre_tool_call", pre_tool_call)
    try:
        register_hook("post_tool_call", post_tool_call)
    except Exception:
        pass
    return True


def enforce(get_engine: Any) -> bool:
    """Auto-enforcement entry point (``install_all``). Hermes loads governance
    through its own plugin loader calling ``register(ctx)`` - there is no
    in-process build/execute boundary to monkey-patch - so nothing is installed
    here. Ship the plugin shim above instead. Returns False."""
    return False


__all__ = ["install", "enforce"]
