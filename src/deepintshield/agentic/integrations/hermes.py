"""Hermes Agent (NousResearch/hermes-agent) enforcement.

Hermes exposes one central ``model_tools.handle_function_call`` dispatcher for
every built-in, plugin and MCP tool. Constructing ``DeepintShield`` inside a
Hermes host plugin automatically guards that dispatcher; the import watcher
also covers Hermes loading ``model_tools`` afterward.

Minimal-code surface - a Hermes plugin that reuses the shared PDP core:

    # ~/.hermes/plugins/deepintshield/__init__.py
    from deepintshield import DeepintShield
    _shield = DeepintShield(virtual_key="sk-ds-...")
    def register(ctx):
        pass

The older ``shield.agentic.hermes(ctx)`` helper remains supported. When the
automatic dispatcher guard is active, it deliberately avoids a duplicate
decision.

Everything else stays native: the terminal sandbox, the approval UI, the tool
registry, and (one config line) the LLM leg - set Hermes's ``base_url`` to the
gateway so cache/guardrails/budgets/cost light up with zero code.

Hooks fire fail-open in Hermes (errors are caught + logged by the runtime), so
fail-CLOSED enforcement is implemented INSIDE the callback: a DENY, approval,
or PDP/setup failure returns an explicit ``{"action": "block", ...}`` rather
than relying on an exception.
Note: ``pre_tool_call`` can block but not rewrite args (Hermes issue #44582), so
MASK obligations are surfaced with a stable block code rather than silently
mutating - use ``transform_tool_result`` for output redaction if needed.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
from typing import Any

from ..errors import (
    AGENTIC_ERROR,
    AGENT_ACCESS_DENIED,
    AGENT_APPROVAL_PENDING,
    AGENT_OBLIGATION_UNSUPPORTED,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    GuardrailDenied,
    normalize_agentic_error_code,
    public_agentic_boundary,
    public_agentic_error_details,
)
from ..gate import resolve
from ..types import Verdict

log = logging.getLogger(__name__)


def _error_code(error: BaseException, fallback: str = AGENTIC_ERROR) -> str:
    return normalize_agentic_error_code(getattr(error, "code", ""), fallback)


def _block_directive(code: str) -> dict[str, Any]:
    details = public_agentic_error_details(code)
    return {
        "action": "block",
        "code": details["code"],
        "message": details["message"],
        "remediation": details["action"],
        "dashboard_path": details["dashboard_path"],
    }


def _blocked_result(code: str) -> str:
    details = public_agentic_error_details(code)
    return json.dumps(
        {
            "error": details["code"],
            "status": "blocked",
            "message": details["message"],
            "action": details["action"],
            "dashboard_path": details["dashboard_path"],
        },
        ensure_ascii=False,
    )


def _pre_tool_call(engine: Any, tool_name: str, args: Any) -> dict[str, Any] | None:
    """Return a Hermes block directive on a blocking verdict, else ``None``."""
    call_args = args if isinstance(args, dict) else {"args": args}
    try:
        decision = resolve(engine, tool_name, (), call_args)
    except GuardrailDenied as error:
        return _block_directive(_error_code(error, AGENT_ACCESS_DENIED))
    except GuardrailApprovalPending as error:
        return _block_directive(_error_code(error, AGENT_APPROVAL_PENDING))
    except Exception as exc:
        # Hermes catches hook exceptions and otherwise continues dispatch, so
        # propagating here would be fail-open. Return its explicit block
        # directive for every PDP/setup failure instead.
        code = _error_code(exc)
        log.warning("deepintshield hermes PEP blocked %s: %s", tool_name, code)
        return _block_directive(code)
    # MASK can't rewrite args in Hermes' pre-hook (#44582); block with the
    # stable unsupported-obligation code so the obligation isn't dropped.
    if decision.verdict == Verdict.MASK and decision.obligations:
        return _block_directive(AGENT_OBLIGATION_UNSUPPORTED)
    return None


@public_agentic_boundary
def install(ctx: Any, engine: Any) -> bool:
    """Register the PDP hooks on a Hermes plugin context. Returns True if at
    least the enforcement hook was registered. Idempotent per context."""
    register_hook = getattr(ctx, "register_hook", None)
    if not callable(register_hook):
        raise GovernanceConfigurationError(
            framework="hermes",
            reason="plugin context exposes no register_hook boundary",
        ) from None

    def pre_tool_call(tool_name: str, args: Any = None, task_id: Any = None, **_: Any) -> Any:
        if _automatic_guard_active():
            return None
        return _pre_tool_call(engine, tool_name, args)

    def post_tool_call(tool_name: str, args: Any = None, result: Any = None, **_: Any) -> None:
        # Observation hook: the pre-call decide already fingerprinted + logged the
        # call server-side; this is the seam for extra audit/drift emission.
        return None

    try:
        register_hook("pre_tool_call", pre_tool_call)
    except Exception:
        raise GovernanceConfigurationError(
            framework="hermes",
            reason="could not register pre_tool_call enforcement hook",
        ) from None
    try:
        register_hook("post_tool_call", post_tool_call)
    except Exception:
        pass
    return True


def _automatic_guard_active() -> bool:
    try:
        import model_tools

        return bool(
            getattr(
                model_tools.handle_function_call,
                "_deepintshield_guarded",
                False,
            )
        )
    except Exception:
        return False


def enforce() -> bool:
    """Guard Hermes' central dispatcher, including aliases imported before the
    SDK client was constructed.

    Hermes intentionally catches plugin exceptions and continues. The wrapper
    therefore converts every deny/approval/setup failure to Hermes' normal JSON
    tool-error result so the real handler is never entered.
    """
    try:
        import model_tools
    except Exception:
        return False
    original = getattr(model_tools, "handle_function_call", None)
    if not callable(original):
        return False
    if getattr(original, "_deepintshield_guarded", False):
        return True

    @functools.wraps(original)
    @public_agentic_boundary
    def guarded(function_name: str, function_args: Any, *args: Any, **kwargs: Any) -> Any:
        # Hermes unwraps this bridge recursively to the real tool. Authorizing
        # the bridge too would require an unrelated `tool_call` permission and
        # could reject an otherwise-authorized underlying action.
        if function_name not in {"tool_search", "tool_describe", "tool_call"}:
            try:
                from ..enforcement import ensure_topology_reported, resolve_engine
                from ..execution import execution_scope, mark_execution_outcome

                engine = resolve_engine()
            except Exception as exc:
                code = _error_code(exc)
                log.warning(
                    "deepintshield hermes dispatcher blocked %s: %s",
                    function_name,
                    code,
                )
                return _blocked_result(code)
            with execution_scope(
                engine, model_tools, framework="hermes"
            ) as frame:
                try:
                    ensure_topology_reported(
                        engine,
                        model_tools,
                        required=True,
                    )
                except Exception as exc:
                    code = _error_code(exc)
                    mark_execution_outcome(
                        frame,
                        "blocked",
                        error_code=code,
                        error_summary="",
                    )
                    return _blocked_result(code)
                directive = _pre_tool_call(engine, function_name, function_args)
                if directive is not None:
                    code = normalize_agentic_error_code(
                        directive.get("code"), AGENTIC_ERROR
                    )
                    mark_execution_outcome(
                        frame,
                        "blocked",
                        error_code=code,
                        error_summary="",
                    )
                    return _blocked_result(code)
                return original(function_name, function_args, *args, **kwargs)
        return original(function_name, function_args, *args, **kwargs)

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    try:
        model_tools.handle_function_call = guarded
    except Exception:
        return False

    # `run_agent` and the MCP transport re-export the dispatcher with
    # `from model_tools import ...`; replace only exact aliases of the original,
    # never unrelated functions that happen to share the same name.
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, "handle_function_call", None) is original:
                setattr(module, "handle_function_call", guarded)
        except Exception:
            continue
    return True


__all__ = ["install", "enforce"]
