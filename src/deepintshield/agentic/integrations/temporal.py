"""Temporal enforcement - gate every activity through the PDP.

Temporal has no policy engine of its own ("the workflow *is* the gate"), yet it
runs durable agents (OpenAI Agents SDK via ``temporalio.contrib.openai_agents``,
Codex/Replit-style loops) in production. The activity boundary is the natural
PEP: an ``ActivityInboundInterceptor`` runs OUTSIDE the deterministic workflow
sandbox, so network I/O to ``/decide`` is allowed there.

Automatic surface:
    from deepintshield import DeepintShield
    shield = DeepintShield(virtual_key=...)
    worker = Worker(client, task_queue="q", activities=[...])

Client construction patches ``Worker`` to prepend the enforcement interceptor.
``shield.agentic.temporal()`` remains an idempotent compatibility helper for
applications that already attach the interceptor explicitly.

Everything else stays native: durability, retries, replay, event-history audit.
The LLM leg is a separate one-liner - point the model activity's client at the
gateway (``ds.openai()`` base_url swap) to light up cache/guardrails/cost.

Design mirrors the other adapters: fail-CLOSED on infrastructure errors and
verdicts. A gateway outage fails the activity before its body executes (and may
use the workflow's normal retry policy); a policy DENY raises a NON-retryable
``ApplicationError`` so Temporal doesn't spin forever on a deterministic block.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from ..errors import (
    AGENT_ACCESS_DENIED,
    AGENT_APPROVAL_PENDING,
    GuardrailApprovalPending,
    GuardrailDenied,
    GovernanceConfigurationError,
    _raise_detached_error,
    normalize_agentic_error_code,
    public_agentic_boundary,
    public_agentic_error_details,
)
from ..gate import resolve
from ..obligations import apply_obligations
from ._common import source_fingerprint

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
        @public_agentic_boundary
        async def execute_activity(self, input: Any) -> Any:  # noqa: ANN401
            from ..execution import execution_scope, mark_execution_outcome

            engine = get_engine()
            name = _activity_name(input)
            args = tuple(getattr(input, "args", ()) or ())
            try:
                fp = source_fingerprint(getattr(input, "fn", None))
            except Exception:
                fp = ""
            with execution_scope(
                engine,
                getattr(input, "fn", None) or input,
                framework="temporal",
            ) as frame:
                blocked_error: BaseException | None = None
                try:
                    decision = resolve(
                        engine,
                        name,
                        args,
                        {},
                        tool_fingerprint=fp,
                        tool_callable=getattr(input, "fn", None),
                    )
                except GuardrailDenied as error:
                    code = normalize_agentic_error_code(
                        getattr(error, "code", ""), AGENT_ACCESS_DENIED
                    )
                    mark_execution_outcome(
                        frame,
                        "blocked",
                        error_code=code,
                        error_summary="",
                    )
                    # Non-retryable: a policy DENY is deterministic, retrying is waste.
                    details = public_agentic_error_details(code)
                    blocked_error = ApplicationError(
                        f"{code}: {details['message']}",
                        type=code,
                        non_retryable=True,
                    )
                except GuardrailApprovalPending as error:
                    code = normalize_agentic_error_code(
                        getattr(error, "code", ""), AGENT_APPROVAL_PENDING
                    )
                    mark_execution_outcome(
                        frame,
                        "awaiting_approval",
                        error_code=code,
                        error_summary="",
                    )
                    details = public_agentic_error_details(code)
                    blocked_error = ApplicationError(
                        f"{code}: {details['message']}",
                        type=code,
                        non_retryable=True,
                    )
                if blocked_error is not None:
                    # Temporal deliberately keeps its native protocol error,
                    # but must not retain the internal policy error as context.
                    _raise_detached_error(blocked_error)
                if decision.obligations:
                    masked_args = tuple(
                        apply_obligations(value, decision.obligations)
                        if isinstance(value, dict)
                        else value
                        for value in args
                    )
                    if masked_args != args:
                        try:
                            input.args = masked_args
                        except Exception:
                            from ..errors import GovernanceConfigurationError

                            raise GovernanceConfigurationError(
                                framework="temporal",
                                reason="MASK obligation cannot rewrite activity arguments",
                            ) from None
                return await super().execute_activity(input)

    class _DISInterceptor(Interceptor):
        _deepintshield_interceptor = True

        def intercept_activity(self, next: Any) -> Any:  # noqa: ANN401
            return _DISActivityInbound(next)

    return _DISInterceptor, _DISActivityInbound


@public_agentic_boundary
def interceptor(engine: Any) -> Any:
    """Return the optional compatibility interceptor for explicit wiring."""
    try:
        cls, _ = _make_interceptor_classes(lambda: engine)
    except ImportError as error:
        raise GovernanceConfigurationError(
            framework="temporal",
            reason=str(error),
            code="framework_dependency_missing",
        ) from None
    try:
        return cls()
    except Exception as error:
        raise GovernanceConfigurationError(
            framework="temporal",
            reason=str(error),
            code="framework_integration_unsupported",
        ) from None


def enforce() -> bool:
    """Inject the activity interceptor into every subsequently constructed
    Temporal worker. Worker construction is the official interceptor boundary
    and happens before any activity task can execute."""
    try:
        from temporalio.worker import Worker
    except Exception:
        return False
    original = getattr(Worker, "__init__", None)
    if not callable(original):
        return False
    if getattr(original, "_deepintshield_guarded", False):
        return True
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return False
    if "interceptors" not in signature.parameters:
        return False

    @functools.wraps(original)
    @public_agentic_boundary
    def guarded(self: Any, *args: Any, **kwargs: Any) -> None:
        from ..enforcement import bind_engine, report_topology, resolve_engine

        engine = resolve_engine()
        bound = signature.bind_partial(self, *args, **kwargs)
        existing = list(bound.arguments.get("interceptors", ()) or ())
        if not any(
            bool(getattr(item, "_deepintshield_interceptor", False))
            for item in existing
        ):
            # Prepend so authorization happens before user interceptors and a
            # user interceptor cannot invoke the terminal activity first.
            existing.insert(0, interceptor(engine))
        bound.arguments["interceptors"] = existing
        original(*bound.args, **bound.kwargs)
        if not bind_engine(self, engine, recursive=True):
            from ..errors import GovernanceConfigurationError

            raise GovernanceConfigurationError(
                framework="temporal",
                reason="Worker cannot retain its DeepintShield engine binding",
            ) from None
        report_topology(engine, self, sync=True, required=True)

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    try:
        Worker.__init__ = guarded
    except Exception:
        return False
    return True


__all__ = ["interceptor", "enforce"]
