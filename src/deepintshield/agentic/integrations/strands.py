"""AWS Strands Agents enforcement.

Strands (the AWS agent framework, default runtime for Bedrock AgentCore) exposes
a typed hook system. A ``HookProvider`` subscribing to ``BeforeToolInvocationEvent``
is the PEP: the callback can cancel the invocation or swap the ``selected_tool``,
so a DENY maps to a cancellation / refusal.

Automatic surface:
    from deepintshield import DeepintShield
    from strands import Agent
    shield = DeepintShield(virtual_key="sk-ds-...")
    agent = Agent(model=..., tools=[...])

Client construction patches Strands' final executor boundary.
``shield.agentic.strands()`` remains an idempotent compatibility helper for
applications that already attach a hook provider. The LLM leg stays native via
Strands' ``OpenAIModel(base_url=gateway)`` or its LiteLLM model class pointed at
the gateway.

Fail-CLOSED on both infrastructure errors and blocking verdicts: the selected
tool never runs unless the PDP returns an executable decision.
"""

from __future__ import annotations

import functools
import inspect
from types import SimpleNamespace
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
    public_agentic_error,
)
from ..gate import resolve
from ..obligations import apply_obligations
from ._common import (
    CallbackInventory,
    callback_method_inventory,
    explicit_callback_inventory,
    merge_callback_inventories,
    source_fingerprint,
)

_EXECUTION_BOUNDARIES = (
    (
        "strands",
        "Agent",
        ("__call__", "invoke_async", "stream_async"),
    ),
    (
        "strands.agent",
        "Agent",
        ("__call__", "invoke_async", "stream_async"),
    ),
)

_CALLBACK_FIELDS = (
    "callbacks",
    "callback_handler",
    "before_tool_callback",
    "after_tool_callback",
)


def executable_callbacks(owner: Any, *, prefix: str = "strands") -> CallbackInventory:
    """Inventory explicitly registered Strands callbacks and hook providers.

    The hook registry's callback mapping is a public structural boundary.  We
    read only that mapping and named provider methods—never executor state or
    framework modules—then filter inherited Strands implementations.
    """
    inventories = [
        explicit_callback_inventory(
            owner,
            _CALLBACK_FIELDS,
            framework_modules=("strands",),
            prefix=prefix,
        )
    ]
    for registry_field in ("hooks", "hook_registry", "_hook_registry"):
        try:
            registry = getattr(owner, registry_field)
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        registered: list[Any] = []
        registry_incomplete = False
        for callbacks_field in (
            "callbacks",
            "_callbacks",
            "registered_callbacks",
            "_registered_callbacks",
        ):
            try:
                callback_map = getattr(registry, callbacks_field)
            except AttributeError:
                continue
            except Exception:
                registry_incomplete = True
                continue
            if not isinstance(callback_map, dict):
                registry_incomplete = True
                continue
            for entries in callback_map.values():
                values = entries if isinstance(entries, (list, tuple)) else [entries]
                for entry in values:
                    # Current Strands stores a small _CallbackEntry(callback,
                    # order) record. Reading its declared callback member is not
                    # traversal of executor/framework internals.
                    try:
                        registered.append(getattr(entry, "callback", entry))
                    except Exception:
                        registry_incomplete = True
        inventories.append(
            explicit_callback_inventory(
                SimpleNamespace(values=registered),
                ("values",),
                framework_modules=("strands",),
                prefix=f"{prefix}.{registry_field}",
            )
        )
        if registry_incomplete:
            inventories.append(CallbackInventory(incomplete=True))
    for provider_field in ("hook_providers", "_hook_providers"):
        try:
            providers = getattr(owner, provider_field)
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        inventories.append(
            callback_method_inventory(
                providers,
                ("before_tool", "after_tool", "on_event"),
                framework_modules=("strands",),
                prefix=f"{prefix}.{provider_field}",
            )
        )
    return merge_callback_inventories(*inventories)

def _tool_name(event: Any) -> str:
    use = getattr(event, "tool_use", None)
    if isinstance(use, dict):
        n = use.get("name")
        if isinstance(n, str) and n:
            return n
    for attr in ("tool_name", "name"):
        v = getattr(event, attr, None)
        if isinstance(v, str) and v:
            return v
    tool = getattr(event, "selected_tool", None)
    return getattr(tool, "tool_name", None) or getattr(tool, "name", None) or "tool"


def _tool_args(event: Any) -> dict[str, Any]:
    use = getattr(event, "tool_use", None)
    if isinstance(use, dict) and isinstance(use.get("input"), dict):
        return use["input"]
    return {}


@public_agentic_boundary
def hook_provider(engine: Any) -> Any:
    """Return a Strands ``HookProvider`` that gates every tool invocation through
    the PDP. Imported lazily so ``strands`` is only required on use."""
    try:
        from strands.hooks import HookProvider, HookRegistry
        from strands.experimental.hooks import BeforeToolInvocationEvent
    except ImportError as error:
        raise GovernanceConfigurationError(
            framework="strands",
            reason=str(error),
            code="framework_dependency_missing",
        ) from None

    class DeepintShieldHooks(HookProvider):
        _deepintshield_hook = True

        def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
            registry.add_callback(BeforeToolInvocationEvent, self._before_tool)

        @public_agentic_boundary
        def _before_tool(self, event: Any) -> None:
            # The automatic executor guard runs after every framework callback,
            # against the final selected tool/arguments. Do not decide twice
            # when an application kept the older explicit hook configuration.
            if _automatic_guard_active():
                return
            name = _tool_name(event)
            try:
                fp = source_fingerprint(getattr(getattr(event, "selected_tool", None), "func", None))
            except Exception:
                fp = ""
            blocked_error: BaseException | None = None
            try:
                selected = getattr(event, "selected_tool", None)
                resolve(
                    engine,
                    name,
                    (),
                    _tool_args(event),
                    tool_fingerprint=fp,
                    tool_callable=getattr(selected, "func", None) or selected,
                )
            except (GuardrailDenied, GuardrailApprovalPending) as error:
                # Cancel the invocation - Strands surfaces this as the tool result.
                try:
                    event.selected_tool = None  # swap out the tool → nothing runs
                except Exception:
                    pass
                fallback = (
                    AGENT_APPROVAL_PENDING
                    if isinstance(error, GuardrailApprovalPending)
                    else AGENT_ACCESS_DENIED
                )
                code = normalize_agentic_error_code(
                    getattr(error, "code", ""), fallback
                )
                error.code = code
                blocked_error = public_agentic_error(error)
            if blocked_error is not None:
                # Raise after leaving the internal-error handler so its private
                # reason/payload is not retained through ``__context__``.
                _raise_detached_error(blocked_error)

    try:
        return DeepintShieldHooks()
    except Exception as error:
        raise GovernanceConfigurationError(
            framework="strands",
            reason=str(error),
            code="framework_integration_unsupported",
        ) from None


def _automatic_guard_active() -> bool:
    try:
        from strands.tools.executors._executor import ToolExecutor

        return bool(
            getattr(
                ToolExecutor._invoke_before_tool_call_hook,
                "_deepintshield_guarded",
                False,
            )
        )
    except Exception:
        return False


def enforce() -> bool:
    """Gate Strands' final pre-execution hook automatically.

    The wrapper deliberately decides *after* Strands has run all registered
    callbacks. A callback may replace ``selected_tool`` or rewrite
    ``tool_use["input"]``; authorizing earlier would create a time-of-check /
    time-of-use bypass.
    """
    from ..enforcement import report_topology_for

    # The agent boundary owns the complete model/tool loop; tool-executor
    # decisions nested inside it inherit one execution/session id.
    report_topology_for(_EXECUTION_BOUNDARIES)
    try:
        from strands.tools.executors._executor import ToolExecutor
    except Exception:
        return False

    original = getattr(ToolExecutor, "_invoke_before_tool_call_hook", None)
    if not callable(original) or not inspect.iscoroutinefunction(original):
        return False
    if getattr(original, "_deepintshield_guarded", False):
        return True

    @functools.wraps(original)
    @public_agentic_boundary
    async def guarded(
        agent: Any,
        tool_func: Any,
        tool_use: Any,
        invocation_state: Any,
    ) -> Any:
        result = await original(agent, tool_func, tool_use, invocation_state)
        try:
            event, interrupts = result
        except Exception:
            from ..errors import GovernanceConfigurationError

            raise GovernanceConfigurationError(
                framework="strands",
                reason="before-tool boundary returned an unsupported result",
            ) from None

        # A cancelled/interrupted tool will not execute and needs no PDP trip.
        if interrupts or bool(getattr(event, "cancel_tool", False)):
            return result

        from ..enforcement import bind_engine, ensure_topology_reported, resolve_engine

        engine = resolve_engine(agent)
        bind_engine(agent, engine, recursive=True)
        name = _tool_name(event)
        selected = getattr(event, "selected_tool", None) or tool_func
        try:
            fp = source_fingerprint(
                getattr(selected, "func", None)
                or getattr(selected, "_func", None)
                or selected
            )
        except Exception:
            fp = ""
        call_args = _tool_args(event)
        ensure_topology_reported(engine, agent, required=True)
        decision = resolve(
            engine,
            name,
            (),
            call_args,
            tool_fingerprint=fp,
            tool_callable=getattr(selected, "func", None) or selected,
        )
        masked = apply_obligations(call_args, decision.obligations)
        if masked is not call_args:
            final_use = getattr(event, "tool_use", None)
            if not isinstance(final_use, dict):
                from ..errors import GovernanceConfigurationError

                raise GovernanceConfigurationError(
                    framework="strands",
                    reason="MASK obligation cannot rewrite the tool arguments",
                ) from None
            final_use["input"] = masked
        return result

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    try:
        ToolExecutor._invoke_before_tool_call_hook = staticmethod(guarded)
    except Exception:
        return False
    return True


__all__ = ["hook_provider", "enforce", "executable_callbacks"]
