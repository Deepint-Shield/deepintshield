"""Google ADK (Agent Development Kit) enforcement.

ADK exposes central normal, live and threaded tool dispatch functions. The
automatic adapter governs those final boundaries so callbacks cannot rewrite a
tool or its arguments after authorization.

Automatic surface:
    from deepintshield import DeepintShield
    from google.adk.runners import InMemoryRunner
    shield = DeepintShield(virtual_key="sk-ds-...")
    runner = InMemoryRunner(agent=agent)

``shield.agentic.google_adk()`` remains an idempotent compatibility helper for
applications already using the app-level ``BasePlugin`` API. The LLM leg stays
native via ADK's LiteLLM/GenAI model wrapper pointed at the gateway.

Fail-CLOSED on both a blocking verdict and an authorization infrastructure
error. DENY/approval return ADK's short-circuit result; transport/setup errors
propagate so the runner cannot execute the tool without a decision.
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
    normalize_agentic_error_code,
    public_agentic_boundary,
    public_agentic_error_details,
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
        "google.adk.runners",
        "BaseRunner",
        ("run_async", "run_live", "run"),
    ),
    (
        "google.adk.runners",
        "InMemoryRunner",
        ("run_async", "run_live", "run"),
    ),
    (
        "google.adk.runners",
        "Runner",
        ("run_async", "run_live", "run"),
    ),
)

_AGENT_CALLBACK_FIELDS = (
    "before_agent_callback",
    "after_agent_callback",
    "before_model_callback",
    "after_model_callback",
    "before_tool_callback",
    "after_tool_callback",
    "canonical_before_agent_callbacks",
    "canonical_after_agent_callbacks",
    "canonical_before_model_callbacks",
    "canonical_after_model_callbacks",
    "canonical_before_tool_callbacks",
    "canonical_after_tool_callbacks",
)
_PLUGIN_CALLBACK_METHODS = _AGENT_CALLBACK_FIELDS + (
    "on_user_message_callback",
    "before_run_callback",
    "after_run_callback",
    "on_event_callback",
    "on_tool_error_callback",
    "on_model_error_callback",
    "on_agent_error_callback",
    "on_run_error_callback",
)


def executable_callbacks(owner: Any, *, prefix: str = "google_adk") -> CallbackInventory:
    """Inventory ADK's public agent callbacks and explicitly attached plugins.

    ADK's own runner/flow callback machinery is never traversed. A custom
    plugin override is captured; inherited ADK and DeepIntShield plugin methods
    are excluded by implementation-module ownership.
    """
    inventories = [
        explicit_callback_inventory(
            owner,
            _AGENT_CALLBACK_FIELDS,
            framework_modules=("google.adk",),
            prefix=prefix,
        )
    ]
    for field in ("plugins", "_plugins"):
        try:
            plugins = getattr(owner, field)
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        inventories.append(
            callback_method_inventory(
                plugins,
                _PLUGIN_CALLBACK_METHODS,
                framework_modules=("google.adk",),
                prefix=f"{prefix}.{field}",
            )
        )
    for manager_field in ("plugin_manager", "_plugin_manager"):
        try:
            manager = getattr(owner, manager_field)
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        try:
            plugins = getattr(manager, "plugins")
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        inventories.append(
            callback_method_inventory(
                plugins,
                _PLUGIN_CALLBACK_METHODS,
                framework_modules=("google.adk",),
                prefix=f"{prefix}.{manager_field}.plugins",
            )
        )
    return merge_callback_inventories(*inventories)

def _tool_name(tool: Any) -> str:
    for attr in ("name", "__name__"):
        v = getattr(tool, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(tool).__name__


def _deny_payload(error: BaseException, fallback: str) -> dict[str, Any]:
    """The ADK short-circuit shape: a tool-result dict the model sees instead of
    the real tool output."""
    code = normalize_agentic_error_code(getattr(error, "code", ""), fallback)
    details = public_agentic_error_details(code)
    return {
        "error": code,
        "status": "blocked",
        "message": details["message"],
        "action": details["action"],
        "dashboard_path": details["dashboard_path"],
    }


@public_agentic_boundary
def plugin(engine: Any) -> Any:
    """Return a ``BasePlugin`` instance that gates every ADK tool call through
    the PDP. Imported lazily so ``google.adk`` is only required on use."""
    try:
        from google.adk.plugins.base_plugin import BasePlugin
    except ImportError as error:
        raise GovernanceConfigurationError(
            framework="google-adk",
            reason=str(error),
            code="framework_dependency_missing",
        ) from None

    class DeepintShieldPlugin(BasePlugin):
        _deepintshield_plugin = True

        def __init__(self) -> None:
            super().__init__(name="deepintshield")

        @public_agentic_boundary
        async def before_tool_callback(self, *, tool: Any = None, tool_args: Any = None, tool_context: Any = None, **_: Any) -> Any:  # noqa: ANN401
            # Current ADK releases are guarded at the final tool-dispatch
            # functions. Keep this explicit plugin backward compatible without
            # issuing a second decision for the same invocation.
            if _automatic_guard_active():
                return None
            name = _tool_name(tool)
            call_args = tool_args if isinstance(tool_args, dict) else {"args": tool_args}
            try:
                fp = source_fingerprint(getattr(tool, "func", None) or tool)
            except Exception:
                fp = ""
            try:
                decision = resolve(
                    engine,
                    name,
                    (),
                    call_args,
                    tool_fingerprint=fp,
                    tool_callable=getattr(tool, "func", None) or tool,
                )
            except GuardrailDenied as error:
                return _deny_payload(error, AGENT_ACCESS_DENIED)
            except GuardrailApprovalPending as error:
                return _deny_payload(error, AGENT_APPROVAL_PENDING)
            masked = apply_obligations(call_args, decision.obligations)
            if isinstance(tool_args, dict) and masked is not call_args:
                tool_args.clear()
                tool_args.update(masked)
            return None  # ALLOW → let ADK run the tool

    try:
        return DeepintShieldPlugin()
    except Exception as error:
        raise GovernanceConfigurationError(
            framework="google-adk",
            reason=str(error),
            code="framework_integration_unsupported",
        ) from None


def _automatic_guard_active() -> bool:
    try:
        from google.adk.flows.llm_flows import functions

        for attr in ("__call_tool_async", "_call_tool_in_thread_pool", "__call_tool_live"):
            if getattr(
                getattr(functions, attr, None),
                "_deepintshield_guarded",
                False,
            ):
                return True
    except Exception:
        pass
    try:
        from google.adk.plugins.plugin_manager import PluginManager

        return bool(
            getattr(
                PluginManager.run_before_tool_callback,
                "_deepintshield_guarded",
                False,
            )
        )
    except Exception:
        return False


def _authorize(tool: Any, call_args: Any) -> dict[str, Any]:
    if not isinstance(call_args, dict):
        from ..errors import GovernanceConfigurationError

        raise GovernanceConfigurationError(
            framework="google-adk",
            reason="tool execution boundary exposed non-dictionary arguments",
        ) from None
    from ..enforcement import bind_engine, ensure_topology_reported, resolve_engine

    engine = resolve_engine(tool)
    bind_engine(tool, engine)
    name = _tool_name(tool)
    try:
        fp = source_fingerprint(getattr(tool, "func", None) or tool)
    except Exception:
        fp = ""
    ensure_topology_reported(engine, tool, required=True)
    decision = resolve(
        engine,
        name,
        (),
        call_args,
        tool_fingerprint=fp,
        tool_callable=getattr(tool, "func", None) or tool,
    )
    masked = apply_obligations(call_args, decision.obligations)
    if masked is not call_args:
        call_args.clear()
        call_args.update(masked)
    return call_args


def _patch_final_dispatch(functions: Any) -> bool:
    installed = False
    for attr in ("__call_tool_async", "_call_tool_in_thread_pool"):
        original = getattr(functions, attr, None)
        if not callable(original):
            continue
        if getattr(original, "_deepintshield_guarded", False):
            installed = True
            continue
        if not inspect.iscoroutinefunction(original):
            continue

        @functools.wraps(original)
        @public_agentic_boundary
        async def guarded(
            tool: Any,
            args: Any,
            *positional: Any,
            _original=original,
            **kwargs: Any,
        ) -> Any:
            from ..enforcement import resolve_engine
            from ..execution import execution_scope

            engine = resolve_engine(tool)
            with execution_scope(engine, tool, framework="google_adk"):
                _authorize(tool, args)
                return await _original(tool, args, *positional, **kwargs)

        guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
        setattr(functions, attr, guarded)
        installed = True

    original_live = getattr(functions, "__call_tool_live", None)
    if callable(original_live):
        if getattr(original_live, "_deepintshield_guarded", False):
            installed = True
        elif inspect.isasyncgenfunction(original_live):

            @functools.wraps(original_live)
            @public_agentic_boundary
            async def guarded_live(
                tool: Any,
                args: Any,
                *positional: Any,
                **kwargs: Any,
            ) -> Any:
                from ..enforcement import resolve_engine
                from ..execution import execution_scope

                engine = resolve_engine(tool)
                with execution_scope(engine, tool, framework="google_adk"):
                    _authorize(tool, args)
                    async for item in original_live(
                        tool, args, *positional, **kwargs
                    ):
                        yield item

            guarded_live._deepintshield_guarded = True  # type: ignore[attr-defined]
            setattr(functions, "__call_tool_live", guarded_live)
            installed = True
    return installed


def _patch_plugin_manager_fallback() -> bool:
    """Compatibility boundary for ADK versions without the central dispatch
    helpers used by current releases."""
    try:
        from google.adk.plugins.plugin_manager import PluginManager
    except Exception:
        return False
    original = getattr(PluginManager, "run_before_tool_callback", None)
    if not callable(original) or not inspect.iscoroutinefunction(original):
        return False
    if getattr(original, "_deepintshield_guarded", False):
        return True

    @functools.wraps(original)
    @public_agentic_boundary
    async def guarded(self: Any, *, tool: Any, tool_args: Any, **kwargs: Any) -> Any:
        response = await original(
            self,
            tool=tool,
            tool_args=tool_args,
            **kwargs,
        )
        if response is None:
            from ..enforcement import resolve_engine
            from ..execution import execution_scope

            engine = resolve_engine(tool)
            with execution_scope(engine, tool, framework="google_adk"):
                _authorize(tool, tool_args)
        return response

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    PluginManager.run_before_tool_callback = guarded
    return True


def enforce() -> bool:
    """Install default enforcement at ADK's final normal/live/threaded tool
    dispatchers, falling back to its global plugin manager on older releases."""
    from ..enforcement import report_topology_for

    report_topology_for(_EXECUTION_BOUNDARIES)
    try:
        from google.adk.flows.llm_flows import functions
    except Exception:
        functions = None
    if functions is not None and _patch_final_dispatch(functions):
        return True
    return _patch_plugin_manager_fallback()


__all__ = ["plugin", "enforce", "executable_callbacks"]
