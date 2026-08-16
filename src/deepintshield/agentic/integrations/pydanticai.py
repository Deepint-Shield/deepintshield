"""PydanticAI enforcement.

Current PydanticAI releases execute validated tool calls through
``ToolManager.execute_tool_call``.  Older releases exposed ``Tool.run`` /
``Tool.call`` instead.  DeepintShield installs the newest available final
execution boundary automatically and retains the old boundary as a compatibility
fallback, so ordinary ``Agent.run*`` calls never require ``.shield()`` or an
explicit adapter call.

The final validated-call boundary is intentional: authorization sees the exact
tool and arguments that PydanticAI is about to execute, and MASK obligations can
replace those arguments before the toolset dispatches them.
"""

from __future__ import annotations

import copy
import functools
import inspect
import logging
from typing import Any

from ..enforcement import report_topology_for
from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ..gate import resolve
from ..obligations import apply_obligations
from ._common import (
    install_method_guard,
    set_attr,
    source_fingerprint,
    wrap_callable,
)

log = logging.getLogger(__name__)

# The agent's first run - every @agent.tool has been registered by then.
_TOPOLOGY_BOUNDARIES = (
    ("pydantic_ai", "Agent", ("run_sync", "run", "run_stream")),
    ("pydantic_ai.agent", "Agent", ("run_sync", "run")),
)


def _tool_manager_guard_active() -> bool:
    try:
        from pydantic_ai.tool_manager import ToolManager

        return bool(
            getattr(
                getattr(ToolManager, "execute_tool_call", None),
                "_deepintshield_guarded",
                False,
            )
        )
    except Exception:
        return False


def _validated_tool_name(validated: Any) -> str:
    call = getattr(validated, "call", None)
    name = getattr(call, "tool_name", None)
    if isinstance(name, str) and name:
        return name
    tool = getattr(validated, "tool", None)
    tool_def = getattr(tool, "tool_def", None)
    name = getattr(tool_def, "name", None)
    if isinstance(name, str) and name:
        return name
    return "tool"


def _validated_tool_impl(manager: Any, validated: Any, name: str) -> Any:
    """Best-effort source identity for a current ``ToolsetTool``.

    ``ToolsetTool`` deliberately carries only a schema and its owning toolset.
    Function toolsets retain the user-facing ``Tool`` in a keyed ``tools``
    mapping, so look through the manager and resolved-tool toolsets without
    depending on one private layout.
    """
    resolved = getattr(validated, "tool", None)
    holders = (
        getattr(resolved, "toolset", None),
        getattr(manager, "toolset", None),
    )
    for holder in holders:
        raw = getattr(holder, "tools", None)
        candidate = raw.get(name) if isinstance(raw, dict) else None
        if candidate is None:
            continue
        for attr in ("function", "func", "_func"):
            fn = getattr(candidate, attr, None)
            if callable(fn):
                return fn
        if callable(candidate):
            return candidate
    return resolved


def _with_validated_args(validated: Any, masked: dict[str, Any]) -> Any:
    """Return a shallow validated-call copy with policy-transformed arguments."""
    try:
        updated = copy.copy(validated)
        setattr(updated, "validated_args", masked)
        return updated
    except Exception:
        raise GovernanceConfigurationError(
            framework="pydantic-ai",
            reason="MASK obligation cannot rewrite validated tool arguments",
        ) from None


def _install_tool_manager_guard() -> bool:
    """Patch PydanticAI >=2's final validated tool execution boundary."""
    try:
        from pydantic_ai.tool_manager import ToolManager
    except Exception:
        return False

    original = getattr(ToolManager, "execute_tool_call", None)
    if not callable(original) or not inspect.iscoroutinefunction(original):
        return False
    if getattr(original, "_deepintshield_guarded", False):
        return True

    @functools.wraps(original)
    @public_agentic_boundary
    async def guarded(
        self: Any,
        validated: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Invalid, output, and external calls cannot execute an application tool
        # through this boundary. Preserve PydanticAI's native error/output path
        # without manufacturing an authorization event for work that never ran.
        if not bool(getattr(validated, "args_valid", False)):
            return await original(self, validated, *args, **kwargs)
        resolved_tool = getattr(validated, "tool", None)
        tool_def = getattr(resolved_tool, "tool_def", None)
        kind = getattr(tool_def, "kind", "function")
        if kind in {"output", "external"}:
            return await original(self, validated, *args, **kwargs)
        call_args = getattr(validated, "validated_args", None)
        if not isinstance(call_args, dict):
            raise GovernanceConfigurationError(
                framework="pydantic-ai",
                reason="validated tool execution exposed non-dictionary arguments",
            )

        from ..enforcement import bind_engine, resolve_engine
        from ..execution import current_execution, execution_scope

        # Prefer a framework object already bound by Agent.run*. ToolManager is
        # ephemeral in current PydanticAI, while its toolset is normally the
        # recursively-bound agent toolset. A current execution frame covers
        # wrapper/copy toolsets and preserves request-local multi-client routing.
        owners = (
            self,
            getattr(resolved_tool, "toolset", None),
            getattr(self, "toolset", None),
            resolved_tool,
        )
        engine = None
        for owner in owners:
            if owner is None or getattr(owner, "_deepintshield_engine", None) is None:
                continue
            engine = resolve_engine(owner)
            break
        if engine is None:
            frame = current_execution()
            engine = frame.engine if frame is not None else resolve_engine(self)
        bind_engine(self, engine)
        if resolved_tool is not None:
            bind_engine(resolved_tool, engine, recursive=True)

        name = _validated_tool_name(validated)
        implementation = _validated_tool_impl(self, validated, name)
        try:
            fingerprint = source_fingerprint(implementation)
        except Exception:
            fingerprint = ""
        with execution_scope(engine, self, framework="pydanticai"):
            decision = resolve(
                engine,
                name,
                (),
                call_args,
                tool_fingerprint=fingerprint,
                tool_callable=implementation,
            )
            masked = apply_obligations(call_args, decision.obligations)
            if not isinstance(masked, dict):
                raise GovernanceConfigurationError(
                    framework="pydantic-ai",
                    reason="MASK obligation produced unsupported tool arguments",
                )
            forwarded = (
                _with_validated_args(validated, masked)
                if masked is not call_args
                else validated
            )
            return await original(self, forwarded, *args, **kwargs)

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    try:
        ToolManager.execute_tool_call = guarded
    except Exception:
        return False
    return True


def enforce() -> bool:
    """Install the newest supported non-bypassable PydanticAI boundary.

    Unsupported versions return ``False`` so client/import setup fails closed
    instead of presenting an ungoverned ``Agent.run*`` surface. Also reports
    the agent topology to the GAF registry on first run.
    """
    report_topology_for(_TOPOLOGY_BOUNDARIES)
    if _install_tool_manager_guard():
        return True

    # PydanticAI <=1 compatibility: these releases execute on Tool itself.
    name_fn = lambda self: getattr(self, "name", None) or type(self).__name__
    impl_fn = lambda self: (
        getattr(self, "function", None) or getattr(self, "func", None) or self
    )
    installed = False
    for mod, cls in (("pydantic_ai.tools", "Tool"),):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            continue
        for attr, is_async in (("run", True), ("call", True), ("__call__", False)):
            if install_method_guard(
                base, attr, name_fn, is_async=is_async, impl_fn=impl_fn
            ):
                installed = True
                break
    return installed


_REGISTRY_ATTRS = ("_function_tools", "_function_toolset", "tools")
_FUNC_ATTRS = ("function", "func", "_func")


@public_agentic_boundary
def shield_agent(agent: Any, *, engine: Any) -> Any:
    # ``shield.agentic.pydanticai(agent)`` remains source-compatible, but the
    # automatic ToolManager boundary already covers every tool. Bind ownership
    # only; wrapping the underlying function would decide twice.
    if _tool_manager_guard_active():
        from ..enforcement import bind_engine

        if not bind_engine(agent, engine, recursive=True):
            raise GovernanceConfigurationError(
                framework="pydantic-ai",
                reason="agent cannot retain its DeepintShield engine binding",
            )
        return agent

    registry = _find_registry(agent)
    if registry is None:
        raise GovernanceConfigurationError(
            framework="pydantic-ai",
            reason="agent exposes no enforceable tool registry",
        )
    tools = registry.values() if isinstance(registry, dict) else registry
    hooked = 0
    for tool in tools:
        if _wrap_tool(tool, engine):
            hooked += 1
    if hooked == 0:
        log.warning("pydanticai: no gateable tool functions found on the agent")
    return agent


def _find_registry(agent: Any):
    for attr in _REGISTRY_ATTRS:
        obj = getattr(agent, attr, None)
        if obj is None:
            continue
        # _function_toolset wraps the dict in a `.tools` attribute.
        inner = getattr(obj, "tools", obj)
        if isinstance(inner, dict) and inner:
            return inner
        if isinstance(inner, (list, tuple)) and inner:
            return list(inner)
    return None


def _wrap_tool(tool: Any, engine: Any) -> bool:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
    for attr in _FUNC_ATTRS:
        fn = getattr(tool, attr, None)
        if callable(fn):
            return set_attr(tool, attr, wrap_callable(engine, name, fn))
    return False


__all__ = ["shield_agent"]
