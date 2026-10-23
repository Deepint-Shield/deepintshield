"""OpenAI Agents SDK enforcement - gate each ``FunctionTool`` on an Agent (or
a bare list of tools) by wrapping its async ``on_invoke_tool`` callable.
Mutates in place and returns the same object.
"""

from __future__ import annotations

import functools
import inspect
import logging
import sys
import types
import weakref
from typing import Any

from ..enforcement import bind_engine, report_topology, resolve_engine
from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ..execution import execution_scope
from ..gate import resolve
from ._common import (
    CallbackInventory,
    callback_method_inventory,
    explicit_callback_inventory,
    merge_callback_inventories,
)

log = logging.getLogger(__name__)

_MAX_HANDOFF_AGENTS = 64
_AGENT_HOOK_METHODS = (
    "on_start", "on_end", "on_handoff", "on_tool_start", "on_tool_end",
    "on_llm_start", "on_llm_end",
)


def _handoffs(owner: Any) -> tuple[tuple[Any, ...], bool]:
    try:
        values = getattr(owner, "handoffs", None)
    except Exception:
        return (), True
    if values is None:
        return (), False
    if not isinstance(values, (list, tuple)):
        return (), True
    return tuple(values[:_MAX_HANDOFF_AGENTS]), len(values) > _MAX_HANDOFF_AGENTS


def _factory_handoff_binding(value: Any, name: str) -> Any:
    """Read one binding from the installed SDK's own handoff factory.

    Native Handoff keeps ``on_handoff`` in a generated closure rather than a
    public field. Code and globals must match the installed factory; copying a
    module/name marker cannot make an application function a trusted wrapper.
    """
    module = sys.modules.get("agents.handoffs")
    factory = getattr(module, "handoff", None)
    if not inspect.isfunction(factory):
        return None
    if isinstance(value, functools.partial):
        if value.func is not getattr(module, "_invoke_handoff_with_redaction", None):
            return None
        if len(value.args) != 1:
            return None
        value = value.args[0]
    if not inspect.isfunction(value) or value.__globals__ is not vars(module):
        return None
    generated = (
        constant for constant in factory.__code__.co_consts
        if isinstance(constant, types.CodeType)
    )
    if not any(value.__code__ is code for code in generated):
        return None
    cells = value.__closure__ or ()
    for key, cell in zip(value.__code__.co_freevars, cells):
        if key == name:
            try:
                return cell.cell_contents
            except ValueError:
                return None
    return None


def handoff_agents(owner: Any) -> tuple[tuple[Any, ...], bool]:
    """Return direct declared handoff agents, with a bounded failure flag."""
    handoffs, incomplete = _handoffs(owner)
    children: list[Any] = []
    seen: set[int] = set()
    for handoff in handoffs:
        try:
            if hasattr(handoff, "tools") and hasattr(handoff, "handoffs"):
                child = handoff
            else:
                reference = getattr(handoff, "_agent_ref", None)
                child = reference() if isinstance(reference, weakref.ReferenceType) else None
                if child is None:
                    child = _factory_handoff_binding(
                        getattr(handoff, "on_invoke_handoff", None), "agent",
                    )
                if child is None:
                    incomplete = True
                    continue
        except Exception:
            incomplete = True
            continue
        if id(child) not in seen:
            seen.add(id(child))
            children.append(child)
    return tuple(children), incomplete


def _guardrail_callbacks(owner: Any, fields: tuple[str, ...], prefix: str) -> CallbackInventory:
    inventories: list[CallbackInventory] = []
    for field in fields:
        try:
            guards = getattr(owner, field, None)
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        if guards is None:
            continue
        if not isinstance(guards, (list, tuple)):
            inventories.append(CallbackInventory(incomplete=True))
            continue
        for index, guard in enumerate(guards[:_MAX_HANDOFF_AGENTS]):
            inventories.append(explicit_callback_inventory(
                guard, ("guardrail_function",), framework_modules=("agents",),
                prefix=f"{prefix}.{field}[{index}]",
            ))
        if len(guards) > _MAX_HANDOFF_AGENTS:
            inventories.append(CallbackInventory(incomplete=True))
    return merge_callback_inventories(*inventories)


def executable_callbacks(owner: Any, *, prefix: str = "openai_agents") -> CallbackInventory:
    """Inventory this agent's supported callbacks, excluding child agents."""
    framework_modules = ("agents",)
    inventories = [explicit_callback_inventory(
        owner, ("instructions", "prompt", "tool_use_behavior"),
        framework_modules=framework_modules, prefix=prefix,
    )]
    inventories.append(_guardrail_callbacks(
        owner, ("input_guardrails", "output_guardrails"), prefix,
    ))
    try:
        tools = getattr(owner, "tools", None)
    except Exception:
        tools = None
        inventories.append(CallbackInventory(incomplete=True))
    if tools is not None:
        if not isinstance(tools, (list, tuple)):
            inventories.append(CallbackInventory(incomplete=True))
        else:
            for index, tool in enumerate(tools[:_MAX_HANDOFF_AGENTS]):
                tool_prefix = f"{prefix}.tools[{index}]"
                inventories.append(explicit_callback_inventory(
                    tool,
                    ("is_enabled", "needs_approval", "timeout_error_function",
                     "custom_data_extractor", "_failure_error_function", "failure_error_function"),
                    framework_modules=framework_modules, prefix=tool_prefix,
                ))
                inventories.append(_guardrail_callbacks(
                    tool, ("tool_input_guardrails", "tool_output_guardrails"), tool_prefix,
                ))
            if len(tools) > _MAX_HANDOFF_AGENTS:
                inventories.append(CallbackInventory(incomplete=True))
    try:
        hooks = getattr(owner, "hooks", None)
    except Exception:
        inventories.append(CallbackInventory(incomplete=True))
    else:
        inventories.append(callback_method_inventory(
            hooks, _AGENT_HOOK_METHODS, framework_modules=framework_modules,
            prefix=f"{prefix}.hooks",
        ))
    handoffs, incomplete = _handoffs(owner)
    for index, handoff in enumerate(handoffs):
        handoff_prefix = f"{prefix}.handoffs[{index}]"
        inventories.append(explicit_callback_inventory(
            handoff, ("on_handoff", "on_invoke_handoff", "input_filter", "is_enabled"),
            framework_modules=framework_modules, prefix=handoff_prefix,
        ))
        # The native factory hides these two application callbacks in its
        # own dispatch functions. Inspect only the exact supported bindings.
        for field, binding in (("on_invoke_handoff", "on_handoff"), ("is_enabled", "is_enabled")):
            try:
                callback = _factory_handoff_binding(getattr(handoff, field, None), binding)
            except Exception:
                incomplete = True
                continue
            if callable(callback):
                inventories.append(explicit_callback_inventory(
                    types.SimpleNamespace(callback=callback), ("callback",),
                    framework_modules=framework_modules,
                    prefix=f"{handoff_prefix}.{binding}",
                ))
    if incomplete:
        inventories.append(CallbackInventory(incomplete=True))
    return merge_callback_inventories(*inventories)


def enforce() -> bool:
    """Non-bypassable OpenAI-Agents enforcement: OpenAI-Agents tools hold their
    ``on_invoke_tool`` per-instance (no class method to patch), so we guard at the
    ``Runner`` boundary - instrument every agent's tools just before it runs.
    Idempotent and fail-closed. Returns True if installed."""
    try:
        from agents import Runner
    except Exception:
        return False

    def _govern(args: tuple, kwargs: dict) -> tuple[Any, Any]:
        values = (*args, *kwargs.values())
        agent = next(
            (
                value
                for value in values
                if hasattr(value, "tools") or hasattr(value, "on_invoke_tool")
            ),
            None,
        )
        if agent is None:
            raise GovernanceConfigurationError(
                framework="openai-agents",
                reason="Runner call exposes no governable agent",
            )
        engine = resolve_engine(agent)
        bind_engine(agent, engine, recursive=True)
        # Instrumentation is mandatory. Any unsupported/frozen tool blocks the
        # run instead of silently executing outside the PDP.
        shield_agent(agent, engine=engine)
        # Same boundary, second job: report the agent + its tools to the GAF
        # registry so Registry populates itself for OpenAI-Agents too.
        # First-run capture is bounded and completes before authorization so an
        # unknown native agent is visible in the dashboard immediately.
        report_topology(engine, agent, sync=True, required=True)
        return engine, agent

    installed = False
    for attr in ("run", "run_sync", "run_streamed"):
        descriptor = inspect.getattr_static(Runner, attr, None)
        orig = getattr(Runner, attr, None)
        if orig is None or getattr(orig, "_deepintshield_guarded", False):
            continue
        if attr == "run":
            @functools.wraps(orig)
            @public_agentic_boundary
            async def guarded(*args: Any, _orig=orig, **kwargs: Any) -> Any:
                engine, agent = _govern(args, kwargs)
                with execution_scope(
                    engine, agent, framework="openai_agents"
                ):
                    return await _orig(*args, **kwargs)
        else:
            @functools.wraps(orig)
            @public_agentic_boundary
            def guarded(*args: Any, _orig=orig, **kwargs: Any) -> Any:  # type: ignore[misc]
                engine, agent = _govern(args, kwargs)
                with execution_scope(
                    engine, agent, framework="openai_agents"
                ):
                    return _orig(*args, **kwargs)
        guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
        try:
            # ``Runner`` exposes class/static entry points across SDK versions.
            # `getattr` above already binds a classmethod, so keep the public
            # call shape with a static wrapper; plain instance methods remain
            # descriptors and receive `self`.
            replacement = (
                staticmethod(guarded)
                if isinstance(descriptor, (staticmethod, classmethod))
                else guarded
            )
            setattr(Runner, attr, replacement)
            installed = True
        except Exception:
            continue
    return installed


@public_agentic_boundary
def shield_agent(target: Any, *, engine: Any) -> Any:
    queue = [target]
    seen: set[int] = set()
    while queue:
        owner = queue.pop(0)
        if id(owner) in seen:
            continue
        if len(seen) >= _MAX_HANDOFF_AGENTS:
            raise GovernanceConfigurationError(
                framework="openai-agents", reason="handoff agent inventory exceeds its bound",
            )
        seen.add(id(owner))
        tools = getattr(owner, "tools", None)
        if tools is None:
            tools = owner if isinstance(owner, (list, tuple)) else [owner]
        for tool in tools:
            _wrap_tool(tool, engine)
        children, incomplete = handoff_agents(owner)
        if incomplete:
            raise GovernanceConfigurationError(
                framework="openai-agents", reason="handoff agent inventory is incomplete",
            )
        queue.extend(child for child in children if id(child) not in seen)
    return target


def _wrap_tool(tool: Any, engine: Any) -> None:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
    original = getattr(tool, "on_invoke_tool", None)
    if not callable(original):
        # Plain decorated function tools may expose the raw callable instead.
        original = getattr(tool, "func", None)
        if not callable(original):
            raise GovernanceConfigurationError(
                framework="openai-agents",
                reason=f"tool {name!r} exposes no enforceable callable",
            )

    if getattr(original, "_deepintshield_wrapped", False):
        bound = getattr(original, "_deepintshield_engine", engine)
        if bound is not engine:
            raise GovernanceConfigurationError(
                framework="openai_agents",
                reason=f"tool {name!r} is already bound to another SDK client",
            )
        return

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        @public_agentic_boundary
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            with execution_scope(
                engine, tool, framework="openai_agents"
            ):
                # on_invoke_tool is (context, input_json) - gate on the input payload.
                resolve(engine, name, args, kwargs, tool_callable=original)
                return await original(*args, **kwargs)
    else:
        @functools.wraps(original)
        @public_agentic_boundary
        def wrapped(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            with execution_scope(
                engine, tool, framework="openai_agents"
            ):
                resolve(engine, name, args, kwargs, tool_callable=original)
                return original(*args, **kwargs)

    wrapped._deepintshield_wrapped = True  # type: ignore[attr-defined]
    wrapped._deepintshield_engine = engine  # type: ignore[attr-defined]
    try:
        setattr(tool, "on_invoke_tool" if hasattr(tool, "on_invoke_tool") else "func", wrapped)
    except Exception:  # frozen dataclass
        try:
            object.__setattr__(
                tool,
                "on_invoke_tool" if hasattr(tool, "on_invoke_tool") else "func",
                wrapped,
            )
        except Exception:
            raise GovernanceConfigurationError(
                framework="openai-agents",
                reason="tool callable cannot be instrumented",
            ) from None


__all__ = ["shield_agent"]
