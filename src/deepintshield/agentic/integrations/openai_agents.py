"""OpenAI Agents SDK enforcement - gate each ``FunctionTool`` on an Agent (or
a bare list of tools) by wrapping its async ``on_invoke_tool`` callable.
Mutates in place and returns the same object.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

from ..enforcement import bind_engine, report_topology, resolve_engine
from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ..execution import execution_scope
from ..gate import resolve

log = logging.getLogger(__name__)


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
    tools = getattr(target, "tools", None)
    if tools is None:
        tools = target if isinstance(target, (list, tuple)) else [target]
    for tool in tools:
        _wrap_tool(tool, engine)
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
