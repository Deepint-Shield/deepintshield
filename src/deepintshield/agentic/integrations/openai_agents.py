"""OpenAI Agents SDK enforcement - gate each ``FunctionTool`` on an Agent (or
a bare list of tools) by wrapping its async ``on_invoke_tool`` callable.
Mutates in place and returns the same object.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ..gate import resolve

log = logging.getLogger(__name__)


def enforce(get_engine: Any) -> bool:
    """Non-bypassable OpenAI-Agents enforcement: OpenAI-Agents tools hold their
    ``on_invoke_tool`` per-instance (no class method to patch), so we guard at the
    ``Runner`` boundary - instrument every agent's tools just before it runs.
    Idempotent + fail-open. Returns True if installed."""
    try:
        from agents import Runner
    except Exception:
        return False

    def _govern(args: tuple) -> None:
        agent = next((a for a in args if hasattr(a, "tools") or hasattr(a, "on_invoke_tool")), None)
        if agent is None:
            return
        try:
            shield_agent(agent, engine=get_engine())
        except Exception:
            pass

    installed = False
    for attr in ("run", "run_sync", "run_streamed"):
        orig = getattr(Runner, attr, None)
        if orig is None or getattr(orig, "_deepintshield_guarded", False):
            continue
        if attr == "run":
            @functools.wraps(orig)
            async def guarded(*args: Any, _orig=orig, **kwargs: Any) -> Any:
                _govern(args)
                return await _orig(*args, **kwargs)
        else:
            @functools.wraps(orig)
            def guarded(*args: Any, _orig=orig, **kwargs: Any) -> Any:  # type: ignore[misc]
                _govern(args)
                return _orig(*args, **kwargs)
        guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
        try:
            setattr(Runner, attr, staticmethod(guarded) if isinstance(orig, staticmethod) else guarded)
            installed = True
        except Exception:
            continue
    return installed


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
            raise TypeError(
                f"openai_agents: tool {name!r} exposes no on_invoke_tool/func to gate"
            )

    if getattr(original, "_deepintshield_wrapped", False):
        return

    @functools.wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        # on_invoke_tool is (context, input_json) - gate on the input payload.
        resolve(engine, name, args, kwargs)
        return await original(*args, **kwargs)

    wrapped._deepintshield_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(tool, "on_invoke_tool" if hasattr(tool, "on_invoke_tool") else "func", wrapped)
    except Exception:  # frozen dataclass
        object.__setattr__(
            tool, "on_invoke_tool" if hasattr(tool, "on_invoke_tool") else "func", wrapped
        )


__all__ = ["shield_agent"]
