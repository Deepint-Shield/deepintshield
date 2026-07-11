"""PydanticAI enforcement - gate the function tools registered on a
``pydantic_ai.Agent`` by wrapping each tool's underlying function.

PydanticAI keeps registered tools in an internal registry whose attribute
name has shifted across versions, so this adapter probes the known shapes
(``_function_tools`` / ``_function_toolset.tools`` / ``tools``). Returns the
same agent, mutated in place.
"""

from __future__ import annotations

import logging
from typing import Any

from ._common import install_method_guard, set_attr, wrap_callable

log = logging.getLogger(__name__)


def enforce(get_engine: Any) -> bool:
    """Non-bypassable PydanticAI enforcement: best-effort patch of the ``Tool``
    class run/call so registered tools are gated at execution. PydanticAI's tool
    internals shift across versions, so this safely no-ops if the shape doesn't
    match (explicit ``govern(agent)`` still works). Idempotent + fail-open."""
    name_fn = lambda self: getattr(self, "name", None) or type(self).__name__
    impl_fn = lambda self: getattr(self, "function", None) or getattr(self, "func", None) or self
    installed = False
    for mod, cls in (("pydantic_ai.tools", "Tool"),):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            continue
        for attr, is_async in (("run", True), ("call", True), ("__call__", False)):
            if install_method_guard(base, attr, get_engine, name_fn, is_async=is_async, impl_fn=impl_fn):
                installed = True
                break
    return installed

_REGISTRY_ATTRS = ("_function_tools", "_function_toolset", "tools")
_FUNC_ATTRS = ("function", "func", "_func")


def shield_agent(agent: Any, *, engine: Any) -> Any:
    registry = _find_registry(agent)
    if registry is None:
        raise TypeError(
            "pydanticai: could not locate the agent's tool registry; "
            "register tools with @agent.tool before calling shield.agentic.pydanticai()"
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
