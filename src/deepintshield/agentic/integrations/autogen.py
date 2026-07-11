"""AutoGen (AG2 / autogen-core) enforcement - gate AutoGen ``FunctionTool``
objects or bare callables registered as tools.

AutoGen's tool surface varies across versions, so this adapter probes the
known shapes:
    * ``autogen_core.tools.FunctionTool`` → wrap its underlying ``_func`` /
      ``func`` callable (``run``/``run_json`` delegate to it).
    * an ``AssistantAgent`` exposing ``.tools`` → wrap each tool.
    * a bare callable registered with ``register_function`` → wrap directly.
Accepts a single target or a list; returns the same object(s).
"""

from __future__ import annotations

import logging
from typing import Any

from ._common import as_list, install_method_guard, set_attr, wrap_callable

log = logging.getLogger(__name__)


def enforce(get_engine: Any) -> bool:
    """Non-bypassable AutoGen enforcement: patch ``FunctionTool.run``/``run_json``
    so every tool is gated at execution. Idempotent + fail-open."""
    name_fn = lambda self: getattr(self, "name", None) or type(self).__name__
    impl_fn = lambda self: getattr(self, "_func", None) or getattr(self, "func", None) or self
    installed = False
    for mod, cls in (
        ("autogen_core.tools", "FunctionTool"),
        ("autogen_core.tools", "BaseTool"),
        ("autogen.tools", "FunctionTool"),
    ):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            continue
        for attr in ("run", "run_json"):
            if install_method_guard(base, attr, get_engine, name_fn, is_async=True, impl_fn=impl_fn):
                installed = True
    return installed


def shield_tools(target: Any, *, engine: Any) -> Any:
    tools = getattr(target, "tools", None)
    if tools is not None:
        for tool in tools:
            _wrap_tool(tool, engine)
        return target

    single = not isinstance(target, (list, tuple, set))
    wrapped = [_wrap_tool(t, engine) for t in as_list(target)]
    return wrapped[0] if single else wrapped


def _wrap_tool(tool: Any, engine: Any) -> Any:
    # A bare callable registered as a tool.
    if callable(tool) and not hasattr(tool, "run") and not hasattr(tool, "_func"):
        name = getattr(tool, "__name__", "unknown")
        return wrap_callable(engine, name, tool)

    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
    for attr in ("_func", "func", "run_json", "run"):
        fn = getattr(tool, attr, None)
        if callable(fn):
            if set_attr(tool, attr, wrap_callable(engine, name, fn)):
                return tool
    raise TypeError(
        f"autogen: could not find a callable to gate on tool {name!r} "
        "(expected one of ._func / .func / .run_json / .run)"
    )


__all__ = ["shield_tools"]
