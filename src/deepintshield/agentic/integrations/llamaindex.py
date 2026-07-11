"""LlamaIndex enforcement - gate each ``FunctionTool`` / ``BaseTool`` by
wrapping its ``call`` (and async ``acall``) method. Accepts a single tool or
a list; mutates in place and returns the same object(s).
"""

from __future__ import annotations

import logging
from typing import Any

from ._common import as_list, install_method_guard, set_attr, wrap_callable

log = logging.getLogger(__name__)


def enforce(get_engine: Any) -> bool:
    """Non-bypassable LlamaIndex enforcement: patch the tool base ``call``/``acall``
    so every tool is gated at execution. Idempotent + fail-open."""
    name_fn = lambda self: (
        getattr(getattr(self, "metadata", None), "name", None)
        or getattr(self, "name", None)
        or type(self).__name__
    )
    impl_fn = lambda self: getattr(self, "fn", None) or self
    installed = False
    for mod, cls in (
        ("llama_index.core.tools", "FunctionTool"),
        ("llama_index.core.tools.function_tool", "FunctionTool"),
        ("llama_index.core.tools", "BaseTool"),
    ):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            continue
        if install_method_guard(base, "call", get_engine, name_fn, impl_fn=impl_fn):
            installed = True
        install_method_guard(base, "acall", get_engine, name_fn, is_async=True, impl_fn=impl_fn)
    return installed


def shield_tools(tools: Any, *, engine: Any) -> Any:
    single = not isinstance(tools, (list, tuple, set))
    wrapped = [_wrap_tool(t, engine) for t in as_list(tools)]
    return wrapped[0] if single else wrapped


def _tool_name(tool: Any) -> str:
    meta = getattr(tool, "metadata", None)
    if meta is not None and getattr(meta, "name", None):
        return meta.name
    return getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")


def _wrap_tool(tool: Any, engine: Any) -> Any:
    name = _tool_name(tool)
    hooked = False
    for attr in ("call", "acall", "__call__", "fn"):
        fn = getattr(tool, attr, None)
        if callable(fn):
            if set_attr(tool, attr, wrap_callable(engine, name, fn)):
                hooked = True
                if attr in ("call", "acall"):
                    continue  # wrap both sync + async entry points if present
                break
    if not hooked:
        raise TypeError(
            f"llamaindex: could not find a callable to gate on tool {name!r}"
        )
    return tool


__all__ = ["shield_tools"]
