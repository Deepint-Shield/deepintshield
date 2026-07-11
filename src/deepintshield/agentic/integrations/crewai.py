"""CrewAI enforcement - gate each CrewAI ``BaseTool`` so ``decide()`` runs
before the tool body. Accepts a single tool or a list; mutates in place and
returns the same object(s).
"""

from __future__ import annotations

import logging
from typing import Any

from ._common import as_list, install_method_guard, set_attr, wrap_callable

log = logging.getLogger(__name__)


def enforce(get_engine: Any) -> bool:
    """Non-bypassable CrewAI enforcement: patch ``BaseTool.run`` so every CrewAI
    tool is gated by the PDP at execution - no per-tool ``govern()`` needed.
    Idempotent + fail-open. Returns True if installed."""
    base = None
    for mod, cls in (("crewai.tools", "BaseTool"), ("crewai.tools.base_tool", "BaseTool")):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
            break
        except Exception:
            continue
    if base is None:
        return False
    name_fn = lambda self: getattr(self, "name", None) or type(self).__name__
    impl_fn = lambda self: getattr(self, "_run", None) or getattr(self, "func", None) or self
    for attr in ("run", "_run"):
        if install_method_guard(base, attr, get_engine, name_fn, impl_fn=impl_fn):
            return True
    return False


def shield_tools(tools: Any, *, engine: Any) -> Any:
    single = not isinstance(tools, (list, tuple, set))
    wrapped = [_wrap_tool(t, engine) for t in as_list(tools)]
    return wrapped[0] if single else wrapped


def _wrap_tool(tool: Any, engine: Any) -> Any:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
    hooked = False
    # Structured tools expose `func`; class-based BaseTools implement `_run`.
    for attr in ("func", "_run", "run"):
        fn = getattr(tool, attr, None)
        if callable(fn):
            if set_attr(tool, attr, wrap_callable(engine, name, fn)):
                hooked = True
                break
    if not hooked:
        raise TypeError(
            f"crewai: could not find a callable to gate on tool {name!r} "
            "(expected one of .func / ._run / .run)"
        )
    return tool


__all__ = ["shield_tools"]
