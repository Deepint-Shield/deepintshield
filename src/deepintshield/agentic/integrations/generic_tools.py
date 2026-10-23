"""Explicit enforcement for framework-neutral tool objects.

Structured tools expose ``func``; class-based tools implement ``_run`` or
``run``. This helper preserves the generic ``guard(target)``/``govern(target)``
path without installing hooks for an unsupported framework.
"""

from __future__ import annotations

from typing import Any

from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ._common import as_list, set_attr, wrap_callable


@public_agentic_boundary
def shield_tools(tools: Any, *, engine: Any) -> Any:
    single = not isinstance(tools, (list, tuple, set))
    wrapped = [_wrap_tool(t, engine) for t in as_list(tools)]
    return wrapped[0] if single else wrapped


def _wrap_tool(tool: Any, engine: Any) -> Any:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
    for attr in ("func", "_run", "run"):
        fn = getattr(tool, attr, None)
        if callable(fn) and set_attr(tool, attr, wrap_callable(engine, name, fn)):
            return tool
    raise GovernanceConfigurationError(
        framework="generic",
        reason=f"tool {name!r} exposes no enforceable callable",
    )


__all__ = ["shield_tools"]
