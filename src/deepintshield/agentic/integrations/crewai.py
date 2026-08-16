"""CrewAI enforcement - gate each CrewAI ``BaseTool`` so ``decide()`` runs
before the tool body. Accepts a single tool or a list; mutates in place and
returns the same object(s).
"""

from __future__ import annotations

import logging
from typing import Any

from ..enforcement import report_topology_for
from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ._common import (
    CallbackInventory,
    as_list,
    explicit_callback_inventory,
    install_method_guard,
    set_attr,
    wrap_callable,
)

log = logging.getLogger(__name__)

# Where a Crew starts running - the natural point at which its agents, tasks and
# tools are fully assembled and can be reported to the GAF registry.
_TOPOLOGY_BOUNDARIES = (
    ("crewai", "Crew", ("kickoff", "kickoff_async", "kickoff_for_each", "kickoff_for_each_async")),
    ("crewai.crew", "Crew", ("kickoff", "kickoff_async")),
)

_CALLBACK_FIELDS = (
    # Crew
    "before_kickoff_callbacks",
    "after_kickoff_callbacks",
    "task_callback",
    "step_callback",
    # Agent / Task
    "callbacks",
    "callback",
)


def executable_callbacks(owner: Any, *, prefix: str = "crewai") -> CallbackInventory:
    """Return only CrewAI's documented application callback fields.

    Tool bodies are inventoried separately by the registry.  In particular we
    do not inspect CrewAI executors, event buses, or generated framework
    callbacks, which keeps a callback-free declarative crew scan-free.
    """
    return explicit_callback_inventory(
        owner,
        _CALLBACK_FIELDS,
        framework_modules=("crewai",),
        prefix=prefix,
    )


def enforce() -> bool:
    """Non-bypassable CrewAI enforcement: patch ``BaseTool.run`` so every CrewAI
    tool is gated by the PDP at execution - no per-tool ``govern()`` needed.
    Idempotent and fail-closed at execution. Returns True if installed.

    Also reports the crew's topology to the GAF registry on first kickoff, so
    Registry fills itself in for CrewAI the same way it does for LangGraph."""
    report_topology_for(_TOPOLOGY_BOUNDARIES)
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
        if install_method_guard(base, attr, name_fn, impl_fn=impl_fn):
            return True
    return False


@public_agentic_boundary
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
        raise GovernanceConfigurationError(
            framework="crewai",
            reason=f"tool {name!r} exposes no enforceable callable",
        )
    return tool


__all__ = ["shield_tools", "executable_callbacks"]
