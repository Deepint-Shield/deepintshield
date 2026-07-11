"""Non-bypassable enforcement installer for every supported framework.

The per-framework adapters gate tools *when the developer calls* ``govern()`` /
``guard()``. That still leaves a gap: nothing stops code from skipping that call
and running the agent directly. ``install_all`` closes it by monkey-patching each
framework's build/execute boundary the moment a DeepintShield client is created,
so a tool/graph can't run ungoverned - the developer no longer has to remember.

Design rules every framework installer follows:
  * **Lazy engine** - takes a ``get_engine`` callable resolved on first use, so
    installing at client construction never forces the agentic surface to build.
  * **Only if imported** - we patch a framework only when it's already in
    ``sys.modules`` (never force-import a dep the user isn't using).
  * **Idempotent** - re-installing re-binds the provider, never double-wraps.
  * **Fail-open on infra, fail-CLOSED on a verdict** - a gateway hiccup must not
    break the app, but a DENY must still block. Each installer swallows
    infrastructure errors and re-raises ``GuardrailDenied`` / approval timeouts.

Cooperative defense-in-depth: a determined process can un-patch these or call a
tool's raw function object, so the gateway (MCP/LLM in the call path) stays the
authoritative boundary. For ordinary application code this makes enforcement the
default rather than something to remember.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any, Callable

# top-level import name → enforcement integration module (relative to this package)
_FRAMEWORKS: list[tuple[str, str]] = [
    ("langgraph", ".integrations.langgraph"),
    ("crewai", ".integrations.crewai"),
    ("llama_index", ".integrations.llamaindex"),
    ("autogen", ".integrations.autogen"),
    ("autogen_core", ".integrations.autogen"),
    ("pydantic_ai", ".integrations.pydanticai"),
    ("agents", ".integrations.openai_agents"),  # `openai-agents` imports as `agents`
    ("litellm", ".integrations.litellm"),
]


def install_all(get_engine: Callable[[], Any]) -> list[str]:
    """Install enforcement guards for every supported framework currently
    imported. Returns the list of frameworks guarded. Never raises."""
    installed: list[str] = []
    seen: set[str] = set()
    for mod_name, integ in _FRAMEWORKS:
        if mod_name not in sys.modules or integ in seen:
            continue
        seen.add(integ)
        try:
            m = import_module(integ, package=__package__)
            fn = getattr(m, "enforce", None)
            if callable(fn) and fn(get_engine):
                installed.append(mod_name)
        except Exception:  # a single framework's guard never blocks the others
            continue
    return installed


__all__ = ["install_all"]
