"""``shield_tool`` - the one-line wrapper that turns any Python function into
a PEP-gated tool, plus a process-global default client for the common
"one client, many decorators" case.

The decorator is import-safe even when no client is bound - it raises a clear
error at first call rather than at import time, so tools can be declared in
modules that don't always have a client available.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from .gate import enforce

# May be a DeepintShield, an AgenticSurface, or an AgenticEngine.
_default_client: Optional[object] = None


def set_default_client(client: object) -> None:
    """Bind a process-wide default so subsequent ``@shield_tool(tool="…")``
    declarations don't need an explicit ``client=…``. Accepts a
    ``DeepintShield``, a ``shield.agentic`` surface, or a raw engine."""
    global _default_client
    _default_client = client


def _resolve_engine(client: object):
    target = client if client is not None else _default_client
    if target is None:
        raise RuntimeError(
            "shield_tool requires a client - pass client=… or call "
            "deepintshield.agentic.set_default_client(shield) once at process start."
        )
    agentic = getattr(target, "agentic", None)
    if agentic is not None:  # a DeepintShield
        return agentic.engine
    engine = getattr(target, "engine", None)
    if engine is not None:  # an AgenticSurface
        return engine
    return target  # assume an AgenticEngine


def shield_tool(
    *,
    tool: str,
    client: Optional[object] = None,
    recovery_cost: str = "",
    rag_provenance: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory. Wrap any function as a PEP-gated tool.

    Example::

        @shield_tool(tool="db.write", recovery_cost="high")
        def write_ledger(row: dict) -> dict:
            return db.execute(…)

    Args:
        tool:           The tool name registered in DeepintShield's
                        Tools & Tiering page.
        client:         A ``DeepintShield`` / ``shield.agentic`` / engine. May
                        also be bound globally via ``set_default_client()``.
        recovery_cost:  Optional autonomy-budget hint: "low"/"medium"/"high".
        rag_provenance: Optional hint when the call uses RAG output.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            engine = _resolve_engine(client)
            kwargs = enforce(
                engine,
                tool,
                args,
                kwargs,
                recovery_cost=recovery_cost,
                rag_provenance=rag_provenance,
            )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["shield_tool", "set_default_client"]
