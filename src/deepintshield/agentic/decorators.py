"""``shield_tool`` - the one-line wrapper that turns any Python function into
a PEP-gated tool, with unambiguous client resolution for the common
"one client, many decorators" case.

The decorator is import-safe even when no client is bound - it raises a clear
error at first call rather than at import time, so tools can be declared in
modules that don't always have a client available.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

from .errors import public_agentic_boundary, public_agentic_error_boundary
from .execution import execution_scope
from .gate import enforce_call, preflight

@public_agentic_boundary
def set_default_client(client: object) -> None:
    """Register a client for bare ``@shield_tool`` compatibility.

    It deliberately does not replace a global default. A sole live client is
    selected automatically; multiple clients must pass ``client=...`` or use a
    request-local ``shield.agentic.run()`` scope.
    """
    from .enforcement import register_client

    if hasattr(client, "agentic"):
        register_client(client)


def _resolve_engine(client: object):
    if client is None:
        from .enforcement import resolve_engine

        return resolve_engine()
    target = client
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
    agent: str = "",
    permission: str = "",
    object: str = "",
    delegation_id: str = "",
    action: str = "",
    action_class: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory. Wrap any function as a PEP-gated tool.

    Example::

        @shield_tool(tool="db.write", recovery_cost="high")
        def write_ledger(row: dict) -> dict:
            return db.execute(…)

    Args:
        tool:           The tool name registered in DeepintShield's
                        Tools & Tiering page.
        client:         A ``DeepintShield`` / ``shield.agentic`` / engine. A
                        sole registered client is resolved automatically;
                        multiple clients require an explicit client or run scope.
        recovery_cost:  Optional autonomy-budget hint: "low"/"medium"/"high".
        rag_provenance: Optional hint when the call uses RAG output.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Returns BOTH argument groups: a MASK verdict has to reach positional
        # arguments as well, or the caller chooses whether the obligation
        # applies simply by how they spell the call.
        def gated(engine: Any, args: tuple, kwargs: dict) -> "tuple[tuple, dict]":
            return enforce_call(
                engine,
                tool,
                args,
                kwargs,
                recovery_cost=recovery_cost,
                rag_provenance=rag_provenance,
                tool_callable=fn,
                agent=agent,
                permission=permission,
                object=object,
                delegation_id=delegation_id,
                action=action,
                action_class=action_class,
            )

        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any):
                with public_agentic_error_boundary():
                    engine = _resolve_engine(client)
                    preflight(engine, tool, fn)
                    with execution_scope(engine, fn):
                        args, kwargs = gated(engine, args, kwargs)
                        async for item in fn(*args, **kwargs):
                            yield item
            return wrapper

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                with public_agentic_error_boundary():
                    engine = _resolve_engine(client)
                    preflight(engine, tool, fn)
                    with execution_scope(engine, fn):
                        args, kwargs = gated(engine, args, kwargs)
                        return await fn(*args, **kwargs)
            return wrapper

        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any):
                with public_agentic_error_boundary():
                    engine = _resolve_engine(client)
                    preflight(engine, tool, fn)
                    with execution_scope(engine, fn):
                        args, kwargs = gated(engine, args, kwargs)
                        yield from fn(*args, **kwargs)
            return wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                engine = _resolve_engine(client)
                preflight(engine, tool, fn)
                with execution_scope(engine, fn):
                    args, kwargs = gated(engine, args, kwargs)
                    return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["shield_tool", "set_default_client"]
