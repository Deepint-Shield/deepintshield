"""LangGraph enforcement - gate every node in a compiled graph through the PDP.

Drop-in: no per-tool decorator, no graph-shape change. Mutates the graph in
place and returns it so existing ``invoke()`` code is unchanged. The node name
is the governed tool name the PDP checks.

Interception is done at the node's underlying callable. In current LangGraph a
compiled node is a ``PregelNode`` whose ``.bound`` is a ``RunnableCallable``
holding the user function in ``.func`` / ``.afunc`` - wrapping those intercepts
every invocation regardless of how the Pregel loop dispatches it (the older
approach of wrapping ``runnable.invoke`` silently missed plain function nodes,
because ``PregelNode.runnable`` is ``None``). We wrap ``func``/``afunc`` first
and fall back to ``invoke``/``ainvoke`` for ToolNodes / custom runnables.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ..gate import resolve
from ..obligations import apply_obligations
from ._common import callable_tool_name, source_fingerprint

log = logging.getLogger(__name__)

_RESERVED = ("__start__", "__end__")


def shield_graph(graph: Any, *, engine: Any) -> Any:
    try:
        from langgraph.graph.state import CompiledStateGraph  # noqa: F401
    except Exception as exc:  # pragma: no cover - install-time signal
        raise ImportError(
            "langgraph is required. Install with: pip install 'deepintshield[langgraph]'"
        ) from exc

    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        raise ValueError(f"shield_graph: expected a compiled LangGraph (got {type(graph)!r})")

    for name, node in nodes.items():
        if name in _RESERVED:
            continue
        if not _instrument_node(engine, name, node):
            log.warning("shield_graph: could not gate node %r - skipped", name)
    return graph


def _instrument_node(engine: Any, name: str, node: Any) -> bool:
    """Wrap the node's underlying callable so every invocation is gated. Returns
    True once at least one entry point was wrapped (or was already wrapped)."""
    holders = [getattr(node, "bound", None), getattr(node, "runnable", None), node]
    # Idempotency: if any entry point is already gated (e.g. the compile-guard ran,
    # then govern() was called too), the node is governed - do NOT add a second,
    # coarser invoke-level gate (which would re-key on the node label).
    for holder in holders:
        if holder is None:
            continue
        for attr in ("func", "afunc", "invoke", "ainvoke"):
            fn = getattr(holder, attr, None)
            if callable(fn) and getattr(fn, "_deepintshield_wrapped", False):
                return True
    # Preferred: the RunnableCallable's user function (.func / .afunc).
    for holder in holders:
        if holder is None:
            continue
        wrapped = False
        for attr, is_async in (("func", False), ("afunc", True)):
            fn = getattr(holder, attr, None)
            if callable(fn) and not getattr(fn, "_deepintshield_wrapped", False):
                # Govern the IMPLEMENTATION, not the node label: the tool identity
                # is the function's own name and we bind to its source fingerprint.
                tool_name = callable_tool_name(fn, name)
                fp = source_fingerprint(fn)
                try:
                    setattr(holder, attr, _wrap_fn(engine, tool_name, fn, is_async=is_async, tool_fingerprint=fp))
                    wrapped = True
                except Exception:  # pragma: no cover - frozen attr
                    pass
        if wrapped:
            return True
    # Fallback: ToolNode / custom runnable - wrap invoke / ainvoke.
    for holder in holders:
        if holder is None:
            continue
        wrapped = False
        for attr, is_async in (("invoke", False), ("ainvoke", True)):
            fn = getattr(holder, attr, None)
            if callable(fn) and not getattr(fn, "_deepintshield_wrapped", False):
                try:
                    setattr(holder, attr, _wrap_fn(engine, name, fn, is_async=is_async))
                    wrapped = True
                except Exception:  # pragma: no cover - frozen attr / bound method
                    pass
        if wrapped:
            return True
    return False


def _wrap_fn(engine: Any, tool_name: str, fn: Any, *, is_async: bool, tool_fingerprint: str = ""):
    """Gate one node callable: ask the PDP (raises on a blocking verdict), apply
    any MASK obligations to a dict-shaped state, then run the original. Carries
    the source fingerprint so the decision is bound to the code."""
    if is_async:
        @functools.wraps(fn)
        async def awrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
            decision = resolve(engine, tool_name, (input_value,), {}, tool_fingerprint=tool_fingerprint)
            if isinstance(input_value, dict):
                input_value = apply_obligations(input_value, decision.obligations)
            return await fn(input_value, *args, **kwargs)

        awrapper._deepintshield_wrapped = True  # type: ignore[attr-defined]
        return awrapper

    @functools.wraps(fn)
    def wrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
        decision = resolve(engine, tool_name, (input_value,), {}, tool_fingerprint=tool_fingerprint)
        if isinstance(input_value, dict):
            input_value = apply_obligations(input_value, decision.obligations)
        return fn(input_value, *args, **kwargs)

    wrapper._deepintshield_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def enforce(get_engine: Any) -> bool:
    """Make enforcement non-bypassable: monkey-patch ``StateGraph.compile`` so
    EVERY compiled graph is governed (nodes gated + blueprint registered) before
    it can be invoked. Closes the "forgot to call ``govern()`` / kept a reference
    to the un-governed object / invoked before governing" gap - after ``compile``,
    ``invoke`` always passes through the PDP.

    ``get_engine`` is a zero-arg callable returning the PDP engine, resolved
    lazily on each compile so installing the guard at client construction never
    forces the agentic surface to build until a graph is actually compiled.

    Cooperative defense-in-depth: a determined process can still un-patch this or
    call a node's function object directly, so the gateway (MCP/LLM in the call
    path) remains the authoritative boundary. But for ordinary application code
    this guarantees no compiled graph runs ungoverned.

    Idempotent (re-binds the provider on re-install) and fail-open (a patch /
    instrument error never breaks ``compile``). Returns True if installed.
    """
    try:
        from langgraph.graph.state import StateGraph
    except Exception:  # langgraph not installed - nothing to guard
        return False

    existing = getattr(StateGraph, "compile", None)
    if getattr(existing, "_deepintshield_guarded", False):
        existing._deepintshield_get_engine = get_engine  # type: ignore[attr-defined]
        return True

    orig_compile = StateGraph.compile

    @functools.wraps(orig_compile)
    def guarded_compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        app = orig_compile(self, *args, **kwargs)
        provider = getattr(guarded_compile, "_deepintshield_get_engine", None)
        if provider is not None and not getattr(app, "_deepintshield_governed", False):
            try:
                eng = provider()
                shield_graph(app, engine=eng)
                try:
                    from ..manifest import describe
                    eng.register_blueprint(describe(app))
                except Exception:  # blueprint registration is best-effort
                    pass
                try:
                    setattr(app, "_deepintshield_governed", True)
                except Exception:
                    pass
            except Exception:  # never let governance break a compile
                log.warning("enforce(langgraph): could not govern compiled graph", exc_info=True)
        return app

    guarded_compile._deepintshield_guarded = True  # type: ignore[attr-defined]
    guarded_compile._deepintshield_get_engine = get_engine  # type: ignore[attr-defined]
    StateGraph.compile = guarded_compile  # type: ignore[assignment]
    return True


# Back-compat alias.
enforce_compile = enforce

__all__ = ["shield_graph", "enforce", "enforce_compile"]
