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
import inspect
import logging
import threading
from typing import Any

from ..enforcement import (
    bind_engine,
    ensure_topology_reported,
    report_topology,
    resolve_engine,
    run_detached,
)
from ..errors import (
    DeepIntShieldError,
    GovernanceConfigurationError,
    public_agentic_boundary,
    public_agentic_error_boundary,
)
from ..execution import execution_scope
from ..gate import resolve
from ..obligations import apply_obligations
from ._common import callable_tool_name, source_fingerprint

log = logging.getLogger(__name__)

_RESERVED = ("__start__", "__end__")
_GOVERNANCE_TOKEN = object()
_GOVERNANCE_ATTESTATION = "_deepintshield_governance_attestation"
_DISCOVERY_BARRIER = "_deepintshield_discovery_barrier"
# Registry discovery uses a three-second HTTP timeout. Give its detached call a
# small scheduling margin, then perform the same bounded call on the execution
# thread only when the detached hand-off could not complete.
_FIRST_EXECUTION_DISCOVERY_WAIT_SECONDS = 3.25


class _DiscoveryBarrier:
    """Coordinate non-blocking compile discovery with the first execution.

    Compilation only starts the canonical registration attempt. The first
    native execution waits for that bounded attempt so credential lookup cannot
    race ahead and turn a newly captured registration into a generic association
    failure. Later executions see an already-set event and pay no wait.
    """

    def __init__(self) -> None:
        self.completed = threading.Event()
        self.fallback_lock = threading.Lock()
        self.background_started = False
        self.succeeded = False
        self.error: DeepIntShieldError | None = None


def shield_graph(graph: Any, *, engine: Any) -> Any:
    """Instrument a graph and expose configuration failures as built-ins."""
    with public_agentic_error_boundary():
        return _shield_graph(graph, engine=engine)


def _shield_graph(graph: Any, *, engine: Any) -> Any:
    try:
        from langgraph.graph.state import CompiledStateGraph  # noqa: F401
    except Exception:  # pragma: no cover - install-time signal
        raise GovernanceConfigurationError(
            framework="langgraph",
            code="framework_dependency_missing",
        ) from None

    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        raise GovernanceConfigurationError(
            framework="langgraph",
            reason="adapter requires a compiled LangGraph",
        )

    failures: list[str] = []
    for name, node in nodes.items():
        if name in _RESERVED:
            continue
        if not _instrument_node(engine, name, node):
            failures.append(str(name))
    if failures:
        raise GovernanceConfigurationError(
            framework="langgraph",
            reason="could not instrument node(s): " + ", ".join(sorted(failures)),
        )
    return graph


def _instrument_node(engine: Any, name: str, node: Any) -> bool:
    """Wrap the node's underlying callable so every invocation is gated. Returns
    True only when every executable entry point on the selected holder is
    wrapped and bound to the same engine."""
    holders = [getattr(node, "bound", None), getattr(node, "runnable", None), node]
    # Preferred: wrap every RunnableCallable user function (.func / .afunc) on
    # the first holder that exposes one. Stopping after a wrapped .func while an
    # unwrapped .afunc remained made ``ainvoke`` a bypass on dual-mode nodes.
    for holder in holders:
        if holder is None:
            continue
        candidates = [
            (attr, is_async, getattr(holder, attr, None))
            for attr, is_async in (("func", False), ("afunc", True))
            if callable(getattr(holder, attr, None))
        ]
        if not candidates:
            continue
        complete = True
        for attr, is_async, fn in candidates:
            if getattr(fn, "_deepintshield_wrapped", False):
                bound = getattr(fn, "_deepintshield_engine", engine)
                if bound is not engine:
                    raise GovernanceConfigurationError(
                        framework="langgraph",
                        reason=f"node {name!r} is already bound to another SDK client",
                    )
                continue
            # Govern the IMPLEMENTATION, not the node label: the tool identity
            # is the function's own name and we bind to its source fingerprint.
            tool_name = callable_tool_name(fn, name)
            fp = source_fingerprint(fn)
            try:
                setattr(
                    holder,
                    attr,
                    _wrap_fn(
                        engine,
                        tool_name,
                        fn,
                        is_async=is_async,
                        tool_fingerprint=fp,
                    ),
                )
            except Exception:  # pragma: no cover - frozen attr
                complete = False
        return complete

    # Fallback: ToolNode / custom runnable. Wrap every invoke/ainvoke entry
    # point on the first holder that exposes one.
    for holder in holders:
        if holder is None:
            continue
        candidates = [
            (attr, is_async, getattr(holder, attr, None))
            for attr, is_async in (("invoke", False), ("ainvoke", True))
            if callable(getattr(holder, attr, None))
        ]
        if not candidates:
            continue
        complete = True
        for attr, is_async, fn in candidates:
            if getattr(fn, "_deepintshield_wrapped", False):
                bound = getattr(fn, "_deepintshield_engine", engine)
                if bound is not engine:
                    raise GovernanceConfigurationError(
                        framework="langgraph",
                        reason=f"node {name!r} is already bound to another SDK client",
                    )
                continue
            try:
                setattr(holder, attr, _wrap_fn(engine, name, fn, is_async=is_async))
            except Exception:  # pragma: no cover - frozen attr / bound method
                complete = False
        return complete
    return False


def _wrap_fn(engine: Any, tool_name: str, fn: Any, *, is_async: bool, tool_fingerprint: str = ""):
    """Gate one node callable: ask the PDP (raises on a blocking verdict), apply
    any MASK obligations to a dict-shaped state, then run the original. Carries
    the source fingerprint so the decision is bound to the code."""
    if is_async:
        @functools.wraps(fn)
        async def awrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
            decision = resolve(
                engine,
                tool_name,
                (input_value,),
                {},
                tool_fingerprint=tool_fingerprint,
                tool_callable=fn,
            )
            if isinstance(input_value, dict):
                input_value = apply_obligations(input_value, decision.obligations)
            return await fn(input_value, *args, **kwargs)

        awrapper._deepintshield_wrapped = True  # type: ignore[attr-defined]
        awrapper._deepintshield_engine = engine  # type: ignore[attr-defined]
        return awrapper

    @functools.wraps(fn)
    def wrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
        decision = resolve(
            engine,
            tool_name,
            (input_value,),
            {},
            tool_fingerprint=tool_fingerprint,
            tool_callable=fn,
        )
        if isinstance(input_value, dict):
            input_value = apply_obligations(input_value, decision.obligations)
        return fn(input_value, *args, **kwargs)

    wrapper._deepintshield_wrapped = True  # type: ignore[attr-defined]
    wrapper._deepintshield_engine = engine  # type: ignore[attr-defined]
    return wrapper


def _register_and_report(
    engine: Any,
    app: Any,
    barrier: _DiscoveryBarrier,
) -> None:
    """The two bounded gateway calls a compile triggers, run together on one
    background thread. Canonical registry discovery runs first because it is the
    only endpoint allowed to capture a never-before-seen agent registration.
    An absent legacy route remains compatible, but a modern route's coverage or
    scanner error is retained on the barrier and blocks the first execution.
    Neither call blocks the developer's ``compile()``."""
    try:
        # sync=True: we are already off the critical path, so don't spend a
        # second thread on the canonical post.
        barrier.succeeded = report_topology(engine, app, sync=True)
        from ..manifest import describe

        engine.register_blueprint(describe(app))
    except DeepIntShieldError as exc:
        barrier.error = exc
    finally:
        barrier.completed.set()


def _await_initial_discovery(engine: Any, app: Any) -> None:
    """Wait once for compile discovery, with a bounded synchronous fallback."""
    barrier = getattr(app, _DISCOVERY_BARRIER, None)
    if not isinstance(barrier, _DiscoveryBarrier):
        return

    if barrier.completed.is_set():
        if barrier.error is not None:
            raise barrier.error
        if barrier.succeeded:
            ensure_topology_reported(engine, app, required=True)
            return

    if barrier.background_started and barrier.completed.wait(
        _FIRST_EXECUTION_DISCOVERY_WAIT_SECONDS
    ):
        if barrier.error is not None:
            raise barrier.error
        if barrier.succeeded:
            ensure_topology_reported(engine, app, required=True)
            return

    # ``run_detached`` can reject work when its bounded worker allowance is
    # full, or a detached attempt can fail before signalling. Let exactly one
    # first execution perform the canonical post inline; registry HTTP itself is
    # bounded and digest de-duplication prevents duplicate ingest.
    with barrier.fallback_lock:
        if barrier.error is not None:
            raise barrier.error
        if barrier.succeeded:
            return
        try:
            barrier.succeeded = report_topology(
                engine,
                app,
                sync=True,
                required=True,
            )
            from ..manifest import describe

            engine.register_blueprint(describe(app))
        except DeepIntShieldError as exc:
            barrier.error = exc
            raise
        finally:
            barrier.completed.set()


def _ensure_governed(app: Any, *, wait_for_discovery: bool = False) -> Any:
    """Bind + instrument one compiled graph before any execution entry point."""
    engine = resolve_engine(app)
    attestation = getattr(app, _GOVERNANCE_ATTESTATION, None)
    if (
        isinstance(attestation, tuple)
        and len(attestation) == 2
        and attestation[0] is _GOVERNANCE_TOKEN
        and attestation[1] is engine
    ):
        # Compile performed the O(nodes) instrumentation once. Ordinary invoke
        # stays O(1) before the actual PDP call, while a forged public boolean
        # marker alone cannot bless an uninstrumented graph.
        if wait_for_discovery:
            _await_initial_discovery(engine, app)
        return engine
    bind_engine(app, engine, recursive=True)
    # Merely setting the public marker must never turn an uninstrumented graph
    # into a bypass; only our module-private identity token permits the fast path.
    shield_graph(app, engine=engine)
    barrier = _DiscoveryBarrier()
    try:
        setattr(app, "_deepintshield_governed", True)
        setattr(app, _DISCOVERY_BARRIER, barrier)
        # Write the private attestation last: a partially writable framework
        # object must never retain the fast-path token without its discovery
        # barrier.
        setattr(app, _GOVERNANCE_ATTESTATION, (_GOVERNANCE_TOKEN, engine))
    except Exception:
        raise GovernanceConfigurationError(
            framework="langgraph",
            reason="compiled graph cannot retain its governance binding",
        ) from None
    # Registration starts off the compile path. The first execution waits for
    # its bounded result and propagates any authoritative security code.
    barrier.background_started = run_detached(
        _register_and_report,
        engine,
        app,
        barrier,
    )
    if wait_for_discovery:
        _await_initial_discovery(engine, app)
    return engine


def _install_execution_guards(compiled_cls: Any) -> bool:
    """Cover graphs compiled before DeepintShield was initialized."""
    installed = False
    # Runnable exposes several equivalent execution front doors. Guarding only
    # invoke/stream left a graph compiled before the SDK able to enter through
    # batch/transform/event-stream methods before its nodes were instrumented.
    for attr in (
        "invoke",
        "ainvoke",
        "stream",
        "astream",
        "batch",
        "abatch",
        "batch_as_completed",
        "abatch_as_completed",
        "transform",
        "atransform",
        "astream_events",
        "astream_log",
    ):
        orig = getattr(compiled_cls, attr, None)
        if not callable(orig):
            continue
        if getattr(orig, "_deepintshield_execution_guard", False):
            installed = True
            continue

        if inspect.isasyncgenfunction(orig):
            @functools.wraps(orig)
            async def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary():
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        async for item in _orig(self, *args, **kwargs):
                            yield item
        elif inspect.iscoroutinefunction(orig):
            @functools.wraps(orig)
            async def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary():
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        return await _orig(self, *args, **kwargs)
        elif inspect.isgeneratorfunction(orig):
            @functools.wraps(orig)
            def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary():
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        yield from _orig(self, *args, **kwargs)
        else:
            @functools.wraps(orig)
            def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary():
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        return _orig(self, *args, **kwargs)
        guarded._deepintshield_execution_guard = True  # type: ignore[attr-defined]
        try:
            setattr(compiled_cls, attr, guarded)
            installed = True
        except Exception:
            raise GovernanceConfigurationError(
                framework="langgraph",
                reason=f"could not guard CompiledStateGraph.{attr}",
            ) from None
    return installed


@public_agentic_boundary
def enforce() -> bool:
    """Make enforcement non-bypassable: monkey-patch ``StateGraph.compile`` so
    EVERY compiled graph is governed (nodes gated + blueprint registered) before
    it can be invoked. Closes the "forgot to call ``govern()`` / kept a reference
    to the un-governed object / invoked before governing" gap - after ``compile``,
    ``invoke`` always passes through the PDP.

    Cooperative defense-in-depth: a determined process can still un-patch this or
    call a node's function object directly, so the gateway (MCP/LLM in the call
    path) remains the authoritative boundary. But for ordinary application code
    this guarantees no compiled graph runs ungoverned.

    Idempotent and fail-closed: compile returns a governed graph or raises.
    Returns True if installed.
    """
    try:
        from langgraph.graph.state import CompiledStateGraph, StateGraph
    except Exception:  # langgraph not installed - nothing to guard
        return False

    existing = getattr(StateGraph, "compile", None)
    if getattr(existing, "_deepintshield_guarded", False):
        _install_execution_guards(CompiledStateGraph)
        return True

    orig_compile = StateGraph.compile

    @functools.wraps(orig_compile)
    def guarded_compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        with public_agentic_error_boundary():
            app = orig_compile(self, *args, **kwargs)
            _ensure_governed(app)
            return app

    guarded_compile._deepintshield_guarded = True  # type: ignore[attr-defined]
    StateGraph.compile = guarded_compile  # type: ignore[assignment]
    return _install_execution_guards(CompiledStateGraph)


# Back-compat alias.
enforce_compile = enforce

__all__ = ["shield_graph", "enforce", "enforce_compile"]
