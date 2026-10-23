"""LangGraph enforcement - gate nodes when workspace Agentic Security is on.

Drop-in: no per-tool decorator, no graph-shape change. Mutates the graph in
place and returns it so existing ``invoke()`` code is unchanged. The node name
is the governed tool name the PDP checks. Each top-level run first reads the
authenticated workspace runtime setting. An explicit disabled setting runs the
native graph without discovery, approval checks, or security telemetry.

Interception is done at the node's underlying callable. In current LangGraph a
compiled node is a ``PregelNode`` whose ``.bound`` is a ``RunnableCallable``
holding the user function in ``.func`` / ``.afunc`` - wrapping those intercepts
every invocation regardless of how the Pregel loop dispatches it (the older
approach of wrapping ``runnable.invoke`` silently missed plain function nodes,
because ``PregelNode.runnable`` is ``None``). We wrap ``func``/``afunc`` first
and fall back to ``invoke``/``ainvoke`` (plus an overridden ``stream``/``astream``)
on the executed ``PregelNode.node`` for composite runnables such as
``RunnableParallel``/``RunnableSequence``, subgraphs and custom runnables.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from contextvars import ContextVar
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
from ..runtime import security_enabled, security_scope
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
# Ids of fallback-gated runnables already admitted in the current call chain. A
# compiled subgraph's invoke() drives its own stream(), so without this marker a
# single node entry would ask the PDP twice.
_ADMITTED_RUNNABLES: ContextVar[frozenset[int]] = ContextVar(
    "deepintshield_langgraph_admitted_runnables", default=frozenset()
)


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

    # Fallback: composite runnable / subgraph / custom Runnable. Gate the object
    # the Pregel loop actually dispatches (``PregelNode.node``: bound fused with
    # its channel writers) ahead of the holders themselves: LangGraph flattens a
    # ``RunnableSequence`` bound into it, so a gate on ``bound`` alone would
    # never run. Wrap every invoke/ainvoke entry point on the first holder that
    # exposes one. The loop dispatches through astream() under
    # astream_events()/astream_log(); a composite such as RunnableParallel
    # overrides it without calling ainvoke(), so an overridden stream/astream is
    # gated too. The base Runnable versions already route through the gated
    # invoke/ainvoke.
    # The node's governed code is the composition it dispatches, which is what
    # discovery reported as its artifact. The gate attests that same text (as
    # the decision fingerprint and for the first-run capture) rather than the
    # framework method it wraps: attesting ``invoke`` of the bound runnable
    # never matched the acknowledged artifact, so the first execution
    # re-registered the node under langgraph's own source, an incomplete
    # artifact the gateway refuses as blueprint_coverage_incomplete and
    # quarantines the agent for. An unreadable composition stays without a
    # fingerprint and keeps failing closed exactly as before.
    from ..manifest import _composite_node_source, _text_fingerprint

    composition = _composite_node_source(node)
    composite_fingerprint = _text_fingerprint(composition) if composition else ""
    for holder in [_executed_runnable(node, holders[0]), *holders]:
        if holder is None:
            continue
        candidates = [
            (attr, is_async, False, getattr(holder, attr, None))
            for attr, is_async in (("invoke", False), ("ainvoke", True))
            if callable(getattr(holder, attr, None))
        ]
        if not candidates:
            continue
        candidates += [
            (attr, is_async, True, getattr(holder, attr))
            for attr, is_async in (("stream", False), ("astream", True))
            if _overrides_runnable_method(holder, attr)
        ]
        complete = True
        for attr, is_async, is_stream, fn in candidates:
            if getattr(fn, "_deepintshield_wrapped", False):
                bound = getattr(fn, "_deepintshield_engine", engine)
                if bound is not engine:
                    raise GovernanceConfigurationError(
                        framework="langgraph",
                        reason=f"node {name!r} is already bound to another SDK client",
                    )
                continue
            wrapped = _wrap_fn(
                engine,
                name,
                fn,
                is_async=is_async,
                tool_fingerprint=composite_fingerprint,
                is_stream=is_stream,
                reentry_key=id(holder),
                tool_callable=node,
            )
            try:
                _set_entry_point(holder, attr, wrapped)
            except Exception:  # pragma: no cover - frozen attr / slotted class
                complete = False
                continue
            if getattr(holder, attr, None) is not wrapped:
                # A data descriptor swallowed the assignment.
                complete = False
        return complete
    return False


def _executed_runnable(node: Any, bound: Any) -> Any:
    """What the Pregel loop dispatches for a node: ``PregelNode.node``, the bound
    runnable fused with its channel writers. It is one object per compiled node
    and survives ``with_config`` copies, which share the node table."""
    try:
        executed = getattr(node, "node", None)
    except Exception:  # pragma: no cover - defensive against custom node types
        return None
    if executed is None or executed is node or executed is bound:
        return None
    return executed


def _overrides_runnable_method(holder: Any, attr: str) -> bool:
    """True when ``holder``'s class replaces the base ``Runnable`` method."""
    try:
        from langchain_core.runnables import Runnable
    except Exception:  # pragma: no cover - langgraph always ships langchain-core
        return False
    if not isinstance(holder, Runnable):
        return False
    return getattr(type(holder), attr, None) is not getattr(Runnable, attr, None)


def _set_entry_point(holder: Any, attr: str, value: Any) -> None:
    """Install ``value`` as an instance attribute that shadows the class method.

    Pydantic-backed runnables (``RunnableParallel``, ``RunnableSequence``) reject
    non-field names in their own ``__setattr__``. Their instance ``__dict__``
    still takes precedence over the class method, so write it there directly.
    """
    try:
        setattr(holder, attr, value)
    except (AttributeError, TypeError, ValueError):
        object.__setattr__(holder, attr, value)


def _wrap_fn(
    engine: Any,
    tool_name: str,
    fn: Any,
    *,
    is_async: bool,
    tool_fingerprint: str = "",
    is_stream: bool = False,
    reentry_key: int | None = None,
    tool_callable: Any = None,
):
    """Gate one node callable: ask the PDP (raises on a blocking verdict), apply
    any MASK obligations to a dict-shaped state, then run the original. Carries
    the source fingerprint so the decision is bound to the code.

    ``tool_callable`` is what the gate attests as this node's code when it is
    not ``fn`` itself: a composite node passes its holder, whose source is the
    composition discovery reported, so the first-run capture recognises the
    acknowledged artifact instead of re-registering framework code.

    ``is_stream`` gates a generator entry point (``stream``/``astream``) before
    its first chunk. ``reentry_key`` identifies a fallback holder: its gated
    ``invoke``/``ainvoke`` admit the holder for the duration of the call, so a
    runnable that drives its own ``stream``/``astream`` (a compiled subgraph) or
    whose default ``ainvoke`` runs ``invoke`` on an executor asks the PDP once
    per node entry. Generator wrappers only consult the marker and never set
    it, so an abandoned iterator cannot leave a stale admission behind."""

    def admit(input_value: Any, *, hold: bool) -> tuple[Any, Any]:
        admitted = _ADMITTED_RUNNABLES.get()
        if reentry_key is not None and reentry_key in admitted:
            return input_value, None
        decision = resolve(
            engine,
            tool_name,
            (input_value,),
            {},
            tool_fingerprint=tool_fingerprint,
            tool_callable=fn if tool_callable is None else tool_callable,
        )
        if isinstance(input_value, dict):
            input_value = apply_obligations(input_value, decision.obligations)
        if not hold or reentry_key is None:
            return input_value, None
        return input_value, _ADMITTED_RUNNABLES.set(admitted | {reentry_key})

    def mark(guarded: Any) -> Any:
        guarded._deepintshield_wrapped = True  # type: ignore[attr-defined]
        guarded._deepintshield_engine = engine  # type: ignore[attr-defined]
        return guarded

    if is_async and is_stream:
        @functools.wraps(fn)
        async def astream_wrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
            input_value, _ = admit(input_value, hold=False)
            async for chunk in fn(input_value, *args, **kwargs):
                yield chunk

        return mark(astream_wrapper)

    if is_async:
        @functools.wraps(fn)
        async def awrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
            input_value, token = admit(input_value, hold=True)
            try:
                return await fn(input_value, *args, **kwargs)
            finally:
                _release(token)

        return mark(awrapper)

    if is_stream:
        @functools.wraps(fn)
        def stream_wrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
            input_value, _ = admit(input_value, hold=False)
            yield from fn(input_value, *args, **kwargs)

        return mark(stream_wrapper)

    @functools.wraps(fn)
    def wrapper(input_value: Any = None, *args: Any, **kwargs: Any) -> Any:
        input_value, token = admit(input_value, hold=True)
        try:
            return fn(input_value, *args, **kwargs)
        finally:
            _release(token)

    return mark(wrapper)


def _release(token: Any) -> None:
    """Drop an admission marker; tolerate a reset from a foreign context."""
    if token is None:
        return
    try:
        _ADMITTED_RUNNABLES.reset(token)
    except (RuntimeError, ValueError):  # pragma: no cover - foreign context
        pass


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
    runtime_ready = False
    try:
        with security_scope(engine):
            runtime_ready = True
            from ..managed_execution import capabilities

            if security_enabled(engine) and capabilities(engine):
                # Managed jobs capture registration with their submission. Do
                # not enroll the submitting machine as the remote worker.
                return
            # sync=True: we are already off the critical path, so don't spend a
            # second thread on the canonical post.
            barrier.succeeded = report_topology(engine, app, sync=True)
            from ..manifest import describe

            engine.register_blueprint(describe(app))
    except DeepIntShieldError as exc:
        # Retry failed optional compile-time config probes at execution. Only
        # errors reached after negotiation can be durable blueprint failures.
        if runtime_ready:
            barrier.error = exc
    finally:
        barrier.completed.set()


def _await_initial_discovery(engine: Any, app: Any) -> None:
    """Wait once for compile discovery, with a bounded synchronous fallback."""
    if not security_enabled(engine):
        return
    from ..managed_execution import handoff_if_enabled

    handoff_if_enabled(engine, app)
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
                with public_agentic_error_boundary(), security_scope(resolve_engine(self)):
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        async for item in _orig(self, *args, **kwargs):
                            yield item
        elif inspect.iscoroutinefunction(orig):
            @functools.wraps(orig)
            async def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary(), security_scope(resolve_engine(self)):
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        return await _orig(self, *args, **kwargs)
        elif inspect.isgeneratorfunction(orig):
            @functools.wraps(orig)
            def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary(), security_scope(resolve_engine(self)):
                    engine = _ensure_governed(self, wait_for_discovery=True)
                    with execution_scope(engine, self, framework="langgraph"):
                        yield from _orig(self, *args, **kwargs)
        else:
            @functools.wraps(orig)
            def guarded(self: Any, *args: Any, _orig=orig, **kwargs: Any):
                with public_agentic_error_boundary(), security_scope(resolve_engine(self)):
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
    """Install compile and execution guards for workspace-driven enforcement.

    Every graph retains node guards so enabling security takes effect on its
    next invocation. Runs with Agentic Security enabled require registration
    and PDP authorization. Explicitly disabled workspaces execute native nodes
    without Agentic Security calls.

    Cooperative defense-in-depth: a determined process can still un-patch this or
    call a node's function object directly, so the gateway (MCP/LLM in the call
    path) remains the authoritative boundary.

    Idempotent and fail-closed when security is enabled or cannot be determined.
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
