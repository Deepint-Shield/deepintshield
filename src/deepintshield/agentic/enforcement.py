"""Non-bypassable enforcement installer for every supported framework.

The per-framework adapters gate tools *when the developer calls* ``govern()`` /
``guard()``. That still leaves a gap: nothing stops code from skipping that call
and running the agent directly. ``install_all`` closes it by monkey-patching each
framework's build/execute boundary the moment a DeepintShield client is created,
so a tool/graph can't run ungoverned - the developer no longer has to remember.

Design rules every framework installer follows:
  * **Lazy, client-scoped engine** - framework objects are bound to the selected
    client/engine at build or first-run time. A single live client needs no user
    code; multiple clients require an explicit request-local scope and otherwise
    fail closed. There is no process-global "latest client wins" pointer.
  * **Only if imported** - we patch a framework only when it's already in
    ``sys.modules`` (never force-import a dep the user isn't using). A framework
    imported *after* the client is built is picked up by the import watch below.
  * **Idempotent** - methods are patched once; ownership is resolved per object
    or request context, never rebound globally.
  * **Fail-closed execution** - PDP outages, ambiguous clients, failed
    instrumentation, and an unverified code-blueprint report all block
    execution. Build-time discovery may be detached, but a framework execution
    boundary requires its bounded report to finish successfully.

Two things live here besides the installer:

  * :func:`report_topology` - the single GAF-registry discovery call. Every
    framework's ``enforce`` routes through it at its natural build/first-run
    boundary (compile / kickoff / run), so the Registry populates itself for all
    of them instead of only for LangGraph.
  * :func:`report_topology_on` - patches a class method into such a boundary.

Cooperative defense-in-depth: a determined process can un-patch these or call a
tool's raw function object, so the gateway (MCP/LLM in the call path) stays the
authoritative boundary. For ordinary application code this makes enforcement the
default rather than something to remember.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import sys
import threading
import weakref
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Callable, Iterator

from .errors import GovernanceConfigurationError, public_agentic_boundary

# top-level import name → enforcement integration module (relative to this package)
_FRAMEWORKS: list[tuple[str, str]] = [
    ("langgraph", ".integrations.langgraph"),
    ("langchain_core", ".integrations.langchain"),
    ("crewai", ".integrations.crewai"),
    ("llama_index", ".integrations.llamaindex"),
    ("autogen", ".integrations.autogen"),
    ("autogen_core", ".integrations.autogen"),
    ("autogen_agentchat", ".integrations.autogen"),
    ("pydantic_ai", ".integrations.pydanticai"),
    ("agents", ".integrations.openai_agents"),  # `openai-agents` imports as `agents`
    ("litellm", ".integrations.litellm"),
    ("strands", ".integrations.strands"),
    ("google.adk", ".integrations.google_adk"),
    ("temporalio", ".integrations.temporal"),
    # Hermes Agent's executable Python boundary is this top-level module. The
    # `hermes_cli` package alone contains configuration/UI helpers and is not a
    # proof that the tool dispatcher has been loaded yet.
    ("model_tools", ".integrations.hermes"),
]

_WATCHED = frozenset(name for name, _ in _FRAMEWORKS)

# Frameworks whose guards are installed, so the import watch doesn't re-run
# install_all on every submodule import of an already-guarded framework.
_installed: set[str] = set()
_install_lock = threading.RLock()

# Import-watch state.
_watch_state = threading.local()
_watching = False

# Live clients are weakly held: constructing a second client never silently
# rebinds global wrappers, and a discarded client does not poison later
# single-client resolution. Framework instances are strongly bound to their
# selected engine when first compiled/run.
_clients: "weakref.WeakSet[Any]" = weakref.WeakSet()
_clients_lock = threading.RLock()
_active_engine: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "deepintshield_active_agentic_engine", default=None
)
_BOUND_ENGINE = "_deepintshield_engine"


def register_client(client: Any) -> None:
    """Make one SDK client eligible for zero-code framework governance."""
    if client is None:
        return
    try:
        with _clients_lock:
            _clients.add(client)
    except TypeError:
        raise GovernanceConfigurationError(
            reason="DeepintShield client must support weak references"
        ) from None


def unregister_client(client: Any) -> None:
    with _clients_lock:
        _clients.discard(client)


@contextmanager
def engine_scope(engine: Any) -> Iterator[Any]:
    """Select an engine for this request/task when several clients coexist."""
    token = _active_engine.set(engine)
    try:
        yield engine
    finally:
        _active_engine.reset(token)


def bind_engine(obj: Any, engine: Any, *, recursive: bool = False) -> bool:
    """Attach an engine to a framework object without a global provider.

    ``recursive=True`` also binds common agent/tool containers so a Crew/Agent
    selected at its run boundary passes the same tenant/VK to its tool calls.
    """
    if obj is None or engine is None:
        return False
    bound = _set_binding(obj, engine)
    if not recursive:
        return bound
    seen: set[int] = set()

    def walk(value: Any, depth: int) -> None:
        if value is None or depth > 4 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                walk(item, depth + 1)
            return
        _set_binding(value, engine)
        for attr in (
            "tools",
            "_tools",
            "tools_by_name",
            "agents",
            "_agents",
            "participants",
            "_participants",
            "agent",
            "agent_worker",
            "_agent_worker",
            "_function_tools",
            "_function_toolset",
        ):
            try:
                child = getattr(value, attr, None)
            except Exception:
                continue
            if child is not None and child is not value:
                walk(child, depth + 1)

    walk(obj, 0)
    return bound


def _set_binding(obj: Any, engine: Any) -> bool:
    try:
        setattr(obj, _BOUND_ENGINE, engine)
        return True
    except Exception:
        try:
            object.__setattr__(obj, _BOUND_ENGINE, engine)
            return True
        except Exception:
            return False


def resolve_engine(owner: Any = None) -> Any:
    """Resolve one engine without a last-client-wins fallback.

    Priority is an engine already bound to ``owner``, then the request-local
    :func:`engine_scope`, then the sole live SDK client. Zero or multiple
    candidates are configuration failures and block execution.
    """
    if owner is not None:
        try:
            bound = getattr(owner, _BOUND_ENGINE, None)
        except Exception:
            bound = None
        if bound is not None:
            # Framework objects often outlive a short-lived SDK client (test
            # fixtures, hot reload, credential rotation). A stale binding must
            # not keep a closed/released client's transport alive forever; it
            # is safe to re-select only after the former owner is demonstrably
            # gone or closed.
            parent_ref = getattr(bound, "_parent_ref", None)
            if callable(parent_ref):
                try:
                    parent = parent_ref()
                except Exception:
                    parent = None
                if parent is not None and not bool(
                    getattr(parent, "_deepintshield_closed", False)
                ):
                    return bound
                _set_binding(owner, None)
            else:
                # Third-party/custom engines have no SDK lifecycle marker.
                return bound
    scoped = _active_engine.get()
    if scoped is not None:
        if owner is not None:
            bind_engine(owner, scoped, recursive=True)
        return scoped
    with _clients_lock:
        candidates = [
            client
            for client in list(_clients)
            if not bool(getattr(client, "_deepintshield_closed", False))
        ]
    if len(candidates) != 1:
        if not candidates:
            reason = (
                "no live DeepintShield client; create one before executing "
                "an agent framework"
            )
        else:
            reason = (
                "multiple DeepintShield clients are live; select one with "
                "`with shield.agentic.run(...):`"
            )
        raise GovernanceConfigurationError(reason=reason)
    engine = candidates[0].agentic.engine
    if owner is not None:
        bind_engine(owner, engine, recursive=True)
    return engine


@public_agentic_boundary
def install_all(*, client: Any = None) -> list[str]:
    """Install enforcement guards for every supported framework currently
    imported. Returns the list of frameworks newly guarded.

    If an imported framework version cannot be instrumented, the public setup
    boundary raises a marked ``RuntimeError``; allowing client construction to
    continue would falsely imply that ordinary execution is protected.
    """
    if client is not None:
        register_client(client)
    installed: list[str] = []
    seen: set[str] = set()
    previous_busy = getattr(_watch_state, "busy", False)
    _watch_state.busy = True
    try:
        for mod_name, integ in _FRAMEWORKS:
            if mod_name not in sys.modules or integ in seen:
                continue
            with _install_lock:
                if mod_name in _installed:
                    continue
                # Nested imports return before the enclosing framework has
                # defined its execution classes. Inspecting it at that point
                # causes circular imports and falsely reports an unsupported
                # version. The enclosing import's post-hook retries once all
                # framework modules have finished initializing.
                if any(
                    (name == mod_name or name.startswith(mod_name + "."))
                    and getattr(getattr(module, "__spec__", None), "_initializing", False)
                    for name, module in tuple(sys.modules.items())
                ):
                    continue
                # Keep check → patch → mark atomic. Two clients initialized in
                # parallel must not wrap the same framework method twice and
                # therefore run two PDP decisions for one tool invocation.
                seen.add(integ)
                try:
                    module = import_module(integ, package=__package__)
                    installer = getattr(module, "enforce", None)
                    if not callable(installer) or not installer():
                        raise GovernanceConfigurationError(
                            framework=mod_name,
                            reason="installed framework version exposes no enforceable execution boundary",
                        )
                except GovernanceConfigurationError:
                    raise
                except Exception as exc:
                    raise GovernanceConfigurationError(
                        framework=mod_name,
                        reason=f"failed to install execution guards: {exc}",
                    ) from None
                _installed.add(mod_name)
                installed.append(mod_name)
    finally:
        _watch_state.busy = previous_busy
    # A framework imported later than the client would otherwise stay ungoverned
    # until someone remembered ``shield.agentic.enforce()``.
    _watch_imports()
    return installed


# ──────────────────────────────────────────────────────────────────────────
# Late-import watch
# ──────────────────────────────────────────────────────────────────────────


def _watch_imports() -> bool:
    """Re-run :func:`install_all` when a supported framework is imported *after*
    the DeepintShield client was built.

    ``install_all`` can only patch what is already in ``sys.modules``, so::

        shield = DeepintShield(virtual_key=…)   # crewai not imported yet
        import crewai                            # → previously ungoverned

    left the framework unguarded and its topology unreported. We wrap
    ``builtins.__import__`` (the standard post-import-hook technique) so the
    guards arm themselves the moment the module finishes loading.

    Installed once per process and re-entrancy-guarded per thread. An imported
    supported framework that cannot be guarded makes that import fail closed."""
    global _watching
    if _watching:
        return True
    try:
        import builtins

        original = builtins.__import__
        if getattr(original, "_deepintshield_watch", False):
            _watching = True
            return True

        @functools.wraps(original)
        def watched(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            module = original(name, globals, locals, fromlist, level)
            _rearm(name, level)
            return module

        watched._deepintshield_watch = True  # type: ignore[attr-defined]
        builtins.__import__ = watched
        _watching = True
        return True
    except Exception as exc:  # pragma: no cover - exotic runtime
        raise GovernanceConfigurationError(
            reason=f"could not install supported-framework import guard: {exc}"
        ) from None


def _rearm(name: Any, level: int) -> None:
    """Post-import hook body: arm the guards for a newly-imported framework."""
    if level or getattr(_watch_state, "busy", False):
        return
    # The import hook is process-global but SDK clients are not. Once every
    # client has been closed/released, framework imports are ordinary imports
    # again; attempting to instrument a partially loaded module at that point
    # serves no enforcement purpose and can break explicit adapter construction.
    with _clients_lock:
        if not any(
            not bool(getattr(client, "_deepintshield_closed", False))
            for client in list(_clients)
        ):
            return
    if not isinstance(name, str) or not name:
        return
    matches = [
        root
        for root in _WATCHED
        if name == root or name.startswith(root + ".") or root.startswith(name + ".")
    ]
    if not matches:
        return
    with _install_lock:
        if all(root in _installed for root in matches):
            return
    _watch_state.busy = True  # install_all imports our own integration modules
    try:
        install_all()
    finally:
        _watch_state.busy = False


# ──────────────────────────────────────────────────────────────────────────
# GAF registry discovery
# ──────────────────────────────────────────────────────────────────────────


_BACKGROUND_MAX = 4
_bg_lock = threading.Lock()
_bg_inflight = 0


def run_detached(fn: Callable[..., Any], *args: Any, name: str = "deepintshield-governance") -> bool:
    """Run best-effort governance work off the caller's thread.

    The build boundary we hook (``compile()`` / ``kickoff()``) belongs to the
    developer, so *nothing* optional may be paid for on it. Blueprint
    registration in particular used to run inline on the shared httpx client and
    inherit its 30s request timeout - a blackhole gateway froze ``compile()`` for
    the full 30 seconds. Here it is a daemon thread (never holds up interpreter
    exit), capped at :data:`_BACKGROUND_MAX` concurrent jobs (a loop that
    compiles per request cannot spawn unbounded threads) and never raises.

    Returns True when the job was dispatched."""
    global _bg_inflight
    try:
        if sys.is_finalizing():
            return False
    except Exception:  # pragma: no cover - exotic runtime
        pass
    with _bg_lock:
        if _bg_inflight >= _BACKGROUND_MAX:
            return False
        _bg_inflight += 1
    caller_context = contextvars.copy_context()

    def _body() -> None:
        global _bg_inflight
        try:
            caller_context.run(fn, *args)
        except BaseException:  # telemetry thread: swallow everything, incl. shutdown
            pass
        finally:
            with _bg_lock:
                _bg_inflight -= 1

    try:
        threading.Thread(target=_body, name=name, daemon=True).start()
        return True
    except BaseException:
        with _bg_lock:
            _bg_inflight -= 1
        return False


def report_topology(
    engine: Any,
    obj: Any,
    *,
    name: str = "",
    sync: bool = False,
    required: bool = False,
) -> bool:
    """Report ``obj``'s topology to the GAF registry (agentic-new).

    The single discovery entry point shared by every framework's ``enforce``.
    It used to live inline in the LangGraph compile guard, which is why only
    LangGraph ever populated Registry → Agents | Tools | Networks; hoisting it
    here lets CrewAI / AutoGen / LlamaIndex / PydanticAI / OpenAI-Agents report
    at their own build boundary with three lines each.

    Build-time discovery stays off the invocation path. Execution-time
    attestation is synchronous only for a new digest and is de-duplicated for
    unchanged code; a required report raises a stable code on failure.

    ``sync=True`` posts inline. Framework run boundaries use that mode once so
    a first-seen agent is present in the dashboard before authorization starts;
    compile/build boundaries can still dispatch it from :func:`run_detached`.
    ``required=True`` converts any failed synchronous report into a code-only
    configuration error so changed or partially captured code cannot run on an
    older approval.
    """
    if obj is None or engine is None:
        if required:
            raise GovernanceConfigurationError(code="blueprint_scan_unavailable") from None
        return False
    try:
        from .execution import cache_topology
        from .registry import describe_network, discover

        # Description is already required by discovery. Cache that exact
        # normalized manifest on the framework object so every invocation can
        # snapshot it into the execution ledger without reflecting over the
        # graph again. Frozen models simply skip the cache.
        manifest = describe_network(obj, name=name)
        cache_topology(obj, manifest)
        result = discover(engine, manifest=manifest, name=name, sync=sync)
        if result.get("error") or result.get("throttled") or result.get("shutdown"):
            if required:
                raw_code = str(result.get("error") or "")
                code = (
                    raw_code
                    if raw_code and not raw_code.startswith("registry_discovery_")
                    else "blueprint_scan_unavailable"
                )
                raise GovernanceConfigurationError(code=code) from None
            return False
        return bool(result.get("dispatched") or result.get("deduped") or sync)
    except GovernanceConfigurationError:
        raise
    except Exception:  # optional build-time telemetry must never break compile
        if required:
            raise GovernanceConfigurationError(code="blueprint_scan_unavailable") from None
        return False


def ensure_topology_reported(
    engine: Any,
    obj: Any,
    *,
    name: str = "",
    required: bool = True,
) -> bool:
    """Synchronously attest a changed blueprint and locally skip an unchanged one.

    Source expansion is cached against a bounded bytecode/dependency token by
    :mod:`registry`, so this check performs no HTTP request—and normally no file
    read—when the live callables are unchanged. The marker is written only after
    a durable discovery acknowledgement.
    """
    if obj is None or engine is None:
        if required:
            raise GovernanceConfigurationError(code="blueprint_scan_unavailable") from None
        return False
    try:
        from .registry import describe_network

        manifest = describe_network(obj, name=name)
        fingerprint = str(manifest.get("blueprint_digest") or "").strip()
        if not fingerprint:
            if required:
                raise GovernanceConfigurationError(code="blueprint_scan_unavailable") from None
            return False
        scope = str(getattr(engine, "registry_scope_id", "") or id(engine))
        marker = (scope, fingerprint)
        try:
            if getattr(obj, "_deepintshield_reported_blueprint", None) == marker:
                return True
        except Exception:
            pass
        report_kwargs: dict[str, Any] = {"sync": True, "required": required}
        if name:
            report_kwargs["name"] = name
        if not report_topology(engine, obj, **report_kwargs):
            return False
        try:
            setattr(obj, "_deepintshield_reported_blueprint", marker)
            # Compatibility/diagnostic marker only. It is deliberately never
            # trusted as the fast-path identity.
            setattr(obj, "_deepintshield_reported", True)
        except Exception:
            pass
        return True
    except GovernanceConfigurationError:
        raise
    except Exception:
        if required:
            raise GovernanceConfigurationError(code="blueprint_scan_unavailable") from None
        return False


def report_topology_on(
    cls: Any,
    attrs: tuple[str, ...],
) -> bool:
    """Patch ``cls.attr`` (a run/kickoff-style method) so the receiver's topology
    is reported to the registry the first time it executes.

    This is the discovery twin of ``_common.install_method_guard``: it never
    gates and never changes the return value, it only fires
    :func:`report_topology` for ``self`` before delegating. Reporting is
    once per unchanged local blueprint (the registry's digest dedupe then
    collapses repeats across instances), idempotent across re-installs, and
    fail-closed at execution boundaries."""
    installed = False
    for attr in attrs:
        orig = getattr(cls, attr, None)
        if not callable(orig):
            continue
        if getattr(orig, "_deepintshield_topology", False):
            installed = True
            continue
        reporter = _make_topology_reporter(orig)
        try:
            setattr(cls, attr, reporter)
            installed = True
        except Exception:  # frozen / slotted class
            continue
    return installed


def _make_topology_reporter(orig: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap one bound method so it reports ``self``'s topology before delegating.

    The engine is resolved from the receiver/request context at call time, so
    different instances can safely belong to different clients."""

    def fire(self: Any) -> Any:
        # Engine selection is part of enforcement, not telemetry. Do not swallow
        # zero/ambiguous-client failures: the run must never start ungoverned.
        engine = resolve_engine(self)
        bind_engine(self, engine, recursive=True)
        # First-run discovery is synchronous and bounded. This ordering is what
        # lets an unknown native framework agent become a pending dashboard
        # registration before its first authorization check returns the public
        # lifecycle code. Registry digest dedupe makes later runs free.
        ensure_topology_reported(engine, self, required=True)
        return engine

    def scope(engine: Any, receiver: Any):
        from .execution import execution_scope

        return execution_scope(engine, receiver)

    if inspect.isasyncgenfunction(orig):
        @functools.wraps(orig)
        @public_agentic_boundary
        async def reporter(self: Any, *args: Any, **kwargs: Any):
            engine = fire(self)
            with scope(engine, self):
                async for item in orig(self, *args, **kwargs):
                    yield item
    elif inspect.iscoroutinefunction(orig):
        @functools.wraps(orig)
        @public_agentic_boundary
        async def reporter(self: Any, *args: Any, **kwargs: Any) -> Any:
            engine = fire(self)
            with scope(engine, self):
                return await orig(self, *args, **kwargs)
    elif inspect.isgeneratorfunction(orig):
        @functools.wraps(orig)
        @public_agentic_boundary
        def reporter(self: Any, *args: Any, **kwargs: Any):
            engine = fire(self)
            with scope(engine, self):
                yield from orig(self, *args, **kwargs)
    else:
        @functools.wraps(orig)
        @public_agentic_boundary
        def reporter(self: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            engine = fire(self)
            with scope(engine, self):
                return orig(self, *args, **kwargs)

    reporter._deepintshield_topology = True  # type: ignore[attr-defined]
    return reporter


def report_topology_for(
    candidates: tuple[tuple[str, str, tuple[str, ...]], ...],
) -> bool:
    """Resolve ``(module, class, methods)`` candidates and install the topology
    report on the first ones that exist.

    Frameworks move their orchestration class between releases (CrewAI's
    ``Crew``, AutoGen's team base, LlamaIndex's agent runner), so each adapter
    passes every shape it knows about and this walks them. A candidate whose
    module/class is missing is skipped silently - importing here never forces a
    dependency, because the framework is already loaded by the time
    ``enforce`` runs."""
    installed = False
    for module_name, class_name, attrs in candidates:
        try:
            cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        except Exception:
            continue
        try:
            if report_topology_on(cls, attrs):
                installed = True
        except Exception:
            continue
    return installed


__all__ = [
    "install_all",
    "register_client",
    "unregister_client",
    "resolve_engine",
    "bind_engine",
    "engine_scope",
    "run_detached",
    "report_topology",
    "ensure_topology_reported",
    "report_topology_on",
    "report_topology_for",
]
