"""Shared helpers for the per-framework enforcement adapters.

Every adapter ultimately needs to do the same thing: given a tool's name and
its underlying callable, return a callable that runs ``gate.enforce`` first.
``wrap_callable`` handles both sync and async tools and is idempotent.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..errors import GovernanceConfigurationError, public_agentic_error_boundary
from ..execution import execution_scope
from ..gate import enforce, preflight

_FLAG = "_deepintshield_wrapped"
_CALLBACK_INVENTORY_MAX = 64


@dataclass(frozen=True)
class ExecutableCallback:
    """One application callback explicitly exposed by a framework object.

    ``slot`` is structural metadata (for example ``agent.step_callback``), not
    runtime data.  The registry uses it only to build a deterministic source
    bundle for registration-time blueprint scanning.
    """

    slot: str
    callback: Any


@dataclass(frozen=True)
class CallbackInventory:
    callbacks: tuple[ExecutableCallback, ...] = ()
    incomplete: bool = False


def explicit_callback_inventory(
    owner: Any,
    fields: Sequence[str],
    *,
    framework_modules: Sequence[str],
    prefix: str = "",
) -> CallbackInventory:
    """Inventory only callbacks stored in explicit, adapter-owned fields.

    This deliberately does not walk an object's attributes, MRO, framework
    dispatcher tables, or module globals.  Framework-owned callbacks are
    ignored by module prefix; application callbacks registered in a supported
    public field are returned.  A bounded malformed/raising collection is
    marked incomplete so the blueprint report fails closed rather than silently
    scanning a subset.
    """
    found: list[ExecutableCallback] = []
    incomplete = False
    seen_containers: set[int] = set()
    seen_callbacks: set[tuple[str, tuple[int, int]]] = set()

    def append(slot: str, value: Any, depth: int = 0) -> None:
        nonlocal incomplete
        if value is None:
            return
        if len(found) >= _CALLBACK_INVENTORY_MAX:
            incomplete = True
            return
        if isinstance(value, dict):
            identity = id(value)
            if identity in seen_containers:
                return
            seen_containers.add(identity)
            if depth >= 3:
                if value:
                    incomplete = True
                return
            items = sorted(value.items(), key=lambda item: _structural_order_key(item[0]))
            if len(items) > _CALLBACK_INVENTORY_MAX - len(found):
                incomplete = True
            for key, nested in items[: _CALLBACK_INVENTORY_MAX - len(found)]:
                append(f"{slot}.{_safe_slot_token(key)}", nested, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            identity = id(value)
            if identity in seen_containers:
                return
            seen_containers.add(identity)
            if depth >= 3:
                if value:
                    incomplete = True
                return
            values = list(value)
            if isinstance(value, (set, frozenset)):
                values.sort(key=_structural_order_key)
            if len(values) > _CALLBACK_INVENTORY_MAX - len(found):
                incomplete = True
            for index, nested in enumerate(
                values[: _CALLBACK_INVENTORY_MAX - len(found)]
            ):
                append(f"{slot}[{index}]", nested, depth + 1)
            return
        if not callable(value):
            # Declarative values sometimes share a container with callbacks.
            # They are not executable and must not make an ordinary agent fail.
            return
        if _framework_owned_callable(value, framework_modules):
            return
        identity = (slot, _callable_identity(value))
        if identity in seen_callbacks:
            return
        seen_callbacks.add(identity)
        found.append(ExecutableCallback(slot=slot, callback=value))

    for field in fields:
        try:
            value = getattr(owner, field)
        except AttributeError:
            continue
        except Exception:
            incomplete = True
            continue
        append(f"{prefix}.{field}" if prefix else field, value)
    return CallbackInventory(tuple(found), incomplete)


def merge_callback_inventories(*inventories: CallbackInventory) -> CallbackInventory:
    callbacks: list[ExecutableCallback] = []
    incomplete = False
    for inventory in inventories:
        callbacks.extend(inventory.callbacks)
        incomplete = incomplete or inventory.incomplete
        if len(callbacks) > _CALLBACK_INVENTORY_MAX:
            callbacks = callbacks[:_CALLBACK_INVENTORY_MAX]
            incomplete = True
    return CallbackInventory(tuple(callbacks), incomplete)


def callback_method_inventory(
    owners: Any,
    methods: Sequence[str],
    *,
    framework_modules: Sequence[str],
    prefix: str,
) -> CallbackInventory:
    """Inventory named callback methods on an explicit handler/provider list."""
    values = owners if isinstance(owners, (list, tuple, set, frozenset)) else [owners]
    if isinstance(values, (set, frozenset)):
        values = sorted(values, key=_structural_order_key)
    inventories: list[CallbackInventory] = []
    for index, owner in enumerate(values):
        if owner is None:
            continue
        inventories.append(
            explicit_callback_inventory(
                owner,
                methods,
                framework_modules=framework_modules,
                prefix=f"{prefix}[{index}]",
            )
        )
    return merge_callback_inventories(*inventories)


def _framework_owned_callable(value: Any, prefixes: Sequence[str]) -> bool:
    """True only for implementation code owned by the framework/SDK itself.

    No filesystem traversal is performed.  A custom subclass override has the
    application's module on the bound method and is therefore captured, while
    inherited framework handlers stay out of the upload.
    """
    candidate = value
    if isinstance(candidate, functools.partial):
        candidate = candidate.func
    if inspect.ismethod(candidate):
        candidate = candidate.__func__
    if not inspect.isfunction(candidate):
        candidate = getattr(type(candidate), "__call__", candidate)
    module = str(getattr(candidate, "__module__", "") or "")
    excluded = tuple({*prefixes, "deepintshield"})
    return any(module == root or module.startswith(root + ".") for root in excluded)


def _callable_parts(value: Any) -> tuple[Any, Any]:
    candidate = value.func if isinstance(value, functools.partial) else value
    receiver = getattr(candidate, "__self__", None) if inspect.ismethod(candidate) else None
    function = candidate.__func__ if inspect.ismethod(candidate) else candidate
    if not inspect.isfunction(function):
        function = getattr(type(candidate), "__call__", function)
        receiver = candidate
    return function, receiver


def _callable_identity(value: Any) -> tuple[int, int]:
    function, receiver = _callable_parts(value)
    return id(function), id(receiver) if receiver is not None else 0


def _structural_order_key(value: Any) -> tuple[str, str, str, int]:
    """Side-effect-free order for unordered explicit callback containers."""
    if callable(value):
        function, _ = _callable_parts(value)
        code = getattr(function, "__code__", None)
        return (
            str(getattr(function, "__module__", "") or ""),
            str(getattr(function, "__qualname__", "") or ""),
            type(value).__qualname__,
            int(getattr(code, "co_firstlineno", 0) or 0),
        )
    if isinstance(value, type):
        return value.__module__, value.__qualname__, "type", 0
    if isinstance(value, (str, int, float, bool, type(None))):
        return "builtins", type(value).__qualname__, str(value), 0
    value_type = type(value)
    return value_type.__module__, value_type.__qualname__, "object", 0


def _safe_slot_token(value: Any) -> str:
    if isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def already_wrapped(fn: Callable[..., Any]) -> bool:
    return bool(getattr(fn, _FLAG, False))


def callable_tool_name(fn: Any, fallback: str) -> str:
    """The governed tool identity = the function's own name (``crm_read``), so
    security follows the implementation, not the label the graph happened to give
    the node. Falls back to ``fallback`` (the node/registry name) for anonymous
    callables (lambdas) or objects with no usable ``__name__``."""
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name and name != "<lambda>":
        return name
    return fallback


def source_fingerprint(fn: Any) -> str:
    """A ``"src:<sha256[:16]>"`` digest of the function's source so the platform
    binds to the actual code: editing the body changes the fingerprint, which
    re-decides (code-bound caching) and feeds the registration-time threat scan.
    Best-effort - returns "" when implementation code is unavailable. Executed
    application decorator wrappers and bounded referenced helpers contribute to
    the digest, so hot reload cannot keep an old code-bound decision."""
    try:
        from ..registry import _callable_change_token

        token = _callable_change_token(fn)
        if token.startswith("sha256:"):
            return "src:" + token.removeprefix("sha256:")[:16]
    except Exception:
        pass
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError, ValueError):
        return ""
    src = src.strip()
    if not src:
        return ""
    return "src:" + hashlib.sha256(src.encode("utf-8", "replace")).hexdigest()[:16]


def wrap_callable(
    engine: Any,
    name: str,
    fn: Callable[..., Any],
    *,
    recovery_cost: str = "",
    rag_provenance: str = "",
) -> Callable[..., Any]:
    """Return a PEP-gated version of ``fn``. Preserves sync/async nature.

    Binds to the implementation: the call carries a ``source_fingerprint`` of
    ``fn`` so the PDP can detect when the code changes and so the gateway can
    threat-scan the tool's source (ASI04/T11/T17)."""
    if already_wrapped(fn):
        bound = getattr(fn, "_deepintshield_engine", engine)
        if bound is not engine:
            raise GovernanceConfigurationError(
                reason="callable is already governed by another SDK client"
            )
        return fn

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapped(*args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                preflight(engine, name, fn)
                with execution_scope(engine, fn):
                    kwargs = enforce(
                        engine, name, args, kwargs,
                        recovery_cost=recovery_cost, rag_provenance=rag_provenance,
                        tool_fingerprint=source_fingerprint(fn),
                        tool_callable=fn,
                    )
                    return await fn(*args, **kwargs)

        setattr(awrapped, _FLAG, True)
        awrapped._deepintshield_engine = engine  # type: ignore[attr-defined]
        return awrapped

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with public_agentic_error_boundary():
            preflight(engine, name, fn)
            with execution_scope(engine, fn):
                kwargs = enforce(
                    engine, name, args, kwargs,
                    recovery_cost=recovery_cost, rag_provenance=rag_provenance,
                    tool_fingerprint=source_fingerprint(fn),
                    tool_callable=fn,
                )
                return fn(*args, **kwargs)

    setattr(wrapped, _FLAG, True)
    wrapped._deepintshield_engine = engine  # type: ignore[attr-defined]
    return wrapped


def install_method_guard(
    cls: Any,
    attr: str,
    name_fn: Callable[[Any], str],
    *,
    is_async: bool = False,
    impl_fn: Callable[[Any], Any] | None = None,
) -> bool:
    """Patch ``cls.attr`` (a method taking ``self``) so every call gates through
    the PDP first - the non-bypassable equivalent of wrapping each tool instance.

    ``name_fn(self)`` → the governed tool name; ``impl_fn(self)`` → the underlying
    callable to fingerprint (defaults to ``self``). Idempotent and fail-closed:
    PDP outages, ambiguous client ownership, and policy verdicts all propagate
    before the framework method can execute."""
    orig = getattr(cls, attr, None)
    if not callable(orig):
        return False
    if getattr(orig, "_deepintshield_guarded", False):
        return True

    from ..enforcement import bind_engine, resolve_engine
    from ..gate import enforce as _gate_enforce

    def _gate_context(self: Any) -> tuple[Any, str, str, Any]:
        engine = resolve_engine(self)
        bind_engine(self, engine)
        name = name_fn(self)
        try:
            implementation = (impl_fn or (lambda s: s))(self)
            fp = source_fingerprint(implementation)
        except Exception:
            implementation = None
            fp = ""
        return engine, name, fp, implementation

    if is_async:
        @functools.wraps(orig)
        async def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                engine, name, fp, implementation = _gate_context(self)
                preflight(engine, name, implementation)
                with execution_scope(engine, self):
                    kwargs = _gate_enforce(
                        engine,
                        name,
                        args,
                        kwargs,
                        tool_fingerprint=fp,
                        tool_callable=implementation,
                    )
                    return await orig(self, *args, **kwargs)
    else:
        @functools.wraps(orig)
        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                engine, name, fp, implementation = _gate_context(self)
                preflight(engine, name, implementation)
                with execution_scope(engine, self):
                    kwargs = _gate_enforce(
                        engine,
                        name,
                        args,
                        kwargs,
                        tool_fingerprint=fp,
                        tool_callable=implementation,
                    )
                    return orig(self, *args, **kwargs)

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    guarded._deepintshield_resolves_engine = True  # type: ignore[attr-defined]
    try:
        setattr(cls, attr, guarded)
    except Exception:
        return False
    return True


def set_attr(obj: Any, attr: str, value: Any) -> bool:
    """Best-effort attribute set that tolerates frozen dataclasses / pydantic
    models. Returns True on success."""
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        try:
            object.__setattr__(obj, attr, value)
            return True
        except Exception:
            return False


def as_list(target: Any) -> list[Any]:
    """Normalise a single tool or an iterable of tools to a list (without
    consuming a one-shot generator surprise for the caller)."""
    if isinstance(target, (list, tuple, set)):
        return list(target)
    if isinstance(target, Iterable) and not hasattr(target, "name"):
        return list(target)
    return [target]


__all__ = [
    "CallbackInventory",
    "ExecutableCallback",
    "callback_method_inventory",
    "explicit_callback_inventory",
    "merge_callback_inventories",
    "wrap_callable",
    "already_wrapped",
    "set_attr",
    "as_list",
    "callable_tool_name",
    "source_fingerprint",
]
