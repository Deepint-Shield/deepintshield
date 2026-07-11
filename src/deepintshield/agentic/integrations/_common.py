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
from typing import Any, Callable, Iterable

from ..gate import enforce

_FLAG = "_deepintshield_wrapped"


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
    Best-effort - returns "" when source is unavailable (C/builtins/REPL). Unwraps
    decorators so we hash the real implementation, not a wrapper."""
    try:
        src = inspect.getsource(inspect.unwrap(fn))
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
        return fn

    fp = source_fingerprint(fn)

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapped(*args: Any, **kwargs: Any) -> Any:
            kwargs = enforce(
                engine, name, args, kwargs,
                recovery_cost=recovery_cost, rag_provenance=rag_provenance,
                tool_fingerprint=fp,
            )
            return await fn(*args, **kwargs)

        setattr(awrapped, _FLAG, True)
        return awrapped

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs = enforce(
            engine, name, args, kwargs,
            recovery_cost=recovery_cost, rag_provenance=rag_provenance,
            tool_fingerprint=fp,
        )
        return fn(*args, **kwargs)

    setattr(wrapped, _FLAG, True)
    return wrapped


def install_method_guard(
    cls: Any,
    attr: str,
    get_engine: Callable[[], Any],
    name_fn: Callable[[Any], str],
    *,
    is_async: bool = False,
    impl_fn: Callable[[Any], Any] | None = None,
) -> bool:
    """Patch ``cls.attr`` (a method taking ``self``) so every call gates through
    the PDP first - the non-bypassable equivalent of wrapping each tool instance.

    ``name_fn(self)`` → the governed tool name; ``impl_fn(self)`` → the underlying
    callable to fingerprint (defaults to ``self``). Idempotent (re-binds the engine
    provider), and FAIL-OPEN on infrastructure errors but FAIL-CLOSED on a verdict
    (a ``GuardrailDenied`` / approval timeout propagates and blocks the call)."""
    orig = getattr(cls, attr, None)
    if not callable(orig):
        return False
    if getattr(orig, "_deepintshield_guarded", False):
        orig._deepintshield_get_engine = get_engine  # type: ignore[attr-defined]
        return True

    from ..errors import GuardrailApprovalPending, GuardrailDenied
    from ..gate import enforce as _gate_enforce

    def _gate(self: Any, args: tuple, kwargs: dict) -> dict:
        provider = getattr(guarded, "_deepintshield_get_engine", None)
        if provider is None:
            return kwargs
        name = name_fn(self)
        try:
            fp = source_fingerprint((impl_fn or (lambda s: s))(self))
        except Exception:
            fp = ""
        try:
            return _gate_enforce(provider(), name, args, kwargs, tool_fingerprint=fp)
        except (GuardrailDenied, GuardrailApprovalPending):
            raise  # a verdict blocks - never swallowed
        except Exception:
            return kwargs  # infra hiccup → fail-open, don't break the app

    if is_async:
        @functools.wraps(orig)
        async def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs = _gate(self, args, kwargs)
            return await orig(self, *args, **kwargs)
    else:
        @functools.wraps(orig)
        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs = _gate(self, args, kwargs)
            return orig(self, *args, **kwargs)

    guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
    guarded._deepintshield_get_engine = get_engine  # type: ignore[attr-defined]
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
    "wrap_callable",
    "already_wrapped",
    "set_attr",
    "as_list",
    "callable_tool_name",
    "source_fingerprint",
]
