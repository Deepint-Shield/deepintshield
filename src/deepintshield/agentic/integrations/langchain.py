"""LangChain / LangGraph enforcement via the framework's NATIVE callback system.

This is the thinnest possible integration. LangChain already dispatches an
``on_tool_start`` event to every registered callback handler *before* any tool
runs - a LangChain ``BaseTool``, a ``@tool`` function, a LangGraph ``ToolNode``
or a prebuilt ReAct agent's tools. We supply a single ``BaseCallbackHandler``
that calls the PDP in that hook and aborts the tool when the verdict blocks. The
framework does all of the tool discovery, argument parsing and dispatch; the
vendor code here is one small class.

The application attaches it once and never touches individual tools::

    shield = DeepintShield.from_env()
    agent.invoke({"input": "…"}, config={"callbacks": [shield.agentic.guard()]})

No per-tool decorator, no wrapping, and no parameters: the tool name comes from
LangChain and every policy / tier / identity input is resolved server-side.

Note on MASK: a callback cannot rewrite a tool's arguments, so arg-level masking
isn't applied on this path (the MASK decision is still recorded server-side and
the call proceeds). Use ``shield.agentic.tool`` / a per-framework adapter when
you need the SDK to redact arguments locally before the body runs.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Optional

from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ..obligations import apply_obligations
from ._common import source_fingerprint

log = logging.getLogger(__name__)

# Built lazily on first use so importing this module never hard-requires
# langchain-core (the surface imports it lazily too).
_HANDLER_CLASS: Optional[type] = None


def _handler_class() -> type:
    global _HANDLER_CLASS
    if _HANDLER_CLASS is not None:
        return _HANDLER_CLASS

    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except Exception:  # pragma: no cover - install-time signal
        raise GovernanceConfigurationError(
            framework="langchain",
            code="framework_dependency_missing",
        ) from None

    from ..gate import resolve

    class ShieldCallbackHandler(BaseCallbackHandler):
        """Gates every tool a LangChain/LangGraph run invokes through the PDP."""

        # LangChain swallows callback exceptions by default; opt in to
        # propagation so a PDP DENY actually aborts the tool body.
        raise_error = True

        def __init__(self, engine: Any) -> None:
            super().__init__()
            self._engine = engine

        @public_agentic_boundary
        def on_tool_start(
            self,
            serialized: Any,
            input_str: str,
            *,
            inputs: Any = None,
            **kwargs: Any,
        ) -> None:
            # Current SDK clients guard BaseTool.run/arun directly. Retain this
            # callback for older/custom LangChain runtimes without deciding
            # twice when both integration styles are present.
            if _automatic_guard_active():
                return
            name = ""
            if isinstance(serialized, dict):
                name = serialized.get("name") or ""
            name = name or "tool"
            call_args = inputs if isinstance(inputs, dict) else {"input": input_str}
            # Raises GuardrailDenied on DENY / denied approval and blocks on
            # REQUIRE_APPROVAL; returns on ALLOW (and on MASK, which a callback
            # cannot apply locally). recovery_cost / rag_provenance / tiering are
            # all resolved server-side from the tool name.
            resolve(self._engine, name, (), call_args)

    _HANDLER_CLASS = ShieldCallbackHandler
    return _HANDLER_CLASS


@public_agentic_boundary
def make_handler(engine: Any) -> Any:
    """Return a LangChain ``BaseCallbackHandler`` bound to ``engine``."""
    handler_class = _handler_class()
    try:
        return handler_class(engine)
    except Exception as error:
        raise GovernanceConfigurationError(
            framework="langchain",
            reason=str(error),
            code="framework_integration_unsupported",
        ) from None


def _automatic_guard_active() -> bool:
    try:
        from langchain_core.tools import BaseTool

        return bool(
            getattr(BaseTool.run, "_deepintshield_guarded", False)
            and getattr(BaseTool.arun, "_deepintshield_guarded", False)
        )
    except Exception:
        return False


def _guard_tool(tool: Any, tool_input: Any, trailing: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    from ..enforcement import bind_engine, ensure_topology_reported, resolve_engine
    from ..gate import resolve

    engine = resolve_engine(tool)
    bind_engine(tool, engine)
    name = getattr(tool, "name", None) or type(tool).__name__
    impl = getattr(tool, "_run", None) or getattr(tool, "func", None) or tool
    try:
        fp = source_fingerprint(impl)
    except Exception:
        fp = ""
    ensure_topology_reported(engine, tool, required=True)
    decision = resolve(
        engine,
        str(name),
        (tool_input, *trailing),
        kwargs,
        tool_fingerprint=fp,
        tool_callable=impl,
    )
    if isinstance(tool_input, dict):
        tool_input = apply_obligations(tool_input, decision.obligations)
    return tool_input


def enforce() -> bool:
    """Guard LangChain's sync and async BaseTool execution boundaries."""
    try:
        from langchain_core.tools import BaseTool
    except Exception:
        return False

    installed = True
    for attr in ("run", "arun"):
        original = getattr(BaseTool, attr, None)
        if not callable(original):
            installed = False
            continue
        if getattr(original, "_deepintshield_guarded", False):
            continue
        is_async = inspect.iscoroutinefunction(original)
        if attr == "arun" and not is_async:
            installed = False
            continue

        if is_async:

            @functools.wraps(original)
            @public_agentic_boundary
            async def guarded(
                self: Any,
                tool_input: Any,
                *args: Any,
                _original=original,
                **kwargs: Any,
            ) -> Any:
                tool_input = _guard_tool(self, tool_input, args, kwargs)
                return await _original(self, tool_input, *args, **kwargs)

        else:

            @functools.wraps(original)
            @public_agentic_boundary
            def guarded(  # type: ignore[no-redef]
                self: Any,
                tool_input: Any,
                *args: Any,
                _original=original,
                **kwargs: Any,
            ) -> Any:
                tool_input = _guard_tool(self, tool_input, args, kwargs)
                return _original(self, tool_input, *args, **kwargs)

        guarded._deepintshield_guarded = True  # type: ignore[attr-defined]
        try:
            setattr(BaseTool, attr, guarded)
        except Exception:
            installed = False
    return installed and _automatic_guard_active()


__all__ = ["make_handler", "enforce"]
