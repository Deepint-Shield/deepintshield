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

import logging
from typing import Any, Optional

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
    except Exception as exc:  # pragma: no cover - install-time signal
        raise ImportError(
            "langchain-core is required for shield.agentic.guard(). "
            "Install with: pip install 'deepintshield[langchain]'"
        ) from exc

    from ..gate import resolve

    class ShieldCallbackHandler(BaseCallbackHandler):
        """Gates every tool a LangChain/LangGraph run invokes through the PDP."""

        # LangChain swallows callback exceptions by default; opt in to
        # propagation so a PDP DENY actually aborts the tool body.
        raise_error = True

        def __init__(self, engine: Any) -> None:
            super().__init__()
            self._engine = engine

        def on_tool_start(
            self,
            serialized: Any,
            input_str: str,
            *,
            inputs: Any = None,
            **kwargs: Any,
        ) -> None:
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


def make_handler(engine: Any) -> Any:
    """Return a LangChain ``BaseCallbackHandler`` bound to ``engine``."""
    return _handler_class()(engine)


__all__ = ["make_handler"]
