"""Google ADK (Agent Development Kit) enforcement.

ADK exposes an app-level Plugin API: a ``BasePlugin`` registered once on the
runner covers EVERY agent and tool without per-agent wiring - the closest analog
to ``govern()``. ``before_tool_callback`` is the PEP: returning a dict
short-circuits the tool and hands the message back to the model.

Minimal-code surface:
    from deepintshield import DeepintShield
    from google.adk.runners import InMemoryRunner
    shield = DeepintShield(virtual_key="sk-ds-...")
    runner = InMemoryRunner(agent=agent, plugins=[shield.agentic.google_adk()])

The LLM leg stays native via ADK's LiteLLM/GenAI model wrapper pointed at the
gateway (we already ship ``providers/litellm`` + ``providers/genai``).

Fail-OPEN on infrastructure error, fail-CLOSED on a verdict (DENY returns a
short-circuit dict so the tool body never runs).
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import GuardrailApprovalPending, GuardrailDenied
from ..gate import resolve
from ._common import source_fingerprint

log = logging.getLogger(__name__)


def _tool_name(tool: Any) -> str:
    for attr in ("name", "__name__"):
        v = getattr(tool, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(tool).__name__


def _deny_payload(tool_name: str, reason: str) -> dict[str, Any]:
    """The ADK short-circuit shape: a tool-result dict the model sees instead of
    the real tool output."""
    return {"error": f"deepintshield policy blocked {tool_name!r}: {reason}", "status": "blocked"}


def plugin(engine: Any) -> Any:
    """Return a ``BasePlugin`` instance that gates every ADK tool call through
    the PDP. Imported lazily so ``google.adk`` is only required on use."""
    from google.adk.plugins.base_plugin import BasePlugin

    class DeepintShieldPlugin(BasePlugin):
        def __init__(self) -> None:
            super().__init__(name="deepintshield")

        async def before_tool_callback(self, *, tool: Any = None, tool_args: Any = None, tool_context: Any = None, **_: Any) -> Any:  # noqa: ANN401
            name = _tool_name(tool)
            call_args = tool_args if isinstance(tool_args, dict) else {"args": tool_args}
            try:
                fp = source_fingerprint(getattr(tool, "func", None) or tool)
            except Exception:
                fp = ""
            try:
                resolve(engine, name, (), call_args, tool_fingerprint=fp)
            except GuardrailDenied as d:
                return _deny_payload(name, d.reason or str(d))
            except GuardrailApprovalPending as p:
                return _deny_payload(name, f"approval required (decision {p.decision_id})")
            except Exception:  # infra hiccup → fail-open
                log.debug("deepintshield adk PEP fail-open for %s", name, exc_info=True)
            return None  # ALLOW → let ADK run the tool

    return DeepintShieldPlugin()


def enforce(get_engine: Any) -> bool:
    """Auto-enforcement entry point (``install_all``). ADK plugins are attached
    explicitly to a runner, so there is no import-time boundary to patch; use
    ``shield.agentic.google_adk()``. Returns False."""
    return False


__all__ = ["plugin", "enforce"]
