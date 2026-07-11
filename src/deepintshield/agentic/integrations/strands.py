"""AWS Strands Agents enforcement.

Strands (the AWS agent framework, default runtime for Bedrock AgentCore) exposes
a typed hook system. A ``HookProvider`` subscribing to ``BeforeToolInvocationEvent``
is the PEP: the callback can cancel the invocation or swap the ``selected_tool``,
so a DENY maps to a cancellation / refusal.

Minimal-code surface:
    from deepintshield import DeepintShield
    from strands import Agent
    shield = DeepintShield(virtual_key="sk-ds-...")
    agent = Agent(model=..., tools=[...], hooks=[shield.agentic.strands()])

The LLM leg stays native via Strands' ``OpenAIModel(base_url=gateway)`` or its
LiteLLM model class pointed at the gateway.

Fail-OPEN on infrastructure error, fail-CLOSED on a verdict.
"""

from __future__ import annotations

import logging
from typing import Any

from ..errors import GuardrailApprovalPending, GuardrailDenied
from ..gate import resolve
from ._common import source_fingerprint

log = logging.getLogger(__name__)


def _tool_name(event: Any) -> str:
    use = getattr(event, "tool_use", None)
    if isinstance(use, dict):
        n = use.get("name")
        if isinstance(n, str) and n:
            return n
    for attr in ("tool_name", "name"):
        v = getattr(event, attr, None)
        if isinstance(v, str) and v:
            return v
    tool = getattr(event, "selected_tool", None)
    return getattr(tool, "tool_name", None) or getattr(tool, "name", None) or "tool"


def _tool_args(event: Any) -> dict[str, Any]:
    use = getattr(event, "tool_use", None)
    if isinstance(use, dict) and isinstance(use.get("input"), dict):
        return use["input"]
    return {}


def hook_provider(engine: Any) -> Any:
    """Return a Strands ``HookProvider`` that gates every tool invocation through
    the PDP. Imported lazily so ``strands`` is only required on use."""
    from strands.hooks import HookProvider, HookRegistry
    from strands.experimental.hooks import BeforeToolInvocationEvent

    class DeepintShieldHooks(HookProvider):
        def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
            registry.add_callback(BeforeToolInvocationEvent, self._before_tool)

        def _before_tool(self, event: Any) -> None:
            name = _tool_name(event)
            try:
                fp = source_fingerprint(getattr(getattr(event, "selected_tool", None), "func", None))
            except Exception:
                fp = ""
            try:
                resolve(engine, name, (), _tool_args(event), tool_fingerprint=fp)
            except (GuardrailDenied, GuardrailApprovalPending) as v:
                # Cancel the invocation - Strands surfaces this as the tool result.
                reason = getattr(v, "reason", None) or str(v)
                try:
                    event.selected_tool = None  # swap out the tool → nothing runs
                except Exception:
                    pass
                raise PermissionError(f"deepintshield blocked {name!r}: {reason}") from v
            except Exception:  # infra hiccup → fail-open
                log.debug("deepintshield strands PEP fail-open for %s", name, exc_info=True)

    return DeepintShieldHooks()


def enforce(get_engine: Any) -> bool:
    """Auto-enforcement entry point (``install_all``). Strands hooks are attached
    explicitly to the Agent, so there is no import-time boundary to patch; use
    ``shield.agentic.strands()``. Returns False."""
    return False


__all__ = ["hook_provider", "enforce"]
