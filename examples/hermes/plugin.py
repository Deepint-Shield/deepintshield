"""DeepintShield governance plugin for Hermes Agent (NousResearch/hermes-agent).

Drop this at ``~/.hermes/plugins/deepintshield/__init__.py`` (or ship it as a
package in the ``hermes_agent.plugins`` entry-point group). Hermes calls
``register(ctx)`` on load; ``install`` wires the ``pre_tool_call`` hook that
fires for EVERY tool - built-in, plugin, and MCP - and blocks on a PDP DENY.

For the LLM leg, set Hermes's ``base_url`` (config) to the DeepintShield
gateway so cache/guardrails/budgets/cost light up with zero code.
"""
from deepintshield import DeepintShield
from deepintshield.agentic.integrations.hermes import install

_shield = DeepintShield.from_env()


def register(ctx) -> None:
    """Hermes plugin entry point."""
    install(ctx, _shield.agentic.engine)
    # Equivalent: _shield.agentic.hermes(ctx)
