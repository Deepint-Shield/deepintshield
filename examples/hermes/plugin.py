"""DeepintShield governance plugin for Hermes Agent (NousResearch/hermes-agent).

Drop this at ``~/.hermes/plugins/deepintshield/__init__.py`` (or ship it as a
package in the ``hermes_agent.plugins`` entry-point group). Hermes calls
``register(ctx)`` on load. Constructing the client arms Hermes'
``model_tools.handle_function_call`` dispatcher, which covers every built-in,
plugin and MCP tool and blocks on a PDP denial.

For the LLM leg, set Hermes's ``base_url`` (config) to the DeepintShield
gateway so cache/guardrails/budgets/cost light up with zero code.
"""
from deepintshield import DeepintShield

_shield = DeepintShield.from_env()


def register(ctx) -> None:
    """Hermes plugin entry point; importing this module installs enforcement."""
    del ctx
