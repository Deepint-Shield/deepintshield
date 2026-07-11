"""OpenClaw (formerly Clawdbot → Moltbot) integration.

OpenClaw is a Node/TypeScript agent runtime, so it integrates in two layers:

LAYER 1 - LLM gateway, ZERO code (this module): add a ``models.providers``
entry pointing OpenClaw at the DeepintShield gateway. Every model call then
transits the gateway - VK auth, semantic/exact cache, request coalescing,
guardrails, budgets, cost analytics, catalog fallbacks - with no code change.

LAYER 2 - in-process tool governance (PEP/PDP, grants, code-threat scan): a
small TypeScript plugin (``openclaw.plugin.json`` + ``definePluginEntry``) that
calls the gateway's REST ``/decide`` + approvals + scan APIs on the
``before_tool_call`` / ``after_tool_call`` / ``before_install`` hooks. OpenClaw
plugins run in-process in the Node gateway, so the Python SDK cannot be embedded;
Layer 2 ships as the ``deepintshield-openclaw`` npm plugin (see examples). This
module only produces the Layer-1 config from a live ``DeepintShield`` client.
"""

from __future__ import annotations

from typing import Any


def provider_config(engine: Any, *, gateway_url: str = "", models: Any = None) -> dict:
    """Build the OpenClaw ``models.providers.deepintshield`` config block bound
    to this client's gateway + VK. Merge into ``openclaw.json`` and set
    ``agents.defaults.model.primary`` to ``"deepintshield/<model-id>"``.

    ``models`` is an optional list of ``{"id", "contextWindow", "maxTokens",
    "cost"}`` dicts advertised to OpenClaw; when omitted the runtime resolves
    models dynamically from the gateway catalog.
    """
    base = (gateway_url or engine.gateway_url).rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    provider: dict[str, Any] = {
        "baseUrl": base,
        "apiKey": engine.virtual_key,
        "api": "openai-completions",
    }
    if models:
        provider["models"] = list(models)
    return {"models": {"providers": {"deepintshield": provider}}}


__all__ = ["provider_config"]
