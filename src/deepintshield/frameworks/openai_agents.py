"""OpenAI Agents SDK binder - native ``AsyncOpenAI`` pointed at the gateway,
plus an ``apply()`` shortcut that registers it as the SDK's default client so
every Agent routes through DeepintShield with no further wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def client(shield: "DeepintShield", *, identity: bool = False, **kwargs: Any):
    """Return a native ``openai.AsyncOpenAI`` bound to the gateway."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install openai: pip install 'deepintshield[openai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="openai_agents",
        ) from exc
    return AsyncOpenAI(**shield.openai_config(identity=identity, **kwargs))


def model(shield: "DeepintShield", model: str = "gpt-4o-mini", *, api: str = "chat_completions", identity: bool = False, **kwargs: Any):
    """Native Agents model; Runner retains its normal tools and execution loop.

    Choose ``api="responses"`` for Responses-compatible gateway models. Supply
    ``openai_client`` to reuse a caller-owned AsyncOpenAI connection.
    """
    if api not in {"chat_completions", "responses"}:
        raise ValueError("api must be 'chat_completions' or 'responses'")
    try:
        from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install OpenAI Agents: pip install 'deepintshield[openai-agents]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING, component="openai_agents",
        ) from exc
    native = kwargs.pop("openai_client", None)
    if native is None:
        native = client(shield, identity=identity, **dict(kwargs.pop("client_args", {})))
    model_class = OpenAIResponsesModel if api == "responses" else OpenAIChatCompletionsModel
    return model_class(model=model, openai_client=native, **kwargs)


def apply(shield: "DeepintShield", *, use_for_tracing: bool = False, identity: bool = False, api: str | None = None, **kwargs: Any):
    """Set the gateway-pointed client as the Agents SDK default and return it.

    After this, your existing ``Agent`` / ``Runner`` code is unchanged but all
    model traffic flows through DeepintShield."""
    try:
        from agents import set_default_openai_client, set_default_openai_api
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install the OpenAI Agents SDK: pip install 'deepintshield[openai-agents]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="openai_agents",
        ) from exc
    if api is not None and api not in {"chat_completions", "responses"}:
        raise ValueError("api must be 'chat_completions' or 'responses'")
    c = client(shield, identity=identity, **kwargs)
    set_default_openai_client(c, use_for_tracing=use_for_tracing)
    if api is not None:
        set_default_openai_api(api)
    return c
