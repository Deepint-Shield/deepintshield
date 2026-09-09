"""PydanticAI binder with a native OpenAI client pointed at the gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def model(shield: "DeepintShield", model: str = "gpt-4o-mini", *, api: str = "chat_completions", identity: bool = False, **kwargs: Any):
    """Return a native ``pydantic_ai`` OpenAI-compatible model bound to the
    gateway. Pass it to ``pydantic_ai.Agent(model)``."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install pydantic-ai: pip install 'deepintshield[pydanticai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="pydanticai",
        ) from exc
    if api not in {"chat_completions", "responses"}:
        raise ValueError("api must be 'chat_completions' or 'responses'")
    client = kwargs.pop("openai_client", None)
    client_options = dict(kwargs.pop("client_args", {}))
    for option in ("base_url", "api_key", "default_headers", "timeout", "http_client", "max_retries"):
        if option in kwargs:
            client_options[option] = kwargs.pop(option)
    # Attach gateway headers at the SDK layer, so a caller's custom transport
    # retains authentication and attribution too. Let OpenAI choose its native
    # HTTP implementation (httpx or httpx2, depending on the installed version).
    if client is None:
        client = AsyncOpenAI(**shield.openai_config(identity=identity, **client_options))
    provider = OpenAIProvider(openai_client=client)
    model_class = OpenAIResponsesModel if api == "responses" else OpenAIChatModel
    return model_class(kwargs.pop("model", model), provider=provider, **kwargs)


def agent(shield: "DeepintShield", model_name: str = "gpt-4o-mini", *, instructions: str = "", identity: bool = False, **kwargs: Any):
    """Return a ready ``pydantic_ai.Agent`` bound to the gateway."""
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install pydantic-ai: pip install 'deepintshield[pydanticai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="pydanticai",
        ) from exc
    model_options = kwargs.pop("model_kwargs", {})
    return Agent(model(shield, model_name, identity=identity, **model_options), instructions=instructions, **kwargs)
