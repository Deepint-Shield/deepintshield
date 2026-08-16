"""PydanticAI binder - native ``OpenAIChatModel`` whose provider points at the
gateway. Headers ride on an injected async httpx client since the provider
takes a model + provider rather than loose ``default_headers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from ..errors import ErrorCode, _dependency_error
from ..transport import connection_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


def model(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``pydantic_ai`` OpenAI-compatible model bound to the
    gateway. Pass it to ``pydantic_ai.Agent(model)``."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install pydantic-ai: pip install 'deepintshield[pydanticai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="pydanticai",
        ) from exc
    base_url = kwargs.pop("base_url", shield.openai_base_url())
    headers = connection_headers(shield, identity=identity)
    http_client = kwargs.pop(
        "http_client", httpx.AsyncClient(headers=headers, timeout=shield.timeout)
    )
    provider = OpenAIProvider(
        base_url=base_url,
        api_key=kwargs.pop("api_key", shield.api_key()),
        http_client=http_client,
    )
    return OpenAIChatModel(kwargs.pop("model", model), provider=provider)


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
    return Agent(model(shield, model_name, identity=identity), instructions=instructions, **kwargs)
