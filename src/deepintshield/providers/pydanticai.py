from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_model(shield: "DeepintShield", *, model: str = "gpt-4o-mini", **_kwargs: Any):
    """Build a PydanticAI OpenAI-compatible model bound to the gateway."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install pydantic-ai: pip install 'deepintshield[pydanticai]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="pydanticai",
        ) from exc

    provider = OpenAIProvider(
        base_url=f"{shield.pydanticai_base_url()}/v1",
        api_key=shield.api_key(),
    )
    return OpenAIChatModel(model, provider=provider)


def build_agent(shield: "DeepintShield", *, model: str = "gpt-4o-mini", instructions: str | None = None, **kwargs: Any):
    """Return a ready-to-use ``pydantic_ai.Agent`` bound to DeepintShield."""
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install pydantic-ai: pip install 'deepintshield[pydanticai]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="pydanticai",
        ) from exc

    return Agent(
        build_model(shield, model=model),
        instructions=instructions or "",
        **kwargs,
    )


def build_client(shield: "DeepintShield", **kwargs: Any):
    return build_agent(shield, **kwargs)
