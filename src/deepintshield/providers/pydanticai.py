from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_model(shield: "DeepintShield", *, model: str = "gpt-4o-mini", **kwargs: Any):
    """Build a PydanticAI OpenAI-compatible model bound to the gateway."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install pydantic-ai: pip install 'deepintshield[pydanticai]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="pydanticai",
        ) from exc

    from ..frameworks.pydanticai import model as bind_model

    kwargs.setdefault("base_url", f"{shield.pydanticai_base_url()}/v1")
    return bind_model(shield, model, **kwargs)


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

    model_options = dict(kwargs.pop("model_kwargs", {}))
    for option in ("base_url", "api_key", "http_client", "default_headers", "timeout", "max_retries", "identity", "profile", "settings", "api", "openai_client", "client_args"):
        if option in kwargs:
            model_options[option] = kwargs.pop(option)
    return Agent(
        build_model(shield, model=model, **model_options),
        instructions=instructions or "",
        **kwargs,
    )


def build_client(shield: "DeepintShield", **kwargs: Any):
    return build_agent(shield, **kwargs)
