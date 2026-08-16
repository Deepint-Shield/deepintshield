from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, model: str = "gpt-4o-mini", **kwargs: Any):
    """Return a ``langchain_openai.ChatOpenAI`` pointed at the gateway."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install langchain-openai: pip install 'deepintshield[langchain]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="langchain",
        ) from exc

    return ChatOpenAI(
        model=model,
        openai_api_base=kwargs.pop("openai_api_base", shield.langchain_base_url()),
        openai_api_key=kwargs.pop("openai_api_key", shield.api_key()),
        default_headers={**shield.headers(), **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )
