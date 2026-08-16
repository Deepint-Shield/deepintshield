"""LangGraph / LangChain binder - native ``langchain_openai`` clients pointed
at the gateway. Used for both ``shield.bind("langgraph")`` and
``shield.bind("langchain")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection

if TYPE_CHECKING:
    from ..client import DeepintShield


def model(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``langchain_openai.ChatOpenAI`` bound to the gateway."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install langchain: pip install 'deepintshield[langgraph]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="langgraph",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    return ChatOpenAI(
        model=kwargs.pop("model", model),
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**headers, **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )


def embedder(shield: "DeepintShield", model: str = "text-embedding-3-small", *, identity: bool = False, **kwargs: Any):
    """Return a native ``langchain_openai.OpenAIEmbeddings`` bound to the gateway."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install langchain: pip install 'deepintshield[langgraph]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="langgraph",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    return OpenAIEmbeddings(
        model=kwargs.pop("model", model),
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**headers, **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )
