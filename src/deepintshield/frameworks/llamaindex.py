"""LlamaIndex binder - native OpenAI-compatible LLM and embedding clients
pointed at the gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection, _merge_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


def llm(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return native ``OpenAILike`` for arbitrary gateway model IDs.

    Set ``context_window`` and ``is_function_calling_model`` from the selected
    model's capabilities when using context management or tools.
    """
    try:
        from llama_index.llms.openai_like import OpenAILike
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install llama-index: pip install 'deepintshield[llamaindex]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="llamaindex",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    return OpenAILike(
        model=kwargs.pop("model", model),
        api_base=kwargs.pop("api_base", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers=_merge_headers(headers, kwargs.pop("default_headers", None) or {}),
        is_chat_model=kwargs.pop("is_chat_model", True),
        **kwargs,
    )


def embedder(shield: "DeepintShield", model: str = "text-embedding-3-small", *, identity: bool = False, **kwargs: Any):
    """Return native ``OpenAILikeEmbedding`` for arbitrary embedding model IDs,
    to the gateway."""
    try:
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install llama-index: pip install 'deepintshield[llamaindex]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="llamaindex",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    return OpenAILikeEmbedding(
        model_name=kwargs.pop("model_name", kwargs.pop("model", model)),
        api_base=kwargs.pop("api_base", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers=_merge_headers(headers, kwargs.pop("default_headers", None) or {}),
        **kwargs,
    )
