"""LlamaIndex binder - native ``llama_index`` OpenAI LLM + embedding clients
pointed at the gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..transport import connection

if TYPE_CHECKING:
    from ..client import DeepintShield


def llm(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``llama_index.llms.openai.OpenAI`` bound to the gateway."""
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install llama-index: pip install 'deepintshield[llamaindex]'") from exc
    base_url, headers = connection(shield, identity=identity)
    return OpenAI(
        model=kwargs.pop("model", model),
        api_base=kwargs.pop("api_base", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**headers, **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )


def embedder(shield: "DeepintShield", model: str = "text-embedding-3-small", *, identity: bool = False, **kwargs: Any):
    """Return a native ``llama_index.embeddings.openai.OpenAIEmbedding`` bound
    to the gateway."""
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install llama-index: pip install 'deepintshield[llamaindex]'") from exc
    base_url, headers = connection(shield, identity=identity)
    return OpenAIEmbedding(
        model=kwargs.pop("model", model),
        api_base=kwargs.pop("api_base", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**headers, **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )
