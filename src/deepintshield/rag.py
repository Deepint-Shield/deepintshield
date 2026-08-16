from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from .errors import ErrorCode, _annotate_error
from .types import RetrievedChunk

if TYPE_CHECKING:
    from .client import DeepintShield

# Retrieve-style methods we know how to wrap, in priority order.
_RETRIEVE_METHODS = ("invoke", "retrieve", "_get_relevant_documents", "get_relevant_documents")


def _query_text(query: Any) -> str:
    if isinstance(query, str):
        return query
    for attr in ("query", "query_str", "text"):
        val = getattr(query, attr, None)
        if isinstance(val, str):
            return val
    return str(query)


def build_chunk(content: str, chunk_id: str, document_id: str, **kwargs: Any) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id=document_id, content=content, **kwargs)


def allowed_chunk_ids(response: Mapping[str, Any]) -> set[str]:
    result = response.get("result", {}) or {}
    trace = result.get("trace", {}) or {}
    retrieved = trace.get("retrieved_chunks", []) or []
    return {str(c.get("chunk_id", "")).strip() for c in retrieved if str(c.get("chunk_id", "")).strip()}


def filter_chunks(chunks: Iterable[RetrievedChunk], response: Mapping[str, Any]) -> list[RetrievedChunk]:
    allowed = allowed_chunk_ids(response)
    return [chunk for chunk in chunks if chunk.chunk_id in allowed]


class RAGSurface:
    """RAG-focused helpers: evaluate a query over retrieved chunks, filter, or both."""

    def __init__(self, client: "DeepintShield") -> None:
        self._client = client

    def evaluate(
        self,
        *,
        query: str,
        chunks: Iterable[RetrievedChunk | Mapping[str, Any]],
        source_id: str | None = None,
        requester: str | None = None,
        requester_role: str | None = None,
        app_name: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        persist: bool | None = None,
    ) -> dict[str, Any]:
        chunks_payload = [
            c.to_payload() if isinstance(c, RetrievedChunk) else dict(c)
            for c in chunks
        ]
        body: dict[str, Any] = {
            "query": query,
            "requester": requester or self._client.requester,
            "requester_role": requester_role or self._client.requester_role,
            "app_name": app_name or self._client.app_name,
            "agent_name": agent_name or self._client.agent_name,
            "persist": self._client.persist if persist is None else persist,
            "retrieved_chunks": chunks_payload,
        }
        if source_id:
            body["source_id"] = source_id
        if metadata:
            body["metadata"] = dict(metadata)
        return self._client.request(
            "POST",
            "/api/rag-security/evaluate",
            json_body=body,
            error_code=ErrorCode.RAG_EVALUATION_FAILED,
            require_object=True,
        )

    def filter(
        self,
        *,
        query: str,
        chunks: Iterable[RetrievedChunk],
        source_id: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """Evaluate and return (allowed_chunks, raw_response)."""
        chunk_list = list(chunks)
        response = self.evaluate(query=query, chunks=chunk_list, source_id=source_id, **kwargs)
        return filter_chunks(chunk_list, response), response

    # ── framework retriever hook ─────────────────────────────────────────────

    def guard_retriever(
        self,
        retriever: Any,
        *,
        source_id: str | None = None,
        content_attr: str = "page_content",
        id_key: str = "chunk_id",
        doc_id_key: str = "document_id",
        chunk_mapper: Callable[[int, Any], RetrievedChunk] | None = None,
        **eval_kwargs: Any,
    ) -> Any:
        """Wrap a framework retriever so every retrieved chunk is filtered
        through the gateway's RAG-security evaluate before it reaches the LLM.

        Works with any retriever exposing one of ``invoke`` / ``retrieve`` /
        ``_get_relevant_documents`` (LangChain ``BaseRetriever``, LlamaIndex
        retrievers, or a custom callable object). Mutates the retriever in
        place and returns it, so existing graph/chain wiring is unchanged.

        Unauthorised chunks (per the gateway verdict) are dropped from the
        returned list; the order of allowed chunks is preserved.
        """
        method_name = next(
            (m for m in _RETRIEVE_METHODS if callable(getattr(retriever, m, None))), None
        )
        if method_name is None:
            raise _annotate_error(
                TypeError(
                    "guard_retriever: retriever exposes no known retrieve method "
                    f"({', '.join(_RETRIEVE_METHODS)})"
                ),
                ErrorCode.RAG_RETRIEVER_UNSUPPORTED,
            )
        original = getattr(retriever, method_name)
        if getattr(original, "_deepintshield_wrapped", False):
            return retriever
        surface = self

        @functools.wraps(original)
        def wrapped(query: Any, *args: Any, **kwargs: Any) -> Any:
            docs = original(query, *args, **kwargs)
            if not isinstance(docs, (list, tuple)) or not docs:
                return docs
            return surface._filter_documents(
                _query_text(query), list(docs),
                source_id=source_id, content_attr=content_attr,
                id_key=id_key, doc_id_key=doc_id_key, chunk_mapper=chunk_mapper,
                **eval_kwargs,
            )

        wrapped._deepintshield_wrapped = True  # type: ignore[attr-defined]
        try:
            setattr(retriever, method_name, wrapped)
        except Exception:  # frozen pydantic model
            object.__setattr__(retriever, method_name, wrapped)
        return retriever

    def _filter_documents(
        self,
        query: str,
        docs: list[Any],
        *,
        source_id: str | None,
        content_attr: str,
        id_key: str,
        doc_id_key: str,
        chunk_mapper: Callable[[int, Any], RetrievedChunk] | None,
        **eval_kwargs: Any,
    ) -> list[Any]:
        chunk_ids: list[str] = []
        chunks: list[RetrievedChunk] = []
        for i, doc in enumerate(docs):
            if chunk_mapper is not None:
                chunk = chunk_mapper(i, doc)
            else:
                meta = dict(getattr(doc, "metadata", None) or {})
                cid = str(meta.get(id_key) or i)
                content = getattr(doc, content_attr, None)
                if content is None:
                    content = getattr(doc, "text", None) or str(doc)
                chunk = build_chunk(
                    content=content,
                    chunk_id=cid,
                    document_id=str(meta.get(doc_id_key) or ""),
                )
            chunk_ids.append(chunk.chunk_id)
            chunks.append(chunk)
        allowed, _resp = self.filter(
            query=query, chunks=chunks, source_id=source_id, **eval_kwargs
        )
        allowed_ids = {c.chunk_id for c in allowed}
        return [doc for doc, cid in zip(docs, chunk_ids) if cid in allowed_ids]

    # ── embedding-input guard (Portkey-parity "before request" stage) ─────────

    def guard_embedder(
        self,
        embedder: Any,
        *,
        stage: str = "input",
        raise_on_block: bool = True,
        **guard_kwargs: Any,
    ) -> Any:
        """Wrap a framework embedder so each input string is screened by the
        gateway guardrail *before* it is vectorised.

        This is the input-side complement to :meth:`guard_retriever` (which
        filters retrieved chunks). Works with LangChain (``embed_documents`` /
        ``embed_query``) and LlamaIndex (``get_text_embedding`` /
        ``get_text_embedding_batch`` / ``get_query_embedding``). Mutates the
        embedder in place and returns it. On a blocking verdict the embed call
        raises (``raise_on_block=True``) rather than vectorising the text.
        """
        methods = (
            "embed_documents", "embed_query",
            "get_text_embedding", "get_text_embedding_batch", "get_query_embedding",
        )
        wrapped_any = False
        for name in methods:
            fn = getattr(embedder, name, None)
            if not callable(fn) or getattr(fn, "_deepintshield_wrapped", False):
                continue
            wrapped = self._wrap_embed(fn, stage, raise_on_block, guard_kwargs)
            try:
                setattr(embedder, name, wrapped)
            except Exception:  # frozen model
                object.__setattr__(embedder, name, wrapped)
            wrapped_any = True
        if not wrapped_any:
            raise _annotate_error(
                TypeError(
                    "guard_embedder: no known embed method found "
                    f"({', '.join(methods)})"
                ),
                ErrorCode.RAG_EMBEDDER_UNSUPPORTED,
            )
        return embedder

    def _wrap_embed(self, original, stage, raise_on_block, guard_kwargs):
        surface = self

        @functools.wraps(original)
        def wrapped(text_or_texts: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(text_or_texts, str):
                texts: list[Any] = [text_or_texts]
            elif isinstance(text_or_texts, (list, tuple)):
                texts = list(text_or_texts)
            else:
                texts = []
            for t in texts:
                if isinstance(t, str):
                    surface._client.guard(
                        stage=stage, input=t, raise_on_block=raise_on_block, **guard_kwargs
                    )
            return original(text_or_texts, *args, **kwargs)

        wrapped._deepintshield_wrapped = True  # type: ignore[attr-defined]
        return wrapped
