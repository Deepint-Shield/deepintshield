from __future__ import annotations

import httpx


class _Doc:
    """Minimal stand-in for a LangChain Document."""

    def __init__(self, content: str, chunk_id: str) -> None:
        self.page_content = content
        self.metadata = {"chunk_id": chunk_id}


class _Retriever:
    def invoke(self, query: str):
        return [_Doc("safe content", "0"), _Doc("blocked", "1")]


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/api/rag-security/evaluate"):
        # The gateway safety policy permits only chunk "0".
        return httpx.Response(
            200,
            json={"result": {"trace": {"retrieved_chunks": [{"chunk_id": "0"}]}}},
        )
    return httpx.Response(404, json={})


def test_guard_retriever_drops_policy_rejected_chunks(shield_factory):
    shield = shield_factory(_handler)
    retriever = shield.rag.guard_retriever(_Retriever())
    out = retriever.invoke("what is the Q2 ledger?")
    assert len(out) == 1
    assert out[0].page_content == "safe content"


def test_guard_retriever_is_idempotent(shield_factory):
    shield = shield_factory(_handler)
    r = _Retriever()
    once = shield.rag.guard_retriever(r)
    twice = shield.rag.guard_retriever(once)  # should not double-wrap
    assert twice is once
    assert len(twice.invoke("q")) == 1


class _Embedder:
    def embed_query(self, text):
        return [0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


def _embed_handler(decision: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/guardrails/evaluate"):
            return httpx.Response(200, json={"result": {"decision": decision, "reason": "x"}})
        return httpx.Response(404, json={})

    return handler


def test_guard_embedder_allows_clean_input(shield_factory):
    shield = shield_factory(_embed_handler("allow"))
    emb = shield.rag.guard_embedder(_Embedder())
    assert emb.embed_query("hello world") == [0.1, 0.2]


def test_guard_embedder_blocks_flagged_input(shield_factory):
    import pytest

    from deepintshield import DeepintShieldBlockedError

    shield = shield_factory(_embed_handler("block"))
    emb = shield.rag.guard_embedder(_Embedder())
    with pytest.raises(DeepintShieldBlockedError):
        emb.embed_query("my ssn is 123-45-6789")
