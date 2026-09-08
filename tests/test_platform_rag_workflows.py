"""Offline retrieval-to-generation workflows; provider identities are routing fixtures."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx
import pytest

from deepintshield import DeepintShieldError, build_chunk
from .test_model_compatibility import CHAT_PROVIDERS, native_response


@pytest.mark.parametrize("provider", CHAT_PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
def test_filtered_rag_context_reaches_each_chat_provider_without_rejected_content(shield_factory, provider, stream):
    requests = []
    model = f"{provider}/deployment/current-chat"
    chunks = [build_chunk(content=text, chunk_id=key, document_id="manual") for key, text in
              [("allowed", "Public operating procedure"), ("redacted", "private phone"), ("blocked", "restricted record")]]

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/rag-security/evaluate":
            body = json.loads(request.content)
            assert body["source_id"] == "workspace-kb"
            assert body["query"] == "What is the procedure?"
            assert body["requester"] == "reader@example.test"
            assert body["metadata"] == {"retrieval_id": "fixture-retrieval"}
            return httpx.Response(200, json={"result": {"final_action": "redact", "trace": {
                "retrieved_chunks": [
                    {"chunk_id": "allowed", "document_id": "manual", "decision": "allow"},
                    {"chunk_id": "redacted", "document_id": "manual", "decision": "redact", "sanitized_content": "Contact [REDACTED]"},
                    {"chunk_id": "blocked", "document_id": "manual", "decision": "reject"},
                ],
            }}})
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == model
        assert body["messages"][0]["content"] == "[allowed] Public operating procedure\n[redacted] Contact [REDACTED]"
        assert b"private phone" not in request.content
        assert b"restricted record" not in request.content
        return native_response("chat", model, stream)

    shield = shield_factory(handler, base_url="https://gateway.invalid")
    allowed, _ = shield.rag.filter(query="What is the procedure?", chunks=iter(chunks), source_id="workspace-kb",
                                  requester="reader@example.test", metadata={"retrieval_id": "fixture-retrieval"})
    context = "\n".join(f"[{chunk.chunk_id}] {chunk.content}" for chunk in allowed)
    result = shield.chat(model=model, messages=[{"role": "system", "content": context},
                         {"role": "user", "content": "What is the procedure?"}], stream=stream)
    if stream:
        assert len(list(result)) == 1
    else:
        assert result["choices"][0]["message"]["content"] == "OK"
    assert len(requests) == 2
    assert all(request.headers["x-deepintshield-vk"] == "sk-ds-test" for request in requests)
    assert [chunk.content for chunk in chunks] == ["Public operating procedure", "private phone", "restricted record"]


@dataclass
class Document:
    page_content: str
    metadata: dict


@pytest.mark.parametrize("async_method", ["ainvoke", "aretrieve", "aget_relevant_documents", "_aget_relevant_documents", "invoke"])
def test_async_retriever_cannot_return_unfiltered_documents(shield_factory, async_method):
    originals = [Document("private original", {"chunk_id": "a", "document_id": "doc"}),
                 Document("blocked original", {"chunk_id": "b", "document_id": "doc"})]
    calls = []

    async def retrieve(self, query, **kwargs):
        assert kwargs == {"config": {"trace": "fixture"}}
        return originals

    retriever = type("AsyncRetriever", (), {async_method: retrieve})()

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"result": {"final_action": "redact", "trace": {"retrieved_chunks": [
            {"chunk_id": "a", "decision": "redact", "sanitized_content": "safe context"},
        ]}}})

    shield = shield_factory(handler)
    guarded = shield.rag.guard_retriever(retriever, source_id="async-kb", requester="async-reader")
    assert shield.rag.guard_retriever(guarded) is guarded
    docs = asyncio.run(getattr(guarded, async_method)("query", config={"trace": "fixture"}))
    assert [doc.page_content for doc in docs] == ["safe context"]
    assert originals[0].page_content == "private original"
    assert len(calls) == 1
    assert calls[0]["query"] == "query" and calls[0]["source_id"] == "async-kb"
    assert calls[0]["requester"] == "async-reader"


@pytest.mark.parametrize("status,payload", [(503, {"error": "unavailable"}), (200, []), (200, "invalid")])
def test_async_retrieval_evaluation_failure_stops_generation(shield_factory, status, payload):
    class Retriever:
        async def ainvoke(self, query):
            return [Document("private context", {"chunk_id": "a"})]

    calls = []
    def handler(request):
        calls.append(request.url.path)
        assert request.url.path == "/api/rag-security/evaluate"
        return httpx.Response(status, json=payload)

    shield = shield_factory(handler)
    guarded = shield.rag.guard_retriever(Retriever())
    async def workflow():
        docs = await guarded.ainvoke("question")
        shield.chat(model="openai/fixture", messages=[{"role": "user", "content": docs[0].page_content}])
    with pytest.raises(DeepintShieldError):
        asyncio.run(workflow())
    assert calls == ["/api/rag-security/evaluate"]


def test_retriever_guards_both_sync_and_async_entry_points(shield_factory):
    class Retriever:
        def invoke(self, query):
            return [Document("private", {"chunk_id": "a"})]
        async def ainvoke(self, query):
            return [Document("private", {"chunk_id": "a"})]

    queries = []
    def handler(request):
        queries.append(json.loads(request.content)["query"])
        return httpx.Response(200, json={"result": {"final_action": "block", "trace": {"retrieved_chunks": []}}})
    guarded = shield_factory(handler).rag.guard_retriever(Retriever())
    assert guarded.invoke("sync") == []
    assert asyncio.run(guarded.ainvoke("async")) == []
    assert queries == ["sync", "async"]


def test_async_delegation_filters_once_and_concurrent_queries_stay_isolated(shield_factory):
    class Retriever:
        def invoke(self, query):
            return self.get_relevant_documents(query)
        def get_relevant_documents(self, query):
            return [Document(f"private:{query}", {"chunk_id": "a"})]
        async def ainvoke(self, query):
            return await asyncio.to_thread(self.invoke, query)

    queries = []
    def handler(request):
        body = json.loads(request.content)
        query = body["query"]
        assert body["retrieved_chunks"][0]["content"] == f"private:{query}"
        queries.append(query)
        return httpx.Response(200, json={"result": {"final_action": "redact", "trace": {"retrieved_chunks": [
            {"chunk_id": "a", "decision": "redact", "sanitized_content": f"safe:{query}"},
        ]}}})
    guarded = shield_factory(handler).rag.guard_retriever(Retriever())
    async def concurrent():
        return await asyncio.gather(*(guarded.ainvoke(f"q{index}") for index in range(12)))
    results = asyncio.run(concurrent())
    assert [[doc.page_content for doc in docs] for docs in results] == [[f"safe:q{index}"] for index in range(12)]
    assert sorted(queries) == sorted(f"q{index}" for index in range(12))


def test_async_retriever_failure_does_not_disable_subsequent_filtering(shield_factory):
    class Retriever:
        async def ainvoke(self, query):
            if query == "fail":
                raise RuntimeError("retrieval failed")
            return [Document("private", {"chunk_id": "a"})]
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"result": {"final_action": "block", "trace": {"retrieved_chunks": []}}})
    guarded = shield_factory(handler).rag.guard_retriever(Retriever())
    async def workflow():
        with pytest.raises(RuntimeError, match="retrieval failed"):
            await guarded.ainvoke("fail")
        assert await guarded.ainvoke("then block") == []
    asyncio.run(workflow())
    assert len(calls) == 1
