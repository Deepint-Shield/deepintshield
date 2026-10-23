"""SDK contract for gateway-owned source ID/name resolution."""
from __future__ import annotations

import json
from copy import deepcopy

import httpx
import pytest

from deepintshield import DeepintShieldError, RetrievedChunk


CANONICAL_ID = "b5358154-702d-4a5e-9f67-61be81963dd1"


@pytest.mark.parametrize("source_id", [CANONICAL_ID, "qa-support", "  qa-support  "])
def test_evaluate_sends_source_references_unchanged_without_discovery(shield_factory, source_id):
    chunks = [
        RetrievedChunk("inherited", "doc", "Inherited content"),
        {"chunk_id": "named", "document_id": "doc", "content": "Named content",
         "source_id": "  qa-support  ", "metadata": {"labels": ["support"]}},
        RetrievedChunk("canonical", "doc", "Canonical content", source_id=CANONICAL_ID),
    ]
    originals = deepcopy(chunks)
    requests = []
    trace = {"source_id": CANONICAL_ID, "retrieved_chunks": [
        {"chunk_id": "inherited", "decision": "allow"},
        {"chunk_id": "named", "decision": "allow"},
        {"chunk_id": "canonical", "decision": "allow"},
    ], "citations": [
        {"chunk_id": chunk_id, "document_id": "doc", "source_id": CANONICAL_ID, "source_name": "qa-support"}
        for chunk_id in ["inherited", "named", "canonical"]
    ]}

    def handler(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/rag-security/evaluate"
        assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
        body = json.loads(request.content)
        assert body["source_id"] == source_id
        assert "source_key" not in body
        assert [chunk["source_id"] for chunk in body["retrieved_chunks"]] == [
            "", "  qa-support  ", CANONICAL_ID,
        ]
        return httpx.Response(200, json={"result": {"final_action": "allow", "trace": trace}})

    result = shield_factory(handler).rag.evaluate(query="question", chunks=iter(chunks), source_id=source_id)

    assert len(requests) == 1
    assert result["result"]["trace"] == trace
    assert chunks == originals


def test_filter_applies_canonical_trace_decisions_to_named_and_inherited_chunks(shield_factory):
    chunks = [
        RetrievedChunk("allowed", "doc", "Public context"),
        RetrievedChunk("redacted", "doc", "Private context", source_id="qa-support"),
        RetrievedChunk("blocked", "doc", "Restricted context", source_id="finance"),
    ]
    originals = deepcopy(chunks)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/rag-security/evaluate"
        body = json.loads(request.content)
        assert body["source_id"] == "qa-support"
        assert [chunk["source_id"] for chunk in body["retrieved_chunks"]] == ["", "qa-support", "finance"]
        return httpx.Response(200, json={"result": {"final_action": "redact", "trace": {
            "source_id": CANONICAL_ID,
            "retrieved_chunks": [
                {"chunk_id": "allowed", "document_id": "doc", "decision": "allow"},
                {"chunk_id": "redacted", "document_id": "doc", "decision": "redact", "sanitized_content": "Safe context"},
            ],
            "rejected_chunks": [
                {"chunk_id": "blocked", "document_id": "doc", "decision": "reject"},
            ],
            "citations": [
                {"chunk_id": "allowed", "document_id": "doc", "source_id": CANONICAL_ID, "source_name": "qa-support"},
                {"chunk_id": "redacted", "document_id": "doc", "source_id": CANONICAL_ID, "source_name": "qa-support"},
            ],
        }}})

    allowed, raw = shield_factory(handler).rag.filter(query="question", chunks=iter(chunks), source_id="qa-support")

    assert len(requests) == 1
    assert [(chunk.chunk_id, chunk.content, chunk.source_id) for chunk in allowed] == [
        ("allowed", "Public context", ""), ("redacted", "Safe context", "qa-support"),
    ]
    assert allowed[0] is chunks[0]
    assert allowed[1] is not chunks[1]
    assert chunks == originals
    assert raw["result"]["trace"]["source_id"] == CANONICAL_ID
    assert {citation["source_id"] for citation in raw["result"]["trace"]["citations"]} == {CANONICAL_ID}


@pytest.mark.parametrize("source_id", [None, ""])
def test_empty_source_keeps_workspace_default_request_shape(shield_factory, source_id):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/rag-security/evaluate"
        body = json.loads(request.content)
        assert "source_id" not in body
        assert body["retrieved_chunks"][0]["source_id"] == ""
        return httpx.Response(200, json={"result": {"final_action": "allow", "trace": {
            "retrieved_chunks": [{"chunk_id": "c", "decision": "allow"}],
        }}})

    chunk = RetrievedChunk("c", "doc", "Public context")
    allowed, _ = shield_factory(handler).rag.filter(query="question", chunks=[chunk], source_id=source_id)
    assert allowed == [chunk]
    assert len(requests) == 1


@pytest.mark.parametrize("status,code", [(404, "resource_not_found"), (409, "conflict")])
@pytest.mark.parametrize("entrypoint", ["evaluate", "filter", "guard_retriever"])
def test_source_resolution_errors_propagate_without_default_retry(shield_factory, status, code, entrypoint):
    requests = []
    payload = {"error": "Source reference could not be resolved in this workspace"}

    def handler(request):
        requests.append(request)
        assert request.url.path == "/api/rag-security/evaluate"
        assert json.loads(request.content)["source_id"] == "qa-support"
        return httpx.Response(status, json=payload)

    class Retriever:
        def invoke(self, query):
            return [type("Document", (), {"page_content": "Private context", "metadata": {"chunk_id": "c"}})()]

    shield = shield_factory(handler)
    with pytest.raises(DeepintShieldError) as error:
        if entrypoint == "guard_retriever":
            shield.rag.guard_retriever(Retriever(), source_id="qa-support").invoke("question")
        else:
            getattr(shield.rag, entrypoint)(
                query="question", chunks=[RetrievedChunk("c", "doc", "Private context")], source_id="qa-support",
            )

    assert error.value.status_code == status
    assert error.value.code == code
    assert error.value.payload == payload
    assert len(requests) == 1
