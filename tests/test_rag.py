from __future__ import annotations

import json
from copy import deepcopy

import httpx
import pytest

from deepintshield import ErrorCode, RetrievedChunk, allowed_chunk_ids, build_chunk, filter_chunks


def test_build_chunk_roundtrips_required_fields():
    chunk = build_chunk(content="hi", chunk_id="c1", document_id="d1", trust_score=91)
    assert chunk.chunk_id == "c1"
    assert chunk.document_id == "d1"
    assert chunk.content == "hi"
    assert chunk.trust_score == 91


@pytest.mark.parametrize("factory", [RetrievedChunk, build_chunk])
@pytest.mark.parametrize("legacy_tags", [[], ["private-team"]])
def test_chunk_constructors_reject_removed_access_flag(factory, legacy_tags):
    with pytest.raises(TypeError, match="acl_tags"):
        factory(content="private content", chunk_id="c", document_id="d", acl_tags=legacy_tags)


@pytest.mark.parametrize("legacy_tags", [["private-team"], "private-team", "", {}, False, 0])
def test_evaluate_rejects_removed_access_flag_before_network(shield_factory, legacy_tags):
    def handler(request):
        pytest.fail("Unsupported permission restrictions must fail before sending any content")

    chunks = [{"chunk_id": "c", "document_id": "d", "content": "private content", "acl_tags": legacy_tags}]
    originals = deepcopy(chunks)
    shield = shield_factory(handler)

    with pytest.raises(ValueError, match="Authorize document access") as exc:
        shield.rag.evaluate(query="question", chunks=iter(chunks), source_id="qa-support")

    assert exc.value.code == ErrorCode.INVALID_ARGUMENT.value
    assert chunks == originals


@pytest.mark.parametrize("legacy_tags", [None, []])
def test_evaluate_strips_empty_legacy_placeholders_without_mutating_input(shield_factory, legacy_tags):
    requests = []
    chunks = [
        RetrievedChunk("c1", "d1", "first content", labels=["support"], pii_flags=["email"]),
        {"chunk_id": "c2", "document_id": "d2", "content": "second content", "acl_tags": legacy_tags},
    ]
    originals = deepcopy(chunks)

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["source_id"] == "qa-support"
        assert body["requester"] == "support-user"
        assert body["requester_role"] == "analyst"
        assert all("acl_tags" not in chunk for chunk in body["retrieved_chunks"])
        assert body["retrieved_chunks"][0]["labels"] == ["support"]
        assert body["retrieved_chunks"][0]["pii_flags"] == ["email"]
        return httpx.Response(200, json={"result": {"final_action": "allow"}})

    shield = shield_factory(handler)
    shield.rag.evaluate(
        query="question", chunks=iter(chunks), source_id="qa-support",
        requester="support-user", requester_role="analyst",
    )

    assert len(requests) == 1
    assert chunks == originals


def test_chunk_payload_omits_removed_access_flag():
    chunk = RetrievedChunk("c", "d", "safe content", labels=["support"], pii_flags=["email"])
    payload = chunk.to_payload()
    assert "acl_tags" not in payload
    assert payload["labels"] == ["support"]
    assert payload["pii_flags"] == ["email"]


def test_allowed_chunk_ids_reads_trace():
    response = {
        "result": {
            "trace": {
                "retrieved_chunks": [
                    {"chunk_id": "a"},
                    {"chunk_id": "b"},
                    {"chunk_id": ""},
                ]
            }
        }
    }
    assert allowed_chunk_ids(response) == {"a", "b"}


def test_allowed_chunk_ids_empty_when_no_trace():
    assert allowed_chunk_ids({}) == set()


def test_filter_chunks_removes_blocked():
    chunks = [
        RetrievedChunk(chunk_id="a", document_id="d", content="x"),
        RetrievedChunk(chunk_id="b", document_id="d", content="y"),
    ]
    response = {"result": {"trace": {"retrieved_chunks": [{"chunk_id": "a"}]}}}
    filtered = filter_chunks(chunks, response)
    assert [c.chunk_id for c in filtered] == ["a"]


def test_rag_evaluate_posts_correct_shape(shield_factory):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"result": {"trace": {"retrieved_chunks": [{"chunk_id": "c1"}]}}},
        )

    shield = shield_factory(handler)
    chunks = [build_chunk(content="hello", chunk_id="c1", document_id="d1")]
    result = shield.rag.evaluate(query="what is hello?", chunks=chunks, source_id="kb")

    assert captured["url"].endswith("/api/rag-security/evaluate")
    assert captured["body"]["query"] == "what is hello?"
    assert captured["body"]["source_id"] == "kb"
    assert captured["body"]["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert result["result"]["trace"]["retrieved_chunks"][0]["chunk_id"] == "c1"


def test_rag_filter_returns_allowed_and_raw(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"result": {"trace": {"retrieved_chunks": [{"chunk_id": "keep"}]}}},
        )

    shield = shield_factory(handler)
    chunks = [
        build_chunk(content="a", chunk_id="keep", document_id="d"),
        build_chunk(content="b", chunk_id="drop", document_id="d"),
    ]
    allowed, raw = shield.rag.filter(query="q", chunks=chunks)
    assert [c.chunk_id for c in allowed] == ["keep"]
    assert "result" in raw
