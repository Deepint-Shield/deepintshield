from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from deepintshield import RetrievedChunk, allowed_chunk_ids, filter_chunks


def response_for(*records, decision="redact"):
    return {"result": {"final_action": decision, "trace": {"retrieved_chunks": list(records)}}}


@pytest.mark.parametrize("sanitized", ["[REDACTED]", "", "  " + "safe context " * 100 + "\n"])
def test_filter_uses_complete_sanitized_content_without_mutating_source(sanitized):
    original = RetrievedChunk("a", "doc", "private original", metadata={"source": "kb"})
    response = response_for({"chunk_id": "a", "document_id": "doc", "decision": "redact",
                             "content_preview": "short preview", "sanitized_content": sanitized})
    allowed = filter_chunks([original], response)
    assert len(allowed) == 1
    assert allowed[0].content == sanitized
    assert allowed[0].metadata == {"source": "kb"}
    assert allowed[0] is not original
    assert original.content == "private original"


@pytest.mark.parametrize("replacement", [None, 123, {"text": "safe"}])
def test_redacted_chunk_without_valid_replacement_is_not_returned(replacement):
    original = RetrievedChunk("a", "doc", "private original")
    response = response_for({"chunk_id": "a", "decision": "redact", "sanitized_content": replacement})
    assert filter_chunks([original], response) == []
    assert allowed_chunk_ids(response) == set()


@pytest.mark.parametrize("decision", ["block", "deny", "sandbox", "approval_required", "unexpected"])
def test_request_level_block_cannot_return_an_individually_allowed_chunk(decision):
    response = response_for({"chunk_id": "a", "decision": "allow"}, decision=decision)
    assert filter_chunks([RetrievedChunk("a", "doc", "private")], response) == []


def test_rejected_quarantined_and_ambiguous_chunks_are_excluded():
    originals = [RetrievedChunk(key, "doc", key) for key in ["safe", "blocked", "quarantined", "duplicate", "duplicate", "wrongdoc"]]
    response = response_for(
        {"chunk_id": "safe", "decision": "allow"},
        {"chunk_id": "blocked", "decision": "reject"},
        {"chunk_id": "quarantined", "decision": "quarantine"},
        {"chunk_id": "duplicate", "decision": "allow"},
        {"chunk_id": "wrongdoc", "document_id": "another-doc", "decision": "allow"},
        decision="allow",
    )
    assert filter_chunks(iter(originals), response) == originals[:1]
    response["result"]["trace"]["retrieved_chunks"].append({"chunk_id": "safe", "decision": "reject"})
    assert filter_chunks(originals, response) == []


@pytest.mark.parametrize("sanitized", ["", "would be redacted in enforce mode"])
def test_allowed_chunk_preserves_original_content_including_shadow_mode(sanitized):
    original = RetrievedChunk("a", "doc", "safe allowed content")
    response = response_for({"chunk_id": "a", "decision": "allow", "sanitized_content": sanitized}, decision="allow")
    assert filter_chunks([original], response) == [original]
    assert filter_chunks([original], response)[0] is original


@dataclass(frozen=True)
class FrozenDoc:
    page_content: str
    metadata: dict


@pytest.mark.parametrize("kind", ["plain", "frozen", "pydantic", "text", "nested"])
def test_guard_retriever_propagates_redaction_without_mutating_documents(shield_factory, kind):
    if kind == "pydantic":
        from pydantic import BaseModel, ConfigDict

        class Document(BaseModel):
            model_config = ConfigDict(frozen=True)
            page_content: str
            metadata: dict

        doc = Document(page_content="private original", metadata={"chunk_id": "a"})
        attr = "page_content"
    elif kind == "nested":
        from types import SimpleNamespace

        class Document:
            def __init__(self):
                self.resource = SimpleNamespace(text="private original")
                self.metadata = {"chunk_id": "a"}

            @property
            def text(self):
                return self.resource.text

            @text.setter
            def text(self, value):
                self.resource.text = value

        doc = Document()
        attr = "text"
    elif kind == "frozen":
        doc = FrozenDoc("private original", {"chunk_id": "a"})
        attr = "page_content"
    else:
        from types import SimpleNamespace

        attr = "text" if kind == "text" else "page_content"
        doc = SimpleNamespace(**{attr: "private original", "metadata": {"chunk_id": "a"}})

    class Retriever:
        def invoke(self, query):
            return [doc]

    def handler(request):
        return httpx.Response(200, json=response_for({
            "chunk_id": "a", "decision": "redact", "sanitized_content": "safe replacement",
        }))

    shield = shield_factory(handler)
    guarded = shield.rag.guard_retriever(Retriever())
    for _ in range(2):
        output = guarded.invoke("query")
        assert len(output) == 1
        assert getattr(output[0], attr) == "safe replacement"
        assert output[0].metadata == doc.metadata
        assert output[0] is not doc
        assert getattr(doc, attr) == "private original"
