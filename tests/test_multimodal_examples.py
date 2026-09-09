"""Offline examples contracts against the installed native OpenAI HTTP transport.

Provider identity coverage does not imply that every provider supports every
operation. All requests here use MockTransport; no provider inference is run.
"""
from __future__ import annotations

import base64
import copy
from email import policy
from email.parser import BytesParser
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import sys
import wave
import zlib

import pytest

from .native_sdk_helpers import openai_backend


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "multimodal" / "run.py"
EXPECTED_PROVIDERS = {
    "anthropic", "azure", "bedrock", "bedrock-mantle", "cerebras", "cohere",
    "deepseek", "elevenlabs", "gemini", "groq", "mistral", "ollama", "openai",
    "parasail", "perplexity", "sgl", "vertex", "openrouter", "huggingface",
    "nebius", "xai", "replicate", "vllm", "sarvam", "wafer", "fireworks",
    "opencode-go", "opencode-zen", "runway",
}


def _load_example(name="_offline_multimodal_example"):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_example()


@pytest.fixture
def native_client():
    sdk, http = openai_backend()
    clients = []

    def make(handler):
        client = sdk.OpenAI(
            api_key="sk-offline-example-fixture", base_url="https://gateway.invalid/openai",
            max_retries=0, http_client=http.Client(transport=http.MockTransport(handler)),
        )
        clients.append(client)
        return client, http

    yield make
    for client in clients:
        client.close()


def _png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 96, 96, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\0\0\xff" * 96) * 96)) + chunk(b"IEND", b""))


def _pdf():
    stream = b"0 0 1 rg 10 10 80 80 re f\n"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream"]
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(b"xref\n0 5\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


def _wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    return output.getvalue()


def _chat(text="Blue", finish="stop"):
    return {"id": "chat-fixture", "object": "chat.completion", "created": 0, "model": "fixture/model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def _response(text="Blue", status="completed"):
    return {"id": "resp-fixture", "object": "response", "created_at": 0, "model": "fixture/model", "status": status,
            "output": [{"id": "msg-fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}]}


def _sse(http, events, *, done=False):
    wire = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    if done:
        wire += "data: [DONE]\n\n"
    return http.Response(200, headers={"content-type": "text/event-stream"}, content=wire.encode())


def test_import_is_inert_without_credentials_or_client_construction(monkeypatch, capsys):
    import deepintshield
    sdk, _ = openai_backend()
    monkeypatch.delenv("DEEPINTSHIELD_VIRTUAL_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(deepintshield.DeepintShield, "from_env", lambda *a, **k: pytest.fail("import opened gateway connection"))
    monkeypatch.setattr(sdk, "OpenAI", lambda *a, **k: pytest.fail("import constructed native client"))
    imported = _load_example("_offline_multimodal_inert_import")
    assert callable(imported.run_example)
    assert capsys.readouterr().out == ""


def test_provider_registry_is_complete_without_universal_modality_claim(example):
    assert isinstance(example.PROVIDERS, tuple)
    assert len(example.PROVIDERS) == 29 and set(example.PROVIDERS) == EXPECTED_PROVIDERS
    assert set(example.OPERATIONS) == {"text", "vision", "pdf", "image", "speech", "transcription", "video", "file"}
    assert set(example.PROVIDER_OPERATIONS) == EXPECTED_PROVIDERS
    for operations in example.PROVIDER_OPERATIONS.values():
        assert set(operations) <= set(example.OPERATIONS)
    assert "text" not in example.PROVIDER_OPERATIONS["elevenlabs"]
    assert "text" not in example.PROVIDER_OPERATIONS["runway"]


@pytest.mark.parametrize("provider", sorted(EXPECTED_PROVIDERS - {"elevenlabs", "runway"}))
@pytest.mark.parametrize("prefixed", [False, True])
def test_text_preserves_every_chat_provider_and_nested_model_id(example, native_client, provider, prefixed):
    seen = []
    client, http = native_client(lambda request: seen.append(request) or http.Response(200, json=_chat()))
    model = "release/deployment/model-v99"
    passed = provider + "/" + model if prefixed else model
    result = example.run_example(provider, "text", passed, prompt="A simple greeting", client=client)
    assert result["text"] == "Blue" and not client.is_closed()
    assert len(seen) == 1 and seen[0].url.path == "/openai/chat/completions"
    body = json.loads(seen[0].content)
    assert body["model"] == provider + "/" + model
    assert body["messages"][-1]["content"] == "A simple greeting"
    for name in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "max_output_tokens",
                 "reasoning", "reasoning_effort", "tools"):
        assert name not in body


@pytest.mark.parametrize("provider,operation", [(provider, "vision") for provider in sorted(EXPECTED_PROVIDERS - {"elevenlabs", "runway"})]
                         + [(provider, "pdf") for provider in ("openai", "azure", "anthropic", "bedrock", "gemini", "vertex")])
@pytest.mark.parametrize("stream", [False, True])
def test_image_and_pdf_use_native_parts_and_complete_streams(example, native_client, tmp_path, provider, operation, stream):
    seen = []
    payload = _png() if operation == "vision" else _pdf()
    source = tmp_path / ("local.png" if operation == "vision" else "local.pdf")
    source.write_bytes(payload)
    def handler(request):
        seen.append(request)
        if not stream:
            return http.Response(200, json=_chat() if operation == "vision" else _response())
        if operation == "vision":
            return _sse(http, [
                {"id": "chat-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
                 "choices": [{"index": 0, "delta": {"content": "Blue"}, "finish_reason": None}]},
                {"id": "chat-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ], done=True)
        return _sse(http, [{"type": "response.output_text.delta", "item_id": "msg-fixture", "output_index": 0,
                            "content_index": 0, "delta": "Blue", "sequence_number": 0},
                           {"type": "response.completed", "sequence_number": 1, "response": _response()}])
    client, http = native_client(handler)
    result = example.run_example(provider, operation, "new-release/vision-deployment", input_path=source,
                                 prompt="Describe the attached content.", stream=stream, client=client)
    assert result["text"] == "Blue" and not client.is_closed()
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["model"] == provider + "/new-release/vision-deployment" and body.get("stream", False) is stream
    assert "temperature" not in body and "reasoning_effort" not in body and "max_tokens" not in body
    if operation == "vision":
        assert seen[0].url.path == "/openai/chat/completions"
        parts = body["messages"][-1]["content"]
        image = next(part for part in parts if part["type"] == "image_url")
        data = image["image_url"]["url"]
        assert data.startswith("data:image/png;base64,")
    else:
        assert seen[0].url.path == "/openai/responses"
        parts = body["input"][-1]["content"]
        file = next(part for part in parts if part["type"] == "input_file")
        assert file["filename"] == source.name and "file_type" not in file
        data = file["file_data"]
        assert data.startswith("data:application/pdf;base64,")
    assert base64.b64decode(data.split(",", 1)[1], validate=True) == payload
    assert source.read_bytes() == payload


def _multipart(request):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["content-type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content)
    assert message.is_multipart()
    return {part.get_param("name", header="content-disposition"): (part.get_filename(), part.get_payload(decode=True))
            for part in message.iter_parts()}


@pytest.mark.parametrize("operation", ["text", "vision", "pdf"])
@pytest.mark.parametrize("fault", ["empty", "incomplete"])
def test_textual_json_outputs_require_nonempty_completed_content(example, native_client, operation, fault):
    payload = _response("" if fault == "empty" else "Blue", "incomplete" if fault == "incomplete" else "completed") if operation == "pdf" else _chat("" if fault == "empty" else "Blue", "length" if fault == "incomplete" else "stop")
    client, http = native_client(lambda _: http.Response(200, json=payload))
    with pytest.raises(RuntimeError):
        example.run_example("openai", operation, "gpt-5.5", client=client)
    assert not client.is_closed()


@pytest.mark.parametrize("operation", ["vision", "pdf"])
@pytest.mark.parametrize("fault", ["truncated", "empty", "error"])
def test_streams_reject_partial_empty_or_error_outputs(example, native_client, operation, fault):
    if operation == "pdf":
        events = [{"type": "response.output_text.delta", "item_id": "msg-fixture", "output_index": 0,
                   "content_index": 0, "delta": "Blue", "sequence_number": 0}]
        if fault == "empty":
            events = [{"type": "response.completed", "sequence_number": 0, "response": _response("")}]
        elif fault == "error":
            events.append({"type": "response.failed", "sequence_number": 1,
                           "response": dict(_response(status="failed"), error={"code": "server_error", "message": "fixture failure"})})
    else:
        events = [{"id": "chat-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
                   "choices": [{"index": 0, "delta": {"content": "" if fault == "empty" else "Blue"},
                                "finish_reason": "stop" if fault == "empty" else None}]}]
        if fault == "error":
            events.append({"error": {"type": "server_error", "message": "fixture failure"}})
    client, http = native_client(lambda _: _sse(http, events, done=fault == "empty" and operation != "pdf"))
    sdk, _ = openai_backend()
    with pytest.raises((RuntimeError, sdk.APIError)):
        example.run_example("openai", operation, "gpt-5.5", stream=True, client=client)
    assert not client.is_closed()


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("case,expected", [
    ("delta_only_terminal", None), ("full_snapshot", "Blue"), ("final_priority", "Blue"),
    ("whitespace_snapshot", None), ("empty", None), ("empty_deltas", None),
    ("whitespace_deltas", None), ("failed_event", None), ("incomplete_event", None),
    ("completed_incomplete_status", None), ("completed_error", None), ("truncated", None),
    ("late_error", None), ("wrong_final_priority", None), ("none_terminal_output", None),
    ("refusal_snapshot", None), ("refusal_delta", None), ("reasoning_only", None), ("invalid_delta", None),
])
def test_native_pdf_stream_validates_complete_final_snapshot(example, native_client, tmp_path, provider, case, expected):
    """Successful completion includes an authoritative final output snapshot."""
    deltas = [] if case == "empty" else ["", ""] if case == "empty_deltas" else ["  "] if case == "whitespace_deltas" else ["Red"] if case == "final_priority" else ["Bl", "ue"]
    start = {**_response(status="in_progress"), "output": []}
    events = [{"type": "response.created", "sequence_number": 0, "response": start}]
    events.extend({"type": "response.output_text.delta", "item_id": "msg-fixture", "output_index": 0,
                   "content_index": 0, "delta": delta, "sequence_number": index}
                  for index, delta in enumerate(deltas, 1))
    terminal = {**_response(), "output": []}
    terminal_kind = "response.completed"
    if case == "none_terminal_output":
        terminal["output"] = None
    elif case == "refusal_snapshot":
        terminal["output"] = [{"id": "msg-fixture", "type": "message", "role": "assistant", "status": "completed",
                               "content": [{"type": "refusal", "refusal": "Synthetic refusal"}]}]
    elif case == "refusal_delta":
        events.append({"type": "response.refusal.delta", "sequence_number": len(events), "output_index": 0,
                       "content_index": 0, "item_id": "msg-fixture", "delta": "Synthetic refusal"})
    elif case == "reasoning_only":
        events = [events[0], {"type": "response.reasoning_text.delta", "sequence_number": 1, "output_index": 0,
                              "content_index": 0, "item_id": "reasoning-fixture", "delta": "Blue"}]
    elif case == "invalid_delta":
        events[1]["delta"] = 42
    elif case in ("full_snapshot", "final_priority"):
        terminal = _response()
    elif case == "whitespace_snapshot":
        terminal = _response("  ")
    elif case == "wrong_final_priority":
        terminal = _response("Red")
    elif case in ("failed_event", "incomplete_event", "completed_incomplete_status"):
        terminal["status"] = "failed" if case == "failed_event" else "incomplete"
        if case != "completed_incomplete_status":
            terminal_kind = "response.failed" if case == "failed_event" else "response.incomplete"
    elif case == "completed_error":
        terminal["error"] = {"code": "server_error", "message": "synthetic terminal failure"}
    if case != "truncated":
        events.append({"type": terminal_kind, "sequence_number": len(events), "response": terminal})
    if case == "late_error":
        events.append({"type": "error", "sequence_number": len(events), "code": "server_error",
                       "message": "synthetic failure after completion"})
    seen = []
    client, http = native_client(lambda request: seen.append(request) or _sse(http, events))
    output = tmp_path / "answer.txt"
    if expected is None:
        with pytest.raises(RuntimeError):
            example.run_example(provider, "pdf", "opaque/pdf-model", stream=True, output_path=output, client=client)
        assert not output.exists()
    else:
        result = example.run_example(provider, "pdf", "opaque/pdf-model", stream=True, output_path=output, client=client)
        assert result["text"] == expected and output.read_text() == expected
    assert len(seen) == 1 and seen[0].url.path == "/openai/responses" and not client.is_closed()
    body = json.loads(seen[0].content)
    assert body["stream"] is True and body["model"] == provider + "/opaque/pdf-model"


@pytest.mark.parametrize("output", [None, []])
def test_nonstream_empty_response_snapshot_is_a_controlled_failure(example, native_client, tmp_path, output):
    client, http = native_client(lambda _: http.Response(200, json={**_response(), "output": output}))
    target = tmp_path / "must-not-exist.txt"
    with pytest.raises(RuntimeError, match="text|empty|output"):
        example.run_example("anthropic", "pdf", "opaque/pdf-model", output_path=target, client=client)
    assert not target.exists() and not client.is_closed()


@pytest.mark.parametrize("operation", ["text", "vision"])
@pytest.mark.parametrize("mode", ["json", "same_chunk", "later_chunk"])
def test_chat_refusal_cannot_be_hidden_by_completed_text(example, native_client, tmp_path, operation, mode):
    seen, response_streams = [], []
    def handler(request):
        seen.append(request)
        if mode == "json":
            payload = _chat()
            payload["choices"][0]["message"]["refusal"] = "Synthetic refusal"
            return http.Response(200, json=payload)
        deltas = ([{"content": "Blue", "refusal": "Synthetic refusal"}] if mode == "same_chunk"
                  else [{"content": "Blue"}, {"refusal": "Synthetic refusal"}])
        frames = ["data: " + json.dumps({"id": "chat-refusal", "object": "chat.completion.chunk", "created": 0,
                   "model": "fixture", "choices": [{"index": 0, "delta": delta,
                   "finish_reason": "stop" if index == len(deltas) - 1 else None}]}) + "\n\n"
                  for index, delta in enumerate(deltas)] + ["data: [DONE]\n\n"]
        class TrackedStream(http.SyncByteStream):
            closed = False
            def __iter__(self):
                yield from (frame.encode() for frame in frames)
            def close(self):
                self.closed = True
        response_stream = TrackedStream()
        response_streams.append(response_stream)
        return http.Response(200, headers={"content-type": "text/event-stream"}, stream=response_stream)
    client, http = native_client(handler)
    output = tmp_path / "must-not-exist.txt"
    with pytest.raises(RuntimeError, match="refus"):
        example.run_example("openai", operation, "gpt-5.5", stream=mode != "json", output_path=output, client=client)
    assert len(seen) == 1 and not output.exists() and not client.is_closed()
    if mode != "json":
        assert len(response_streams) == 1 and response_streams[0].closed


@pytest.mark.parametrize("provider,operation", [("openai", "image"), ("openai", "speech"), ("elevenlabs", "speech")])
def test_generated_artifacts_validate_and_save_native_bytes(example, native_client, tmp_path, provider, operation):
    seen = []
    data = _png() if operation == "image" else _wav()
    output = tmp_path / ("generated.png" if operation == "image" else "generated.wav")
    def handler(request):
        seen.append(request)
        if operation == "image":
            return http.Response(200, json={"created": 0, "data": [{"b64_json": base64.b64encode(data).decode()}]})
        return http.Response(200, content=data, headers={"content-type": "audio/wav"})
    client, http = native_client(handler)
    result = example.run_example(provider, operation, "opaque/new-model", prompt="A blue square" if operation == "image" else "Hello",
                                 output_path=output, voice="fixture-voice" if operation == "speech" else None, client=client)
    assert result["bytes"] == len(data) and result["sha256"] == hashlib.sha256(data).hexdigest()
    assert output.read_bytes() == data and not client.is_closed()
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["model"] == provider + "/opaque/new-model"
    assert seen[0].url.path == ("/openai/images/generations" if operation == "image" else "/openai/audio/speech")
    if operation == "speech":
        assert body["voice"] == "fixture-voice" and body["input"] == "Hello"
    else:
        assert body["prompt"] == "A blue square"
        assert body.get("n", 1) == 1
    for parameter in ("temperature", "top_p", "quality", "size", "style", "speed"):
        assert parameter not in body


@pytest.mark.parametrize("payload", [{"created": 0, "data": []}, {"created": 0, "data": [{"b64_json": "!!bad-base64!!"}]},
                                     {"created": 0, "data": [{"b64_json": base64.b64encode(b"not actual image bytes").decode()}]}])
def test_invalid_image_output_never_creates_artifact(example, native_client, tmp_path, payload):
    client, http = native_client(lambda _: http.Response(200, json=payload))
    output = tmp_path / "must-not-exist.png"
    with pytest.raises((RuntimeError, ValueError)):
        example.run_example("openai", "image", "new-image-model", output_path=output, client=client)
    assert not output.exists()


def test_image_url_is_reported_without_untrusted_download(example, native_client, tmp_path):
    seen = []
    payload = {"created": 0, "data": [{"url": "https://provider.invalid/result.png"}]}
    client, http = native_client(lambda request: seen.append(request) or http.Response(200, json=payload))
    result = example.run_example("openai", "image", "new-image-model", client=client)
    assert result["url"] == "https://provider.invalid/result.png" and len(seen) == 1
    output = tmp_path / "must-not-exist.png"
    with pytest.raises((RuntimeError, ValueError)):
        example.run_example("openai", "image", "new-image-model", output_path=output, client=client)
    assert len(seen) == 2 and not output.exists()


def test_empty_speech_response_is_not_written(example, native_client, tmp_path):
    client, http = native_client(lambda _: http.Response(200, content=b"", headers={"content-type": "audio/mpeg"}))
    output = tmp_path / "must-not-exist.mp3"
    with pytest.raises(RuntimeError):
        example.run_example("openai", "speech", "tts-1", voice="alloy", output_path=output, client=client)
    assert not output.exists()


@pytest.mark.parametrize("operation", ["speech", "video"])
def test_http_200_json_error_is_not_a_binary_media_success(example, native_client, tmp_path, operation):
    def handler(request):
        if operation == "video" and request.method == "POST":
            return http.Response(200, json={"id": "job-fixture:openai", "object": "video", "created_at": 0,
                                           "status": "completed", "model": "openai/new-model", "progress": 100,
                                           "seconds": "4", "size": "1280x720"})
        return http.Response(200, json={"error": {"message": "upstream fixture failure"}})
    client, http = native_client(handler)
    output = tmp_path / "must-not-exist.bin"
    with pytest.raises(RuntimeError):
        example.run_example("openai", operation, "new-model", voice="alloy" if operation == "speech" else None,
                            output_path=output, client=client)
    assert not output.exists()


def test_transcription_preserves_real_audio_multipart_and_closes_input_file(example, native_client, tmp_path, monkeypatch):
    source = tmp_path / "voice.wav"
    source.write_bytes(_wav())
    seen = []
    client, http = native_client(lambda request: seen.append(request) or http.Response(200, json={"text": "Hello from audio"}))
    handles = []
    original_open = Path.open
    def track_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == source:
            handles.append(handle)
        return handle
    monkeypatch.setattr(Path, "open", track_open)
    result = example.run_example("openai", "transcription", "whisper-1", input_path=source, client=client)
    assert handles and all(handle.closed for handle in handles)
    assert result["text"] == "Hello from audio" and not client.is_closed()
    assert len(seen) == 1 and seen[0].url.path == "/openai/audio/transcriptions"
    parts = _multipart(seen[0])
    assert parts["model"][1] == b"openai/whisper-1"
    assert parts["file"] == ("voice.wav", source.read_bytes())
    assert "temperature" not in parts and "language" not in parts and "prompt" not in parts


@pytest.mark.parametrize("failure", [None, "inference", "cleanup", "both"])
@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_uploaded_pdf_uses_file_id_and_deletes_remote_file_even_on_error(example, native_client, tmp_path, provider, failure):
    source = tmp_path / "fixture.pdf"
    source.write_bytes(_pdf())
    seen = []
    def handler(request):
        seen.append(request)
        if request.method == "POST" and request.url.path == "/openai/files":
            return http.Response(200, json={"id": "file-fixture", "object": "file", "bytes": source.stat().st_size,
                                            "created_at": 0, "filename": source.name, "purpose": "user_data", "status": "processed"})
        if request.url.path == "/openai/responses":
            if failure in ("inference", "both"):
                return http.Response(403, json={"error": {"message": "fixture policy block", "type": "permission_error"}})
            return http.Response(200, json=_response())
        assert request.method == "DELETE" and request.url.path == "/openai/files/file-fixture"
        if failure == "both":
            return http.Response(500, json={"error": {"message": "fixture cleanup failure", "type": "server_error"}})
        return http.Response(200, json={"id": "file-fixture", "object": "file", "deleted": failure != "cleanup"})
    client, http = native_client(handler)
    sdk, _ = openai_backend()
    if failure:
        error_type = sdk.PermissionDeniedError if failure in ("inference", "both") else RuntimeError
        with pytest.raises(error_type):
            example.run_example(provider, "file", "gpt-5.5", input_path=source, client=client)
    else:
        result = example.run_example(provider, "file", "gpt-5.5", input_path=source, client=client)
        assert result["text"] == "Blue" and result["file_id"] == "file-fixture" and result["deleted"] is True
    assert not client.is_closed()
    assert len(seen) == 3 and seen[-1].method == "DELETE"
    uploaded = _multipart(seen[0])
    assert uploaded["file"] == (source.name, source.read_bytes())
    assert uploaded["purpose"][1] == (b"assistants" if provider == "azure" else b"user_data")
    assert uploaded["provider"][1] == provider.encode()
    body = json.loads(seen[1].content)
    assert body["model"] == provider + "/gpt-5.5"
    parts = body["input"][-1]["content"]
    assert {"type": "input_file", "file_id": "file-fixture"} in parts
    assert seen[-1].url.params["provider"] == provider


@pytest.mark.parametrize("provider,opaque_id", [("openai", "job-with:opaque:id"), ("runway", "job-with:opaque:id"), ("runway", "org/task:42")])
def test_video_uses_bounded_native_job_lifecycle_and_saves_download(example, native_client, tmp_path, provider, opaque_id):
    seen = []
    artifact = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    job_id = opaque_id + ":" + provider
    def handler(request):
        seen.append(request)
        if request.method == "POST":
            return http.Response(200, json={"id": job_id, "object": "video", "created_at": 0, "status": "queued",
                                           "model": provider + "/opaque/new-model", "progress": 0, "seconds": "4", "size": "1280x720"})
        if request.url.path.endswith("/content"):
            return http.Response(200, content=artifact, headers={"content-type": "video/mp4"})
        return http.Response(200, json={"id": job_id, "object": "video", "created_at": 0, "status": "completed",
                                       "model": provider + "/opaque/new-model", "progress": 100, "seconds": "4", "size": "1280x720"})
    client, http = native_client(handler)
    output = tmp_path / "generated.mp4"
    reference = None
    parameters = None
    if "/" in opaque_id:
        reference = tmp_path / "reference.png"
        reference.write_bytes(_png())
        parameters = {"seconds": "4", "size": "1280x720", "extra_body": {"seed": 42}}
    result = example.run_example(provider, "video", "opaque/new-model", prompt="Slow moving ocean waves",
                                 output_path=output, input_path=reference, parameters=parameters, client=client)
    assert result["bytes"] == len(artifact) and result["sha256"] == hashlib.sha256(artifact).hexdigest()
    assert output.read_bytes() == artifact and not client.is_closed()
    assert [(request.method, request.url.path) for request in seen] == [
        ("POST", "/openai/videos"), ("GET", "/openai/videos/" + job_id),
        ("GET", "/openai/videos/" + job_id + "/content"),
    ]
    if "/" in opaque_id:
        assert b"org%2Ftask:42:runway" in seen[1].url.raw_path
        assert b"%252F" not in seen[1].url.raw_path
    if seen[0].headers["content-type"].startswith("multipart/"):
        parts = _multipart(seen[0])
        assert parts["model"][1] == (provider + "/opaque/new-model").encode()
        assert parts["prompt"][1] == b"Slow moving ocean waves"
        if reference is not None:
            assert parts["input_reference"] == (reference.name, reference.read_bytes())
            assert parts["seconds"][1] == b"4" and parts["size"][1] == b"1280x720" and parts["seed"][1] == b"42"
    else:
        body = json.loads(seen[0].content)
        assert body["model"] == provider + "/opaque/new-model"
        assert body["prompt"] == "Slow moving ocean waves"
    assert all(not {"seconds", "size", "seed"}.intersection(request.url.params) for request in seen[1:])


def test_failed_video_job_does_not_download_or_write_output(example, native_client, tmp_path):
    seen = []
    payload = {"id": "job-failed:runway", "object": "video", "created_at": 0, "status": "failed", "model": "runway/gen4.5",
               "progress": 0, "seconds": "4", "size": "1280x720", "error": {"code": "generation_failed", "message": "fixture failure"}}
    client, http = native_client(lambda request: seen.append(request) or http.Response(200, json=payload))
    output = tmp_path / "must-not-exist.mp4"
    with pytest.raises(RuntimeError):
        example.run_example("runway", "video", "gen4.5", output_path=output, client=client)
    assert not output.exists() and len(seen) == 1


def test_pending_video_has_a_finite_poll_limit_without_download(example, native_client, monkeypatch):
    seen = []
    payload = {"id": "job-pending:runway", "object": "video", "created_at": 0, "status": "queued", "model": "runway/gen4.5",
               "progress": 0, "seconds": "4", "size": "1280x720"}
    def handler(request):
        seen.append(request)
        assert len(seen) <= 100, "example performed unbounded video polling"
        assert not request.url.path.endswith("/content")
        return http.Response(200, json=payload)
    client, http = native_client(handler)
    monkeypatch.setattr(example.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="bounded|timed|complete"):
        example.run_example("runway", "video", "gen4.5", client=client)
    assert 1 < len(seen) <= 100 and not client.is_closed()


@pytest.mark.parametrize("operation,stream", [("vision", False), ("vision", True), ("pdf", False), ("pdf", True), ("file", False)])
@pytest.mark.parametrize("answer", ["Blue", "The visible color of the rectangle is blue.", "Red",
                                    "I cannot read the attachment.", "Blue or red.", "Blue and red.", "It might be blue."])
def test_generated_blue_fixture_requires_an_unambiguous_correct_answer(example, native_client, operation, stream, answer):
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path == "/openai/files":
            return http.Response(200, json={"id": "file-fixture", "object": "file", "created_at": 0, "bytes": 100,
                                           "filename": "blue.pdf", "purpose": "user_data"})
        if request.method == "DELETE":
            return http.Response(200, json={"id": "file-fixture", "object": "file", "deleted": True})
        if stream:
            if operation == "vision":
                return _sse(http, [{"id": "chat-fixture", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
                                    "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": "stop"}]}], done=True)
            return _sse(http, [{"type": "response.completed", "sequence_number": 0, "response": _response(answer)}])
        return http.Response(200, json=_chat(answer) if operation == "vision" else _response(answer))
    client, http = native_client(handler)
    if answer in ("Blue", "The visible color of the rectangle is blue."):
        assert example.run_example("openai", operation, "gpt-5.5", stream=stream, client=client)["text"] == answer
    else:
        with pytest.raises(RuntimeError):
            example.run_example("openai", operation, "gpt-5.5", stream=stream, client=client)
    if operation == "file":
        assert len(seen) == 3 and seen[-1].method == "DELETE"


@pytest.mark.parametrize("arguments", [["--list"], ["--provider", "runway", "--operation", "video", "--model", "gen4.5", "--dry-run"]])
def test_list_and_dry_run_do_not_construct_clients_or_write_files(example, monkeypatch, capsys, arguments):
    import deepintshield
    monkeypatch.setattr(deepintshield.DeepintShield, "from_env", lambda *a, **k: pytest.fail("read-only CLI opened gateway"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("read-only CLI opened a file"))
    assert example.main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == EXPECTED_PROVIDERS if arguments == ["--list"] else output["dry_run"] is True


@pytest.mark.parametrize("operation", ["vision", "pdf"])
def test_default_native_attachment_is_a_real_blue_fixture(example, native_client, operation):
    seen = []
    client, http = native_client(lambda request: seen.append(request) or http.Response(200, json=_chat() if operation == "vision" else _response()))
    example.run_example("openai", operation, "gpt-5.5", client=client)
    body = json.loads(seen[0].content)
    parts = body["messages" if operation == "vision" else "input"][0]["content"]
    data_url = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url") if operation == "vision" else next(part["file_data"] for part in parts if part["type"] == "input_file")
    data = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if operation == "vision":
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        offset, chunks = 8, {}
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            kind, value = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
            crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
            assert crc == zlib.crc32(kind + value) & 0xffffffff
            chunks[kind] = value
            offset += 12 + length
        assert offset == len(data) and chunks[b"IEND"] == b""
        width, height, depth, color, *_ = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
        assert width >= 96 and height >= 96 and depth == 8 and color == 2
        assert zlib.decompress(chunks[b"IDAT"]) == (b"\0" + b"\0\0\xff" * width) * height
    else:
        assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
        start = int(re.search(rb"startxref\s+(\d+)\s+%%EOF", data).group(1))
        assert data[start:].startswith(b"xref\n")
        entries = data[start:].splitlines()
        count = int(entries[1].split()[1])
        for index, entry in enumerate(entries[3:count + 2], 1):
            offset = int(entry.split()[0])
            assert data[offset:].startswith(f"{index} 0 obj\n".encode())
        image = re.search(rb"/Subtype /Image(?P<dictionary>.*?)stream\n", data, re.S)
        assert image is not None and b"/FlateDecode" in image["dictionary"]
        width = int(re.search(rb"/Width (\d+)", image["dictionary"])[1])
        height = int(re.search(rb"/Height (\d+)", image["dictionary"])[1])
        length = int(re.search(rb"/Length (\d+)", image["dictionary"])[1])
        assert width >= 96 and height >= 96
        assert zlib.decompress(data[image.end():image.end() + length]) == b"\0\0\xff" * width * height


def test_existing_output_is_never_overwritten_or_dispatched(example, tmp_path):
    output = tmp_path / "user-image.png"
    output.write_bytes(b"existing-user-artifact")
    with pytest.raises(ValueError, match="output|overwrite"):
        example.run_example("openai", "image", "new-model", output_path=output, client=_NoClientAccess())
    assert output.read_bytes() == b"existing-user-artifact"


@pytest.mark.parametrize("provider_error", [False, True])
def test_owned_connection_uses_environment_and_closes_both_clients(example, monkeypatch, provider_error):
    from deepintshield import DeepintShield
    sdk, http = openai_backend()
    monkeypatch.setenv("DEEPINTSHIELD_VIRTUAL_KEY", "sk-offline-example-environment")
    monkeypatch.setenv("DEEPINTSHIELD_BASE_URL", "https://gateway.invalid")
    seen, shields, transports = [], [], []
    original = DeepintShield.openai_config
    def options(shield, **kwargs):
        shields.append(shield)
        def handler(request):
            seen.append(request)
            return http.Response(403, json={"error": {"message": "fixture policy block", "type": "permission_error"}}) if provider_error else http.Response(200, json=_chat())
        transport = http.Client(transport=http.MockTransport(handler))
        transports.append(transport)
        return {**original(shield, **kwargs), "http_client": transport, "max_retries": 0}
    monkeypatch.setattr(DeepintShield, "openai_config", options)
    if provider_error:
        with pytest.raises(sdk.PermissionDeniedError):
            example.run_example("openai", "text", "gpt-5.5")
    else:
        assert example.run_example("openai", "text", "gpt-5.5")["text"] == "Blue"
    assert len(seen) == len(shields) == len(transports) == 1
    assert seen[0].url.path == "/openai/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-offline-example-environment"
    assert transports[0].is_closed and shields[0]._client.is_closed


@pytest.mark.parametrize("operation", ["vision", "pdf"])
@pytest.mark.parametrize("stream", [False, True])
def test_inline_guardrail_block_preserves_native_error_without_retry_or_file_changes(example, native_client, tmp_path, operation, stream):
    seen = []
    message = "Attachment redaction requires a sanitized copy; the request was blocked before provider dispatch."
    payload = {"error": {"type": "guardrail_blocked", "code": "guardrail_blocked", "message": message}}
    client, http = native_client(lambda request: seen.append(request) or http.Response(403, json=payload))
    source = tmp_path / ("synthetic.png" if operation == "vision" else "synthetic.pdf")
    fixture = _png() if operation == "vision" else _pdf()
    source.write_bytes(fixture)
    output = tmp_path / "must-not-exist.txt"
    sdk, _ = openai_backend()
    with pytest.raises(sdk.PermissionDeniedError) as caught:
        example.run_example("anthropic", operation, "claude-sonnet-4-5", input_path=source,
                            output_path=output, stream=stream, client=client)
    assert caught.value.status_code == 403 and caught.value.code == "guardrail_blocked"
    assert caught.value.body["message"] == message and caught.value.body["type"] == "guardrail_blocked"
    assert len(seen) == 1 and not client.is_closed()
    assert seen[0].url.path == ("/openai/chat/completions" if operation == "vision" else "/openai/responses")
    assert json.loads(seen[0].content)["stream"] is stream
    assert source.read_bytes() == fixture and set(tmp_path.iterdir()) == {source}


class _NoClientAccess:
    def __getattr__(self, name):
        pytest.fail("invalid example accessed the native client: " + name)


@pytest.mark.parametrize("provider,operation,model,options", [
    ("not-a-provider", "text", "model", {}),
    ("openai", "not-an-operation", "model", {}),
    ("elevenlabs", "text", "model", {}),
    ("runway", "vision", "model", {}),
    ("openai", "text", "", {}),
    ("openai", "text", "   ", {}),
    ("openai", "text", "anthropic/claude-sonnet-4-5", {}),
    ("openai", "speech", "tts-1", {}),
    ("openai", "transcription", "whisper-1", {}),
])
def test_invalid_or_unconfigured_requests_fail_before_native_io(example, provider, operation, model, options):
    with pytest.raises(ValueError):
        example.run_example(provider, operation, model, client=_NoClientAccess(), **options)


@pytest.mark.parametrize("operation", ["image", "speech", "transcription", "video", "file"])
def test_unsupported_stream_operations_fail_before_io(example, operation):
    with pytest.raises(ValueError, match="stream"):
        example.run_example("openai", operation, "model", stream=True, client=_NoClientAccess())


@pytest.mark.parametrize("provider,operation,parameters", [
    ("openai", "text", {"reasoning_effort": "low", "max_completion_tokens": 1024, "extra_body": {"vendor_setting": {"mode": "fixture"}}}),
    ("openai", "pdf", {"reasoning": {"effort": "low"}, "max_output_tokens": 1024}),
    ("openai", "image", {"size": "1024x1024", "n": 2, "extra_body": {"seed": 42}}),
    ("sarvam", "speech", {"extra_body": {"language_code": "hi-IN"}}),
    ("openai", "transcription", {"language": "en", "response_format": "json"}),
])
def test_explicit_native_parameters_are_preserved_without_mutating_caller(example, native_client, tmp_path, provider, operation, parameters):
    seen = []
    original = copy.deepcopy(parameters)
    source = tmp_path / "voice.wav"
    source.write_bytes(_wav())
    def handler(request):
        seen.append(request)
        if operation == "speech":
            return http.Response(200, content=_wav(), headers={"content-type": "audio/wav"})
        payload = {"text": "Hello"} if operation == "transcription" else {"created": 0, "data": [{"b64_json": base64.b64encode(_png()).decode()}]} if operation == "image" else _response() if operation == "pdf" else _chat()
        return http.Response(200, json=payload)
    client, http = native_client(handler)
    result = example.run_example(provider, operation, "new-release/model", parameters=parameters,
                                 input_path=source if operation == "transcription" else None,
                                 voice="fixture-voice" if operation == "speech" else None, client=client)
    assert result["status"] == "completed" and parameters == original
    assert len(seen) == 1
    if operation == "transcription":
        parts = _multipart(seen[0])
        body = {name: value[1].decode() for name, value in parts.items() if name != "file"}
    else:
        body = json.loads(seen[0].content)
    assert body["model"] == provider + "/new-release/model"
    for name, value in original.items():
        if name == "extra_body":
            for nested, expected in value.items():
                assert body[nested] == expected
        else:
            assert body[name] == value


@pytest.mark.parametrize("parameters", [
    {"model": "anthropic/unselected-model"}, {"input": "replaced"}, {"messages": []}, {"stream": True},
    {"prompt": "replaced"}, {"provider": "anthropic"},
    {"extra_body": {"model": "anthropic/unselected-model"}}, {"extra_body": {"input": "replaced"}},
    {"extra_body": {"messages": []}}, {"extra_body": {"stream": True}},
    {"extra_body": {"prompt": "replaced"}}, {"extra_body": {"provider": "anthropic"}},
    {"extra_query": {"provider": "anthropic"}},
])
def test_native_parameters_cannot_replace_selected_provider_model_or_input(example, parameters):
    with pytest.raises(ValueError):
        example.run_example("openai", "text", "gpt-5.5", parameters=parameters, client=_NoClientAccess())


@pytest.mark.parametrize("parameters", [None, {}, {"extra_body": {"language_code": ""}}])
def test_sarvam_speech_requires_explicit_language_before_dispatch(example, parameters):
    with pytest.raises(ValueError, match="language_code"):
        example.run_example("sarvam", "speech", "bulbul:v2", voice="anushka", parameters=parameters, client=_NoClientAccess())


def test_file_parameters_apply_only_to_inference_and_preserve_cleanup(example, native_client):
    seen = []
    def handler(request):
        seen.append(request)
        if request.method == "DELETE":
            return http.Response(200, json={"id": "file-fixture", "object": "file", "deleted": True})
        if request.url.path == "/openai/files":
            return http.Response(200, json={"id": "file-fixture", "object": "file", "created_at": 0, "bytes": 100,
                                           "filename": "blue.pdf", "purpose": "user_data"})
        return http.Response(200, json=_response())
    client, http = native_client(handler)
    parameters = {"max_output_tokens": 1024, "extra_body": {"vendor_setting": {"mode": "fixture"}}}
    result = example.run_example("openai", "file", "gpt-5.5", parameters=parameters, client=client)
    assert result["deleted"] is True and len(seen) == 3
    upload = _multipart(seen[0])
    assert "max_output_tokens" not in upload and "vendor_setting" not in upload
    inference = json.loads(seen[1].content)
    assert inference["max_output_tokens"] == 1024 and inference["vendor_setting"] == {"mode": "fixture"}
    assert dict(seen[2].url.params) == {"provider": "openai"}
