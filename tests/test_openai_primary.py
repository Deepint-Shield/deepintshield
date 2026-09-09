"""Native OpenAI inference contracts across installed SDK HTTP generations."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from .native_sdk_helpers import openai_backend
from .test_model_compatibility import MODELS, native_response


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_native_async_openai_keeps_provider_models_and_parameters(shield_factory, model, api, stream):
    _, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response(api, model, stream, http)

    shield = shield_factory(lambda _: pytest.fail("connection must not perform discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        async with shield.async_openai(http_client=transport, max_retries=0) as client:
            if api == "chat":
                response = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}], stream=stream)
            else:
                response = await client.responses.create(model=model, input="Hi", stream=stream)
            if stream:
                assert [chunk async for chunk in response]
                await response.close()
            else:
                assert response.model == model
    assert len(seen) == 1
    request = seen[0]
    assert request.url.path == ("/openai/chat/completions" if api == "chat" else "/openai/responses")
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    body = json.loads(request.content)
    assert body["model"] == model
    for key in ("temperature", "top_p", "max_tokens", "reasoning", "reasoning_effort", "tools", "tool_choice"):
        assert key not in body


@pytest.mark.asyncio
async def test_openai_defaults_use_installed_transport_and_close(shield_factory):
    sdk, http = openai_backend()
    shield = shield_factory(lambda _: pytest.fail("unexpected I/O"))
    with shield.openai() as sync:
        assert isinstance(sync, sdk.OpenAI)
        assert isinstance(sync._client, http.Client)
        assert sync._client.follow_redirects is False
        assert len(sync._client.event_hooks["request"]) == 2
    assert sync.is_closed()
    async with shield.providers.async_openai() as native:
        assert isinstance(native, sdk.AsyncOpenAI)
        assert isinstance(native._client, http.AsyncClient)
        assert native._client.follow_redirects is False
        assert len(native._client.event_hooks["request"]) == 2
    assert native.is_closed()


def test_openai_config_uses_only_existing_connection_environment(monkeypatch):
    from deepintshield import DeepintShield
    monkeypatch.setenv("DEEPINTSHIELD_VIRTUAL_KEY", "sk-ds-env")
    monkeypatch.setenv("DEEPINTSHIELD_BASE_URL", "https://gateway.invalid/prefix/")
    with DeepintShield.from_env() as shield:
        custom_headers = {"X-DeepIntShield-Agent": "reviewer"}
        options = shield.openai_config(default_headers=custom_headers, max_retries=4)
        assert options["base_url"] == "https://gateway.invalid/prefix/openai"
        assert options["api_key"] == "sk-ds-env"
        assert options["max_retries"] == 4
        assert options["default_headers"]["X-DeepIntShield-Agent"] == "reviewer"
        assert custom_headers == {"X-DeepIntShield-Agent": "reviewer"}
        options["default_headers"]["x-custom"] = "one client"
        assert "x-custom" not in shield.openai_config()["default_headers"]


@pytest.mark.parametrize("operation", ["embedding", "speech", "transcription", "image", "video"])
def test_native_openai_non_chat_operations_use_the_same_gateway(shield_factory, operation):
    _, http = openai_backend()
    seen = []
    payloads = {
        "embedding": {"object": "list", "data": [{"index": 0, "object": "embedding", "embedding": [0.1, 0.2]}], "model": "cohere/embed-v4.0", "usage": {"prompt_tokens": 1, "total_tokens": 1}},
        "transcription": {"text": "Hello"},
        "image": {"created": 0, "data": [{"b64_json": "aW1hZ2U="}]},
        "video": {"id": "video-test", "object": "video", "created_at": 0, "status": "queued", "model": "runway/gen4.5", "progress": 0, "seconds": "4", "size": "1280x720"},
    }

    def handler(request):
        seen.append(request)
        if operation == "speech":
            return http.Response(200, content=b"audio-bytes", headers={"content-type": "audio/mpeg"})
        return http.Response(200, json=payloads[operation])

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    with http.Client(transport=http.MockTransport(handler)) as transport:
        with shield.openai(http_client=transport, max_retries=0) as client:
            if operation == "embedding":
                assert client.embeddings.create(model="cohere/embed-v4.0", input=["Hello"], encoding_format="float").data[0].embedding == [0.1, 0.2]
            elif operation == "speech":
                assert client.audio.speech.create(model="elevenlabs/eleven_multilingual_v2", input="Hello", voice="voice-id").content == b"audio-bytes"
            elif operation == "transcription":
                assert client.audio.transcriptions.create(model="openai/whisper-1", file=("audio.wav", b"audio-bytes", "audio/wav")).text == "Hello"
            elif operation == "image":
                assert client.images.generate(model="openai/gpt-image-1", prompt="A tree").data[0].b64_json == "aW1hZ2U="
            else:
                if not hasattr(client, "videos"):
                    pytest.skip("installed OpenAI version predates the native videos resource")
                assert client.videos.create(model="runway/gen4.5", prompt="Ocean waves").id == "video-test"
    assert len(seen) == 1
    suffix = {"embedding": "embeddings", "speech": "audio/speech", "transcription": "audio/transcriptions", "image": "images/generations", "video": "videos"}[operation]
    assert str(seen[0].url) == f"https://gateway.invalid/openai/{suffix}"
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"


@pytest.mark.asyncio
async def test_invalid_async_transport_remains_unchanged(shield_factory):
    _, http = openai_backend()
    shield = shield_factory(lambda _: pytest.fail("unexpected request"))
    with http.Client() as transport:
        hooks = deepcopy(transport.event_hooks)
        with pytest.raises(TypeError, match="http_client"):
            shield.async_openai(http_client=transport)
        assert transport.event_hooks == hooks
        assert not transport.is_closed


@pytest.mark.asyncio
async def test_native_aiohttp_transport_is_accepted_and_sends_gateway_headers(shield_factory):
    """Use an actual local HTTP server: aiohttp has its own request path."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    sdk, _ = openai_backend()
    pytest.importorskip("aiohttp")
    try:
        transport = sdk.DefaultAioHttpClient()
    except (RuntimeError, ImportError):
        pytest.skip("installed OpenAI aiohttp optional transport dependencies are missing")
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps({"id": "chat-1", "object": "chat.completion", "created": 0,
                               "model": "anthropic/opaque-model", "choices": [{"index": 0,
                               "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url=f"http://127.0.0.1:{server.server_port}")
        async with shield.async_openai(http_client=transport, max_retries=0) as client:
            result = await client.chat.completions.create(model="anthropic/opaque-model", messages=[{"role": "user", "content": "Hi"}])
            assert result.choices[0].message.content == "Hello"
        assert transport.is_closed
        assert len(seen) == 1
        path, headers, payload = seen[0]
        assert path == "/openai/chat/completions"
        assert {k.lower(): v for k, v in headers.items()}["x-deepintshield-vk"] == "sk-ds-test"
        assert payload["model"] == "anthropic/opaque-model"
    finally:
        await transport.aclose()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
