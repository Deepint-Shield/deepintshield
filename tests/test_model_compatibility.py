"""Offline request contracts for model IDs accepted by the gateway.

These check SDK serialization and streaming against a mock transport. Provider
availability, access permissions, and inference support are server concerns.
"""

from __future__ import annotations

import json

import httpx
import pytest

from deepintshield._prompt_cache import PROVIDER_OPENAI, build_request_hook
from .native_sdk_helpers import openai_backend


# New models and provider/deployment prefixes must stay opaque to the SDK.
CHAT_PROVIDERS = (
    "anthropic", "azure", "bedrock", "bedrock-mantle", "cerebras", "cohere",
    "deepseek", "gemini", "groq", "mistral", "ollama", "openai", "parasail",
    "perplexity", "sgl", "vertex", "openrouter", "huggingface", "nebius", "xai",
    "replicate", "vllm", "sarvam", "wafer", "fireworks", "opencode-go", "opencode-zen",
)
MODELS = [
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "openai/gpt-5.6-sol",
    "anthropic/new-chat-model",
    "gemini/new-chat-model",
    "custom-provider/deployment/new-model",
] + [f"{provider}/new-release/deployment-model" for provider in CHAT_PROVIDERS]


def native_response(protocol: str, model: str, stream: bool, http=httpx):
    if protocol == "chat":
        body = {
            "id": "chat-mock",
            "created": 0,
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if stream:
            body["object"] = "chat.completion.chunk"
            body["choices"][0]["delta"] = body["choices"][0].pop("message")
            wire = f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n"
    else:
        body = {
            "id": "resp-mock",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": model,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "output": [{
                "id": "msg-mock", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
            }],
        }
        if stream:
            event = {"type": "response.completed", "sequence_number": 0, "response": body}
            wire = f"event: response.completed\ndata: {json.dumps(event)}\n\n"
    if stream:
        return http.Response(200, headers={"content-type": "text/event-stream"}, stream=http.ByteStream(wire.encode()))
    return http.Response(200, json=body)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("protocol", ["chat", "responses"])
def test_native_openai_paths_preserve_model_ids_without_sampling_or_tool_defaults(
    shield_factory, model, stream, protocol,
):
    _, http = openai_backend()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return native_response(protocol, model, stream, http)

    shield = shield_factory(lambda _: pytest.fail("native requests must use the selected transport"), base_url="https://gateway.invalid")
    with http.Client(
        transport=http.MockTransport(handler),
        event_hooks={"request": [build_request_hook(PROVIDER_OPENAI)]},
    ) as transport:
        with shield.openai(http_client=transport, max_retries=0) as client:
            if protocol == "chat":
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hello"}],
                    stream=stream,
                )
            else:
                response = client.responses.create(model=model, instructions="Be helpful.", input="Hello", stream=stream)
            if stream:
                assert list(response)
                response.close()
            else:
                assert response.model == model
                if protocol == "responses":
                    assert response.output_text == "OK"
                else:
                    assert response.choices[0].message.content == "OK"

    assert len(seen) == 1
    request = seen[0]
    suffix = "chat/completions" if protocol == "chat" else "responses"
    assert str(request.url) == f"https://gateway.invalid/openai/{suffix}"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    body = json.loads(request.content)
    assert body["model"] == model
    assert body["stream"] is stream
    if "/" in model and not model.startswith("openai/"):
        assert "prompt_cache_key" not in body
    for parameter in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "max_output_tokens", "tools", "tool_choice", "reasoning", "reasoning_effort"):
        assert parameter not in body


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
def test_unified_chat_preserves_explicit_reasoning_and_completion_limits(shield_factory, model, stream):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return native_response("chat", model, stream)

    shield = shield_factory(handler, base_url="https://gateway.invalid")
    response = shield.chat(
        model=model, messages=[{"role": "user", "content": "Hello"}],
        stream=stream, max_completion_tokens=512, reasoning_effort="none",
    )
    if stream:
        assert list(response)
    else:
        assert response["choices"][0]["message"]["content"] == "OK"
    assert len(seen) == 1
    assert str(seen[0].url) == "https://gateway.invalid/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body == {
        "model": model, "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream, "max_completion_tokens": 512, "reasoning_effort": "none",
    }


def test_native_responses_preserves_explicit_reasoning_tools_and_output_limit(shield_factory):
    _, http = openai_backend()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return native_response("responses", "gpt-5.6-sol", False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}]
    with http.Client(transport=http.MockTransport(handler)) as transport:
        with shield.openai(http_client=transport, max_retries=0) as client:
            client.responses.create(
                model="gpt-5.6-sol", input="Hello", reasoning={"effort": "high"},
                max_output_tokens=512, tools=tools,
            )
    body = json.loads(seen[0].content)
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_output_tokens"] == 512
    assert body["tools"] == tools
    assert "temperature" not in body
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("provider,path", [
    ("elevenlabs", "/v1/audio/speech"),
    ("runway", "/v1/videos"),
])
def test_non_chat_provider_model_ids_remain_opaque_on_their_own_operations(shield_factory, provider, path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "mock-operation", "status": "completed"})

    shield = shield_factory(handler, base_url="https://gateway.invalid")
    model = f"{provider}/new-release/model"
    result = shield.request("POST", path, json_body={"model": model, "input": "Hello"})
    assert result["id"] == "mock-operation"
    assert str(seen[0].url) == "https://gateway.invalid" + path
    assert json.loads(seen[0].content) == {"model": model, "input": "Hello"}
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"


def test_provider_contract_matrix_covers_all_builtin_identities():
    assert len(set(CHAT_PROVIDERS) | {"elevenlabs", "runway"}) == 29
