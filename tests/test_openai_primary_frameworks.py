"""Real framework model interfaces against a mock OpenAI-compatible gateway.

These isolate inference serialization. Existing Agentic suites separately test
the unchanged framework authorization and native tool-execution boundaries.
"""

from __future__ import annotations

import json
import importlib

import pytest

from .native_sdk_helpers import openai_backend
from .test_model_compatibility import CHAT_PROVIDERS, native_response

MODELS = [f"{provider}/new-release/deployment" for provider in CHAT_PROVIDERS]


def assert_request(seen, model, protocol="chat"):
    assert len(seen) == 1
    request = seen[0]
    path = "chat/completions" if protocol == "chat" else "responses"
    assert str(request.url) == f"https://gateway.invalid/openai/{path}"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    assert request.headers["x-deepintshield-agent"] == "deepintshield-test-agent"
    assert json.loads(request.content)["model"] == model


def mock_crewai_clients(monkeypatch, sdk, sync, asynchronous):
    # CrewAI constructs both native clients with the same connection options;
    # replace only their HTTP transports, not model execution or serialization.
    module = importlib.import_module("crewai.llms.providers.openai.completion")
    monkeypatch.setattr(module, "OpenAI", lambda **options: sdk.OpenAI(**{**options, "http_client": sync}))
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **options: sdk.AsyncOpenAI(**{**options, "http_client": asynchronous}))


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_crewai_native_openai_gateway(shield_factory, monkeypatch, model):
    pytest.importorskip("crewai")
    sdk, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    with http.Client(transport=http.MockTransport(handler)) as transport:
        async with http.AsyncClient(transport=http.MockTransport(handler)) as asynchronous:
            mock_crewai_clients(monkeypatch, sdk, transport, asynchronous)
            native = shield.crewai().llm(model, max_retries=0)
            assert native.call([{"role": "user", "content": "Hi"}]) == "OK"
    assert_request(seen, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_crewai_native_async_responses_gateway(shield_factory, monkeypatch, model):
    pytest.importorskip("crewai")
    sdk, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("responses", model, False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    with http.Client(transport=http.MockTransport(handler)) as sync:
        async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
            mock_crewai_clients(monkeypatch, sdk, sync, transport)
            native = shield.crewai().llm(model, api="responses", max_retries=0)
            assert await native.acall([{"role": "user", "content": "Hi"}]) == "OK"
    assert_request(seen, model, "responses")


@pytest.mark.parametrize("model", MODELS)
def test_llamaindex_openai_like_preserves_custom_chat_models(shield_factory, model):
    pytest.importorskip("llama_index.llms.openai_like")
    from llama_index.core.llms import ChatMessage
    _, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    with http.Client(transport=http.MockTransport(handler)) as transport:
        native = shield.llamaindex().llm(model, http_client=transport, max_retries=0, context_window=8192)
        assert native.chat([ChatMessage(role="user", content="Hi")]).message.content == "OK"
        assert native.metadata.context_window == 8192
    assert_request(seen, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_llamaindex_openai_like_async_stream(shield_factory, model):
    pytest.importorskip("llama_index.llms.openai_like")
    from llama_index.core.llms import ChatMessage
    _, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, True, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        native = shield.llamaindex().llm(model, async_http_client=transport, max_retries=0)
        response = await native.astream_chat([ChatMessage(role="user", content="Hi")])
        assert [chunk async for chunk in response]
    assert_request(seen, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_llamaindex_openai_like_embedding_preserves_custom_model_and_text(shield_factory, asynchronous):
    pytest.importorskip("llama_index.embeddings.openai_like")
    _, http = openai_backend()
    model = "cohere/embed-v4.0"
    seen = []

    def handler(request):
        seen.append(request)
        return http.Response(200, json={"object": "list", "model": model,
            "data": [{"index": 0, "object": "embedding", "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    with http.Client(transport=http.MockTransport(handler)) as sync:
        async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
            native = shield.llamaindex().embedder(model, http_client=sync, async_http_client=transport, max_retries=0)
            result = await native.aget_text_embedding("Hello") if asynchronous else native.get_text_embedding("Hello")
            assert result == [0.1, 0.2]
    assert len(seen) == 1
    assert str(seen[0].url) == "https://gateway.invalid/openai/embeddings"
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"
    assert json.loads(seen[0].content)["model"] == model
    assert json.loads(seen[0].content)["input"] == ["Hello"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_pydanticai_native_gateway_models(shield_factory, model, protocol):
    pytest.importorskip("pydantic_ai.models.openai")
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    _, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response(protocol, model, False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        native = shield.bind("pydanticai").model(model, api="responses" if protocol == "responses" else "chat_completions", http_client=transport)
        result = await native.request([ModelRequest(parts=[UserPromptPart("Hi")])], None, ModelRequestParameters())
        assert result.parts[0].content == "OK"
    assert_request(seen, model, protocol)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_openai_agents_native_model_gateway(shield_factory, model, protocol):
    pytest.importorskip("agents")
    from agents import ModelSettings
    from agents.models.interface import ModelTracing
    _, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response(protocol, model, False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        async with shield.openai_agents().client(http_client=transport) as client:
            native = shield.openai_agents().model(model, api="responses" if protocol == "responses" else "chat_completions", openai_client=client)
            response = await native.get_response(system_instructions=None, input="Hi", model_settings=ModelSettings(), tools=[], output_schema=None, handoffs=[], tracing=ModelTracing.DISABLED)
            assert response.output[0].content[0].text == "OK"
    assert_request(seen, model, protocol)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
async def test_autogen_native_gateway_models_preserve_explicit_metadata(shield_factory, model, stream):
    pytest.importorskip("autogen_ext.models.openai")
    from autogen_core.models import UserMessage
    _, http = openai_backend()
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, stream, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        info = {"vision": False, "function_calling": True, "json_output": True, "structured_output": False, "family": "unknown"}
        native = shield.autogen().client(model, model_info=info, http_client=transport)
        if stream:
            assert [chunk async for chunk in native.create_stream([UserMessage(content="Hi", source="user")])]
        else:
            assert (await native.create([UserMessage(content="Hi", source="user")])).content == "OK"
        assert native.model_info == info
        await native.close()
    assert_request(seen, model)
