"""Native LangChain/LangGraph model and embedding transport contracts."""

from __future__ import annotations

import json

import httpx
import pytest

from .test_model_compatibility import CHAT_PROVIDERS, native_response


MODELS = [f"{provider}/new-release/deployment-model" for provider in CHAT_PROVIDERS]


def build_model(shield, surface, model, **kwargs):
    if surface == "shortcut":
        return shield.langchain(model=model, **kwargs)
    return shield.bind(surface).model(model=model, **kwargs)


def assert_chat_request(request, surface, model):
    route = "langchain" if surface == "shortcut" else "openai"
    assert str(request.url) == f"https://gateway.invalid/{route}/chat/completions"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    assert request.headers["x-deepintshield-agent"] == "reviewer"
    assert request.headers.get_list("x-deepintshield-agent") == ["reviewer"]
    body = json.loads(request.content)
    assert body["model"] == model
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    for parameter in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "reasoning_effort", "tools"):
        assert parameter not in body


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("surface", ["shortcut", "langchain", "langgraph"])
@pytest.mark.parametrize("stream", [False, True])
def test_langchain_native_sync_preserves_provider_models(shield_factory, model, surface, stream):
    pytest.importorskip("langchain_openai")
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, stream)

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        native = build_model(
            shield, surface, model, http_client=transport, max_retries=0,
            default_headers={"X-DeepIntShield-Agent": "reviewer"},
        )
        response = list(native.stream("Hello")) if stream else native.invoke("Hello")
        assert response
        if not stream:
            assert response.content == "OK"
    assert len(seen) == 1
    assert_chat_request(seen[0], surface, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("surface", ["shortcut", "langchain", "langgraph"])
@pytest.mark.parametrize("stream", [False, True])
async def test_langchain_native_async_preserves_provider_models(shield_factory, model, surface, stream):
    pytest.importorskip("langchain_openai")
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, stream)

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        native = build_model(
            shield, surface, model, http_async_client=transport, max_retries=0,
            default_headers={"X-DeepIntShield-Agent": "reviewer"},
        )
        response = [chunk async for chunk in native.astream("Hello")] if stream else await native.ainvoke("Hello")
        assert response
        if not stream:
            assert response.content == "OK"
    assert len(seen) == 1
    assert_chat_request(seen[0], surface, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["langchain", "langgraph"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_langchain_public_responses_selection(shield_factory, surface, asynchronous):
    pytest.importorskip("langchain_openai")
    model = "anthropic/opaque-responses-model"
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("responses", model, False)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as sync:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            native = shield.bind(surface).model(model, use_responses_api=True, http_client=sync,
                                                 http_async_client=transport, max_retries=0)
            result = await native.ainvoke("Hello") if asynchronous else native.invoke("Hello")
            assert "OK" in str(result.content)
    assert len(seen) == 1
    assert str(seen[0].url) == "https://gateway.invalid/openai/responses"
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"
    assert json.loads(seen[0].content)["model"] == model


@pytest.mark.parametrize("surface", ["langchain", "langgraph"])
@pytest.mark.parametrize("model", ["cohere/embed-v4.0", "gemini/gemini-embedding-001", "custom/deployment/new-embedding"])
def test_langchain_embeddings_keep_raw_text_for_gateway_models(shield_factory, surface, model):
    pytest.importorskip("langchain_openai")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"object": "list", "model": model, "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        ], "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        embedder = shield.bind(surface).embedder(model=model, http_client=transport, max_retries=0)
        assert embedder.embed_documents(["Original text for the provider tokenizer"]) == [[0.1, 0.2]]
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["model"] == model
    assert body["input"] == ["Original text for the provider tokenizer"]
    assert str(seen[0].url) == "https://gateway.invalid/openai/embeddings"
