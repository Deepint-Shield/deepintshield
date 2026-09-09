"""Exercise real LiteLLM serialization; only the network and PDP are mocked."""

from __future__ import annotations

import json
from copy import deepcopy

import httpx
import pytest

from .test_model_compatibility import CHAT_PROVIDERS, native_response


MODELS = [f"{provider}/new-release/deployment-model" for provider in CHAT_PROVIDERS] + [
    "gpt-6-astra", "custom-provider/deployment/new-model",
]


@pytest.fixture
def litellm_wire(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    native = pytest.importorskip("litellm")
    # Authorization is covered by test_agentic_new_enforcement.py. These
    # contracts isolate native SDK routing without requiring a running PDP.
    from deepintshield.agentic.integrations import litellm as integration
    monkeypatch.setattr(integration, "_check", lambda kwargs, args: kwargs)
    from litellm.llms.openai.openai import OpenAIChatCompletion
    native.in_memory_llm_clients_cache.flush_cache()
    return OpenAIChatCompletion


def assert_wire(request, model, stream, messages, original):
    assert str(request.url) == "https://gateway.invalid/litellm/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-ds-test"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    assert request.headers["x-deepintshield-agent"] == "reviewer"
    body = json.loads(request.content)
    assert body["model"] == model
    assert body["messages"] == messages == original
    assert body.get("stream", False) is stream
    assert body["max_completion_tokens"] == 512
    assert body["reasoning_effort"] == "none"
    assert body["provider_option"] == {"value": "unchanged"}
    assert "temperature" not in body
    assert "custom_llm_provider" not in body


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
def test_litellm_native_sync_preserves_gateway_provider_models(shield_factory, monkeypatch, litellm_wire, model, stream):
    seen = []
    messages = [{"role": "user", "content": "Hello"}]
    original = deepcopy(messages)

    def handler(request):
        seen.append(request)
        return native_response("chat", model, stream)

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        monkeypatch.setattr(litellm_wire, "_get_sync_http_client", staticmethod(lambda: transport))
        response = shield.litellm().completion(
            model=model, messages=messages, stream=stream,
            max_completion_tokens=512, reasoning_effort="none",
            extra_body={"provider_option": {"value": "unchanged"}},
            extra_headers={"X-DeepIntShield-Agent": "reviewer"},
        )
        if stream:
            assert list(response)
        else:
            assert response.choices[0].message.content == "OK"
    assert len(seen) == 1
    assert_wire(seen[0], model, stream, messages, original)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
async def test_litellm_native_async_preserves_gateway_provider_models(shield_factory, monkeypatch, litellm_wire, model, stream):
    seen = []
    messages = [{"role": "user", "content": "Hello"}]
    original = deepcopy(messages)

    def handler(request):
        seen.append(request)
        return native_response("chat", model, stream)

    shield = shield_factory(lambda _: pytest.fail("unexpected direct request"), base_url="https://gateway.invalid")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        monkeypatch.setattr(litellm_wire, "_get_async_http_client", staticmethod(lambda **kwargs: transport))
        response = await shield.litellm().acompletion(
            model=model, messages=messages, stream=stream,
            max_completion_tokens=512, reasoning_effort="none",
            extra_body={"provider_option": {"value": "unchanged"}},
            extra_headers={"X-DeepIntShield-Agent": "reviewer"},
        )
        if stream:
            assert [chunk async for chunk in response]
        else:
            assert response.choices[0].message.content == "OK"
    assert len(seen) == 1
    assert_wire(seen[0], model, stream, messages, original)
