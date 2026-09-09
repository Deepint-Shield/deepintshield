"""Real Strands and ADK model interfaces; mock gateway/PDP, native execution."""

import json

import pytest

from .native_sdk_helpers import openai_backend
from .test_model_compatibility import CHAT_PROVIDERS, native_response


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", CHAT_PROVIDERS)
async def test_strands_native_openai_stream_gateway(shield_factory, provider):
    pytest.importorskip("strands.models.openai")
    _, http = openai_backend()
    model = f"{provider}/new-release/deployment"
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, True, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        native = shield.strands().model(model, client_args={"http_client": transport})
        chunks = [chunk async for chunk in native.stream(messages=[{"role": "user", "content": [{"text": "Hi"}]}])]
        assert chunks
    assert len(seen) == 1
    assert str(seen[0].url) == "https://gateway.invalid/openai/chat/completions"
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"
    assert json.loads(seen[0].content)["model"] == model


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", CHAT_PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
async def test_google_adk_native_connector_gateway(shield_factory, monkeypatch, provider, stream):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    pytest.importorskip("google.adk.models.lite_llm")
    litellm = pytest.importorskip("litellm")
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types
    from litellm.llms.openai.openai import OpenAIChatCompletion
    from deepintshield.agentic.integrations import litellm as enforcement
    # Focus on ADK/LiteLLM/OpenAI serialization; authorization is exercised by
    # the existing fail-closed integration suite, without a running PDP here.
    monkeypatch.setattr(enforcement, "_check", lambda kwargs, args: kwargs)
    litellm.in_memory_llm_clients_cache.flush_cache()
    _, http = openai_backend()
    model = f"{provider}/new-release/deployment"
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, stream, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        monkeypatch.setattr(OpenAIChatCompletion, "_get_async_http_client", staticmethod(lambda **kwargs: transport))
        native = shield.google_adk().model(model)
        request = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="Hi")])])
        response = [chunk async for chunk in native.generate_content_async(request, stream=stream)]
        assert response
    assert len(seen) == 1
    assert str(seen[0].url) == "https://gateway.invalid/openai/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-ds-test"
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"
    assert json.loads(seen[0].content)["model"] == model


@pytest.mark.asyncio
async def test_temporal_native_activity_environment_runs_inference_example(shield_factory, monkeypatch):
    """Temporal executes the real example activity; no workflow service needed.

    ActivityEnvironment does not install Worker interceptors. Their allow/deny
    behavior is covered separately by the enforcement suite.
    """
    import importlib.util
    from pathlib import Path

    testing = pytest.importorskip("temporalio.testing")
    from deepintshield import DeepintShield

    path = Path(__file__).parents[1] / "examples" / "temporal" / "inference.py"
    spec = importlib.util.spec_from_file_location("temporal_inference_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    _, http = openai_backend()
    model = "anthropic/opaque-model"
    monkeypatch.setenv("DEEPINTSHIELD_MODEL", model)
    seen = []

    def handler(request):
        seen.append(request)
        return native_response("chat", model, False, http)

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery"), base_url="https://gateway.invalid")
    monkeypatch.setattr(DeepintShield, "from_env", classmethod(lambda cls: shield))
    native_factory = shield.async_openai
    async with http.AsyncClient(transport=http.MockTransport(handler)) as transport:
        monkeypatch.setattr(shield, "async_openai", lambda: native_factory(http_client=transport, max_retries=0))
        assert await testing.ActivityEnvironment().run(example.infer, "Hello")
    assert len(seen) == 1
    assert str(seen[0].url) == "https://gateway.invalid/openai/chat/completions"
    assert seen[0].headers["x-deepintshield-vk"] == "sk-ds-test"
    assert json.loads(seen[0].content)["model"] == model
