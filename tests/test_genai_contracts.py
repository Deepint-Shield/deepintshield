"""Native Google SDK contracts exercised entirely through mocked HTTP traffic."""

import json
import logging

import httpx
import pytest

pytest.importorskip("google.genai")
from google.genai import types


MODEL = "gemini-3.5-flash"


@pytest.fixture
def afc_warnings(monkeypatch, caplog):
    from google.genai import models

    # Newer upstream releases log once per class/process. Reset only in tests
    # so an earlier direct call cannot hide a warning from the recommended API.
    for model_class in (models.Models, models.AsyncModels):
        if hasattr(model_class, "_logged_afc_warning"):
            monkeypatch.setattr(model_class, "_logged_afc_warning", False)
    caplog.set_level(logging.WARNING, logger="google_genai.models")
    return lambda: [
        record.getMessage() for record in caplog.records
        if "Direct use of automatic function calling" in record.getMessage()
    ]


def _response(request, parts):
    body = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}
    if request.url.path.endswith(":streamGenerateContent"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=f"data: {json.dumps(body)}\n\n")
    return httpx.Response(200, json=body)


def _assert_gateway_request(request, *, stream):
    operation = "streamGenerateContent" if stream else "generateContent"
    assert request.url.host == "gateway.invalid"
    assert request.url.path == f"/genai/v1beta/models/{MODEL}:{operation}"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    assert request.headers["x-deepintshield-agent"] == "deepintshield-test-agent"
    if stream:
        assert request.url.params["alt"] == "sse"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("disable_afc", [False, True])
def test_genai_plain_content_keeps_native_defaults(shield_factory, stream, disable_afc, afc_warnings):
    requests = []

    def handler(request):
        requests.append(request)
        return _response(request, [{"text": "hello"}])

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery call"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_client=transport))
        try:
            # SDK 2.22 warns about default AFC even when there are no tools. The
            # warning must not be confused with a failed model request.
            config = {"automatic_function_calling": {"disable": True}} if disable_afc else None
            if stream:
                text = "".join(chunk.text or "" for chunk in client.models.generate_content_stream(model=MODEL, contents="hello", config=config))
            else:
                text = client.models.generate_content(model=MODEL, contents="hello", config=config).text
            assert text == "hello"
            assert len(requests) == 1
            _assert_gateway_request(requests[0], stream=stream)
            body = json.loads(requests[0].content)
            assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
            assert not body.get("tools")
            assert not body.get("generationConfig")
            assert "automatic_function_calling" not in body
            if disable_afc:
                assert not afc_warnings()
        finally:
            client.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_chat", [False, True])
def test_genai_automatic_function_roundtrip_preserves_signatures(shield_factory, stream, use_chat, afc_warnings):
    requests = []
    calls = []

    def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return _response(request, [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}},
                                       "thoughtSignature": "c2lnbmF0dXJl"}])
        return _response(request, [{"text": "5"}])

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery call"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_client=transport))
        try:
            config = types.GenerateContentConfig(tools=[add])
            if use_chat:
                chat = client.chats.create(model=MODEL, config=config)
                response = chat.send_message_stream("2 + 3?") if stream else chat.send_message("2 + 3?")
            else:
                generate = client.models.generate_content_stream if stream else client.models.generate_content
                response = generate(model=MODEL, contents="2 + 3?", config=config)
            text = "".join(chunk.text or "" for chunk in response) if stream else response.text
            assert text == "5"
            assert calls == [(2, 3)]
            assert len(requests) == 2
            for request in requests:
                _assert_gateway_request(request, stream=stream)
            followup = json.loads(requests[1].content)["contents"]
            function_call = followup[-2]["parts"][0]
            assert function_call["functionCall"]["name"] == "add"
            assert function_call["thoughtSignature"] == "c2lnbmF0dXJl"
            assert followup[-1]["parts"][0]["functionResponse"] == {"name": "add", "response": {"result": 5}}
            assert config.tools == [add], "SDK options remain reusable after a tool round trip"
            if use_chat:
                assert not afc_warnings()
                history = chat.get_history()
                assert history[0].parts[0].text == "2 + 3?"
                assert history[-1].parts[0].text == "5"
        finally:
            client.close()


def test_genai_disabling_afc_returns_tool_call_without_execution(shield_factory):
    calls = []

    def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    def handler(request):
        return _response(request, [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}}])

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery call"), base_url="https://gateway.invalid")
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_client=transport))
        try:
            response = client.models.generate_content(model=MODEL, contents="2 + 3?", config={
                "tools": [add], "automatic_function_calling": {"disable": True},
            })
            assert response.function_calls[0].name == "add"
            assert calls == []
        finally:
            client.close()


@pytest.mark.parametrize("as_dict", [False, True])
def test_genai_custom_http_options_retain_gateway_routing(shield_factory, as_dict):
    requests = []
    callbacks = []

    def handler(request):
        requests.append(request)
        return _response(request, [{"text": "hello"}])

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery call"), base_url="https://gateway.invalid")
    settings = {"timeout": 1234, "client_args": {
        "transport": httpx.MockTransport(handler), "event_hooks": {"request": [callbacks.append]},
    }}
    options = settings if as_dict else types.HttpOptions(**settings)
    client = shield.genai(http_options=options)
    try:
        assert client.models.generate_content(model=MODEL, contents="hello").text == "hello"
        assert len(requests) == len(callbacks) == 1
        _assert_gateway_request(requests[0], stream=False)
        assert "base_url" not in settings
        if not as_dict:
            assert options.base_url is None
            assert options.headers is None
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_chat", [False, True])
@pytest.mark.parametrize("disable_afc", [False, True])
async def test_genai_async_content_and_chat(shield_factory, stream, use_chat, disable_afc, afc_warnings):
    requests = []

    def handler(request):
        requests.append(request)
        return _response(request, [{"text": "hello"}])

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery call"), base_url="https://gateway.invalid")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_async_client=transport))
        try:
            config = {"automatic_function_calling": {"disable": True}} if disable_afc else None
            if use_chat:
                chat = client.aio.chats.create(model=MODEL, config=config)
                response = await (chat.send_message_stream("hello") if stream else chat.send_message("hello"))
            else:
                generate = client.aio.models.generate_content_stream if stream else client.aio.models.generate_content
                response = await generate(model=MODEL, contents="hello", config=config)
            text = "".join([chunk.text or "" async for chunk in response]) if stream else response.text
            assert text == "hello"
            assert len(requests) == 1
            _assert_gateway_request(requests[0], stream=stream)
            if use_chat or disable_afc:
                assert not afc_warnings()
        finally:
            await client.aio.aclose()
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_genai_async_chat_executes_callable_roundtrip(shield_factory, stream, afc_warnings):
    requests = []
    calls = []

    async def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return _response(request, [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}}])
        return _response(request, [{"text": "5"}])

    shield = shield_factory(lambda _: pytest.fail("unexpected discovery call"), base_url="https://gateway.invalid")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_async_client=transport))
        try:
            chat = client.aio.chats.create(model=MODEL, config={"tools": [add]})
            response = await (chat.send_message_stream("2 + 3?") if stream else chat.send_message("2 + 3?"))
            text = "".join([chunk.text or "" async for chunk in response]) if stream else response.text
            assert text == "5"
            assert calls == [(2, 3)]
            assert not afc_warnings()
            assert len(requests) == 2
            for request in requests:
                _assert_gateway_request(request, stream=stream)
            result = json.loads(requests[1].content)["contents"][-1]["parts"][0]["functionResponse"]
            assert result == {"name": "add", "response": {"result": 5}}
        finally:
            await client.aio.aclose()
            client.close()
