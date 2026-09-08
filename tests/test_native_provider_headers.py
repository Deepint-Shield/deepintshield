from __future__ import annotations

import json
import importlib

import httpx
import pytest


def _native_response(provider: str, stream: bool, http=httpx):
    if provider == "openai":
        body = {"id": "reply", "object": "chat.completion", "model": "test-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
        if stream:
            body["object"] = "chat.completion.chunk"
            body["choices"][0]["delta"] = body["choices"][0].pop("message")
            data = f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n"
    elif provider == "anthropic":
        body = {"id": "reply", "type": "message", "role": "assistant", "model": "test-model",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}}
        if stream:
            data = 'event: message_start\ndata: ' + json.dumps({"type": "message_start", "message": body})
            data += '\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
    else:
        body = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
        if stream:
            data = f"data: {json.dumps(body)}\n\n"
    if stream:
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=data)
    return http.Response(200, json=body)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "genai"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("override", ["none", "unnamed", "shield", "constructor", "request"])
def test_native_provider_outbound_agent_selectors(shield_factory, provider, stream, override):
    sdk = pytest.importorskip("google.genai" if provider == "genai" else provider)
    http = httpx
    if provider == "anthropic":
        backend = next(cls.__module__.partition(".")[0] for cls in sdk.DefaultHttpxClient.__mro__ if cls.__name__ == "Client")
        http = importlib.import_module(backend)
    seen = []
    callback_requests = []

    def handler(request):
        seen.append(request)
        return _native_response(provider, stream, http)

    # Two profiles sharing a VK and transport must keep their selectors separate.
    # Calling the first again after the second also catches mutable hook state.
    native_clients = []
    with http.Client(transport=http.MockTransport(handler), event_hooks={"request": [callback_requests.append]}) as http_client:
        try:
            for profile in ("planner", "auditor"):
                extra = {"X-DeepIntShield-Agent": "explicit-" + profile}
                shield = shield_factory(
                    lambda _: pytest.fail("native header construction must not trigger Agentic discovery"),
                    base_url="http://gateway.invalid", agent_name="" if override == "unnamed" else profile,
                    default_headers=extra if override == "shield" else None,
                )
                if provider == "genai":
                    from google.genai.types import HttpOptions
                    options = HttpOptions(httpx_client=http_client, timeout=1234,
                                          headers=extra if override == "constructor" else None)
                    client = shield.genai(http_options=options)
                    # Adding gateway headers must not mutate caller-owned options.
                    assert options.headers == (extra if override == "constructor" else None)
                    assert options.base_url is None
                else:
                    client = getattr(shield, provider)(http_client=http_client, max_retries=0,
                                                       default_headers=extra if override == "constructor" else None)
                native_clients.append(client)

            for index in (0, 1, 0):
                profile = ("planner", "auditor")[index]
                extra = {"X-DeepIntShield-Agent": "request-" + profile} if override == "request" else {}
                client = native_clients[index]
                if provider == "genai":
                    method = client.models.generate_content_stream if stream else client.models.generate_content
                    response = method(model="test-model", contents="hello", config={"http_options": {"headers": extra}})
                else:
                    method = client.chat.completions.create if provider == "openai" else client.messages.create
                    response = method(model="test-model", messages=[{"role": "user", "content": "hello"}],
                                      max_tokens=5, stream=stream, extra_headers=extra)
                if stream:
                    assert list(response)
                else:
                    assert response is not None
                request = seen[-1]
                expected = profile
                if override == "unnamed":
                    expected = None
                elif override in ("shield", "constructor"):
                    expected = "explicit-" + profile
                elif override == "request":
                    expected = "request-" + profile
                assert request.headers.get_list("x-deepintshield-agent") == ([] if expected is None else [expected])
                assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
                assert request.headers["x-deepintshield-app"] == "deepintshield"
                assert "x-agent-subject" not in request.headers
                assert "x-deepintshield-agent-principal" not in request.headers
                assert "x-agent-token" not in request.headers
                assert request.url.host == "gateway.invalid"
            assert len(seen) == len(callback_requests) == 3
        finally:
            for client in native_clients:
                client.close()
