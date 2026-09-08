"""Exercise the installed Anthropic SDK's own HTTP transport, without network calls."""

from __future__ import annotations

import importlib
import json

import pytest

from deepintshield.transport import _install_agent_selector_header_hook


def anthropic_backend():
    sdk = pytest.importorskip("anthropic")
    backend = next(cls.__module__.partition(".")[0] for cls in sdk.DefaultHttpxClient.__mro__ if cls.__name__ == "Client")
    return sdk, importlib.import_module(backend)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
def test_default_anthropic_transport_preserves_cache_wire_and_gateway_headers(shield_factory, monkeypatch, stream, passthrough):
    sdk, http = anthropic_backend()
    seen = []
    native_transport = sdk.DefaultHttpxClient

    def handler(request):
        seen.append(request)
        message = {
            "id": "msg-mock", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
            "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if stream:
            wire = "event: message_start\ndata: " + json.dumps({"type": "message_start", "message": message})
            wire += '\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n'
            return http.Response(200, headers={"content-type": "text/event-stream"}, stream=http.ByteStream(wire.encode()))
        return http.Response(200, json=message)

    def with_mock_transport(**kwargs):
        return native_transport(transport=http.MockTransport(handler), **kwargs)

    monkeypatch.setattr(sdk, "DefaultHttpxClient", with_mock_transport)
    shield = shield_factory(lambda _: pytest.fail("unexpected direct gateway call"), base_url="https://gateway.invalid", agent_name="planner")
    with shield.anthropic(passthrough=passthrough, max_retries=0) as client:
        assert isinstance(client._client, http.Client)
        assert client._client.follow_redirects is False
        response = client.messages.create(
            model="claude-opus-4-8", max_tokens=256, stream=stream,
            system="Be helpful.", messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}],
            extra_headers={"X-DeepIntShield-Agent": "reviewer"},
        )
        if stream:
            assert list(response)
            response.close()
        else:
            assert response.content[0].text == "OK"
    assert len(seen) == 1
    request = seen[0]
    route = "anthropic_passthrough" if passthrough else "anthropic"
    assert str(request.url) == f"https://gateway.invalid/{route}/v1/messages"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    assert request.headers.get_list("x-deepintshield-agent") == ["reviewer"]
    assert b"".join(request.stream) == request.content
    assert int(request.headers["content-length"]) == len(request.content)
    body = json.loads(request.content)
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["max_tokens"] == 256
    assert "temperature" not in body
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_native_async_transport_gets_an_awaitable_idempotent_selector_hook():
    _, http = anthropic_backend()
    seen = []

    async def existing_hook(request):
        seen.append(request)

    async with http.AsyncClient(
        transport=http.MockTransport(lambda _: http.Response(200, json={"ok": True})),
        event_hooks={"request": [existing_hook]},
    ) as client:
        _install_agent_selector_header_hook(client)
        _install_agent_selector_header_hook(client)
        response = await client.get("https://gateway.invalid", headers=[("x-deepintshield-agent", "first"), ("X-DeepIntShield-Agent", "second")])
        assert response.status_code == 200
        assert len(client.event_hooks["request"]) == 2
        assert seen[0].headers.get_list("x-deepintshield-agent") == ["second"]


@pytest.mark.asyncio
async def test_invalid_caller_transport_is_neither_modified_nor_closed(shield_factory):
    _, http = anthropic_backend()
    shield = shield_factory(lambda _: pytest.fail("unexpected gateway request"))
    async with http.AsyncClient() as custom:
        original_hooks = list(custom.event_hooks["request"])
        with pytest.raises(TypeError, match="http_client"):
            shield.anthropic(http_client=custom)
        assert custom.event_hooks["request"] == original_hooks
        assert not custom.is_closed
