from __future__ import annotations

import gc
import json
import weakref
from typing import Iterator

import httpx
import pytest

from deepintshield import (
    ChatCompletionStream,
    DeepintShield,
    DeepintShieldError,
    get_exception_error_code,
)


class _TrackingStream(httpx.SyncByteStream):
    def __init__(self, *items: bytes | BaseException) -> None:
        self.items = items
        self.reads = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for item in self.items:
            self.reads += 1
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self.closed = True


def _stream_response(
    wire: _TrackingStream,
    *,
    status_code: int = 200,
    content_type: str = "text/event-stream; charset=utf-8",
) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": content_type},
        stream=wire,
    )


def test_stream_is_lazy_yields_json_and_consumes_terminal_event(shield_factory):
    wire = _TrackingStream(
        b': keepalive\n\ndata: {"id":"one","choices":[]}\n\n',
        b'data: [DONE]\n\n',
    )
    shield = shield_factory(lambda _request: _stream_response(wire))

    stream = shield.chat(model="model", messages=[], stream=True)

    assert isinstance(stream, ChatCompletionStream)
    assert wire.reads == 0
    assert stream.closed is False
    assert next(stream) == {"id": "one", "choices": []}
    assert wire.reads == 1
    with pytest.raises(StopIteration):
        next(stream)
    assert wire.reads == 2
    assert stream.closed is True
    assert wire.closed is True


def test_stream_supports_multiline_data_and_utf8_bom(shield_factory):
    wire = _TrackingStream(
        b'\xef\xbb\xbfdata: {"id":"one",\n'
        b'data: "choices":[]}\n\n'
        b'data: [DONE]\n\n'
    )
    shield = shield_factory(lambda _request: _stream_response(wire))

    with shield.chat(model="model", messages=[], stream=True) as stream:
        assert list(stream) == [{"id": "one", "choices": []}]

    assert stream.closed is True
    assert wire.closed is True


@pytest.mark.parametrize("newline", [b"\r\n", b"\r"], ids=["crlf", "cr"])
def test_stream_supports_all_sse_line_endings(newline: bytes, shield_factory):
    wire = _TrackingStream(
        newline.join(
            [
                b'data: {"id":"one"}',
                b"",
                b"data: [DONE]",
                b"",
            ]
        )
    )
    shield = shield_factory(lambda _request: _stream_response(wire))

    assert list(shield.chat(model="model", messages=[], stream=True)) == [
        {"id": "one"}
    ]
    assert wire.closed is True


def test_terminal_done_is_accepted_without_a_final_blank_delimiter(shield_factory):
    wire = _TrackingStream(
        b'data: {"id":"one"}\n\n',
        b'data: [DONE]',
    )
    shield = shield_factory(lambda _request: _stream_response(wire))

    stream = shield.chat(model="model", messages=[], stream=True)

    assert list(stream) == [{"id": "one"}]
    assert stream.closed is True
    assert wire.closed is True


def test_breaking_iteration_closes_without_reading_the_full_body(shield_factory):
    wire = _TrackingStream(
        b'data: {"id":"one"}\n\n',
        b'data: {"id":"two"}\n\n',
        b'data: [DONE]\n\n',
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    for chunk in stream:
        assert chunk == {"id": "one"}
        break

    assert stream.closed is True
    assert wire.closed is True
    assert wire.reads == 1


def test_explicit_close_is_idempotent_and_stops_iteration(shield_factory):
    wire = _TrackingStream(
        b'data: {"id":"one"}\n\n',
        b'data: [DONE]\n\n',
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    stream.close()
    stream.close()

    assert stream.closed is True
    assert wire.closed is True
    assert wire.reads == 0
    with pytest.raises(StopIteration):
        next(stream)


def test_malformed_json_event_closes_and_does_not_retain_raw_decoder_context(
    shield_factory,
):
    wire = _TrackingStream(
        b'data: {"private":"unterminated"\n\n',
        b'data: [DONE]\n\n',
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "private" not in repr(caught.value.to_dict())
    assert stream.closed is True
    assert wire.closed is True


def test_non_object_json_event_is_malformed(shield_factory):
    wire = _TrackingStream(b'data: [1,2,3]\n\ndata: [DONE]\n\n')
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert stream.closed is True


def test_eof_before_done_is_a_truncated_stream(shield_factory):
    wire = _TrackingStream(b'data: {"id":"one"}\n\n')
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    assert next(stream) == {"id": "one"}
    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.details["reason"] == "premature_eof"
    assert stream.closed is True
    assert wire.closed is True


def test_unterminated_nonterminal_event_is_not_yielded_at_eof(shield_factory):
    wire = _TrackingStream(b'data: {"id":"must-not-yield"}')
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.details["reason"] == "premature_eof"
    assert stream.closed is True


def test_structured_sse_error_preserves_recognized_gateway_code(shield_factory):
    wire = _TrackingStream(
        b'event: error\n'
        b'data: {"error":{"code":"RATE_LIMITED","message":"slow down"}}\n\n'
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "rate_limited"
    assert caught.value.message == "slow down"
    assert caught.value.retryable is True
    assert stream.closed is True


def test_top_level_sse_error_preserves_recognized_gateway_code(shield_factory):
    wire = _TrackingStream(
        b'event: error\n'
        b'data: {"code":"FEATURE_LOCKED","message":"upgrade"}\n\n'
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "feature_locked"
    assert caught.value.message == "upgrade"
    assert stream.closed is True
    assert wire.closed is True


def test_decoder_limit_failure_is_coded_and_closes_without_retaining_data(
    shield_factory,
):
    private_integer = "9" * 5_000
    wire = _TrackingStream(f"data: {private_integer}\n\n".encode())
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert private_integer not in repr(caught.value.to_dict())
    assert stream.closed is True
    assert wire.closed is True


@pytest.mark.parametrize(
    "event",
    [
        b"data: " + (b"x" * (1024 * 1024 + 1)) + b"\n\n",
        b"data: " + (b"x" * 600_000) + b"\n"
        b"data: " + (b"y" * 600_000) + b"\n\n",
    ],
    ids=["single-line", "multiline"],
)
def test_successful_stream_event_size_is_bounded(event: bytes, shield_factory):
    wire = _TrackingStream(event)
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.details["reason"] == "event_too_large"
    assert caught.value.details["max_event_bytes"] == 1024 * 1024
    assert caught.value.__context__ is None
    assert stream.closed is True
    assert wire.closed is True


def test_invalid_utf8_stream_event_is_coded_and_closed(shield_factory):
    wire = _TrackingStream(b"data: \xff\n\n")
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.details["reason"] == "invalid_utf8"
    assert caught.value.__context__ is None
    assert stream.closed is True
    assert wire.closed is True


def test_midstream_timeout_keeps_httpx_type_adds_code_and_closes(shield_factory):
    wire = _TrackingStream(
        b'data: {"id":"one"}\n\n',
        httpx.ReadTimeout("stream stalled"),
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    assert next(stream) == {"id": "one"}
    with pytest.raises(httpx.ReadTimeout) as caught:
        next(stream)

    assert get_exception_error_code(caught.value) == "transport_timeout"
    assert stream.closed is True
    assert wire.closed is True


def test_http_error_is_raised_before_returning_a_stream_and_body_is_bounded(
    shield_factory,
):
    payload = {"error": {"code": "FEATURE_LOCKED", "message": "upgrade"}}
    wire = _TrackingStream(json.dumps(payload).encode())
    shield = shield_factory(
        lambda _request: _stream_response(
            wire,
            status_code=402,
            content_type="application/json",
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert caught.value.code == "feature_locked"
    assert wire.reads == 1
    assert wire.closed is True


def test_large_http_error_body_is_bounded_and_remaining_chunks_are_not_read(
    shield_factory,
):
    wire = _TrackingStream(*(b"x" * 8192 for _ in range(20)))
    shield = shield_factory(
        lambda _request: _stream_response(
            wire,
            status_code=500,
            content_type="text/plain",
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert caught.value.code == "chat_request_failed"
    assert caught.value.details["response_truncated"] is True
    assert len(caught.value.payload["raw"].encode()) == 64 * 1024
    assert wire.reads == 9
    assert wire.closed is True


def test_http_error_json_decoder_limit_still_uses_stable_error_contract(
    shield_factory,
):
    private_integer = b"9" * 5_000
    wire = _TrackingStream(private_integer)
    shield = shield_factory(
        lambda _request: _stream_response(
            wire,
            status_code=500,
            content_type="application/json",
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert caught.value.code == "chat_request_failed"
    assert caught.value.retryable is False
    assert caught.value.__context__ is None
    assert private_integer.decode() not in repr(caught.value.to_dict())
    assert wire.closed is True


def test_initial_transport_error_is_raised_before_returning_a_stream(shield_factory):
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect stalled")

    shield = shield_factory(timeout)

    with pytest.raises(httpx.ConnectTimeout) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert get_exception_error_code(caught.value) == "transport_timeout"


def test_non_sse_success_is_rejected_and_closed_before_return(shield_factory):
    wire = _TrackingStream(b'{"choices":[]}')
    shield = shield_factory(
        lambda _request: _stream_response(wire, content_type="application/json")
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert caught.value.code == "chat_stream_invalid_event"
    assert wire.closed is True


def test_content_type_must_be_the_exact_sse_media_type(shield_factory):
    wire = _TrackingStream(b'data: [DONE]\n\n')
    shield = shield_factory(
        lambda _request: _stream_response(
            wire,
            content_type="application/x-text/event-streamish",
        )
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert caught.value.code == "chat_stream_invalid_event"
    assert wire.closed is True


def test_missing_sse_content_type_is_rejected_and_closed_before_return(
    shield_factory,
):
    wire = _TrackingStream(b'data: [DONE]\n\n')
    shield = shield_factory(
        lambda _request: httpx.Response(200, stream=wire)
    )

    with pytest.raises(DeepintShieldError) as caught:
        shield.chat(model="model", messages=[], stream=True)

    assert caught.value.code == "chat_stream_invalid_event"
    assert caught.value.details["content_type"] == ""
    assert wire.closed is True


def test_closing_owner_makes_next_read_fail_closed_and_releases_response(
    shield_factory,
):
    wire = _TrackingStream(
        b'data: {"id":"one"}\n\n',
        b'data: [DONE]\n\n',
    )
    shield = shield_factory(lambda _request: _stream_response(wire))
    stream = shield.chat(model="model", messages=[], stream=True)

    shield.close()

    with pytest.raises(DeepintShieldError) as caught:
        next(stream)
    assert caught.value.code == "client_closed"
    assert stream.closed is True
    assert wire.closed is True
    assert wire.reads == 0


def test_stream_keeps_temporary_owner_alive_until_close(mock_transport):
    wire = _TrackingStream(
        b'data: {"id":"one"}\n\n',
        b'data: [DONE]\n\n',
    )

    def make_stream():
        owner = DeepintShield(virtual_key="vk")
        owner._client.close()
        owner._client = httpx.Client(
            transport=mock_transport(lambda _request: _stream_response(wire))
        )
        owner_ref = weakref.ref(owner)
        return owner.chat(model="model", messages=[], stream=True), owner_ref

    stream, owner_ref = make_stream()
    gc.collect()

    assert owner_ref() is not None
    assert list(stream) == [{"id": "one"}]
    assert stream.closed is True
    gc.collect()
    assert owner_ref() is None
