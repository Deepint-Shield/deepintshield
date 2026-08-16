"""Lazy synchronous chat-completion streaming over Server-Sent Events."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Iterator, Mapping, NoReturn

import httpx

from .errors import DeepintShieldError, ErrorCode, _annotate_error

if TYPE_CHECKING:
    from .client import DeepintShield


_SKIP = object()
_END = object()
_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_STREAM_EVENT_BYTES = 1024 * 1024


class _SSELineError(Exception):
    """Internal bounded parser failure that never retains response content."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _decode_sse_line(raw: bytearray) -> str | None:
    """Decode a UTF-8 SSE line without leaking decoder input through errors."""
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _bounded_sse_lines(
    response: httpx.Response,
    *,
    limit: int = _MAX_STREAM_EVENT_BYTES,
) -> Iterator[str]:
    """Yield SSE lines while bounding the undecoded line buffer.

    ``httpx.Response.iter_lines`` buffers until a delimiter and therefore lets
    an untrusted peer grow one line without bound. SSE permits LF, CRLF, and CR
    delimiters, so this parser handles each form over bounded byte chunks.
    """
    line = bytearray()
    after_cr = False
    # Preserve httpx's native chunk boundaries. Supplying ``chunk_size`` makes
    # httpx coalesce short network chunks, which can read ahead past a complete
    # event and break early-close/timeout semantics.
    for chunk in response.iter_bytes():
        for value in chunk:
            if after_cr:
                after_cr = False
                if value == 0x0A:
                    continue
            if value in (0x0A, 0x0D):
                decoded = _decode_sse_line(line)
                line.clear()
                if decoded is None:
                    raise _SSELineError("invalid_utf8")
                yield decoded
                after_cr = value == 0x0D
                continue
            line.append(value)
            if len(line) > limit:
                line.clear()
                raise _SSELineError("event_too_large")
    if line:
        decoded = _decode_sse_line(line)
        line.clear()
        if decoded is None:
            raise _SSELineError("invalid_utf8")
        yield decoded


def _bounded_response_payload(
    response: httpx.Response,
    *,
    limit: int = _MAX_ERROR_BODY_BYTES,
) -> tuple[dict[str, Any], bool]:
    """Read at most ``limit`` bytes from a non-successful stream and close it."""
    body = bytearray()
    truncated = False
    try:
        try:
            for chunk in response.iter_bytes(chunk_size=8192):
                remaining = (limit + 1) - len(body)
                if remaining <= 0:
                    truncated = True
                    break
                body.extend(chunk[:remaining])
                if len(body) > limit:
                    truncated = True
                    del body[limit:]
                    break
        except httpx.HTTPError:
            # The HTTP status is already authoritative. If its diagnostic body
            # cannot be read, preserve the status and use the generic message.
            pass
    finally:
        with suppress(Exception):
            response.close()

    if not body:
        return {}, truncated
    try:
        decoded = json.loads(bytes(body))
    except (ValueError, UnicodeDecodeError, RecursionError):
        # Python's integer-string and nesting limits raise ValueError and
        # RecursionError rather than JSONDecodeError. An HTTP error response
        # must still reach the caller through the stable SDK error contract.
        decoded = {"raw": bytes(body).decode("utf-8", errors="replace")}
    if isinstance(decoded, dict):
        return decoded, truncated
    return {"data": decoded}, truncated


class ChatCompletionStream(Iterator[dict[str, Any]]):
    """Single-consumer, closeable iterator over chat SSE JSON objects.

    A successful response body is consumed only as the caller requests chunks.
    Use the object as a context manager, consume it fully, or call :meth:`close`
    to release the response deterministically.
    """

    def __init__(
        self,
        response: httpx.Response,
        owner: "DeepintShield",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._response = response
        # Keep a temporary-expression client alive for the stream lifetime.
        # The owner does not retain this stream, so this creates no cycle.
        self._owner: DeepintShield | None = owner
        self._lines = _bounded_sse_lines(response)
        self._details = dict(details or {})
        self._data_lines: list[str] = []
        self._data_bytes = 0
        self._event_name = ""
        self._first_line = True
        self._terminal_received = False
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the underlying streaming response has been released."""
        return self._closed or self._response.is_closed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # Returning a tiny generator gives ``for`` loops a finally boundary:
        # breaking or raising out of the loop releases the response instead of
        # leaving it open until the stream object itself is discarded.
        try:
            while True:
                try:
                    yield self.__next__()
                except StopIteration:
                    return
        finally:
            self.close()

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        owner = self._owner
        if owner is None or bool(getattr(owner, "_deepintshield_closed", False)):
            self.close()
            raise DeepintShieldError(
                "DeepintShield client is closed",
                code=ErrorCode.CLIENT_CLOSED,
                details=self._details,
            )
        if self._response.is_closed:
            self._raise_premature_eof()

        while True:
            line_error: str | None = None
            try:
                line = next(self._lines)
            except StopIteration:
                if self._data_lines:
                    event = self._decode_pending_event()
                    if event is _END:
                        self.close()
                        raise StopIteration
                self._raise_premature_eof()
            except httpx.TimeoutException as exc:
                self.close()
                raise _annotate_error(
                    exc,
                    ErrorCode.TRANSPORT_TIMEOUT,
                    details=self._details,
                )
            except httpx.HTTPError as exc:
                self.close()
                raise _annotate_error(
                    exc,
                    ErrorCode.TRANSPORT_ERROR,
                    details=self._details,
                )
            except _SSELineError as exc:
                line_error = exc.reason
            except BaseException:
                # Includes cooperative cancellation/KeyboardInterrupt and any
                # unexpected decoder failure: never strand the connection.
                self.close()
                raise

            if line_error is not None:
                self._raise_invalid_event(line_error)

            if self._first_line:
                line = line.removeprefix("\ufeff")
                self._first_line = False
            if line == "":
                event = self._decode_pending_event()
                if event is _SKIP:
                    continue
                if event is _END:
                    self.close()
                    raise StopIteration
                return event
            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                next_size = self._data_bytes + len(value.encode("utf-8"))
                if self._data_lines:
                    next_size += 1
                if next_size > _MAX_STREAM_EVENT_BYTES:
                    self._raise_invalid_event("event_too_large")
                self._data_lines.append(value)
                self._data_bytes = next_size
            elif field == "event":
                self._event_name = value
            # SSE ``id``/``retry`` and extension fields do not carry chunks.

    def _decode_pending_event(self) -> dict[str, Any] | object:
        data_lines = self._data_lines
        event_name = self._event_name
        self._data_lines = []
        self._data_bytes = 0
        self._event_name = ""
        if not data_lines:
            return _SKIP

        data = "\n".join(data_lines)
        data_lines.clear()
        if not data.strip():
            return _SKIP
        if data.strip() == "[DONE]":
            self._terminal_received = True
            return _END

        decode_failed = False
        try:
            decoded = json.loads(data)
        except (ValueError, RecursionError):
            # Raise only after leaving the decoder handler. JSONDecodeError
            # retains its input in ``.doc`` and stream data may be sensitive.
            # Python's integer-string and nesting guards raise sibling errors,
            # which are malformed events under the same public contract.
            decoded = None
            decode_failed = True
        except BaseException:
            # Cooperative cancellation or an unexpected decoder failure must
            # never strand the streaming response.
            self.close()
            raise
        if decode_failed or not isinstance(decoded, dict):
            data = ""
            decoded = None
            self.close()
            raise DeepintShieldError(
                "Chat stream returned a malformed JSON event",
                code=ErrorCode.CHAT_STREAM_INVALID_EVENT,
                details={**self._details, "event": event_name or "message"},
            ) from None

        is_error = (
            event_name.strip().lower() == "error"
            or ("error" in decoded and decoded.get("error") is not None)
            or str(decoded.get("type") or "").strip().lower() == "error"
        )
        if is_error:
            self.close()
            # Gateway streams use both nested ``error.code`` and top-level
            # ``code``/``error_code`` shapes. Route every structured error
            # event through the same normalization precedence as HTTP errors.
            raise DeepintShieldError.from_response(
                200,
                decoded,
                fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
                details={**self._details, "event": event_name or "error"},
            ) from None
        return decoded

    def _raise_invalid_event(self, reason: str) -> NoReturn:
        self.close()
        raise DeepintShieldError(
            "Chat stream returned a malformed event",
            code=ErrorCode.CHAT_STREAM_INVALID_EVENT,
            details={
                **self._details,
                "reason": reason,
                "max_event_bytes": _MAX_STREAM_EVENT_BYTES,
            },
        ) from None

    def _raise_premature_eof(self) -> None:
        self.close()
        raise DeepintShieldError(
            "Chat stream ended before the terminal [DONE] event",
            code=ErrorCode.CHAT_STREAM_INVALID_EVENT,
            details={**self._details, "reason": "premature_eof"},
        ) from None

    def close(self) -> None:
        """Release the response. Safe to call repeatedly."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self._data_lines.clear()
        self._data_bytes = 0
        self._owner = None
        with suppress(Exception):
            self._response.close()

    def __enter__(self) -> "ChatCompletionStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        with suppress(Exception):
            self.close()


__all__ = ["ChatCompletionStream"]
