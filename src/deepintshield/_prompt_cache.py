"""Provider-native prompt cache injection for the DeepintShield SDK.

This module ships a thin ``httpx`` request hook that injects cache hints into
outbound requests heading for the DeepintShield gateway. The intent is to make
provider-native prompt caching automatic for SDK users without requiring any
code changes on the caller side and without exposing a per-call parameter
(see the gateway's workspace-level switch in Caching settings for control).

Performance contract
--------------------
The hook runs synchronously inside the httpx event loop for every outbound
request. To honor the "zero added latency" requirement:

* Non-POST / non-JSON requests short-circuit before any parsing.
* The injection is O(messages + tools), no allocations beyond the marker dict.
* httpx.Client is fully thread-safe; the hook is stateless, so concurrent
  requests use the same hook instance without locks.

When the gateway's workspace switch disables prompt caching, it strips the
markers we emit here (see plugins/semanticcache/promptcache.go). Net result on
disable: ~50us of wasted work per request, no provider-side impact.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Callable, Iterable

import httpx


PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"


@lru_cache(maxsize=8)
def _request_byte_stream_type(request_type: type) -> type:
    """Keep rewritten request streams in the HTTP library that owns the request."""
    for cls in request_type.__mro__:
        package = cls.__module__.partition(".")[0]
        if package == "httpx2":
            # Optional: only imported for a request from an installed httpx2 SDK.
            from httpx2 import ByteStream
            return ByteStream
        if package == "httpx":
            return httpx.ByteStream
    return httpx.ByteStream

# Anthropic accepts up to 4 cache_control markers per request. We place them in
# priority order: system → tools → last static user/assistant block.
_DEFAULT_BREAKPOINTS: tuple[str, ...] = ("system", "tools")


def _build_cache_marker(ttl: str) -> dict[str, str]:
    """Anthropic's cache_control marker. ``ttl`` is "5m" (default) or "1h"."""
    marker: dict[str, str] = {"type": "ephemeral"}
    if ttl and ttl != "5m":
        marker["ttl"] = ttl
    return marker


def _mark_anthropic_block_cacheable(block: Any, marker: dict[str, str]) -> bool:
    """Attach the cache_control marker to a single content block if missing.

    Returns True if a marker was added (so callers can stop after the budget of
    4 markers is exhausted).
    """
    if not isinstance(block, dict):
        return False
    if "cache_control" in block:
        # Caller already marked this block - respect it.
        return False
    block["cache_control"] = marker
    return True


def _inject_anthropic(
    body: dict[str, Any],
    *,
    ttl: str,
    breakpoints: Iterable[str],
) -> None:
    """Mark the static prefix unless the caller already configured caching.

    Explicit breakpoints and automatic (top-level) caching share a four-slot
    limit. Adding our own markers can also put a short TTL before a caller's
    longer TTL, which Anthropic rejects. Preserve caller-managed caching in
    full rather than augmenting a valid request with incompatible markers.
    """
    if "cache_control" in body:
        return
    for section in ("system", "tools", "messages"):
        blocks = body.get(section)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if "cache_control" in block:
                return
            content = block.get("content") if section == "messages" else None
            if isinstance(content, list) and any(
                isinstance(item, dict) and "cache_control" in item for item in content
            ):
                return

    marker = _build_cache_marker(ttl)
    requested = {b for b in breakpoints if b in {"system", "tools", "large_blocks"}}

    if "system" in requested:
        system = body.get("system")
        if isinstance(system, list) and system:
            # System is already a list of blocks - mark the last one.
            _mark_anthropic_block_cacheable(system[-1], marker)
        elif isinstance(system, str) and system.strip():
            # Promote string-form system prompts into block form so we can attach
            # cache_control. Anthropic accepts both shapes.
            body["system"] = [
                {"type": "text", "text": system, "cache_control": marker},
            ]

    if "tools" in requested:
        tools = body.get("tools")
        if isinstance(tools, list) and tools:
            _mark_anthropic_block_cacheable(tools[-1], marker)

    if "large_blocks" in requested:
        # Mark the last content block of the last "user" message if it's
        # text and large enough to be worth caching.
        messages = body.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if not isinstance(msg, dict) or msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if not isinstance(content, list) or not content:
                    break
                last = content[-1]
                if isinstance(last, dict) and last.get("type") == "text":
                    text = last.get("text") or ""
                    if isinstance(text, str) and len(text) >= 12000:  # ~4096 tokens
                        _mark_anthropic_block_cacheable(last, marker)
                break


def _compute_prefix_hash(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> str:
    """Stable 16-char SHA-256 prefix of (system messages + tool defs).

    OpenAI uses this as a cache partition key - same hash → same cache bucket.
    Identical prefixes from different conversations therefore share cache
    entries. Returns "" when there's no meaningful static prefix to hash.
    """
    static_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            try:
                static_parts.append(json.dumps(msg, sort_keys=True, separators=(",", ":")))
            except (TypeError, ValueError):
                continue
    if tools:
        try:
            static_parts.append(json.dumps(tools, sort_keys=True, separators=(",", ":")))
        except (TypeError, ValueError):
            pass
    if not static_parts:
        return ""
    digest = hashlib.sha256("|".join(static_parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _inject_openai(body: dict[str, Any]) -> None:
    """Set ``prompt_cache_key`` on an OpenAI chat completions body.

    OpenAI's automatic prompt caching fires on any prefix >= 1024 tokens
    regardless of this field; ``prompt_cache_key`` is an opt-in partition hint
    that improves hit rate when the same client has many concurrent sessions.
    Skipped if the caller already set the key.
    """
    if "prompt_cache_key" in body:
        return
    # The OpenAI-compatible gateway serves every provider. An OpenAI-specific
    # hint must not leak into requests routed to another provider or a custom
    # deployment. Keep model IDs opaque; OpenAI fine-tunes may contain slashes.
    provider = body.get("provider")
    if provider is not None and provider != PROVIDER_OPENAI:
        return
    model = body.get("model")
    if isinstance(model, str) and "/" in model and not model.startswith(("openai/", "ft:")):
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    tools_raw = body.get("tools")
    tools = tools_raw if isinstance(tools_raw, list) else None
    prefix_hash = _compute_prefix_hash(messages, tools)
    if prefix_hash:
        body["prompt_cache_key"] = prefix_hash


def build_request_hook(
    provider: str,
    *,
    anthropic_ttl: str = "5m",
    breakpoints: Iterable[str] = _DEFAULT_BREAKPOINTS,
) -> Callable[[httpx.Request], None]:
    """Build a request event hook that injects prompt-cache markers.

    The returned callable is safe to attach to ``httpx.Client(event_hooks={...})``
    and is reused across requests without per-call allocations beyond the
    in-place mutation of the body dict.
    """
    breakpoints_tuple = tuple(breakpoints)

    def hook(request: httpx.Request) -> None:
        if request.method != "POST":
            return
        ct = request.headers.get("content-type", "")
        if "json" not in ct.lower():
            return

        raw = request.content
        if not raw:
            return
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(body, dict):
            return

        if provider == PROVIDER_ANTHROPIC:
            _inject_anthropic(body, ttl=anthropic_ttl, breakpoints=breakpoints_tuple)
        elif provider == PROVIDER_OPENAI:
            _inject_openai(body)
        else:
            return

        # Re-serialize the mutated body. httpx checks content-length
        # headers, so we keep them consistent.
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        request.headers["content-length"] = str(len(encoded))
        # httpx routes the actual transmitted bytes through `request.stream`
        # - a single-shot byte iterator built when the Request was first
        # constructed. Updating only `_content` leaves the stream pointing
        # at the original bytes, so on the wire httpx emits the *old* body
        # while advertising the *new* content-length. The result on the
        # transport layer is `LocalProtocolError: Too little data for
        # declared Content-Length` (h11 hits EOF on the iterator before
        # the announced byte count). Reattach the stream so the new body
        # is what actually gets sent.
        request._content = encoded  # type: ignore[attr-defined]
        request.stream = _request_byte_stream_type(type(request))(encoded)

    return hook


def build_http_client(
    provider: str,
    *,
    anthropic_ttl: str = "5m",
    breakpoints: Iterable[str] = _DEFAULT_BREAKPOINTS,
    timeout: float = 60.0,
) -> httpx.Client:
    """Synchronous httpx client with the prompt-cache hook attached."""
    hook = build_request_hook(
        provider,
        anthropic_ttl=anthropic_ttl,
        breakpoints=breakpoints,
    )
    return httpx.Client(timeout=timeout, event_hooks={"request": [hook]})


def build_async_http_client(
    provider: str,
    *,
    anthropic_ttl: str = "5m",
    breakpoints: Iterable[str] = _DEFAULT_BREAKPOINTS,
    timeout: float = 60.0,
) -> httpx.AsyncClient:
    """Async httpx client variant. Wraps the same sync hook in an async shim so
    we keep a single source of truth for the injection logic.
    """
    sync_hook = build_request_hook(
        provider,
        anthropic_ttl=anthropic_ttl,
        breakpoints=breakpoints,
    )

    async def async_hook(request: httpx.Request) -> None:
        sync_hook(request)

    return httpx.AsyncClient(timeout=timeout, event_hooks={"request": [async_hook]})
