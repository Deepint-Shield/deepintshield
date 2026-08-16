"""Tests for the prompt-cache injection hook used by the OpenAI and Anthropic providers."""

from __future__ import annotations

import json

import httpx
import pytest

from deepintshield._prompt_cache import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    _compute_prefix_hash,
    _inject_anthropic,
    _inject_openai,
    build_request_hook,
)


def _make_request(method: str, url: str, *, body: dict | None = None, content_type: str = "application/json") -> httpx.Request:
    """Build a real httpx.Request so we exercise the same code path as the SDK."""
    if body is not None:
        content = json.dumps(body).encode("utf-8")
        return httpx.Request(method, url, content=content, headers={"content-type": content_type})
    return httpx.Request(method, url)


# ─────────────────────── anthropic in-memory injection ──────────────────────────


def test_inject_anthropic_marks_system_string():
    body = {"system": "You are helpful.", "messages": []}
    _inject_anthropic(body, ttl="5m", breakpoints=("system",))
    assert body["system"] == [
        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
    ]


def test_inject_anthropic_marks_system_blocks_last_only():
    body = {
        "system": [
            {"type": "text", "text": "block-a"},
            {"type": "text", "text": "block-b"},
        ],
        "messages": [],
    }
    _inject_anthropic(body, ttl="5m", breakpoints=("system",))
    assert "cache_control" not in body["system"][0]
    assert body["system"][1]["cache_control"] == {"type": "ephemeral"}


def test_inject_anthropic_marks_last_tool_only():
    body = {
        "system": "ignored",
        "tools": [
            {"name": "tool_a", "input_schema": {}},
            {"name": "tool_b", "input_schema": {}},
        ],
    }
    _inject_anthropic(body, ttl="5m", breakpoints=("tools",))
    assert "cache_control" not in body["tools"][0]
    assert body["tools"][1]["cache_control"] == {"type": "ephemeral"}


def test_inject_anthropic_respects_caller_cache_control():
    body = {"system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}
    _inject_anthropic(body, ttl="5m", breakpoints=("system",))
    # Caller's marker wins.
    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_inject_anthropic_emits_ttl_only_when_non_default():
    body = {"system": "x"}
    _inject_anthropic(body, ttl="1h", breakpoints=("system",))
    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ────────────────────────── openai prefix hash ──────────────────────────────────


def test_compute_prefix_hash_is_stable_for_identical_system_and_tools():
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    assert _compute_prefix_hash(messages, tools) == _compute_prefix_hash(messages, tools)


def test_compute_prefix_hash_excludes_user_messages():
    base_messages = [{"role": "system", "content": "S"}]
    hash_one = _compute_prefix_hash(base_messages + [{"role": "user", "content": "Q1"}], None)
    hash_two = _compute_prefix_hash(base_messages + [{"role": "user", "content": "Q2"}], None)
    assert hash_one == hash_two


def test_compute_prefix_hash_returns_empty_without_static_prefix():
    assert _compute_prefix_hash([{"role": "user", "content": "hi"}], None) == ""


def test_inject_openai_sets_prompt_cache_key():
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
        ],
    }
    _inject_openai(body)
    assert "prompt_cache_key" in body
    assert len(body["prompt_cache_key"]) == 16


def test_inject_openai_respects_caller_key():
    body = {"messages": [{"role": "system", "content": "S"}], "prompt_cache_key": "caller-set"}
    _inject_openai(body)
    assert body["prompt_cache_key"] == "caller-set"


def test_inject_openai_noop_without_system_messages():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    _inject_openai(body)
    assert "prompt_cache_key" not in body


# ────────────────────────── hook fast paths ─────────────────────────────────────


def test_hook_skips_non_post():
    hook = build_request_hook(PROVIDER_OPENAI)
    req = _make_request("GET", "https://example.com/v1/models")
    hook(req)  # must not raise
    assert req.content == b""


def test_hook_skips_non_json():
    hook = build_request_hook(PROVIDER_OPENAI)
    req = _make_request(
        "POST",
        "https://example.com/v1/audio",
        body={"messages": [{"role": "system", "content": "S"}]},
        content_type="multipart/form-data",
    )
    original = req.content
    hook(req)
    assert req.content == original  # body untouched


def test_hook_mutates_openai_post_body():
    hook = build_request_hook(PROVIDER_OPENAI)
    req = _make_request(
        "POST",
        "https://example.com/openai/v1/chat/completions",
        body={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )
    hook(req)
    decoded = json.loads(req.content)
    assert "prompt_cache_key" in decoded
    # content-length stays consistent with the new body
    assert req.headers["content-length"] == str(len(req.content))


def test_hook_mutates_anthropic_post_body():
    hook = build_request_hook(PROVIDER_ANTHROPIC)
    req = _make_request(
        "POST",
        "https://example.com/anthropic/v1/messages",
        body={
            "model": "claude-3-5-sonnet-latest",
            "system": "You are helpful.",
            "tools": [{"name": "lookup", "input_schema": {}}],
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    hook(req)
    decoded = json.loads(req.content)
    # System promoted to block + marked
    assert isinstance(decoded["system"], list)
    assert decoded["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # Tool marked
    assert decoded["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_hook_handles_malformed_json_gracefully():
    hook = build_request_hook(PROVIDER_OPENAI)
    req = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions",
        content=b"not-valid-json{",
        headers={"content-type": "application/json"},
    )
    hook(req)  # should not raise
    assert req.content == b"not-valid-json{"


def test_hook_no_op_when_body_already_has_prompt_cache_key():
    hook = build_request_hook(PROVIDER_OPENAI)
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
        "prompt_cache_key": "user-set",
    }
    req = _make_request("POST", "https://example.com/openai/v1/chat/completions", body=body)
    hook(req)
    decoded = json.loads(req.content)
    assert decoded["prompt_cache_key"] == "user-set"


# ─────────────────────── plumbing tests ─────────────────────────────────────────


def test_openai_provider_attaches_hook_by_default():
    importlib_util = pytest.importorskip("importlib.util")
    if importlib_util.find_spec("openai") is None:
        pytest.skip("openai is not installed")
    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-ds-1")
    client = shield.openai()
    # Internal: openai stores its httpx client at ._client._client (sync transport)
    http_client = getattr(client, "_client", None)
    assert http_client is not None
    # The httpx.Client's request event hooks should contain our injection hook
    hooks = getattr(http_client, "_event_hooks", {})
    request_hooks = hooks.get("request") or []
    assert any(callable(h) for h in request_hooks)


def test_anthropic_provider_attaches_hook_by_default():
    importlib_util = pytest.importorskip("importlib.util")
    if importlib_util.find_spec("anthropic") is None:
        pytest.skip("anthropic is not installed")
    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-ds-1")
    client = shield.anthropic()
    http_client = getattr(client, "_client", None)
    assert http_client is not None
    hooks = getattr(http_client, "_event_hooks", {})
    request_hooks = hooks.get("request") or []
    assert any(callable(h) for h in request_hooks)
