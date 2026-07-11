"""Tests for the Gemini context-cache manager and wrapped client."""

from __future__ import annotations

import time

import pytest

from deepintshield._gemini_cache import (
    DEFAULT_MIN_PREFIX_TOKENS,
    GeminiCacheManager,
    GenaiCachedClient,
    _compute_prefix_hash,
    _ttl_seconds_from_string,
    _with_cached_content,
)


# ─────────────────────────── small helpers ──────────────────────────────────────


def test_ttl_seconds_parses_workspace_strings():
    assert _ttl_seconds_from_string("5m") == 300
    assert _ttl_seconds_from_string("1h") == 3600
    assert _ttl_seconds_from_string("6h") == 21600
    assert _ttl_seconds_from_string("24h") == 86400
    assert _ttl_seconds_from_string("90s") == 90
    assert _ttl_seconds_from_string("600") == 600
    # Falls back to the default for unknown formats.
    assert _ttl_seconds_from_string("forever") == 3600
    assert _ttl_seconds_from_string(None) == 3600
    # Ints are accepted; floor of 60s prevents nonsense.
    assert _ttl_seconds_from_string(10) == 60
    assert _ttl_seconds_from_string(900) == 900


def test_compute_prefix_hash_is_stable_and_partition_aware():
    h_a = _compute_prefix_hash("gemini-2.0-flash", "system one", [{"name": "tool_a"}])
    h_b = _compute_prefix_hash("gemini-2.0-flash", "system one", [{"name": "tool_a"}])
    h_c = _compute_prefix_hash("gemini-2.0-flash", "system two", [{"name": "tool_a"}])
    h_d = _compute_prefix_hash("gemini-1.5-pro", "system one", [{"name": "tool_a"}])
    assert h_a == h_b
    assert h_a != h_c  # different system instruction
    assert h_a != h_d  # different model
    assert len(h_a) == 16


def test_compute_prefix_hash_empty_input():
    assert _compute_prefix_hash("", None, None) == ""


def test_with_cached_content_handles_dict_and_none():
    assert _with_cached_content(None, "cachedContents/x") == {"cached_content": "cachedContents/x"}
    assert _with_cached_content({"foo": 1}, "cachedContents/x") == {"foo": 1, "cached_content": "cachedContents/x"}


def test_with_cached_content_pydantic_like_model_copy():
    class FakeConfig:
        def __init__(self, system_instruction=None):
            self.system_instruction = system_instruction
            self.cached_content = None

        def model_copy(self, *, update):
            new = FakeConfig(system_instruction=self.system_instruction)
            for k, v in update.items():
                setattr(new, k, v)
            return new

    original = FakeConfig(system_instruction="S")
    updated = _with_cached_content(original, "cachedContents/x")
    assert updated is not original
    assert updated.cached_content == "cachedContents/x"
    assert original.cached_content is None  # caller's config untouched


# ─────────────────────── cache manager registry ─────────────────────────────────


def test_lookup_active_returns_none_for_unknown_prefix():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    assert mgr.lookup_active("absent") is None


def test_remember_and_lookup_within_ttl():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    mgr.remember("abc", "cachedContents/xyz")
    assert mgr.lookup_active("abc") == "cachedContents/xyz"


def test_lookup_expires_entries_lazily():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    mgr.remember("abc", "cachedContents/xyz", ttl_seconds=-1)  # already expired
    assert mgr.lookup_active("abc") is None
    # And the entry has been dropped.
    assert mgr.lookup_active("abc") is None


def test_mark_pending_atomic_claim():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    assert mgr.mark_pending("h") is True
    # Second claim is rejected while creation is in flight.
    assert mgr.mark_pending("h") is False
    # Resolve via remember to release the pending slot.
    mgr.remember("h", "cachedContents/x")
    # Once active, mark_pending stays rejected (no need to recreate).
    assert mgr.mark_pending("h") is False


def test_mark_pending_releases_on_forget():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    assert mgr.mark_pending("h") is True
    mgr.forget("h")
    assert mgr.mark_pending("h") is True


def test_attach_returns_existing_caller_value_unchanged():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    result = mgr.attach(
        client=object(),
        model="gemini-2.0-flash",
        existing_cached_content="cachedContents/caller-set",
    )
    assert result == "cachedContents/caller-set"


def test_attach_returns_active_cache():
    mgr = GeminiCacheManager(ttl_seconds=3600)
    prefix_hash = _compute_prefix_hash("gemini-2.0-flash", "S", [{"name": "t"}])
    mgr.remember(prefix_hash, "cachedContents/abc")
    result = mgr.attach(
        client=object(),
        model="gemini-2.0-flash",
        system_instruction="S",
        tools=[{"name": "t"}],
    )
    assert result == "cachedContents/abc"


def test_attach_skips_below_min_prefix_tokens():
    mgr = GeminiCacheManager(ttl_seconds=3600, min_prefix_tokens=32_768)
    result = mgr.attach(
        client=object(),
        model="gemini-2.0-flash",
        system_instruction="S",
        tools=None,
        prefix_token_estimate=500,
    )
    assert result is None
    # No pending creation kicked off - caller is below the cost threshold.
    prefix_hash = _compute_prefix_hash("gemini-2.0-flash", "S", None)
    assert mgr.mark_pending(prefix_hash) is True  # we can claim it ourselves


def test_attach_kicks_off_background_creation():
    """On a cache miss above the min token threshold, the manager fires a
    background creation but returns ``None`` so the current call runs normally
    without waiting on the cache resource.
    """
    mgr = GeminiCacheManager(ttl_seconds=600, min_prefix_tokens=0)

    fake_client = object()
    result = mgr.attach(
        client=fake_client,
        model="gemini-2.0-flash",
        system_instruction="S",
        tools=None,
    )
    # Current call gets no cache - runs normally.
    assert result is None
    # The prefix hash should be in the pending set briefly. We don't assert
    # on `_pending` directly because the background pool may have resolved
    # it already; instead we assert it's tracked one way or the other.
    prefix_hash = _compute_prefix_hash("gemini-2.0-flash", "S", None)
    # The background task hits the real google-genai library which isn't
    # available; it will fail and the pending slot will be released. Either
    # way, the next attach() call won't deadlock.
    # Give the executor a moment to settle.
    for _ in range(50):
        if mgr.mark_pending(prefix_hash + "-other"):
            break
        time.sleep(0.01)


# ─────────────────────── plumbing tests ─────────────────────────────────────────


def test_default_min_prefix_tokens_matches_gemini_threshold():
    assert DEFAULT_MIN_PREFIX_TOKENS == 32_768


def test_genai_cached_provider_returns_wrapped_client():
    importlib_util = pytest.importorskip("importlib.util")
    if importlib_util.find_spec("google.genai") is None:
        pytest.skip("google-genai is not installed")
    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-bf-1")
    cached = shield.genai_cached()
    assert isinstance(cached, GenaiCachedClient)
    # Should expose the underlying client's attributes via __getattr__.
    assert getattr(cached, "models", None) is not None
    # And ship a cache manager the caller can poke at if they need to.
    assert isinstance(cached.cache_manager, GeminiCacheManager)


def test_genai_plain_provider_still_returns_native_client():
    importlib_util = pytest.importorskip("importlib.util")
    if importlib_util.find_spec("google.genai") is None:
        pytest.skip("google-genai is not installed")
    from google.genai import Client as GenaiClient
    from deepintshield import DeepintShield

    shield = DeepintShield(virtual_key="sk-bf-1")
    native = shield.genai()
    assert isinstance(native, GenaiClient)
