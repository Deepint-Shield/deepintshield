"""Gemini context-cache management for the DeepintShield SDK.

Google Gemini doesn't have inline cache markers like Anthropic - instead it
exposes an explicit ``cachedContents`` resource with its own lifecycle:

1. Client calls ``client.caches.create(...)`` with the static portion of the
   prompt (system instruction, tools, large reference docs) and a TTL.
2. Google returns a resource name like ``cachedContents/abc123def`` and an
   expiration time.
3. Subsequent ``generate_content`` calls pass ``cached_content=<name>`` in
   their config; Google reuses the KV state for the prefix and applies the
   model's current cached-input pricing.

This module wraps that lifecycle so customers don't have to manage cache
resources by hand. Design choices for bounded request-path overhead and
parallel use:

* **Per-process registry** - a thread-safe dict of ``prefix_hash → (name,
  expires_at_ms)``. Lookups are O(1), no I/O.
* **Fire-and-forget creation** - the first request that sees a prefix doesn't
  wait for the cache resource to exist. It runs through the provider as a
  normal call, while a background thread creates the cache for the *next*
  call. Resource creation is kept off the first call's provider critical path;
  the local hash/lookup/scheduling work is still measurable.
* **Lazy TTL eviction** - expired entries are dropped at lookup time. No
  background sweeper, no extra timers.
* **Workspace switch respected** - the gateway strips ``cached_content`` from
  outbound requests when the workspace has prompt caching disabled or
  ``google`` not in the provider allow-list, so the SDK's injection becomes
  a no-op end-to-end.
* **Caller-aware** - if the caller already passed ``cached_content`` on a
  call's config, we leave it alone.

The SDK exposes ``shield.gemini_cache_manager()`` plus an opt-in wrapped
``shield.genai_cached()`` client. Cache storage prices and minimum eligible
prefix sizes vary by model and can change, so applications should validate
reuse frequency against the provider's current requirements and pricing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Conservative default heuristic for deciding whether to create a cache.
# Provider eligibility and break-even points vary by model and pricing; callers
# can override this value for their measured workload.
DEFAULT_MIN_PREFIX_TOKENS = 32_768

# Default TTL on the Gemini side. Customers override this via the workspace
# config field `prompt_cache_google_ttl` or by passing ``ttl`` directly to
# the manager.
DEFAULT_TTL_SECONDS = 3600  # "1h"

# Background pool for cache resource creation. One thread per process is plenty
# because creation is rare (once per unique prefix per TTL window) and the
# work is just a single HTTP call to Google.
_CREATION_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dis-gemini-cache")


def _ttl_seconds_from_string(ttl: str | int | None) -> int:
    """Parse a workspace-style TTL string ("5m", "1h", "6h", "24h") into seconds.

    Falls through to ``DEFAULT_TTL_SECONDS`` on unknown values so misconfigured
    workspaces never throw at request time - they just use the default.
    """
    if ttl is None:
        return DEFAULT_TTL_SECONDS
    if isinstance(ttl, int):
        return max(int(ttl), 60)
    s = ttl.strip().lower()
    if s.endswith("h") and s[:-1].isdigit():
        return int(s[:-1]) * 3600
    if s.endswith("m") and s[:-1].isdigit():
        return int(s[:-1]) * 60
    if s.endswith("s") and s[:-1].isdigit():
        return int(s[:-1])
    if s.isdigit():
        return int(s)
    return DEFAULT_TTL_SECONDS


def _compute_prefix_hash(
    model: str,
    system_instruction: Any,
    tools: Any,
) -> str:
    """Stable 16-char hash of (model + system + tools).

    Returns "" when nothing meaningful is available to cache. The hash is the
    registry key, so identical prefixes share a single cache resource even
    across many concurrent calls in the same process.
    """
    parts: list[str] = [model or ""]
    if system_instruction is not None:
        try:
            parts.append(json.dumps(system_instruction, sort_keys=True, separators=(",", ":"), default=str))
        except (TypeError, ValueError):
            parts.append(str(system_instruction))
    if tools:
        try:
            parts.append(json.dumps(tools, sort_keys=True, separators=(",", ":"), default=str))
        except (TypeError, ValueError):
            parts.append(str(tools))
    if len(parts) == 1 and not parts[0]:
        return ""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class GeminiCacheManager:
    """Per-process registry of Gemini ``cachedContents`` resources.

    Thread-safe. Use one instance per ``DeepintShield`` client; the wrapped
    ``shield.genai_cached()`` constructor builds one for you.
    """

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS, min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS):
        self._ttl_seconds = max(int(ttl_seconds), 60)
        self._min_prefix_tokens = max(int(min_prefix_tokens), 0)
        self._lock = threading.Lock()
        # prefix_hash -> (cached_content_name, expires_at_unix_ms)
        self._entries: dict[str, tuple[str, int]] = {}
        # prefix_hash -> True while a creation is in flight, so we don't
        # double-submit while a cache is being built.
        self._pending: set[str] = set()

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def lookup_active(self, prefix_hash: str) -> str | None:
        """Return the active ``cachedContents/...`` name for the prefix, or None.

        O(1) under a single lock. Expired entries are dropped lazily here so
        no background sweeper is required.
        """
        if not prefix_hash:
            return None
        now_ms = int(time.time() * 1000)
        with self._lock:
            entry = self._entries.get(prefix_hash)
            if entry is None:
                return None
            name, expires_at_ms = entry
            if now_ms >= expires_at_ms:
                del self._entries[prefix_hash]
                return None
            return name

    def remember(self, prefix_hash: str, name: str, *, ttl_seconds: int | None = None) -> None:
        """Stash a freshly-created cache resource."""
        if not prefix_hash or not name:
            return
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        expires_at_ms = int(time.time() * 1000) + int(ttl) * 1000
        with self._lock:
            self._entries[prefix_hash] = (name, expires_at_ms)
            self._pending.discard(prefix_hash)

    def forget(self, prefix_hash: str) -> None:
        with self._lock:
            self._entries.pop(prefix_hash, None)
            self._pending.discard(prefix_hash)

    def mark_pending(self, prefix_hash: str) -> bool:
        """Atomically claim the right to create the cache for ``prefix_hash``.

        Returns True if the caller should proceed with creation; False if
        another thread is already creating it.
        """
        if not prefix_hash:
            return False
        with self._lock:
            if prefix_hash in self._pending or prefix_hash in self._entries:
                return False
            self._pending.add(prefix_hash)
            return True

    def schedule_creation(
        self,
        prefix_hash: str,
        creator: "callable[[], tuple[str, int] | None]",
    ) -> None:
        """Fire-and-forget creation. ``creator`` must return ``(name, ttl_seconds)``
        on success or None on failure (in which case the pending slot is
        released so a later request can retry).
        """
        if not self.mark_pending(prefix_hash):
            return

        def _run() -> None:
            try:
                result = creator()
            except Exception:  # pragma: no cover - swallow background errors
                result = None
            if result is None:
                self.forget(prefix_hash)
                return
            name, ttl = result
            self.remember(prefix_hash, name, ttl_seconds=ttl)

        _CREATION_POOL.submit(_run)

    # ─────────────────────── higher-level convenience ──────────────────────────

    def attach(
        self,
        client: Any,
        *,
        model: str,
        system_instruction: Any = None,
        tools: Any = None,
        contents: Any = None,
        existing_cached_content: str | None = None,
        prefix_token_estimate: int | None = None,
    ) -> str | None:
        """Return the ``cached_content`` resource name to attach to this call.

        Behavior:

        * If the caller already passed ``cached_content`` on the config, we
          honor it and skip cache management entirely (returns the existing
          value unchanged).
        * If a valid cache exists for this prefix, returns its name.
        * Otherwise, kicks off a background cache creation for the *next*
          call and returns ``None`` so the current call goes through as a
          normal Gemini request.

        ``prefix_token_estimate`` is optional; when supplied, we skip cache
        creation for prefixes below the configured minimum (the storage
        cost would outweigh the discount).
        """
        if existing_cached_content:
            return existing_cached_content

        prefix_hash = _compute_prefix_hash(model, system_instruction, tools)
        if not prefix_hash:
            return None

        active = self.lookup_active(prefix_hash)
        if active:
            return active

        # Below the minimum prefix size, caching is unprofitable.
        if prefix_token_estimate is not None and prefix_token_estimate < self._min_prefix_tokens:
            return None

        # Kick off background creation for *next* call. Uses the genai client's
        # native caches.create call so all auth/headers/base_url are inherited.
        def _create() -> tuple[str, int] | None:
            return _create_gemini_cache(
                client=client,
                model=model,
                system_instruction=system_instruction,
                tools=tools,
                contents=contents,
                ttl_seconds=self._ttl_seconds,
            )

        self.schedule_creation(prefix_hash, _create)
        return None


def _create_gemini_cache(
    *,
    client: Any,
    model: str,
    system_instruction: Any,
    tools: Any,
    contents: Any,
    ttl_seconds: int,
) -> tuple[str, int] | None:
    """Create a Gemini cachedContents resource via the native genai client.

    Returns ``(name, ttl_seconds)`` on success, or None on any failure (in
    which case the pending slot is freed so a later call can try again).
    The native ``caches.create`` call inherits the gateway base_url + auth
    from the wrapped client, so the gateway sees this call for accounting.
    """
    try:
        from google.genai import types as genai_types
    except ImportError:
        return None

    try:
        config_kwargs: dict[str, Any] = {"ttl": f"{ttl_seconds}s"}
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = tools
        # Gemini requires at least one of contents/system/tools to be set on
        # the cached resource. When the caller hasn't supplied static
        # ``contents`` we lean on system + tools (the common case).
        if contents:
            config_kwargs["contents"] = contents

        cache = client.caches.create(
            model=model,
            config=genai_types.CreateCachedContentConfig(**config_kwargs),
        )
        name = getattr(cache, "name", None)
        if not name:
            return None
        return name, ttl_seconds
    except Exception:  # pragma: no cover - swallow upstream failures
        return None


# ─────────────────────── wrapped genai.Client surface ──────────────────────────


class _ModelsProxy:
    """Drop-in for ``client.models`` that intercepts ``generate_content`` to
    inject a cache reference when one is available.

    Everything else is forwarded to the native ``models`` object unchanged via
    ``__getattr__``.
    """

    def __init__(self, native_models: Any, cache_manager: GeminiCacheManager) -> None:
        self._native = native_models
        self._cache = cache_manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def generate_content(self, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
        config = self._inject_cached_content(model=model, config=config)
        return self._native.generate_content(model=model, contents=contents, config=config, **kwargs)

    async def generate_content_async(self, *, model: str, contents: Any, config: Any = None, **kwargs: Any) -> Any:
        config = self._inject_cached_content(model=model, config=config)
        return await self._native.generate_content_async(model=model, contents=contents, config=config, **kwargs)

    def _inject_cached_content(self, *, model: str, config: Any) -> Any:
        """Mutate a copy of ``config`` to point at the cached resource, if any."""
        # config may be a GenerateContentConfig object, a dict, or None.
        existing = _read_cached_content(config)
        system_instruction = _read_attr(config, "system_instruction")
        tools = _read_attr(config, "tools")
        cache_name = self._cache.attach(
            client=self._client_ref(),
            model=model,
            system_instruction=system_instruction,
            tools=tools,
            existing_cached_content=existing,
        )
        if cache_name is None or cache_name == existing:
            return config
        return _with_cached_content(config, cache_name)

    def _client_ref(self) -> Any:
        # We need the parent client to call client.caches.create. The wrapper
        # injects this back-reference at construction time.
        return getattr(self, "_parent_client", None)


def _read_attr(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _read_cached_content(config: Any) -> str | None:
    if config is None:
        return None
    value = _read_attr(config, "cached_content")
    if isinstance(value, str) and value:
        return value
    return None


def _with_cached_content(config: Any, name: str) -> Any:
    """Return a copy of ``config`` with ``cached_content`` set to ``name``.

    For ``GenerateContentConfig`` objects we use ``model_copy`` when
    available (pydantic); falling back to a dict overlay. For raw dicts we
    shallow-copy and update. ``None`` configs become dicts.
    """
    if config is None:
        return {"cached_content": name}
    if isinstance(config, dict):
        out = dict(config)
        out["cached_content"] = name
        return out
    # Pydantic-style model (google.genai uses pydantic v2).
    copy_fn = getattr(config, "model_copy", None)
    if callable(copy_fn):
        try:
            return copy_fn(update={"cached_content": name})
        except Exception:  # pragma: no cover - fallback below
            pass
    # Last-resort: set on the object directly. Mutates the caller's config,
    # which is mildly invasive but the only option for non-pydantic shapes.
    try:
        setattr(config, "cached_content", name)
    except Exception:  # pragma: no cover
        pass
    return config


class GenaiCachedClient:
    """Wrapper around ``google.genai.Client`` that auto-manages context cache.

    Forwards every attribute except ``models`` to the underlying client.
    ``models`` is replaced with a proxy that intercepts ``generate_content``
    to inject ``cached_content`` references.
    """

    def __init__(self, native_client: Any, cache_manager: GeminiCacheManager) -> None:
        self._native = native_client
        self._cache = cache_manager
        proxy = _ModelsProxy(native_client.models, cache_manager)
        # Back-reference so the proxy can invoke client.caches.create when it
        # needs to populate the registry.
        proxy._parent_client = native_client  # type: ignore[attr-defined]
        self._models_proxy = proxy

    @property
    def models(self) -> _ModelsProxy:
        return self._models_proxy

    @property
    def cache_manager(self) -> GeminiCacheManager:
        return self._cache

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)


def env_ttl_seconds() -> int:
    """Fallback TTL pulled from the env so containers can override without code.

    Honors ``DEEPINTSHIELD_GEMINI_CACHE_TTL`` (e.g. ``"6h"``). The full
    workspace-managed TTL ships through the gateway's settings page.
    """
    return _ttl_seconds_from_string(os.environ.get("DEEPINTSHIELD_GEMINI_CACHE_TTL"))
