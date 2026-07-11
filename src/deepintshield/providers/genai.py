from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._gemini_cache import GenaiCachedClient, GeminiCacheManager, env_ttl_seconds

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, passthrough: bool = False, **kwargs: Any):
    """Return a native ``google.genai.Client`` pointed at the gateway."""
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install google-genai: pip install 'deepintshield[genai]'") from exc

    base_url = shield.genai_passthrough_base_url() if passthrough else shield.genai_base_url()
    return genai.Client(
        api_key=kwargs.pop("api_key", shield.api_key()),
        http_options=kwargs.pop("http_options", HttpOptions(base_url=base_url, headers=shield.headers())),
        **kwargs,
    )


def build_cached_client(
    shield: "DeepintShield",
    *,
    passthrough: bool = False,
    ttl_seconds: int | None = None,
    min_prefix_tokens: int | None = None,
    **kwargs: Any,
) -> GenaiCachedClient:
    """Return a Gemini client with automatic ``cachedContents`` lifecycle.

    Drop-in for ``google.genai.Client`` (forwards every method that isn't
    ``models``). On each ``generate_content`` call:

    * Hashes the static prefix (model + system_instruction + tools).
    * If the SDK already has an active ``cachedContents/...`` resource for
      that prefix (within its TTL), passes it as ``cached_content`` so
      Gemini reuses the KV state - billed at ~25% of the normal input rate.
    * Otherwise lets the call run as a normal Gemini request and kicks off
      a fire-and-forget background creation of a cache resource for the
      *next* call. The current call therefore pays no extra latency.

    Caching is governed by the workspace switch on the gateway: when the
    workspace has prompt caching disabled or ``google`` not in the provider
    allow-list, the gateway strips ``cached_content`` before forwarding and
    this becomes a no-op end-to-end.

    Phase 4 ships this as an opt-in constructor - call ``shield.genai_cached()``
    explicitly - because Gemini's cache *storage* is metered and only
    profitable for large repeating prefixes (≥ ~32K tokens).
    """
    native = build_client(shield, passthrough=passthrough, **kwargs)
    cache_manager = GeminiCacheManager(
        ttl_seconds=ttl_seconds if ttl_seconds is not None else env_ttl_seconds(),
        min_prefix_tokens=min_prefix_tokens if min_prefix_tokens is not None else 32_768,
    )
    return GenaiCachedClient(native, cache_manager)
