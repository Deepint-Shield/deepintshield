from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._gemini_cache import GenaiCachedClient, GeminiCacheManager, env_ttl_seconds
from ..errors import ErrorCode, _dependency_error
from ..transport import (
    connection_headers,
    _install_agent_selector_header_hook,
    _normalize_agent_selector_headers,
    _normalize_agent_selector_headers_async,
)

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, passthrough: bool = False, **kwargs: Any):
    """Return a native ``google.genai.Client`` pointed at the gateway."""
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install google-genai: pip install 'deepintshield[genai]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="genai",
        ) from exc

    base_url = shield.genai_passthrough_base_url() if passthrough else shield.genai_base_url()
    supplied_options = kwargs.pop("http_options", None)
    if isinstance(supplied_options, HttpOptions):
        options = supplied_options.model_copy()
    else:
        options = HttpOptions(**(supplied_options or {}))
    options.base_url = options.base_url or base_url
    options.headers = connection_headers(shield, extra=options.headers)
    # Preserve caller options, transports, and hooks. Native sync/async clients
    # use the same stateless selector normalization, including stream requests.
    for client_attr, args_attr, hook in (
        ("httpx_client", "client_args", _normalize_agent_selector_headers),
        ("httpx_async_client", "async_client_args", _normalize_agent_selector_headers_async),
    ):
        native_http_client = getattr(options, client_attr, None)
        if native_http_client is not None:
            _install_agent_selector_header_hook(native_http_client)
        else:
            client_args = dict(getattr(options, args_attr, None) or {})
            event_hooks = dict(client_args.get("event_hooks") or {})
            request_hooks = list(event_hooks.get("request") or [])
            if hook not in request_hooks:
                request_hooks.append(hook)
            event_hooks["request"] = request_hooks
            client_args["event_hooks"] = event_hooks
            setattr(options, args_attr, client_args)
    return genai.Client(
        api_key=kwargs.pop("api_key", shield.api_key()),
        http_options=options,
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
