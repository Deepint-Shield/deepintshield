from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._prompt_cache import PROVIDER_OPENAI, build_http_client

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, passthrough: bool = False, **kwargs: Any):
    """Return a native ``openai.OpenAI`` client pre-wired to the DeepintShield gateway.

    The returned client includes an httpx event hook that adds a stable
    ``prompt_cache_key`` to outbound chat completion requests so OpenAI's
    automatic prompt cache partitions cleanly per logical prefix. Caching
    itself is governed by the workspace-level Provider Prompt Caching switch
    on the gateway - if the workspace has it disabled the gateway strips the
    key before forwarding.

    Callers passing their own ``http_client`` are respected (no hook is
    attached); they're assumed to manage caching themselves.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install openai: pip install 'deepintshield[openai]'") from exc

    base_url = shield.openai_passthrough_base_url() if passthrough else shield.openai_base_url()
    http_client = kwargs.pop("http_client", None)
    if http_client is None:
        http_client = build_http_client(PROVIDER_OPENAI, timeout=shield.timeout)

    return OpenAI(
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**shield.headers(), **(kwargs.pop("default_headers", None) or {})},
        http_client=http_client,
        **kwargs,
    )
