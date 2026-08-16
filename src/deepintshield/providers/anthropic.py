from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._prompt_cache import PROVIDER_ANTHROPIC, build_http_client
from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, passthrough: bool = False, **kwargs: Any):
    """Return a native ``anthropic.Anthropic`` client pointed at the gateway.

    The returned client includes an httpx event hook that marks the static
    portions of the prompt (``system`` + ``tools`` by default) with Anthropic's
    ``cache_control: {"type": "ephemeral"}`` so Anthropic reuses KV state for
    the prefix on repeat calls. Caching itself is governed by the workspace's
    Provider Prompt Caching switch - disabled workspaces have the markers
    stripped at the gateway before they reach Anthropic.

    Pass a custom ``http_client`` to bypass injection entirely; the SDK trusts
    the caller's transport in that case.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install anthropic: pip install 'deepintshield[anthropic]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="anthropic",
        ) from exc

    base_url = shield.anthropic_passthrough_base_url() if passthrough else shield.anthropic_base_url()
    http_client = kwargs.pop("http_client", None)
    if http_client is None:
        http_client = build_http_client(PROVIDER_ANTHROPIC, timeout=shield.timeout)

    return anthropic.Anthropic(
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**shield.headers(), **(kwargs.pop("default_headers", None) or {})},
        http_client=http_client,
        **kwargs,
    )
