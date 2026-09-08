from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._prompt_cache import PROVIDER_ANTHROPIC, build_request_hook
from ..errors import ErrorCode, _dependency_error
from ..transport import connection_headers, _install_agent_selector_header_hook

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

    Pass a custom ``http_client`` to manage caching yourself; only
    case-insensitive agent selector override handling is added to that client.
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
    owns_http_client = http_client is None
    if http_client is None:
        # Use the installed SDK's public transport class: older releases use
        # httpx, while current releases require httpx2 and reject httpx.Client.
        http_client = anthropic.DefaultHttpxClient(
            timeout=shield.timeout,
            follow_redirects=False,
            event_hooks={"request": [build_request_hook(PROVIDER_ANTHROPIC)]},
        )

    try:
        client = anthropic.Anthropic(
            base_url=kwargs.pop("base_url", base_url),
            api_key=kwargs.pop("api_key", shield.api_key()),
            default_headers=connection_headers(shield, extra=kwargs.pop("default_headers", None)),
            http_client=http_client,
            **kwargs,
        )
    except Exception:
        if owns_http_client:
            http_client.close()
        raise
    # Let the SDK validate caller-supplied transports before mutating hooks.
    _install_agent_selector_header_hook(http_client)
    return client
