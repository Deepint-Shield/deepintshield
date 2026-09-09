from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._prompt_cache import PROVIDER_OPENAI, build_request_hook
from ..errors import ErrorCode, _dependency_error
from ..transport import openai_connection, _install_agent_selector_header_hook

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, passthrough: bool = False, identity: bool = False, **kwargs: Any):
    """Return a native ``openai.OpenAI`` client pre-wired to the DeepintShield gateway.

    The returned client includes an httpx event hook that adds a stable
    ``prompt_cache_key`` to outbound chat completion requests so OpenAI's
    automatic prompt cache partitions cleanly per logical prefix. Caching
    itself is governed by the workspace-level Provider Prompt Caching switch
    on the gateway - if the workspace has it disabled the gateway strips the
    key before forwarding.

    Callers passing their own ``http_client`` manage caching themselves; only
    case-insensitive agent selector override handling is added to that client.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install openai: pip install 'deepintshield[openai]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="openai",
        ) from exc

    client = OpenAI(**openai_connection(shield, passthrough=passthrough, identity=identity, **kwargs))
    # Let the installed SDK choose and validate its own transport. OpenAI 2.x
    # uses httpx; 3.x uses httpx2, whose client classes are not interchangeable.
    transport = client._client
    if kwargs.get("http_client") is None and isinstance(getattr(transport, "event_hooks", None), dict):
        transport.follow_redirects = False
        transport.event_hooks.setdefault("request", []).append(build_request_hook(PROVIDER_OPENAI))
    _install_agent_selector_header_hook(transport)
    return client


def build_async_client(shield: "DeepintShield", *, passthrough: bool = False, identity: bool = False, **kwargs: Any):
    """Return native ``openai.AsyncOpenAI`` with the same gateway connection.

    Client construction performs no requests. Use ``async with`` or await
    ``client.close()`` to release the native transport after use.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install openai: pip install 'deepintshield[openai]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="openai",
        ) from exc
    client = AsyncOpenAI(**openai_connection(shield, passthrough=passthrough, identity=identity, **kwargs))
    transport = client._client
    if kwargs.get("http_client") is None and isinstance(getattr(transport, "event_hooks", None), dict):
        transport.follow_redirects = False
        sync_hook = build_request_hook(PROVIDER_OPENAI)

        async def cache_hook(request):
            sync_hook(request)

        transport.event_hooks.setdefault("request", []).append(cache_hook)
    _install_agent_selector_header_hook(transport)
    return client
