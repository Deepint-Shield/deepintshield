"""Transport layer (L1) - the single source of truth for pointing any
framework's *native* client at the DeepintShield gateway.

The compatibility layer keeps gateway endpoint, virtual-key, attribution, and
optional agent-identity header construction in one place. Supported framework
integrations can generally reuse their application logic while swapping model
client construction; provider-specific extensions still require validation.

``connection()`` returns ``(base_url, headers)`` for manual wiring;
``http_client()`` returns a ready ``httpx.Client`` carrying the same. Both are
consumed by :mod:`deepintshield.frameworks` and re-exported on the client as
``shield.connection()`` / ``shield.http_client()``.
"""

from __future__ import annotations

from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any, Mapping, Optional

import httpx

if TYPE_CHECKING:
    from .client import DeepintShield


def _merge_headers(*sources: Mapping[str, str]) -> dict[str, str]:
    """HTTP field names are case-insensitive; the last source wins."""
    result: dict[str, str] = {}
    names: dict[str, str] = {}
    for source in sources:
        for name, value in source.items():
            previous = names.get(name.lower())
            if previous is not None:
                result.pop(previous)
            result[name] = value
            names[name.lower()] = name
    return result


def _normalize_agent_selector_headers(request: httpx.Request) -> None:
    # Native SDKs merge default/extra headers as case-sensitive dictionaries.
    # HTTPX retains both spellings, so select the final explicit override before
    # sending rather than emitting two contradictory profile selectors.
    for name in ("x-deepintshield-agent", "x-agent-subject"):
        values = request.headers.get_list(name)
        if len(values) > 1:
            request.headers[name] = values[-1]


async def _normalize_agent_selector_headers_async(request: httpx.Request) -> None:
    _normalize_agent_selector_headers(request)


def _install_agent_selector_header_hook(client: Any) -> None:
    # Both httpx and httpx2 expose async send methods, but their AsyncClient
    # classes are unrelated. Match the operation instead of one package type.
    event_hooks = getattr(client, "event_hooks", None)
    send = getattr(client, "send", None)
    # Native SDKs can accept other public transports (for example aiohttp).
    # Preserve them: HTTPX event hooks are an optional transport capability.
    if not isinstance(event_hooks, dict) or not callable(send):
        return
    hook = _normalize_agent_selector_headers_async if iscoroutinefunction(send) else _normalize_agent_selector_headers
    hooks = event_hooks.setdefault("request", [])
    if hook not in hooks:
        hooks.append(hook)


def connection_headers(
    shield: "DeepintShield",
    *,
    identity: bool = False,
    extra: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build the header set the gateway needs for transparent traffic:
    the VK, attribution (app / agent / requester), and - when ``identity`` is
    set - a best-effort ``X-Agent-Token`` for agent identity.

    ``identity=True`` triggers lazy agentic discovery (one network call) the
    first time; it defaults off so chat traffic never blocks on it.
    """
    defaults = {"x-deepintshield-app": shield.app_name}
    # Omitted rather than sent empty when unset: an empty selector is not an
    # identity, and sending one would have the gateway resolve "the agent this
    # key happens to be bound to" instead of the one the caller meant.
    if str(getattr(shield, "agent_name", "") or "").strip():
        defaults["x-deepintshield-agent"] = shield.agent_name
    defaults["x-deepintshield-requester"] = shield.requester
    defaults["x-deepintshield-requester-role"] = shield.requester_role
    h = _merge_headers(defaults, shield.headers())
    if identity:
        token = shield._agent_token()
        if token:
            h = _merge_headers(h, {"X-Agent-Token": token})
    if extra:
        h = _merge_headers(h, extra)
    return h


def connection(
    shield: "DeepintShield",
    *,
    provider: str = "openai",
    identity: bool = False,
    extra: Optional[Mapping[str, str]] = None,
) -> tuple[str, dict[str, str]]:
    """Return ``(base_url, headers)`` for a guarded gateway route.

    ``provider`` selects the route: ``"openai"`` (default, OpenAI-compatible -
    every framework's OpenAI client posts to ``…/openai/chat/completions``),
    or any other gateway-mounted provider (``"anthropic"``, ``"genai"``,
    ``"bedrock"``, ``"litellm"`` …). The gateway routes by URL path, so no
    provider header is needed.
    """
    return shield.endpoint(provider), connection_headers(shield, identity=identity, extra=extra)


def openai_connection(
    shield: "DeepintShield", *, passthrough: bool = False, identity: bool = False, **kwargs: Any,
) -> dict[str, Any]:
    """Fresh native OpenAI constructor options; no discovery or inference I/O.

    ``identity=True`` retains the existing opt-in identity lookup. Caller
    options, including custom transports and API-key callables, pass through.
    The base URL identifies the protocol; ``provider/model`` selects routing
    separately on each native inference call.
    """
    base_url = shield.openai_passthrough_base_url() if passthrough else shield.openai_base_url()
    kwargs.setdefault("base_url", base_url)
    if "api_key" not in kwargs:
        kwargs["api_key"] = shield.api_key()
    kwargs.setdefault("timeout", shield.timeout)
    kwargs["default_headers"] = connection_headers(shield, identity=identity, extra=kwargs.get("default_headers"))
    return kwargs


def http_client(
    shield: "DeepintShield",
    *,
    provider: str = "openai",
    identity: bool = False,
    base_url: Optional[str] = None,
    extra: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
) -> httpx.Client:
    """A plain ``httpx.Client`` pre-loaded with the gateway base URL + headers.
    Hand it to any SDK that accepts a custom ``http_client``."""
    return httpx.Client(
        base_url=base_url if base_url is not None else shield.endpoint(provider),
        headers=connection_headers(shield, identity=identity, extra=extra),
        timeout=timeout if timeout is not None else shield.timeout,
    )


__all__ = ["connection", "connection_headers", "http_client", "openai_connection"]
