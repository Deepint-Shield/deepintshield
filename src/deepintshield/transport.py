"""Transport layer (L1) - the single source of truth for pointing any
framework's *native* client at the DeepintShield gateway.

The whole "minimal touch" promise rests here: a user keeps 100% of their
framework code and only swaps where the model client points (``base_url``)
and what key it carries (the virtual key). Everything else - attribution and,
optionally, agent identity - is injected as headers in one place.

``connection()`` returns ``(base_url, headers)`` for manual wiring;
``http_client()`` returns a ready ``httpx.Client`` carrying the same. Both are
consumed by :mod:`deepintshield.frameworks` and re-exported on the client as
``shield.connection()`` / ``shield.http_client()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import httpx

if TYPE_CHECKING:
    from .client import DeepintShield


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
    h = dict(shield.headers())  # content-type + x-bf-vk + default_headers
    h.setdefault("x-bf-app", shield.app_name)
    h.setdefault("x-bf-agent", shield.agent_name)
    h.setdefault("x-bf-requester", shield.requester)
    h.setdefault("x-bf-requester-role", shield.requester_role)
    if identity:
        token = shield._agent_token()
        if token:
            h["X-Agent-Token"] = token
    if extra:
        h.update(dict(extra))
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


__all__ = ["connection", "connection_headers", "http_client"]
