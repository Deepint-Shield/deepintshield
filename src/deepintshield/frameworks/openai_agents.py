"""OpenAI Agents SDK binder - native ``AsyncOpenAI`` pointed at the gateway,
plus an ``apply()`` shortcut that registers it as the SDK's default client so
every Agent routes through DeepintShield with no further wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection

if TYPE_CHECKING:
    from ..client import DeepintShield


def client(shield: "DeepintShield", *, identity: bool = False, **kwargs: Any):
    """Return a native ``openai.AsyncOpenAI`` bound to the gateway."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install openai: pip install 'deepintshield[openai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="openai_agents",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    return AsyncOpenAI(
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**headers, **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )


def apply(shield: "DeepintShield", *, use_for_tracing: bool = False, identity: bool = False, **kwargs: Any):
    """Set the gateway-pointed client as the Agents SDK default and return it.

    After this, your existing ``Agent`` / ``Runner`` code is unchanged but all
    model traffic flows through DeepintShield."""
    try:
        from agents import set_default_openai_client
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install the OpenAI Agents SDK: pip install 'deepintshield[openai-agents]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="openai_agents",
        ) from exc
    c = client(shield, identity=identity, **kwargs)
    set_default_openai_client(c, use_for_tracing=use_for_tracing)
    return c
