"""AutoGen (AG2) binder - native ``autogen_ext`` OpenAI chat completion client
pointed at the gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..transport import connection

if TYPE_CHECKING:
    from ..client import DeepintShield


def model_client(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``OpenAIChatCompletionClient`` bound to the gateway.
    Pass it as ``model_client=`` to any AutoGen agent."""
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install autogen: pip install 'deepintshield[autogen]'") from exc
    base_url, headers = connection(shield, identity=identity)
    return OpenAIChatCompletionClient(
        model=kwargs.pop("model", model),
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers={**headers, **(kwargs.pop("default_headers", None) or {})},
        **kwargs,
    )


# Convenience alias.
client = model_client
