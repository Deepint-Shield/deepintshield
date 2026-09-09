"""AutoGen (AG2) binder - native ``autogen_ext`` OpenAI chat completion client
pointed at the gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection, _merge_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


def model_client(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``OpenAIChatCompletionClient`` bound to the gateway.
    Pass it as ``model_client=`` to any AutoGen agent."""
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install autogen: pip install 'deepintshield[autogen]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="autogen",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    options = {
        "model": kwargs.pop("model", model),
        "base_url": kwargs.pop("base_url", base_url),
        "api_key": kwargs.pop("api_key", shield.api_key()),
        "default_headers": _merge_headers(headers, kwargs.pop("default_headers", None) or {}),
        **kwargs,
    }
    try:
        return OpenAIChatCompletionClient(**options)
    except ValueError as exc:
        if (
            "model_info is required" not in str(exc)
            or options.get("model_info") is not None
            or options.get("model_capabilities") is not None
        ):
            raise
        # AutoGen's bundled model list cannot know every gateway deployment.
        # Permit text requests without guessing tool/vision/JSON support. Users
        # can supply model_info to enable capabilities verified for their model.
        options["model_info"] = {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
        }
        return OpenAIChatCompletionClient(**options)


# Convenience alias.
client = model_client
