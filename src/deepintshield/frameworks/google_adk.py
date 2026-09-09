"""Google ADK's optional native LiteLlm connector to the OpenAI gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


def model(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return ADK LiteLlm; native ADK still runs agents, tools, and sessions.

    ADK requires its LiteLLM connector for this route. The outer ``openai/``
    transport prefix is removed by LiteLLM; the gateway model ID is preserved.
    """
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install the ADK gateway connector: pip install 'deepintshield[google-adk]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING, component="google_adk",
        ) from exc
    headers = connection_headers(shield, identity=identity, extra=kwargs.pop("extra_headers", None))
    base_url = kwargs.pop("api_base", None) or kwargs.pop("base_url", None) or shield.openai_base_url()
    kwargs.pop("base_url", None)
    key = kwargs.pop("api_key", None) or shield.api_key()
    return LiteLlm(
        model=f"openai/{model}", api_base=base_url, api_key=key,
        extra_headers=headers, **kwargs,
    )
