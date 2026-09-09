"""CrewAI's native OpenAI-compatible LLM; CrewAI owns agent execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection, _merge_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


def llm(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``crewai.LLM`` bound to the gateway. Pass the result as
    the ``llm=`` of any CrewAI ``Agent``."""
    try:
        from crewai import LLM
        from crewai.llms.providers.openai.completion import OpenAICompletion
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install crewai: pip install 'deepintshield[crewai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="crewai",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    model = kwargs.pop("model", model)
    # CrewAI's public factory strips one transport prefix in custom mode.
    # Always add that outer prefix, preserving the complete gateway ID even
    # when it already starts with openai/ or includes deployment slashes.
    if "custom_openai" in getattr(OpenAICompletion, "model_fields", {}):
        kwargs["custom_openai"] = True
        model = f"openai/{model}"
    else:
        # Earlier native CrewAI releases select the same OpenAI implementation
        # through the public provider argument and retain the model verbatim.
        kwargs["provider"] = "openai"
    extra = kwargs.pop("extra_headers", None) or {}
    headers = _merge_headers(headers, kwargs.pop("default_headers", None) or {}, extra)
    return LLM(
        model=model,
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        default_headers=headers,
        **kwargs,
    )


# Alias so ``shield.crewai().model(...)`` also works.
model = llm
