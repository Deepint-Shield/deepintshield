"""CrewAI binder - native ``crewai.LLM`` (LiteLLM under the hood) pointed at
the gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection

if TYPE_CHECKING:
    from ..client import DeepintShield


def llm(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return a native ``crewai.LLM`` bound to the gateway. Pass the result as
    the ``llm=`` of any CrewAI ``Agent``."""
    try:
        from crewai import LLM
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install crewai: pip install 'deepintshield[crewai]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="crewai",
        ) from exc
    base_url, headers = connection(shield, identity=identity)
    # LiteLLM needs a provider-prefixed model name for OpenAI-compatible bases.
    model = kwargs.pop("model", model)
    if "/" not in model:
        model = f"openai/{model}"
    return LLM(
        model=model,
        base_url=kwargs.pop("base_url", base_url),
        api_key=kwargs.pop("api_key", shield.api_key()),
        extra_headers={**headers, **(kwargs.pop("extra_headers", None) or {})},
        **kwargs,
    )


# Alias so ``shield.crewai().model(...)`` also works.
model = llm
