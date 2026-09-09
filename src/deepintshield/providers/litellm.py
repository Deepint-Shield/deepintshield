from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


class LiteLLMShield:
    """Native LiteLLM sync/async completions routed through the gateway."""

    def __init__(self, shield: "DeepintShield") -> None:
        self._shield = shield

    def completion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover
            raise _dependency_error(
                "Install litellm: pip install 'deepintshield[litellm]'",
                code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
                component="litellm",
            ) from exc

        return completion(**self._request_kwargs(model=model, messages=messages, **kwargs))

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any):
        try:
            from litellm import acompletion
        except ImportError as exc:  # pragma: no cover
            raise _dependency_error(
                "Install litellm: pip install 'deepintshield[litellm]'",
                code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
                component="litellm",
            ) from exc

        return await acompletion(**self._request_kwargs(model=model, messages=messages, **kwargs))

    def _request_kwargs(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        # LiteLLM's provider prefix selects a wire protocol and is removed
        # before transmission. Add an outer OpenAI prefix so the gateway gets
        # the original provider/model ID over its OpenAI-compatible route.
        # Without it, e.g. anthropic/... is sent as native Messages and Bedrock
        # attempts AWS authentication locally instead of using the virtual key.
        base_url = kwargs.pop("api_base", None) or kwargs.pop("base_url", None) or self._shield.litellm_base_url()
        kwargs.pop("base_url", None)
        api_key = kwargs.pop("api_key", None) or self._shield.api_key()
        headers = connection_headers(self._shield, extra=kwargs.pop("extra_headers", None))
        # The gateway owns model capabilities, including new model IDs that
        # are not in LiteLLM's local OpenAI catalog. Keep explicit parameters
        # such as reasoning_effort instead of rejecting or dropping them here.
        allowed_params = list(dict.fromkeys([*(kwargs.pop("allowed_openai_params", None) or []), *kwargs]))
        return {
            **kwargs,
            "model": f"openai/{model}",
            "messages": messages,
            "custom_llm_provider": "openai",
            "api_base": base_url,
            "api_key": api_key,
            "extra_headers": headers,
            "allowed_openai_params": allowed_params,
        }

    # Allow ``shield.litellm()(model=..., messages=...)`` to forward to completion.
    __call__ = completion


def build_client(shield: "DeepintShield") -> LiteLLMShield:
    return LiteLLMShield(shield)
