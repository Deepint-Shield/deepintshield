from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


class LiteLLMShield:
    """Convenience wrapper around ``litellm.completion`` that always injects gateway routing."""

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

        return completion(
            model=model,
            messages=messages,
            base_url=kwargs.pop("base_url", self._shield.litellm_base_url()),
            extra_headers={**self._shield.headers(), **(kwargs.pop("extra_headers", None) or {})},
            **kwargs,
        )

    # Allow ``shield.litellm()(model=..., messages=...)`` to forward to completion.
    __call__ = completion


def build_client(shield: "DeepintShield") -> LiteLLMShield:
    return LiteLLMShield(shield)
