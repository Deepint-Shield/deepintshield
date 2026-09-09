from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import DeepintShield


class ProviderRegistry:
    """Lazy accessor for provider builders, so optional deps stay optional."""

    def __init__(self, client: "DeepintShield") -> None:
        self._client = client

    def openai(self, **kwargs: Any):
        from . import openai as mod
        return mod.build_client(self._client, **kwargs)

    def anthropic(self, **kwargs: Any):
        from . import anthropic as mod
        return mod.build_client(self._client, **kwargs)

    def async_openai(self, **kwargs: Any):
        from . import openai as mod
        return mod.build_async_client(self._client, **kwargs)

    def bedrock(self, **kwargs: Any):
        from . import bedrock as mod
        return mod.build_client(self._client, **kwargs)

    def genai(self, **kwargs: Any):
        from . import genai as mod
        return mod.build_client(self._client, **kwargs)

    def langchain(self, **kwargs: Any):
        from . import langchain as mod
        return mod.build_client(self._client, **kwargs)

    def langgraph(self):
        from .langgraph import LangGraphShield
        return LangGraphShield(self._client)

    def litellm(self):
        from .litellm import LiteLLMShield
        return LiteLLMShield(self._client)

    def pydanticai(self, **kwargs: Any):
        from . import pydanticai as mod
        return mod.build_agent(self._client, **kwargs)


__all__ = ["ProviderRegistry"]
