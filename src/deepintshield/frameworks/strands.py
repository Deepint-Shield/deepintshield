"""Strands' public OpenAI model; Strands owns the agent and tool lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error

if TYPE_CHECKING:
    from ..client import DeepintShield


def model(shield: "DeepintShield", model: str = "gpt-4o-mini", *, identity: bool = False, **kwargs: Any):
    """Return native OpenAIModel using an arbitrary gateway ``provider/model``.

    Pass inference controls as Strands ``params`` and connection controls as
    ``client_args``. Supplied native ``client`` objects remain caller-owned.
    """
    try:
        from strands.models.openai import OpenAIModel
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install Strands OpenAI support: pip install 'deepintshield[strands]'",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING, component="strands",
        ) from exc
    client_args = dict(kwargs.pop("client_args", {}))
    if kwargs.get("client") is None:
        client_args = shield.openai_config(identity=identity, **client_args)
    return OpenAIModel(model_id=kwargs.pop("model_id", model), client_args=client_args, **kwargs)
