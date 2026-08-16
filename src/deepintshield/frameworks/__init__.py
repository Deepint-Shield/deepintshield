"""Framework binders (L1) - return *native* framework objects pre-pointed at
the DeepintShield gateway.

These are the "drop-in" layer: the user keeps all of their framework code and
only obtains their model / embedding client from here so traffic flows
through the gateway (where input/output/RAG guardrails + observability run
server-side). No DeepintShield types leak into their agent code.

Accessed via ``shield.bind("crewai")`` or the convenience accessors
``shield.crewai()`` / ``shield.openai_agents()`` / ``shield.llamaindex()`` /
``shield.autogen()`` - each returns a :class:`FrameworkBinder`.
"""

from __future__ import annotations

import functools
import importlib
from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _annotate_error

if TYPE_CHECKING:
    from ..client import DeepintShield

# Public framework name → module name within this package.
_ALIASES = {
    "langchain": "langgraph",  # both bind native langchain_openai clients
    "openai-agents": "openai_agents",
    "llama-index": "llamaindex",
    "llama_index": "llamaindex",
    "ag2": "autogen",
    "pydantic-ai": "pydanticai",
    "pydantic_ai": "pydanticai",
}

_KNOWN = {"langgraph", "crewai", "openai_agents", "llamaindex", "autogen", "pydanticai"}


class FrameworkBinder:
    """Thin façade dispatching ``.model()/.embedder()/.client()/.apply()`` to
    the underlying framework module, bound to one ``DeepintShield``."""

    def __init__(self, shield: "DeepintShield", name: str, module: Any) -> None:
        self._shield = shield
        self._name = name
        self._module = module

    def __getattr__(self, attr: str):
        fn = getattr(self._module, attr, None)
        if not callable(fn):
            raise _annotate_error(
                AttributeError(
                    f"framework {self._name!r} has no binder {attr!r} "
                    f"(available: {', '.join(self._public_binders())})"
                ),
                ErrorCode.FRAMEWORK_BINDER_ATTRIBUTE_MISSING,
                details={"framework": self._name, "binder": attr},
            )
        return functools.partial(fn, self._shield)

    def _public_binders(self) -> list[str]:
        return sorted(
            n for n in dir(self._module)
            if not n.startswith("_") and callable(getattr(self._module, n, None))
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<FrameworkBinder {self._name}>"


def get_binder(shield: "DeepintShield", framework: str) -> FrameworkBinder:
    name = _ALIASES.get(framework.lower(), framework.lower().replace("-", "_"))
    if name not in _KNOWN:
        raise _annotate_error(
            ValueError(
                f"unknown framework {framework!r}; known: {', '.join(sorted(_KNOWN))}"
            ),
            ErrorCode.FRAMEWORK_BINDER_NOT_FOUND,
            details={"framework": framework},
        )
    module = importlib.import_module(f".{name}", __package__)
    return FrameworkBinder(shield, name, module)


__all__ = ["FrameworkBinder", "get_binder"]
