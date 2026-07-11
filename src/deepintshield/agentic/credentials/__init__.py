"""Agent identity credential implementations.

A credential is anything that can produce an agent identity token to present
alongside the platform VK on every PEP request. The right implementation is
auto-selected by :class:`AgenticEngine` based on the ``provider_type`` it
discovers from the gateway - developers almost never instantiate these
directly.
"""

from .base import AgentCredential, StaticAgentCredential
from .entra import EntraAgentCredential
from .oidc import OIDCCredential
from .zeroid import ZeroIDCredential

__all__ = [
    "AgentCredential",
    "EntraAgentCredential",
    "ZeroIDCredential",
    "OIDCCredential",
    "StaticAgentCredential",
]
