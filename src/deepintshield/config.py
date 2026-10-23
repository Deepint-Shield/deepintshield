from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .errors import ErrorCode, _annotate_error

DEFAULT_BASE_URL = "https://app.deepintshield.com"

# The registering developer's e-mail. One local part, one domain with a dot, no
# whitespace. Only the shape is checked, here and at the gateway: the gateway
# maps the address to an identity in the workspace directory and notifies it.
_REGISTERED_BY_RE = re.compile(r"^[^\s@]{1,64}@[^\s@]+\.[^\s@]+$")
_REGISTERED_BY_MAX_LENGTH = 254
# Where the registering e-mail came from. It travels beside the address as
# ``registered_by_source``. "explicit" is the ``registered_by`` setting: the
# gateway maps that address to an identity in the workspace directory and
# notifies it. The address is not verified and grants nothing; the
# administrator confirms the owner, and one registration keeps its first
# address. "requester" is an address derived from an e-mail-shaped
# ``requester``. ``requester`` names whoever the agent acts for, often an end
# user, so that address is only a fallback: the gateway ignores it, without
# refusing, for people the organization has not placed in the workspace.
_REGISTERED_BY_EXPLICIT = "explicit"
_REGISTERED_BY_REQUESTER = "requester"
_REGISTERED_BY_SOURCES = frozenset({_REGISTERED_BY_EXPLICIT, _REGISTERED_BY_REQUESTER})


def _normalize_base_url(value: str | None) -> str:
    if value is None:
        return DEFAULT_BASE_URL
    cleaned = value.strip().rstrip("/")
    return cleaned or DEFAULT_BASE_URL


def _email_shaped(value: object) -> str:
    """Return ``value`` as a lowercase e-mail address, or ``""`` when it is not one."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    if not cleaned or len(cleaned) > _REGISTERED_BY_MAX_LENGTH:
        return ""
    return cleaned if _REGISTERED_BY_RE.fullmatch(cleaned) else ""


def _normalize_registered_by(
    value: str | None,
    *,
    details: Mapping[str, Any] | None = None,
) -> str | None:
    """Validate the ``registered_by`` setting.

    ``None`` keeps the derivation from ``requester``; ``""`` explicitly names no
    registering identity; anything else must be an e-mail address and is
    returned lowercased. An invalid value raises an annotated ``ValueError``
    carrying ``configuration_error`` and the caller-supplied ``details``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise _annotate_error(
            TypeError("registered_by must be an e-mail address string"),
            ErrorCode.CONFIGURATION_ERROR,
            details=details,
        ) from None
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    normalized = _email_shaped(cleaned)
    if not normalized:
        raise _annotate_error(
            ValueError("registered_by must be an e-mail address"),
            ErrorCode.CONFIGURATION_ERROR,
            details=details,
        ) from None
    return normalized


@dataclass(slots=True)
class ShieldConfig:
    virtual_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    app_name: str = "deepintshield"
    # No default. A shared literal meant every unnamed workload in a workspace
    # claimed the same registry row: the first one to run enrolled it, and the
    # second got a bare 403 with nothing pointing at the collision. The name is
    # an identity, so it has to be chosen, not inherited.
    agent_name: str = ""
    requester: str = "sdk-user"
    requester_role: str = "member"
    # The developer who files the agent registration. The gateway maps the
    # address to an identity in the workspace directory, proposes that
    # identity as the owner and notifies the address; it is not verified and
    # grants nothing. None sends an e-mail-shaped ``requester`` in its place
    # as a fallback the gateway may ignore; "" sends no registering identity.
    # Never a header: it travels only in the registry discover payload.
    registered_by: str | None = None
    persist: bool = True
    default_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = _normalize_base_url(self.base_url)

    @classmethod
    def from_env(cls) -> "ShieldConfig":
        base_url = os.getenv("DEEPINTSHIELD_BASE_URL") or os.getenv(
            "DEEPINTSHIELD_GATEWAY_URL"
        )
        timeout_value = os.getenv("DEEPINTSHIELD_TIMEOUT", "30")
        try:
            timeout = float(timeout_value)
        except ValueError as exc:
            raise _annotate_error(
                exc,
                ErrorCode.CONFIGURATION_ERROR,
                details={"environment_variable": "DEEPINTSHIELD_TIMEOUT"},
            ) from None
        registered_by = _normalize_registered_by(
            os.getenv("DEEPINTSHIELD_REGISTERED_BY"),
            details={"environment_variable": "DEEPINTSHIELD_REGISTERED_BY"},
        )
        return cls(
            virtual_key=(os.getenv("DEEPINTSHIELD_VIRTUAL_KEY") or "").strip(),
            base_url=_normalize_base_url(base_url),
            timeout=timeout,
            app_name=os.getenv("DEEPINTSHIELD_APP_NAME", "deepintshield"),
            agent_name=(os.getenv("DEEPINTSHIELD_AGENT_NAME") or "").strip(),
            requester=os.getenv("DEEPINTSHIELD_REQUESTER", "sdk-user"),
            requester_role=os.getenv("DEEPINTSHIELD_REQUESTER_ROLE", "member"),
            registered_by=registered_by,
            persist=os.getenv("DEEPINTSHIELD_PERSIST", "true").lower() not in {"0", "false", "no"},
        )
