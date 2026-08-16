from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .errors import ErrorCode, _annotate_error

DEFAULT_BASE_URL = "https://app.deepintshield.com"


def _normalize_base_url(value: str | None) -> str:
    if value is None:
        return DEFAULT_BASE_URL
    cleaned = value.strip().rstrip("/")
    return cleaned or DEFAULT_BASE_URL


@dataclass(slots=True)
class ShieldConfig:
    virtual_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    app_name: str = "deepintshield"
    agent_name: str = "deepintshield-agent"
    requester: str = "sdk-user"
    requester_role: str = "member"
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
        return cls(
            virtual_key=(os.getenv("DEEPINTSHIELD_VIRTUAL_KEY") or "").strip(),
            base_url=_normalize_base_url(base_url),
            timeout=timeout,
            app_name=os.getenv("DEEPINTSHIELD_APP_NAME", "deepintshield"),
            agent_name=os.getenv("DEEPINTSHIELD_AGENT_NAME", "deepintshield-agent"),
            requester=os.getenv("DEEPINTSHIELD_REQUESTER", "sdk-user"),
            requester_role=os.getenv("DEEPINTSHIELD_REQUESTER_ROLE", "member"),
            persist=os.getenv("DEEPINTSHIELD_PERSIST", "true").lower() not in {"0", "false", "no"},
        )
