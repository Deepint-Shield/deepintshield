from __future__ import annotations

from typing import Any


class DeepintShieldError(Exception):
    def __init__(self, message: str, status_code: int | None = None, payload: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}

    @classmethod
    def from_response(cls, status_code: int, payload: dict | None) -> "DeepintShieldError":
        error = (payload or {}).get("error", {})
        # `error` may be a nested object ({"message": ...}) OR a plain string - the
        # gateway's FEATURE_LOCKED / feature-gate bodies put the human message
        # directly in `error`. Handle both so a 402 never crashes the SDK.
        error_message = error.get("message") if isinstance(error, dict) else error
        message = (
            error_message
            or (payload or {}).get("message")
            or f"DeepintShield request failed with status {status_code}"
        )
        return cls(message=message, status_code=status_code, payload=payload or {})


class DeepintShieldBlockedError(DeepintShieldError):
    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        decision: str | None = None,
        reason: str | None = None,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, status_code=status_code, payload=payload)
        self.stage = stage
        self.decision = decision
        self.reason = reason
