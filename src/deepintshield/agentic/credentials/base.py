"""AgentCredential - the protocol every identity provider implementation
follows. The engine only ever calls ``get_token()`` on the abstract type;
the concrete provider (Entra / ZeroID / OIDC) handles the wire details.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Protocol


class AgentCredential(Protocol):
    """Returns a fresh agent identity token to inject into the
    ``X-Agent-Token`` header alongside the platform VK bearer.

    Implementations MUST:
      * cache tokens in-process for their TTL,
      * refresh silently when ttl < 5 min,
      * be safe for concurrent callers,
      * never block the hot path on a network roundtrip if the cached token
        is still valid.
    """

    def get_token(self) -> str: ...

    @property
    def provider_type(self) -> str: ...


class StaticAgentCredential:
    """Returns a pre-baked token verbatim. For local dev / tests where you
    want a known-good token without running an FIC exchange.

    Do not use in production - the token won't refresh.
    """

    def __init__(self, token: str, provider_type: str = "static") -> None:
        self._token = token
        self._provider_type = provider_type

    def get_token(self) -> str:
        return self._token

    @property
    def provider_type(self) -> str:
        return self._provider_type


class _CachedToken:
    """Tiny in-process cache used by FIC-based credentials.

    Thread-safe single-flight: only one caller performs the refresh at a
    time; the rest wait on the lock and pick up the fresh value.
    """

    REFRESH_BEFORE_EXPIRY_SECONDS = 300  # 5-minute pre-emptive refresh

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def get(self) -> Optional[str]:
        if not self._token:
            return None
        if time.time() >= self._expires_at - self.REFRESH_BEFORE_EXPIRY_SECONDS:
            return None
        return self._token

    def set(self, token: str, expires_in_seconds: float) -> None:
        with self._lock:
            self._set_unlocked(token, expires_in_seconds)

    def get_or_refresh(self, refresh: Callable[[], tuple[str, float]]) -> str:
        """Return a warm token or refresh it exactly once.

        The refresh callback deliberately runs while the lock is held: one
        caller performs the network exchange and concurrent callers wait for
        that result.  Updating the cache uses the unlocked helper so the
        non-reentrant ``threading.Lock`` is never acquired twice by the same
        thread.
        """
        cached = self.get()
        if cached:
            return cached
        with self._lock:
            cached = self.get()
            if cached:
                return cached
            token, ttl = refresh()
            self._set_unlocked(token, ttl)
            return token

    def _set_unlocked(self, token: str, expires_in_seconds: float) -> None:
        self._token = token
        self._expires_at = time.time() + expires_in_seconds

    def lock(self) -> threading.Lock:
        """Compatibility escape hatch for older credential implementations.

        New implementations should call :meth:`get_or_refresh`; exposing the
        lock remains harmless for third-party credentials that imported this
        private helper before the single-flight API existed.
        """
        return self._lock
