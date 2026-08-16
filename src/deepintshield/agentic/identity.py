"""Principal identity resolution against the GAF directory (agentic-new).

The ReBAC/GAF authorization model is written against *subjects*
(``user:<collision-resistant-id>``, ``service_account:etl-nightly``), but application code
only ever knows an email or a username. This module is the one-hop bridge: it
POSTs the human-readable identifier to ``/api/agentic-new/identity/resolve`` with
the VK bearer and gets back the canonical subject, creating the directory
principal on first sight (JIT provisioning) so an operator sees real users in
Organization → Users without any manual import.

Two properties matter for the hot path:

  * **One dict lookup after the first call** - a process-local LRU (512 entries)
    keyed on gateway + kind + identifier, so a per-request ``as_user(email)`` in
    a web handler costs nothing after warm-up.
  * **Fail-soft** - an offline/older gateway must never break a run. On any
    error the subject is derived locally with the *same* normalization rule the
    server applies, so the decision still carries a stable, correct-looking OBO
    leg and reconciles the moment the gateway is reachable again.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from .errors import GovernanceConfigurationError

log = logging.getLogger(__name__)

_RESOLVE_PATH = "/api/agentic-new/identity/resolve"
_TIMEOUT_SECONDS = 3.0
_CACHE_MAXSIZE = 512

# Everything outside [a-z0-9] collapses to a single dash - "Alice@Corp.com"
# becomes "alice-corp-com". Must stay byte-identical to the Go-side rule.
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Principal kinds a directory row can have; anything else degrades to "user".
_KINDS = ("user", "service_account")
# Subject types whose prefix is accepted verbatim on an already-typed subject.
#
# EXACTLY the two DIRECTORY kinds, mirroring the Go ``normalizeSubject`` switch
# (framework/agenticnew/identity.go). ``agent:`` is deliberately NOT here: an
# agent is a *registry* entity, not a directory principal, so the server derives
# ``agent:helper`` as a collision-resistant ``user:…`` subject rather than
# honouring the prefix. The SDK used to accept it verbatim, which meant an offline-minted subject
# (``agent:helper``) checked tuples the server would never write - the one input
# class where the fail-soft fallback produced a *wrong* answer instead of the
# same answer. Keep this tuple byte-identical to the Go switch.
_SUBJECT_TYPES = ("user", "service_account")
# Longest slug kept intact; lossy normalization always carries 128 digest bits
# so distinct Entra identifiers cannot collapse onto one privileged subject.
_MAX_SLUG = 96
_DIGEST_HEX_LENGTH = 32
_MAX_READABLE_SLUG = _MAX_SLUG - 1 - _DIGEST_HEX_LENGTH

_cache: "OrderedDict[tuple[str, str, str, str], str]" = OrderedDict()
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class PrincipalBinding:
    """Request-local identity retained after directory resolution.

    ``subject`` is the canonical value used by the PDP.  The source email or
    username is intentionally retained as well because the VK-authenticated
    registry ingest does not trust a caller-provided subject hint; it derives
    and verifies the principal from email/username instead.
    """

    subject: str
    email: str = ""
    username: str = ""
    display_name: str = ""
    kind: str = "user"

    def manifest(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.email:
            payload["email"] = self.email.strip().lower()
        if self.username:
            payload["username"] = self.username.strip()
        if self.subject:
            payload["subject"] = self.subject.strip()
        if self.display_name:
            payload["display_name"] = self.display_name.strip()
        payload["kind"] = self.kind if self.kind in _KINDS else "user"
        return payload


def slugify(value: str) -> str:
    """``agenticnew.SubjectSlug``: lowercase, every run of non-``[a-z0-9]``
    collapsed to a single ``-``, trimmed. ``"Alice@Corp.com"`` →
    ``"alice-corp-com"``. Returns ``""`` for an input with nothing to keep."""
    return _SLUG_RE.sub("-", (value or "").strip().lower()).strip("-")


def local_subject(value: str, kind: str = "user") -> str:
    """Derive the canonical subject locally, without touching the network.

    Byte-for-byte the Go ``normalizeSubject``/``DeriveSubject`` pair, which is
    what makes the fallback path safe: a subject minted while the gateway was
    down still matches the tuples the server writes later. An already-typed
    *directory* subject (``user:`` / ``service_account:``) keeps its type and
    only has its id re-slugified; anything else - including ``agent:…``, which
    is a registry entity rather than a directory principal - is typed by
    ``kind`` (``agent:helper`` becomes a collision-resistant ``user:…``
    subject, exactly as the server does)."""
    raw = (value or "").strip()
    marker = raw.find(":")
    if marker > 0:
        typ = raw[:marker].lower()
        if typ in _SUBJECT_TYPES:
            local = raw[marker + 1 :].strip().lower()
            slug = slugify(local)
            if slug and slug == local:
                return f"{typ}:{slug}"
            return _derive_subject(typ, local)
    return _derive_subject(kind, raw)


def _derive_subject(kind: str, identifier: str) -> str:
    """Type + collision-resistant slug, byte-identical to Go DeriveSubject.

    Any lossy normalization (including punctuation in an email) appends 128
    SHA-256 bits. This prevents identifiers such as ``alice.k@corp`` and
    ``alice-k@corp`` from collapsing onto the same privileged OpenFGA subject.
    """
    k = (kind or "").strip().lower()
    if k != "service_account":
        k = "user"
    normalized = (identifier or "").strip().lower()
    slug = slugify(normalized)
    digest = _short_hash(normalized)
    if not slug:
        slug = "id-" + digest
    elif slug != normalized:
        if len(slug) > _MAX_READABLE_SLUG:
            slug = slug[:_MAX_READABLE_SLUG].rstrip("-")
        slug += "-" + digest
    if len(slug) > _MAX_SLUG:
        slug = slug[:_MAX_READABLE_SLUG].rstrip("-") + "-" + digest
    return f"{k}:{slug}"


def _short_hash(value: str) -> str:
    """First 16 bytes of SHA-256, hex - the Go side's ``h[:16]``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_DIGEST_HEX_LENGTH]


def resolve_principal(
    engine: Any,
    *,
    email: Optional[str] = None,
    username: Optional[str] = None,
    subject: Optional[str] = None,
    display_name: Optional[str] = None,
    kind: str = "user",
) -> str:
    """Resolve a human identifier to its canonical GAF subject.

    Pass one of ``email`` / ``username`` / ``subject``; when several are given
    the precedence matches the server's (``subject`` > ``email`` > ``username``)
    so the cached key and the offline fallback agree with it. The gateway creates
    the ``agenticnew_principals`` row on first sight, so this doubles as JIT
    provisioning. Returns the canonical typed subject string; never
    raises - on any failure it falls back to :func:`local_subject`, which
    produces the identical string offline.
    """
    identifier = (subject or email or username or "").strip()
    if not identifier:
        raise GovernanceConfigurationError(
            framework="agentic-identity",
            reason="principal identifier is required",
            code="principal_identifier_missing",
        ) from None

    fallback = local_subject(identifier, kind)
    # Scope by SDK client as well as gateway. Two tenants commonly share one
    # host and can resolve the same email to the same deterministic subject, but
    # each tenant still needs its own JIT directory row; a gateway-global cache
    # previously skipped the second tenant's provisioning call.
    key = (
        _gateway_url(engine),
        _identity_scope(engine),
        kind,
        identifier.lower(),
    )

    cached = _cache_get(key)
    if cached is not None:
        return cached

    resolved = _post_resolve(
        engine,
        email=email,
        username=username,
        subject=subject,
        display_name=display_name,
        kind=kind,
    )
    if not resolved:
        # Fail-soft: cache nothing so the next call retries a recovered gateway.
        return fallback
    _cache_put(key, resolved)
    return resolved


def obo_actor_chain(engine: Any, agent_subject: str = "") -> list[str]:
    """The delegation chain the PDP should see for this call.

    Once ``shield.agentic.as_user(…)`` binds a human, the chain is
    ``["user:<derived-id>", "<selected-registry-agent>"]``. On older gateways
    that do not advertise a selected Registry principal it contains only the
    user, or is empty when no user is selected. The SDK never invents a shared
    ``agent:sdk`` identity; the authenticated server resolves the missing agent."""
    if not agent_subject:
        try:
            agent_subject = str(getattr(engine, "agent_subject", "") or "").strip()
        except Exception:
            agent_subject = ""
    user = (getattr(engine, "acting_user", "") or "").strip()
    return [subject for subject in (user, agent_subject) if subject]


def principal_manifest(engine: Any) -> dict[str, str]:
    """Return the trusted-source identity payload for registry discovery."""
    binding = getattr(engine, "principal_binding", None)
    if isinstance(binding, PrincipalBinding):
        return binding.manifest()
    user = (getattr(engine, "acting_user", "") or "").strip()
    return {"subject": user, "kind": "user"} if user else {}


def clear_cache() -> None:
    """Drop the process-local resolution cache (tests / long-lived workers that
    want to pick up a renamed principal)."""
    with _cache_lock:
        _cache.clear()


# ── internals ─────────────────────────────────────────────────────────────


def _cache_get(key: tuple[str, str, str, str]) -> Optional[str]:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)  # LRU recency
        return hit


def _cache_put(key: tuple[str, str, str, str], value: str) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAXSIZE:
            _cache.popitem(last=False)


def _gateway_url(engine: Any) -> str:
    try:
        return str(getattr(engine, "gateway_url", "") or "")
    except Exception:  # a half-built parent client must not break resolution
        return ""


def _identity_scope(engine: Any) -> str:
    try:
        scope = str(getattr(engine, "registry_scope_id", "") or "")
    except Exception:
        scope = ""
    return scope or f"engine-{id(engine)}"


def _post_resolve(
    engine: Any,
    *,
    email: Optional[str],
    username: Optional[str],
    subject: Optional[str],
    display_name: Optional[str],
    kind: str,
) -> str:
    """One POST to /identity/resolve. Returns "" on any failure (never raises)."""
    payload: dict[str, str] = {"kind": kind if kind in _KINDS else "user"}
    for field, value in (
        ("email", email),
        ("username", username),
        ("subject", subject),
        ("display_name", display_name),
    ):
        if value:
            payload[field] = str(value).strip()
    try:
        resp = engine._http.post(
            f"{engine.gateway_url}{_RESOLVE_PATH}",
            json=payload,
            headers=engine._headers(include_agent_token=True),
            timeout=_TIMEOUT_SECONDS,
        )
        if resp.status_code >= 300:
            log.debug("identity resolve http %s - using local subject", resp.status_code)
            return ""
        body = resp.json()
    except Exception as exc:  # offline gateway / older server / bad JSON
        log.debug("identity resolve unavailable (%s) - using local subject", exc)
        return ""
    if not isinstance(body, dict):
        return ""
    resolved = body.get("subject") or ""
    return resolved if isinstance(resolved, str) else ""


__all__ = [
    "PrincipalBinding",
    "resolve_principal",
    "local_subject",
    "slugify",
    "obo_actor_chain",
    "principal_manifest",
    "clear_cache",
]
