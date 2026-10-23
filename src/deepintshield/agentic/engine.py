"""AgenticEngine - the low-level Agentic (PDP) transport/control primitive.

Applications use ``shield.agentic`` or a native framework adapter. Those
supported boundaries translate the engine's internal typed failures into
marked built-in Python exceptions with trusted details. Direct engine use and
its code-only typed errors remain available for SDK internals and backward
compatibility, but are not the supported application error contract.

Unlike the standalone ``deepintshield_agents`` client it was migrated from,
the engine does not take its own ``gateway_url``/``virtual_key`` - it binds
to the parent :class:`~deepintshield.client.DeepintShield` and reuses its
``base_url``, ``virtual_key`` and shared httpx client. The user therefore
passes *only* virtual key + base URL to ``DeepintShield`` and everything
agentic (discovery, credential selection, agent-token minting) happens
automatically.

What it does at decide() time:
    1. POST /api/agentic-new/decide for the authoritative OpenFGA/GAF verdict.
    2. Return that verdict directly. The SDK never asks a hidden second PDP to
       replace or tighten canonical evidence after GAF succeeds.
    3. Only when both canonical Agentic-New routes are genuinely absent (an
       older gateway returning 404/405/501) use the legacy PDP as an explicitly
       marked compatibility source. Canonical errors always fail closed.
    4. Authorization calls use the authenticated key plus the Agentic selector
       and, when configured, workload proof:
       - Authorization: Bearer <virtual_key>  (platform VK)
       - X-Agent-Subject: agent:<registry-key> (untrusted profile selector)
       - X-Agent-Token: <fresh agent token from the credential>, when configured
    5. Return the Decision; the gate translates blocking verdicts
       into exceptions.

Discovery + credential build are lazy (first decide(), not at construction)
so building a ``DeepintShield`` never performs network I/O.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import os
import re
import threading
import time
import uuid
import weakref
from typing import TYPE_CHECKING, Optional

import httpx

from ..config import _REGISTERED_BY_EXPLICIT, _REGISTERED_BY_REQUESTER, _email_shaped
from ..errors import DeepintShieldError as SDKConfigurationError
from ..errors import get_exception_error_code
from .actions import infer_action_class
from .credentials.base import AgentCredential, StaticAgentCredential
from .errors import (
    _REGISTRATION_EMAIL_MAX_LENGTH,
    _REGISTRATION_EMAIL_RE,
    _REGISTRATION_MESSAGE_CODES,
    _REGISTRATION_REFERENCE_RE,
    _RUNTIME_KEY_THUMBPRINT_RE,
    _registration_fragment,
    AGENT_APPROVAL_PENDING,
    AGENT_CONFIGURATION_ERROR,
    AGENT_GATEWAY_UNAVAILABLE,
    AGENT_INVALID_GATEWAY_RESPONSE,
    DeepIntShieldError,
    GatewayUnavailable,
    GovernanceConfigurationError,
    GuardrailApprovalPending,
    normalize_agentic_error_code,
    public_agentic_error_details,
)
from .identity import PrincipalBinding
from .registry import _registry_key as _manifest_registry_key
from .types import Decision, DelegationContext, GAFDecision, Verdict, VKCredentialInfo

if TYPE_CHECKING:
    from ..client import DeepintShield

log = logging.getLogger(__name__)
_PRINCIPAL_UNSET = object()
# Best-effort transport runs on every LLM request; a broken host credential
# would otherwise repeat the same warning per call. Each code prints once per
# process, like the registry's enrolment lifecycle line.
_credential_failure_logged: set[str] = set()
_credential_failure_log_lock = threading.Lock()
_REGISTRATION_STATE_RE = re.compile(r"^[a-z][a-z_]{0,31}$")
_REGISTRATION_SUBJECT_RE = re.compile(r"^agent:[A-Za-z0-9][A-Za-z0-9:_.-]{0,191}$")
_REGISTRATION_PENDING_CODE = "agent_registration_pending"
# Why a provisional runtime key can be unusable in this process: the runtime
# extra is not installed, the gateway address or its broker settings are not
# ones the credential accepts, or the key store cannot hold the key. Only for
# the gateway's provisional offer to a pending registration do these mean
# "still pending"; anywhere else they surface unchanged.
_PROVISIONAL_KEY_UNUSABLE_CODES = frozenset(
    {
        "credential_dependency_missing",
        "agent_runtime_configuration_error",
        "agent_runtime_key_unavailable",
    }
)
# The one refusal of a provisional runtime key that makes the engine ask for
# the agent's credential settings again: the gateway no longer accepts that
# key at enrolment or challenge, which is what an administrator's decision
# (a VK-trusted approval, a denial, an identity declared elsewhere) leaves
# behind. Every other failure, including an outage, a 429, a conflict and a
# key still waiting for authorization, surfaces unchanged.
_RUNTIME_AUTHENTICATION_FAILED = "agent_runtime_authentication_failed"
_PROVISIONAL_REREAD_ENDPOINTS = frozenset({"enroll", "challenge"})
# A credential the gateway marked optional (``credential_optional``) that
# failed is not tried again for this long, so an approved agent running on its
# virtual key does not repeat a failing build or enrolment on every call.
_OPTIONAL_CREDENTIAL_RETRY_SECONDS = 60.0
# One line per process and agent says that the agent runs on its virtual key.
_optional_credential_logged: set[str] = set()
_optional_credential_log_lock = threading.Lock()


def _log_credential_failure(error: BaseException, fallback_code: str) -> None:
    """Log trusted guidance once per process per code, never provider prose."""
    details = public_agentic_error_details(
        get_exception_error_code(error)
        or ("credential_dependency_missing" if isinstance(error, ImportError) else fallback_code)
    )
    payload = getattr(error, "payload", None)
    if (
        details["code"] in _REGISTRATION_MESSAGE_CODES
        and isinstance(payload, dict)
        and "registration_context" in payload
    ):
        # The engine already reported this registration state with its
        # reference, key thumbprint and notification e-mail. Repeating it in
        # the generic form would give the developer two lines for one state.
        return
    with _credential_failure_log_lock:
        if details["code"] in _credential_failure_logged:
            return
        _credential_failure_logged.add(details["code"])
    message = details["message"]
    action = details["action"]
    if action and not message.endswith(action):
        message = f"{message} {action}"
    log.warning("DeepintShield: %s (%s)", message, details["code"])


def _first_fragment(pattern: "re.Pattern[str]", limit: int, *candidates: object) -> str:
    for candidate in candidates:
        value = candidate.lower() if pattern is _REGISTRATION_EMAIL_RE and isinstance(candidate, str) else candidate
        fragment = _registration_fragment(value, pattern, limit)
        if fragment:
            return fragment
    return ""


_DISCOVERY_TTL_SECONDS = 3600.0
# Several execution/identity consumers may request the same credential binding
# concurrently during one gated call. Cache only fail-closed configuration
# codes for a very short window so a pending registration produces one gateway
# read, while an operator approval becomes visible on the next normal retry.
_CREDENTIAL_FAILURE_TTL_SECONDS = 0.25
# Blueprint transport probing stays bounded so a genuinely absent/older gateway
# cannot freeze ``compile()``. Once a modern server answers authoritatively,
# coverage and scanner failures are governance decisions and fail closed.
_BLUEPRINT_TIMEOUT_SECONDS = 5.0
_BLUEPRINT_CONFIGURATION_CODES = frozenset(
    {
        "invalid_blueprint_manifest",
        "blueprint_manifest_too_large",
        "blueprint_coverage_incomplete",
        "blueprint_remote_tool_unverified",
        "blueprint_mcp_inventory_unavailable",
        "blueprint_model_unavailable",
        "blueprint_registration_failed",
        "blueprint_scan_unavailable",
    }
)
_ENDPOINT_PROBE_TTL_SECONDS = 60.0
_AGENTIC_INCLUDED_PLANS = frozenset(
    {"business", "enterprise", "enterprise-vpc", "enterprise_vpc"}
)
_AGENTIC_EXCLUDED_PLANS = frozenset({"community", "developer", "team"})
_GAF_FIELDS = frozenset(
    {
        "agent",
        "user",
        "permission",
        "object",
        "delegation_id",
        "action",
        "action_class",
    }
)
def _prompt_digest(prompt: object) -> str:
    """SHA-256 of a prompt, or "" when there is none.

    The prompt itself never leaves the process by default. A digest still lets
    the server bind a decision to the exact text that drove it and notice when
    the same verdict is being replayed against different input, without the
    gateway ever holding the words - which is the zero-retention posture the
    rest of this path already keeps.
    """
    text = str(prompt or "").strip()
    if not text:
        return ""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class AgenticEngine:
    """Low-level gateway runtime for the canonical Agentic-New GAF data plane.

    GAF is the sole verdict whenever available. The legacy PDP is reachable
    only as an older-gateway compatibility fallback and never participates in
    a successful canonical decision. Applications normally reach this through
    ``shield.agentic`` so internal typed errors are translated to built-ins.
    """

    def __init__(
        self,
        parent: "DeepintShield",
        *,
        agent_credential: Optional[AgentCredential] = None,
        approval_poll_timeout_seconds: float = 300.0,
        approval_poll_interval_seconds: float = 2.0,
    ) -> None:
        # Avoid a client → surface → engine → client reference cycle. Framework
        # objects retain their concrete selected engine, while connection access
        # fails closed if the owning SDK client has been released.
        self._parent_ref = weakref.ref(parent)
        self._cred_info: Optional[VKCredentialInfo] = None
        self._cred_info_fetched_at: float = 0.0
        self._cred_info_failure_code = ""
        self._cred_info_failure_at = 0.0
        self._agent_credential: Optional[AgentCredential] = agent_credential
        # The credential this engine built from the gateway's provisional offer
        # to a pending registration, while it is the active one. Only its
        # failure to enrol can be read as the plain pending state; a supplied
        # or declared-identity credential never is.
        self._provisional_credential: Optional[AgentCredential] = None
        # The credential this engine built itself from a credential-info
        # answer, while it is the active one (never one the application
        # supplied), and when such a credential last failed under an answer
        # the gateway marked ``credential_optional`` (with that answer). Only
        # the server-sent flag makes a failure non-fatal; see
        # ``_optional_credential_failure``.
        self._built_credential: Optional[AgentCredential] = None
        self._optional_credential_failed_at = 0.0
        self._optional_credential_failed_for: Optional[VKCredentialInfo] = None
        # Provisional credentials set aside after the gateway refused their
        # key, so a concurrent call that met the same refusal reads the new
        # answer instead of surfacing it (see _set_aside_refused_provisional_key).
        self._set_aside_credentials: "weakref.WeakSet[object]" = weakref.WeakSet()
        # Whether this process can hold and use a gateway runtime key. None
        # until first asked, then fixed for the engine's lifetime: it is
        # judged from local facts alone (see ``_runtime_key_capable``).
        self._runtime_key_capability: Optional[bool] = None
        # Credential discovery/build is lazy and shared by every gated call.
        # An RLock makes the first fetch/build single-flight under concurrent
        # agent invocations while allowing agent_credential → credential_info.
        self._credential_lock = threading.RLock()
        # A DeepintShield client represents one agent identity, even when
        # several clients share the same virtual key.  Snapshot the selector
        # from the existing per-client agent_name so credential discovery,
        # token acquisition, and every later Agentic-New call cannot drift to
        # another binding if the parent attribute is mutated mid-run.
        raw_agent_name = str(getattr(parent, "agent_name", "") or "").strip()
        agent_key = _registry_key(raw_agent_name) if raw_agent_name else ""
        self._agent_subject_selector = f"agent:{agent_key}" if agent_key else ""
        requester = str(getattr(parent, "requester", "") or "").strip()
        self._requester_hint = "" if requester == "sdk-user" else requester
        # The registering developer: an explicit ``registered_by`` (possibly
        # "" for none), else an e-mail-shaped requester. Snapshotted like the
        # agent selector so a mutated parent cannot change the registrant of
        # an already filed sighting. The source travels with the address: the
        # gateway maps an explicit registrant to an identity in the workspace
        # directory and notifies it, and uses a requester-derived one only as
        # a fallback, because ``requester`` may name an end user the
        # organization has not placed in the workspace. Neither is verified,
        # and neither grants anything.
        registered_by = getattr(parent, "registered_by", None)
        registered_by_source = _REGISTERED_BY_EXPLICIT
        if registered_by is None:
            registered_by = _email_shaped(self._requester_hint)
            registered_by_source = _REGISTERED_BY_REQUESTER
        self._registered_by_email = str(registered_by or "")
        self._registered_by_source = (
            registered_by_source if self._registered_by_email else ""
        )
        # What the gateway told this process about its registration: reference,
        # state and the enrolled runtime key. Pure memory, filled from response
        # bodies the SDK already reads; it never triggers a request.
        self._registration_reference: dict[str, str] = {}
        self._runtime_key_thumbprint = ""
        self._registration_memory_lock = threading.Lock()
        self._default_principal: Optional[PrincipalBinding] = None
        self._default_principal_lock = threading.Lock()
        self._approval_timeout = approval_poll_timeout_seconds
        self._approval_interval = approval_poll_interval_seconds
        # ContextVars make one long-lived SDK client safe to share across async
        # requests and worker threads.  The stable default preserves the old
        # per-client session grouping; ``surface.run()`` / ``start_run()`` binds
        # a request-local run id without mutating other callers.
        self._default_session_id = self.new_session_id()
        self._session_context: contextvars.ContextVar[Optional[str]] = (
            contextvars.ContextVar(
                f"deepintshield_session_{id(self)}", default=None
            )
        )
        self._principal_context: contextvars.ContextVar[object] = (
            contextvars.ContextVar(
                f"deepintshield_principal_{id(self)}", default=_PRINCIPAL_UNSET
            )
        )
        # Discovery dedupe is scoped to this client instance. Identical graphs
        # owned by another VK/client must still register independently.
        self.registry_scope_id = uuid.uuid4().hex
        # Native framework run/compile reporters normally complete discovery
        # before authorization. The shared gate uses these fields for a
        # single-flight minimal fallback when a framework version exposes only
        # its final tool boundary.
        self._registration_capture_lock = threading.Lock()
        self._registration_capture_ready = False
        self._registration_captured_tools: set[str] = set()
        self._registration_captured_tool_attestations: dict[str, str] = {}
        self._registration_minimal_tools: set[str] = set()
        self._registration_capture_retry_at = 0.0
        self._registration_capture_retry_by_tool: dict[str, float] = {}
        self._registration_capture_error_code = ""
        self._gaf_unavailable_until = 0.0
        self._legacy_unavailable_until = 0.0
        # None until credential discovery runs, False only when the canonical
        # route explicitly returned a compatibility status (404/405/501).
        # This prevents a missing decide handler on a modern/GAF-aware gateway
        # from silently falling through to the retired PDP.
        self._canonical_credential_info_available: Optional[bool] = None

    @staticmethod
    def new_session_id() -> str:
        return "sdk-" + uuid.uuid4().hex[:12]

    @property
    def session_id(self) -> str:
        return self._session_context.get() or self._default_session_id

    @property
    def bound_session_id(self) -> str:
        """The request-local session override, excluding the client fallback.

        Automatic framework execution uses this to preserve an explicit
        ``shield.agentic.run(session_id=...)`` while still assigning a fresh id
        to every ordinary top-level framework invocation.
        """
        return str(self._session_context.get() or "")

    @property
    def execution_id(self) -> str:
        """Current first-class workflow execution id, when inside a run."""
        from .execution import current_execution_id

        return current_execution_id(self)

    @session_id.setter
    def session_id(self, value: str) -> None:
        clean = str(value or "").strip()
        self._session_context.set(clean or None)

    def bind_session(self, session_id: str = ""):
        """Bind a session in the current context and return a reset token."""
        clean = str(session_id or "").strip() or self.new_session_id()
        return self._session_context.set(clean)

    def reset_session(self, token: object) -> None:
        self._session_context.reset(token)  # type: ignore[arg-type]

    @property
    def principal_binding(self) -> Optional[PrincipalBinding]:
        scoped = self._principal_context.get()
        if scoped is not _PRINCIPAL_UNSET:
            return scoped if isinstance(scoped, PrincipalBinding) else None
        return self._requester_principal()

    def _requester_principal(self) -> Optional[PrincipalBinding]:
        """Lazily resolve ``DeepintShield(requester=...)`` on first gated use."""
        if not self._requester_hint:
            return None
        if self._default_principal is not None:
            return self._default_principal
        with self._default_principal_lock:
            if self._default_principal is not None:
                return self._default_principal
            from .identity import resolve_principal

            hint = self._requester_hint
            email = hint if "@" in hint else None
            username = None if email else hint
            subject = resolve_principal(self, email=email, username=username)
            self._default_principal = PrincipalBinding(
                subject=subject,
                email=(email or "").lower(),
                username=username or "",
                display_name=hint,
                kind="user",
            )
            return self._default_principal

    def bind_principal(self, binding: Optional[PrincipalBinding]):
        """Bind an acting principal in the current context; return reset token."""
        return self._principal_context.set(binding)

    def reset_principal(self, token: object) -> None:
        self._principal_context.reset(token)  # type: ignore[arg-type]

    @property
    def registered_by_email(self) -> str:
        """The developer filing this agent's registration; ``""`` when none."""
        return self._registered_by_email

    @property
    def registered_by_source(self) -> str:
        """Where :attr:`registered_by_email` came from; ``""`` when there is none.

        ``"explicit"`` for the ``registered_by`` setting
        (``DEEPINTSHIELD_REGISTERED_BY``), which the gateway maps to an
        identity in the workspace directory and notifies; ``"requester"`` for
        an address derived from an e-mail-shaped ``requester``, a fallback the
        gateway ignores, without refusing, for people the organization has not
        placed in the workspace. The address is not verified and grants
        nothing: the administrator confirms the owner.
        """
        return self._registered_by_source

    # ──────────────────────────────────────────────────────────────────
    # Registration memory (no network)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _registration_identity(body: dict, registration: dict) -> str:
        """The registration a gateway body is about, by the id of its row."""
        return _first_fragment(
            _REGISTRATION_REFERENCE_RE, 128,
            registration.get("id"), body.get("registration_id"), body.get("reference"),
        )

    def _note_registration(self, body: object) -> None:
        """Remember the registration a gateway body describes.

        Accepts a 202 discover envelope (``registration``/``registration_state``),
        a 409/403 credential-info body (``error`` + ``registration``), a
        ``registry/self`` body (``state``/``registration_id``) and the runtime
        broker's pending payload (``registration_id``). Every value is
        validated; nothing else in the body is retained.

        ``notify_email`` is remembered only from the gateway: ``notify_email``
        on a 409 or ``registry/self`` body, ``registration.requester_email`` on
        a 202 body. It is the gateway's confirmation that this registration has
        a recorded address to notify, and the only source of the "You will be
        notified at ..." sentence.
        """
        if not isinstance(body, dict):
            return
        registration = body.get("registration")
        if not isinstance(registration, dict):
            registration = {}
        reference = self._registration_identity(body, registration)
        # The reference the gateway prints for the developer. It is the row id
        # under the current contract; when a gateway shows something else, the
        # shown value is what later messages about this registration repeat.
        shown_reference = _first_fragment(
            _REGISTRATION_REFERENCE_RE, 128, body.get("reference"),
        )
        state = _first_fragment(
            _REGISTRATION_STATE_RE, 32,
            *(
                value.strip().lower() if isinstance(value, str) else value
                for value in (
                    body.get("registration_state"), registration.get("status"), body.get("state"),
                )
            ),
        )
        agent_subject = _first_fragment(
            _REGISTRATION_SUBJECT_RE, 198,
            registration.get("agent_subject"), body.get("agent_subject"),
        )
        notify_email = _first_fragment(
            _REGISTRATION_EMAIL_RE, _REGISTRATION_EMAIL_MAX_LENGTH,
            body.get("notify_email"), registration.get("requester_email"),
        )
        if reference or state or agent_subject or notify_email:
            with self._registration_memory_lock:
                memory = dict(self._registration_reference)
                if reference:
                    if memory.get("registration_id") not in (None, reference):
                        # A different registration. The address the gateway
                        # confirmed and the reference it showed belonged to
                        # the earlier one; each is kept only when this body
                        # states it again below.
                        memory.pop("notify_email", None)
                        memory.pop("reference", None)
                    memory["registration_id"] = reference
                    if shown_reference and shown_reference != reference:
                        memory["reference"] = shown_reference
                    elif shown_reference:
                        memory.pop("reference", None)
                if state:
                    memory["state"] = state
                if agent_subject:
                    memory["agent_subject"] = agent_subject
                if notify_email:
                    memory["notify_email"] = notify_email
                self._registration_reference = memory
        self._note_runtime_key(body.get("runtime_key_thumbprint"))

    def _note_runtime_key(self, thumbprint: object) -> None:
        """Remember the enrolled runtime key's RFC 7638 thumbprint."""
        value = _registration_fragment(thumbprint, _RUNTIME_KEY_THUMBPRINT_RE, 43)
        if value:
            self._runtime_key_thumbprint = value

    def _registration_known_pending(self) -> bool:
        """Whether the gateway has already told this process the row is pending.

        A ``credential_bootstrap`` block is read as pending unless the gateway
        marked it ``credential_optional``: that block belongs to an approved
        agent running on its virtual key. Nothing but that server-sent flag
        changes the reading, and a block that says ``registration_state:
        pending`` is pending whatever else it says.
        """
        if self._registration_reference.get("state") == "pending":
            return True
        if self._cred_info_failure_code == "agent_registration_pending":
            return True
        info = self._cred_info
        if info is not None and (
            (info.credential_bootstrap is True and not self._credential_optional(info))
            or str(info.registration_state or "").strip().lower() == "pending"
        ):
            return True
        return False

    def _registration_context(self, payload: object = None) -> dict[str, str]:
        """Validated reference/thumbprint/notification address for the developer message.

        A gateway echo in ``payload`` (``reference``, ``runtime_key_thumbprint``,
        ``notify_email``, the 202 ``registration`` object) wins over memory. The
        reference and the runtime-key thumbprint then fall back to what this
        process knows itself. A reference the gateway showed earlier for the
        same registration is repeated when ``payload`` carries none, so every
        message about one registration cites one reference.

        ``notify_email`` never does. It is the address the gateway confirmed it
        will notify, from this body or an earlier one about the same
        registration, and is empty when the gateway confirmed none. The local
        ``registered_by`` setting is deliberately not consulted: a gateway that
        predates the field, or one that set aside a requester-derived address,
        sends no e-mail, and the message must not promise one.
        """
        body = payload if isinstance(payload, dict) else {}
        registration = body.get("registration")
        if not isinstance(registration, dict):
            registration = {}
        memory = self._registration_reference
        try:
            credential_thumbprint = getattr(self._agent_credential, "key_thumbprint", "")
        except Exception:
            credential_thumbprint = ""
        # A remembered confirmation speaks for the registration it was given
        # for, not for another one this body may name. Identity is compared
        # the way it was remembered; the reference shown below may be a
        # different field of the same body.
        named = self._registration_identity(body, registration)
        same_registration = not named or memory.get("registration_id") in (None, named)
        remembered_notify_email = memory.get("notify_email") if same_registration else ""
        # The reference the gateway showed for this registration earlier. A
        # body that names the same registration without one, or no body at
        # all (the short negative cache, the gate), cites the same reference
        # the developer has already been given.
        remembered_reference = memory.get("reference") if same_registration else ""
        return {
            "registration_id": _first_fragment(
                _REGISTRATION_REFERENCE_RE, 128,
                body.get("reference"), remembered_reference, registration.get("id"),
                body.get("registration_id"), memory.get("registration_id"),
            ),
            "key_thumbprint": _first_fragment(
                _RUNTIME_KEY_THUMBPRINT_RE, 43,
                body.get("runtime_key_thumbprint"), body.get("key_thumbprint"),
                self._runtime_key_thumbprint, credential_thumbprint,
            ),
            "notify_email": _first_fragment(
                _REGISTRATION_EMAIL_RE, _REGISTRATION_EMAIL_MAX_LENGTH,
                body.get("notify_email"), registration.get("requester_email"),
                remembered_notify_email,
            ),
            "state": _first_fragment(
                _REGISTRATION_STATE_RE, 32,
                *(
                    value.strip().lower() if isinstance(value, str) else value
                    for value in (
                        body.get("registration_state"), registration.get("status"),
                        memory.get("state"),
                    )
                ),
            ),
        }

    def _attach_registration_context(self, exc: BaseException) -> None:
        """Give a registration-state error the context its public text needs."""
        code = normalize_agentic_error_code(getattr(exc, "code", ""), "")
        if code not in _REGISTRATION_MESSAGE_CODES:
            return
        payload = getattr(exc, "payload", None)
        if not isinstance(payload, dict):
            payload = {}
            try:
                exc.payload = payload  # type: ignore[attr-defined]
            except Exception:
                return
        self._note_registration(payload)
        self._note_runtime_key(payload.get("key_thumbprint"))
        payload["registration_context"] = self._registration_context(payload)

    def _observe_registration_error(self, exc: BaseException) -> None:
        """Attach context and print the single pending line once per process.

        Diagnostics only. It never raises, so the governance error it
        describes always reaches the caller unchanged. While a discovery
        report asks for workload proof the line is held back, and the report
        calls this again once its sighting has been answered
        (``registry._post_discover``): that answer may name the address the
        gateway will notify.
        """
        try:
            code = normalize_agentic_error_code(getattr(exc, "code", ""), "")
            if code not in _REGISTRATION_MESSAGE_CODES:
                return
            self._attach_registration_context(exc)
            from .registry import _log_enrolment_state

            payload = getattr(exc, "payload", None)
            _log_enrolment_state(
                {**(payload if isinstance(payload, dict) else {}), "error": code},
                engine=self,
            )
        except Exception as error:  # pragma: no cover - defensive
            log.debug("registration context unavailable: %s", type(error).__name__)

    @property
    def acting_user(self) -> str:
        binding = self.principal_binding
        return binding.subject if binding is not None else ""

    @acting_user.setter
    def acting_user(self, value: str) -> None:
        """Backward-compatible subject setter, now request-local."""
        subject = str(value or "").strip()
        self._principal_context.set(
            PrincipalBinding(subject=subject) if subject else None
        )

    # ──────────────────────────────────────────────────────────────────
    # Parent-backed connection details
    # ──────────────────────────────────────────────────────────────────

    def _parent(self) -> "DeepintShield":
        parent = self._parent_ref()
        if parent is None:
            raise GatewayUnavailable(reason="DeepintShield client has been released")
        return parent

    @property
    def gateway_url(self) -> str:
        return self._parent().base_url

    @property
    def virtual_key(self) -> str:
        parent = self._parent()
        try:
            return parent.virtual_key_or_raise()
        except SDKConfigurationError:
            raise GovernanceConfigurationError(
                framework="agentic",
                reason="virtual key is missing",
                code="virtual_key_missing",
            ) from None

    @property
    def _http(self) -> httpx.Client:
        return self._parent()._client

    # ──────────────────────────────────────────────────────────────────
    # Discovery + agent-credential bootstrap
    # ──────────────────────────────────────────────────────────────────

    def fetch_security_enabled(self, *, require_tasks: bool = False) -> bool:
        """Read the workspace's authenticated effective Agentic setting.

        This deliberately bypasses registration/credential bootstrap. Only an
        explicit, internally consistent disabled response permits native
        execution without Agentic calls; failed requests never permit it.
        """
        try:
            response = self._http.get(
                f"{self.gateway_url}/api/agentic-new/runtime-config",
                headers=self._headers(include_agent_token=False),
                timeout=3.0,
            )
        except DeepIntShieldError:
            raise
        except Exception as exc:
            raise GatewayUnavailable(reason=type(exc).__name__) from None
        if self._route_not_implemented(response):
            if require_tasks:
                raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_security_unavailable")
            return True
        if response.status_code >= 400:
            self._raise_agentic_http_error(response)
        # Gateways predating this endpoint serve their SPA index for unknown
        # GET paths, including /api paths, as 200 text/html. Retain the old
        # governed path in that case; HTML can never opt a workspace out.
        if (
            response.status_code == 200
            and response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            == "text/html"
        ):
            if require_tasks:
                raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_security_unavailable")
            return True
        try:
            data = response.json()
            enabled = data["security_enabled"]
            available = data["available"]
            plan = data["plan"]
            reason = data["reason"]
            valid = type(enabled) is bool and type(available) is bool
            if enabled:
                valid = (
                    valid and available and reason == "enabled"
                    and plan in _AGENTIC_INCLUDED_PLANS
                )
            elif reason == "plan_not_included":
                valid = valid and not available and plan in _AGENTIC_EXCLUDED_PLANS
            elif reason == "workspace_disabled":
                valid = valid and available and plan in _AGENTIC_INCLUDED_PLANS
            else:
                valid = False
            if not valid:
                raise ValueError("invalid runtime configuration")
            if require_tasks and (not enabled or data.get("task_mandates_supported") is not True):
                raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_security_unavailable")
            return enabled
        except DeepIntShieldError:
            raise
        except Exception:
            raise DeepIntShieldError(code=AGENT_INVALID_GATEWAY_RESPONSE) from None

    @property
    def credential_info(self) -> VKCredentialInfo:
        """Lazy-loaded discovery info. Cached for 1h; auto-refreshed on a
        401/403 from /decide."""
        failure_now = time.monotonic()
        if (
            self._cred_info_failure_code
            and failure_now - self._cred_info_failure_at
            <= _CREDENTIAL_FAILURE_TTL_SECONDS
        ):
            raise self._cached_credential_failure() from None
        now = time.time()
        if (
            self._cred_info is not None
            and now - self._cred_info_fetched_at <= _DISCOVERY_TTL_SECONDS
        ):
            return self._cred_info
        with self._credential_lock:
            failure_now = time.monotonic()
            if (
                self._cred_info_failure_code
                and failure_now - self._cred_info_failure_at
                <= _CREDENTIAL_FAILURE_TTL_SECONDS
            ):
                raise self._cached_credential_failure() from None
            now = time.time()
            if (
                self._cred_info is None
                or now - self._cred_info_fetched_at > _DISCOVERY_TTL_SECONDS
            ):
                try:
                    fetched = self._fetch_credential_info()
                except GovernanceConfigurationError as exc:
                    self._cred_info_failure_code = exc.code
                    self._cred_info_failure_at = time.monotonic()
                    raise
                self._cred_info = fetched
                self._cred_info_fetched_at = time.time()
                self._cred_info_failure_code = ""
                self._cred_info_failure_at = 0.0
            return self._cred_info

    def _cached_credential_failure(self) -> GovernanceConfigurationError:
        """The short negative cache re-raises with the remembered context."""
        error = GovernanceConfigurationError(code=self._cred_info_failure_code)
        try:
            self._attach_registration_context(error)
        except Exception:  # pragma: no cover - defensive
            pass
        return error

    def set_agent_credential(self, cred: AgentCredential) -> None:
        """Override the auto-built credential (advanced; usually for tests
        where you want a known-good static token)."""
        with self._credential_lock:
            self._agent_credential = cred
            self._provisional_credential = None
            self._forget_optional_credential()
            try:
                self._set_aside_credentials.discard(cred)
            except TypeError:  # pragma: no cover - not weakly referenceable
                pass

    def _clear_credential_failure_cache(self) -> None:
        """Forget a short-lived fail-closed lookup after discovery changed state."""
        with self._credential_lock:
            self._cred_info_failure_code = ""
            self._cred_info_failure_at = 0.0

    @property
    def agent_subject(self) -> str:
        """Canonical principal selected from this VK's Registry associations.

        The selector header asks the gateway for this client's Registry profile,
        but the returned credential-info value remains authoritative. Older
        gateways omit the field; GAF requests then send an empty claim so the
        server can fill it from the authenticated virtual-key context.
        """
        return str(self.credential_info.agent_subject or "").strip()

    @property
    def agent_credential(self) -> Optional[AgentCredential]:
        """Returns the active credential, building one lazily from the
        discovered info if no override was set."""
        if self._agent_credential is not None:
            return self._agent_credential
        with self._credential_lock:
            if self._agent_credential is not None:
                return self._agent_credential
            info = self.credential_info
            if not info.agent_configured and not info.runtime_auth_mode:
                # LLM-only VK - no agent token needed.
                return None
            if self._optional_credential_resting(info):
                # The gateway marked this credential optional and building it
                # failed moments ago: the agent runs on its virtual key.
                return None
            try:
                credential = self._build_credential_for(info)
            except Exception as exc:
                if self._credential_optional(info):
                    # An approved agent the gateway admits on its virtual key:
                    # the declared identity it cannot build here is optional.
                    self._rest_optional_credential(info, exc, None)
                    return None
                if not isinstance(exc, DeepIntShieldError):
                    raise
                pending = self._provisional_key_fallback(exc, info)
                if pending is None:
                    raise
                raise pending from None
            self._agent_credential = credential
            self._provisional_credential = (
                credential if self._provisional_pending(info) else None
            )
            self._built_credential = credential
            return self._agent_credential

    # ──────────────────────────────────────────────────────────────────
    # PDP calls
    # ──────────────────────────────────────────────────────────────────

    def decide(self, dc: DelegationContext) -> Decision:
        """Synchronously obtain the canonical Agentic-New authorization result.

        Retries once with fresh discovery info on 401/403 to handle a Registry
        profile or credential change made mid-process. A legacy decision is
        accepted only from a gateway that does not implement either canonical
        Agentic-New route."""
        from .runtime import passthrough_decision, security_enabled

        if not security_enabled(self):
            return passthrough_decision()
        # Stamp the request-local session unless the caller supplied one.
        if not dc.session_id:
            dc.session_id = self.session_id
        for attempt in range(2):
            try:
                return self._decide_once(dc)
            except DeepIntShieldError:
                raise
            except httpx.HTTPStatusError as exc:
                if attempt == 0 and exc.response.status_code in (401, 403):
                    with self._credential_lock:
                        self._cred_info = None
                        self._cred_info_fetched_at = 0.0
                        self._cred_info_failure_code = ""
                        self._cred_info_failure_at = 0.0
                        self._agent_credential = None
                        self._provisional_credential = None
                        self._forget_optional_credential()
                    continue
                self._raise_agentic_http_error(exc.response)
            except httpx.HTTPError as exc:
                raise GatewayUnavailable(reason=type(exc).__name__) from None
            except Exception as exc:
                # Framework applications should never need to understand an
                # httpx/Pydantic/credential implementation exception. Retain
                # the type for diagnostics, but expose only the public code.
                raise GatewayUnavailable(reason=type(exc).__name__) from None
        raise GatewayUnavailable(code=AGENT_GATEWAY_UNAVAILABLE)  # pragma: no cover

    def register_blueprint(
        self,
        manifest: object,
        *,
        timeout: float = _BLUEPRINT_TIMEOUT_SECONDS,
    ) -> Optional[str]:
        """Register the agent's declared tool surface (manifest) with the server
        BEFORE the run - the server stores the declared topology for full-graph
        visualization, policy pre-validation, and declared-vs-observed drift.

        Transport failure and an older gateway without this compatibility route
        remain non-fatal. Any response from an implemented route is
        authoritative: incomplete source coverage, an unverified MCP claim or a
        required scanner failure raises a stable code-only Agentic exception.
        Returns the server-assigned blueprint id, or None for the two legacy
        compatibility cases."""
        from .runtime import security_enabled

        if not security_enabled(self):
            return None
        try:
            payload = manifest.to_dict() if hasattr(manifest, "to_dict") else manifest
            # Stamp the per-process session + principal so the blueprint binds to
            # the same execution the decisions will carry.
            if isinstance(payload, dict):
                payload.setdefault("session_id", self.session_id)
            resp = self._http.post(
                f"{self.gateway_url}/api/agentic-security/blueprints",
                json=payload,
                headers=self._headers(include_agent_token=False),
                timeout=timeout,
            )
        except DeepIntShieldError:
            raise
        except httpx.HTTPError:
            # Optional compatibility transport: do not emit exception prose to
            # stderr. The canonical first-run discovery/decision boundary still
            # fails closed independently when the gateway is required.
            log.debug("blueprint_registration_transport_unavailable")
            return None
        except Exception:
            raise GovernanceConfigurationError(
                framework="agentic",
                code="invalid_blueprint_manifest",
            ) from None

        if self._route_not_implemented(resp):
            log.debug("blueprint_registration_route_unavailable")
            return None
        if resp.status_code >= 400:
            code = self._response_error_code(resp)
            if code in _BLUEPRINT_CONFIGURATION_CODES:
                raise self._agentic_configuration_error(resp) from None
            self._raise_agentic_http_error(resp)
        try:
            body = resp.json()
        except Exception:
            raise DeepIntShieldError(code=AGENT_INVALID_GATEWAY_RESPONSE) from None
        blueprint_id = body.get("blueprint_id") if isinstance(body, dict) else None
        if not isinstance(blueprint_id, str) or not blueprint_id.strip():
            raise DeepIntShieldError(code=AGENT_INVALID_GATEWAY_RESPONSE) from None
        return blueprint_id.strip()

    def poll_approval(self, decision_id: str) -> Decision:
        """Poll an approval produced by the older-gateway compatibility PDP.

        Canonical Agentic-New approvals never call this method: they return
        control immediately with their canonical approval id so the workflow
        can persist and resume without a second authorization data source.
        """
        deadline = time.time() + self._approval_timeout
        while time.time() < deadline:
            try:
                resp = self._http.get(
                    f"{self.gateway_url}/api/agentic-security/approvals/{decision_id}",
                    headers=self._headers(include_agent_token=False),
                )
            except DeepIntShieldError:
                raise
            except Exception as exc:
                raise GatewayUnavailable(reason=type(exc).__name__) from None
            if resp.status_code >= 500 or resp.status_code in (400, 401, 403, 409):
                self._raise_agentic_http_error(resp)
            if resp.status_code == 200:
                # Be defensive: a transient/non-JSON body (proxy error page, SPA
                # fallback) must not crash the poll - treat it as "still pending"
                # and keep waiting until the deadline (→ GuardrailApprovalPending).
                try:
                    state = resp.json().get("state", "pending")
                except Exception:  # noqa: BLE001
                    state = "pending"
                if state == "approved":
                    return Decision(verdict=Verdict.ALLOW, decision_id=decision_id)
                if state in ("denied", "expired"):
                    return Decision(
                        verdict=Verdict.DENY,
                        decision_id=decision_id,
                        reason="approval denied",
                    )
            time.sleep(self._approval_interval)
        raise GuardrailApprovalPending(
            decision_id=decision_id,
            code=AGENT_APPROVAL_PENDING,
        ) from None

    # ──────────────────────────────────────────────────────────────────
    # Token injection (consumed by the L1 transport too)
    # ──────────────────────────────────────────────────────────────────

    def agent_token(self, *, fail_closed: bool = False) -> Optional[str]:
        """Return a fresh agent token, or None if this VK has no agent
        identity configured.

        Transparent LLM transport keeps the historical best-effort behavior.
        Authorization calls pass ``fail_closed=True``: if a VK is configured
        for workload identity, a credential/setup/token failure must block
        before the PDP request rather than silently dropping the agent proof.

        Two gateway statements change that, on both paths:

        * an answer marked ``credential_optional`` (an approved agent the
          gateway admits on its virtual key): a failure to build its
          credential or obtain its token sends no token, logs one line per
          process and is not retried for a minute;
        * a provisional runtime key refused at enrolment or challenge with
          ``agent_runtime_authentication_failed``: the offer is set aside and
          credential-info is asked once more, and the call continues with the
          new answer. Exactly one re-read; every other failure is unchanged.
        """
        from .runtime import scoped_security_enabled

        if scoped_security_enabled(self) is False:
            return None
        if fail_closed:
            try:
                return self._fail_closed_agent_token()
            except ImportError as exc:
                raise GovernanceConfigurationError(
                    framework="agent-credential",
                    reason=type(exc).__name__,
                ) from None
            except DeepIntShieldError as exc:
                # A pending registration or runtime key stops the governed
                # call here. Remember what the gateway said (reference, key
                # thumbprint) and print the one developer-facing line.
                self._observe_registration_error(exc)
                raise
            except Exception as exc:
                raise GatewayUnavailable(
                    reason=type(exc).__name__
                ) from None
        return self._best_effort_agent_token()

    def _fail_closed_agent_token(self) -> Optional[str]:
        """The authorization path of :meth:`agent_token`; errors are translated there."""
        # A second pass happens only after a refused provisional key was set
        # aside and credential-info answered again: one re-read, never a loop.
        for attempt in range(2):
            cred = self.agent_credential
            if cred is None:
                return None
            if self._optional_credential_resting_for(cred):
                return None
            try:
                token = str(cred.get_token() or "").strip()
            except Exception as exc:
                if attempt == 0 and self._set_aside_refused_provisional_key(cred, exc):
                    self._reread_after_refused_provisional_key()
                    continue
                if self._optional_credential_failure(cred, exc):
                    return None
                if isinstance(exc, DeepIntShieldError):
                    pending = self._provisional_token_fallback(cred, exc)
                    if pending is not None:
                        raise pending from None
                raise
            if not token:
                error = GovernanceConfigurationError(
                    framework="agent-credential",
                    reason="configured credential returned an empty agent token",
                )
                if self._optional_credential_failure(cred, error):
                    return None
                raise error
            return token
        return None  # pragma: no cover - every pass returns or raises

    def _best_effort_agent_token(self) -> Optional[str]:
        """The transparent-traffic path of :meth:`agent_token`."""
        for attempt in range(2):
            try:
                cred = self.agent_credential
            except ImportError as exc:
                if self._broker_identity_required():
                    raise
                _log_credential_failure(exc, "credential_dependency_missing")
                return None
            except Exception as exc:  # discovery failure shouldn't break LLM traffic
                if self._broker_identity_required():
                    raise
                _log_credential_failure(exc, "credential_configuration_error")
                return None
            if cred is None:
                return None
            if self._optional_credential_resting_for(cred):
                return None
            try:
                return cred.get_token()
            except Exception as exc:
                if attempt == 0 and self._set_aside_refused_provisional_key(cred, exc):
                    try:
                        self._reread_after_refused_provisional_key()
                    except Exception as reread_error:
                        # The offer is gone, so this reads as any lookup
                        # failure of transparent traffic: a pending answer
                        # has been reported and the traffic goes on.
                        if self._broker_identity_required():
                            raise
                        _log_credential_failure(reread_error, "credential_configuration_error")
                        return None
                    continue
                if self._optional_credential_failure(cred, exc):
                    return None
                if self._provisional_token_fallback(cred, exc) is not None:
                    # The registration is simply pending, and has been reported as
                    # such: transparent traffic goes on without an agent token,
                    # as it does when the gateway answers a pending lookup itself.
                    return None
                if getattr(cred, "provider_type", "") == "gateway_broker":
                    raise
                _log_credential_failure(exc, "credential_exchange_failed")
                return None
        return None  # pragma: no cover - every pass returns or raises

    def _broker_identity_required(self) -> bool:
        """Whether the cached answer names a broker identity traffic must present.

        A gateway-broker answer fails closed even on transparent traffic,
        unless the gateway marked it ``credential_optional``.
        """
        info = self._cred_info
        return (
            info is not None
            and info.runtime_auth_mode == "gateway_broker"
            and not self._credential_optional(info)
        )

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _decide_once(self, dc: DelegationContext) -> Decision:
        """One canonical decision attempt, with an older-gateway fallback.

        Once GAF returns a valid response, its verdict and evidence are final.
        The legacy route is contacted only after the canonical route is absent
        *and* canonical credential discovery already established that this is
        an older gateway.
        """
        gaf_call = self._post_gaf_decide(dc)
        if gaf_call is not None:
            gaf, gaf_decision_id = gaf_call
            return self._decision_from_gaf(gaf, gaf_decision_id)

        from .task_context import current_task_context, require_task_security
        if current_task_context() is not None:
            require_task_security(False)

        legacy = self._post_legacy_compat_decide(dc)
        if legacy is None:
            raise GatewayUnavailable(
                reason=(
                    "canonical Agentic-New is not implemented by this older "
                    "gateway and its compatibility PDP is unavailable"
                )
            )
        proceed = (
            legacy.proceed
            if legacy.proceed is not None
            else legacy.verdict in (Verdict.ALLOW, Verdict.MASK)
        )
        return legacy.model_copy(
            update={
                # Preserve truthful evidence: no GAF verdict exists on this
                # compatibility path, and callers can identify it explicitly.
                "gaf_checked": False,
                "gaf_verdict": "",
                "mode": "legacy-compat",
                "proceed": bool(proceed),
            }
        )

    def _post_gaf_decide(
        self, dc: DelegationContext
    ) -> Optional[tuple[GAFDecision, str]]:
        if time.monotonic() < self._gaf_unavailable_until:
            return None
        payload = self._gaf_payload(dc)
        resp = self._http.post(
            f"{self.gateway_url}/api/agentic-new/decide",
            json=payload,
            headers=self._headers(
                include_agent_token=True, path="/api/agentic-new/decide"
            ),
        )
        if self._route_not_implemented(resp):
            from .task_context import current_task_context, require_task_security
            if current_task_context() is not None:
                require_task_security(False)
            # _gaf_payload() normally runs credential discovery first. The
            # explicit check also covers callers that supplied both a custom
            # agent principal and credential override.
            if self._canonical_credential_info_available is None:
                _ = self.credential_info
            if self._canonical_credential_info_available is not False:
                raise GovernanceConfigurationError(
                    framework="agentic-new",
                    reason=(
                        "the canonical credential-info route is available but "
                        "the canonical decision route is missing"
                    ),
                )
            self._gaf_unavailable_until = (
                time.monotonic() + _ENDPOINT_PROBE_TTL_SECONDS
            )
            return None
        if self._endpoint_absent(resp, "GAF PDP"):
            # An explicitly unconfigured GAF route is a modern gateway
            # configuration error, not evidence of an older gateway.
            raise GovernanceConfigurationError(
                framework="agentic-new",
                reason="the canonical Agentic-New PDP is not configured",
            )
        if resp.status_code >= 500:
            self._raise_agentic_http_error(resp)
        resp.raise_for_status()
        try:
            decision = GAFDecision.model_validate(resp.json())
        except Exception as exc:
            raise DeepIntShieldError(
                type(exc).__name__,
                code=AGENT_INVALID_GATEWAY_RESPONSE,
            ) from None
        if payload.get("task_mandate"):
            from .task_context import capture_task_receipt, require_task_security

            proceeds = decision.proceed if decision.proceed is not None else decision.allow
            if proceeds:
                # A gateway ignoring the new fields must never authorize a
                # task by accidentally falling back to its ordinary decision.
                require_task_security(bool(
                    decision.allow and decision.verdict == Verdict.ALLOW
                    and decision.task_receipt and decision.task_id
                ))
                capture_task_receipt(decision.task_id, decision.task_receipt)
        decision_id = (
            decision.decision_id
            or decision.approval_id
            or resp.headers.get("x-request-id", "").strip()
            or "gaf-" + uuid.uuid4().hex[:16]
        )
        return decision, decision_id

    def _post_legacy_compat_decide(
        self, dc: DelegationContext
    ) -> Optional[Decision]:
        """Call the retired PDP only for a proven older-gateway fallback."""
        from .task_context import current_task_context, require_task_security
        if current_task_context() is not None:
            require_task_security(False)
        if self._canonical_credential_info_available is not False:
            raise GovernanceConfigurationError(
                framework="agentic-new",
                reason=(
                    "legacy authorization fallback was refused because this "
                    "gateway exposes canonical Agentic-New capabilities"
                ),
            )
        if time.monotonic() < self._legacy_unavailable_until:
            return None
        # Agentic-New-only fields are never sent to the legacy Go contract.
        payload = dc.model_dump(mode="json", exclude=_GAF_FIELDS)
        resp = self._http.post(
            f"{self.gateway_url}/api/agentic-security/decide",
            json=payload,
            headers=self._headers(
                include_agent_token=True, path="/api/agentic-security/decide"
            ),
        )
        if self._route_not_implemented(resp) or self._endpoint_absent(
            resp, "legacy PDP"
        ):
            self._legacy_unavailable_until = (
                time.monotonic() + _ENDPOINT_PROBE_TTL_SECONDS
            )
            return None
        if resp.status_code >= 500:
            self._raise_agentic_http_error(resp)
        resp.raise_for_status()
        try:
            return Decision.model_validate(resp.json())
        except Exception as exc:
            raise DeepIntShieldError(
                type(exc).__name__,
                code=AGENT_INVALID_GATEWAY_RESPONSE,
            ) from None

    # Backward-compatible private name, now preserving canonical semantics
    # instead of exposing a hidden legacy authorization path.
    def _post_decide(self, dc: DelegationContext) -> Decision:
        return self._decide_once(dc)

    def _gaf_payload(self, dc: DelegationContext) -> dict[str, object]:
        from .task_context import task_payload
        raw_tool = str(dc.tool or "").strip()
        tool_name = (
            raw_tool.split(":", 1)[1]
            if raw_tool.lower().startswith("tool:")
            else raw_tool
        )
        tool_key = _registry_key(tool_name)
        action = str(dc.action or "").strip()
        action_class = (
            str(dc.action_class or "").strip().lower()
            # A named operation is more specific than its containing tool. This
            # matters for multi-operation tools such as read_records/delete:
            # without an explicit class, the destructive action must select the
            # write permission/HITL path rather than inherit the tool's read name.
            or _action_class(action or tool_key)
        )
        # Only an explicit GAF claim or the server-selected Registry principal is
        # sent. The Go data plane rejects subjects that do not match a persisted
        # association and fills an empty claim when selection is unambiguous.
        agent = str(dc.agent or "").strip() or self.agent_subject

        user = str(dc.user or "").strip() or self.acting_user
        if not user:
            user = next(
                (
                    str(subject).strip()
                    for subject in dc.actor_chain
                    if str(subject).strip().startswith(("user:", "service_account:"))
                ),
                "",
            )

        permission = str(dc.permission or "").strip() or "holder"
        object_id = str(dc.object or "").strip()
        if not object_id:
            # Discovery declares this exact default for the agent review.
            # The reviewed access plan includes it only after an operator
            # confirms approval. Match Go PermissionSubject; missing tuples deny.
            object_id = _permission_subject(f"{tool_key}:{action_class}")
        tool_subject = f"tool:{tool_key}"
        return {
            "agent": agent,
            "user": user,
            "permission": permission,
            "object": object_id,
            "tool": tool_subject,
            # Empty is meaningful only when exactly one registered action
            # exists: the server selects that sole action. Actionless tools are
            # denied; discovery always supplies a conservative `invoke` action.
            "action": action,
            "delegation_id": str(dc.delegation_id or "").strip(),
            "action_class": action_class,
            "args_digest": str(dc.args_digest or "").strip(),
            # One ordinary framework invocation owns one execution/session.
            # The server also accepts these values in headers, but carrying
            # them in the body keeps MockTransport/custom transports and audit
            # replay deterministic.
            "execution_id": self.execution_id or str(dc.session_id or "").strip(),
            "session_id": str(dc.session_id or self.session_id).strip(),
            # ── context the canonical PDP could not previously see ──────────
            #
            # These four travelled only to the legacy engine, so the SDK's own
            # docstrings described operands the enforced path never received:
            # decide(prompt=...) was never scanned, recovery_cost never
            # escalated, tool_fingerprint never bound, and delegation depth was
            # unreachable because actor_chain stopped at the legacy body.
            #
            # prompt is sent as a DIGEST by default. The raw text is scanned
            # server-side only when a workspace opts in, so zero-retention
            # stays the default rather than something to remember.
            "tool_fingerprint": str(
                getattr(dc.context, "tool_fingerprint", "") or ""
            ).strip(),
            "recovery_cost": str(getattr(dc.context, "recovery_cost", "") or "").strip(),
            "actor_chain": [
                str(subject).strip() for subject in (dc.actor_chain or []) if str(subject).strip()
            ],
            "prompt_digest": _prompt_digest(getattr(dc, "prompt", "")),
            **task_payload(),
        }

    @staticmethod
    def _decision_from_gaf(gaf: GAFDecision, decision_id: str) -> Decision:
        proceed = gaf.proceed if gaf.proceed is not None else bool(gaf.allow)
        if proceed:
            effective_verdict = Verdict.ALLOW
        elif gaf.verdict == Verdict.REQUIRE_APPROVAL:
            effective_verdict = Verdict.REQUIRE_APPROVAL
        else:
            # ``proceed`` is the authoritative execution bit. A malformed or
            # forward-version response such as verdict=ALLOW/proceed=false must
            # not pass through merely because the enum looks non-blocking.
            effective_verdict = Verdict.DENY
        return Decision(
            verdict=effective_verdict,
            reason=gaf.reason,
            decision_id=decision_id,
            mode=gaf.mode,
            latency_us=gaf.latency_us,
            gaf_checked=True,
            gaf_verdict=gaf.verdict.value,
            failed_check=gaf.failed_check,
            checks=gaf.checks,
            proceed=proceed,
            would_block=gaf.would_block,
            approval_id=gaf.approval_id,
            task_id=gaf.task_id,
            task_receipt=gaf.task_receipt,
        )

    @staticmethod
    def _route_not_implemented(resp: httpx.Response) -> bool:
        """True only for statuses that identify an absent route surface.

        410 is included because a deployment that has finished migrating off
        the legacy PDP retires it with Gone rather than Not Found - and Gone is
        the more definite of the two. Without it the fallback would raise a raw
        transport error and re-probe a route the server has said is permanently
        removed.
        """
        return resp.status_code in (404, 405, 410, 501)

    @staticmethod
    def _response_error_detail(resp: httpx.Response) -> str:
        """Extract a bounded message from flat or DeepIntShield error envelopes."""

        def find_message(value: object, seen: set[int]) -> str:
            if isinstance(value, str):
                return value.strip()
            if not isinstance(value, dict) or id(value) in seen:
                return ""
            seen.add(id(value))
            for key in ("message", "detail", "error"):
                message = find_message(value.get(key), seen)
                if message:
                    return message
            return ""

        try:
            detail = find_message(resp.json(), set())
        except Exception:
            detail = resp.text.strip()
        # Do not let an intermediary HTML page or oversized upstream error flood
        # exception logs. Agentic server messages are intentionally short.
        return detail[:500]

    @staticmethod
    def _response_error_code(resp: httpx.Response) -> str:
        """Read only documented public-code envelope fields.

        Arbitrary nested objects may contain third-party ``code`` attributes;
        recursively promoting those values would turn payload data into the
        SDK's public exception contract.
        """
        try:
            body = resp.json()
        except Exception:
            return ""
        if not isinstance(body, dict):
            return ""
        error = body.get("error")
        if isinstance(error, dict):
            code = normalize_agentic_error_code(error.get("code"), "")
            if code:
                return code
        return normalize_agentic_error_code(body.get("code"), "")

    def _agentic_configuration_error(
        self, resp: httpx.Response
    ) -> GovernanceConfigurationError:
        """Preserve the server code and leave human remediation to Agentic UI."""
        detail = self._response_error_detail(resp)
        error = GovernanceConfigurationError(
            framework="agentic-new",
            reason=detail,
            code=self._response_error_code(resp) or AGENT_CONFIGURATION_ERROR,
            status_code=resp.status_code,
            payload=self._response_json_object(resp),
        )
        # A registration-state body (credential-info 409/403) names the
        # registration; keep it for the developer message and log it once.
        self._observe_registration_error(error)
        return error

    def _raise_agentic_http_error(self, resp: httpx.Response) -> None:
        """Translate every Agentic HTTP failure to the public code contract."""
        if resp.status_code < 500:
            raise self._agentic_configuration_error(resp) from None
        raise GatewayUnavailable(
            reason=self._response_error_detail(resp),
            code=self._response_error_code(resp) or AGENT_GATEWAY_UNAVAILABLE,
            status_code=resp.status_code,
            payload=self._response_json_object(resp),
        ) from None

    @staticmethod
    def _response_json_object(resp: httpx.Response) -> dict:
        try:
            body = resp.json()
        except Exception:
            return {}
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _endpoint_absent(resp: httpx.Response, label: str) -> bool:
        if resp.status_code != 503:
            return False
        # The Agentic-New handler uses 503 to say that no OpenFGA store is
        # configured. A generic 503 is an outage and must fail closed.
        try:
            body = resp.json()
            detail = " ".join(
                str(body.get(k, "")) for k in ("error", "message", "detail")
            )
        except Exception:
            detail = resp.text
        absent = "not enabled" in detail.lower() or "not configured" in detail.lower()
        if absent:
            log.debug("%s unavailable: %s", label, detail)
        return absent

    def _fetch_credential_info(self) -> VKCredentialInfo:
        # Agentic-New is the canonical identity-provider store. Keep the old
        # discovery route as a one-release compatibility fallback so an updated
        # SDK can still run against an older gateway, but never fall back after
        # an authentication/configuration failure (403/409/5xx): those must stay
        # fail closed.
        try:
            resp = self._http.get(
                f"{self.gateway_url}/api/agentic-new/credential-info",
                # A client able to hold a runtime key says so here, and only
                # then does the gateway offer a pending registration its
                # provisional key. The older route below predates that key
                # and is asked exactly as before.
                headers={
                    **self._headers(include_agent_token=False),
                    **self._provisional_key_headers(),
                },
                follow_redirects=False,
            )
        except DeepIntShieldError:
            raise
        except Exception as exc:
            raise GatewayUnavailable(reason=type(exc).__name__) from None
        canonical_route = not self._route_not_implemented(resp)
        if not canonical_route:
            self._canonical_credential_info_available = False
            try:
                resp = self._http.get(
                    f"{self.gateway_url}/api/agentic-security/vk-credential-info",
                    headers=self._headers(include_agent_token=False),
                    follow_redirects=False,
                )
            except DeepIntShieldError:
                raise
            except Exception as exc:
                raise GatewayUnavailable(reason=type(exc).__name__) from None
        else:
            # Even an auth/config error proves the modern route exists; record
            # this before raise_for_status so decide can never downgrade.
            self._canonical_credential_info_available = True
        if resp.status_code >= 400:
            self._raise_agentic_http_error(resp)
        try:
            return VKCredentialInfo.model_validate(resp.json())
        except Exception as exc:
            raise DeepIntShieldError(
                type(exc).__name__,
                code=AGENT_INVALID_GATEWAY_RESPONSE,
            ) from None

    # ── provisional runtime key: capability and fallback ───────────────

    def _runtime_key_capable(self) -> bool:
        """Whether this process can hold and use a gateway runtime key.

        Decided once per engine, on first use, from local facts alone: the
        ``runtime`` extra imports, the runtime key store is usable and the
        gateway address is one the broker credential accepts. Nothing is
        requested, no key is generated and nothing is written; the rules are
        the credential's own (``gateway_broker.runtime_key_capable``).

        Judging it never fails the lookup that asked: an error while judging,
        including an optional dependency that cannot be imported, means "not
        capable", and the client keeps the flow it had before the capability
        existed.
        """
        capable = self._runtime_key_capability
        if capable is None:
            with self._credential_lock:
                capable = self._runtime_key_capability
                if capable is None:
                    try:
                        from .credentials.gateway_broker import runtime_key_capable

                        capable = runtime_key_capable(self.gateway_url) is True
                    except Exception:
                        capable = False
                    except BaseException as error:
                        # A broken native build of the optional dependency
                        # fails its import with pyo3's PanicException, which
                        # is not an ``Exception``. This lookup is the first
                        # place such a process imports it, so it must not be
                        # the place a virtual-key flow starts failing.
                        # Interrupts and exits are never absorbed.
                        if type(error).__name__ != "PanicException":
                            raise
                        capable = False
                    self._runtime_key_capability = capable
        return capable

    def _provisional_key_headers(self) -> dict[str, str]:
        """The header that asks the gateway for a provisional runtime key.

        Empty unless this process can hold such a key, so a client that
        cannot keeps receiving the pending answer it always received.
        """
        if not self._runtime_key_capable():
            return {}
        from .credentials.gateway_broker import provisional_key_headers

        return provisional_key_headers(True)

    @staticmethod
    def _provisional_pending(info: Optional[VKCredentialInfo]) -> bool:
        """Whether ``info`` is the provisional offer to a pending registration.

        The block has to say both: a ``provisional`` key, and a registration
        whose state is ``pending``. Nothing else the engine remembers can
        stand in for either statement.
        """
        return (
            info is not None
            and getattr(info, "provisional", False) is True
            and str(getattr(info, "registration_state", "") or "").strip().lower() == "pending"
        )

    def _provisional_key_fallback(
        self, exc: BaseException, info: Optional[VKCredentialInfo],
    ) -> Optional[GovernanceConfigurationError]:
        """Read a provisional runtime key this process cannot use as pending.

        A gateway offers a pending registration its provisional key only to a
        client that asked for one, but an older gateway offers it to every
        client, and a key store can prove unusable for one scope. When
        building or enrolling that key fails for a local reason, nothing is
        wrong with the registration: it is pending, as it was before the
        offer existed. This returns the ``agent_registration_pending`` error
        to raise in place of ``exc``, exactly as if the gateway had answered
        the lookup with it. The sighting is then filed with the virtual key
        alone and the developer reads the plain pending line, which names no
        runtime key because this process holds none it could cite.

        The offer and its key are set aside, so the next lookup asks the
        gateway again and an administrator's decision is seen on the next run
        or retry.

        Returns None, and ``exc`` must surface unchanged, for any other
        failure and for any block that is not a provisional offer to a
        pending registration. A declared or approved identity never has its
        credential errors read as "pending".
        """
        if not self._provisional_pending(info):
            return None
        code = normalize_agentic_error_code(getattr(exc, "code", ""), "")
        if code not in _PROVISIONAL_KEY_UNUSABLE_CODES:
            return None
        return self._provisional_offer_pending(info, code)

    def _provisional_offer_pending(
        self, info: Optional[VKCredentialInfo], reason: str,
    ) -> GovernanceConfigurationError:
        """Set a provisional offer aside and return the plain pending error.

        The engine is left exactly where a 409 ``agent_registration_pending``
        from the gateway leaves it, including the short negative cache, and
        the developer reads the pending line without a runtime-key clause.
        """
        with self._credential_lock:
            if self._cred_info is info:
                # The same state a 409 from the gateway leaves behind,
                # including its short negative cache.
                self._cred_info = None
                self._cred_info_fetched_at = 0.0
                self._cred_info_failure_code = _REGISTRATION_PENDING_CODE
                self._cred_info_failure_at = time.monotonic()
            if self._agent_credential is self._provisional_credential:
                self._agent_credential = None
            if self._built_credential is self._provisional_credential:
                self._built_credential = None
            self._provisional_credential = None
            self._runtime_key_thumbprint = ""
        log.debug("provisional runtime key not usable here (%s); the registration stays pending", reason)
        error = GovernanceConfigurationError(
            framework="agentic-new",
            code=_REGISTRATION_PENDING_CODE,
            payload={
                "registration": {
                    "id": getattr(info, "registration_id", ""),
                    "agent_subject": getattr(info, "agent_subject", ""),
                    "status": "pending",
                },
            },
        )
        self._observe_registration_error(error)
        return error

    def _provisional_token_fallback(
        self, credential: object, exc: BaseException,
    ) -> Optional[GovernanceConfigurationError]:
        """:meth:`_provisional_key_fallback` for a key that failed to enrol.

        Applies to the credential this engine built from a provisional offer,
        and only until the gateway has accepted its key: a failure after that
        belongs to an approved binding and is not a registration state.
        """
        if credential is None or credential is not self._provisional_credential:
            return None
        if getattr(credential, "enrolled", None) is not False:
            return None
        return self._provisional_key_fallback(exc, self._cred_info)

    # ── a refused provisional key: one re-read ─────────────────────────

    def _set_aside_refused_provisional_key(self, credential: object, exc: BaseException) -> bool:
        """Set aside a provisional offer whose key the gateway refused.

        Applies only to the credential this engine built from a provisional
        offer to a pending registration, and only when the gateway's
        enrolment or challenge endpoint refused it with
        ``agent_runtime_authentication_failed``. That refusal is what an
        administrator's decision leaves behind: after a VK-trusted approval or
        a denial the gateway no longer accepts the key, while the offer may
        still sit in the hour-long credential-info cache of a process that
        kept running.

        The offer, its credential and the cached answer are dropped, so the
        next lookup reads the gateway's current answer and replaces the
        cache. Returns False, leaving everything in place, for any other
        failure: an outage, a 429, a conflict and a key still waiting for
        authorization surface unchanged.
        """
        if credential is None or not isinstance(exc, DeepIntShieldError):
            return False
        if normalize_agentic_error_code(getattr(exc, "code", ""), "") != _RUNTIME_AUTHENTICATION_FAILED:
            return False
        endpoint = getattr(exc, "runtime_endpoint", "")
        if endpoint not in _PROVISIONAL_REREAD_ENDPOINTS:
            return False
        with self._credential_lock:
            if credential is self._provisional_credential:
                if self._agent_credential is credential:
                    self._agent_credential = None
                if self._built_credential is credential:
                    self._built_credential = None
                self._provisional_credential = None
                self._cred_info = None
                self._cred_info_fetched_at = 0.0
                self._cred_info_failure_code = ""
                self._cred_info_failure_at = 0.0
                # The gateway no longer holds that key for this registration.
                self._runtime_key_thumbprint = ""
                try:
                    self._set_aside_credentials.add(credential)
                except TypeError:  # pragma: no cover - not weakly referenceable
                    pass
            elif credential not in self._set_aside_credentials or credential is self._agent_credential:
                return False
            # Otherwise a concurrent call already set this very offer aside;
            # this one reads the answer that call left behind.
        log.debug("provisional runtime key refused at %s; reading the credential settings once more", endpoint)
        return True

    def _reread_after_refused_provisional_key(self) -> None:
        """Ask credential-info once more after a refused provisional key.

        The new answer replaces the cached one. An approved, declared or
        plainly pending answer lets the call continue (a 409 pending answer is
        raised as the gateway gave it). A gateway that repeats its provisional
        offer is read as the plain pending state, without a runtime-key
        clause, and no second enrolment is attempted in this call.
        """
        info = self.credential_info
        if self._provisional_pending(info):
            raise self._provisional_offer_pending(info, _RUNTIME_AUTHENTICATION_FAILED)

    # ── a credential the gateway marked optional ───────────────────────

    @staticmethod
    def _credential_optional(info: Optional[VKCredentialInfo]) -> bool:
        """Whether the gateway marked this answer's credential optional.

        The server-sent ``credential_optional`` flag alone decides; nothing
        the engine remembers or infers stands in for it.
        """
        return info is not None and getattr(info, "credential_optional", False) is True

    def _forget_optional_credential(self) -> None:
        """Drop the built credential's marker and its failure memory (caller holds the lock)."""
        self._built_credential = None
        self._optional_credential_failed_at = 0.0
        self._optional_credential_failed_for = None

    def _optional_credential_resting(self, info: Optional[VKCredentialInfo]) -> bool:
        """Whether an optional credential failed for this very answer moments ago."""
        failed_at = self._optional_credential_failed_at
        return (
            failed_at > 0.0
            and info is not None
            and info is self._optional_credential_failed_for
            and self._credential_optional(info)
            and time.monotonic() - failed_at <= _OPTIONAL_CREDENTIAL_RETRY_SECONDS
        )

    def _optional_credential_resting_for(self, credential: object) -> bool:
        return (
            credential is not None
            and credential is self._built_credential
            and self._optional_credential_resting(self._cred_info)
        )

    def _optional_credential_failure(self, credential: object, exc: BaseException) -> bool:
        """Whether a failure of ``credential`` leaves the call on the virtual key.

        True only for a credential this engine built itself while the cached
        credential-info answer is one the gateway marked
        ``credential_optional``. The failure is then remembered for a minute
        and reported by one line per process; no token is sent. A credential
        the application supplied, and every answer without the flag, keep
        failing closed.
        """
        if credential is None or credential is not self._built_credential:
            return False
        info = self._cred_info
        if not self._credential_optional(info):
            return False
        self._rest_optional_credential(info, exc, credential)
        return True

    def _rest_optional_credential(
        self, info: Optional[VKCredentialInfo], exc: BaseException, credential: object,
    ) -> None:
        """Remember an optional credential's failure and say once that the agent runs on its key.

        ``credential`` is the one that failed, or None when building it did.
        The line cites a runtime key only when one was enrolled: the
        thumbprint of a key the gateway holds pending, or of one it accepted.
        """
        payload = getattr(exc, "payload", None)
        thumbprint = _registration_fragment(
            payload.get("key_thumbprint") if isinstance(payload, dict) else "",
            _RUNTIME_KEY_THUMBPRINT_RE, 43,
        )
        if not thumbprint and getattr(credential, "enrolled", None) is True:
            # The gateway accepted this key earlier; cite it.
            try:
                thumbprint = _registration_fragment(
                    getattr(credential, "key_thumbprint", ""), _RUNTIME_KEY_THUMBPRINT_RE, 43,
                )
            except Exception:
                thumbprint = ""
        with self._credential_lock:
            self._optional_credential_failed_at = time.monotonic()
            self._optional_credential_failed_for = info
        code = normalize_agentic_error_code(
            get_exception_error_code(exc)
            or ("credential_dependency_missing" if isinstance(exc, ImportError) else ""),
            "",
        )
        log.debug("optional agent credential unavailable (%s); the call continues on the virtual key", code or type(exc).__name__)
        key = self._agent_subject_selector or "agent"
        with _optional_credential_log_lock:
            if key in _optional_credential_logged:
                return
            _optional_credential_logged.add(key)
        clause = f" (runtime key SHA256:{thumbprint})" if thumbprint else ""
        log.warning(
            "DeepintShield: this agent runs on its virtual key; its declared identity is not active yet%s.",
            clause,
        )

    def _build_credential_for(self, info: VKCredentialInfo) -> AgentCredential:
        if info.runtime_auth_mode == "gateway_broker":
            from .credentials.gateway_broker import GatewayBrokerCredential

            if info.runtime_broker is None or (
                info.agent_subject and info.agent_subject != self._agent_subject_selector
            ):
                raise GovernanceConfigurationError(
                    framework="gateway-broker", code="agent_runtime_configuration_error",
                )
            return GatewayBrokerCredential(
                http=self._http, gateway_url=self.gateway_url, virtual_key=self.virtual_key,
                agent_subject=self._agent_subject_selector, broker=info.runtime_broker,
                accepts_provisional_key=self._runtime_key_capable(),
            )
        if info.runtime_auth_mode:
            raise GovernanceConfigurationError(
                framework="gateway-broker", code="agent_runtime_configuration_error",
            )
        if info.require_proof_of_possession and info.provider_type != "generic_oidc":
            raise GovernanceConfigurationError(
                framework="agent-credential",
                reason="this credential adapter has no verified DPoP token issuance protocol",
                code="credential_dpop_provider_unsupported",
            )
        if info.provider_type == "entra_agent_id":
            from .credentials.entra import EntraAgentCredential

            return EntraAgentCredential(
                authority=info.authority,
                blueprint_client_id=info.blueprint_client_id,
                agent_identity_client_id=info.agent_identity_client_id,
                gateway_audience=info.gateway_audience,
                scopes=info.scopes,
                fic_audience=info.fic_audience,
                exchange_endpoint=info.exchange_endpoint,
            )
        if info.provider_type == "zeroid":
            from .credentials.zeroid import ZeroIDCredential

            return ZeroIDCredential(
                exchange_endpoint=(
                    os.environ.get("ZEROID_EXCHANGE_ENDPOINT", "") or info.exchange_endpoint
                ),
                gateway_audience=info.gateway_audience,
                scopes=info.scopes,
            )
        if info.provider_type == "generic_oidc":
            from .credentials.oidc import OIDCCredential

            # A provisioned client can have narrower scopes than its shared
            # verification provider. An explicit empty value requests no scope.
            workload_scopes = os.environ.get("OIDC_SCOPES")
            # The gateway stores only VERIFICATION config for a generic_oidc
            # provider (issuer / audience / JWKS), matching how the UI is
            # configured. The token endpoint and client id used to MINT the
            # agent token live with the agent, so fall back to the SDK
            # environment when credential-info does not supply them. The client
            # secret is read from OIDC_CLIENT_SECRET by OIDCCredential itself.
            return OIDCCredential(
                exchange_endpoint=(
                    os.environ.get("OIDC_TOKEN_ENDPOINT", "")
                    or info.exchange_endpoint
                ),
                client_id=(
                    info.blueprint_client_id
                    or os.environ.get("OIDC_CLIENT_ID", "")
                ),
                gateway_audience=info.gateway_audience,
                scopes=(workload_scopes.split() if workload_scopes is not None else info.scopes),
                issuer=info.issuer,
                require_proof_of_possession=info.require_proof_of_possession,
            )
        # Dev shortcut: respect a known-good token from env for offline
        # iteration where the gateway is reachable but Azure isn't.
        if dev_token := os.environ.get("DEEPINTSHIELD_DEV_AGENT_TOKEN", ""):
            return StaticAgentCredential(dev_token)
        raise GovernanceConfigurationError(
            framework="agent-credential",
            reason=f"unknown provider_type: {info.provider_type!r}",
            code="credential_provider_unsupported",
        )

    def _headers(
        self, *, include_agent_token: bool, method: str = "POST", path: str = ""
    ) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.virtual_key}",
            "Content-Type": "application/json",
            "User-Agent": f"deepintshield-python/{_sdk_version()}",
        }
        if self._agent_subject_selector:
            h["X-Agent-Subject"] = self._agent_subject_selector
        if include_agent_token:
            h.update(self.workload_headers(method, path))
        return h

    def workload_headers(self, method: str, uri: str, *, token: str = "") -> dict[str, str]:
        """Acquire a token and its proof from one immutable credential choice.

        Also used by MCP's native per-request hook. Holding the credential
        lock prevents a concurrent identity refresh from switching the signer
        between token acquisition and proof construction.
        """
        with self._credential_lock:
            token = token or self.agent_token(fail_closed=True) or ""
            if not token:
                return {}
            return {
                "X-Agent-Token": token,
                **self._dpop_headers(method, uri, access_token=token),
            }

    def _dpop_headers(
        self, method: str, path: str, access_token: Optional[str] = None
    ) -> dict[str, str]:
        """Proof headers for one request, using the credential's enrolled key.

        Legacy issuer credentials retain their existing optional proof behavior.
        Gateway sessions always require their credential's specific signer and
        fail closed when it is unavailable.
        """
        uri = path if path.startswith(("https://", "http://")) else f"{self.gateway_url.rstrip('/')}{path}"
        credential = self._agent_credential
        signer = getattr(credential, "proof_headers", None)
        if callable(signer):
            return signer(method, uri, access_token or "")
        if str(access_token or "").startswith("dsrs_"):
            raise GovernanceConfigurationError(
                framework="gateway-broker", code="agent_runtime_key_unavailable",
            )
        try:
            from .dpop import dpop_headers

            return dpop_headers(
                method, uri, access_token=access_token
            )
        except Exception:
            return {}


def _registry_key(value: str) -> str:
    """Match Go ``AgenticNewSanitizeKey`` for OpenFGA agent/tool ids."""
    return _manifest_registry_key(value) or "tool"


def _permission_slug(value: str) -> str:
    """Match Go ``AgenticNewSlugify`` used by ``PermissionSubject``."""
    out: list[str] = []
    last_dash = False
    for char in str(value or "").strip().lower():
        if "a" <= char <= "z" or "0" <= char <= "9":
            out.append(char)
            last_dash = False
        elif char in "_-.:":
            out.append("-")
            last_dash = True
        elif char in " \t/\\" and not last_dash and out:
            out.append("-")
            last_dash = True
    return "".join(out).strip("-") or "tool-read"


def _permission_subject(value: str) -> str:
    """Match Go's versioned, collision-resistant ``PermissionSubject``."""
    canonical = str(value or "").strip().lower()
    readable = _permission_slug(canonical)
    readable = readable[:64].rstrip("-._") or readable[:64]

    payload = bytearray(b"deepintshield/openfga/object-id/v1")
    for component in ("permission", canonical):
        encoded = component.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    suffix = hashlib.sha256(payload).hexdigest()[:32]
    return f"permission:v1-{readable}--{suffix}"


def _action_class(tool: str) -> str:
    return infer_action_class(tool)


def _sdk_version() -> str:
    try:
        from ..version import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0.0.0"


__all__ = ["AgenticEngine"]
