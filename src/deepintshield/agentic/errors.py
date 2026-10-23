"""Internal Agentic errors and the standard Python public error contract.

SDK internals retain typed errors with code-only strings and private audit
metadata. Application/framework boundaries translate those errors to marked
``PermissionError``, ``RuntimeError``, or ``ConnectionError`` instances. Their
strings combine the stable code with trusted catalogue text; raw gateway data
is never rendered. Existing typed classes remain for SDK implementation and
backward compatibility, but new application code does not catch them.
"""

from __future__ import annotations

import functools
import inspect
import re
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, NoReturn, TypedDict

from ..errors import (
    _REGISTRATION_CONTACT,
    DeepintShieldError,
    ErrorCode,
    get_error_definition,
    get_exception_error_code,
)


AGENTIC_ERROR = "agentic_error"
AGENT_CONFIGURATION_ERROR = "governance_configuration_error"
AGENT_ACCESS_DENIED = "guardrail_denied"
AGENT_APPROVAL_PENDING = "require_approval"
AGENT_OBLIGATION_UNSUPPORTED = "mask_obligation_unsupported"
AGENT_GATEWAY_UNAVAILABLE = "gateway_unavailable"
AGENT_INVALID_GATEWAY_RESPONSE = "invalid_gateway_response"

_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
# Registration outcomes carry a developer-facing reference, runtime-key
# thumbprint and notification e-mail. Every fragment is validated locally and
# dropped when it does not match; server prose and review paths are never
# rendered.
_REGISTRATION_PENDING_CODES = frozenset(
    {
        ErrorCode.AGENT_REGISTRATION_PENDING.value,
        ErrorCode.AGENT_RUNTIME_ENROLLMENT_PENDING.value,
    }
)
_REGISTRATION_MESSAGE_CODES = _REGISTRATION_PENDING_CODES | {
    ErrorCode.REGISTRATION_REQUESTER_MISMATCH.value,
}
_REGISTRATION_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_.-]{0,127}$")
# RFC 7638 JWK thumbprint: 32 bytes, base64url without padding.
_RUNTIME_KEY_THUMBPRINT_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_REGISTRATION_EMAIL_RE = re.compile(r"^[^\s@]{1,64}@[^\s@]+\.[^\s@]+$")
_REGISTRATION_EMAIL_MAX_LENGTH = 254
_PUBLIC_ERROR_CODE_ATTR = "_deepintshield_error_code"
_PUBLIC_ERROR_MESSAGE_ATTR = "_deepintshield_error_message"
_PUBLIC_ERROR_ACTION_ATTR = "_deepintshield_error_action"
_PUBLIC_ERROR_DASHBOARD_ATTR = "_deepintshield_dashboard_path"


class PublicAgenticErrorDetails(TypedDict):
    code: str
    message: str
    action: str
    dashboard_path: str


_DENIAL_CODES = frozenset(
    {
        AGENT_ACCESS_DENIED,
        "agent_access_denied",
        "agent_lacks_perm",
        "agent_retired",
        "approval_access_denied",
        "authorization_denied",
        "authz_engine_error",
        "context_deny",
        "no_store",
        "obo_action_not_allowed",
        "obo_agent_lacks_perm",
        "obo_delegation_required",
        "obo_no_acts_for",
        "obo_scope_mismatch",
        "obo_tool_not_allowed",
        "obo_user_lacks_perm",
        "obo_user_mismatch",
    }
)


def normalize_agentic_error_code(value: object, fallback: str) -> str:
    """Return a bounded public error code, never arbitrary server prose."""
    candidate = (
        value.strip().lower().replace("-", "_")
        if isinstance(value, str)
        else ""
    )
    if _ERROR_CODE.fullmatch(candidate):
        return candidate
    fallback_code = (
        fallback.strip().lower().replace("-", "_")
        if isinstance(fallback, str)
        else ""
    )
    if not fallback_code:
        return "" if isinstance(fallback, str) else AGENTIC_ERROR
    if _ERROR_CODE.fullmatch(fallback_code):
        return fallback_code
    return AGENTIC_ERROR


def public_agentic_error_details(value: object) -> PublicAgenticErrorDetails:
    """Return trusted operator guidance for a stable Agentic error code.

    ``value`` may be a code or a marked built-in exception. Unknown valid
    codes receive generic guidance; raw exception/server text is never copied.
    """
    raw = (
        get_exception_error_code(value)
        if isinstance(value, BaseException)
        else value
    )
    code = normalize_agentic_error_code(raw, AGENTIC_ERROR)
    # All trusted text comes from the SDK-wide immutable catalogue.  Unknown
    # (but syntactically safe) future gateway codes retain their code while
    # receiving the generic Agentic guidance.
    # Retirement is a terminal lifecycle state with safe, specific guidance;
    # ordinary authorization denials retain their generic explanation.
    lookup = (
        AGENT_ACCESS_DENIED
        if code in _DENIAL_CODES and code not in {
            ErrorCode.AGENT_RETIRED.value,
            ErrorCode.AGENT_LACKS_PERM.value,
            ErrorCode.OBO_AGENT_LACKS_PERM.value,
        }
        else code
    )
    definition = get_error_definition(lookup) or get_error_definition(
        ErrorCode.AGENTIC_ERROR
    )
    assert definition is not None  # static catalogue invariant
    return {
        "code": code,
        "message": definition.description,
        "action": definition.action,
        "dashboard_path": definition.dashboard_path,
    }


def _registration_fragment(value: object, pattern: re.Pattern[str], limit: int = 128) -> str:
    """Return ``value`` when it is a bounded, well-formed fragment, else ``""``."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned or len(cleaned) > limit or not pattern.fullmatch(cleaned):
        return ""
    # An echoed fragment reaches the developer's terminal and logs; control
    # and format characters are never part of an identifier or an address.
    if not cleaned.isprintable():
        return ""
    return cleaned


def format_registration_message(
    code: object,
    *,
    reference: object = "",
    thumbprint: object = "",
    email: object = "",
    notify: bool = True,
) -> str:
    """Render the developer-facing registration message for ``code``.

    ``Registration pending. Reference <id> · runtime key SHA256:<thumbprint>.
    Contact your administrator. You will be notified at <email>.`` for the two
    pending codes. ``registration_requester_mismatch`` means one thing: the
    agent is already being registered under a different e-mail. It renders
    its refusal lead, the reference when the gateway named the registration,
    and the contact sentence. It never carries the runtime-key clause or the
    notification sentence: the recorded address belongs to whoever registered
    first. Fragments are omitted when unknown or invalid. Other codes render
    their static catalogue description unchanged.

    ``email`` is the address the gateway confirmed it will notify, and nothing
    else: pass the gateway's echo (``notify_email``, or the 202 body's
    ``registration.requester_email``), never the locally configured
    ``registered_by``. A gateway that recorded no address sends no e-mail, so
    the sentence is left out rather than promised.
    """
    normalized = normalize_agentic_error_code(code, AGENTIC_ERROR)
    description = public_agentic_error_details(normalized)["message"]
    if normalized not in _REGISTRATION_MESSAGE_CODES:
        return description
    pending = normalized in _REGISTRATION_PENDING_CODES
    suffix = f" {_REGISTRATION_CONTACT}"
    lead = description[: -len(suffix)] if description.endswith(suffix) else description
    parts = [lead]
    reference_text = _registration_fragment(reference, _REGISTRATION_REFERENCE_RE)
    thumbprint_text = (
        _registration_fragment(thumbprint, _RUNTIME_KEY_THUMBPRINT_RE, 43) if pending else ""
    )
    email_text = _registration_fragment(
        email.lower() if isinstance(email, str) else "",
        _REGISTRATION_EMAIL_RE,
        _REGISTRATION_EMAIL_MAX_LENGTH,
    )
    if reference_text and thumbprint_text:
        parts.append(f"Reference {reference_text} · runtime key SHA256:{thumbprint_text}.")
    elif reference_text:
        parts.append(f"Reference {reference_text}.")
    elif thumbprint_text:
        parts.append(f"Runtime key SHA256:{thumbprint_text}.")
    parts.append(_REGISTRATION_CONTACT)
    if notify and email_text and pending:
        parts.append(f"You will be notified at {email_text}.")
    return " ".join(parts)


def public_agentic_error_message(code: object, payload: object = None) -> str:
    """Return the trusted message for ``code``, with registration context.

    Registration codes read the SDK-attached ``payload["registration_context"]``
    (``registration_id``, ``key_thumbprint``, ``notify_email``), which the
    engine fills from validated gateway echoes and its own memory.
    ``notify_email`` is filled from the gateway only, so the notification
    sentence appears exactly when the gateway confirmed an address. Every
    other code renders its static catalogue description; raw gateway prose is
    never consulted.
    """
    normalized = normalize_agentic_error_code(code, AGENTIC_ERROR)
    if normalized not in _REGISTRATION_MESSAGE_CODES:
        return public_agentic_error_details(normalized)["message"]
    context = payload.get("registration_context") if isinstance(payload, Mapping) else None
    if not isinstance(context, Mapping):
        context = {}
    return format_registration_message(
        normalized,
        reference=context.get("registration_id", ""),
        thumbprint=context.get("key_thumbprint", ""),
        email=context.get("notify_email", ""),
    )


class DeepIntShieldError(DeepintShieldError):
    """Base for code-first Agentic failures.

    ``detail`` is deliberately not included in ``str(exc)``. It is retained for
    structured logging and compatibility, while the server-side audit record
    and Agentic interface remain authoritative for human explanation.
    """

    default_code = AGENTIC_ERROR

    def __init__(
        self,
        detail: str = "",
        *,
        code: str = "",
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        normalized_code = normalize_agentic_error_code(code, self.default_code)
        self.detail = str(detail or "")
        super().__init__(
            message=normalized_code,
            status_code=status_code,
            payload=payload or {},
            # The SDK-wide catalogue intentionally contains only codes known
            # to this release. Agentic gateways may introduce a syntactically
            # valid code before the client upgrades, so use generic trusted
            # metadata in the base class and restore the bounded wire code
            # below instead of collapsing it to ``sdk_error``.
            code=(
                normalized_code
                if get_error_definition(normalized_code) is not None
                else ErrorCode.AGENTIC_ERROR
            ),
        )
        self.code = normalized_code

    def __str__(self) -> str:
        return self.code


class GuardrailDenied(DeepIntShieldError):
    """The authorization decision denied execution; the body did not run."""

    default_code = AGENT_ACCESS_DENIED

    def __init__(
        self,
        reason: str = "",
        decision_id: str = "",
        policy_id: str = "",
        tool: str = "",
        *,
        code: str = "",
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason or "")
        self.decision_id = str(decision_id or "")
        self.policy_id = str(policy_id or "")
        self.tool = str(tool or "")
        super().__init__(
            self.reason,
            code=code,
            status_code=status_code,
            payload=payload,
        )


class GuardrailApprovalPending(DeepIntShieldError):
    """Execution is paused until the durable approval is decided in Agentic."""

    default_code = AGENT_APPROVAL_PENDING

    def __init__(
        self,
        decision_id: str = "",
        approvers: list[str] | None = None,
        reason: str = "",
        *,
        code: str = "",
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.decision_id = str(decision_id or "")
        self.approvers = list(approvers or [])
        self.reason = str(reason or "")
        super().__init__(
            self.reason,
            code=code,
            status_code=status_code,
            payload=payload,
        )


class GuardrailMasked(DeepIntShieldError):
    """The gateway returned an obligation the SDK cannot apply locally."""

    default_code = AGENT_OBLIGATION_UNSUPPORTED

    def __init__(
        self,
        obligations: list[str] | None = None,
        value: Any = None,
        *,
        code: str = "",
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.obligations = list(obligations or [])
        self.value = value
        super().__init__(
            code=code,
            status_code=status_code,
            payload=payload,
        )


class GatewayUnavailable(DeepIntShieldError):
    """The authoritative Agentic gateway could not be reached or used."""

    default_code = AGENT_GATEWAY_UNAVAILABLE

    def __init__(
        self,
        reason: str = "",
        *,
        code: str = "",
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason or "")
        super().__init__(
            self.reason,
            code=code,
            status_code=status_code,
            payload=payload,
        )


class GovernanceConfigurationError(DeepIntShieldError):
    """Execution cannot be governed safely with the current configuration."""

    default_code = AGENT_CONFIGURATION_ERROR

    def __init__(
        self,
        reason: str = "",
        framework: str = "",
        *,
        code: str = "",
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.reason = str(reason or "")
        self.framework = str(framework or "")
        super().__init__(
            self.reason,
            code=code,
            status_code=status_code,
            payload=payload,
        )


def public_agentic_error(error: BaseException) -> BaseException:
    """Translate an internal Agentic error to a standard Python exception.

    Framework and application boundaries expose only a built-in exception with
    one stable machine-readable code.  Internal SDK code keeps the richer typed
    errors above so policy handling and audit metadata do not have to be folded
    into application control flow.

    The marker makes a translated error distinguishable from an application's
    own ``RuntimeError``/``PermissionError`` without importing an SDK exception
    class.  Passing an already translated error is idempotent.
    """
    marked = normalize_agentic_error_code(
        getattr(error, _PUBLIC_ERROR_CODE_ATTR, ""),
        "",
    )
    if marked:
        return error
    if not isinstance(error, DeepIntShieldError):
        return error

    code = normalize_agentic_error_code(error.code, error.default_code)
    details = public_agentic_error_details(code)
    message = public_agentic_error_message(code, getattr(error, "payload", None))
    rendered = f"{code}: {message}"
    if isinstance(error, GuardrailDenied) or code in _DENIAL_CODES:
        translated: BaseException = PermissionError(rendered)
    elif isinstance(error, GatewayUnavailable) or code in {
        AGENT_GATEWAY_UNAVAILABLE,
        AGENT_INVALID_GATEWAY_RESPONSE,
    }:
        translated = ConnectionError(rendered)
    else:
        # Approval, configuration, unsupported obligations and blueprint
        # lifecycle failures all mean the governed operation cannot proceed in
        # its current state. RuntimeError is the standard Python boundary.
        translated = RuntimeError(rendered)
    setattr(translated, _PUBLIC_ERROR_CODE_ATTR, code)
    setattr(translated, _PUBLIC_ERROR_MESSAGE_ATTR, message)
    setattr(translated, _PUBLIC_ERROR_ACTION_ATTR, details["action"])
    setattr(translated, _PUBLIC_ERROR_DASHBOARD_ATTR, details["dashboard_path"])
    return translated


def _raise_detached_error(error: BaseException) -> NoReturn:
    """Raise a public/protocol error without retaining an internal context.

    Callers must invoke this only after leaving any handler for a private SDK
    exception.  Keeping the raise in this tiny frame also ensures traceback
    locals contain only the already-sanitized public error.
    """
    error.__context__ = None
    error.__cause__ = None
    try:
        raise error from None
    finally:
        # Be defensive if an interpreter or framework assigns implicit context
        # while propagating the exception through its own callback machinery.
        error.__context__ = None
        error.__cause__ = None


@contextmanager
def public_agentic_error_boundary() -> Iterator[None]:
    """Expose internal Agentic failures as safe, marked built-in exceptions.

    Only SDK Agentic errors are translated. Application and framework bugs,
    including an ordinary ``RuntimeError``, propagate untouched. ``from None``
    prevents internal transport/configuration context from leaking through the
    default traceback.
    """
    translated: BaseException | None = None
    try:
        yield
    except DeepIntShieldError as error:
        # Do not raise while handling ``error``.  ``raise ... from None`` hides
        # implicit context from the rendered traceback, but Python would still
        # retain the internal exception (and its private payload) through
        # ``__context__``.  Leaving the handler first makes the public object a
        # genuinely detached, safe boundary value.
        translated = public_agentic_error(error)
    if translated is not None:
        _raise_detached_error(translated)


def public_agentic_boundary(function: Any) -> Any:
    """Mark a callable as an application-facing Agentic error boundary."""
    if inspect.isasyncgenfunction(function):
        @functools.wraps(function)
        async def async_generator(*args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                async for item in function(*args, **kwargs):
                    yield item
        return async_generator
    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def asynchronous(*args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                return await function(*args, **kwargs)
        return asynchronous
    if inspect.isgeneratorfunction(function):
        @functools.wraps(function)
        def generator(*args: Any, **kwargs: Any) -> Any:
            with public_agentic_error_boundary():
                yield from function(*args, **kwargs)
        return generator

    @functools.wraps(function)
    def synchronous(*args: Any, **kwargs: Any) -> Any:
        with public_agentic_error_boundary():
            return function(*args, **kwargs)
    return synchronous


__all__ = [
    "AGENTIC_ERROR",
    "AGENT_CONFIGURATION_ERROR",
    "AGENT_ACCESS_DENIED",
    "AGENT_APPROVAL_PENDING",
    "AGENT_OBLIGATION_UNSUPPORTED",
    "AGENT_GATEWAY_UNAVAILABLE",
    "AGENT_INVALID_GATEWAY_RESPONSE",
    "normalize_agentic_error_code",
    "PublicAgenticErrorDetails",
    "format_registration_message",
    "public_agentic_error_details",
    "public_agentic_error_message",
    "public_agentic_error",
    "public_agentic_boundary",
    "public_agentic_error_boundary",
    "DeepIntShieldError",
    "GuardrailDenied",
    "GuardrailApprovalPending",
    "GuardrailMasked",
    "GatewayUnavailable",
    "GovernanceConfigurationError",
]
