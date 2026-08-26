"""Shared digest + obligation helpers used by every framework adapter.

Zero-data-retention: raw argument values never leave the process - only a
SHA-256 digest of the canonicalised arguments is sent to the PDP.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re


def digest(args: tuple, kwargs: dict) -> str:
    """sha256 of the canonicalised arguments. Sort keys so identical calls
    (regardless of kwarg order) produce identical digests, which in turn
    produce L1 cache hits on the PEP side."""
    payload = {"args": list(args), "kwargs": kwargs}
    try:
        canonical = json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = repr(payload)
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


_PII_FIELDS = {"email", "phone", "ssn", "credit_card", "tax_id", "address"}


def mask_pii(d: dict) -> dict:
    """Redact known PII fields by name. Leaves shape intact so the receiving
    tool's signature still validates."""
    out = {}
    for k, v in d.items():
        if k.lower() in _PII_FIELDS and isinstance(v, str):
            out[k] = "***"
        else:
            out[k] = v
    return out


_SECRET_FIELDS = {
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "private_key",
    "credential",
    "credentials",
    "authorization",
    "auth",
    "session_key",
}

# Common issued-credential shapes. Matching the VALUE as well as the field name
# matters because a secret handed to a tool as `note="sk-live-..."` is still a
# secret, and the field name says nothing.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

_PHI_FIELDS = {
    "diagnosis",
    "diagnoses",
    "icd",
    "icd10",
    "icd_code",
    "mrn",
    "medical_record_number",
    "patient_id",
    "patient_name",
    "treatment",
    "medication",
    "prescription",
    "lab_result",
    "health_plan_id",
    "insurance_id",
}

_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

_REDACTED = "***"


def _luhn_valid(digits: str) -> bool:
    """Card-number check so ordinary long numbers - order ids, timestamps,
    account references - are not mangled into ``***``."""
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = ord(char) - 48
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_card_numbers_in_text(value: str) -> str:
    def replace(match: "re.Match[str]") -> str:
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return _REDACTED
        return match.group(0)

    return _CARD_CANDIDATE.sub(replace, value)


def _redact_by_field(d: dict, fields: set) -> dict:
    out = {}
    for k, v in d.items():
        if k.lower() in fields:
            out[k] = _REDACTED
        else:
            out[k] = v
    return out


def redact_secrets(d: dict) -> dict:
    """Redact credential-shaped fields AND credential-shaped values."""
    out = _redact_by_field(d, _SECRET_FIELDS)
    for k, v in out.items():
        if not isinstance(v, str):
            continue
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(v):
                out[k] = _REDACTED
                break
    return out


def redact_phi(d: dict) -> dict:
    """Redact protected-health-information fields by name."""
    return _redact_by_field(d, _PHI_FIELDS)


_BANK_FIELDS = {
    "account_number",
    "bank_account",
    "bank_account_number",
    "iban",
    "routing_number",
    "sort_code",
    "swift",
    "bic",
}


def redact_bank_accounts(d: dict) -> dict:
    """Redact bank-account identifiers by field name."""
    return _redact_by_field(d, _BANK_FIELDS)


def redact_card_numbers(d: dict) -> dict:
    """Redact Luhn-valid card numbers wherever they appear in string values."""
    out = {}
    for k, v in d.items():
        out[k] = _redact_card_numbers_in_text(v) if isinstance(v, str) else v
    return out


# Field names that carry a raw secret VALUE, as opposed to naming one. The
# secrets templates pass a key/identifier in and expect the value never to be
# handled in the clear, so a call that carries one inline is the case this
# covers.
_VALUE_FIELDS = {
    "value",
    "values",
    "secret_value",
    "new_value",
    "old_value",
    "plaintext",
    "plain_text",
    "cleartext",
}


def redact_values(d: dict) -> dict:
    """Redact raw value-carrying fields by name.

    Scoped to the ARGUMENTS of a call, like every handler here. The
    ``tool-secrets-read`` template pairs this with ``return:fingerprint-only``,
    which constrains the RESULT and is the gateway's to enforce - this side
    cannot see a result it never receives.
    """
    return _redact_by_field(d, _VALUE_FIELDS)


# The obligations this side can actually carry out. A verdict may attach an
# obligation this table does not know; that stays a no-op here and is the
# gateway's to enforce. What must NOT happen is a shipped policy template
# promising redaction that nothing performs - which is what
# redact:secrets / redact:phi / redact:card-numbers were before this table.
_HANDLERS = {
    "mask:pii": mask_pii,
    "redact:secrets": redact_secrets,
    "redact:phi": redact_phi,
    "redact:card-numbers": redact_card_numbers,
    "redact:bank-accounts": redact_bank_accounts,
    "redact:value": redact_values,
}


def apply_obligations(kwargs: dict, obligations: list[str]) -> dict:
    """Apply known obligations to the kwargs. Unknown obligations are no-ops
    (the gateway already validated them; this is the SDK side's best-effort
    defence in depth)."""
    if not obligations:
        return kwargs
    out = dict(kwargs)
    for ob in obligations:
        handler = _HANDLERS.get(str(ob).strip().lower())
        if handler is not None:
            out = handler(out)
    return out


def apply_obligations_to_call(
    tool_callable: object,
    args: tuple,
    kwargs: dict,
    obligations: list[str],
) -> "tuple[tuple, dict]":
    """Apply obligations across BOTH positional and keyword arguments.

    A MASK verdict used to be bypassable by calling the tool positionally:
    only kwargs were ever masked, so ``send(email)`` leaked what
    ``send(email=…)`` redacted. Since the caller chooses the calling
    convention, that made the obligation optional in practice.

    Positionals are bound to their parameter names, redacted as a mapping, and
    written back by position. The argument DIGEST is deliberately untouched -
    it is computed upstream from the original call shape, and changing it here
    would invalidate every approval already pinned to it.

    Anything that cannot be bound - a builtin with no signature, ``*args``
    packing - is passed through unchanged rather than guessed at.
    """
    masked_kwargs = apply_obligations(kwargs, obligations)
    if not obligations or not args:
        return args, masked_kwargs
    try:
        signature = inspect.signature(tool_callable)  # type: ignore[arg-type]
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return args, masked_kwargs

    positional_names: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            break
        if name not in bound.arguments:
            break
        positional_names.append(name)
        if len(positional_names) == len(args):
            break
    if len(positional_names) != len(args):
        return args, masked_kwargs

    as_mapping = {name: args[index] for index, name in enumerate(positional_names)}
    redacted = apply_obligations(as_mapping, obligations)
    return tuple(redacted[name] for name in positional_names), masked_kwargs


__all__ = [
    "digest",
    "mask_pii",
    "redact_secrets",
    "redact_phi",
    "redact_card_numbers",
    "redact_bank_accounts",
    "redact_values",
    "apply_obligations",
    "apply_obligations_to_call",
]
