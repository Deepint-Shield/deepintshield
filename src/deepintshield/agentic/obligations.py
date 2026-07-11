"""Shared digest + obligation helpers used by every framework adapter.

Zero-data-retention: raw argument values never leave the process - only a
SHA-256 digest of the canonicalised arguments is sent to the PDP.
"""

from __future__ import annotations

import hashlib
import json


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


def apply_obligations(kwargs: dict, obligations: list[str]) -> dict:
    """Apply known obligations to the kwargs. Unknown obligations are no-ops
    (the gateway already validated them; this is the SDK side's best-effort
    defence in depth)."""
    if not obligations:
        return kwargs
    out = dict(kwargs)
    for ob in obligations:
        if ob == "mask:pii":
            out = mask_pii(out)
        # Add more handlers here as DeepintShield ships new obligation types
        # (rate-limit, time-box). They're enforced server-side too.
    return out


__all__ = ["digest", "mask_pii", "apply_obligations"]
