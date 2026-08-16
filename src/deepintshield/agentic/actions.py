"""Fail-safe action impact inference shared by discovery and authorization."""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKENS = re.compile(r"[a-z0-9]+")

# Concrete mutation verbs take precedence over read hints for compound names
# such as ``read_then_delete``. Generic verbs (invoke/run/execute/call) are not
# read evidence and therefore naturally fall through to the safe write default.
_MUTATION_TOKENS = frozenset(
    {
        "append",
        "approve",
        "archive",
        "assign",
        "cancel",
        "charge",
        "commit",
        "create",
        "delete",
        "deploy",
        "disable",
        "drop",
        "edit",
        "enable",
        "grant",
        "insert",
        "invite",
        "merge",
        "modify",
        "move",
        "patch",
        "pay",
        "post",
        "publish",
        "put",
        "remove",
        "rename",
        "revoke",
        "send",
        "set",
        "submit",
        "transfer",
        "update",
        "upload",
        "upsert",
        "write",
    }
)

_READ_TOKENS = frozenset(
    {
        "check",
        "count",
        "describe",
        "download",
        "fetch",
        "find",
        "get",
        "health",
        "inspect",
        "list",
        "lookup",
        "preview",
        "query",
        "read",
        "retrieve",
        "search",
        "select",
        "status",
        "view",
    }
)


def infer_action_class(name: str) -> str:
    """Infer ``read`` only from an explicit read-only verb; default to write.

    An opaque ``invoke``/``run``/custom name can mutate state, so classifying it
    as read would bypass write/HITL policy. Explicit manifest ``action_class``
    values are handled by callers and always take precedence over inference.
    """
    separated = _CAMEL_BOUNDARY.sub("_", str(name or ""))
    tokens = set(_TOKENS.findall(separated.lower()))
    if tokens & _MUTATION_TOKENS:
        return "write"
    if tokens & _READ_TOKENS:
        return "read"
    return "write"


__all__ = ["infer_action_class"]
