"""Safe exception boundary for results from the upstream MCP Python SDK.

This module deliberately has no dependency on an MCP package.  Both supported
upstream SDK generations expose result attributes with slightly different
names, so the boundary uses small, bounded structural reads and returns the
same DeepintShield exception contract in either case.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..errors import ERROR_CATALOG, DeepintShieldError, ErrorCode


_MAX_LEGACY_JSON_BYTES = 16 * 1024
_MAX_IDENTIFIER_LENGTH = 128
_MAX_EXCEPTION_NODES = 64
_MAX_EXCEPTION_DEPTH = 8
_MAX_CONTENT_BLOCKS = 32

_RESPONSE_CODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,127}$")

_GAF_CODES = frozenset(
    {
        ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED.value,
        ErrorCode.MCP_TOOL_AUTHORIZATION_UNAVAILABLE.value,
        ErrorCode.MCP_TOOL_APPROVAL_REQUIRED.value,
    }
)

_MISSING = object()


def _read_field(value: object, name: str) -> object:
    """Read one mapping key or object attribute without trusting its repr."""
    if isinstance(value, Mapping):
        try:
            return value[name] if name in value else _MISSING
        except Exception:
            return _MISSING
    try:
        return getattr(value, name)
    except Exception:
        return _MISSING


def _read_fields(value: object, names: tuple[str, ...]) -> tuple[object, ...]:
    found: list[object] = []
    for name in names:
        item = _read_field(value, name)
        if item is not _MISSING:
            found.append(item)
    return tuple(found)


def _safe_label(value: object, *, default: str = "") -> str:
    if isinstance(value, str) and _SAFE_LABEL.fullmatch(value):
        return value
    return default


def _safe_identifier(value: object) -> str:
    if not isinstance(value, str) or len(value) > _MAX_IDENTIFIER_LENGTH:
        return ""
    return value if _SAFE_IDENTIFIER.fullmatch(value) else ""


def _safe_response_code(value: object) -> str:
    if not isinstance(value, str) or len(value) > 64:
        return ""
    return value if _RESPONSE_CODE.fullmatch(value) else ""


def _safe_mcp_code(value: object) -> int | None:
    # bool is an int subclass but is not a JSON-RPC error code.
    if type(value) is int and -(2**31) <= value <= 2**31 - 1:
        return value
    return None


def _is_envelope(value: object) -> bool:
    if isinstance(value, Mapping):
        return True
    return any(
        _read_field(value, name) is not _MISSING
        for name in (
            "code",
            "error_code",
            "response_code",
            "decision_id",
            "approval_id",
            "error",
            "param",
            "data",
            "details",
        )
    )


def _candidate_containers(value: object) -> tuple[object, ...]:
    """Return only documented error envelopes, with a fixed traversal bound."""
    if not _is_envelope(value):
        return ()

    containers: list[object] = [value]
    error = _read_field(value, "error")
    if error is not _MISSING and _is_envelope(error):
        containers.append(error)

    # HTTP-style errors place safe decision evidence under ``param``.  Some
    # upstream adapters use ``data`` or ``details`` for the same envelope.
    for parent in tuple(containers):
        for key in ("param", "data", "details"):
            nested = _read_field(parent, key)
            if (
                nested is not _MISSING
                and _is_envelope(nested)
                and all(nested is not present for present in containers)
            ):
                containers.append(nested)
    return tuple(containers[:8])


def _evidence(value: object) -> dict[str, object]:
    """Extract allowlisted, bounded fields from one structured envelope."""
    response_code = ""
    mcp_code: int | None = None
    decision_id = ""
    approval_id = ""

    for container in _candidate_containers(value):
        if not response_code and mcp_code is None:
            for key in ("code", "error_code", "response_code"):
                candidate = _read_field(container, key)
                if candidate is _MISSING:
                    continue
                response_code = _safe_response_code(candidate)
                mcp_code = _safe_mcp_code(candidate)
                if response_code or mcp_code is not None:
                    break
        if not decision_id:
            decision_id = _safe_identifier(_read_field(container, "decision_id"))
        if not approval_id:
            approval_id = _safe_identifier(_read_field(container, "approval_id"))

    result: dict[str, object] = {}
    if response_code:
        result["response_code"] = response_code
    if mcp_code is not None:
        result["mcp_code"] = mcp_code
    if decision_id:
        result["decision_id"] = decision_id
    if approval_id:
        result["approval_id"] = approval_id
    return result


def _legacy_json_sources(result: object) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    contents = _read_fields(result, ("content",))
    if not contents:
        return ()
    content = contents[0]
    if type(content) not in (list, tuple):
        return ()
    for index, block in enumerate(content):
        if index >= _MAX_CONTENT_BLOCKS:
            break
        block_type = _read_field(block, "type")
        text = _read_field(block, "text")
        if block_type not in (_MISSING, "text") or not isinstance(text, str):
            continue
        # Check the cheaper character bound first so a very large hostile
        # string is never encoded merely to reject it.
        if len(text) > _MAX_LEGACY_JSON_BYTES:
            continue
        try:
            if len(text.encode("utf-8")) > _MAX_LEGACY_JSON_BYTES:
                continue
            decoded = json.loads(text)
        except (UnicodeError, ValueError, RecursionError):
            continue
        if isinstance(decoded, Mapping):
            sources.append(decoded)
    return tuple(sources)


def _result_sources(result: object) -> tuple[object, ...]:
    """Return evidence sources in protocol trust order.

    MCP metadata is transport-controlled and therefore takes precedence over
    structured tool content.  Structured content in turn takes precedence over
    the bounded JSON-text compatibility path.
    """
    sources: list[object] = []
    sources.extend(_read_fields(result, ("_meta", "meta")))
    sources.extend(
        _read_fields(result, ("structuredContent", "structured_content"))
    )
    sources.extend(_legacy_json_sources(result))
    return tuple(sources)


def _merge_evidence(sources: Iterable[object]) -> dict[str, object]:
    # Keep decision/approval identifiers bound to the same protocol envelope
    # that supplied the selected error code. A lower-trust structured/tool-text
    # source must never decorate a higher-trust metadata code with fake IDs.
    for source in sources:
        current = _evidence(source)
        if "response_code" in current or "mcp_code" in current:
            return current
    return {}


def _trusted_error(code: ErrorCode, details: Mapping[str, object]) -> DeepintShieldError:
    definition = ERROR_CATALOG[code.value]
    return DeepintShieldError(
        definition.description,
        code=code,
        details=details,
    )


def _result_error_flag(result: object) -> tuple[bool, bool]:
    """Return ``(is_error, malformed)`` for v1/v2 result aliases."""
    flags = _read_fields(result, ("isError", "is_error"))
    if not flags:
        return False, False
    if any(type(flag) is not bool for flag in flags):
        return False, True
    # Conflicting aliases fail closed; either generation reporting an error is
    # sufficient to keep a tool failure from looking successful.
    return any(flag is True for flag in flags), False


def mcp_error_from_result(result: object) -> DeepintShieldError | None:
    """Map a typed or mapping MCP tool result onto the SDK error contract.

    Error content is never used as exception prose.  Only a literal protocol
    error flag enables interpretation of the allowlisted structured evidence;
    a successful tool cannot spoof a canonical GAF error through its content.
    """
    if result is None or isinstance(result, (str, bytes, bytearray)):
        return _trusted_error(ErrorCode.MCP_PROTOCOL_ERROR, {})

    recognizable = _read_fields(
        result,
        (
            "isError",
            "is_error",
            "content",
            "structuredContent",
            "structured_content",
            "_meta",
            "meta",
        ),
    )
    if not recognizable:
        return _trusted_error(ErrorCode.MCP_PROTOCOL_ERROR, {})

    is_error, malformed = _result_error_flag(result)
    if malformed:
        return _trusted_error(ErrorCode.MCP_PROTOCOL_ERROR, {})
    if not is_error:
        return None

    details = _merge_evidence(_result_sources(result))
    response_code = details.get("response_code")
    if isinstance(response_code, str) and response_code in _GAF_CODES:
        return _trusted_error(ErrorCode(response_code), details)
    return _trusted_error(ErrorCode.MCP_EXECUTION_FAILED, details)


def raise_for_mcp_result(result: object) -> object:
    """Return ``result`` unchanged, or raise its safe coded SDK exception."""
    error = mcp_error_from_result(result)
    if error is not None:
        raise error from None
    return result


def _walk_exceptions(error: BaseException) -> tuple[BaseException, ...]:
    """Flatten nested exception groups with deterministic resource bounds."""
    found: list[BaseException] = []
    stack: list[tuple[BaseException, int]] = [(error, 0)]
    seen: set[int] = set()
    while stack and len(found) < _MAX_EXCEPTION_NODES:
        current, depth = stack.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        found.append(current)
        if depth >= _MAX_EXCEPTION_DEPTH:
            continue
        children = _read_field(current, "exceptions")
        if type(children) not in (tuple, list):
            continue
        # Reverse so ExceptionGroup's public ordering remains the search order.
        for child in reversed(children[:_MAX_EXCEPTION_NODES]):
            if isinstance(child, BaseException):
                stack.append((child, depth + 1))
    return tuple(found)


def _exception_sources(error: BaseException) -> tuple[object, ...]:
    envelope: dict[str, object] = {}
    if isinstance(error, DeepintShieldError):
        # Bind the trusted SDK code and its allowlisted evidence together.
        envelope["code"] = error.code
        envelope["details"] = error.details
    for name in ("error", "data", "details"):
        value = _read_field(error, name)
        if value is not _MISSING:
            envelope[name] = value
    code = _read_field(error, "code")
    if code is not _MISSING and "code" not in envelope:
        envelope["code"] = code
    return (envelope,) if envelope else ()


def mcp_error_from_exception(
    error: BaseException,
    operation: str,
    fallback_code: ErrorCode,
) -> DeepintShieldError:
    """Convert any upstream MCP exception into one safe SDK exception.

    Nested exception groups are inspected for a canonical GAF code and a
    numeric JSON-RPC/MCP code.  Raw messages, arguments, payloads, and causes
    are intentionally not copied to the outward exception.
    """
    # Cancellation and process-control signals are not MCP failures. Keeping
    # them outside the coded boundary preserves cooperative task cancellation
    # and application shutdown for every framework integration.
    if not isinstance(error, Exception):
        raise error

    nodes = _walk_exceptions(error)
    selected_details: dict[str, object] = {}
    selected_node: BaseException = error
    selected_code: ErrorCode | None = None

    # A DeepintShieldError has already crossed the SDK's trusted catalogue
    # boundary. Preserve its precise stable code while rebuilding the outward
    # exception from safe catalogue prose and allowlisted evidence only.
    if isinstance(error, DeepintShieldError):
        try:
            selected_code = ErrorCode(error.code)
        except ValueError:
            selected_code = None
        else:
            selected_details = _merge_evidence(_exception_sources(error))

    # An exact canonical boundary code outranks generic transport siblings.
    if selected_code is None:
        for node in nodes:
            details = _merge_evidence(_exception_sources(node))
            response_code = details.get("response_code")
            if isinstance(response_code, str) and response_code in _GAF_CODES:
                selected_code = ErrorCode(response_code)
                selected_details = details
                selected_node = node
                break

    if selected_code is None:
        try:
            selected_code = (
                fallback_code
                if isinstance(fallback_code, ErrorCode)
                else ErrorCode(fallback_code)
            )
        except (TypeError, ValueError):
            selected_code = ErrorCode.MCP_PROTOCOL_ERROR
        for node in nodes:
            details = _merge_evidence(_exception_sources(node))
            if any(
                key in details
                for key in (
                    "mcp_code",
                    "response_code",
                    "decision_id",
                    "approval_id",
                )
            ):
                selected_details = details
                selected_node = node
                break

    safe_details: dict[str, object] = {}
    safe_operation = _safe_label(operation)
    if safe_operation:
        safe_details["operation"] = safe_operation
    upstream_type = _safe_label(type(selected_node).__name__, default="Exception")
    safe_details["upstream_type"] = upstream_type
    for key in ("mcp_code", "response_code", "decision_id", "approval_id"):
        if key in selected_details:
            safe_details[key] = selected_details[key]
    return _trusted_error(selected_code, safe_details)


__all__ = [
    "mcp_error_from_exception",
    "mcp_error_from_result",
    "raise_for_mcp_result",
]
