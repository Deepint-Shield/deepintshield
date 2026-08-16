"""Internal helpers for preserving the canonical MCP authorization boundary."""

from __future__ import annotations

from ...errors import DeepintShieldError, ErrorCode


_MCP_AUTHORIZATION_BOUNDARY_CODES = frozenset(
    {
        ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED.value,
        ErrorCode.MCP_TOOL_AUTHORIZATION_UNAVAILABLE.value,
        ErrorCode.MCP_TOOL_APPROVAL_REQUIRED.value,
    }
)


def is_mcp_authorization_boundary_error(error: BaseException) -> bool:
    """Return whether ``error`` must leave a convenience tool runner intact."""
    if not isinstance(error, DeepintShieldError):
        return False
    if error.code in _MCP_AUTHORIZATION_BOUNDARY_CODES:
        return True
    # Preserve fail-closed behavior for a bounded future server code even
    # before this SDK release knows its catalogue metadata. ``from_response``
    # keeps that value only in diagnostic details and assigns a generic public
    # code, so inspect the retained wire code as well.
    response_code = error.details.get("response_code")
    return isinstance(response_code, str) and (
        response_code.startswith("mcp_tool_authorization_")
        or response_code == ErrorCode.MCP_TOOL_APPROVAL_REQUIRED.value
    )
