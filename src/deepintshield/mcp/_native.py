"""Thin bridge to the official MCP Python SDK.

DeepIntShield owns only gateway connection metadata and the public coded-error
boundary.  The upstream SDK owns Streamable HTTP, protocol negotiation,
sessions, requests, pagination, and native result types.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from importlib import metadata
from typing import TYPE_CHECKING, Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import ValidationError

from ..errors import DeepintShieldError, ErrorCode
from ._errors import mcp_error_from_exception, raise_for_mcp_result

if TYPE_CHECKING:
    from ..client import DeepintShield


_MCP_REQUIREMENT = "mcp>=1.29,<2"
_STABLE_V1_VERSION = re.compile(
    r"^1\.(?P<minor>\d+)(?:\.\d+)?(?:\.post\d+)?(?:\+[A-Za-z0-9.-]+)?$"
)


def _require_supported_mcp() -> None:
    """Fail clearly when the imported MCP distribution is outside v1.29+."""
    try:
        installed = metadata.version("mcp")
    except metadata.PackageNotFoundError:
        installed = ""

    match = _STABLE_V1_VERSION.fullmatch(installed)
    if match is not None and int(match.group("minor")) >= 29:
        return

    details = {"component": "mcp", "requirement": _MCP_REQUIREMENT}
    if installed:
        details["installed_version"] = installed
    raise DeepintShieldError(
        "A supported official MCP Python SDK is required",
        code=ErrorCode.MCP_DEPENDENCY_MISSING,
        details=details,
    ) from None


def _request_error_context(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[str, ErrorCode]:
    """Choose the public boundary from this request, never session-wide state."""
    request = args[0] if args else kwargs.get("request")
    try:
        method = getattr(getattr(request, "root", request), "method", None)
    except Exception:
        method = None
    if type(method) is not str:
        return "send_request", ErrorCode.MCP_PROTOCOL_ERROR
    if method == "tools/call":
        return "call_tool", ErrorCode.MCP_EXECUTION_FAILED
    if method == "tools/list":
        return "list_tools", ErrorCode.MCP_DISCOVERY_FAILED
    if method == "initialize":
        return "connect", ErrorCode.MCP_CONNECTION_FAILED
    return "send_request", ErrorCode.MCP_PROTOCOL_ERROR


class _DeepIntShieldClientSession(ClientSession):
    """Official session with one small, stable exception boundary."""

    def __init__(
        self,
        *args: Any,
        authentication_failures: list[BaseException] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Filled by the gateway workload hook when issuing credentials for a
        # request fails.  The transport then cancels the request waiting for
        # its response, so a bare cancellation here means "look in this list".
        self._authentication_failures = authentication_failures

    def _authentication_failure(self) -> BaseException | None:
        failures = self._authentication_failures
        if not failures:
            return None
        return failures.pop(0)

    async def send_request(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await super().send_request(*args, **kwargs)
        except asyncio.CancelledError:
            reason = self._authentication_failure()
            if reason is None:
                raise
            operation, fallback_code = _request_error_context(args, kwargs)
            raise mcp_error_from_exception(
                reason,
                operation=operation,
                fallback_code=fallback_code,
            ) from None
        except Exception as error:
            operation, fallback_code = _request_error_context(args, kwargs)
            # The native session validates the remote result here. Invalid
            # result schemas remain protocol failures, not operation failures.
            if isinstance(error, ValidationError):
                fallback_code = ErrorCode.MCP_PROTOCOL_ERROR
            raise mcp_error_from_exception(
                error,
                operation=operation,
                fallback_code=fallback_code,
            ) from None

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await super().list_tools(*args, **kwargs)
        except asyncio.CancelledError:
            reason = self._authentication_failure()
            if reason is None:
                raise
            raise mcp_error_from_exception(
                reason,
                operation="list_tools",
                fallback_code=ErrorCode.MCP_DISCOVERY_FAILED,
            ) from None
        except Exception as error:
            raise mcp_error_from_exception(
                error,
                operation="list_tools",
                fallback_code=ErrorCode.MCP_DISCOVERY_FAILED,
            ) from None

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = await super().call_tool(*args, **kwargs)
            return raise_for_mcp_result(result)
        except asyncio.CancelledError:
            reason = self._authentication_failure()
            if reason is None:
                raise
            raise mcp_error_from_exception(
                reason,
                operation="call_tool",
                fallback_code=ErrorCode.MCP_EXECUTION_FAILED,
            ) from None
        except Exception as error:
            raise mcp_error_from_exception(
                error,
                operation="call_tool",
                fallback_code=ErrorCode.MCP_EXECUTION_FAILED,
            ) from None


def _connection_error(error: BaseException) -> DeepintShieldError:
    return mcp_error_from_exception(
        error,
        operation="connect",
        fallback_code=ErrorCode.MCP_CONNECTION_FAILED,
    )


@asynccontextmanager
async def open_mcp_session(
    shield: "DeepintShield",
    *,
    identity: bool = False,
    extra_headers: Mapping[str, str] | None = None,
    terminate_on_close: bool = True,
) -> AsyncGenerator[_DeepIntShieldClientSession, None]:
    """Open an initialized official MCP v1 session on ``shield``'s gateway.

    ``shield.timeout`` bounds individual MCP protocol requests.  Streamable
    HTTP reads may remain open for up to 300 seconds, matching the upstream
    transport's long-lived response behavior.
    """
    _require_supported_mcp()
    if getattr(shield, "_deepintshield_closed", False):
        raise DeepintShieldError(
            "DeepintShield client is closed",
            code=ErrorCode.CLIENT_CLOSED,
        )
    shield.virtual_key_or_raise()
    from ._workload import gateway_task_headers, gateway_workload_hook

    # Acquire workload credentials in the per-request hook so short-lived
    # tokens renew during long sessions instead of freezing in default headers.
    url, headers = shield.connection(provider="mcp", identity=False, extra=extra_headers)
    headers = gateway_task_headers(shield, headers)

    stack = AsyncExitStack()
    session: _DeepIntShieldClientSession | None = None
    # The workload hook records why it could not issue credentials for a
    # request.  The transport answers such a request with a cancellation, so
    # the recorded reason - not the cancellation - is what the caller needs.
    authentication_failures: list[BaseException] = []
    try:
        timeout = httpx.Timeout(shield.timeout, read=300.0)
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                follow_redirects=False,
                event_hooks={
                    "request": [
                        gateway_workload_hook(
                            shield,
                            url,
                            identity=identity,
                            headers=headers,
                            failures=authentication_failures,
                        )
                    ]
                },
            )
        )
        read_stream, write_stream, _get_session_id = await stack.enter_async_context(
            streamable_http_client(
                url,
                http_client=http_client,
                terminate_on_close=terminate_on_close,
            )
        )
        session = _DeepIntShieldClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=shield.timeout),
            authentication_failures=authentication_failures,
        )
        await stack.enter_async_context(session)
        await session.initialize()
    except BaseException as error:
        try:
            await stack.aclose()
        except BaseException:
            # The first setup failure is the actionable one.  Never let a
            # secondary teardown failure replace it or leak native details.
            pass
        if authentication_failures:
            raise _connection_error(authentication_failures[0]) from None
        if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
        raise _connection_error(error) from None

    try:
        # Initialization has completed and the object remains an upstream
        # ClientSession, so callers and frameworks consume native MCP types.
        yield session
    except BaseException as user_error:
        try:
            await stack.__aexit__(
                type(user_error),
                user_error,
                user_error.__traceback__,
            )
        except BaseException:
            # The exception raised by application code is always primary.
            pass
        raise
    else:
        try:
            await stack.aclose()
        except Exception as error:
            raise _connection_error(error) from None


__all__ = ["open_mcp_session"]
