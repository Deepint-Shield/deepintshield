"""Generic MCP client built on top of a DeepintShield instance.

Designed to work with *any* MCP server connected to the gateway. There is no
per-server logic anywhere in this module.
"""
from __future__ import annotations

import json
import uuid
import warnings
from typing import TYPE_CHECKING, Any, AsyncContextManager, Iterable, Mapping, NoReturn

from ..errors import DeepintShieldError, ErrorCode
from .tool import MCPResult, Tool, normalize_result

if TYPE_CHECKING:
    from mcp import ClientSession

    from ..client import DeepintShield


_WARNED_LEGACY_METHODS: set[str] = set()


def _reject_non_json_constant(value: str) -> NoReturn:
    """Reject Python's permissive NaN/Infinity JSON extensions."""
    raise ValueError(f"non-standard JSON constant: {value}")


class MCPClient:
    """DeepIntShield's thin MCP gateway surface.

    New code should use :meth:`connect`, which delegates protocol handling,
    sessions, native types, and pagination to the official MCP Python SDK.
    :meth:`connection` exposes only the URL/header seam required by native
    framework integrations.

    The synchronous REST helpers and custom provider adapters remain as
    compatibility shims for the 2.x line and are planned for removal in 3.0.
    """

    def __init__(self, shield: "DeepintShield") -> None:
        self._shield = shield
        self._anthropic_tool_names: dict[str, str] = {}

    # ───────────────────── preferred native MCP boundary ───────────────────

    def connection(
        self,
        *,
        identity: bool = False,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Return the canonical ``/mcp`` URL and gateway headers.

        Pass this pair to framework-native integrations such as OpenAI
        Agents or ``langchain-mcp-adapters``. Request-scoped credentials are
        session-scoped in Streamable HTTP, so create a separate connection for
        each caller/delegated subject rather than mutating shared headers.
        """
        self._shield.virtual_key_or_raise()
        return self._shield.connection(
            provider="mcp",
            identity=identity,
            extra=extra_headers,
        )

    def connect(
        self,
        *,
        identity: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        terminate_on_close: bool = True,
    ) -> AsyncContextManager["ClientSession"]:
        """Open an official MCP Python SDK session against DeepIntShield.

        Install ``deepintshield[mcp]`` first. The yielded object subclasses the
        official ``mcp.ClientSession`` and therefore returns official MCP
        types without DeepIntShield-owned schema or content normalization.
        Transport, protocol, discovery, and tool failures cross this boundary
        as :class:`DeepintShieldError` with a stable ``.code``.
        """
        if self._shield._deepintshield_closed:
            raise DeepintShieldError(
                "DeepintShield client is closed",
                code=ErrorCode.CLIENT_CLOSED,
            )
        try:
            from ._native import open_mcp_session
        except ImportError:
            raise DeepintShieldError(
                "The official MCP Python SDK is required",
                code=ErrorCode.MCP_DEPENDENCY_MISSING,
                details={"component": "mcp", "requirement": "mcp>=1.29,<2"},
            ) from None

        return open_mcp_session(
            self._shield,
            identity=identity,
            extra_headers=extra_headers,
            terminate_on_close=terminate_on_close,
        )

    @staticmethod
    def error_from_result(result: Any) -> DeepintShieldError | None:
        """Return a coded exception for a failed native MCP result, if any."""
        from ._errors import mcp_error_from_result

        return mcp_error_from_result(result)

    @staticmethod
    def raise_for_result(result: Any) -> Any:
        """Raise a coded exception for failure, otherwise return ``result``.

        Framework interceptors can call this before converting an official
        ``CallToolResult`` into model-visible content.
        """
        from ._errors import raise_for_mcp_result

        return raise_for_mcp_result(result)

    @staticmethod
    def raise_for_error(error: Exception, *, operation: str = "mcp") -> NoReturn:
        """Translate a caught third-party MCP failure to one SDK exception."""
        from ._errors import mcp_error_from_exception

        raise mcp_error_from_exception(
            error,
            operation=operation,
            fallback_code=ErrorCode.MCP_EXECUTION_FAILED,
        ) from None

    @staticmethod
    def _warn_legacy(method: str) -> None:
        if method in _WARNED_LEGACY_METHODS:
            return
        _WARNED_LEGACY_METHODS.add(method)
        warnings.warn(
            f"MCPClient.{method}() is a deprecated 2.x compatibility shim; "
            "use MCPClient.connect() or MCPClient.connection() with an official "
            "MCP/framework SDK. It will be removed in DeepIntShield 3.0.",
            DeprecationWarning,
            stacklevel=3,
        )

    # ─────────────────────────── direct execution ────────────────────────────

    def call(
        self,
        *,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        call_id: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> MCPResult:
        """Execute a single MCP tool.

        Either pass the ``arguments=`` mapping or use ``**kwargs``. Tool name
        is *bare* (no ``<server>-`` prefix); the SDK adds the prefix. Use
        ``extra_headers`` for request-scoped gateway credentials such as
        ``X-MCP-Subject-Token`` or ``X-Agent-Token``; they are never added to
        the tool argument object.
        """
        self._warn_legacy("call")
        merged_args = dict(arguments or {})
        merged_args.update(kwargs)
        qualified = f"{server}-{tool}"
        encode_failed = False
        try:
            encoded_arguments = json.dumps(merged_args, allow_nan=False)
        except (TypeError, ValueError, OverflowError, RecursionError):
            # Circular, excessively nested, and unsupported values are all
            # invalid MCP argument objects. Raise after leaving the encoder
            # handler so private object reprs are not chained as context.
            encoded_arguments = ""
            encode_failed = True
        if encode_failed:
            raise DeepintShieldError(
                "MCP tool arguments must be JSON serializable",
                code=ErrorCode.MCP_ARGUMENTS_INVALID,
                details={"tool": qualified},
            ) from None
        payload = {
            "id": call_id or f"sdk-{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": qualified,
                "arguments": encoded_arguments,
            },
        }
        body = self._shield.request(
            "POST",
            "/v1/mcp/tool/execute",
            json_body=payload,
            extra_headers=self._agent_identity_headers(extra_headers),
            error_code=ErrorCode.MCP_EXECUTION_FAILED,
            require_object=True,
        )
        error = body.get("error")
        response_code = error.get("code") if isinstance(error, Mapping) else None
        if (
            isinstance(response_code, str)
            and response_code.strip().lower().replace("-", "_")
            == ErrorCode.MCP_TOOL_APPROVAL_REQUIRED.value
        ):
            # Canonical GAF deliberately represents REQUIRE_APPROVAL as HTTP
            # 202: the request was accepted for review, but no tool executed.
            # The generic transport treats all 2xx responses as success, so
            # the MCP surface must preserve this fail-closed outcome instead
            # of manufacturing an empty successful MCPResult.
            raise DeepintShieldError.from_response(
                202,
                body,
                fallback_code=ErrorCode.MCP_EXECUTION_FAILED,
                details={"method": "POST", "path": "/v1/mcp/tool/execute"},
            )
        return normalize_result(qualified, body)

    def call_qualified(self, qualified_name: str, arguments: Mapping[str, Any] | str, **kwargs: Any) -> MCPResult:
        """Execute by qualified ``<server>-<tool>`` name. Accepts a JSON
        string for ``arguments`` (matching the OpenAI tool_call shape)."""
        self._warn_legacy("call_qualified")
        if isinstance(arguments, str):
            decode_failed = False
            try:
                args_dict = (
                    json.loads(arguments, parse_constant=_reject_non_json_constant)
                    if arguments
                    else {}
                )
            except (ValueError, RecursionError):
                # Leave the JSON handler before raising so the decoder's
                # private ``doc`` (the raw tool arguments), integer-limit
                # error, or nesting failure is not retained as context.
                args_dict = {}
                decode_failed = True
            if decode_failed:
                raise DeepintShieldError(
                    "MCP tool arguments must be valid JSON",
                    code=ErrorCode.MCP_ARGUMENTS_INVALID,
                    details={"tool": qualified_name},
                ) from None
            if not isinstance(args_dict, dict):
                raise DeepintShieldError(
                    "MCP tool arguments must decode to an object",
                    code=ErrorCode.MCP_ARGUMENTS_INVALID,
                    details={"tool": qualified_name},
                )
        else:
            args_dict = dict(arguments or {})
        server, _, tool = qualified_name.partition("-")
        if not tool:
            raise DeepintShieldError(
                f"Tool name '{qualified_name}' is missing a server prefix",
                code=ErrorCode.MCP_TOOL_NAME_INVALID,
                details={"tool": qualified_name},
            )
        return self.call(server=server, tool=tool, arguments=args_dict, **kwargs)

    # ───────────────────────── agent identity headers ────────────────────────

    def _agent_identity_headers(
        self, extra_headers: Mapping[str, str] | None
    ) -> dict[str, str]:
        """Attach the client's agent selector to a gateway MCP call.

        A DeepintShield client represents one agent identity. The PDP decide
        path already sends ``X-Agent-Subject``; the brokered MCP path used to
        send nothing, so a virtual key bound to more than one active agent was
        refused with ``mcp_tool_authorization_unavailable`` on execute while
        its decide calls succeeded. The selector is derived locally (no
        discovery round-trip). Workload tokens stay request-scoped: pass
        ``X-Agent-Token`` through ``extra_headers`` as before. Explicit caller
        headers always win; nothing is added for a client without an
        ``agent_name``.
        """
        headers = dict(extra_headers or {})
        agent_name = str(getattr(self._shield, "agent_name", "") or "").strip()
        if not agent_name:
            return headers
        if any(str(key).lower() == "x-agent-subject" for key in headers):
            return headers
        from ..agentic.registry import _registry_key

        agent_key = _registry_key(agent_name)
        if agent_key:
            headers["X-Agent-Subject"] = f"agent:{agent_key}"
        return headers

    # ─────────────────────────── discovery (optional) ────────────────────────

    def list_tools(
        self,
        *,
        server: str | None = None,
        admin_token: str | None = None,
    ) -> list[Tool]:
        """Discover tools via the workspace-scoped MCP registry API.

        Current gateways accept the client's Virtual Key. ``admin_token`` is
        retained for older or separately protected deployments and, when
        supplied, is sent as ``Authorization: Bearer ...``. If discovery is
        unavailable in your environment, supply tool definitions manually.
        """
        self._warn_legacy("list_tools")
        headers: dict[str, str] = self._agent_identity_headers(None)
        if admin_token:
            headers["Authorization"] = f"Bearer {admin_token}"
        payload = self._shield.request(
            "GET",
            "/api/mcp/clients",
            extra_headers=headers,
            error_code=ErrorCode.MCP_DISCOVERY_FAILED,
            require_object=True,
        ) or {}
        clients = payload.get("clients") or payload.get("data") or payload
        tools: list[Tool] = []
        for entry in clients or []:
            config = entry.get("config", entry) if isinstance(entry, dict) else {}
            client_name = (config.get("name") or entry.get("name") or "") if isinstance(entry, dict) else ""
            if not client_name:
                continue
            if server is not None and client_name != server:
                continue
            for raw_tool in (entry.get("tools") if isinstance(entry, dict) else None) or []:
                tool_name = raw_tool.get("name", "")
                prefix = f"{client_name}-"
                if tool_name.startswith(prefix):
                    tool_name = tool_name[len(prefix):]
                tools.append(
                    Tool(
                        server=client_name,
                        name=tool_name,
                        description=raw_tool.get("description") or "",
                        schema=raw_tool.get("parameters") or raw_tool.get("inputSchema") or {},
                    )
                )
        return tools

    # ─────────────────────────── adapters ────────────────────────────────────

    def to_openai(self, tools: Iterable[Tool]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI / LiteLLM ``tools=`` array shape."""
        self._warn_legacy("to_openai")
        from .adapters import openai as _openai
        return _openai.to_openai(tools)

    def run_openai_tool_calls(
        self,
        tool_calls: Iterable[Any],
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute each OpenAI ``tool_call`` and return the matching list of
        ``role: tool`` messages to append to the conversation. Request-scoped
        delegated or workload credentials can be supplied with
        ``extra_headers``."""
        self._warn_legacy("run_openai_tool_calls")
        from .adapters import openai as _openai
        return _openai.run_tool_calls(
            self,
            tool_calls,
            extra_headers=extra_headers,
        )

    def to_anthropic(self, tools: Iterable[Tool]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic Messages API ``tools=`` array shape."""
        self._warn_legacy("to_anthropic")
        from .adapters import anthropic as _anthropic
        return _anthropic.to_anthropic(tools, name_map=self._anthropic_tool_names)

    def run_anthropic_tool_uses(
        self,
        content: Iterable[Any],
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute each ``tool_use`` block in an assistant message and return
        the corresponding ``tool_result`` blocks for the next user turn.
        Request-scoped delegated or workload credentials can be supplied with
        ``extra_headers``."""
        self._warn_legacy("run_anthropic_tool_uses")
        from .adapters import anthropic as _anthropic
        return _anthropic.run_tool_uses(
            self,
            content,
            extra_headers=extra_headers,
            name_map=self._anthropic_tool_names,
        )

    def to_langchain(self, tools: Iterable[Tool]) -> list[Any]:
        """Wrap each MCP tool as a LangChain ``BaseTool`` that can be passed
        to LangChain agents or LangGraph ``ToolNode``s."""
        self._warn_legacy("to_langchain")
        from .adapters import langchain as _langchain
        return _langchain.to_langchain(self, tools)
