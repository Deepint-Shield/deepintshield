from __future__ import annotations

import importlib
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from deepintshield.errors import DeepintShieldError


class _Shield:
    timeout = 17.0

    def __init__(self) -> None:
        self._deepintshield_closed = False
        self.virtual_key_checked = False
        self.connection_calls: list[dict[str, Any]] = []

    def virtual_key_or_raise(self) -> str:
        self.virtual_key_checked = True
        return "sk-ds-test"

    def connection(self, **kwargs: Any) -> tuple[str, dict[str, str]]:
        self.connection_calls.append(kwargs)
        return (
            "https://gateway.example/mcp",
            {
                "x-deepintshield-vk": "sk-ds-test",
                "x-request-scope": "delegated",
            },
        )


@pytest.fixture
def native_module(monkeypatch):
    state: dict[str, Any] = {
        "events": [],
        "version": "1.29.0",
        "initialize_error": None,
        "send_error": None,
        "list_error": None,
        "call_error": None,
        "session_exit_error": None,
        "transport_exit_error": None,
        "http_exit_error": None,
        "send_result": object(),
        "list_result": SimpleNamespace(tools=[]),
        "call_result": SimpleNamespace(isError=False, content=[]),
    }

    class FakeClientSession:
        def __init__(self, read_stream, write_stream, **kwargs):
            state["session"] = self
            state["session_args"] = (read_stream, write_stream)
            state["session_kwargs"] = kwargs

        async def __aenter__(self):
            state["events"].append("session-enter")
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            state["events"].append("session-exit")
            if state["session_exit_error"] is not None:
                raise state["session_exit_error"]
            return False

        async def initialize(self):
            state["events"].append("initialize")
            if state["initialize_error"] is not None:
                raise state["initialize_error"]
            state["initialized"] = True
            return SimpleNamespace(protocolVersion="2025-06-18")

        async def send_request(self, *args, **kwargs):
            if state["send_error"] is not None:
                raise state["send_error"]
            return state["send_result"]

        async def list_tools(self, *args, **kwargs):
            if state["list_error"] is not None:
                raise state["list_error"]
            return state["list_result"]

        async def call_tool(self, *args, **kwargs):
            if state["call_error"] is not None:
                raise state["call_error"]
            return state["call_result"]

    @asynccontextmanager
    async def fake_streamable_http_client(
        url,
        *,
        http_client,
        terminate_on_close,
    ):
        state["transport_args"] = {
            "url": url,
            "http_client": http_client,
            "terminate_on_close": terminate_on_close,
        }
        state["events"].append("transport-enter")
        try:
            yield ("read-stream", "write-stream", lambda: "session-id")
        finally:
            state["events"].append("transport-exit")
            if state["transport_exit_error"] is not None:
                raise state["transport_exit_error"]

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            state["http_client"] = self
            state["http_kwargs"] = kwargs

        async def __aenter__(self):
            state["events"].append("http-enter")
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            state["events"].append("http-exit")
            if state["http_exit_error"] is not None:
                raise state["http_exit_error"]
            return False

    fake_mcp = ModuleType("mcp")
    fake_mcp.__path__ = []
    fake_mcp.ClientSession = FakeClientSession
    fake_client = ModuleType("mcp.client")
    fake_client.__path__ = []
    fake_transport = ModuleType("mcp.client.streamable_http")
    fake_transport.streamable_http_client = fake_streamable_http_client

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", fake_client)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", fake_transport)
    sys.modules.pop("deepintshield.mcp._native", None)

    native = importlib.import_module("deepintshield.mcp._native")
    monkeypatch.setattr(native.metadata, "version", lambda _name: state["version"])
    monkeypatch.setattr(native.httpx, "AsyncClient", FakeAsyncClient)
    try:
        yield native, state
    finally:
        sys.modules.pop("deepintshield.mcp._native", None)
        package = sys.modules.get("deepintshield.mcp")
        if package is not None and getattr(package, "_native", None) is native:
            delattr(package, "_native")


@pytest.mark.asyncio
async def test_open_session_uses_gateway_connection_and_native_types(native_module):
    native, state = native_module
    shield = _Shield()

    async with native.open_mcp_session(
        shield,
        identity=True,
        extra_headers={"x-request-scope": "delegated"},
        terminate_on_close=False,
    ) as session:
        assert state["initialized"] is True
        assert await session.list_tools() is state["list_result"]

    assert shield.virtual_key_checked is True
    assert shield.connection_calls == [
        {
            "provider": "mcp",
            "identity": True,
            "extra": {"x-request-scope": "delegated"},
        }
    ]
    assert state["transport_args"] == {
        "url": "https://gateway.example/mcp",
        "http_client": state["http_client"],
        "terminate_on_close": False,
    }
    assert state["http_kwargs"]["headers"]["x-deepintshield-vk"] == "sk-ds-test"
    assert state["http_kwargs"]["follow_redirects"] is False
    assert state["http_kwargs"]["timeout"].connect == shield.timeout
    assert state["http_kwargs"]["timeout"].read == 300.0
    assert state["session_args"] == ("read-stream", "write-stream")
    assert state["session_kwargs"]["read_timeout_seconds"] == timedelta(
        seconds=shield.timeout
    )
    assert state["events"] == [
        "http-enter",
        "transport-enter",
        "session-enter",
        "initialize",
        "session-exit",
        "transport-exit",
        "http-exit",
    ]


@pytest.mark.asyncio
async def test_native_tool_and_protocol_failures_are_simple_coded_errors(
    native_module,
):
    native, state = native_module
    state["call_result"] = SimpleNamespace(
        isError=True,
        content=[],
        structuredContent={
            "code": "mcp_tool_authorization_denied",
            "decision_id": "decision-123",
        },
    )

    async with native.open_mcp_session(_Shield()) as session:
        with pytest.raises(DeepintShieldError) as denied:
            await session.call_tool("server-tool", {"value": 1})
        assert denied.value.code == "mcp_tool_authorization_denied"
        assert denied.value.details["decision_id"] == "decision-123"

        state["call_error"] = RuntimeError("private upstream diagnostic")
        with pytest.raises(DeepintShieldError) as execution:
            await session.call_tool("server-tool", {})
        assert execution.value.code == "mcp_execution_failed"
        assert "private upstream diagnostic" not in repr(execution.value.to_dict())

        state["list_error"] = RuntimeError("private list diagnostic")
        with pytest.raises(DeepintShieldError) as discovery:
            await session.list_tools()
        assert discovery.value.code == "mcp_discovery_failed"

        state["send_error"] = RuntimeError("private protocol diagnostic")
        with pytest.raises(DeepintShieldError) as protocol:
            await session.send_request(object(), object)
        assert protocol.value.code == "mcp_protocol_error"


@pytest.mark.asyncio
async def test_initialization_and_normal_cleanup_failures_use_connection_code(
    native_module,
):
    native, state = native_module
    state["initialize_error"] = RuntimeError("private setup diagnostic")

    with pytest.raises(DeepintShieldError) as setup:
        async with native.open_mcp_session(_Shield()):
            pytest.fail("an uninitialized session must never be yielded")

    assert setup.value.code == "mcp_connection_failed"
    assert state["events"][-3:] == [
        "session-exit",
        "transport-exit",
        "http-exit",
    ]

    state["initialize_error"] = None
    state["transport_exit_error"] = RuntimeError("private cleanup diagnostic")
    with pytest.raises(DeepintShieldError) as cleanup:
        async with native.open_mcp_session(_Shield()):
            pass
    assert cleanup.value.code == "mcp_connection_failed"


@pytest.mark.asyncio
async def test_user_exception_is_not_replaced_by_cleanup_failure(native_module):
    native, state = native_module
    state["transport_exit_error"] = RuntimeError("cleanup failed")

    application_error = LookupError("application failure")
    with pytest.raises(LookupError) as caught:
        async with native.open_mcp_session(_Shield()):
            raise application_error

    assert caught.value is application_error
    assert state["events"][-3:] == [
        "session-exit",
        "transport-exit",
        "http-exit",
    ]


@pytest.mark.asyncio
async def test_unsupported_official_sdk_version_has_dependency_code(native_module):
    native, state = native_module
    state["version"] = "2.0.0"
    shield = _Shield()

    with pytest.raises(DeepintShieldError) as caught:
        async with native.open_mcp_session(shield):
            pytest.fail("unsupported MCP versions must not be entered")

    assert caught.value.code == "mcp_dependency_missing"
    assert caught.value.details == {
        "component": "mcp",
        "requirement": "mcp>=1.29,<2",
        "installed_version": "2.0.0",
    }
    assert shield.connection_calls == []


@pytest.mark.asyncio
async def test_session_rechecks_closed_state_when_context_is_entered(native_module):
    native, state = native_module
    shield = _Shield()
    context = native.open_mcp_session(shield)
    shield._deepintshield_closed = True

    with pytest.raises(DeepintShieldError) as caught:
        async with context:
            pytest.fail("a client closed after context construction must stay closed")

    assert caught.value.code == "client_closed"
    assert shield.connection_calls == []
    assert state["events"] == []
