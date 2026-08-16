from __future__ import annotations

import sys

import pytest

from deepintshield import DeepintShield, DeepintShieldError


def test_mcp_connection_reuses_canonical_gateway_headers() -> None:
    shield = DeepintShield(
        virtual_key="sk-ds-test",
        base_url="https://gateway.example/",
        default_headers={"x-installation": "sdk-test"},
    )
    try:
        url, headers = shield.mcp.connection(
            identity=False,
            extra_headers={"X-MCP-Subject-Token": "delegated-token"},
        )
    finally:
        shield.close()

    assert url == "https://gateway.example/mcp"
    assert headers["x-deepintshield-vk"] == "sk-ds-test"
    assert headers["x-deepintshield-app"] == "deepintshield"
    assert headers["x-installation"] == "sdk-test"
    assert headers["X-MCP-Subject-Token"] == "delegated-token"


def test_mcp_connection_requires_a_virtual_key() -> None:
    shield = DeepintShield(base_url="https://gateway.example")
    try:
        with pytest.raises(DeepintShieldError) as caught:
            shield.mcp.connection()
    finally:
        shield.close()

    assert caught.value.code == "virtual_key_missing"


def test_mcp_connect_rejects_a_closed_client_before_opening_transport() -> None:
    shield = DeepintShield(
        virtual_key="sk-ds-test",
        base_url="https://gateway.example",
    )
    shield.close()

    with pytest.raises(DeepintShieldError) as caught:
        shield.mcp.connect()

    assert caught.value.code == "client_closed"


def test_external_tool_error_uses_the_simple_execution_code() -> None:
    shield = DeepintShield(
        virtual_key="sk-ds-test",
        base_url="https://gateway.example",
    )
    try:
        with pytest.raises(DeepintShieldError) as caught:
            shield.mcp.raise_for_error(
                RuntimeError("private framework diagnostic"),
                operation="framework_tool",
            )
    finally:
        shield.close()

    assert caught.value.code == "mcp_execution_failed"
    assert "private framework diagnostic" not in repr(caught.value.to_dict())


def test_missing_official_mcp_dependency_is_one_coded_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shield = DeepintShield(
        virtual_key="sk-ds-test",
        base_url="https://gateway.example",
    )
    monkeypatch.setitem(sys.modules, "deepintshield.mcp._native", None)
    try:
        with pytest.raises(DeepintShieldError) as caught:
            shield.mcp.connect()
    finally:
        shield.close()

    assert caught.value.code == "mcp_dependency_missing"
    assert caught.value.details == {
        "component": "mcp",
        "requirement": "mcp>=1.29,<2",
    }
