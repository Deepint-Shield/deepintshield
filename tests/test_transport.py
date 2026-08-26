from __future__ import annotations

import pytest

from deepintshield import DeepintShield


def test_connection_points_at_gateway_with_vk():
    shield = DeepintShield(virtual_key="sk-ds-x", base_url="http://gw.example")
    base_url, headers = shield.connection()
    assert base_url == "http://gw.example/openai"
    assert headers["x-deepintshield-vk"] == "sk-ds-x"
    # Attribution rides as headers for transparent (L1) traffic.
    assert headers["x-deepintshield-app"] == "deepintshield"
    # No agent name configured means NO agent attribution. There used to be a
    # "deepintshield-agent" default, which made every unnamed workload on the
    # estate share one attribution identity - so the header said something the
    # platform could not act on. Absent is the honest answer.
    assert "x-deepintshield-agent" not in headers
    # Identity is opt-in so chat never blocks on discovery.
    assert "X-Agent-Token" not in headers


def test_connection_carries_the_configured_agent_name():
    shield = DeepintShield(
        virtual_key="sk-ds-x", base_url="http://gw.example", agent_name="billing-bot"
    )
    _base, headers = shield.connection()
    assert headers["x-deepintshield-agent"] == "billing-bot"


def test_connection_extra_headers_merge():
    shield = DeepintShield(virtual_key="sk-ds-x", base_url="http://gw.example")
    _base, headers = shield.connection(extra={"x-custom": "1"})
    assert headers["x-custom"] == "1"


def test_http_client_prewired():
    shield = DeepintShield(virtual_key="sk-ds-x", base_url="http://gw.example")
    client = shield.http_client()
    assert str(client.base_url).rstrip("/") == "http://gw.example/openai"
    assert client.headers["x-deepintshield-vk"] == "sk-ds-x"
    client.close()


def test_bind_unknown_framework_raises():
    shield = DeepintShield(virtual_key="sk-ds-x")
    with pytest.raises(ValueError):
        shield.bind("not-a-framework")


def test_framework_binder_accessors():
    shield = DeepintShield(virtual_key="sk-ds-x")
    for name in ("crewai", "openai_agents", "llamaindex", "autogen"):
        binder = getattr(shield, name)()
        assert name.split("_")[0] in repr(binder).lower()


def test_bind_aliases():
    shield = DeepintShield(virtual_key="sk-ds-x")
    # langchain aliases to the langgraph binder; pydantic-ai normalises too.
    assert repr(shield.bind("langchain"))  # does not raise
    assert repr(shield.bind("pydantic-ai"))
    assert repr(shield.bind("llama-index"))


def test_connection_provider_selects_route():
    shield = DeepintShield(virtual_key="sk-ds-x", base_url="http://gw.example")
    assert shield.connection()[0] == "http://gw.example/openai"
    assert shield.connection(provider="anthropic")[0] == "http://gw.example/anthropic"
    assert shield.connection(provider="genai")[0] == "http://gw.example/genai"


def test_create_headers_parity_helper():
    shield = DeepintShield(virtual_key="sk-ds-x", base_url="http://gw.example")
    headers = shield.create_headers()
    assert headers["x-deepintshield-vk"] == "sk-ds-x"
    assert headers["x-deepintshield-app"] == "deepintshield"
    assert "X-Agent-Token" not in headers
