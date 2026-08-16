from __future__ import annotations

import importlib

import pytest

from deepintshield import DeepintShield
from deepintshield.config import DEFAULT_BASE_URL


def _skip_if_missing(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed")


def test_openai_client_points_at_gateway():
    _skip_if_missing("openai")
    shield = DeepintShield(virtual_key="sk-ds-1")
    client = shield.openai()
    assert str(client.base_url).rstrip("/") == f"{DEFAULT_BASE_URL}/openai"
    assert client.default_headers.get("x-deepintshield-vk") == "sk-ds-1"


def test_openai_passthrough_uses_v1_suffix():
    _skip_if_missing("openai")
    shield = DeepintShield(virtual_key="sk-ds-1")
    client = shield.openai(passthrough=True)
    assert str(client.base_url).rstrip("/") == f"{DEFAULT_BASE_URL}/openai_passthrough/v1"


def test_anthropic_client_points_at_gateway():
    _skip_if_missing("anthropic")
    shield = DeepintShield(virtual_key="sk-ds-1")
    client = shield.anthropic()
    assert str(client.base_url).rstrip("/") == f"{DEFAULT_BASE_URL}/anthropic"
    assert client.default_headers.get("x-deepintshield-vk") == "sk-ds-1"


def test_anthropic_passthrough_uses_passthrough_suffix():
    _skip_if_missing("anthropic")
    shield = DeepintShield(virtual_key="sk-ds-1")
    client = shield.anthropic(passthrough=True)
    assert str(client.base_url).rstrip("/") == f"{DEFAULT_BASE_URL}/anthropic_passthrough"


def test_langchain_returns_chat_openai_pointing_at_gateway():
    _skip_if_missing("langchain_openai")
    shield = DeepintShield(virtual_key="sk-ds-1")
    llm = shield.langchain(model="gpt-4o-mini")
    assert getattr(llm, "model_name", None) or getattr(llm, "model", None) == "gpt-4o-mini"
    openai_api_base = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None)
    assert "app.deepintshield.com/langchain" in str(openai_api_base)


def test_litellm_shield_is_constructible():
    _skip_if_missing("litellm")
    shield = DeepintShield(virtual_key="sk-ds-1")
    wrapper = shield.litellm()
    assert callable(getattr(wrapper, "completion", None))
