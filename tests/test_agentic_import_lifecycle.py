"""Framework startup must finish before automatic enforcement inspects it."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest


@pytest.mark.parametrize("supported", [True, False])
def test_late_framework_submodule_import_finishes_before_guard_install(tmp_path, supported):
    package = tmp_path / "litellm"
    package.mkdir()
    (package / "helpers.py").write_text("value = 1\n")
    (package / "__init__.py").write_text(
        "from litellm.helpers import value\n"
        + ("def completion(*args, **kwargs):\n    return 'raw'\n" if supported else "")
    )
    script = f"""
import sys
sys.path.insert(0, {str(tmp_path)!r})
from deepintshield import DeepintShield
with DeepintShield(virtual_key="sk-ds-test", base_url="http://127.0.0.1:1"):
    try:
        import litellm
    except RuntimeError as error:
        assert not {supported!r}, str(error)
        assert getattr(error, "_deepintshield_error_code", "") == "governance_configuration_error"
    else:
        assert {supported!r}, "unsupported framework imported ungoverned"
        assert getattr(litellm.completion, "_deepintshield_guarded", False)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(importlib.util.find_spec("langchain_openai") is None, reason="langchain-openai is not installed")
def test_langchain_provider_first_import_after_client_creation():
    result = subprocess.run(
        [sys.executable, "-c", """
from deepintshield import DeepintShield
with DeepintShield(virtual_key="sk-ds-test", base_url="http://127.0.0.1:1") as shield:
    model = shield.langchain(model="test-model")
    assert model.model_name == "test-model"
    assert str(model.openai_api_base).rstrip("/") == "http://127.0.0.1:1/langchain"
    from langchain_core.tools import BaseTool
    assert getattr(BaseTool.run, "_deepintshield_guarded", False)
    assert getattr(BaseTool.arun, "_deepintshield_guarded", False)
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
