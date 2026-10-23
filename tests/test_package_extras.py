"""Installation contracts that prevent pip falling back to the old SDK."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
)["project"]
EXTRAS = PROJECT["optional-dependencies"]


def _requirements(extra: str, python: str) -> dict[str, Requirement]:
    environment = {
        **default_environment(),
        "python_version": python,
        "python_full_version": f"{python}.0",
        "extra": extra,
    }
    requirements = [Requirement(raw) for raw in EXTRAS[extra]]
    return {
        canonicalize_name(requirement.name): requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(environment)
    }


@pytest.mark.parametrize("python", ["3.10", "3.11", "3.12", "3.13", "3.14"])
def test_all_keeps_every_integration_supported_by_the_interpreter(python):
    aggregate = _requirements("all", python)
    expected: dict[str, set[str]] = {}
    for name in EXTRAS:
        if name in {"all", "dev"}:
            continue
        for package, requirement in _requirements(name, python).items():
            expected.setdefault(package, set()).update(requirement.extras)
    if python == "3.14":
        # A transitive compatibility bound for Google ADK + AutoGen.
        expected["google-api-core"] = set()

    assert set(aggregate) == set(expected)
    for package, requested_extras in expected.items():
        assert requested_extras <= aggregate[package].extras, package


def test_removed_framework_is_not_an_installation_dependency():
    assert "crewai" not in EXTRAS
    for python in ("3.10", "3.13", "3.14"):
        assert "crewai" not in _requirements("all", python)


@pytest.mark.parametrize("python", ["3.13", "3.14"])
def test_aggregate_has_a_shared_openai_and_mcp_version_set(python):
    aggregate = _requirements("all", python)
    # Published dependency constraints, from each version's PyPI JSON metadata.
    # Full resolver checks run separately against the built distribution.
    framework_metadata = {
        "openai-agents": ("0.20.0", ["openai>=2.45.0,<3", "mcp>=1.19,<3"]),
        "pydantic-ai-slim": ("2.31.0", ["openai>=2.45.0"]),
        "strands-agents": ("1.55.0", ["openai>=1.68,<3", "mcp>=1.23,<2.2"]),
    }
    shared_versions = {"openai": "2.54.0", "mcp": "1.29.0"}

    for package, version in shared_versions.items():
        assert version in aggregate[package].specifier
    for package, (version, dependencies) in framework_metadata.items():
        if package not in aggregate:
            continue
        assert version in aggregate[package].specifier, package
        for raw in dependencies:
            dependency = Requirement(raw)
            assert shared_versions[dependency.name] in dependency.specifier, package

    # OpenAI 3 would make the Strands and LlamaIndex installations conflict.
    assert "3.0.0" not in aggregate["openai"].specifier


def test_python_314_google_adk_and_autogen_can_share_protobuf_five():
    aggregate = _requirements("all", "3.14")
    google_api_core = aggregate["google-api-core"]
    assert "2.25.2" in google_api_core.specifier
    # 2.26+ requires grpcio-status>=1.75.1 on Python 3.14, requiring protobuf 6.
    assert "2.26.0" not in google_api_core.specifier
    assert "google-api-core" not in _requirements("all", "3.13")
    # ADK 1.36+ requires GenAI 2, unavailable with the compatible AI Platform
    # releases selected by Google API Core 2.25.x.
    assert "1.35.0" in aggregate["google-adk"].specifier
    assert "1.36.0" not in aggregate["google-adk"].specifier
    assert "1.39.1" in _requirements("all", "3.13")["google-adk"].specifier
    # AutoGen 0.7.5 and grpcio-status 1.71.2 requirements from PyPI metadata.
    assert "5.29.6" in Requirement("protobuf~=5.29.3").specifier
    assert "5.29.6" in Requirement("protobuf>=5.26.1,<6.0dev").specifier
