"""A program whose server run ended for its identity reports the bundle it submitted.

Under "every workspace" a plain agent's first run is handed to the server and
approved as VK-trusted; the server run then ends in preparation because that
identity cannot use one, and the program runs on its own machine from the next
start. Its approval is pinned to the fingerprint of the SUBMITTED program
bundle. The gateway's "not offered" capabilities answer then carries
``report_program_bundle``, and the local discovery manifest is built exactly as
the submission built it, so the approved fingerprint still matches.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

from deepintshield import DeepintShield
from deepintshield.agentic import managed_execution as managed
from deepintshield.agentic.errors import GovernanceConfigurationError

_CAPABILITIES = "/api/agentic-new/managed-execution/capabilities"
_SCRIPT = "from langgraph.graph import StateGraph, START, END\n\nkey = 'sk-ds-managed-test'\nprint('math')\n"


def engine_for(handler):
    return SimpleNamespace(
        gateway_url="https://gateway.example", virtual_key="sk-ds-managed-test",
        _http=httpx.Client(transport=httpx.MockTransport(handler)),
        _headers=lambda **_: {"Authorization": "Bearer sk-ds-managed-test", "X-Agent-Subject": "agent:math"},
    )


@pytest.fixture
def program(tmp_path, monkeypatch):
    """The developer's script, as ``__main__`` names it."""
    script = tmp_path / "math_agent.py"
    script.write_text(_SCRIPT)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script), raising=False)
    return script


def _manifest():
    return {
        "agents": [{"key": "math"}],
        "network": {"digest": "sha256:topology"},
        "code_artifacts": [{
            "subject_kind": "agent", "subject_key": "math", "language": "python",
            "source": "def increment(state):\n    return state\n",
            "digest": "sha256:" + "1" * 64, "self_attested_content_digest": "sha256:" + "2" * 64,
        }],
        "code_artifact_coverage": [{"subject_kind": "agent", "subject_key": "math", "status": "captured"}],
    }


# --- the flag is read only from a "not offered" capabilities answer ------------


@pytest.mark.parametrize(
    ("status", "answer", "flag"),
    [
        pytest.param(200, {"enabled": False, "report_program_bundle": True}, True, id="asked_for"),
        pytest.param(200, {"enabled": False}, False, id="plain_not_offered"),
        pytest.param(200, {"enabled": False, "report_program_bundle": "true"}, False, id="not_a_boolean"),
        pytest.param(200, {"enabled": False, "report_program_bundle": 1}, False, id="not_true"),
        pytest.param(200, {"enabled": True, "report_program_bundle": True}, False, id="an_offer_is_no_fall_back"),
        pytest.param(404, {"enabled": False, "report_program_bundle": True}, False, id="no_capabilities_route"),
    ],
)
def test_flag_is_read_only_from_a_not_offered_capabilities_answer(status, answer, flag):
    engine = engine_for(lambda request: httpx.Response(status, json=answer))
    try:
        settings = managed.capabilities(engine)
        assert engine._managed_execution_report_bundle is flag
        # The answer itself is unchanged: "not offered" is still an empty dict.
        assert settings == (answer if answer.get("enabled") is True and status == 200 else {})
    finally:
        engine._http.close()


def test_a_later_answer_without_the_flag_clears_it():
    answers = iter([
        {"enabled": False, "report_program_bundle": True},
        {"enabled": False},
    ])
    engine = engine_for(lambda request: httpx.Response(200, json=next(answers)))
    try:
        assert managed.capabilities(engine) == {}
        assert engine._managed_execution_report_bundle is True
        assert managed.capabilities(engine, refresh_disabled=True) == {}
        assert engine._managed_execution_report_bundle is False
    finally:
        engine._http.close()


def test_an_engine_that_is_never_asked_carries_no_flag(program, monkeypatch):
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("packaged without the flag"))
    engine = SimpleNamespace(virtual_key="sk-ds-managed-test")
    assert managed.program_bundle_manifest(engine, _manifest()) is None
    assert not hasattr(engine, "_managed_execution_report_bundle")


# --- the reported manifest -------------------------------------------------------


def test_with_the_flag_the_manifest_is_the_one_a_submission_builds(program):
    # The real packager pins the script's langgraph import from installed metadata.
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    from deepintshield.agentic.registry import _wire_attested_payload

    engine = SimpleNamespace(virtual_key="sk-ds-managed-test", _managed_execution_report_bundle=True)
    manifest = _manifest()
    submitted = _wire_attested_payload(engine, managed.augment_manifest(engine, manifest, managed.package_script(program, {})["files"]))
    reported = _wire_attested_payload(engine, managed.program_bundle_manifest(engine, manifest))
    ordinary = _wire_attested_payload(engine, manifest)
    assert reported == submitted
    assert reported["blueprint_digest"] == submitted["blueprint_digest"] != ordinary["blueprint_digest"]
    assert manifest == _manifest(), "the caller's manifest is never changed"
    assert "sk-ds-managed-test" not in json.dumps(reported)
    assert reported["code_artifacts"][0]["source"].startswith("# file: math_agent.py\n")


def test_without_the_flag_the_ordinary_manifest_is_reported(program, monkeypatch):
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("packaged without the flag"))
    for flag in (False, None, "true", 1):
        engine = SimpleNamespace(virtual_key="sk-ds-managed-test", _managed_execution_report_bundle=flag)
        assert managed.program_bundle_manifest(engine, _manifest()) is None


def test_a_worker_never_reports_a_second_bundle(program, monkeypatch):
    monkeypatch.setattr(managed, "_WORKER_FILES", {"math_agent.py": _SCRIPT})
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a worker packaged its program"))
    engine = SimpleNamespace(virtual_key="sk-ds-managed-test", _managed_execution_report_bundle=True)
    assert managed.program_bundle_manifest(engine, _manifest()) is None


def _refuse(code):
    def package(*_):
        raise GovernanceConfigurationError(framework="managed-execution", code=code)

    return package


@pytest.mark.parametrize(
    "obstacle",
    ["dependency_missing", "bundle_too_large", "unexpected_error", "computed_import", "no_entry_file", "not_python", "no_agent"],
)
def test_a_program_that_cannot_be_packaged_reports_the_ordinary_manifest(program, monkeypatch, obstacle):
    engine = SimpleNamespace(virtual_key="sk-ds-managed-test", _managed_execution_report_bundle=True)
    manifest = _manifest()
    if obstacle == "dependency_missing":
        monkeypatch.setattr(managed, "package_script", _refuse("managed_execution_dependency_missing"))
    elif obstacle == "bundle_too_large":
        monkeypatch.setattr(managed, "package_script", lambda *_: {"files": {"math_agent.py": "x" * 70000}})
    elif obstacle == "unexpected_error":
        monkeypatch.setattr(managed, "package_script", lambda *_: (_ for _ in ()).throw(OSError("PRIVATE path")))
    elif obstacle == "computed_import":
        # What the real packager refuses.
        program.write_text("import importlib\nimportlib.import_module(variable)\n")
    elif obstacle == "no_entry_file":
        monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)
    elif obstacle == "not_python":
        notebook = program.with_suffix(".ipynb")
        notebook.write_text("{}")
        monkeypatch.setattr(sys.modules["__main__"], "__file__", str(notebook))
    elif obstacle == "no_agent":
        manifest["agents"] = []
    assert managed.program_bundle_manifest(engine, manifest) is None


# --- end to end: a compiled LangGraph program, the real packager ------------------


def _increment(state):
    return {"number": state["number"] + 1}


def increment(state):
    return {"number": state["number"] + 1}


def _compiled(node):
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(dict)
    # "increment" is the node; its function is named alike, or apart.
    graph.add_node("increment", increment if node == "named_alike" else _increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile()


def _gateway(capabilities, discoveries, submissions):
    def handler(request):
        path = request.url.path
        if path.endswith("/runtime-config"):
            return httpx.Response(200, json={"security_enabled": True, "available": True, "plan": "business", "reason": "enabled"})
        if path == _CAPABILITIES:
            return httpx.Response(200, json=capabilities)
        if path.endswith("/vk-credential-info"):
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
        if path.endswith("/registry/discover"):
            discoveries.append(json.loads(request.content))
            # A field of the same name anywhere else is never read as the flag.
            return httpx.Response(200, json={"network_id": "math-network", "registration_state": "active",
                                             "blueprint_scan_status": "complete", "report_program_bundle": True})
        if path.endswith("/managed-execution/submissions"):
            submissions.append(json.loads(request.content))
            return httpx.Response(202, json={"job": {"id": "run-1", "status": "waiting_identity"}})
        if path.endswith("/agentic-new/executions"):
            return httpx.Response(200, json={})
        return httpx.Response(404)

    return handler


def _start(monkeypatch, capabilities, node):
    """One program start: compile under a fresh client; returns what reached the gateway."""
    from deepintshield.agentic import execution, registry
    from deepintshield.agentic.integrations import langgraph as integration

    def run_now(function, *args):
        function(*args)
        return True

    monkeypatch.setattr(integration, "run_detached", run_now)
    monkeypatch.setattr(execution, "_dispatch", run_now)
    discoveries, submissions = [], []
    registry.reset_dedupe()
    try:
        with DeepintShield(virtual_key="sk-ds-managed-test", base_url="https://gateway.example", agent_name="math") as client:
            client._client.close()
            client._client = httpx.Client(transport=httpx.MockTransport(_gateway(capabilities, discoveries, submissions)))
            engine = client.agentic.engine
            app = _compiled(node)
            if capabilities.get("enabled") is True:
                # The first execution hands the program to the server.
                with pytest.raises(GovernanceConfigurationError, match="managed_execution_identity_pending"):
                    integration._await_initial_discovery(engine, app)
            # What a local start reported before this change: the graph described,
            # that manifest described again by discovery, then attested.
            ordinary = registry._wire_attested_payload(engine, registry.describe_network(registry.describe_network(app)))
            return SimpleNamespace(engine=engine, discoveries=discoveries, submissions=submissions, ordinary=ordinary)
    finally:
        registry.reset_dedupe()


# Every part of a manifest the gateway derives the code fingerprint from.
_FINGERPRINTED = ("blueprint_digest", "nodes", "edges", "tools", "code_artifacts", "code_artifact_coverage", "self_attested_code_digest")


@pytest.mark.parametrize("node", ["named_alike", "named_apart"])
def test_the_next_start_reports_the_fingerprint_the_submission_had(program, monkeypatch, node):
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    first = _start(monkeypatch, {"enabled": True}, node)
    assert first.discoveries == [] and len(first.submissions) == 1
    submitted = first.submissions[0]["manifest"]
    assert submitted["blueprint_digest"] != first.ordinary["blueprint_digest"], "a submission binds the whole program"

    # The run ended for its identity; the next start runs here and is asked for the bundle.
    second = _start(monkeypatch, {"enabled": False, "report_program_bundle": True}, node)
    assert second.submissions == [] and len(second.discoveries) == 1
    assert second.engine._managed_execution_report_bundle is True
    reported = second.discoveries[0]
    assert reported["blueprint_digest"] == submitted["blueprint_digest"]
    assert {key: reported.get(key) for key in _FINGERPRINTED} == {key: submitted.get(key) for key in _FINGERPRINTED}
    assert reported["code_artifacts"][0]["source"].startswith("# file: math_agent.py\n")
    assert "sk-ds-managed-test" not in json.dumps(reported)


@pytest.mark.parametrize("node", ["named_alike", "named_apart"])
def test_a_plain_not_offered_start_keeps_the_ordinary_fingerprint(program, monkeypatch, node):
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("an ordinary local program was packaged"))
    start = _start(monkeypatch, {"enabled": False}, node)
    assert start.submissions == [] and len(start.discoveries) == 1
    reported = start.discoveries[0]
    assert {key: reported.get(key) for key in _FINGERPRINTED} == {key: start.ordinary.get(key) for key in _FINGERPRINTED}
    # The discovery answer carried a field of the same name: it is not the flag.
    assert start.engine._managed_execution_report_bundle is False


@pytest.mark.parametrize("node", ["named_alike", "named_apart"])
def test_a_flagged_start_that_cannot_be_packaged_keeps_the_ordinary_fingerprint(program, monkeypatch, node):
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    monkeypatch.setattr(managed, "package_script", _refuse("managed_execution_dependency_missing"))
    start = _start(monkeypatch, {"enabled": False, "report_program_bundle": True}, node)
    assert start.submissions == [] and len(start.discoveries) == 1
    reported = start.discoveries[0]
    assert {key: reported.get(key) for key in _FINGERPRINTED} == {key: start.ordinary.get(key) for key in _FINGERPRINTED}
