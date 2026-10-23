from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

from deepintshield import DeepintShield, __version__
from deepintshield.agentic import managed_execution as managed
from deepintshield.agentic import managed_runner
from deepintshield.agentic.errors import GovernanceConfigurationError, public_agentic_error_boundary


def engine_for(handler):
    return SimpleNamespace(
        gateway_url="https://gateway.example", virtual_key="sk-ds-managed-test",
        _http=httpx.Client(transport=httpx.MockTransport(handler)),
        _headers=lambda **_: {"Authorization": "Bearer sk-ds-managed-test", "X-Agent-Subject": "agent:math"},
    )


@pytest.fixture
def invocation_handoff(tmp_path, monkeypatch):
    from deepintshield.agentic import registry

    script = tmp_path / "entry.py"
    script.write_text("raise AssertionError('must not execute locally')\n")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script))
    monkeypatch.setattr(managed, "package_script", lambda *_: {"entrypoint": "entry.py", "files": {"entry.py": script.read_text()}, "requirements": []})
    monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": [{"key": "math"}]})
    monkeypatch.setattr(registry, "_wire_attested_payload", lambda _, payload: payload)
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(managed, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    engines = []

    def create(handler, **settings):
        engine = engine_for(handler)
        engine._managed_execution_capabilities = {
            "enabled": True, "submission_ids": True, "result_polling": True,
            "poll_interval_seconds": 1, "wait_timeout_seconds": 30, **settings,
        }
        engines.append(engine)
        return engine

    yield create, script, clock
    for engine in engines:
        engine._http.close()


def test_identified_handoff_polls_exact_invocation_and_prints_result(invocation_handoff, capsys):
    create, _, clock = invocation_handoff
    calls = []
    jobs = iter([
        {"id": "run-1", "status": "queued"},
        {"id": "run-1", "status": "running"},
        {"id": "run-1", "status": "succeeded", "result": "{'number': 21}\n"},
    ])

    def handler(request):
        calls.append(request.content)
        assert request.method == "POST" and request.url.path.endswith("/submissions")
        return httpx.Response(202, json={"job": next(jobs)})

    engine = create(handler)
    with pytest.raises(SystemExit) as stopped:
        managed.handoff_if_enabled(engine, object())
    assert stopped.value.code == 0
    assert len(calls) == 3 and len(set(calls)) == 1
    assert managed.uuid.UUID(json.loads(calls[0])["submission_id"]).version == 4
    assert clock.sleeps == [1, 1]
    assert capsys.readouterr().out == "{'number': 21}\n"


def test_fresh_handoffs_have_distinct_invocations_for_identical_program(invocation_handoff):
    create, _, clock = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": f"run-{len(bodies)}", "status": "waiting_approval"}})

    for _ in range(2):
        with pytest.raises(GovernanceConfigurationError, match="agent_registration_pending"):
            managed.handoff_if_enabled(create(handler), object())
    assert bodies[0].pop("submission_id") != bodies[1].pop("submission_id")
    assert bodies[0] == bodies[1]
    assert clock.sleeps == []


def test_lost_submission_response_retries_same_id_and_source(invocation_handoff, capsys):
    create, script, _ = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            raise httpx.ReadTimeout("PRIVATE response was lost", request=request)
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    engine = create(handler)
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_submission_failed"):
        managed.handoff_if_enabled(engine, object())
    script.write_text("print('different source must not be submitted on retry')\n")
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert capsys.readouterr().out == "21\n"


def test_cached_operator_hold_refreshes_same_invocation_after_approval(invocation_handoff, capsys):
    create, _, clock = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(request.content)
        job = {"id": "run-1", "status": "waiting_approval"} if len(bodies) == 1 else {"id": "run-1", "status": "succeeded", "result": "21\n"}
        return httpx.Response(202, json={"job": job})

    engine = create(handler)
    with pytest.raises(GovernanceConfigurationError, match="agent_registration_pending"):
        managed.handoff_if_enabled(engine, object())
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert clock.sleeps == []
    assert capsys.readouterr().out == "21\n"


@pytest.mark.parametrize(("identified", "polling"), [(False, False), (False, True), (True, False)])
def test_gateway_capabilities_gate_invocation_ids_and_polling(invocation_handoff, identified, polling):
    create, _, clock = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "queued"}})

    engine = create(handler, submission_ids=identified, result_polling=polling)
    for _ in range(2):
        with pytest.raises(GovernanceConfigurationError, match="managed_execution_pending"):
            managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 1
    assert ("submission_id" in bodies[0]) is identified
    assert clock.sleeps == []


@pytest.mark.parametrize(("status", "code"), [
    ("waiting_identity", "managed_execution_identity_pending"),
    ("waiting_runtime", "managed_execution_runtime_pending"),
    ("waiting_approval", "agent_registration_pending"),
    ("failed", "managed_execution_failed"),
    ("cancelled", "managed_execution_cancelled"),
    ("reconcile_required", "managed_execution_reconcile_required"),
])
def test_identified_handoff_stops_immediately_on_operator_hold_or_failure(invocation_handoff, status, code):
    create, _, clock = invocation_handoff
    calls = []

    def handler(request):
        calls.append(request.content)
        return httpx.Response(202, json={"job": {"id": "run-1", "status": status}})

    with pytest.raises(GovernanceConfigurationError, match=code):
        managed.handoff_if_enabled(create(handler), object())
    assert len(calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(("timeout", "expected_time"), [(0, 0), (2.5, 2.5), (1000, 60)])
def test_identified_polling_deadline_is_bounded_and_keeps_invocation(invocation_handoff, timeout, expected_time):
    create, _, clock = invocation_handoff
    bodies, request_timeouts = [], []

    def handler(request):
        bodies.append(request.content)
        request_timeouts.append(request.extensions["timeout"]["read"])
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "running"}})

    engine = create(handler, wait_timeout_seconds=timeout)
    for _ in range(2):
        with pytest.raises(GovernanceConfigurationError, match="managed_execution_pending"):
            managed.handoff_if_enabled(engine, object())
    assert clock.now == 2 * expected_time
    assert len(set(bodies)) == 1
    assert all(0 < value <= 30 for value in request_timeouts)
    assert engine._managed_execution_submission["job"]["id"] == "run-1"


def test_identified_polling_rejects_changed_job_without_printing_output(invocation_handoff, capsys):
    create, _, _ = invocation_handoff
    jobs = iter([
        {"id": "run-1", "status": "queued"},
        {"id": "different-run", "status": "succeeded", "result": "unrelated output"},
    ])
    engine = create(lambda _: httpx.Response(202, json={"job": next(jobs)}))
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_response_invalid"):
        managed.handoff_if_enabled(engine, object())
    assert engine._managed_execution_submission["job"]["id"] == "run-1"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("settings", [
    {"wait_timeout_seconds": True}, {"wait_timeout_seconds": -1},
    {"wait_timeout_seconds": float("inf")}, {"wait_timeout_seconds": "30"},
    {"poll_interval_seconds": float("nan")}, {"poll_interval_seconds": -1},
])
def test_invalid_polling_settings_fail_without_waiting(invocation_handoff, settings):
    create, _, clock = invocation_handoff
    engine = create(lambda _: httpx.Response(202, json={"job": {"id": "run-1", "status": "queued"}}), **settings)
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_response_invalid"):
        managed.handoff_if_enabled(engine, object())
    assert clock.sleeps == []


@pytest.mark.parametrize("response", [httpx.Response(404), httpx.Response(200, json={"enabled": False})])
def test_unconfigured_gateway_does_not_package_or_submit(response, monkeypatch):
    calls = []
    engine = engine_for(lambda request: calls.append(request.url.path) or response)
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("disabled mode packaged source"))
    managed.handoff_if_enabled(engine, object())
    managed.handoff_if_enabled(engine, object())
    # Asked before the first invocation only: from then on the program runs here.
    assert calls == ["/api/agentic-new/managed-execution/capabilities"]
    assert engine._managed_execution_local is True


@pytest.mark.parametrize("local_fallback", [True, False])
@pytest.mark.parametrize("refusal", [(404, None), (200, {"enabled": False})])
def test_program_that_ran_locally_is_not_handed_over_once_server_runs_are_offered(refusal, local_fallback, monkeypatch):
    probes = []

    def handler(request):
        if not request.url.path.endswith("/capabilities"):
            pytest.fail("a program that already ran here was submitted")
        probes.append(request.url.path)
        # Only the first answer is a no: one lookup failed, or the setting changed while the program ran.
        if len(probes) == 1:
            return httpx.Response(refusal[0], json=refusal[1])
        return httpx.Response(200, json={"enabled": True, "local_fallback": local_fallback, "submission_ids": True, "result_polling": True})

    engine = engine_for(handler)
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a program that already ran here was packaged"))
    try:
        assert managed.handoff_if_enabled(engine, object()) is None
        # The server runs a script from its first line, so it would repeat the invocation that ran here.
        assert managed.handoff_if_enabled(engine, object()) is None
        assert len(probes) == 1
        assert engine._managed_execution_local is True
        assert _nothing_was_handed_to_the_server(engine)
    finally:
        engine._http.close()


def test_concurrent_first_invocations_agree_where_server_runs_are_not_offered(monkeypatch):
    contenders = 6
    together, probes = threading.Barrier(contenders), []

    def handler(request):
        if not request.url.path.endswith("/capabilities"):
            pytest.fail("a program that runs here was submitted")
        probes.append(request.url.path)
        if len(probes) > 1:
            return httpx.Response(200, json={"enabled": True})
        # Long enough for the other invocations to be waiting for this answer.
        threading.Event().wait(0.3)
        return httpx.Response(200, json={"enabled": False})

    engine = engine_for(handler)
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a program that runs here was packaged"))
    results = []

    def invoke():
        together.wait(10)
        results.append(managed.handoff_if_enabled(engine, object()))

    threads = [threading.Thread(target=invoke) for _ in range(contenders)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        # One answer for all of them: none asks again and is told otherwise.
        assert results == [None] * contenders
        assert len(probes) == 1
        assert _nothing_was_handed_to_the_server(engine)
    finally:
        engine._http.close()


def test_engine_that_is_never_asked_is_left_as_it_was(monkeypatch):
    # Without a gateway nothing is asked, and nothing is written to an engine the SDK does not own.
    assert managed.handoff_if_enabled(object(), object()) is None
    engine = engine_for(lambda _: pytest.fail("a worker asked for a server run"))
    monkeypatch.setattr(managed, "_WORKER_FILES", {})
    try:
        assert managed.handoff_if_enabled(engine, object()) is None
        assert not hasattr(engine, "_managed_execution_local") and not hasattr(engine, "_managed_execution_capabilities")
    finally:
        engine._http.close()


def test_program_that_runs_locally_does_not_wait_for_a_hand_off_in_progress():
    engine = engine_for(lambda _: httpx.Response(200, json={"enabled": False}))
    holding, over, results = threading.Event(), threading.Event(), []

    def hand_off():
        # Another engine's hand-off keeps this lock for as long as it follows its job.
        with managed._LOCK:
            holding.set()
            over.wait(30)

    other = threading.Thread(target=hand_off)
    later = threading.Thread(target=lambda: results.append(managed.handoff_if_enabled(engine, object())))
    try:
        assert managed.handoff_if_enabled(engine, object()) is None
        other.start()
        assert holding.wait(10)
        later.start()
        later.join(2)
        assert results == [None]
    finally:
        over.set()
        for thread in (other, later):
            if thread.is_alive():
                thread.join(10)
        engine._http.close()


@pytest.mark.parametrize("code", ["agent_registration_denied", "agent_quarantined", "agent_retired"])
def test_blocked_identity_never_submits(code):
    calls = []
    engine = engine_for(lambda request: calls.append(request.method) or httpx.Response(403, json={"error": {"code": code}}))
    with pytest.raises(GovernanceConfigurationError, match=code):
        managed.handoff_if_enabled(engine, object())
    assert calls == ["GET"]


def test_redirect_is_not_followed_or_used_for_upload():
    calls = []
    engine = engine_for(lambda request: calls.append(str(request.url)) or httpx.Response(302, headers={"Location": "https://other.example/upload"}))
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_response_invalid"):
        managed.handoff_if_enabled(engine, object())
    assert calls == ["https://gateway.example/api/agentic-new/managed-execution/capabilities"]


@pytest.mark.parametrize("context", ["requester", "user", "service_account", "acting_user", "task_scope", "task_environment"])
def test_delegated_context_is_rejected_before_packaging_or_upload(context, monkeypatch):
    from contextlib import nullcontext
    from deepintshield.agentic.identity import PrincipalBinding
    from deepintshield.agentic.task_context import task_scope

    calls = []
    engine = engine_for(lambda request: calls.append(request.method) or httpx.Response(200, json={"enabled": True}))
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("delegated source was packaged"))
    if context == "requester":
        engine._requester_hint = "  alice@example.com  "
    elif context in ("user", "service_account"):
        engine.principal_binding = PrincipalBinding(subject="user:alice", kind=context)
    elif context == "acting_user":
        engine.acting_user = "user:alice"
    elif context == "task_environment":
        monkeypatch.setenv("DEEPINTSHIELD_TASK_MANDATE", "private-test-mandate")
    with task_scope("private-test-mandate") if context == "task_scope" else nullcontext():
        with pytest.raises(GovernanceConfigurationError, match="managed_execution_context_unsupported"):
            managed.handoff_if_enabled(engine, object())
    assert calls == ["GET"]


def test_whitespace_requester_is_autonomous_and_disabled_mode_preserves_delegation(monkeypatch):
    engine = engine_for(lambda _: httpx.Response(200, json={"enabled": False}))
    engine._requester_hint = " \t "
    managed._require_autonomous_context(engine)
    engine._requester_hint = "alice@example.com"
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("disabled mode packaged source"))
    managed.handoff_if_enabled(engine, object())


def test_disabled_security_preserves_native_execution_without_capability_probe(monkeypatch):
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    from langgraph.graph import StateGraph, START, END
    from deepintshield.agentic.integrations import langgraph as integration

    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.url.path.endswith("/runtime-config")
        return httpx.Response(200, json={"security_enabled": False, "available": True, "plan": "business", "reason": "workspace_disabled"})

    def run_now(function, *args):
        function(*args)
        return True

    monkeypatch.setattr(integration, "run_detached", run_now)
    with DeepintShield(virtual_key="sk-ds-managed-test", base_url="https://gateway.example", agent_name="math") as client:
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        graph = StateGraph(dict)
        graph.add_node("increment", lambda state: {"number": state["number"] + 1})
        graph.add_edge(START, "increment")
        graph.add_edge("increment", END)
        assert graph.compile().invoke({"number": 1}) == {"number": 2}
    assert calls
    assert set(calls) == {"/api/agentic-new/runtime-config"}


@pytest.mark.parametrize(("status", "reason", "expected"), [
    ("queued", "", "managed_execution_pending"),
    ("preparing", "", "managed_execution_pending"),
    ("running", "", "managed_execution_pending"),
    ("waiting_identity", "", "managed_execution_identity_pending"),
    # An older gateway keeps such a run waiting instead of ending it.
    ("waiting_identity", "managed_gateway_runtime_required", "managed_execution_identity_pending"),
    ("waiting_runtime", "", "managed_execution_runtime_pending"),
    ("waiting_runtime", "managed_dependency_unavailable", "managed_dependency_unavailable"),
    ("waiting_runtime", "managed_python_version_unavailable", "managed_python_version_unavailable"),
    ("waiting_runtime", "agent_runtime_authentication_failed", "managed_execution_runtime_pending"),
    ("waiting_runtime", "PRIVATE credential https://secret.example", "managed_execution_runtime_pending"),
    ("waiting_approval", "", "agent_registration_pending"),
    ("waiting_approval", "agent_access_setup_pending", "managed_execution_access_pending"),
    ("waiting_approval", "agent_access_plan_pending", "managed_execution_access_pending"),
    ("waiting_approval", "agent_blueprint_review_pending", "agent_blueprint_review_pending"),
    ("waiting_approval", "managed_blueprint_changed", "managed_blueprint_changed"),
    ("waiting_approval", "agent_tool_identity_review_required", "agent_tool_identity_review_required"),
    ("failed", "agent_runtime_authentication_failed", "agent_runtime_authentication_failed"),
    ("failed", "managed_dependency_unavailable", "managed_dependency_unavailable"),
    ("failed", "managed_python_version_unavailable", "managed_python_version_unavailable"),
    ("failed", "private_future_failure", "managed_execution_failed"),
    # Without the gateway's local fall-back the ended run is reported as failed.
    # Catalogued since 3.2.0: a listed workspace now names the reason instead of the generic failure.
    ("failed", "managed_gateway_runtime_required", "managed_gateway_runtime_required"),
    ("failed", "PRIVATE credential https://secret.example", "managed_execution_failed"),
    ("cancelled", "execution_cancelled", "managed_execution_cancelled"),
    ("expired", "execution_expired", "managed_execution_expired"),
    ("reconcile_required", "managed_execution_uncertain", "managed_execution_reconcile_required"),
    ("unknown", "", "managed_execution_response_invalid"),
])
def test_saved_handoff_reports_safe_state_and_never_submits_twice(tmp_path, monkeypatch, capsys, status, reason, expected):
    from deepintshield.agentic import registry

    calls = []
    response_body = {
        "job": {"id": "private-job-1", "status": status, "error_code": reason},
        "review_path": "/workspace/private-review?registration=private-registration",
        "message": "PRIVATE server prose",
    }

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, json={"enabled": True})
        assert request.method == "POST" and request.url.path.endswith("/submissions")
        return httpx.Response(202, json=response_body)

    engine = engine_for(handler)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "entry.py"))
    monkeypatch.setattr(managed, "package_script", lambda *_: {"entrypoint": "entry.py", "files": {"entry.py": "raise AssertionError('must not execute locally')\n"}, "requirements": []})
    monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": [{"key": "math"}]})
    monkeypatch.setattr(registry, "_wire_attested_payload", lambda _, payload: payload)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                with public_agentic_error_boundary():
                    managed.handoff_if_enabled(engine, object())
            assert caught.value._deepintshield_error_code == expected
            if expected == "agent_registration_pending":
                assert str(caught.value) == "agent_registration_pending: Registration pending. Contact your administrator."
            else:
                assert "Please contact your administrator." in str(caught.value)
            assert "PRIVATE" not in str(caught.value)
            assert "private-" not in str(caught.value)
            assert "https://" not in str(caught.value)
            assert caught.value.__context__ is None
        assert engine._managed_execution_submission == response_body
        assert calls == [("GET", "/api/agentic-new/managed-execution/capabilities"), ("POST", "/api/agentic-new/managed-execution/submissions")]
        assert capsys.readouterr().out == ""
    finally:
        engine._http.close()


@pytest.mark.parametrize("output", ["{'number': 21}\n", "Result: 21", "first\nsecond\n", "résultat: 结果\n", ""])
def test_succeeded_handoff_prints_exact_script_stdout_without_execution(capsys, output):
    engine = engine_for(lambda _: pytest.fail("completed cached job contacted gateway"))
    engine._managed_execution_capabilities = {"enabled": True}
    engine._managed_execution_submission = {"job": {"id": "private-job", "status": "succeeded", "result": output}, "review_path": "/workspace/private-review"}
    try:
        with pytest.raises(SystemExit) as stopped:
            managed.handoff_if_enabled(engine, object())
        assert stopped.value.code == 0
        assert engine._managed_execution_submission["job"]["result"] == output
        assert capsys.readouterr().out == output
    finally:
        engine._http.close()


@pytest.mark.parametrize("output", [{"number": 21}, [21], 21, True, "x" * (64 * 1024 + 1), "\ud800"])
def test_succeeded_handoff_rejects_malformed_or_oversize_stdout(capsys, output):
    engine = engine_for(lambda _: pytest.fail("completed cached job contacted gateway"))
    engine._managed_execution_capabilities = {"enabled": True}
    engine._managed_execution_submission = {"job": {"id": "private-job", "status": "succeeded", "result": output}}
    try:
        with pytest.raises(GovernanceConfigurationError, match="managed_execution_response_invalid"):
            managed.handoff_if_enabled(engine, object())
        assert capsys.readouterr().out == ""
    finally:
        engine._http.close()


@pytest.mark.parametrize("job", [{"status": "succeeded"}, {"status": "succeeded", "result": None}])
def test_completed_job_without_retained_output_reports_unavailable(capsys, job):
    engine = engine_for(lambda _: pytest.fail("completed cached job contacted gateway"))
    engine._managed_execution_capabilities = {"enabled": True}
    engine._managed_execution_submission = {"job": {"id": "private-job", **job}}
    try:
        with pytest.raises(RuntimeError, match="managed_execution_output_unavailable: The saved execution output is unavailable."):
            with public_agentic_error_boundary():
                managed.handoff_if_enabled(engine, object())
        assert capsys.readouterr().out == ""
    finally:
        engine._http.close()


def test_succeeded_handoff_scrubs_credentials_and_terminal_controls(capsys):
    engine = engine_for(lambda _: pytest.fail("completed cached job contacted gateway"))
    engine._managed_execution_capabilities = {"enabled": True}
    output = (
        "21\n" + engine.virtual_key + "\nBearer secret-token\nvirtual_key:reporter-id\n"
        "-----BEGIN PRIVATE KEY-----\nprivate-key\n-----END PRIVATE KEY-----\n"
        "\x1b[31mvalue\x1b[0m\x00\x7f\x85\n\x1b]52;c;clipboard-payload\x07end\t21"
    )
    engine._managed_execution_submission = {"job": {"id": "private-job", "status": "succeeded", "result": output}}
    try:
        with pytest.raises(SystemExit):
            managed.handoff_if_enabled(engine, object())
        assert capsys.readouterr().out == "21\n[REDACTED]\n[REDACTED]\n[REDACTED]\n[REDACTED]\nvalue\nend\t21"
    finally:
        engine._http.close()


def test_completed_submission_returns_full_bounded_stdout_without_local_continuation(tmp_path, monkeypatch, capsys):
    from deepintshield.agentic import registry

    output = "x" * (64 * 1024 - 2)
    result = {"job": {"id": "completed-job", "status": "succeeded", "result": output}, "review_path": "/workspace/private-review"}
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, json={"enabled": True})
        assert request.method == "POST" and request.url.path.endswith("/submissions")
        response = httpx.Response(202, json=result)
        assert len(response.content) > 65536
        return response

    engine = engine_for(handler)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "entry.py"))
    monkeypatch.setattr(managed, "package_script", lambda *_: {"entrypoint": "entry.py", "files": {"entry.py": "raise AssertionError('must not execute locally')\n"}, "requirements": []})
    monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": [{"key": "math"}]})
    monkeypatch.setattr(registry, "_wire_attested_payload", lambda _, payload: payload)
    try:
        with pytest.raises(SystemExit) as stopped:
            managed.handoff_if_enabled(engine, object())
        assert stopped.value.code == 0
        assert capsys.readouterr().out == output
        assert calls == ["GET", "POST"]
        assert engine._managed_execution_submission == result
    finally:
        engine._http.close()


def test_completed_script_stdout_printed_by_cli_and_script_does_not_continue(tmp_path):
    script = tmp_path / "completed_fixture.py"
    script.write_text(
        "from types import SimpleNamespace\n"
        "from deepintshield.agentic.managed_execution import handoff_if_enabled\n"
        "engine = SimpleNamespace(gateway_url='https://gateway.example', _managed_execution_capabilities={'enabled':True}, "
        "_managed_execution_submission={'job':{'id':'saved-job','status':'succeeded','result':\"{'number': 21}\\n\"}})\n"
        "handoff_if_enabled(engine, object())\n"
        "raise AssertionError('must not execute locally')\n"
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert result.stdout == "{'number': 21}\n"
    assert result.stderr == ""


def test_pending_handoff_exits_nonzero_with_public_error_and_no_review_url(tmp_path):
    script = tmp_path / "pending_fixture.py"
    script.write_text(
        "from types import SimpleNamespace\n"
        "from deepintshield.agentic.managed_execution import handoff_if_enabled\n"
        "from deepintshield.agentic.errors import public_agentic_error_boundary\n"
        "engine = SimpleNamespace(gateway_url='https://private-gateway.example', _managed_execution_capabilities={'enabled':True}, _managed_execution_submission={'job':{'id':'private-job','status':'waiting_approval'},'review_path':'/workspace/private-review'})\n"
        "with public_agentic_error_boundary():\n"
        "    handoff_if_enabled(engine, object())\n"
        "print('must-not-continue')\n"
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "RuntimeError: agent_registration_pending: Registration pending. Contact your administrator." in result.stderr
    assert "private-gateway" not in result.stderr
    assert "private-job" not in result.stderr
    assert "private-review" not in result.stderr


def test_package_contains_only_reachable_python_files_and_pins(tmp_path, monkeypatch):
    (tmp_path / "entry.py").write_text("import os\nfrom package import helper\nimport external_demo\n")
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "__init__.py").write_text("")
    (tmp_path / "package" / "helper.py").write_text("from . import nested\n")
    (tmp_path / "package" / "nested.py").write_text("VALUE=3\n")
    (tmp_path / "private.env").write_text("secret-not-to-upload")
    (tmp_path / "unrelated.py").write_text("private='not-imported'\n")
    monkeypatch.setattr(managed.importlib.metadata, "packages_distributions", lambda: {"external_demo": ["external-demo"]})
    monkeypatch.setattr(managed.importlib.metadata, "version", lambda _: "1.2.3")
    bundle = managed.package_script(tmp_path / "entry.py", {})
    assert list(bundle["files"]) == ["entry.py", "package/__init__.py", "package/helper.py", "package/nested.py"]
    assert bundle["requirements"] == ["external-demo==1.2.3"]
    assert "secret-not-to-upload" not in json.dumps(bundle)


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python 3.11+ infers wheel import mappings natively")
@pytest.mark.parametrize("partially_declared", [False, True])
def test_wheel_records_supply_missing_import_mapping_without_importing_code(tmp_path, monkeypatch, partially_declared):
    def distribution(name, files):
        return SimpleNamespace(
            metadata={"Name": name},
            read_text=lambda _: None,
            files=[managed.importlib.metadata.PackagePath(file) for file in files],
        )

    # Python 3.10 supplies no mappings for modern wheels without top_level.txt.
    monkeypatch.setattr(
        managed.importlib.metadata, "packages_distributions",
        lambda: {"shared_demo": ["demo-core"]} if partially_declared else {},
    )
    monkeypatch.setattr(managed.importlib.metadata, "distributions", lambda: [
        distribution("demo-core", ["shared_demo/core.py"]),
        distribution("demo-extension", ["shared_demo/extensions/__init__.py"]),
        distribution("native-demo", ["native_demo" + managed.importlib.machinery.EXTENSION_SUFFIXES[0]]),
        distribution("invalid-records", [
            "../../private.py", "/private/secret.py", "invalid-name/module.py",
            "metadata.dist-info/metadata.py", "type_only.pyi",
        ]),
        distribution("invalid name", ["invalid_metadata/module.py"]),
        distribution("missing-records", []),
    ])
    monkeypatch.setattr(managed.importlib.metadata, "version", lambda _: "1.2.3")
    script = tmp_path / "entry.py"
    script.write_text("import shared_demo\nimport native_demo\nraise AssertionError('must not execute')\n")
    bundle = managed.package_script(script, {})
    assert bundle["requirements"] == [
        "demo-core==1.2.3", "demo-extension==1.2.3", "native-demo==1.2.3",
    ]
    assert set(managed._packages_distributions()) == {"shared_demo", "native_demo"}
    assert "shared_demo" not in sys.modules
    assert "native_demo" not in sys.modules


@pytest.mark.parametrize("source", ["import importlib\nimportlib.import_module(variable)\n", "from importlib import import_module as load\nload(variable)\n", "exec('print(1)')\n", "from ..escape import value\n"])
def test_unsupported_code_loaders_fail_before_submission(tmp_path, source):
    script = tmp_path / "entry.py"
    script.write_text(source)
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_bundle_unsupported"):
        managed.package_script(script, {})


def test_package_rejects_symlinks_and_bounds(tmp_path):
    script = tmp_path / "entry.py"
    script.write_text("import linked\n")
    original = tmp_path / "original.py"
    original.write_text("VALUE=1\n")
    (tmp_path / "linked.py").symlink_to(original)
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_bundle_unsupported"):
        managed.package_script(script, {})
    script.write_text("value = '" + "a" * 300 + "'\n")
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_bundle_too_large"):
        managed.package_script(script, {"max_bundle_bytes": 100})
    script.write_text("import original\n")
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_bundle_too_large"):
        managed.package_script(script, {"max_files": 1})


def test_missing_import_fails_with_actionable_code(tmp_path, monkeypatch):
    script = tmp_path / "entry.py"
    script.write_text("import absent_dependency\n")
    monkeypatch.setattr(managed.importlib.metadata, "packages_distributions", lambda: {})
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_dependency_missing"):
        managed.package_script(script, {})


def test_repository_folder_does_not_hide_installed_sdk_dependency(tmp_path):
    script = tmp_path / "entry.py"
    script.write_text("from deepintshield import DeepintShield\n")
    (tmp_path / "deepintshield").mkdir()
    (tmp_path / "deepintshield" / "private.py").write_text("not_selected=True\n")
    bundle = managed.package_script(script, {})
    assert list(bundle["files"]) == ["entry.py"]
    assert f"deepintshield=={__version__}" in bundle["requirements"]


def test_full_bundle_artifact_is_same_on_submit_and_worker(monkeypatch):
    from deepintshield.agentic.registry import _wire_attested_payload

    engine = engine_for(lambda _: httpx.Response(404))
    manifest = {"agents": [{"key": "math"}], "network": {"digest": "sha256:topology"}, "code_artifacts": [], "code_artifact_coverage": []}
    files = {"entry.py": "key='sk-ds-managed-test'\nprint(21)\n"}
    submitted = _wire_attested_payload(engine, managed.augment_manifest(engine, manifest, files))
    changed = _wire_attested_payload(engine, managed.augment_manifest(engine, manifest, {"entry.py": "print(22)\n"}))
    assert changed["blueprint_digest"] != submitted["blueprint_digest"]
    monkeypatch.setattr(managed, "_WORKER_FILES", files)
    worker = _wire_attested_payload(engine, manifest)
    assert submitted == worker
    assert "sk-ds-managed-test" not in json.dumps(submitted)
    assert "print(21)" in worker["code_artifacts"][0]["source"]


@pytest.mark.parametrize("mode", ["invoke", "stream", "ainvoke", "astream", "batch", "abatch"])
def test_standalone_math_handoffs_before_any_node_runs(tmp_path, monkeypatch, capsys, mode):
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    safe = {"DEEPINTSHIELD_AGENT_NAME": "math", "DEEPINTSHIELD_BASE_URL": "https://gateway.example", "DEEPINTSHIELD_VIRTUAL_KEY": "sk-ds-managed-test", "DEEPINTSHIELD_REQUESTER": ""}
    script = tmp_path / "entry.py"
    # Keep this fixture independent of developer scripts and their credentials.
    source = (
        "from typing import TypedDict\n"
        "from deepintshield import DeepintShield\n"
        "from langgraph.graph import StateGraph, START, END\n"
        "class State(TypedDict):\n"
        "    number: int\n"
        "def add_two(state: State):\n"
        "    return {'number': state['number'] + 2}\n"
        "def multiply_by_three(state: State):\n"
        "    return {'number': state['number'] * 3}\n"
        "with DeepintShield.from_env() as shield:\n"
        "    builder = StateGraph(State)\n"
        "    builder.add_node('add_two', add_two)\n"
        "    builder.add_node('multiply_by_three', multiply_by_three)\n"
        "    builder.add_edge(START, 'add_two')\n"
        "    builder.add_edge('add_two', 'multiply_by_three')\n"
        "    builder.add_edge('multiply_by_three', END)\n"
        "    graph = builder.compile()\n"
        "    result = graph.invoke({'number': 5})\n"
        "    print(result)\n"
    )
    if mode == "stream":
        source = source.replace("result = graph.invoke({'number': 5})", "result = list(graph.stream({'number': 5}))")
    elif mode == "ainvoke":
        source = "import asyncio\n" + source.replace("result = graph.invoke({'number': 5})", "result = asyncio.run(graph.ainvoke({'number': 5}))")
    elif mode == "astream":
        source = "import asyncio\nasync def collect(graph):\n    return [chunk async for chunk in graph.astream({'number': 5})]\n" + source.replace("result = graph.invoke({'number': 5})", "result = asyncio.run(collect(graph))")
    elif mode == "batch":
        source = source.replace("result = graph.invoke({'number': 5})", "result = graph.batch([{'number': 5}])")
    elif mode == "abatch":
        source = "import asyncio\n" + source.replace("result = graph.invoke({'number': 5})", "result = asyncio.run(graph.abatch([{'number': 5}]))")
    script.write_text(source)
    calls, bodies, executed = [], [], []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/runtime-config"):
            return httpx.Response(404)
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, json={"enabled": True, "max_bundle_bytes": 2097152, "max_files": 64})
        if request.url.path.endswith("/submissions"):
            bodies.append(json.loads(request.content))
            return httpx.Response(202, json={"job": {"id": "job-1", "status": "waiting_approval"}, "review_path": "/workspace/agentic-new/overview?review_registration=reg-1"})
        pytest.fail(f"unexpected local governance request: {request.url.path}")

    clients = []

    def from_env(cls):
        client = DeepintShield(virtual_key=safe["DEEPINTSHIELD_VIRTUAL_KEY"], base_url=safe["DEEPINTSHIELD_BASE_URL"], agent_name="math", requester="")
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(DeepintShield, "from_env", classmethod(from_env))
    old_trace = sys.getprofile()

    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == str(script) and frame.f_code.co_name in {"add_two", "multiply_by_three"}:
            executed.append(frame.f_code.co_name)

    try:
        sys.setprofile(trace)
        with pytest.raises(RuntimeError) as caught:
            runpy.run_path(str(script), run_name="__main__")
        assert caught.value._deepintshield_error_code == "agent_registration_pending"
        assert "Registration pending. Contact your administrator." in str(caught.value)
    finally:
        sys.setprofile(old_trace)
        for client in clients:
            client.close()
    assert executed == []
    assert len(bodies) == 1
    assert bodies[0]["files"] == {"entry.py": script.read_text()}
    assert any("# file: entry.py" in artifact["source"] for artifact in bodies[0]["manifest"]["code_artifacts"])
    assert not any(path.endswith("/registry/discover") for path in calls)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("approval_budget", [0, 7.5, 240])
def test_worker_restores_operator_config_and_owned_key(tmp_path, monkeypatch, capsys, approval_budget):
    script = tmp_path / "entry.py"
    script.write_text("import os\nfrom deepintshield import DeepintShield\nos.environ['DEEPINTSHIELD_BASE_URL']='http://localhost:8080'\nos.environ['DEEPINTSHIELD_VIRTUAL_KEY']='source-key'\nDeepintShield.from_env()\nprint(21)\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"environment": {"DEEPINTSHIELD_BASE_URL": "https://worker-gateway.example", "DEEPINTSHIELD_VIRTUAL_KEY": "approved-key", "DEEPINTSHIELD_AGENT_NAME": "math"}, "runtime_private_key": "private-test-key", "requirements": [], "approval_wait_timeout_seconds": approval_budget}))
    observed = {}
    prior_deadline = managed._WORKER_APPROVAL_DEADLINE

    def from_env(cls):
        observed.update({key: value for key, value in os.environ.items() if key.startswith("DEEPINTSHIELD_")})
        key = Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_FILE"])
        assert key.read_text() == "private-test-key"
        assert key.stat().st_mode & 0o777 == 0o600
        assert managed.capabilities(object()) == {}
        remaining = managed._WORKER_APPROVAL_DEADLINE - managed.time.monotonic()
        assert approval_budget - 2 <= remaining <= approval_budget
        return object()

    monkeypatch.setattr(DeepintShield, "from_env", classmethod(from_env))
    assert managed_runner.main([str(script), "--config", str(config)]) == 0
    assert observed["DEEPINTSHIELD_BASE_URL"] == "https://worker-gateway.example"
    assert observed["DEEPINTSHIELD_VIRTUAL_KEY"] == "approved-key"
    assert not Path(observed["DEEPINTSHIELD_RUNTIME_KEY_FILE"]).exists()
    assert capsys.readouterr().out == "21\n"
    assert managed._WORKER_APPROVAL_DEADLINE == prior_deadline


def test_worker_missing_pinned_dependency_never_runs_script(tmp_path, capsys):
    script = tmp_path / "entry.py"
    script.write_text("raise RuntimeError('source-must-not-run')\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"environment": {}, "runtime_private_key": "private-test-key", "requirements": ["missing-deepintshield-demo==999.0"]}))
    assert managed_runner.main([str(script), "--config", str(config)]) == 1
    assert capsys.readouterr().err == "DEEPINTSHIELD_MANAGED_ERROR:managed_dependency_unavailable\n"


def test_worker_requires_installed_packages_not_matching_versions(monkeypatch):
    def installed(name):
        if name == "langgraph":
            return "1.2.6"
        raise managed_runner.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(managed_runner.importlib.metadata, "version", installed)
    requirements = ["DeepIntShield==2.8.3", "langgraph==1.1.6"]
    original = list(requirements)
    # Any client SDK or framework release runs on the worker's own releases.
    assert managed_runner._dependencies_available(requirements)
    assert requirements == original
    assert managed_runner._dependencies_available(["deepintshield==4.0.0"])
    assert not managed_runner._dependencies_available(["langgraph==1.2.6", "missing==1.0.0"])
    assert not managed_runner._dependencies_available(["langgraph>=1.2.6"])
    assert not managed_runner._dependencies_available(["langgraph==1.2.6"] * 65)


def test_worker_runs_submission_from_any_client_python_and_sdk_version(tmp_path, capsys):
    script = tmp_path / "entry.py"
    script.write_text("from deepintshield.version import __version__\nprint(__version__)\n")
    configuration = {
        "environment": {
            "DEEPINTSHIELD_BASE_URL": "https://gateway.example",
            "DEEPINTSHIELD_VIRTUAL_KEY": "fixture-key",
            "DEEPINTSHIELD_AGENT_NAME": "fixture-agent",
        },
        "runtime_private_key": "fixture-private-key",
        # An older SDK pin and a pytest release that is not installed as such:
        # only presence matters, and pytest is present in the test interpreter.
        "requirements": ["deepintshield==2.8.3", "pytest==0.0.1"],
        "python_version": "3.9",
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration))
    assert managed_runner.main([str(script), "--config", str(config)]) == 0
    output = capsys.readouterr()
    assert output.out == f"{__version__}\n"
    assert output.err == ""
    assert json.loads(config.read_text()) == configuration


def test_worker_rejects_malformed_python_version_as_configuration(tmp_path, capsys):
    script = tmp_path / "entry.py"
    script.write_text("raise RuntimeError('source-must-not-run')\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"environment": {}, "runtime_private_key": "private-test-key", "requirements": [], "python_version": "3"}))
    assert managed_runner.main([str(script), "--config", str(config)]) == 1
    assert capsys.readouterr().err == "DEEPINTSHIELD_MANAGED_ERROR:managed_execution_configuration_invalid\n"


def test_submit_then_worker_executes_math_over_real_loopback_http(tmp_path, monkeypatch, capsys):
    """Exercise the actual HTTP client, launcher, key proofs, and node execution.

    Only the reporting identity bootstrap is redirected for the submitting
    process. The worker uses the production from_env override and mounted key.
    No source or credentials are loaded from the workspace's real test2.py.
    """
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    from deepintshield.agentic import execution, registry
    from deepintshield.agentic.credentials import runtime_key
    from deepintshield.agentic.dpop import DPoPKey
    from deepintshield.agentic.integrations import langgraph as integration
    from tests.test_agentic_gateway_broker import Broker, PREFIX, config

    script = tmp_path / "entry.py"
    script.write_text(
        "import os\n"
        "os.environ['DEEPINTSHIELD_BASE_URL']='http://localhost:8080'\n"
        "os.environ['DEEPINTSHIELD_AGENT_NAME']='source-agent'\n"
        "os.environ['DEEPINTSHIELD_VIRTUAL_KEY']='source-vk'\n"
        "os.environ['DEEPINTSHIELD_REQUESTER']=''\n"
        "from typing import TypedDict\n"
        "from deepintshield import DeepintShield\n"
        "from langgraph.graph import StateGraph, START, END\n"
        "class State(TypedDict):\n"
        "    number: int\n"
        "def add_two(state: State):\n"
        "    return {'number': state['number'] + 2}\n"
        "def multiply_by_three(state: State):\n"
        "    return {'number': state['number'] * 3}\n"
        "with DeepintShield.from_env() as shield:\n"
        "    builder = StateGraph(State)\n"
        "    builder.add_node('add_two', add_two)\n"
        "    builder.add_node('multiply_by_three', multiply_by_three)\n"
        "    builder.add_edge(START, 'add_two')\n"
        "    builder.add_edge('add_two', 'multiply_by_three')\n"
        "    builder.add_edge('multiply_by_three', END)\n"
        "    graph = builder.compile()\n"
        "    result = graph.invoke({'number': 5})\n"
        "    print(result)\n"
    )
    script_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    broker = Broker(approved=True, broker=config(agent_subject="agent:math-agent"))
    phase = "submission"
    submitted, discoveries, decisions, calls, errors = [], [], [], [], []

    class Gateway(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def handle_request(self):
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request = httpx.Request(self.command, origin + self.path, headers=dict(self.headers), content=raw)
                calls.append((phase, self.path))
                assert request.headers["Authorization"] == "Bearer sk-ds-test"
                if self.path.endswith("/runtime-config"):
                    response = httpx.Response(200, json={"security_enabled": True, "available": True, "plan": "business", "reason": "enabled"})
                elif self.path.endswith("/capabilities"):
                    assert phase == "submission", "worker attempted another handoff"
                    response = httpx.Response(200, json={"enabled": True})
                elif self.path.endswith("/submissions"):
                    assert phase == "submission"
                    submitted.append(json.loads(raw))
                    response = httpx.Response(202, json={"job": {"id": "hermetic-job", "status": "waiting_approval"}, "review_path": "/workspace/agentic-new/overview?review_registration=hermetic-reg"})
                elif self.path == "/api/agentic-security/blueprints":
                    response = httpx.Response(404)
                elif self.path == "/api/agentic-new/credential-info" or self.path.startswith(PREFIX):
                    response = broker(request)
                else:
                    broker(request)  # Verify ES256 signature, nonce, ath, URI, VK, agent, and key continuity.
                    if self.path.endswith("/registry/discover"):
                        body = json.loads(raw)
                        discoveries.append(body)
                        assert body["blueprint_digest"] == submitted[0]["manifest"]["blueprint_digest"]
                        assert body["self_attested_code_digest"] == submitted[0]["manifest"]["self_attested_code_digest"]
                        assert body["code_artifacts"] == submitted[0]["manifest"]["code_artifacts"]
                        response = httpx.Response(200, json={"network_id": "math-network", "registration_state": "active", "blueprint_scan_status": "complete"})
                    elif self.path.endswith("/decide"):
                        decisions.append(json.loads(raw))
                        response = httpx.Response(200, json={"verdict": "ALLOW", "allow": True, "proceed": True, "mode": "enforce", "would_block": False})
                    elif self.path == "/api/agentic-new/executions":
                        response = httpx.Response(200, json={})
                    else:
                        raise AssertionError("unexpected test gateway route")
            except BaseException as error:
                errors.append(str(error))
                response = httpx.Response(500, json={"error": {"code": "hermetic_gateway_assertion_failed"}})
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    origin = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def run_now(function, *args):
        function(*args)
        return True

    monkeypatch.setattr(integration, "run_detached", run_now)
    monkeypatch.setattr(execution, "_dispatch", run_now)
    for name in ("DEEPINTSHIELD_BASE_URL", "DEEPINTSHIELD_AGENT_NAME", "DEEPINTSHIELD_VIRTUAL_KEY", "DEEPINTSHIELD_REQUESTER"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    try:
        with monkeypatch.context() as submitting:
            # Safe redirect of copied source's example settings; use real HTTP.
            submitting.setattr(DeepintShield, "from_env", classmethod(lambda cls: cls(base_url=origin, virtual_key="sk-ds-test", agent_name="math-agent", requester="")))
            with pytest.raises(RuntimeError) as stopped:
                runpy.run_path(str(script), run_name="__main__")
            assert stopped.value._deepintshield_error_code == "agent_registration_pending"
        assert len(submitted) == 1
        assert not decisions
        assert capsys.readouterr().out == ""

        phase = "worker"
        key = DPoPKey()
        worker_directory = tmp_path / "worker_bundle"
        worker_directory.mkdir()
        for name, content in submitted[0]["files"].items():
            target = worker_directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        worker_script = worker_directory / submitted[0]["entrypoint"]
        config_path = tmp_path / "worker.json"
        config_path.write_text(json.dumps({
            "environment": {"DEEPINTSHIELD_BASE_URL": origin, "DEEPINTSHIELD_VIRTUAL_KEY": "sk-ds-test", "DEEPINTSHIELD_AGENT_NAME": "math-agent", "DEEPINTSHIELD_REQUESTER": "", "DEEPINTSHIELD_RUNTIME_KIND": "key"},
            "runtime_private_key": runtime_key._private_bytes(key).decode("ascii"),
            "requirements": submitted[0]["requirements"],
            "python_version": submitted[0]["python_version"],
        }))
        config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        assert managed_runner.main(["--config", str(config_path), str(worker_script)]) == 0, errors
        output = capsys.readouterr()
        assert output.out == "{'number': 21}\n"
        assert "DEEPINTSHIELD_MANAGED_ERROR" not in output.err
        assert not errors
        assert discoveries
        assert [decision["tool"] for decision in decisions] == ["tool:add_two", "tool:multiply_by_three"]
        assert all(decision["agent"] == "agent:math-agent" for decision in decisions)
        assert broker.jkt == key.thumbprint
        assert sum(path.endswith("/submissions") for _, path in calls) == 1
        assert not any(stage == "worker" and path.endswith("/capabilities") for stage, path in calls)
        assert hashlib.sha256(script.read_bytes()).hexdigest() == script_digest
        assert hashlib.sha256(worker_script.read_bytes()).hexdigest() == script_digest
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_digest
    finally:
        registry.reset_dedupe()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("initial", [httpx.Response(404), httpx.Response(200, json={"enabled": False})])
def test_compile_opt_out_is_rechecked_before_first_invocation(initial, tmp_path, monkeypatch):
    from deepintshield.agentic import registry
    from deepintshield.agentic.integrations import langgraph as integration

    paths, local_discovery = [], []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/capabilities"):
            return initial if paths.count(request.url.path) == 1 else httpx.Response(200, json={"enabled": True})
        assert request.url.path.endswith("/submissions")
        return httpx.Response(202, json={"job": {"id": "saved-after-config-change", "status": "waiting_runtime"}})

    engine = engine_for(handler)
    engine.register_blueprint = lambda _: None
    monkeypatch.setattr(integration, "report_topology", lambda *_, **__: local_discovery.append("compile") or True)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "entry.py"))
    monkeypatch.setattr(managed, "package_script", lambda *_: {"entrypoint": "entry.py", "files": {"entry.py": "raise AssertionError('must not execute locally')\n"}, "requirements": []})
    monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": [{"key": "math"}]})
    monkeypatch.setattr(registry, "_wire_attested_payload", lambda _, payload: payload)
    barrier = integration._DiscoveryBarrier()
    app = SimpleNamespace()
    setattr(app, integration._DISCOVERY_BARRIER, barrier)
    integration._register_and_report(engine, app, barrier)
    assert barrier.succeeded
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_runtime_pending"):
        integration._await_initial_discovery(engine, app)
    assert paths.count("/api/agentic-new/managed-execution/capabilities") == 2
    assert paths.count("/api/agentic-new/managed-execution/submissions") == 1
    assert local_discovery == ["compile"]
    assert engine._managed_execution_submission["job"]["id"] == "saved-after-config-change"


@pytest.mark.parametrize("budget", [-1, 241, "60", True, None, float("nan"), float("inf")])
def test_worker_rejects_invalid_approval_deadline_before_source(tmp_path, monkeypatch, capsys, budget):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"environment": {}, "runtime_private_key": "private-test-key", "approval_wait_timeout_seconds": budget}))
    monkeypatch.setattr(managed_runner.runpy, "run_path", lambda *_a, **_kw: pytest.fail("invalid configuration executed source"))
    assert managed_runner.main([str(tmp_path / "entry.py"), "--config", str(config)]) == 1
    assert capsys.readouterr().err == "DEEPINTSHIELD_MANAGED_ERROR:managed_execution_configuration_invalid\n"


# --- local fall-back: the gateway offers server runs to every workspace ---------

_serves_requests = managed._serves_requests


@pytest.fixture(autouse=True)
def standalone_script(monkeypatch):
    """Tests run as a standalone script unless they say otherwise.

    The test process itself is started by pytest, which is installed code, and
    several tests call the handoff from worker threads on purpose.
    """
    monkeypatch.setattr(managed, "_serves_requests", lambda: False)


@pytest.fixture
def own_script(tmp_path, monkeypatch):
    """The process was started from the user's own file, as ``python agent.py`` is."""
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "agent.py"), raising=False)


def _called_from(module, function, *args):
    """Call ``function`` from a frame that belongs to ``module``, without importing it."""
    namespace = {"__name__": module}
    exec("def call(function, *args):\n    return function(*args)\n", namespace)
    return namespace["call"](function, *args)


def test_a_plain_script_is_not_taken_for_a_service(own_script):
    assert _serves_requests() is False
    # Nor when its graph is invoked some calls down in its own modules.
    assert _called_from("agent.graph", _called_from, "agent.nodes", _serves_requests) is False


def test_a_serving_framework_that_is_only_imported_does_not_mark_a_service(own_script, monkeypatch):
    # Importing mcp or google.genai loads starlette and uvicorn into a script that serves nothing.
    with monkeypatch.context() as imported:
        for name in (*managed._SERVING_MODULES, *managed._SERVING_PREFIXES):
            imported.setitem(sys.modules, name, object())
        assert _serves_requests() is False


@pytest.mark.parametrize("framework", sorted({*managed._SERVING_MODULES, *managed._SERVING_PREFIXES}))
def test_a_call_from_inside_a_serving_framework_marks_a_service(own_script, framework):
    assert _called_from(framework, _serves_requests) is True
    # The framework calls the program's handler, which invokes the graph further down.
    assert _called_from(framework + ".routing", _called_from, "agent.handlers", _serves_requests) is True


@pytest.mark.parametrize(("module", "serves"), [
    ("aiohttp.web_app", True),
    ("aiohttp.web_protocol", True),
    ("aiohttp.client", False),
    ("aiohttp", False),
    ("mcp.server.lowlevel.server", True),
    ("mcp.client.session", False),
    # A program's own module that is merely named after a framework.
    ("flask_app", False),
    ("project.celery", False),
])
def test_only_the_serving_part_of_a_package_marks_a_service(own_script, module, serves):
    assert _called_from(module, _serves_requests) is serves


def test_a_worker_thread_marks_a_service(own_script):
    seen = []
    thread = threading.Thread(target=lambda: seen.append(_serves_requests()))
    thread.start()
    thread.join(10)
    assert seen == [True]


@pytest.mark.parametrize(("entry", "installed"), [
    # ``python -m pytest`` and ``python -m uvicorn`` run someone else's file.
    ("venv/lib/python3.12/site-packages/pytest/__main__.py", True),
    ("usr/lib/python3/dist-packages/uvicorn/__main__.py", True),
    # The program's own file, in a directory that is only named like one.
    ("my-site-packages/agent.py", False),
])
def test_only_an_entry_file_in_an_installed_package_marks_a_service(tmp_path, monkeypatch, entry, installed):
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / entry), raising=False)
    assert _serves_requests() is installed


def test_an_entry_file_in_the_standard_library_marks_a_service(monkeypatch):
    # ``python -m timeit``: the standard library is wherever its own modules say they are.
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(Path(os.__file__).with_name("timeit.py")), raising=False)
    assert _serves_requests() is True


@pytest.mark.parametrize("layout", [
    "library reported through a link",
    "entry file given through a link",
    "installed file linked from a cache",
    "site-packages reached through a link",
])
def test_an_installed_entry_file_is_recognized_through_a_link(tmp_path, monkeypatch, layout):
    import sysconfig

    library = tmp_path / "cellar" / "python3.12"
    library.mkdir(parents=True)
    (tmp_path / "opt").symlink_to(tmp_path / "cellar")
    linked = tmp_path / "opt" / "python3.12"
    reported, script = library, library / "timeit.py"
    if layout == "library reported through a link":
        # Homebrew names its library by a stable link to the versioned directory.
        reported = linked
    elif layout == "entry file given through a link":
        script = linked / "timeit.py"
    elif layout == "installed file linked from a cache":
        (tmp_path / "cache" / "pytest").mkdir(parents=True)
        (tmp_path / "site-packages").mkdir()
        (tmp_path / "site-packages" / "pytest").symlink_to(tmp_path / "cache" / "pytest")
        script = tmp_path / "site-packages" / "pytest" / "__main__.py"
    else:
        (tmp_path / "lib" / "site-packages" / "pytest").mkdir(parents=True)
        (tmp_path / "vendor").symlink_to(tmp_path / "lib" / "site-packages")
        script = tmp_path / "vendor" / "pytest" / "__main__.py"
    monkeypatch.setattr(sysconfig, "get_paths", lambda *_, **__: {"stdlib": str(reported)})
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script), raising=False)
    assert _serves_requests() is True
    # The program's own file beside the library is still its own.
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "cellar" / "python3.12-tools" / "agent.py"), raising=False)
    assert _serves_requests() is False


@pytest.mark.parametrize("entry", ["missing", None, ""])
def test_a_process_without_an_entry_file_is_not_taken_for_a_service(tmp_path, monkeypatch, entry):
    # An interactive session, ``python -c`` and a notebook have no entry file,
    # and the directory they were started in does not stand in for one.
    (tmp_path / "site-packages").mkdir()
    monkeypatch.chdir(tmp_path / "site-packages")
    main = sys.modules["__main__"]
    if entry == "missing":
        monkeypatch.delattr(main, "__file__", raising=False)
    else:
        monkeypatch.setattr(main, "__file__", entry, raising=False)
    assert _serves_requests() is False


def test_local_fallback_hands_over_a_script_that_only_imports_a_serving_framework(invocation_handoff, monkeypatch, capsys):
    create, _, _ = invocation_handoff
    monkeypatch.setattr(managed, "_serves_requests", _serves_requests)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    engine = create(handler, local_fallback=True)
    with monkeypatch.context() as imported, pytest.raises(SystemExit):
        for name in ("starlette", "uvicorn"):
            imported.setitem(sys.modules, name, object())
        managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 1 and bodies[0]["entrypoint"] == "entry.py"
    assert not hasattr(engine, "_managed_execution_local")
    assert capsys.readouterr().out == "21\n"


@pytest.mark.parametrize("sign", ["serving framework", "installed entry file", "worker thread"])
def test_local_fallback_recognizes_a_service_by_each_sign(invocation_handoff, monkeypatch, tmp_path, sign):
    create, _, _ = invocation_handoff
    monkeypatch.setattr(managed, "_serves_requests", _serves_requests)
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a service was packaged"))
    engine = create(lambda _: pytest.fail("a service was submitted"), local_fallback=True)
    results = []

    def invoke():
        results.append(managed.handoff_if_enabled(engine, object()))

    if sign == "serving framework":
        _called_from("uvicorn.protocols.http.h11_impl", invoke)
    elif sign == "installed entry file":
        monkeypatch.setattr(sys.modules["__main__"], "__file__", str(tmp_path / "site-packages" / "pytest" / "__main__.py"))
        invoke()
    else:
        thread = threading.Thread(target=invoke)
        thread.start()
        thread.join(10)
    assert results == [None]
    assert engine._managed_execution_local is True
    assert _nothing_was_handed_to_the_server(engine)


def test_local_fallback_keeps_a_service_local(invocation_handoff, monkeypatch):
    create, _, _ = invocation_handoff
    monkeypatch.setattr(managed, "_serves_requests", lambda: True)
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a service was packaged"))
    engine = create(lambda _: pytest.fail("a service was submitted"), local_fallback=True)
    assert managed.handoff_if_enabled(engine, object()) is None
    assert engine._managed_execution_local is True
    assert _nothing_was_handed_to_the_server(engine)
    # The choice holds for the life of the process.
    monkeypatch.setattr(managed, "_serves_requests", lambda: False)
    assert managed.handoff_if_enabled(engine, object()) is None
    assert _nothing_was_handed_to_the_server(engine)


def test_a_service_is_still_handed_over_where_an_operator_chose_server_runs(invocation_handoff, monkeypatch):
    create, _, _ = invocation_handoff
    monkeypatch.setattr(managed, "_serves_requests", lambda: True)
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    # No local_fallback: the workspace was listed on purpose, as before.
    engine = create(handler)
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 1


def test_no_service_fallback_once_a_submission_exists(invocation_handoff, monkeypatch, capsys):
    create, _, _ = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    engine = create(handler, local_fallback=True)
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    # The server owns this run now, whatever the process looks like later.
    monkeypatch.setattr(managed, "_serves_requests", lambda: True)
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    assert not hasattr(engine, "_managed_execution_local")
    capsys.readouterr()


_UNPACKAGEABLE = [
    "managed_execution_bundle_unsupported",
    "managed_execution_dependency_missing",
    "managed_execution_bundle_too_large",
]
_DELEGATED = ["requester", "user", "service_account", "acting_user", "task_scope", "task_environment"]


def _delegated(engine, context, monkeypatch):
    """Put ``engine`` into one delegated context; returns the scope to run the handoff in."""
    from contextlib import nullcontext
    from deepintshield.agentic.identity import PrincipalBinding
    from deepintshield.agentic.task_context import task_scope

    if context == "requester":
        engine._requester_hint = "  alice@example.com  "
    elif context in ("user", "service_account"):
        engine.principal_binding = PrincipalBinding(subject="user:alice", kind=context)
    elif context == "acting_user":
        engine.acting_user = "user:alice"
    elif context == "task_environment":
        monkeypatch.setenv("DEEPINTSHIELD_TASK_MANDATE", "private-test-mandate")
    return task_scope("private-test-mandate") if context == "task_scope" else nullcontext()


def _nothing_was_handed_to_the_server(engine):
    return not hasattr(engine, "_managed_execution_submission") and not hasattr(engine, "_managed_execution_wire")


@pytest.mark.parametrize("context", _DELEGATED)
def test_local_fallback_keeps_a_delegated_context_on_the_local_path(context, monkeypatch):
    calls = []
    engine = engine_for(lambda request: calls.append(request.method) or httpx.Response(200, json={"enabled": True, "local_fallback": True}))
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("delegated source was packaged"))
    try:
        with _delegated(engine, context, monkeypatch):
            # Returning lets the program run locally, exactly as it did before server runs were offered.
            assert managed.handoff_if_enabled(engine, object()) is None
            assert managed.handoff_if_enabled(engine, object()) is None
        assert calls == ["GET"]
        assert engine._managed_execution_local is True
        assert _nothing_was_handed_to_the_server(engine)
    finally:
        engine._http.close()


@pytest.mark.parametrize("code", _UNPACKAGEABLE)
def test_local_fallback_keeps_a_program_that_cannot_be_packaged_local(invocation_handoff, monkeypatch, code):
    create, _, clock = invocation_handoff
    attempts = []

    def refuse(*_):
        attempts.append(code)
        raise GovernanceConfigurationError(framework="managed-execution", code=code)

    monkeypatch.setattr(managed, "package_script", refuse)
    engine = create(lambda _: pytest.fail("a program that cannot be packaged was submitted"), local_fallback=True)
    assert managed.handoff_if_enabled(engine, object()) is None
    # The choice is remembered: later invocations return at once without packaging again.
    assert managed.handoff_if_enabled(engine, object()) is None
    assert attempts == [code]
    assert engine._managed_execution_local is True
    assert _nothing_was_handed_to_the_server(engine)
    assert clock.sleeps == []


@pytest.mark.parametrize("program", ["no entry script", "not a python file", "code loader", "missing import", "too large"])
def test_local_fallback_covers_what_the_real_packager_refuses(tmp_path, monkeypatch, program):
    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(request.url.path)
        return httpx.Response(200, json={"enabled": True, "local_fallback": True, "max_bundle_bytes": 200})

    main = sys.modules["__main__"]
    script = tmp_path / "entry.py"
    script.write_text({
        "code loader": "exec('print(1)')\n",
        "missing import": "import absent_dependency_for_fallback\n",
        "too large": "value = '" + "a" * 300 + "'\n",
    }.get(program, "print(1)\n"))
    if program == "no entry script":
        # An interactive session or notebook has no entry file to package.
        monkeypatch.delattr(main, "__file__", raising=False)
    elif program == "not a python file":
        monkeypatch.setattr(main, "__file__", str(tmp_path / "console-script"), raising=False)
    else:
        monkeypatch.setattr(main, "__file__", str(script), raising=False)
    if program == "missing import":
        monkeypatch.setattr(managed.importlib.metadata, "packages_distributions", lambda: {})
    engine = engine_for(handler)
    try:
        assert managed.handoff_if_enabled(engine, object()) is None
        assert managed.handoff_if_enabled(engine, object()) is None
        assert posts == []
        assert engine._managed_execution_local is True
        assert _nothing_was_handed_to_the_server(engine)
    finally:
        engine._http.close()


@pytest.mark.parametrize("obstacle", ["no agent to bind", "source over the review limit", "bundle over the gateway limit"])
def test_local_fallback_covers_a_bundle_that_cannot_be_bound_or_sent(invocation_handoff, monkeypatch, obstacle):
    from deepintshield.agentic import registry

    create, script, _ = invocation_handoff
    settings = {"local_fallback": True}
    if obstacle == "no agent to bind":
        monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": []})
    elif obstacle == "source over the review limit":
        monkeypatch.setattr(managed, "package_script", lambda *_: {"entrypoint": "entry.py", "files": {"entry.py": "#" * 70000}, "requirements": []})
    else:
        settings["max_bundle_bytes"] = 16
    engine = create(lambda _: pytest.fail("a program that cannot be packaged was submitted"), **settings)
    assert managed.handoff_if_enabled(engine, object()) is None
    assert managed.handoff_if_enabled(engine, object()) is None
    assert engine._managed_execution_local is True
    assert _nothing_was_handed_to_the_server(engine)


@pytest.mark.parametrize("flag", [None, False, "true", 1, {"enabled": True}])
@pytest.mark.parametrize("obstacle", ["requester", "task_scope", *_UNPACKAGEABLE])
def test_without_the_local_fallback_flag_nothing_changes(invocation_handoff, monkeypatch, flag, obstacle):
    create, _, clock = invocation_handoff
    settings = {} if flag is None else {"local_fallback": flag}
    engine = create(lambda _: pytest.fail("the program was submitted"), **settings)
    expected = "managed_execution_context_unsupported"
    scope = _delegated(engine, obstacle, monkeypatch)
    if obstacle in _UNPACKAGEABLE:
        expected = obstacle

        def refuse(*_):
            raise GovernanceConfigurationError(framework="managed-execution", code=obstacle)

        monkeypatch.setattr(managed, "package_script", refuse)
    with scope:
        for _ in range(2):
            with pytest.raises(GovernanceConfigurationError, match=expected):
                managed.handoff_if_enabled(engine, object())
    assert not hasattr(engine, "_managed_execution_local")
    assert _nothing_was_handed_to_the_server(engine)
    assert clock.sleeps == []


@pytest.mark.parametrize("failure", ["invalid gateway limit", "scan unavailable", "unexpected defect"])
def test_local_fallback_still_raises_every_other_error(invocation_handoff, monkeypatch, failure):
    from deepintshield.agentic import registry

    create, _, _ = invocation_handoff
    settings = {"local_fallback": True}
    if failure == "invalid gateway limit":
        settings["max_bundle_bytes"] = "large"
        expected = pytest.raises(GovernanceConfigurationError, match="managed_execution_response_invalid")
    elif failure == "scan unavailable":
        def unavailable(_):
            raise GovernanceConfigurationError(framework="managed-execution", code="blueprint_scan_unavailable")

        monkeypatch.setattr(registry, "describe_network", unavailable)
        expected = pytest.raises(GovernanceConfigurationError, match="blueprint_scan_unavailable")
    else:
        monkeypatch.setattr(managed, "package_script", lambda *_: 1 / 0)
        expected = pytest.raises(ZeroDivisionError)
    engine = create(lambda _: pytest.fail("the program was submitted"), **settings)
    with expected:
        managed.handoff_if_enabled(engine, object())
    assert not hasattr(engine, "_managed_execution_local")
    assert _nothing_was_handed_to_the_server(engine)


def test_local_fallback_does_not_stop_a_program_the_server_can_take(invocation_handoff, capsys):
    create, _, _ = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    engine = create(handler, local_fallback=True)
    with pytest.raises(SystemExit) as stopped:
        managed.handoff_if_enabled(engine, object())
    assert stopped.value.code == 0
    assert len(bodies) == 1 and bodies[0]["entrypoint"] == "entry.py"
    assert not hasattr(engine, "_managed_execution_local")
    assert capsys.readouterr().out == "21\n"


# ── a server run the agent's identity cannot use ─────────────────────────

_IDENTITY_ENDED_RUN = {
    "id": "run-1", "status": "failed", "stage": "prepare", "error_code": "managed_gateway_runtime_required",
}


def test_a_server_run_the_identity_cannot_use_returns_the_program_here(invocation_handoff, caplog):
    import logging

    create, _, clock = invocation_handoff
    posts = []

    def handler(request):
        posts.append(request.content)
        return httpx.Response(202, json={"job": dict(_IDENTITY_ENDED_RUN)})

    engine = create(handler, local_fallback=True)
    caplog.set_level(logging.INFO, logger="deepintshield.agentic.managed_execution")
    # Returning lets the program run on this machine: it never ran on the server.
    assert managed.handoff_if_enabled(engine, object()) is None
    assert engine._managed_execution_local is True
    # Later invocations stay here and send nothing.
    assert managed.handoff_if_enabled(engine, object()) is None
    assert len(posts) == 1
    assert clock.sleeps == []
    assert caplog.messages == [
        "DeepintShield: this agent's identity cannot use a server run; the program runs on this machine."
    ]


def test_a_run_ended_for_its_identity_while_followed_returns_the_program_here(invocation_handoff):
    create, _, clock = invocation_handoff
    posts = []
    jobs = iter([{"id": "run-1", "status": "queued", "stage": "prepare"}, dict(_IDENTITY_ENDED_RUN)])

    def handler(request):
        posts.append(request.content)
        return httpx.Response(202, json={"job": next(jobs)})

    engine = create(handler, local_fallback=True)
    assert managed.handoff_if_enabled(engine, object()) is None
    assert engine._managed_execution_local is True
    # One invocation, polled once: the same submission both times.
    assert len(posts) == 2 and posts[0] == posts[1]
    assert clock.sleeps == [1]


def test_a_process_that_kept_running_follows_its_run_home_and_submits_nothing_new(invocation_handoff):
    # Run 1 waits for an identity; the administrator approves the agent as
    # VK-trusted; the same process invokes the graph again.
    create, _, _ = invocation_handoff
    posts = []

    def handler(request):
        posts.append(request.content)
        job = {"id": "run-1", "status": "waiting_identity", "stage": "prepare"} if len(posts) == 1 else dict(_IDENTITY_ENDED_RUN)
        return httpx.Response(202, json={"job": job})

    engine = create(handler, local_fallback=True)
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_identity_pending"):
        managed.handoff_if_enabled(engine, object())
    assert managed.handoff_if_enabled(engine, object()) is None
    assert managed.handoff_if_enabled(engine, object()) is None
    # The second request asked about the same invocation; nothing new was sent.
    assert len(posts) == 2 and posts[0] == posts[1]
    assert engine._managed_execution_local is True


def test_the_next_start_after_the_run_ended_runs_locally_without_packaging(monkeypatch):
    # The gateway counts the ended run and no longer offers this program a server run.
    calls = []
    engine = engine_for(lambda request: calls.append(request.url.path) or httpx.Response(200, json={"enabled": False}))
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a program that runs here was packaged"))
    try:
        assert managed.handoff_if_enabled(engine, object()) is None
        assert managed.handoff_if_enabled(engine, object()) is None
        assert calls == ["/api/agentic-new/managed-execution/capabilities"]
        assert engine._managed_execution_local is True
        assert _nothing_was_handed_to_the_server(engine)
    finally:
        engine._http.close()


@pytest.mark.parametrize(
    ("offered", "job"),
    [
        # A workspace listed by name: no local fall-back was offered.
        pytest.param(False, dict(_IDENTITY_ENDED_RUN), id="no_local_fallback"),
        # The run reached its program: running it here would repeat it.
        pytest.param(True, {**_IDENTITY_ENDED_RUN, "stage": "execute"}, id="reached_its_program"),
        pytest.param(True, {**_IDENTITY_ENDED_RUN, "stage": ""}, id="stage_unknown"),
        pytest.param(True, {**_IDENTITY_ENDED_RUN, "error_code": "managed_identity_unavailable"}, id="other_failure"),
        pytest.param(True, {**_IDENTITY_ENDED_RUN, "error_code": "agent_runtime_authentication_failed"}, id="refused_key"),
    ],
)
def test_only_a_run_ended_for_its_identity_before_its_program_comes_home(invocation_handoff, offered, job):
    create, _, _ = invocation_handoff
    posts = []

    def handler(request):
        posts.append(request.content)
        return httpx.Response(202, json={"job": dict(job)})

    engine = create(handler, local_fallback=True) if offered else create(handler)
    # A catalogued failure code is reported as itself (managed_gateway_runtime_required since 3.2.0);
    # an uncatalogued one stays the generic managed_execution_failed. Either way the program never comes home.
    expected = job["error_code"] if job["error_code"] in {"agent_runtime_authentication_failed", "managed_gateway_runtime_required"} else "managed_execution_failed"
    with pytest.raises(GovernanceConfigurationError, match=expected):
        managed.handoff_if_enabled(engine, object())
    assert not hasattr(engine, "_managed_execution_local")
    assert len(posts) == 1


@pytest.mark.parametrize("context", _DELEGATED)
def test_no_local_fallback_once_a_submission_exists(invocation_handoff, monkeypatch, context):
    create, _, _ = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(request.content)
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "waiting_approval"}})

    engine = create(handler, local_fallback=True)
    with pytest.raises(GovernanceConfigurationError, match="agent_registration_pending"):
        managed.handoff_if_enabled(engine, object())
    # The server owns the run now. A delegated context must not start a second, local one.
    with _delegated(engine, context, monkeypatch):
        for _ in range(2):
            with pytest.raises(GovernanceConfigurationError, match="managed_execution_context_unsupported"):
                managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 1
    assert not hasattr(engine, "_managed_execution_local")
    assert engine._managed_execution_submission["job"]["id"] == "run-1"


def test_no_local_fallback_after_a_submission_whose_answer_was_lost(invocation_handoff, monkeypatch, capsys):
    create, _, _ = invocation_handoff
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            raise httpx.ReadTimeout("PRIVATE response was lost", request=request)
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    engine = create(handler, local_fallback=True)
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_submission_failed"):
        managed.handoff_if_enabled(engine, object())
    assert isinstance(engine._managed_execution_wire, bytes)
    # The gateway may already be running it: neither a delegated context nor a
    # program that could no longer be packaged may run it locally as well.
    engine._requester_hint = "alice@example.com"
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_context_unsupported"):
        managed.handoff_if_enabled(engine, object())
    engine._requester_hint = ""
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a persisted submission was packaged again"))
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert not hasattr(engine, "_managed_execution_local")
    assert capsys.readouterr().out == "21\n"


@pytest.mark.parametrize("identified", [True, False])
@pytest.mark.parametrize("code", _UNPACKAGEABLE)
def test_gateway_refusing_a_submission_is_not_a_local_fallback(invocation_handoff, identified, code):
    create, _, _ = invocation_handoff
    posts = []

    def handler(request):
        posts.append(request.url.path)
        return httpx.Response(413, json={"error": {"code": code}})

    engine = create(handler, local_fallback=True, submission_ids=identified)
    with pytest.raises(GovernanceConfigurationError, match=code):
        managed.handoff_if_enabled(engine, object())
    assert len(posts) == 1
    assert not hasattr(engine, "_managed_execution_local")


def test_remembered_local_choice_outlives_the_context_that_caused_it(invocation_handoff, monkeypatch):
    create, _, _ = invocation_handoff
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("a locally running program was packaged"))
    engine = create(lambda _: pytest.fail("a locally running program was submitted"), local_fallback=True)
    engine._requester_hint = "alice@example.com"
    assert managed.handoff_if_enabled(engine, object()) is None
    # The program is already running here; a later autonomous invocation must not hand it to the server too.
    engine._requester_hint = ""
    assert managed.handoff_if_enabled(engine, object()) is None
    assert engine._managed_execution_local is True
    assert _nothing_was_handed_to_the_server(engine)


def test_concurrent_first_invocations_package_once_and_agree(invocation_handoff, monkeypatch):
    create, _, _ = invocation_handoff
    attempts = []
    contenders = 6
    together = threading.Barrier(contenders)

    def refuse(*_):
        attempts.append(threading.get_ident())
        raise GovernanceConfigurationError(framework="managed-execution", code="managed_execution_bundle_unsupported")

    monkeypatch.setattr(managed, "package_script", refuse)
    engine = create(lambda _: pytest.fail("a program that cannot be packaged was submitted"), local_fallback=True)
    results = []

    def invoke():
        together.wait(10)
        results.append(managed.handoff_if_enabled(engine, object()))

    threads = [threading.Thread(target=invoke) for _ in range(contenders)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == [None] * contenders
    assert len(attempts) == 1


@pytest.mark.parametrize("local_fallback", [True, False])
def test_local_fallback_reaches_the_ordinary_local_registration_and_run(monkeypatch, local_fallback):
    """A delegated program keeps its whole local path: discovery, the PDP and the nodes themselves."""
    pytest.importorskip("langgraph", reason="requires the optional LangGraph integration")
    from langgraph.graph import StateGraph, START, END
    from deepintshield.agentic import execution, registry
    from deepintshield.agentic.integrations import langgraph as integration

    calls, discoveries, decisions = [], [], []

    def handler(request):
        path = request.url.path
        calls.append((request.method, path))
        if path.endswith("/runtime-config"):
            return httpx.Response(200, json={"security_enabled": True, "available": True, "plan": "business", "reason": "enabled"})
        if path.endswith("/managed-execution/capabilities"):
            return httpx.Response(200, json={"enabled": True, "local_fallback": local_fallback, "submission_ids": True, "result_polling": True})
        if path.endswith("/vk-credential-info"):
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
        if path.endswith("/registry/discover"):
            discoveries.append(json.loads(request.content))
            return httpx.Response(200, json={"network_id": "math-network", "registration_state": "active", "blueprint_scan_status": "complete"})
        if path.endswith("/decide"):
            decisions.append(json.loads(request.content))
            return httpx.Response(200, json={"verdict": "ALLOW", "allow": True, "proceed": True, "mode": "enforce", "would_block": False})
        if path.endswith("/agentic-new/executions"):
            return httpx.Response(200, json={})
        assert not path.endswith("/submissions"), "a delegated program was handed to the server"
        return httpx.Response(404)

    def run_now(function, *args):
        function(*args)
        return True

    monkeypatch.setattr(integration, "run_detached", run_now)
    monkeypatch.setattr(execution, "_dispatch", run_now)
    monkeypatch.setattr(managed, "package_script", lambda *_: pytest.fail("delegated source was packaged"))
    registry.reset_dedupe()
    try:
        with DeepintShield(virtual_key="sk-ds-managed-test", base_url="https://gateway.example", agent_name="math", requester="alice@example.com") as client:
            client._client.close()
            client._client = httpx.Client(transport=httpx.MockTransport(handler))
            engine = client.agentic.engine
            graph = StateGraph(dict)
            graph.add_node("increment", lambda state: {"number": state["number"] + 1})
            graph.add_edge(START, "increment")
            graph.add_edge("increment", END)
            app = graph.compile()
            # Compile skips registry discovery whenever the gateway offers server runs.
            barrier = getattr(app, integration._DISCOVERY_BARRIER)
            assert barrier.completed.is_set() and not barrier.succeeded and barrier.error is None
            assert discoveries == []
            if not local_fallback:
                # An operator-selected workspace keeps today's refusal of a delegated context.
                with pytest.raises(GovernanceConfigurationError, match="managed_execution_context_unsupported"):
                    integration._await_initial_discovery(engine, app)
                assert discoveries == [] and not hasattr(engine, "_managed_execution_local")
                return
            # The first execution falls back, then performs the canonical discovery post itself.
            integration._await_initial_discovery(engine, app)
            assert barrier.succeeded and barrier.error is None
            assert len(discoveries) == 1
            assert discoveries[0]["principal"]["email"] == "alice@example.com"
            assert [agent["key"] for agent in discoveries[0]["agents"]]
            assert engine._managed_execution_local is True
            assert _nothing_was_handed_to_the_server(engine)
            # And the program then runs locally under the PDP, as it does today.
            assert app.invoke({"number": 1}) == {"number": 2}
            assert app.invoke({"number": 5}) == {"number": 6}
            assert [decision["tool"] for decision in decisions] == ["tool:increment"] * 2
            assert len(discoveries) == 1
            assert not any(path.endswith("/submissions") for _, path in calls)
    finally:
        registry.reset_dedupe()


# --- local fall-back: the gateway names what its worker image has installed ------


def _needing(monkeypatch, script, *requirements):
    """Make the packaged program need ``requirements``; returns a list that grows with every packaging."""
    packaged = []

    def package(*_):
        packaged.append(requirements)
        return {"entrypoint": "entry.py", "files": {"entry.py": script.read_text()}, "requirements": list(requirements)}

    monkeypatch.setattr(managed, "package_script", package)
    return packaged


def _taking(bodies):
    """A gateway that takes every submission, records it in ``bodies`` and has run it."""
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(202, json={"job": {"id": "run-1", "status": "succeeded", "result": "21\n"}})

    return handler


def test_program_whose_packages_the_worker_has_is_handed_over(invocation_handoff, monkeypatch, capsys):
    create, script, _ = invocation_handoff
    # Releases the worker does not have: only the names are compared.
    _needing(monkeypatch, script, "deepintshield==0.0.1", "langgraph==99.0")
    bodies = []
    engine = create(_taking(bodies), local_fallback=True, worker_packages=["deepintshield", "httpx", "langgraph"])
    with pytest.raises(SystemExit) as stopped:
        managed.handoff_if_enabled(engine, object())
    assert stopped.value.code == 0
    assert len(bodies) == 1 and bodies[0]["requirements"] == ["deepintshield==0.0.1", "langgraph==99.0"]
    assert not hasattr(engine, "_managed_execution_local")
    assert capsys.readouterr().out == "21\n"


@pytest.mark.parametrize("installed", [["deepintshield", "langgraph"], ["deepintshield", "langgraph", "pandas-stubs"], []])
def test_program_that_needs_a_package_the_worker_lacks_stays_local(invocation_handoff, monkeypatch, installed):
    create, script, clock = invocation_handoff
    packaged = _needing(monkeypatch, script, "deepintshield==2.8.3", "langgraph==1.1.6", "pandas==2.2.0")
    engine = create(lambda _: pytest.fail("a program the worker cannot run was submitted"), local_fallback=True, worker_packages=installed)
    assert managed.handoff_if_enabled(engine, object()) is None
    # The choice is remembered: later invocations return at once without packaging again.
    assert managed.handoff_if_enabled(engine, object()) is None
    assert len(packaged) == 1
    assert engine._managed_execution_local is True
    assert _nothing_was_handed_to_the_server(engine)
    assert clock.sleeps == []


@pytest.mark.parametrize(("requirement", "installed", "present"), [
    ("Foo_Bar.baz==1.0", "foo-bar-baz", True),
    ("typing_extensions==4.12.2", "typing-extensions", True),
    ("ruamel.yaml==0.18.6", "ruamel-yaml", True),
    ("A__b--C.-_d==1", "a-b-c-d", True),
    # Another distribution, whose name only begins or reads the same.
    ("foo-bar==1.0", "foo-bar-baz", False),
    ("foobar==1.0", "foo-bar", False),
])
def test_worker_package_names_are_compared_as_python_normalizes_them(invocation_handoff, monkeypatch, capsys, requirement, installed, present):
    create, script, _ = invocation_handoff
    _needing(monkeypatch, script, requirement)
    bodies = []
    engine = create(_taking(bodies), local_fallback=True, worker_packages=[installed])
    if present:
        with pytest.raises(SystemExit):
            managed.handoff_if_enabled(engine, object())
    else:
        assert managed.handoff_if_enabled(engine, object()) is None
    assert len(bodies) == (1 if present else 0)
    assert getattr(engine, "_managed_execution_local", False) is not present
    capsys.readouterr()


@pytest.mark.parametrize("field", [
    None, "deepintshield", 7, True, {"deepintshield": "2.8.3"}, ["deepintshield", 7], ["deepintshield", None], [["deepintshield"]],
])
def test_worker_package_list_that_is_not_a_list_of_names_is_ignored(invocation_handoff, monkeypatch, capsys, field):
    create, script, _ = invocation_handoff
    _needing(monkeypatch, script, "pandas==2.2.0")
    bodies = []
    engine = create(_taking(bodies), local_fallback=True, worker_packages=field)
    # As if the gateway had named nothing: the server decides, as it did before.
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(engine, object())
    assert len(bodies) == 1
    assert not hasattr(engine, "_managed_execution_local")
    capsys.readouterr()


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_worker_package_list_is_ignored_without_the_local_fallback_flag(invocation_handoff, monkeypatch, capsys, flag):
    create, script, _ = invocation_handoff
    _needing(monkeypatch, script, "pandas==2.2.0")
    bodies = []
    settings = {"worker_packages": ["deepintshield"], **({} if flag is None else {"local_fallback": flag})}
    # A workspace an operator listed keeps today's path: the program is sent, and the server answers for it.
    with pytest.raises(SystemExit):
        managed.handoff_if_enabled(create(_taking(bodies), **settings), object())
    assert len(bodies) == 1 and bodies[0]["requirements"] == ["pandas==2.2.0"]
    capsys.readouterr()


@pytest.mark.parametrize("worker_has_it", [True, False])
def test_worker_package_list_meets_what_the_real_packager_pins(tmp_path, monkeypatch, worker_has_it):
    from deepintshield.agentic import registry

    posts = []

    def handler(request):
        if request.method == "POST":
            posts.append(json.loads(request.content))
            return httpx.Response(202, json={"job": {"id": "run-1", "status": "waiting_approval"}})
        installed = ["deepintshield", "httpx"] if worker_has_it else ["deepintshield", "httpx-sse"]
        return httpx.Response(200, json={"enabled": True, "local_fallback": True, "worker_packages": installed})

    script = tmp_path / "entry.py"
    script.write_text("import httpx\n")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script), raising=False)
    monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": [{"key": "math"}]})
    monkeypatch.setattr(registry, "_wire_attested_payload", lambda _, payload: payload)
    engine = engine_for(handler)
    try:
        if worker_has_it:
            with pytest.raises(GovernanceConfigurationError, match="agent_registration_pending"):
                managed.handoff_if_enabled(engine, object())
            assert [body["requirements"] for body in posts] == [["httpx==" + managed.importlib.metadata.version("httpx")]]
            assert not hasattr(engine, "_managed_execution_local")
        else:
            assert managed.handoff_if_enabled(engine, object()) is None
            assert managed.handoff_if_enabled(engine, object()) is None
            assert posts == []
            assert engine._managed_execution_local is True
            assert _nothing_was_handed_to_the_server(engine)
    finally:
        engine._http.close()
