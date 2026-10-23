"""A live worker waits at the denied boundary; completed operations never replay."""
from types import SimpleNamespace

import pytest

from deepintshield.agentic import gate, managed_execution as managed
from deepintshield.agentic.errors import GuardrailApprovalPending, GuardrailDenied, GovernanceConfigurationError
from deepintshield.agentic.types import Decision, Verdict


def decision(verdict):
    return Decision(verdict=verdict, decision_id="decision", gaf_checked=True, approval_id="exact-operation")


@pytest.fixture
def worker_clock(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(managed, "_WORKER_FILES", {"entry.py": "# hermetic fixture"})
    monkeypatch.setattr(managed, "_WORKER_APPROVAL_DEADLINE", 110.0)
    monkeypatch.setattr(managed.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(managed.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(gate, "preflight", lambda *_, **__: None)
    return clock


def engine_with(callback):
    return SimpleNamespace(virtual_key="fixture", agent_subject="agent:math", acting_user="", _approval_timeout=300.0, _approval_interval=1.0, decide=callback)


def test_wait_resumes_only_blocked_operation_with_identical_context(worker_clock):
    calls, executed = [], []
    def check(context):
        calls.append(context.model_dump())
        if context.tool == "multiply_by_three" and len(calls) < 4:
            return decision(Verdict.REQUIRE_APPROVAL)
        return decision(Verdict.ALLOW)
    engine = engine_with(check)
    def call(name, value):
        args, kwargs = gate.enforce_call(engine, name, (value,), {})
        executed.append(name)
        return args[0] + 2 if name == "add_two" else args[0] * 3
    value = call("add_two", 5)
    assert call("multiply_by_three", value) == 21
    assert executed == ["add_two", "multiply_by_three"]
    assert [call["tool"] for call in calls] == ["add_two", "multiply_by_three", "multiply_by_three", "multiply_by_three"]
    assert calls[1] == calls[2] == calls[3]


def test_ordinary_client_still_returns_pending_immediately(worker_clock, monkeypatch):
    monkeypatch.setattr(managed, "_WORKER_FILES", None)
    checks = []
    engine = engine_with(lambda context: checks.append(context) or decision(Verdict.REQUIRE_APPROVAL))
    with pytest.raises(GuardrailApprovalPending):
        gate.resolve(engine, "add_two", (5,), {})
    assert len(checks) == 1
    assert worker_clock[0] == 100.0


def test_deny_during_wait_stops_without_invoking_tool(worker_clock):
    checks = []
    def check(context):
        checks.append(context)
        return decision(Verdict.REQUIRE_APPROVAL if len(checks) == 1 else Verdict.DENY)
    with pytest.raises(GuardrailDenied):
        gate.resolve(engine_with(check), "add_two", (5,), {})
    assert len(checks) == 2


def test_worker_deadline_is_shared_across_operations(worker_clock):
    checks = []
    engine = engine_with(lambda context: checks.append(context) or decision(Verdict.REQUIRE_APPROVAL))
    for _ in range(2):
        with pytest.raises(GuardrailApprovalPending):
            gate.resolve(engine, "add_two", (5,), {})
    assert worker_clock[0] == 110.0
    assert len(checks) == 11  # initial+9 polls, then next initial check; no new wait


def test_mutated_arguments_cannot_use_approval_for_previous_digest(worker_clock, monkeypatch):
    argument = {"number": 5}
    count = [0]
    def check(context):
        count[0] += 1
        if count[0] == 1:
            return decision(Verdict.REQUIRE_APPROVAL)
        argument["number"] = 500
        return decision(Verdict.ALLOW)
    with pytest.raises(GuardrailDenied):
        gate.resolve(engine_with(check), "add_two", (argument,), {})


def test_worker_cancel_interrupts_wait_without_retry(worker_clock, monkeypatch):
    checks = []
    def interrupt(_seconds):
        raise KeyboardInterrupt()
    monkeypatch.setattr(managed.time, "sleep", interrupt)
    engine = engine_with(lambda context: checks.append(context) or decision(Verdict.REQUIRE_APPROVAL))
    with pytest.raises(KeyboardInterrupt):
        gate.resolve(engine, "add_two", (5,), {})
    assert len(checks) == 1


def test_wait_cannot_downgrade_to_legacy_approval(worker_clock):
    count = [0]
    def check(_):
        count[0] += 1
        return decision(Verdict.REQUIRE_APPROVAL) if count[0] == 1 else Decision(verdict=Verdict.ALLOW, decision_id="legacy")
    with pytest.raises(GovernanceConfigurationError):
        gate.resolve(engine_with(check), "add_two", (5,), {})


def test_missing_approval_id_returns_pending_without_wait(worker_clock):
    checks = []
    pending = decision(Verdict.REQUIRE_APPROVAL)
    pending.approval_id = ""
    engine = engine_with(lambda context: checks.append(context) or pending)
    with pytest.raises(GuardrailApprovalPending):
        gate.resolve(engine, "add_two", (5,), {})
    assert len(checks) == 1
    assert worker_clock[0] == 100.0


def test_active_async_loop_returns_pending_without_blocking(worker_clock):
    import asyncio

    checks = []
    engine = engine_with(lambda context: checks.append(context) or decision(Verdict.REQUIRE_APPROVAL))
    async def run():
        with pytest.raises(GuardrailApprovalPending):
            gate.resolve(engine, "add_two", (5,), {})
        await asyncio.sleep(0)
    asyncio.run(run())
    assert len(checks) == 1
    assert worker_clock[0] == 100.0
