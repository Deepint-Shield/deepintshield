from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from deepintshield.agentic import current_task_context, memory_reference, task_scope
from deepintshield.agentic.credentials import StaticAgentCredential
from deepintshield.agentic.errors import DeepIntShieldError, GovernanceConfigurationError
from deepintshield.agentic.task_context import TaskContext, capture_task_receipt, task_execution_scope, task_payload
from deepintshield.agentic.task_memory import commit_task_memory
from deepintshield.agentic.types import DelegationContext
from tests.test_agentic_runtime_config import runtime_config


def _dc():
    return DelegationContext(tool="reader", action="read", agent="agent:test", args_digest="sha256:test")


def _engine(shield_factory, handler, **kwargs):
    engine = shield_factory(handler, **kwargs).agentic.engine
    engine.set_agent_credential(StaticAgentCredential("workload-token"))
    return engine


def test_receipt_ancestry_is_carried_and_raw_memory_is_never_sent(shield_factory):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"verdict": "ALLOW", "allow": True,
            "task_id": "task-1", "task_receipt": f"receipt-{len(requests)}"})

    engine = _engine(shield_factory, handler)
    with task_scope("signed-secret", input_receipts=["external-receipt"], memory_inputs=[memory_reference("notes", b"private bytes", 2)]):
        first = engine.decide(_dc())
        second = engine.decide(_dc())
        assert first.task_receipt == "receipt-1" and second.task_id == "task-1"
        assert current_task_context().last_receipt == "receipt-2"
        assert "signed-secret" not in repr(current_task_context())
    assert current_task_context() is None
    assert requests[0]["input_receipts"] == ["external-receipt"]
    assert requests[1]["input_receipts"] == ["receipt-1"]
    assert requests[0]["memory_inputs"][0]["digest"] == memory_reference("notes", b"private bytes", 2).digest
    assert "private bytes" not in json.dumps(requests)


def test_denial_does_not_advance_ancestry(shield_factory):
    engine = _engine(shield_factory, lambda _: httpx.Response(200, json={"verdict": "DENY", "allow": False}))
    with task_scope("signed", input_receipts=["parent"]):
        decision = engine.decide(_dc())
        assert decision.proceed is False
        assert current_task_context().input_receipts == ("parent",)


def test_task_receipt_cannot_convert_shadow_denial_to_permission(shield_factory):
    engine = _engine(shield_factory, lambda _: httpx.Response(200, json={
        "verdict": "DENY", "allow": False, "proceed": True, "mode": "shadow",
        "task_id": "task", "task_receipt": "unexpected-receipt",
    }))
    with task_scope("signed"):
        with pytest.raises(GovernanceConfigurationError, match="agent_task_security_unavailable"):
            engine.decide(_dc())
        assert current_task_context().last_receipt == ""


def test_repeated_task_receipt_cannot_authorize_another_call(shield_factory):
    engine = _engine(shield_factory, lambda _: httpx.Response(200, json={
        "verdict": "ALLOW", "allow": True, "task_id": "task", "task_receipt": "same-receipt",
    }))
    with task_scope("signed"):
        assert engine.decide(_dc()).proceed is True
        with pytest.raises(GovernanceConfigurationError, match="agent_task_context_invalid"):
            engine.decide(_dc())


@pytest.mark.parametrize("status,body", [(404, {}), (200, {"verdict": "ALLOW", "allow": True})])
def test_task_cannot_fall_back_or_accept_unaware_gateway(shield_factory, status, body):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(status, json=body)

    engine = _engine(shield_factory, handler)
    engine._canonical_credential_info_available = False
    with task_scope("signed"):
        with pytest.raises(GovernanceConfigurationError, match="agent_task_security_unavailable"):
            engine.decide(_dc())
    assert paths == ["/api/agentic-new/decide"]


def test_task_never_uses_cached_legacy_fallback(shield_factory):
    engine = _engine(shield_factory, lambda _: pytest.fail("legacy request"))
    engine._gaf_unavailable_until = float("inf")
    engine._canonical_credential_info_available = False
    with task_scope("signed"):
        with pytest.raises(GovernanceConfigurationError): engine.decide(_dc())
        with pytest.raises(GovernanceConfigurationError): engine._post_legacy_compat_decide(_dc())


def test_environment_task_cannot_become_runtime_disabled_passthrough(shield_factory, monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_TASK_MANDATE", "signed-env")
    engine = _engine(shield_factory, lambda _: pytest.fail("unexpected data-plane request"),
        runtime_config_handler=lambda _: httpx.Response(200, json=runtime_config("business", enabled=False)))
    with pytest.raises(GovernanceConfigurationError, match="agent_task_security_unavailable"):
        engine.decide(_dc())


def test_task_added_inside_preexisting_disabled_security_scope_cannot_passthrough(shield_factory):
    from deepintshield.agentic.runtime import security_enabled, security_scope

    engine = _engine(shield_factory, lambda _: pytest.fail("unexpected data-plane request"),
        runtime_config_handler=lambda _: httpx.Response(200, json=runtime_config("business", enabled=False)))
    with security_scope(engine):
        with task_scope("signed"):
            with pytest.raises(GovernanceConfigurationError, match="agent_task_security_unavailable"):
                security_enabled(engine)
            with pytest.raises(GovernanceConfigurationError, match="agent_task_security_unavailable"):
                with security_scope(engine): pass


def test_environment_scope_resets_ancestry_between_native_executions(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_TASK_MANDATE", "signed-env")
    for _ in range(2):
        with task_execution_scope(new_execution=True):
            assert task_payload()["input_receipts"] == []
            capture_task_receipt("task-env", "receipt-one")
            with task_execution_scope(new_execution=False):
                assert task_payload()["input_receipts"] == ["receipt-one"]
    assert task_payload()["input_receipts"] == []


def test_parallel_scopes_and_nested_scopes_do_not_mutate_other_branches():
    async def run():
        with task_scope("parent", input_receipts=["root"]):
            async def child(number):
                capture_task_receipt("task", f"child-{number}")
                await asyncio.sleep(0)
                return current_task_context().input_receipts
            results = await asyncio.gather(child(1), child(2))
            assert results == [("child-1",), ("child-2",)]
            assert current_task_context().input_receipts == ("root",)
            with task_scope("nested"):
                capture_task_receipt("nested-task", "nested-receipt")
            assert current_task_context().mandate == "parent"
    asyncio.run(run())


def test_task_context_copies_mutable_inputs_and_rejects_cross_task_receipt():
    parents = ["root"]
    context = TaskContext("secret", parents)
    parents.append("mutated")
    assert context.input_receipts == ("root",)
    with task_scope("signed"):
        capture_task_receipt("one", "receipt-one")
        with pytest.raises(GovernanceConfigurationError): capture_task_receipt("two", "receipt-two")


@pytest.mark.parametrize("kwargs", [
    {"input_receipts": ["x"] * 33},
    {"memory_inputs": [memory_reference(str(i), b"x", 1) for i in range(17)]},
    {"memory_inputs": [memory_reference("same", b"x", 1)] * 2},
])
def test_task_bounds_fail_before_network(kwargs):
    with pytest.raises(GovernanceConfigurationError):
        with task_scope("signed", **kwargs): pass


def test_commit_task_memory_hashes_actual_bytes_and_uses_bound_workload_header(shield_factory):
    requests = []

    def handler(request):
        if request.url.path.endswith("credential-info"):
            return httpx.Response(200, json={"agent_subject": "agent:test"})
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["X-Agent-Token"] == "workload-token"
        assert request.url.path == "/api/agentic-new/tasks/task-one/memory"
        return httpx.Response(200, json={"memory": {
            "task_id": "task-one", "receipt_id": body["receipt_id"], "slot": body["slot"],
            "digest": body["digest"], "version": body["expected_version"] + 1,
            "agent": "agent:test", "updated_at": "2026-10-03T00:00:00Z",
        }})

    engine = _engine(shield_factory, handler)
    with task_scope("signed"):
        capture_task_receipt("task-one", "writer-receipt")
        memory = commit_task_memory(engine, "notes", b"application secret", 0)
        assert memory["version"] == 1
        assert current_task_context().memory_inputs == ()
    assert requests[0]["digest"] == memory_reference("notes", b"application secret", 1).digest
    assert "application secret" not in json.dumps(requests)


def test_commit_requires_writer_receipt_and_rejects_wrong_response(shield_factory):
    engine = _engine(shield_factory, lambda request: httpx.Response(200, json={"agent_subject": "agent:test", "memory": {}}))
    with task_scope("signed"):
        with pytest.raises(GovernanceConfigurationError, match="agent_task_receipt_required"):
            commit_task_memory(engine, "notes", b"value", 0)
        capture_task_receipt("task", "writer")
        with pytest.raises(DeepIntShieldError, match="invalid_gateway_response"):
            commit_task_memory(engine, "notes", b"value", 0)
