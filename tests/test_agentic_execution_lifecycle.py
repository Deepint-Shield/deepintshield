from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import httpx
import pytest

from deepintshield.agentic import enforcement
langgraph = pytest.importorskip("langgraph")


class LifecycleRecorder:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.decisions: list[dict] = []
        self._condition = threading.Condition()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/vk-credential-info"):
            return httpx.Response(
                200,
                json={
                    "provider_type": "",
                    "agent_configured": False,
                    "agent_subject": "agent:lifecycle-test",
                },
            )
        if path.endswith("/agentic-new/registry/discover"):
            return httpx.Response(
                200, json={"network_id": "network-test", "agents_upserted": 1}
            )
        if path.endswith("/agentic-new/executions"):
            body = json.loads(request.content)
            with self._condition:
                self.events.append(body)
                self._condition.notify_all()
            return httpx.Response(
                201,
                json={
                    "execution": {
                        "execution_id": body["execution_id"],
                        "status": body.get("event", "running"),
                    },
                    "created": body.get("event") == "start",
                },
            )
        if path.endswith("/agentic-new/decide"):
            body = json.loads(request.content)
            with self._condition:
                self.decisions.append(body)
                self._condition.notify_all()
            return httpx.Response(
                200,
                headers={"x-request-id": f"d-{len(self.decisions)}"},
                json={
                    "verdict": "ALLOW",
                    "allow": True,
                    "reason": "allowed",
                    "mode": "enforce",
                    "proceed": True,
                },
            )
        # A canonical result is the sole runtime authorization verdict.
        return httpx.Response(404)

    def wait_for_events(self, count: int, timeout: float = 5.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.events) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return list(self.events)


def _graph(*, fail: bool = False):
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    def increment(state: State) -> State:
        if fail:
            raise RuntimeError("raw state must never enter execution telemetry")
        return {"value": state["value"] + 1}

    graph = StateGraph(State)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph.compile()


def _terminal_by_id(events: list[dict]) -> dict[str, dict]:
    return {
        event["execution_id"]: event
        for event in events
        if event.get("event") in {"complete", "failed", "blocked", "cancelled"}
    }


def test_plain_langgraph_invocations_get_fresh_correlated_executions(
    shield_factory,
):
    recorder = LifecycleRecorder()
    shield_factory(recorder.handler)
    app = _graph()

    assert app.invoke({"value": 1}) == {"value": 2}
    assert app.invoke({"value": 4}) == {"value": 5}

    events = recorder.wait_for_events(4)
    starts = [event for event in events if event.get("event") == "start"]
    terminals = _terminal_by_id(events)
    assert len(starts) == 2
    assert len({event["execution_id"] for event in starts}) == 2
    assert set(terminals) == {event["execution_id"] for event in starts}
    assert {event["event"] for event in terminals.values()} == {"complete"}

    assert len(recorder.decisions) == 2
    assert {
        (decision["execution_id"], decision["session_id"])
        for decision in recorder.decisions
    } == {
        (event["execution_id"], event["execution_id"])
        for event in starts
    }
    assert all(event["framework"] == "langgraph" for event in starts)
    assert all(event["topology"]["nodes"] for event in starts)
    # Terminal events intentionally omit the immutable topology snapshot.
    assert all("topology" not in event for event in terminals.values())


def test_langgraph_failure_is_terminal_and_zero_data_retention_safe(
    shield_factory,
):
    recorder = LifecycleRecorder()
    shield_factory(recorder.handler)
    app = _graph(fail=True)

    with pytest.raises(RuntimeError):
        app.invoke({"value": 7})

    events = recorder.wait_for_events(2)
    failed = next(event for event in events if event.get("event") == "failed")
    assert failed["error_code"] == "RuntimeError"
    assert failed["error_summary"] == "workflow raised RuntimeError"
    assert "raw state" not in json.dumps(events)
    assert recorder.decisions[0]["execution_id"] == failed["execution_id"]


def test_agentic_failure_telemetry_contains_only_the_policy_code(shield_factory):
    class DeniedRecorder(LifecycleRecorder):
        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/agentic-new/decide"):
                body = json.loads(request.content)
                with self._condition:
                    self.decisions.append(body)
                    self._condition.notify_all()
                return httpx.Response(
                    200,
                    json={
                        "verdict": "DENY",
                        "allow": False,
                        "reason": "obo_tool_not_allowed",
                        "mode": "enforce",
                        "proceed": False,
                    },
                )
            return super().handler(request)

    recorder = DeniedRecorder()
    shield_factory(recorder.handler)
    app = _graph()

    with pytest.raises(PermissionError) as caught:
        app.invoke({"value": 7})

    assert caught.value._deepintshield_error_code == "obo_tool_not_allowed"
    events = recorder.wait_for_events(2)
    blocked = next(event for event in events if event.get("event") == "blocked")
    assert blocked["error_code"] == "obo_tool_not_allowed"
    assert "error_summary" not in blocked


def test_langgraph_stream_and_async_entry_points_keep_scope_until_consumed(
    shield_factory,
):
    recorder = LifecycleRecorder()
    shield_factory(recorder.handler)
    app = _graph()

    assert list(app.stream({"value": 10}))

    async def run_async() -> None:
        assert await app.ainvoke({"value": 20}) == {"value": 21}
        streamed = []
        async for item in app.astream({"value": 30}):
            streamed.append(item)
        assert streamed

    asyncio.run(run_async())

    events = recorder.wait_for_events(6)
    starts = [event for event in events if event.get("event") == "start"]
    assert len(starts) == 3
    assert len({event["execution_id"] for event in starts}) == 3
    assert set(_terminal_by_id(events)) == {
        event["execution_id"] for event in starts
    }
    assert len({decision["execution_id"] for decision in recorder.decisions}) == 3


def test_concurrent_invocations_do_not_leak_session_context(shield_factory):
    recorder = LifecycleRecorder()
    shield = shield_factory(recorder.handler)
    app = _graph()
    default_session = shield.agentic.engine.session_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda value: app.invoke({"value": value}), range(16)))

    assert results == [{"value": value + 1} for value in range(16)]
    events = recorder.wait_for_events(32, timeout=8.0)
    starts = [event for event in events if event.get("event") == "start"]
    assert len(starts) == 16
    execution_ids = {event["execution_id"] for event in starts}
    assert len(execution_ids) == 16
    assert {decision["execution_id"] for decision in recorder.decisions} == execution_ids
    assert all(
        decision["execution_id"] == decision["session_id"]
        for decision in recorder.decisions
    )
    assert shield.agentic.engine.session_id == default_session


def test_shared_framework_boundary_preserves_explicit_run_id(shield_factory):
    recorder = LifecycleRecorder()
    shield = shield_factory(recorder.handler)

    @shield.agentic.tool("crew_task")
    def crew_task() -> str:
        return shield.agentic.engine.session_id

    class Crew:
        def __init__(self) -> None:
            self.agents = []
            self.tasks = []

        def kickoff(self) -> str:
            return crew_task()

    assert enforcement.report_topology_on(Crew, ("kickoff",))
    crew = Crew()
    enforcement.bind_engine(crew, shield.agentic.engine, recursive=True)

    with shield.agentic.run(session_id="caller-run-42"):
        assert crew.kickoff() == "caller-run-42"

    events = recorder.wait_for_events(2)
    assert [event["execution_id"] for event in events] == [
        "caller-run-42",
        "caller-run-42",
    ]
    assert [event["event"] for event in events] == ["start", "complete"]
    assert recorder.decisions[0]["execution_id"] == "caller-run-42"
    assert recorder.decisions[0]["session_id"] == "caller-run-42"
