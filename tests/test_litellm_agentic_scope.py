"""Native inference stays native; application tools own Agentic authorization."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager

import httpx
import pytest

from .test_model_compatibility import native_response


MODEL = "custom-provider/deployment-model"
MESSAGES = [{"role": "user", "content": "Write a short report"}]
TOOL = "reports.generate"


def generate_report(client, *, stream=False):
    """An application tool may call a model as part of its implementation."""
    response = client.completion(model=MODEL, messages=MESSAGES, stream=stream)
    if stream:
        return "".join(chunk.choices[0].delta.content or "" for chunk in response)
    return response.choices[0].message.content


async def generate_report_async(client, *, stream=False):
    """Keep the application's async tool boundary inspectable for discovery."""
    response = await client.acompletion(model=MODEL, messages=MESSAGES, stream=stream)
    if stream:
        return "".join([
            chunk.choices[0].delta.content or "" async for chunk in response
        ])
    return response.choices[0].message.content


@pytest.fixture
def native_litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    native = pytest.importorskip("litellm")
    from litellm.llms.openai.openai import OpenAIChatCompletion

    native.in_memory_llm_clients_cache.flush_cache()
    return OpenAIChatCompletion


@contextmanager
def run_scope(shield, kind, session_id):
    if kind == "run":
        with shield.agentic.run(session_id=session_id):
            yield
    elif kind == "legacy":
        shield.agentic.start_run(session_id)
        try:
            yield
        finally:
            shield.agentic.end_run()
    else:
        yield


def task_rendezvous(parties):
    ready = asyncio.Event()
    arrived = 0

    async def wait():
        nonlocal arrived
        arrived += 1
        if arrived == parties:
            ready.set()
        await ready.wait()

    return wait


class Gateway:
    def __init__(self, outcome="allow"):
        self.outcome = outcome
        self.requests = []
        self.models = []
        self.decisions = []
        self.discoveries = []
        self.events = []
        self.condition = threading.Condition()

    def model(self, request):
        self.models.append(request)
        body = json.loads(request.content)
        return native_response("chat", body["model"], body.get("stream", False))

    def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/runtime-config"):
            if self.outcome == "runtime-error":
                return httpx.Response(503)
            return httpx.Response(200, json={
                "security_enabled": True, "available": True,
                "reason": "enabled", "plan": "business",
            })
        if path.endswith(("/vk-credential-info", "/agentic-new/credential-info")):
            return httpx.Response(200, json={
                "provider_type": "", "agent_configured": False,
                "agent_subject": request.headers.get("X-Agent-Subject", ""),
            })
        if path.endswith("/agentic-new/registry/discover"):
            self.discoveries.append(json.loads(request.content))
            if self.outcome == "blueprint-error":
                return httpx.Response(503)
            return httpx.Response(200, json={"blueprint_scan_status": "complete"})
        if path.endswith("/agentic-new/executions"):
            body = json.loads(request.content)
            with self.condition:
                self.events.append(body)
                self.condition.notify_all()
            return httpx.Response(201, json={"created": True})
        if path.endswith("/agentic-new/decide"):
            self.decisions.append(json.loads(request.content))
            if self.outcome == "pdp-error":
                return httpx.Response(503)
            if self.outcome == "invalid-decision":
                return httpx.Response(200, json={"unexpected": True})
            verdict = {
                "allow": "ALLOW", "deny": "DENY", "approval": "REQUIRE_APPROVAL",
            }[self.outcome]
            return httpx.Response(200, json={
                "verdict": verdict,
                "allow": verdict == "ALLOW", "proceed": verdict == "ALLOW",
                "mode": "enforce", "reason": "allowed" if verdict == "ALLOW" else "tool_not_allowed",
                "approval_id": "approval-report" if verdict == "REQUIRE_APPROVAL" else "",
            })
        return httpx.Response(404)

    def wait_for_terminal(self, session_id):
        with self.condition:
            assert self.condition.wait_for(
                lambda: any(
                    event["session_id"] == session_id and event["event"] != "start"
                    for event in self.events
                ),
                timeout=5,
            ), "application execution telemetry did not finish"


def make_shield(shield_factory, gateway, index=0):
    return shield_factory(
        gateway.handle,
        runtime_config_handler=gateway.handle,
        base_url=f"https://gateway-{index}.invalid",
        agent_name=f"report-agent-{index}",
    )


def assert_native_request(request, index, stream):
    assert str(request.url) == f"https://gateway-{index}.invalid/litellm/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-ds-test"
    assert request.headers["x-deepintshield-vk"] == "sk-ds-test"
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["messages"] == MESSAGES
    assert body.get("stream", False) is stream


@pytest.mark.parametrize("scope", ["plain", "run", "legacy"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("client_count", [1, 2])
def test_native_sync_inference_does_not_enter_agentic(
    shield_factory, monkeypatch, native_litellm, scope, stream, client_count,
):
    gateway = Gateway()
    clients = [make_shield(shield_factory, gateway, index) for index in range(client_count)]
    with httpx.Client(transport=httpx.MockTransport(gateway.model)) as transport:
        monkeypatch.setattr(native_litellm, "_get_sync_http_client", staticmethod(lambda: transport))
        for index, shield in enumerate(clients):
            with run_scope(shield, scope, f"native-{index}"):
                assert generate_report(shield.litellm(), stream=stream) == "OK"
            if scope != "plain":
                assert not shield.agentic.engine.bound_session_id

    assert gateway.requests == []
    assert len(gateway.models) == client_count
    for index, request in enumerate(gateway.models):
        assert_native_request(request, index, stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["plain", "run", "legacy"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("client_count", [1, 2])
async def test_native_async_inference_does_not_enter_agentic(
    shield_factory, monkeypatch, native_litellm, scope, stream, client_count,
):
    gateway = Gateway()
    clients = [make_shield(shield_factory, gateway, index) for index in range(client_count)]
    wait_until_ready = task_rendezvous(client_count)

    async def invoke(index, shield):
        with run_scope(shield, scope, f"native-{index}"):
            await wait_until_ready()
            assert await generate_report_async(shield.litellm(), stream=stream) == "OK"
            if scope != "plain":
                assert shield.agentic.engine.bound_session_id == f"native-{index}"
        if scope != "plain":
            assert not shield.agentic.engine.bound_session_id

    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway.model)) as transport:
        monkeypatch.setattr(native_litellm, "_get_async_http_client", staticmethod(lambda **kwargs: transport))
        await asyncio.gather(*(invoke(index, shield) for index, shield in enumerate(clients)))

    assert gateway.requests == []
    assert len(gateway.models) == client_count
    for index, request in enumerate(sorted(gateway.models, key=lambda request: request.url.host)):
        assert_native_request(request, index, stream)


OUTCOMES = [
    ("allow", None),
    ("deny", PermissionError),
    ("approval", RuntimeError),
    ("pdp-error", ConnectionError),
    ("invalid-decision", ConnectionError),
    ("blueprint-error", RuntimeError),
    ("runtime-error", ConnectionError),
]


def assert_application_boundary(gateway, outcome, source_name, stream):
    assert len(gateway.models) == (1 if outcome == "allow" else 0)
    if gateway.models:
        assert_native_request(gateway.models[0], 0, stream)
    if outcome == "runtime-error":
        assert gateway.discoveries == []
        assert gateway.decisions == []
        return
    assert len(gateway.discoveries) == 1
    manifest = gateway.discoveries[0]
    assert [tool["key"] for tool in manifest["tools"]] == [TOOL]
    assert not manifest.get("code_artifacts_incomplete")
    assert any(
        artifact["subject_key"] == TOOL and f"def {source_name}(" in artifact["source"]
        for artifact in manifest["code_artifacts"]
    )
    if outcome == "blueprint-error":
        assert gateway.decisions == []
        return
    assert len(gateway.decisions) == 1
    assert gateway.decisions[0]["tool"] == f"tool:{TOOL}"
    assert gateway.decisions[0]["session_id"] == "report-run"
    gateway.wait_for_terminal("report-run")


@pytest.mark.parametrize("outcome,error_type", OUTCOMES)
@pytest.mark.parametrize("stream", [False, True])
def test_application_sync_tool_gates_before_native_inference(
    shield_factory, monkeypatch, native_litellm, outcome, error_type, stream,
):
    gateway = Gateway(outcome)
    shield = make_shield(shield_factory, gateway)
    tool = shield.agentic.tool(TOOL)(generate_report)
    with httpx.Client(transport=httpx.MockTransport(gateway.model)) as transport:
        monkeypatch.setattr(native_litellm, "_get_sync_http_client", staticmethod(lambda: transport))
        with shield.agentic.run(session_id="report-run"):
            if error_type is None:
                assert tool(shield.litellm(), stream=stream) == "OK"
            else:
                with pytest.raises(error_type) as caught:
                    tool(shield.litellm(), stream=stream)
                assert caught.value._deepintshield_error_code
    assert_application_boundary(gateway, outcome, "generate_report", stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,error_type", OUTCOMES)
@pytest.mark.parametrize("stream", [False, True])
async def test_application_async_tool_gates_before_native_inference(
    shield_factory, monkeypatch, native_litellm, outcome, error_type, stream,
):
    gateway = Gateway(outcome)
    shield = make_shield(shield_factory, gateway)
    tool = shield.agentic.tool(TOOL)(generate_report_async)
    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway.model)) as transport:
        monkeypatch.setattr(native_litellm, "_get_async_http_client", staticmethod(lambda **kwargs: transport))
        with shield.agentic.run(session_id="report-run"):
            if error_type is None:
                assert await tool(shield.litellm(), stream=stream) == "OK"
            else:
                with pytest.raises(error_type) as caught:
                    await tool(shield.litellm(), stream=stream)
                assert caught.value._deepintshield_error_code
    assert_application_boundary(gateway, outcome, "generate_report_async", stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["run", "legacy"])
async def test_concurrent_tools_keep_their_client_and_session(
    shield_factory, monkeypatch, native_litellm, scope,
):
    gateways = [Gateway("allow"), Gateway("deny")]
    shields = [make_shield(shield_factory, gateway, index) for index, gateway in enumerate(gateways)]
    tools = [shield.agentic.tool(TOOL)(generate_report_async) for shield in shields]
    wait_until_ready = task_rendezvous(2)

    async def invoke(index):
        shield = shields[index]
        with run_scope(shield, scope, f"isolated-{index}"):
            await wait_until_ready()
            if index == 0:
                assert await tools[index](shield.litellm(), stream=True) == "OK"
            else:
                with pytest.raises(PermissionError):
                    await tools[index](shield.litellm(), stream=True)
            assert shield.agentic.engine.bound_session_id == f"isolated-{index}"
        assert not shield.agentic.engine.bound_session_id

    def model(request):
        index = 0 if request.url.host == "gateway-0.invalid" else 1
        return gateways[index].model(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as transport:
        monkeypatch.setattr(native_litellm, "_get_async_http_client", staticmethod(lambda **kwargs: transport))
        await asyncio.gather(invoke(0), invoke(1))

    assert len(gateways[0].models) == 1
    assert gateways[1].models == []
    for index, gateway in enumerate(gateways):
        assert len(gateway.decisions) == 1
        assert gateway.decisions[0]["tool"] == f"tool:{TOOL}"
        assert gateway.decisions[0]["session_id"] == f"isolated-{index}"
        assert gateway.decisions[0]["agent"] == f"agent:report-agent-{index}"
        gateway.wait_for_terminal(f"isolated-{index}")
