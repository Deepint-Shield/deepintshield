"""Real native GenAI AFC dispatch with mocked model and governance HTTP."""

import json

import httpx
import pytest

pytest.importorskip("google.genai")
from google.genai import types

from deepintshield.errors import get_exception_error_code


def _model_response(request, parts):
    body = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}
    if request.url.path.endswith(":streamGenerateContent"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(body)}\n\n")
    return httpx.Response(200, json=body)


def _clients(shield_factory, monkeypatch, outcome):
    from deepintshield.agentic import execution

    # Dispatch/receipt telemetry has its own contract tests. Keep this test's
    # request evidence focused on code capture and the authorization boundary.
    monkeypatch.setattr(execution, "_dispatch", lambda *_args: False)
    manifests, decisions, model_requests = [], [], []

    def governance(request):
        path = request.url.path
        if path.endswith("/runtime-config"):
            return httpx.Response(200, json={"available": True, "security_enabled": outcome != "disabled", "plan": "business", "reason": "workspace_disabled" if outcome == "disabled" else "enabled"})
        if path.endswith("/credential-info"):
            return httpx.Response(200, json={"agent_configured": False})
        if path.endswith("/registry/discover"):
            manifests.append(json.loads(request.content))
            if outcome == "coverage":
                return httpx.Response(400, json={"error": {"code": "blueprint_coverage_incomplete"}})
            return httpx.Response(200, json={"blueprint_scan_status": "complete"})
        if path.endswith("/agentic-new/decide"):
            decisions.append(json.loads(request.content))
            if outcome == "outage":
                return httpx.Response(503, json={"error": {"code": "gateway_unavailable"}})
            return httpx.Response(200, json={
                "verdict": "REQUIRE_APPROVAL" if outcome == "approval" else "DENY" if outcome == "deny" else "ALLOW",
                "allow": outcome == "allow", "proceed": outcome == "allow",
                "reason": "context_deny" if outcome == "deny" else "",
                "decision_id": "decision-1", "approval_id": "approval-1", "mode": "enforce",
            })
        return httpx.Response(404)

    def model(request):
        if outcome != "disabled":
            assert manifests, "source must be captured before the first model request"
        model_requests.append(request)
        if len(model_requests) == 1:
            return _model_response(request, [{"functionCall": {"name": "add", "args": {"a": 2, "b": 3}}}])
        return _model_response(request, [{"text": "5"}])

    shield = shield_factory(governance, runtime_config_handler=governance, base_url="https://gateway.invalid")
    return shield, model, manifests, decisions, model_requests


def _assert_outcome(outcome, calls, manifests, decisions, model_requests):
    if outcome in {"allow", "disabled"}:
        assert calls == [(2, 3)]
        assert len(model_requests) == 2
    else:
        assert calls == []
        # A denied/pending call is not swallowed and sent to the model as an
        # ordinary function error, and no additional model turn is generated.
        assert len(model_requests) == (0 if outcome == "coverage" else 1)
    if outcome == "disabled":
        assert manifests == decisions == []
    else:
        assert manifests
        assert manifests[0].get("code_artifacts_incomplete") is not True
        assert any("def add" in artifact["source"] for artifact in manifests[0]["code_artifacts"])
        assert all("_AFCGovernanceStop" not in artifact["source"] for artifact in manifests[0]["code_artifacts"])
        assert bool(decisions) == (outcome != "coverage")


_ERRORS = {"deny": "context_deny", "approval": "require_approval", "coverage": "blueprint_coverage_incomplete", "outage": "gateway_unavailable"}


@pytest.mark.parametrize("outcome", ["allow", "disabled", "deny", "approval", "coverage", "outage"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chat", [False, True])
def test_native_genai_sync_tool_governance(shield_factory, monkeypatch, outcome, stream, chat):
    calls = []

    def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    shield, handler, manifests, decisions, requests = _clients(shield_factory, monkeypatch, outcome)
    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_client=transport))
        config = types.GenerateContentConfig(tools=[add])

        def run():
            if chat:
                session = client.chats.create(model="gemini-test", config=config)
                response = session.send_message_stream("add") if stream else session.send_message("add")
            else:
                generate = client.models.generate_content_stream if stream else client.models.generate_content
                response = generate(model="gemini-test", contents="add", config=config)
            return "".join(item.text or "" for item in response) if stream else response.text

        try:
            if outcome in _ERRORS:
                with pytest.raises(Exception) as caught:
                    run()
                assert get_exception_error_code(caught.value) == _ERRORS[outcome]
            else:
                assert run() == "5"
            assert config.tools == [add]
            _assert_outcome(outcome, calls, manifests, decisions, requests)
        finally:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["allow", "disabled", "deny", "approval", "coverage", "outage"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chat", [False, True])
async def test_native_genai_async_tool_governance(shield_factory, monkeypatch, outcome, stream, chat):
    calls = []

    async def add(a: int, b: int) -> int:
        """Add two integers."""
        calls.append((a, b))
        return a + b

    shield, handler, manifests, decisions, requests = _clients(shield_factory, monkeypatch, outcome)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = shield.genai(http_options=types.HttpOptions(httpx_async_client=transport))
        config = {"tools": [add]}

        async def run():
            if chat:
                session = client.aio.chats.create(model="gemini-test", config=config)
                response = await (session.send_message_stream("add") if stream else session.send_message("add"))
            else:
                generate = client.aio.models.generate_content_stream if stream else client.aio.models.generate_content
                response = await generate(model="gemini-test", contents="add", config=config)
            return "".join([item.text or "" async for item in response]) if stream else response.text

        try:
            if outcome in _ERRORS:
                with pytest.raises(Exception) as caught:
                    await run()
                assert get_exception_error_code(caught.value) == _ERRORS[outcome]
            else:
                assert await run() == "5"
            assert config["tools"] == [add]
            _assert_outcome(outcome, calls, manifests, decisions, requests)
        finally:
            await client.aio.aclose()
            client.close()


@pytest.mark.asyncio
async def test_native_genai_stream_close_releases_underlying_response():
    from deepintshield.agentic.integrations.google_genai import _guard_generation

    closed = []

    async def response():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    async def generate(**_kwargs):
        return response()

    guarded = _guard_generation(generate, object(), asynchronous=True, stream=True)
    stream = await guarded()
    assert await anext(stream) == "first"
    await stream.aclose()
    assert closed == [True]
