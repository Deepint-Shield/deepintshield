"""Replay actual Go provider-converter events with native OpenAI stream helpers.

These synthetic SSE fixtures were exported by the Anthropic stream snapshot
tests, TestGeminiResponsesNativeSDKFixtures, and TestMuxNativeSDKFixtures through
the actual converters. They contain no provider credentials or user inputs.
Keep their native event ordering and terminal snapshots when updating fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .native_sdk_helpers import openai_backend
from .test_multimodal_examples import _load_example


FIXTURES = Path(__file__).with_name("fixtures") / "native_responses"


@pytest.mark.parametrize("fixture,provider,text,types", [
    ("anthropic-mixed.sse", "anthropic", "Blue. Done.", ["reasoning", "message", "function_call", "message"]),
    ("anthropic-mcp.sse", "anthropic", "Blue.", ["mcp_call", "mcp_call", "message"]),
    ("gemini-text.sse", "gemini", "Blue", ["message"]),
    ("gemini-mixed.sse", "gemini", "FirstBlue", ["message", "reasoning", "function_call", "reasoning", "message"]),
    ("mux-text.sse", "openai", "Blue", ["message"]),
    ("mux-reasoning.sse", "openai", "Blue", ["reasoning", "message"]),
    ("mux-tools.sse", "openai", "", ["function_call", "function_call"]),
    ("mux-mixed.sse", "openai", "FirstBlue", ["message", "reasoning", "function_call", "function_call", "message"]),
    ("mux-refusal.sse", "openai", "", ["message"]),
])
@pytest.mark.parametrize("consume_events_first", [False, True])
def test_real_converter_events_support_native_final_response_helper(fixture, provider, text, types, consume_events_first):
    sdk, http = openai_backend()
    wire = (FIXTURES / fixture).read_bytes()
    seen = []
    def handler(request):
        seen.append(request)
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=wire)
    with http.Client(transport=http.MockTransport(handler)) as transport:
        with sdk.OpenAI(api_key="sk-offline-native-stream-fixture", base_url="https://gateway.invalid/openai",
                        http_client=transport, max_retries=0) as client:
            with client.responses.stream(model=provider + "/synthetic-model", input="Read the synthetic color fixture.") as stream:
                if consume_events_first:
                    events = []
                    for event in stream:
                        if event.type == "response.created":
                            assert event.response.object == "response"
                            assert event.response.status == "in_progress"
                            assert event.response.output == []
                        events.append(event)
                    assert events[0].type == "response.created" and events[-1].type == "response.completed"
                final = stream.get_final_response()
                assert stream.get_final_response() is final
                assert final.status == "completed" and final.error is None
                assert final.object == "response"
                assert final.output_text == text
                assert [item.type for item in final.output] == types
                assert [item.status for item in final.output] == (["completed", "failed", "completed"] if fixture == "anthropic-mcp.sse" else ["completed"] * len(types))
                outputs = [item.model_dump(mode="json") for item in final.output]
                assert len({item["id"] for item in outputs}) == len(outputs)
                if fixture == "anthropic-mcp.sse":
                    assert [item["id"] for item in outputs[:2]] == ["mcp_first", "mcp_second"]
                    assert [json.loads(item["arguments"]) for item in outputs[:2]] == [{"order": 1}, {"order": 2}]
                    assert json.loads(outputs[0]["output"]) == [
                        {"type": "text", "text": "blue"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "synthetic-only"}},
                    ]
                    assert outputs[1]["output"] == "synthetic tool failure"
                    assert outputs[1]["error"] == "MCP tool returned an error"
                elif fixture == "anthropic-mixed.sse":
                    call = outputs[2]
                    assert call["name"] == "record_color" and call["call_id"] == "call_synthetic_color"
                    assert json.loads(call["arguments"]) == {"color": "blue"}
                    assert outputs[0]["summary"][0]["text"] == "Check the synthetic image."
                    assert outputs[0]["content"][0]["signature"] == "synthetic-signature-only"
                elif fixture == "gemini-mixed.sse":
                    call = outputs[2]
                    assert call["name"] == "lookup_color" and call["call_id"] == "call-synthetic_ts_dG9vbC1zaWduYXR1cmU"
                    assert json.loads(call["arguments"]) == {"value": 1}
                    assert outputs[1]["summary"][0]["text"] == "Check the color"
                    assert outputs[1]["encrypted_content"] == "cmVhc29uaW5nLXNpZ25hdHVyZQ=="
                    assert outputs[3]["encrypted_content"] == "b3BhcXVlLXNpZ25hdHVyZQ=="
                elif fixture == "gemini-text.sse":
                    assert outputs[0]["content"][0]["signature"] == "dGV4dC1zaWduYXR1cmU="
                elif fixture in {"mux-tools.sse", "mux-mixed.sse"}:
                    calls = [item for item in outputs if item["type"] == "function_call"]
                    assert [item["name"] for item in calls] == ["lookup_color", "count_colors"]
                    assert [item["call_id"] for item in calls] == ["call-color", "call-count"]
                    assert [json.loads(item["arguments"]) for item in calls] == [{"color": "blue"}, {"count": 1}]
                elif fixture == "mux-refusal.sse":
                    assert outputs[0]["content"] == [{"type": "refusal", "refusal": "I cannot help with that."}]
                if fixture in {"mux-reasoning.sse", "mux-mixed.sse"}:
                    reasoning = next(item for item in outputs if item["type"] == "reasoning")
                    assert reasoning["summary"] == [{"type": "summary_text", "text": "Check the color"}]
    assert len(seen) == 1 and seen[0].url.path == "/openai/responses"
    assert json.loads(seen[0].content)["stream"] is True


@pytest.mark.parametrize("fixture", [
    "gemini-incomplete.sse", "gemini-failed.sse", "gemini-interrupted.sse",
    "mux-incomplete.sse", "mux-interrupted.sse", "mux-filtered.sse", "mux-failed.sse",
])
@pytest.mark.parametrize("consumer", ["native_final_helper", "example"])
def test_actual_converter_failure_after_text_is_never_success(fixture, consumer):
    sdk, http = openai_backend()
    wire = (FIXTURES / fixture).read_bytes()
    seen = []
    def handler(request):
        seen.append(request)
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=wire)
    with http.Client(transport=http.MockTransport(handler)) as transport:
        with sdk.OpenAI(api_key="sk-offline-native-stream-fixture", base_url="https://gateway.invalid/openai",
                        http_client=transport, max_retries=0) as client:
            with pytest.raises(RuntimeError):
                if consumer == "native_final_helper":
                    with client.responses.stream(model="gemini/synthetic-model", input="Read the synthetic color fixture.") as stream:
                        stream.get_final_response()
                else:
                    _load_example().run_example("gemini", "pdf", "synthetic-model", stream=True, client=client)
    assert len(seen) == 1


def test_actual_mux_refusal_is_preserved_as_refusal_by_example():
    sdk, http = openai_backend()
    wire = (FIXTURES / "mux-refusal.sse").read_bytes()
    seen = []
    def handler(request):
        seen.append(request)
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=wire)
    with http.Client(transport=http.MockTransport(handler)) as transport:
        with sdk.OpenAI(api_key="sk-offline-native-stream-fixture", base_url="https://gateway.invalid/openai",
                        http_client=transport, max_retries=0) as client:
            with pytest.raises(RuntimeError, match="refus"):
                _load_example().run_example("openai", "pdf", "synthetic-model", stream=True, client=client)
    assert len(seen) == 1
