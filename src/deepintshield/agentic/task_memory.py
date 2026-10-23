"""Explicit authenticated commitments for application-owned memory bytes."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from .errors import AGENT_INVALID_GATEWAY_RESPONSE, DeepIntShieldError, GatewayUnavailable, GovernanceConfigurationError
from .runtime import security_enabled
from .task_context import current_task_context, memory_reference


def commit_task_memory(
    engine, slot: str, value: bytes, expected_version: int,
    *, receipt_id: str = "", task_id: str = "",
) -> dict:
    """Commit a digest, never content or a claim that a tool completed safely."""
    current = current_task_context()
    if current is None or type(expected_version) is not int or expected_version < 0:
        raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_context_invalid")
    security_enabled(engine)
    reference = memory_reference(slot, value, expected_version + 1)
    selected_task = task_id or current.task_id
    receipt = receipt_id or current.last_receipt
    if not selected_task or not receipt:
        raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_receipt_required")
    if current.task_id and selected_task != current.task_id:
        raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_context_invalid")
    path = "/api/agentic-new/tasks/" + quote(selected_task, safe="") + "/memory"
    try:
        response = engine._http.post(
            engine.gateway_url + path,
            headers=engine._headers(include_agent_token=True, path=path),
            json={
                "agent": engine.agent_subject, "task_mandate": current.mandate,
                "receipt_id": receipt, "slot": slot, "digest": reference.digest,
                "expected_version": expected_version,
            },
        )
        if response.status_code >= 400:
            engine._raise_agentic_http_error(response)
        memory = response.json().get("memory")
        if (not isinstance(memory, dict) or memory.get("task_id") != selected_task
                or memory.get("receipt_id") != receipt or memory.get("slot") != slot
                or type(memory.get("version")) is not int or memory.get("version") != reference.version
                or memory.get("digest") != reference.digest):
            raise DeepIntShieldError(code=AGENT_INVALID_GATEWAY_RESPONSE)
        return memory
    except DeepIntShieldError:
        raise
    except httpx.HTTPError as exc:
        raise GatewayUnavailable(reason=type(exc).__name__) from None
    except Exception as exc:
        raise DeepIntShieldError(type(exc).__name__, code=AGENT_INVALID_GATEWAY_RESPONSE) from None
