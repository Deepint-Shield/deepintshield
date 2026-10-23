"""Per-request gateway authentication for the native MCP transport."""

from __future__ import annotations

import asyncio
import json

import httpx

from ..agentic.errors import GovernanceConfigurationError
from ..agentic.task_context import task_payload


def gateway_task_headers(shield, headers: dict[str, str]) -> dict[str, str]:
    """Snapshot one task per MCP session, after authenticated capability discovery."""
    headers = dict(headers)
    payload = task_payload()
    existing = {key.lower(): value for key, value in headers.items()}
    mandate = payload.get("task_mandate") or existing.get("x-deepintshield-task-mandate") or existing.get("x-ds-task-mandate")
    if not mandate:
        if any(existing.get(name) for name in (
            "x-deepintshield-input-receipts", "x-ds-input-receipts",
            "x-deepintshield-memory-inputs", "x-ds-memory-inputs",
        )):
            raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_context_invalid")
        return headers
    # MCP executes remotely; discover support BEFORE sending any tool call.
    shield.agentic.engine.fetch_security_enabled(require_tasks=True)
    if payload:
        for key in list(headers):
            canonical = key.lower()
            if canonical.startswith("x-ds-"):
                canonical = "x-deepintshield-" + canonical[len("x-ds-"):]
            if canonical in {"x-deepintshield-task-mandate", "x-deepintshield-input-receipts", "x-deepintshield-memory-inputs"}:
                del headers[key]
        headers.update({
            "X-Deepintshield-Task-Mandate": payload["task_mandate"],
            "X-Deepintshield-Input-Receipts": json.dumps(payload["input_receipts"], separators=(",", ":")),
            "X-Deepintshield-Memory-Inputs": json.dumps(payload["memory_inputs"], separators=(",", ":")),
        })
    return headers


def gateway_workload_hook(
    shield,
    url: str,
    *,
    identity: bool,
    headers: dict[str, str],
    failures: list[BaseException] | None = None,
):
    """Issue fresh DPoP on the configured gateway URL only, never a redirect.

    The hook runs inside the MCP transport's own task group.  When it raises,
    the transport cancels the request that was waiting for the response, and
    the caller sees a bare cancellation instead of the reason - an agent that
    is not registered read as ``Cancelled via cancel scope``.  ``failures``
    keeps the reason so the session can raise it in the cancellation's place.
    """
    target = httpx.URL(url)
    origin = (target.scheme, target.host, target.port)
    explicit_token = next((value for key, value in headers.items() if key.lower() == "x-agent-token"), "")
    task_present = any(key.lower() in {"x-deepintshield-task-mandate", "x-ds-task-mandate"} for key in headers)

    async def authenticate(request: httpx.Request) -> None:
        try:
            if (request.url.scheme, request.url.host, request.url.port) != origin or request.url.path != target.path:
                raise GovernanceConfigurationError(framework="mcp", code="agent_workload_destination_invalid")
            if not (identity or task_present or explicit_token):
                return
            proof_headers = await asyncio.to_thread(
                shield.agentic.engine.workload_headers, request.method, str(request.url), token=explicit_token,
            )
            request.headers.update(proof_headers)
        except Exception as error:
            if failures is not None:
                failures.append(error)
            raise

    return authenticate
