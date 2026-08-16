"""Automatic, zero-wait workflow execution lifecycle.

Framework adapters enter :func:`execution_scope` at their native ``invoke`` /
``run`` / ``kickoff`` boundary.  The scope gives every top-level invocation a
fresh execution id, binds that id as the request-local agentic session, and
emits start/terminal lifecycle events to the Agentic execution ledger.

Authorization is still synchronous and fail-closed.  Lifecycle delivery is
metadata-only, best-effort telemetry and always runs on the existing bounded
background executor, so an unavailable telemetry route never delays or permits
a governed tool call.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import queue
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from .errors import (
    DeepIntShieldError,
    GuardrailApprovalPending,
    GuardrailDenied,
    normalize_agentic_error_code,
)

_EXECUTION_PATH = "/api/agentic-new/executions"
_EXECUTION_TIMEOUT_SECONDS = 3.0
_TOPOLOGY_CACHE_ATTR = "_deepintshield_execution_topology"
_EVENT_QUEUE_MAX = 8192
_EVENT_WORKERS = 4
_event_queue: "queue.Queue[tuple[contextvars.Context, Any, tuple[Any, ...]]]" = (
    queue.Queue(maxsize=_EVENT_QUEUE_MAX)
)
_workers_lock = threading.Lock()
_workers_started = False


@dataclass
class ExecutionFrame:
    """One request-local workflow invocation.

    ``start_delivered`` coordinates the two detached jobs.  The terminal job
    waits in the background (never on the application thread) until the start
    attempt has completed, preserving event order even when a workflow finishes
    immediately.
    """

    engine: Any
    execution_id: str
    session_id: str
    target: Any
    framework: str
    started_at: str
    started_monotonic_ns: int
    start_delivered: Any = field(default=None, repr=False)
    terminal_event: str = ""
    error_code: str = ""
    error_summary: str = ""

    def __post_init__(self) -> None:
        if self.start_delivered is None:
            self.start_delivered = threading.Event()


# A stack (instead of one process-global "current execution") supports nested
# calls across different SDK clients while ContextVar isolation keeps async
# tasks and copied request contexts independent.
_execution_stack: contextvars.ContextVar[tuple[ExecutionFrame, ...]] = (
    contextvars.ContextVar("deepintshield_execution_stack", default=())
)


def current_execution(engine: Any = None) -> Optional[ExecutionFrame]:
    """Return the innermost active execution, optionally for one engine."""
    stack = _execution_stack.get()
    if engine is None:
        return stack[-1] if stack else None
    for frame in reversed(stack):
        if frame.engine is engine:
            return frame
    return None


def current_execution_id(engine: Any = None) -> str:
    frame = current_execution(engine)
    return frame.execution_id if frame is not None else ""


@contextmanager
def execution_scope(
    engine: Any,
    target: Any,
    *,
    framework: str = "",
) -> Iterator[ExecutionFrame]:
    """Scope one native framework invocation to one execution/session.

    A nested native entry point (LangGraph's ``invoke`` calling ``stream``, a
    Crew invoking another governed runnable, etc.) reuses the outer frame and
    emits no duplicate lifecycle events.  An explicit
    ``with shield.agentic.run(session_id=...)`` remains authoritative: the
    framework uses that caller-supplied session rather than replacing it.
    """
    active = current_execution(engine)
    if active is not None:
        yield active
        return

    bound_session = str(getattr(engine, "bound_session_id", "") or "").strip()
    new_session = getattr(engine, "new_session_id", None)
    execution_id = bound_session or (
        str(new_session())
        if callable(new_session)
        else "sdk-" + uuid.uuid4().hex[:12]
    )
    session_token = None
    bind_session = getattr(engine, "bind_session", None)
    if not bound_session and callable(bind_session):
        session_token = bind_session(execution_id)

    cached = _cached_topology(target)
    detected_framework = str(
        framework
        or (cached or {}).get("framework")
        or _framework_name(target)
    ).strip().lower()
    frame = ExecutionFrame(
        engine=engine,
        execution_id=execution_id,
        session_id=execution_id,
        target=target,
        framework=detected_framework,
        started_at=_utc_now(),
        started_monotonic_ns=time.monotonic_ns(),
    )
    stack_token = _execution_stack.set((*_execution_stack.get(), frame))
    _emit_start(frame)
    try:
        yield frame
    except BaseException as exc:
        if frame.terminal_event:
            event = frame.terminal_event
            code = frame.error_code
            summary = frame.error_summary
        else:
            event, code, summary = _failure_metadata(exc)
        _emit_finish(frame, event=event, error_code=code, error_summary=summary)
        raise
    else:
        _emit_finish(
            frame,
            event=frame.terminal_event or "complete",
            error_code=frame.error_code,
            error_summary=frame.error_summary,
        )
    finally:
        _execution_stack.reset(stack_token)
        if session_token is not None:
            reset_session = getattr(engine, "reset_session", None)
            if callable(reset_session):
                reset_session(session_token)


def cache_topology(target: Any, topology: dict[str, Any]) -> bool:
    """Cache an already-described topology on a framework object.

    Discovery/build boundaries call this once, allowing every later execution
    to snapshot the graph without repeating reflection. Frozen framework models
    simply decline the cache and are described in the detached start worker.
    """
    if target is None or not isinstance(topology, dict):
        return False
    try:
        setattr(target, _TOPOLOGY_CACHE_ATTR, topology)
        return True
    except Exception:
        try:
            object.__setattr__(target, _TOPOLOGY_CACHE_ATTR, topology)
            return True
        except Exception:
            return False


def mark_execution_outcome(
    frame: ExecutionFrame,
    event: str,
    *,
    error_code: str = "",
    error_summary: str = "",
) -> None:
    """Set a non-exception terminal outcome for integrations that short-circuit.

    Some frameworks represent a policy denial as an ordinary tool-result value
    instead of raising. Their adapters use this hook so the execution ledger
    records ``blocked`` rather than incorrectly treating the return as success.
    """
    clean = str(event or "").strip().lower()
    if clean not in {
        "complete",
        "failed",
        "blocked",
        "cancelled",
        "awaiting_approval",
    }:
        return
    frame.terminal_event = clean
    frame.error_code = str(error_code or "")[:64]
    frame.error_summary = str(error_summary or "")[:512]


def _cached_topology(target: Any) -> Optional[dict[str, Any]]:
    if target is None:
        return None
    try:
        value = getattr(target, _TOPOLOGY_CACHE_ATTR, None)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _emit_start(frame: ExecutionFrame) -> None:
    if not _dispatch(_post_start, frame):
        # Do not make the terminal worker wait for a start job that was never
        # scheduled (e.g. bounded executor saturation during shutdown).
        frame.start_delivered.set()


def _emit_finish(
    frame: ExecutionFrame,
    *,
    event: str,
    error_code: str = "",
    error_summary: str = "",
) -> None:
    completed_at = _utc_now()
    duration_us = max(
        0, (time.monotonic_ns() - frame.started_monotonic_ns) // 1_000
    )
    _dispatch(
        _post_finish,
        frame,
        event,
        error_code,
        error_summary,
        completed_at,
        duration_us,
    )


def _dispatch(fn: Any, *args: Any) -> bool:
    """Enqueue lifecycle telemetry without waiting or spawning per-run threads.

    Execution volume is much higher than topology discovery volume, so sharing
    discovery's four-*job* cap would silently lose most events during a burst.
    This dedicated bounded queue keeps four long-lived daemon workers and
    accepts 8192 events in O(1), preserving application latency and bounding
    memory at the same time.
    """
    try:
        if sys.is_finalizing():
            return False
    except Exception:
        pass
    _ensure_workers()
    try:
        _event_queue.put_nowait((contextvars.copy_context(), fn, args))
        return True
    except queue.Full:
        return False


def _ensure_workers() -> None:
    global _workers_started
    if _workers_started:
        return
    with _workers_lock:
        if _workers_started:
            return
        for index in range(_EVENT_WORKERS):
            threading.Thread(
                target=_event_worker,
                name=f"deepintshield-execution-{index + 1}",
                daemon=True,
            ).start()
        _workers_started = True


def _event_worker() -> None:
    while True:
        context, fn, args = _event_queue.get()
        try:
            context.run(fn, *args)
        except BaseException:
            # Lifecycle is metadata telemetry. Authorization is enforced on the
            # caller thread and is never weakened by a reporting failure.
            pass
        finally:
            _event_queue.task_done()


def _post_start(frame: ExecutionFrame) -> None:
    try:
        manifest = _cached_topology(frame.target)
        if manifest is None:
            from .registry import describe_network

            manifest = describe_network(
                frame.target,
                framework=frame.framework,
            )
            cache_topology(frame.target, manifest)
        network = manifest.get("network") if isinstance(manifest, dict) else {}
        if not isinstance(network, dict):
            network = {}
        payload: dict[str, Any] = {
            "execution_id": frame.execution_id,
            "session_id": frame.session_id,
            "event": "start",
            "framework": str(
                frame.framework or manifest.get("framework") or ""
            ).strip().lower(),
            "network_key": str(network.get("key") or "").strip(),
            "started_at": frame.started_at,
            "topology": _wire_topology(manifest),
        }
        payload.update(_actor_payload(frame.engine))
        _post_event(frame.engine, payload)
    finally:
        frame.start_delivered.set()


def _post_finish(
    frame: ExecutionFrame,
    event: str,
    error_code: str,
    error_summary: str,
    completed_at: str,
    duration_us: int,
) -> None:
    # This wait is entirely off the caller's thread. It prevents a fast
    # terminal event from creating a topology-less terminal row before the
    # detached start snapshot arrives.
    frame.start_delivered.wait(_EXECUTION_TIMEOUT_SECONDS + 0.5)
    payload: dict[str, Any] = {
        "execution_id": frame.execution_id,
        "session_id": frame.session_id,
        "event": event,
        "duration_us": duration_us,
    }
    if event != "awaiting_approval":
        payload["completed_at"] = completed_at
    if error_code:
        payload["error_code"] = error_code[:64]
    if error_summary:
        payload["error_summary"] = error_summary[:512]
    _post_event(frame.engine, payload)


def _post_event(engine: Any, payload: dict[str, Any]) -> None:
    """Send one lifecycle event. Caller is always a telemetry worker."""
    response = engine._http.post(
        f"{engine.gateway_url}{_EXECUTION_PATH}",
        json=payload,
        headers=engine._headers(include_agent_token=True),
        timeout=_EXECUTION_TIMEOUT_SECONDS,
    )
    # Telemetry remains best effort, but reading/closing the response ensures
    # pooled connections are promptly reusable.
    if response.status_code >= 400:
        return


def _actor_payload(engine: Any) -> dict[str, str]:
    try:
        binding = engine.principal_binding
    except Exception:
        binding = None
    if binding is None:
        return {}
    subject = str(getattr(binding, "subject", "") or "").strip()
    kind = str(getattr(binding, "kind", "") or "").strip().lower()
    if not subject:
        return {}
    if kind not in ("user", "service_account"):
        kind = "service_account" if subject.startswith("service_account:") else "user"
    return {"actor_subject": subject, "actor_kind": kind}


def _wire_topology(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Project a discovery manifest onto the immutable execution wire shape."""
    nodes = [
        _select(item, ("id", "label", "kind", "model", "ref", "x", "y"))
        for item in _items(manifest.get("nodes"))
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]
    edges = [
        _select(item, ("from", "to", "label", "kind"))
        for item in _items(manifest.get("edges"))
        if isinstance(item, dict)
        and str(item.get("from") or "").strip()
        and str(item.get("to") or "").strip()
    ]
    tools: list[dict[str, Any]] = []
    for item in _items(manifest.get("tools")):
        if not isinstance(item, dict) or not str(item.get("key") or "").strip():
            continue
        tool = _select(item, ("key", "name", "schema_digest"))
        server_key = item.get("server_key") or item.get("server")
        if server_key:
            tool["server_key"] = server_key
        actions = [
            _select(
                action,
                (
                    "key",
                    "name",
                    "description",
                    "action_class",
                    "risk",
                    "schema_digest",
                ),
            )
            for action in _items(item.get("actions"))
            if isinstance(action, dict) and str(action.get("key") or "").strip()
        ]
        if actions:
            tool["actions"] = actions
        tools.append(tool)
    return {"nodes": nodes, "edges": edges, "tools": tools}


def _items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return []


def _select(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: source[key]
        for key in keys
        if key in source and source[key] not in (None, "")
    }


def _framework_name(target: Any) -> str:
    module = str(getattr(type(target), "__module__", "") or "").lower()
    for token, label in (
        ("langgraph", "langgraph"),
        ("crewai", "crewai"),
        ("autogen", "autogen"),
        ("llama_index", "llamaindex"),
        ("pydantic_ai", "pydanticai"),
        ("agents", "openai_agents"),
        ("strands", "strands"),
        ("google.adk", "google_adk"),
        ("temporalio", "temporal"),
        ("model_tools", "hermes"),
    ):
        if token in module:
            return label
    return "generic"


def _failure_metadata(exc: BaseException) -> tuple[str, str, str]:
    code = type(exc).__name__ or "WorkflowError"
    if isinstance(exc, GuardrailApprovalPending):
        return "awaiting_approval", exc.code, ""
    if isinstance(exc, GuardrailDenied):
        return "blocked", exc.code, ""
    if isinstance(exc, DeepIntShieldError):
        return "failed", exc.code, ""
    translated_code = normalize_agentic_error_code(
        getattr(exc, "_deepintshield_error_code", ""),
        "",
    )
    if translated_code:
        event = (
            "awaiting_approval"
            if translated_code in {"require_approval", "guardrail_approval_pending"}
            else "blocked"
        )
        return event, translated_code, ""
    if isinstance(exc, GeneratorExit) or code in {
        "CancelledError",
        "KeyboardInterrupt",
        "SystemExit",
    }:
        return "cancelled", code, "workflow execution was cancelled"
    # Never persist ``str(exc)``: framework exceptions routinely include raw
    # tool arguments, prompts, credentials, or model output. The type is enough
    # to diagnose the failure class while preserving zero data retention.
    return "failed", code, f"workflow raised {code}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def topology_signature(topology: dict[str, Any]) -> str:
    """Client-side diagnostic signature (the SDK omits it on writes).

    The server remains authoritative and computes its canonical signature. This
    helper is useful for tests/logging without risking a client/server
    canonicalization mismatch.
    """
    raw = json.dumps(topology, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


__all__ = [
    "ExecutionFrame",
    "execution_scope",
    "current_execution",
    "current_execution_id",
    "cache_topology",
    "mark_execution_outcome",
    "topology_signature",
]
