from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GuardrailStage = Literal["input", "output", "action", "mcp", "rag"]
GuardrailDecision = Literal[
    "allow", "allow_with_redaction", "block", "deny", "redact", "monitor", "sandbox", "approval_required"
]
NON_BLOCKING_DECISIONS = frozenset({"allow", "allow_with_redaction", "redact", "monitor"})


@dataclass(slots=True)
class RetrievedChunk:
    """Retrieved content and its security metadata.

    ``source_id`` accepts a registered source ID or exact source name within
    the authenticated workspace. Empty values inherit the evaluation's
    ``source_id``, or use workspace defaults when neither is supplied.
    ``source_name`` is descriptive metadata, not a source lookup selector.
    Authorize document access in the application or retriever before creating
    chunks for safety evaluation. RAG filtering does not grant document access.
    """

    chunk_id: str
    document_id: str
    content: str
    document_version: str = "v1"
    offset_start: int = 0
    offset_end: int = 0
    source_id: str = ""
    source_name: str = ""
    source_health: str = "healthy"
    trust_score: int = 80
    injection_score: int = 0
    labels: list[str] = field(default_factory=list)
    pii_flags: list[str] = field(default_factory=list)
    quarantined: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["offset_end"] <= 0:
            payload["offset_end"] = len(self.content)
        if not payload["metadata"]:
            payload.pop("metadata", None)
        return payload


@dataclass(slots=True)
class ToolInvocation:
    tool_name: str
    tool_input: str | dict[str, Any]
    server_label: str = ""
    action_class: str = "read"
    domains: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["metadata"]:
            payload.pop("metadata", None)
        return payload


@dataclass(slots=True)
class GuardrailResult:
    decision: str
    stage: str
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    # Enforcement mode reported by the gateway: "sync"/"enforce" actually blocks,
    # "shadow"/"async" reports the verdict without blocking. Empty when the
    # gateway does not report it. Additive and optional - existing callers are
    # unaffected.
    mode: str = ""

    @property
    def allowed(self) -> bool:
        """Classify the verdict independently of the gateway enforcement mode."""
        return (self.decision or "").lower() in NON_BLOCKING_DECISIONS

    @property
    def blocked(self) -> bool:
        return not self.allowed

    @classmethod
    def from_response(cls, stage: str, payload: dict[str, Any]) -> "GuardrailResult":
        inner = payload.get("result", payload)
        if not isinstance(inner, dict):
            raise ValueError("The gateway returned an invalid guardrail result")
        decision = inner.get("decision")
        if isinstance(decision, dict):
            decision = decision.get("decision", "")
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError("The gateway returned no guardrail decision")
        return cls(
            decision=decision.strip().lower(),
            stage=stage,
            reason=(inner.get("reason") or "").strip(),
            raw=payload,
            mode=(inner.get("mode") or "").strip().lower(),
        )
