"""Request-local task attenuation and authorization ancestry.

Receipts identify prior gateway authorizations, not completed tool executions or
verified output values. Sibling async tasks inherit a snapshot; joining branches
requires explicitly supplying their server receipt IDs in a new task scope.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterator, Sequence

from .errors import GovernanceConfigurationError


def _invalid() -> GovernanceConfigurationError:
    return GovernanceConfigurationError(framework="agentic-task", code="agent_task_context_invalid")


def _identifier(value: str, limit: int) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= limit
            and value == value.strip() and not any(c in value for c in "*\x00\r\n"))


@dataclass(frozen=True)
class TaskMemoryReference:
    slot: str
    version: int
    digest: str

    def __post_init__(self) -> None:
        if (not _identifier(self.slot, 128) or type(self.version) is not int
                or self.version < 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.digest)):
            raise _invalid()


def memory_reference(slot: str, value: bytes, version: int) -> TaskMemoryReference:
    """Hash the bytes actually read; the gateway compares this with its commitment."""
    if not isinstance(value, bytes):
        raise _invalid()
    return TaskMemoryReference(slot, version, "sha256:" + hashlib.sha256(value).hexdigest())


@dataclass(frozen=True)
class TaskContext:
    mandate: str = field(repr=False)
    input_receipts: tuple[str, ...] = ()
    memory_inputs: tuple[TaskMemoryReference, ...] = ()
    task_id: str = ""
    last_receipt: str = ""
    _explicit: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_receipts", tuple(self.input_receipts))
        object.__setattr__(self, "memory_inputs", tuple(self.memory_inputs))
        if (not isinstance(self.mandate, str) or not self.mandate
                or len(self.mandate) > 4096 or self.mandate != self.mandate.strip()
                or not self.mandate.isascii() or any(c in self.mandate for c in "\x00\r\n")):
            raise _invalid()
        if len(self.input_receipts) > 32 or any(not _identifier(r, 64) for r in self.input_receipts):
            raise _invalid()
        if len(self.memory_inputs) > 16 or any(not isinstance(r, TaskMemoryReference) for r in self.memory_inputs):
            raise _invalid()
        if len({r.slot for r in self.memory_inputs}) != len(self.memory_inputs):
            raise _invalid()


_context: contextvars.ContextVar[TaskContext | None] = contextvars.ContextVar("deepintshield_task", default=None)


def current_task_context() -> TaskContext | None:
    current = _context.get()
    if current is not None and current._explicit:
        return current
    mandate = os.environ.get("DEEPINTSHIELD_TASK_MANDATE", "")
    if not mandate:
        return None
    if current is not None and current.mandate == mandate:
        return current
    return TaskContext(mandate, _explicit=False)


@contextmanager
def task_scope(
    mandate: str | None = None, *, input_receipts: Sequence[str] = (),
    memory_inputs: Sequence[TaskMemoryReference] = (),
) -> Iterator[TaskContext]:
    """Bind a signed mandate and input references without altering framework APIs."""
    previous = current_task_context()
    selected = mandate if mandate is not None else (previous.mandate if previous else "")
    context = TaskContext(selected, tuple(input_receipts), tuple(memory_inputs))
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


@contextmanager
def task_execution_scope(*, new_execution: bool) -> Iterator[None]:
    """Environment-only mandates start fresh ancestry for each native execution."""
    current = current_task_context()
    if not new_execution or current is None or current._explicit:
        yield
        return
    token = _context.set(TaskContext(current.mandate, _explicit=False))
    try:
        yield
    finally:
        _context.reset(token)


def task_payload() -> dict:
    current = current_task_context()
    if current is None:
        return {}
    return {
        "task_mandate": current.mandate,
        "input_receipts": list(current.input_receipts),
        "memory_inputs": [dict(slot=r.slot, version=r.version, digest=r.digest) for r in current.memory_inputs],
    }


def capture_task_receipt(task_id: str, receipt: str) -> None:
    current = current_task_context()
    if current is None:
        return
    if (not _identifier(task_id, 64) or not _identifier(receipt, 64)
            or (current.task_id and current.task_id != task_id)
            or receipt == current.last_receipt or receipt in current.input_receipts):
        raise _invalid()
    # The server receipt retains its parents. Carry the latest predecessor so
    # long sequential runs do not overflow the bounded ancestry input list.
    _context.set(replace(current, task_id=task_id, last_receipt=receipt, input_receipts=(receipt,)))


def require_task_security(enabled: bool) -> bool:
    if not enabled and current_task_context() is not None:
        raise GovernanceConfigurationError(framework="agentic-task", code="agent_task_security_unavailable")
    return enabled
