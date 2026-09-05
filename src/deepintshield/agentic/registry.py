"""Automatic agent/tool/network discovery for the GAF registry (agentic-new).

``manifest.describe`` already extracts an agent's declared *tool surface* for the
agentic-security blueprint. The GAF registry wants one level more structure: the
**network** (a typed node/edge graph), the **agents** that run inside it, the
**tools** they reach and the **MCP servers** those tools live on - so the
Registry tab can render the real topology and the authorization layer can write
``agent → organization`` / ``tool → server`` tuples without anyone hand-editing
relationships.

Two entry points:

  * :func:`describe_network` - pure, offline introspection of a compiled
    LangGraph / CrewAI crew / LlamaIndex or AutoGen agent (with a generic
    fallback for everything else), producing the ``/registry/discover`` payload
    plus a stable ``sha256`` digest over the topology. The digest is what makes
    the server-side upsert idempotent and what turns a changed graph into a
    drift event.
  * :func:`discover` - POST that payload. Explicit discovery is fire-and-forget
    by default and de-duplicated per digest (at most one post per 5 minutes), so
    a hot loop that rebuilds an unchanged graph cannot hammer the gateway.
    Framework execution boundaries require one durable acknowledgement for a
    new or changed blueprint before user code runs; acknowledged unchanged
    blueprints add no repeated scan request.

No runtime data: node labels, tool names, edges, digests, plus a bounded and
credential-redacted implementation snapshot used only for registration-time
blueprint scanning. Arguments and prompts are never captured; the gateway
discards source after analysis and persists only digests/findings.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import logging
import marshal
import os
import re
import sys
import threading
import time
import weakref
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from .actions import infer_action_class
from .errors import (
    DeepIntShieldError,
    GovernanceConfigurationError,
    normalize_agentic_error_code,
)

log = logging.getLogger(__name__)

_DISCOVER_PATH = "/api/agentic-new/registry/discover"
_TIMEOUT_SECONDS = 3.0
_REGISTRATION_CAPTURE_RETRY_SECONDS = 30.0

# At most one post per client/manifest/principal scope per this many seconds.
# A graph compiled per web request therefore reports once for that client, but
# an identical graph owned by another VK/client is never suppressed.
REPOST_INTERVAL_SECONDS = 300.0
_SEEN_MAXSIZE = 256
# Ceiling on concurrently-dispatched background posts. The digest dedupe already
# collapses a hot loop that rebuilds the *same* graph; this bounds the pathological
# case of a loop building a *different* graph every iteration, which would
# otherwise spawn one OS thread per compile against a slow/hung gateway.
_INFLIGHT_MAX = 4

# Source is registration evidence, not telemetry. Keep uploads small and
# predictable even when a framework wrapper exposes a generated callable with a
# very large body. The gateway independently enforces the same contract.
_CODE_ARTIFACT_MAX_BYTES = 64 * 1024
_CODE_BUNDLE_MAX_BYTES = 512 * 1024
_CODE_ARTIFACT_MAX_COUNT = 64
_CODE_HELPER_MAX_DEPTH = 2
_CODE_HELPER_MAX_COUNT = 16
_CODE_CONTAINER_MAX_DEPTH = 3
_CODE_CONTAINER_MAX_ITEMS = 128
_CODE_WRAPPER_MAX_COUNT = 8
_SOURCE_CACHE_MAXSIZE = 256
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COVERAGE_PRIORITY = {
    "captured": 0,
    "partial": 1,
    "truncated": 2,
    "omitted": 3,
    "missing": 4,
}

# Redact literal credential material before it leaves the process. The scanner
# still sees the security-relevant assignment/symbol (and can flag a hard-coded
# secret), while the value itself never reaches the gateway or its model.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|passwd|private[_-]?key|secret(?:[_-]?key)?)\b\s*[:=]\s*)"
    r"(?P<quote>['\"])(?P<value>[^'\"\r\n]{4,})(?P=quote)"
)
_TOKEN_LITERAL_RE = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|"
    r"gh[opusr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

_LG_RESERVED = ("__start__", "__end__")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Object-id sanitiser for subject keys. Unlike the slug it preserves case and
# underscores, because a tool key IS the governed tool identity the PDP decides
# on (``crm_read``) - dash-slugging it would break the join with the decisions.
# A Python default repr - "<__main__.Plain object at 0x10b9082f0>". It is
# manifest.describe's last-resort name for an object with no name/__name__, and
# it embeds a MEMORY ADDRESS: left alone it registers a junk tool whose key
# changes every process, so the network digest changes every run and a perfectly
# static agent looks like continuous topology drift.
_REPR_RE = re.compile(r"at 0x[0-9A-Fa-f]{4,}|^<.+>$")
# Name tokens that say nothing about what an agent can do.
_STOPWORDS = frozenset({"agent", "node", "step", "task", "run", "call", "chain", "fn", "func"})
# scoped dedupe key → time.monotonic() of the last dispatch.
_seen: dict[str, float] = {}
_seen_lock = threading.Lock()
# Background posts currently in flight (guarded by _seen_lock).
_inflight = 0


class _ReportFlight:
    """Completion state for one digest posted by a background reporter."""

    __slots__ = ("completed", "succeeded")

    def __init__(self) -> None:
        self.completed = threading.Event()
        self.succeeded = False


_report_flights: dict[str, _ReportFlight] = {}

# Function -> (local bytecode/dependency token, expanded source, incomplete).
# Weak keys avoid retaining application callables. The token is cheap to recompute
# and invalidates the cached source when hot reload replaces a function, wrapper,
# or directly referenced application helper.
_expanded_source_cache: weakref.WeakKeyDictionary[
    Any, tuple[str, str, bool]
] = weakref.WeakKeyDictionary()
_expanded_source_cache_lock = threading.Lock()


def _redact_source(source: str) -> tuple[str, bool]:
    """Remove likely credential *values* without hiding dangerous behavior."""
    redacted = False

    def assignment(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return f'{match.group(1)}{match.group("quote")}<redacted-secret>{match.group("quote")}'

    source = _SECRET_ASSIGNMENT_RE.sub(assignment, source)
    if _PRIVATE_KEY_BLOCK_RE.search(source):
        redacted = True
        source = _PRIVATE_KEY_BLOCK_RE.sub("<redacted-private-key>", source)
    if _TOKEN_LITERAL_RE.search(source):
        redacted = True
        source = _TOKEN_LITERAL_RE.sub("<redacted-token>", source)
    return source, redacted


def _bounded_source(source: str) -> tuple[str, bool]:
    """UTF-8 byte cap that never returns a broken Unicode sequence."""
    encoded = source.replace("\r\n", "\n").replace("\r", "\n").encode(
        "utf-8", "replace"
    )
    if len(encoded) <= _CODE_ARTIFACT_MAX_BYTES:
        return encoded.decode("utf-8"), False
    return encoded[:_CODE_ARTIFACT_MAX_BYTES].decode("utf-8", "ignore"), True


def _normalised_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _source_digest(source: str) -> str:
    return "sha256:" + hashlib.sha256(
        _normalised_source(source).encode("utf-8", "replace")
    ).hexdigest()


def _python_function(value: Any) -> Any:
    """Return the directly executed Python function without unwrapping it."""
    if inspect.ismethod(value):
        value = _attr(value, "__func__", value)
    if inspect.isfunction(value):
        return value
    # Callable application objects commonly keep their implementation here.
    call = _attr(type(value), "__call__", None) if value is not None else None
    if inspect.ismethod(call):
        call = _attr(call, "__func__", call)
    return call if inspect.isfunction(call) else None


def _callable_layers(value: Any) -> tuple[list[Any], bool]:
    """Executed wrapper chain, outermost first, with a strict cycle/count cap."""
    first = _python_function(value)
    if first is None:
        return [], False
    layers: list[Any] = []
    seen: set[int] = set()
    current = first
    incomplete = False
    while current is not None:
        identity = id(current)
        if identity in seen:
            incomplete = True
            break
        seen.add(identity)
        if len(layers) >= _CODE_WRAPPER_MAX_COUNT:
            incomplete = True
            break
        layers.append(current)
        # A Python function's dict is inert; avoid invoking an arbitrary
        # descriptor merely to follow ``__wrapped__``.
        wrapped = vars(current).get("__wrapped__")
        current = _python_function(wrapped) if wrapped is not None else None
    return layers, incomplete


def _source_file(value: Any) -> str:
    try:
        return inspect.getabsfile(value)
    except (OSError, TypeError, ValueError):
        return ""
    except Exception:
        return ""


def _direct_source(value: Any) -> str:
    """Source for this exact function, bypassing ``inspect`` auto-unwrapping."""
    fn = _python_function(value)
    if fn is None:
        return inspect.getsource(value)
    lines, line_number = inspect.findsource(fn)
    return "".join(inspect.getblock(lines[line_number:]))


def _application_source_root(value: Any) -> str:
    """Smallest source tree that represents the callable's top-level package."""
    source_file = _source_file(value)
    if not source_file:
        return ""
    root = os.path.realpath(os.path.dirname(source_file))
    module = str(_attr(value, "__module__", "") or "")
    parts = [part for part in module.split(".") if part and part != "__main__"]
    # ``pkg.sub.module`` lives at ``<root>/pkg/sub/module.py``. Starting from
    # its directory, one ascent reaches the top-level ``pkg`` source tree.
    for _ in range(max(0, len(parts) - 2)):
        root = os.path.dirname(root)
    return root


def _is_application_function(value: Any, application_root: str) -> bool:
    fn = _python_function(value)
    source_file = os.path.realpath(_source_file(fn)) if fn is not None else ""
    if not source_file or not application_root:
        return False
    try:
        return os.path.commonpath((application_root, source_file)) == application_root
    except (OSError, ValueError):
        return False


def _function_identity(value: Any) -> tuple[str, str, int]:
    fn = _python_function(value)
    code = _attr(fn, "__code__", None)
    return (
        _source_file(fn),
        str(_attr(fn, "__qualname__", "") or ""),
        int(_attr(code, "co_firstlineno", 0) or 0),
    )


def _local_function_dependencies(
    value: Any, application_root: str
) -> tuple[list[Any], bool]:
    """Direct same-application functions, including ``module.helper()`` calls.

    Only already-imported module dictionaries are inspected. No module is
    imported, no descriptor is invoked, and every candidate must live beneath
    the root callable's top-level package source tree. Bounded built-in
    containers are traversed as well: dispatch tables are executable
    dependencies, and ignoring their callable values would let a target be
    replaced after approval without changing the scanned entrypoint source.

    The boolean result reports an incomplete traversal. Callers bind it into
    the change token and fail closed through the normal coverage contract.
    """
    fn = _python_function(value)
    code = _attr(fn, "__code__", None)
    if fn is None or code is None:
        return [], False
    try:
        closure = inspect.getclosurevars(fn)
    except Exception:
        return [], True
    values: dict[str, Any] = {}
    values.update(closure.globals)
    values.update(closure.nonlocals)
    names = set(_attr(code, "co_names", ()) or ())
    names.update(_attr(code, "co_freevars", ()) or ())
    dependencies: dict[tuple[str, str, int], Any] = {}
    visited_containers: set[int] = set()
    traversed_items = 0
    incomplete = False

    def include(candidate: Any, depth: int = 0) -> None:
        nonlocal incomplete, traversed_items
        candidate_fn = _python_function(candidate)
        if candidate_fn is not None:
            if candidate_fn is fn or not _is_application_function(
                candidate_fn, application_root
            ):
                return
            dependencies.setdefault(_function_identity(candidate_fn), candidate_fn)
            return

        if not isinstance(candidate, (dict, list, tuple, set, frozenset)):
            return
        identity = id(candidate)
        if identity in visited_containers:
            return
        visited_containers.add(identity)
        if depth >= _CODE_CONTAINER_MAX_DEPTH:
            if candidate:
                incomplete = True
            return
        values = (
            list(candidate.values())
            if isinstance(candidate, dict)
            else list(candidate)
        )
        remaining = _CODE_CONTAINER_MAX_ITEMS - traversed_items
        if remaining <= 0:
            if values:
                incomplete = True
            return
        if len(values) > remaining:
            incomplete = True
            values = values[:remaining]
        traversed_items += len(values)
        for nested in values:
            include(nested, depth + 1)

    for name in sorted(names):
        value_for_name = values.get(name)
        include(value_for_name)
        if inspect.ismodule(value_for_name):
            module_values = vars(value_for_name)
            for attribute in sorted(names):
                include(module_values.get(attribute))
    return [dependencies[key] for key in sorted(dependencies)], incomplete


def _safe_runtime_constant(value: Any, depth: int = 0) -> Any:
    """Side-effect-free shape used only by the local change detector."""
    if depth > 3:
        return ["depth"]
    if value is None or isinstance(value, (bool, int, float, str)):
        return [type(value).__name__, value]
    if isinstance(value, bytes):
        return ["bytes", hashlib.sha256(value).hexdigest()]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_safe_runtime_constant(v, depth + 1) for v in value[:64]]]
    if isinstance(value, dict):
        items = []
        for key, item in list(value.items())[:64]:
            if isinstance(key, (str, int, float, bool, type(None))):
                items.append(
                    [_safe_runtime_constant(key, depth + 1), _safe_runtime_constant(item, depth + 1)]
                )
        return ["dict", sorted(items, key=lambda item: json.dumps(item[0], sort_keys=True))]
    return ["object", type(value).__module__, type(value).__qualname__]


def _callable_change_token(value: Any) -> str:
    """Cheap local digest of executed wrappers, code, defaults, and helpers."""
    layers, wrapper_incomplete = _callable_layers(value)
    if not layers:
        return ""
    implementation = layers[-1]
    application_root = _application_source_root(implementation)
    roots = [
        layer for layer in layers if _is_application_function(layer, application_root)
    ] or [implementation]
    queue = [(root, 0) for root in roots]
    visited: set[tuple[str, str, int]] = set()
    entries: list[tuple[tuple[str, str, int], bytes, Any]] = []
    incomplete = wrapper_incomplete
    while queue:
        fn, depth = queue.pop(0)
        identity = _function_identity(fn)
        if identity in visited:
            continue
        visited.add(identity)
        if len(entries) >= _CODE_HELPER_MAX_COUNT + _CODE_WRAPPER_MAX_COUNT:
            incomplete = True
            continue
        code = _attr(fn, "__code__", None)
        if code is None:
            incomplete = True
            continue
        try:
            code_bytes = marshal.dumps(code)
        except Exception:
            incomplete = True
            continue
        defaults = [
            _safe_runtime_constant(_attr(fn, "__defaults__", None)),
            _safe_runtime_constant(_attr(fn, "__kwdefaults__", None)),
        ]
        entries.append((identity, code_bytes, defaults))
        children, dependency_incomplete = _local_function_dependencies(
            fn, application_root
        )
        incomplete = incomplete or dependency_incomplete
        if children and depth >= _CODE_HELPER_MAX_DEPTH:
            incomplete = True
            continue
        queue.extend((child, depth + 1) for child in children)
        queue.sort(key=lambda item: (_function_identity(item[0]), item[1]))
    if not entries:
        return ""
    digest = hashlib.sha256()
    for identity, code_bytes, defaults in sorted(entries, key=lambda item: item[0]):
        metadata = json.dumps(
            [identity, defaults], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8", "replace")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(code_bytes).to_bytes(8, "big"))
        digest.update(code_bytes)
    digest.update(b"partial" if incomplete else b"complete")
    return "sha256:" + digest.hexdigest()


def _expanded_callable_source(value: Any) -> tuple[str, bool]:
    """Read a callable plus bounded wrappers and application helper source."""
    layers, wrapper_incomplete = _callable_layers(value)
    cache_key = layers[0] if layers else None
    change_token = _callable_change_token(value) if cache_key is not None else ""
    if cache_key is not None and change_token:
        try:
            with _expanded_source_cache_lock:
                cached = _expanded_source_cache.get(cache_key)
            if cached is not None and cached[0] == change_token:
                return cached[1], cached[2]
        except (TypeError, RuntimeError):
            pass

    try:
        target = layers[-1] if layers else inspect.unwrap(value)
        source = _direct_source(target)
    except (OSError, TypeError, ValueError, AttributeError):
        return "", True
    except Exception:
        return "", True
    if not isinstance(source, str) or not source.strip():
        return "", True

    root_fn = _python_function(target)
    application_root = _application_source_root(root_fn)
    if root_fn is None or not application_root:
        return source, False

    wrapper_sources: list[tuple[tuple[str, str, int], str]] = []
    roots = [root_fn]
    incomplete = wrapper_incomplete
    for wrapper in layers[:-1]:
        if not _is_application_function(wrapper, application_root):
            continue
        roots.append(wrapper)
        try:
            wrapper_source = _direct_source(wrapper)
        except (OSError, TypeError, ValueError, AttributeError):
            incomplete = True
            continue
        except Exception:
            incomplete = True
            continue
        if not isinstance(wrapper_source, str) or not wrapper_source.strip():
            incomplete = True
            continue
        wrapper_sources.append((_function_identity(wrapper), wrapper_source))

    visited = {_function_identity(root) for root in roots}
    queue: list[tuple[Any, int]] = []
    for root in roots:
        dependencies, dependency_incomplete = _local_function_dependencies(
            root, application_root
        )
        incomplete = incomplete or dependency_incomplete
        queue.extend((dependency, 1) for dependency in dependencies)
    helpers: list[tuple[tuple[str, str, int], str]] = []
    while queue:
        helper, depth = queue.pop(0)
        identity = _function_identity(helper)
        if identity in visited:
            continue
        visited.add(identity)
        if len(helpers) >= _CODE_HELPER_MAX_COUNT:
            incomplete = True
            continue
        try:
            helper_source = _direct_source(helper)
        except (OSError, TypeError, ValueError, AttributeError):
            incomplete = True
            continue
        except Exception:
            incomplete = True
            continue
        if not isinstance(helper_source, str) or not helper_source.strip():
            incomplete = True
            continue
        helpers.append((identity, helper_source))
        children, dependency_incomplete = _local_function_dependencies(
            helper, application_root
        )
        incomplete = incomplete or dependency_incomplete
        if children and depth >= _CODE_HELPER_MAX_DEPTH:
            incomplete = True
            continue
        queue.extend((child, depth + 1) for child in children)
        queue.sort(
            key=lambda item: (
                _function_identity(item[0]),
                item[1],
            )
        )

    expanded = source.rstrip()
    for (_, qualname, _), wrapper_source in sorted(wrapper_sources):
        if wrapper_source.strip() in expanded:
            continue
        expanded += (
            f"\n\n# deepintshield: application decorator wrapper {qualname}\n"
            f"{wrapper_source.rstrip()}"
        )
    for (_, qualname, _), helper_source in sorted(helpers):
        # A nested helper can already be inside the root function's source.
        # Still traverse it above, but do not duplicate its bytes on the wire.
        if helper_source.strip() in expanded:
            continue
        expanded += (
            f"\n\n# deepintshield: referenced local helper {qualname}\n"
            f"{helper_source.rstrip()}"
        )
    result = (expanded + "\n", incomplete)
    if cache_key is not None and change_token:
        try:
            with _expanded_source_cache_lock:
                if len(_expanded_source_cache) >= _SOURCE_CACHE_MAXSIZE:
                    _expanded_source_cache.clear()
                _expanded_source_cache[cache_key] = (change_token, result[0], result[1])
        except (TypeError, RuntimeError):
            pass
    return result


def _source_artifact(
    value: Any,
    *,
    subject_kind: str,
    subject_key: str,
    source: str = "",
) -> dict[str, Any]:
    """Build transient, digest-bound source evidence without executing code."""
    incomplete = {
        "subject_kind": subject_kind,
        "subject_key": _key(subject_key),
        "_incomplete": True,
    }
    dependency_incomplete = False
    if not source:
        source, dependency_incomplete = _expanded_callable_source(value)
        if not source:
            return incomplete
    if not isinstance(source, str) or not source.strip():
        return incomplete
    # This digest is calculated before redaction and is explicitly only an SDK
    # self-attestation. It detects secret-literal-only drift for local dedupe;
    # the gateway must never treat a reporter-provided digest as proof.
    self_attested_digest = _source_digest(source)
    source, redacted = _redact_source(source)
    source, truncated = _bounded_source(source)
    if not source.strip():
        return incomplete
    digest = "sha256:" + hashlib.sha256(source.encode("utf-8", "replace")).hexdigest()
    artifact: dict[str, Any] = {
        "subject_kind": subject_kind,
        "subject_key": _key(subject_key),
        "language": "python",
        "digest": digest,
        "self_attested_content_digest": self_attested_digest,
        "source": source,
    }
    if not artifact["subject_key"]:
        return incomplete
    if redacted:
        artifact["redacted"] = True
    if truncated:
        artifact["truncated"] = True
        artifact["_incomplete"] = True
        artifact["_coverage_status"] = "truncated"
    elif dependency_incomplete:
        artifact["_incomplete"] = True
        artifact["_coverage_status"] = "partial"
    return artifact


def _callback_code_artifacts(
    inventory: Any,
    *,
    subject_key: str,
) -> list[dict[str, Any]]:
    """Build one slot-bound artifact for each adapter-declared callback.

    The adapter has already selected explicit public callback fields.  This
    helper reads only those callables, and binds their structural slot into the
    scanned bytes using an opaque digest (no hook/event names or runtime data
    need leave the process).  Missing/partial source is retained as coverage
    evidence so a required execution report fails closed.
    """
    raw_callbacks = tuple(_attr(inventory, "callbacks", ()) or ())
    incomplete = bool(_attr(inventory, "incomplete", False))
    ordered = sorted(
        raw_callbacks,
        key=lambda entry: (
            str(_attr(entry, "slot", "") or ""),
            _function_identity(_attr(entry, "callback", None)),
        ),
    )
    artifacts: list[dict[str, Any]] = []
    for entry in ordered:
        slot = str(_attr(entry, "slot", "") or "")
        callback = _attr(entry, "callback", None)
        source, dependency_incomplete = _expanded_callable_source(callback)
        if not source:
            artifacts.append(
                _source_artifact(
                    None,
                    subject_kind="agent",
                    subject_key=subject_key,
                )
            )
            continue
        slot_digest = hashlib.sha256(slot.encode("utf-8", "replace")).hexdigest()
        artifact = _source_artifact(
            None,
            subject_kind="agent",
            subject_key=subject_key,
            source=f"# deepintshield-callback-slot-sha256:{slot_digest}\n{source}",
        )
        if dependency_incomplete and artifact.get("source"):
            artifact["_incomplete"] = True
            artifact["_coverage_status"] = "partial"
        artifacts.append(artifact)
    if incomplete:
        artifacts.append(
            _source_artifact(
                None,
                subject_kind="agent",
                subject_key=subject_key,
            )
        )
    return artifacts


def _attach_callback_inventory(node: dict[str, Any], inventory: Any) -> bool:
    """Mark and source-bind a node iff an adapter found application callbacks."""
    callbacks = tuple(_attr(inventory, "callbacks", ()) or ())
    incomplete = bool(_attr(inventory, "incomplete", False))
    if not callbacks and not incomplete:
        return False
    node["executable"] = True
    node.setdefault("_code_artifacts", []).extend(
        _callback_code_artifacts(inventory, subject_key=str(node.get("id") or ""))
    )
    return True


def _append_callback_router(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    inventory: Any,
    *,
    framework: str,
) -> str:
    """Represent network-level callbacks without reclassifying declarative agents."""
    callbacks = tuple(_attr(inventory, "callbacks", ()) or ())
    incomplete = bool(_attr(inventory, "incomplete", False))
    if not callbacks and not incomplete:
        return ""
    used = {str(node.get("id") or "") for node in nodes}
    base = _key(f"{framework}-callbacks") or "application-callbacks"
    key = base
    suffix = 2
    while key in used:
        key = f"{base}-{suffix}"
        suffix += 1
    node: dict[str, Any] = {
        "id": key,
        "label": f"{_label(framework)} callbacks",
        "kind": "router",
        "ref": f"callback:{_key(framework)}",
    }
    _attach_callback_inventory(node, inventory)
    nodes.append(node)
    edges.append({"from": "input", "to": key, "kind": "callback"})
    edges.append({"from": key, "to": "output", "kind": "callback"})
    return key


def _normalise_code_artifact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    kind = str(raw.get("subject_kind") or "").strip().lower()
    key = _key(raw.get("subject_key") or "")
    language = str(raw.get("language") or "python").strip().lower()
    source = raw.get("source")
    if (
        kind not in {"agent", "tool"}
        or not key
        or language != "python"
        or not isinstance(source, str)
    ):
        return {}
    # Recompute the digest after local redaction/capping. A caller-supplied
    # digest is never trusted as evidence for bytes we did not actually send.
    artifact = _source_artifact(
        None,
        subject_kind=kind,
        subject_key=key,
        source=source,
    )
    attestation = str(raw.get("self_attested_content_digest") or "").strip().lower()
    if _SHA256_DIGEST_RE.fullmatch(attestation):
        artifact["self_attested_content_digest"] = attestation
    if raw.get("redacted") is True:
        artifact["redacted"] = True
    if raw.get("truncated") is True:
        artifact["truncated"] = True
        artifact["_coverage_status"] = "truncated"
    if artifact and (raw.get("truncated") is True or raw.get("_incomplete") is True):
        artifact["_incomplete"] = True
    status = str(raw.get("_coverage_status") or "").strip().lower()
    if status in _COVERAGE_PRIORITY:
        artifact["_coverage_status"] = status
    return artifact


def _coverage_entry(raw: Any, *, status: str = "missing") -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    kind = str(raw.get("subject_kind") or "").strip().lower()
    key = _key(raw.get("subject_key") or "")
    declared = str(
        raw.get("status") or raw.get("_coverage_status") or status
    ).strip().lower()
    if kind not in {"agent", "tool"} or not key or declared not in _COVERAGE_PRIORITY:
        return {}
    return {"subject_kind": kind, "subject_key": key, "status": declared}


def _merge_coverage(
    coverage: dict[tuple[str, str], dict[str, str]],
    entry: dict[str, str],
) -> None:
    if not entry:
        return
    identity = (entry["subject_kind"], entry["subject_key"])
    previous = coverage.get(identity)
    if (
        previous is None
        or _COVERAGE_PRIORITY[entry["status"]]
        > _COVERAGE_PRIORITY[previous["status"]]
    ):
        coverage[identity] = entry


def _self_attested_code_digest(
    artifacts: list[dict[str, Any]],
    coverage: list[dict[str, str]],
) -> str:
    parts = [
        "content:"
        + ":".join(
            (
                item["subject_kind"],
                item["subject_key"],
                str(item.get("self_attested_content_digest") or item["digest"]),
            )
        )
        for item in artifacts
    ]
    parts.extend(
        f"coverage:{item['subject_kind']}:{item['subject_key']}:{item['status']}"
        for item in coverage
    )
    if not parts:
        return ""
    return "sha256:" + hashlib.sha256(
        "\n".join(sorted(parts)).encode("utf-8")
    ).hexdigest()


def _collect_code_artifacts(
    nodes: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool, list[dict[str, str]]]:
    candidates: list[dict[str, Any]] = []
    for node in nodes:
        artifact = node.pop("_code_artifact", None)
        if isinstance(artifact, dict):
            candidates.append(artifact)
        artifacts = node.pop("_code_artifacts", None)
        if isinstance(artifacts, (list, tuple)):
            candidates.extend(
                item for item in artifacts if isinstance(item, dict)
            )
    for tool in tools:
        artifact = tool.pop("_code_artifact", None)
        if isinstance(artifact, dict):
            candidates.append(artifact)
        artifacts = tool.pop("_code_artifacts", None)
        if isinstance(artifacts, (list, tuple)):
            candidates.extend(
                item for item in artifacts if isinstance(item, dict)
            )

    # Stable de-duplication and a total byte cap keep a graph with repeated
    # wrapper objects from amplifying registration traffic.
    by_subject: dict[tuple[str, str, str], dict[str, Any]] = {}
    coverage: dict[tuple[str, str], dict[str, str]] = {}
    used = 0
    incomplete = False
    for artifact in candidates:
        clean = _normalise_code_artifact(artifact)
        if not clean or clean.get("_incomplete") is True:
            incomplete = True
        if not clean or "source" not in clean:
            _merge_coverage(coverage, _coverage_entry(artifact))
            continue
        status = str(clean.get("_coverage_status") or "captured")
        key = (clean["subject_kind"], clean["subject_key"], clean["digest"])
        if key in by_subject:
            _merge_coverage(coverage, _coverage_entry(clean, status=status))
            continue
        size = len(clean["source"].encode("utf-8", "replace"))
        if (
            len(by_subject) >= _CODE_ARTIFACT_MAX_COUNT
            or used + size > _CODE_BUNDLE_MAX_BYTES
        ):
            incomplete = True
            _merge_coverage(
                coverage,
                _coverage_entry({**clean, "_coverage_status": "omitted"}),
            )
            continue
        clean.pop("_incomplete", None)
        clean.pop("_coverage_status", None)
        by_subject[key] = clean
        used += size
        _merge_coverage(coverage, _coverage_entry(clean, status=status))
    ledger = [coverage[key] for key in sorted(coverage)]
    incomplete = incomplete or any(item["status"] != "captured" for item in ledger)
    return [by_subject[key] for key in sorted(by_subject)], incomplete, ledger


def _blueprint_digest(
    topology_digest: str,
    artifacts: list[dict[str, Any]],
    *,
    incomplete: bool = False,
    coverage: Optional[list[dict[str, str]]] = None,
) -> str:
    parts = [
        f"topology:{topology_digest}",
        "coverage:incomplete" if incomplete else "coverage:complete",
    ]
    parts.extend(
        f"code:{item['subject_kind']}:{item['subject_key']}:{item['digest']}:"
        f"{item.get('self_attested_content_digest') or item['digest']}"
        for item in sorted(
            artifacts,
            key=lambda item: (item["subject_kind"], item["subject_key"], item["digest"]),
        )
    )
    parts.extend(
        f"executable:{item['subject_kind']}:{item['subject_key']}:{item['status']}"
        for item in (coverage or [])
    )
    return "sha256:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """``getattr`` that also survives a *raising* descriptor.

    Plain ``getattr(obj, name, default)`` only swallows ``AttributeError``; a
    pydantic/SQLAlchemy/lazy-proxy property that raises anything else propagates
    straight out of introspection. Introspection itself must remain total, so
    every attribute probe goes through here; failed executable capture is then
    represented explicitly and rejected by a required execution barrier.
    """
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


_MISSING = object()


def _has(obj: Any, name: str) -> bool:
    """``hasattr`` with the same raising-descriptor guarantee as :func:`_attr`.

    (``hasattr`` re-raises anything that isn't ``AttributeError``, so the bare
    form would let a hostile ``@property`` escape framework detection.)"""
    return _attr(obj, name, _MISSING) is not _MISSING


# ──────────────────────────────────────────────────────────────────────────
# Description
# ──────────────────────────────────────────────────────────────────────────


def describe_network(
    obj: Any,
    *,
    name: str = "",
    framework: str = "",
) -> dict[str, Any]:
    """Introspect ``obj`` and return the ``/registry/discover`` payload.

    Accepts, in this order: a **plain dict** already shaped like a manifest
    (normalised and re-digested, never introspected), a **compiled LangGraph**,
    an **uncompiled ``StateGraph``**, a **CrewAI ``Crew``**, an AutoGen team, a
    LlamaIndex agent, and **any other object** via the generic fallback.

    Framework detection is duck-typed - no framework is ever imported here, and
    every attribute access goes through :func:`_attr`, so describing an object
    from a library that isn't installed, a half-built one, or one whose
    properties raise, degrades to the generic shape instead of raising.
    ``name`` overrides the network key; ``framework`` overrides the
    auto-detected framework label.

    **Never raises.** A total introspection failure returns a fail-closed,
    incomplete-coverage payload. The network helper remains safe to call from a
    framework hook; a required execution boundary will reject that payload.
    """
    try:
        return _describe_network(obj, name=name, framework=framework)
    except Exception as exc:  # introspection must never break the caller's run
        log.debug("describe_network fell back to an empty topology: %s", exc)
        return _empty_payload(framework or "generic", name)


def _describe_network(obj: Any, *, name: str, framework: str) -> dict[str, Any]:
    # A dict is never a framework object - it is a manifest the caller built (or
    # one we produced earlier and cached). Introspecting it would stringify the
    # whole mapping into a bogus "tool", so route it to the normaliser instead.
    if isinstance(obj, dict):
        return _from_manifest(obj, name=name, framework=framework)

    detected, nodes, edges, tools, servers = _introspect(obj)
    fw = framework or detected
    nodes, edges = _normalise_topology(nodes, edges)
    digest = _digest(nodes, edges)
    agents = _agents_from(nodes, edges)
    artifacts, artifacts_incomplete, coverage = _collect_code_artifacts(nodes, tools)
    # Extractors may carry private assembly hints for the derived agents list;
    # never put those non-contract fields on ManifestNode's wire shape.
    for node in nodes:
        node.pop("_capabilities", None)
    net_name = name or _network_name(obj)
    key = _slug(net_name) or f"{fw}-{digest[7:19]}"
    payload = {
        "framework": fw,
        "network": {
            "key": key,
            "name": net_name or key,
            "digest": digest,
        },
        "nodes": nodes,
        "edges": edges,
        "agents": agents,
        "tools": tools,
        "servers": servers,
        "code_artifacts": artifacts,
        "blueprint_digest": _blueprint_digest(
            digest,
            artifacts,
            incomplete=artifacts_incomplete,
            coverage=coverage,
        ),
    }
    if coverage:
        payload["code_artifact_coverage"] = coverage
    attestation = _self_attested_code_digest(artifacts, coverage)
    if attestation:
        payload["self_attested_code_digest"] = attestation
    if artifacts_incomplete:
        payload["code_artifacts_incomplete"] = True
    return payload


def _empty_payload(framework: str, name: str = "") -> dict[str, Any]:
    """Fail-closed payload used when framework introspection fails outright."""
    fw = _slug(framework) or "generic"
    digest = _digest([], [])
    key = _slug(name) or f"{fw}-{digest[7:19]}"
    return {
        "framework": fw,
        "network": {"key": key, "name": name or key, "digest": digest},
        "nodes": [],
        "edges": [],
        "agents": [],
        "tools": [],
        "servers": [],
        "code_artifacts": [],
        "code_artifacts_incomplete": True,
        "blueprint_digest": _blueprint_digest(digest, [], incomplete=True),
    }


def _introspect(obj: Any) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    """Route to the right extractor. Order mirrors ``AgenticSurface._dispatch``
    so discovery and enforcement always agree on the framework."""
    if _is_langgraph(obj):
        return _from_langgraph(obj)
    if _is_crewai(obj):
        return _from_crewai(obj)
    if _is_autogen(obj):
        return _from_autogen(obj)
    if _is_llamaindex(obj):
        return _from_llamaindex(obj)
    if _is_google_adk(obj):
        return _from_google_adk(obj)
    if _is_strands(obj):
        return _from_strands(obj)
    if _is_temporal(obj):
        return _from_temporal(obj)
    return _from_generic(obj)


# ── LangGraph ─────────────────────────────────────────────────────────────


def _is_langgraph(obj: Any) -> bool:
    """Compiled graph (``.nodes`` + ``.invoke``) or an uncompiled ``StateGraph``
    (``.nodes`` + ``.compile``)."""
    return isinstance(_attr(obj, "nodes"), dict) and (
        callable(_attr(obj, "invoke")) or callable(_attr(obj, "compile"))
    )


def _from_langgraph(app: Any) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    nodes: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    # Node key → the id we publish, so the edges below line up after the
    # __start__/__end__ rewrite and after ToolNode expansion. The two reserved
    # keys are seeded rather than read from ``app.nodes``, because a compiled
    # graph lists ``__start__`` there but never ``__end__`` - relying on the
    # dict would silently drop every edge into the terminal node.
    alias: dict[str, str] = {"__start__": "input", "__end__": "output"}
    # Router → tool edges synthesised when one ToolNode holds several tools.
    edges_from_router: list[tuple[str, str]] = []

    for key, node in _safe_items(_attr(app, "nodes")):
        if key in _LG_RESERVED:
            continue
        node_tools = _node_tools(node)
        if node_tools:
            # A ToolNode is not an agent - it *is* the tools it holds. Publish
            # the tools as nodes so the graph shows what the agent can reach.
            keys: list[str] = []
            for tool in node_tools:
                t = _tool_entry(tool)
                tools.setdefault(t["key"], t)
                _merge_server(servers, tool)
                keys.append(t["key"])
                nodes.append({
                    "id": t["key"], "label": t["name"], "kind": "tool", "ref": f"tool:{t['key']}",
                })
            if len(keys) == 1:
                alias[key] = keys[0]
            else:
                # Fan-out node: keep a router so the incoming edges stay connected.
                alias[key] = key
                nodes.append({"id": key, "label": _label(key), "kind": "router"})
                for tool_key in keys:
                    edges_from_router.append((key, tool_key))
            continue
        agent_node: dict[str, Any] = {
            "id": key,
            "label": _label(key),
            "kind": "router" if _looks_like_router(key) else "agent",
        }
        model = _node_model(node)
        if model:
            agent_node["model"] = model
        nodes.append(agent_node)
        alias[key] = key
        # LangGraph's enforcement adapter gates every executable node by the
        # underlying callable name. Register that exact identity as a tool too,
        # even when the visual node remains an agent/router. Otherwise strict
        # action lookup denies ordinary function nodes as "tool not registered"
        # while only ToolNode contents appear in the catalog.
        fn = _node_callable(node)
        if callable(fn):
            fn_name = _attr(fn, "__name__", None)
            governed_name = (
                fn_name
                if isinstance(fn_name, str)
                and fn_name
                and fn_name != "<lambda>"
                else key
            )
            entry = _tool_entry(fn, name_override=governed_name)
            tools.setdefault(entry["key"], entry)
            agent_node["_capabilities"] = [entry["key"]]
            artifact = entry.get("_code_artifact")
            if isinstance(artifact, dict):
                artifact = dict(artifact)
                artifact["subject_kind"] = "agent"
                artifact["subject_key"] = key
                agent_node["_code_artifact"] = artifact
        else:
            # This is still an executable graph node. An unknown Runnable shape
            # must not disappear from the coverage ledger as though it contained
            # no code.
            agent_node["_code_artifact"] = _source_artifact(
                None,
                subject_kind="agent",
                subject_key=key,
            )

    edges: list[dict[str, Any]] = [
        {"from": alias.get(src, src), "to": alias.get(dst, dst), **extra}
        for src, dst, extra in _langgraph_edges(app)
    ]
    edges += [{"from": src, "to": dst, "kind": "tool"} for src, dst in edges_from_router]
    _merge_server_specs(servers, _declared_servers(app))
    return "langgraph", nodes, edges, list(tools.values()), list(servers.values())


def _langgraph_edges(app: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """Edges from the drawable graph (has __start__/__end__ and conditional
    flags); falls back to the builder's raw edge set + branch specs."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    get_graph = _attr(app, "get_graph")
    drawable = None
    if callable(get_graph):
        try:
            drawable = get_graph()
        except Exception:  # uncompiled builder / unsupported version
            drawable = None
    if drawable is not None:
        for e in _attr(drawable, "edges") or []:
            src = _attr(e, "source")
            dst = _attr(e, "target")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            extra: dict[str, Any] = {}
            label = _attr(e, "data")
            if isinstance(label, str) and label:
                extra["label"] = label
            if _attr(e, "conditional", False):
                extra["kind"] = "conditional"
            out.append((src, dst, extra))
        if out:
            return out
    # Uncompiled StateGraph: `.edges` is an unordered *set* of (src, dst) pairs
    # and conditional edges live in `.branches`, not `.edges` - reading only the
    # set produced a disconnected graph (every conditional hop missing) and a
    # payload whose node/edge order changed run to run. Sort the set and fold in
    # the branch targets so an uncompiled builder describes the same topology a
    # compiled one does.
    builder = _attr(app, "builder", app)
    pairs: list[tuple[str, str]] = []
    for pair in _attr(builder, "edges") or []:
        try:
            src, dst = pair
        except Exception:
            continue
        if isinstance(src, str) and isinstance(dst, str):
            pairs.append((src, dst))
    out += [(src, dst, {}) for src, dst in sorted(pairs)]
    out += _builder_branch_edges(builder)
    return out


def _builder_branch_edges(builder: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """Conditional edges declared on an uncompiled builder.

    ``StateGraph.branches`` is ``{source: {branch_name: BranchSpec}}`` and the
    reachable targets are ``BranchSpec.ends`` (``{path_value: node}``) plus the
    optional ``.then``. Entirely best-effort: an unknown branch shape yields no
    edges rather than an exception."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    branches = _attr(builder, "branches")
    for src, specs in _safe_items(branches):
        for _, spec in _safe_items(specs):
            targets: list[str] = []
            ends = _attr(spec, "ends")
            if isinstance(ends, dict):
                targets += [v for v in ends.values() if isinstance(v, str)]
            elif isinstance(ends, (list, tuple, set)):
                targets += [v for v in ends if isinstance(v, str)]
            then = _attr(spec, "then")
            if isinstance(then, str) and then:
                targets.append(then)
            for dst in sorted(set(targets)):
                out.append((src, dst, {"kind": "conditional"}))
    return out


def _node_callable(node: Any) -> Any:
    """The user function behind a compiled LangGraph node (``PregelNode.bound``
    holds a ``RunnableCallable`` with ``.func`` / ``.afunc``), or the node itself
    when it was added as a plain function."""
    for holder in (_attr(node, "bound", None), _attr(node, "runnable", None), node):
        if holder is None:
            continue
        for attr in ("func", "afunc"):
            fn = _attr(holder, attr, None)
            if callable(fn):
                return fn
    return node if callable(node) else None


def _node_tools(node: Any) -> list[Any]:
    """Tools held by a ToolNode-style node (``tools_by_name`` / ``tools``)."""
    for holder in (node, _attr(node, "bound", None), _attr(node, "runnable", None)):
        if holder is None:
            continue
        by_name = _attr(holder, "tools_by_name", None)
        if isinstance(by_name, dict) and by_name:
            return list(by_name.values())
        tools = _attr(holder, "tools", None)
        if isinstance(tools, (list, tuple)) and tools:
            return list(tools)
    return []


def _node_model(node: Any) -> str:
    """Best-effort model id for an LLM-backed node."""
    holders = [node, _attr(node, "bound", None), _attr(node, "runnable", None)]
    holders += [_attr(h, "bound", None) for h in list(holders) if h is not None]
    for holder in holders:
        if holder is None:
            continue
        for attr in ("model_name", "model_id", "deployment_name", "model"):
            value = _attr(holder, attr, None)
            if isinstance(value, str) and value:
                return value
    return ""


# ── CrewAI ────────────────────────────────────────────────────────────────


def _is_crewai(obj: Any) -> bool:
    return isinstance(_attr(obj, "agents", None), (list, tuple)) and _has(obj, "tasks")


def _from_crewai(crew: Any) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    from .integrations._common import merge_callback_inventories
    from .integrations.crewai import executable_callbacks

    nodes: list[dict[str, Any]] = [{"id": "input", "label": "Input", "kind": "input"}]
    edges: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    previous = "input"

    for index, agent in enumerate(_attr(crew, "agents", None) or []):
        key = _slug(_agent_name(agent))
        if not key:
            continue
        entry: dict[str, Any] = {"id": key, "label": _label(key), "kind": "agent"}
        model = _crew_model(agent)
        if model:
            entry["model"] = model
        _attach_callback_inventory(
            entry,
            executable_callbacks(agent, prefix=f"agent[{index}]"),
        )
        nodes.append(entry)
        edges.append({"from": previous, "to": key})
        previous = key
        for tool in _iter_tools(_attr(agent, "tools", None)):
            t = _tool_entry(tool)
            tools.setdefault(t["key"], t)
            _merge_server(servers, tool)
            nodes.append({
                "id": t["key"], "label": t["name"], "kind": "tool", "ref": f"tool:{t['key']}",
            })
            edges.append({"from": key, "to": t["key"], "kind": "tool"})

    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    edges.append({"from": previous, "to": "output"})
    network_callbacks = [executable_callbacks(crew, prefix="crew")]
    for index, task in enumerate(_attr(crew, "tasks", None) or []):
        network_callbacks.append(
            executable_callbacks(task, prefix=f"task[{index}]")
        )
    _append_callback_router(
        nodes,
        edges,
        merge_callback_inventories(*network_callbacks),
        framework="crewai",
    )
    for tool in _iter_tools(_attr(crew, "tools", None)):
        t = _tool_entry(tool)
        tools.setdefault(t["key"], t)
        _merge_server(servers, tool)
    _merge_server_specs(servers, _declared_servers(crew))
    return "crewai", nodes, edges, list(tools.values()), list(servers.values())


def _crew_model(agent: Any) -> str:
    llm = _attr(agent, "llm", None)
    for holder in (llm, agent):
        if holder is None:
            continue
        for attr in ("model_name", "model", "model_id"):
            value = _attr(holder, attr, None)
            if isinstance(value, str) and value:
                return value
    return llm if isinstance(llm, str) else ""


def _agent_name(agent: Any) -> str:
    for attr in ("name", "role", "id"):
        value = _attr(agent, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    fn_name = _attr(agent, "__name__", None)  # a bare function used as an agent
    if isinstance(fn_name, str) and fn_name and fn_name != "<lambda>":
        return fn_name
    return type(agent).__name__


# ── AutoGen ───────────────────────────────────────────────────────────────


def _is_autogen(obj: Any) -> bool:
    """A team/group chat: a list of participants and no CrewAI-style tasks."""
    participants = _attr(obj, "_participants", None) or _attr(obj, "agents", None)
    return isinstance(participants, (list, tuple)) and bool(participants) and not _has(obj, "tasks")


def _from_autogen(team: Any) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    from .integrations.autogen import executable_callbacks

    participants = _attr(team, "_participants", None) or _attr(team, "agents", None) or []
    nodes: list[dict[str, Any]] = [{"id": "input", "label": "Input", "kind": "input"}]
    edges: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    previous = "input"

    for index, agent in enumerate(participants):
        key = _slug(_agent_name(agent))
        if not key:
            continue
        entry: dict[str, Any] = {"id": key, "label": _label(key), "kind": "agent"}
        model = _autogen_model(agent)
        if model:
            entry["model"] = model
        _attach_callback_inventory(
            entry,
            executable_callbacks(agent, prefix=f"participant[{index}]"),
        )
        nodes.append(entry)
        edges.append({"from": previous, "to": key})
        previous = key
        for tool in _iter_tools(_attr(agent, "_tools", None) or _attr(agent, "tools", None)):
            t = _tool_entry(tool)
            tools.setdefault(t["key"], t)
            _merge_server(servers, tool)
            nodes.append({
                "id": t["key"], "label": t["name"], "kind": "tool", "ref": f"tool:{t['key']}",
            })
            edges.append({"from": key, "to": t["key"], "kind": "tool"})

    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    edges.append({"from": previous, "to": "output"})
    _append_callback_router(
        nodes,
        edges,
        executable_callbacks(team, prefix="team"),
        framework="autogen",
    )
    _merge_server_specs(servers, _declared_servers(team))
    return "autogen", nodes, edges, list(tools.values()), list(servers.values())


def _autogen_model(agent: Any) -> str:
    client = _attr(agent, "_model_client", None) or _attr(agent, "model_client", None)
    for holder in (client, agent):
        if holder is None:
            continue
        for attr in ("model", "model_name", "model_id"):
            value = _attr(holder, attr, None)
            if isinstance(value, str) and value:
                return value
    return ""


# ── LlamaIndex ────────────────────────────────────────────────────────────


def _is_llamaindex(obj: Any) -> bool:
    if any(_has(obj, a) for a in ("agent_worker", "_agent_worker", "tool_retriever")):
        return True
    return "llama_index" in _attr(type(obj), "__module__", "")


def _from_llamaindex(agent: Any) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    from .integrations.llamaindex import executable_callbacks

    key = _slug(_agent_name(agent)) or "llamaindex-agent"
    entry: dict[str, Any] = {"id": key, "label": _label(key), "kind": "agent"}
    model = _crew_model(agent)
    if model:
        entry["model"] = model
    _attach_callback_inventory(
        entry,
        executable_callbacks(agent, prefix="agent"),
    )
    nodes: list[dict[str, Any]] = [
        {"id": "input", "label": "Input", "kind": "input"},
        entry,
    ]
    edges: list[dict[str, Any]] = [{"from": "input", "to": key}]
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}

    raw = _attr(agent, "tools", None)
    if raw is None:
        worker = _attr(agent, "agent_worker", None) or _attr(agent, "_agent_worker", None)
        raw = _attr(worker, "tools", None) or _attr(agent, "_tools", None)
    for tool in _iter_tools(raw):
        t = _tool_entry(tool)
        tools.setdefault(t["key"], t)
        _merge_server(servers, tool)
        nodes.append({"id": t["key"], "label": t["name"], "kind": "tool", "ref": f"tool:{t['key']}"})
        edges.append({"from": key, "to": t["key"], "kind": "tool"})

    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    edges.append({"from": key, "to": "output"})
    _merge_server_specs(servers, _declared_servers(agent))
    return "llamaindex", nodes, edges, list(tools.values()), list(servers.values())


# ── Google ADK / Strands / Temporal ─────────────────────────────────────


def _module_name(obj: Any) -> str:
    return str(_attr(type(obj), "__module__", "") or "")


def _is_google_adk(obj: Any) -> bool:
    """An ADK runner owns ``.agent``; direct ADK agents are accepted too."""
    if _module_name(obj).startswith("google.adk"):
        return True
    root = _attr(obj, "agent", None)
    return root is not None and _module_name(root).startswith("google.adk")


def _from_google_adk(
    runner_or_agent: Any,
) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    """Describe an ADK runner's complete root/sub-agent/tool tree."""
    from .integrations.google_adk import executable_callbacks

    root = _attr(runner_or_agent, "agent", None) or runner_or_agent
    nodes: list[dict[str, Any]] = [
        {"id": "input", "label": "Input", "kind": "input"}
    ]
    edges: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    seen: set[int] = set()

    def visit(agent: Any, parent: str) -> str:
        if agent is None or id(agent) in seen:
            return ""
        seen.add(id(agent))
        name = _agent_name(agent)
        key = _key(name) or "google-adk-agent"
        entry: dict[str, Any] = {
            "id": key,
            "label": name or _label(key),
            "kind": "agent",
        }
        model = _framework_model(agent)
        if model:
            entry["model"] = model
        _attach_callback_inventory(
            entry,
            executable_callbacks(agent, prefix=f"agent.{key}"),
        )
        nodes.append(entry)
        edges.append({"from": parent, "to": key})

        for tool in _iter_tools(_attr(agent, "tools", None)):
            tool_entry = _tool_entry(tool)
            tools.setdefault(tool_entry["key"], tool_entry)
            _merge_server(servers, tool)
            nodes.append(
                {
                    "id": tool_entry["key"],
                    "label": tool_entry["name"],
                    "kind": "tool",
                    "ref": f"tool:{tool_entry['key']}",
                }
            )
            edges.append(
                {"from": key, "to": tool_entry["key"], "kind": "tool"}
            )
        for child in _iter_tools(_attr(agent, "sub_agents", None)):
            visit(child, key)
        return key

    root_key = visit(root, "input")
    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    if root_key:
        edges.append({"from": root_key, "to": "output"})
    if runner_or_agent is not root:
        _append_callback_router(
            nodes,
            edges,
            executable_callbacks(runner_or_agent, prefix="runner"),
            framework="google-adk",
        )
    _merge_server_specs(servers, _declared_servers(runner_or_agent))
    return "google_adk", nodes, edges, list(tools.values()), list(servers.values())


def _is_strands(obj: Any) -> bool:
    return _module_name(obj).startswith("strands") and _has(obj, "tool_registry")


def _from_strands(
    agent: Any,
) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    from .integrations.strands import executable_callbacks

    key = _key(_agent_name(agent)) or "strands-agent"
    nodes: list[dict[str, Any]] = [
        {"id": "input", "label": "Input", "kind": "input"},
        {"id": key, "label": _agent_name(agent), "kind": "agent"},
    ]
    model = _framework_model(agent)
    if model:
        nodes[1]["model"] = model
    _attach_callback_inventory(
        nodes[1],
        executable_callbacks(agent, prefix="agent"),
    )
    edges: list[dict[str, Any]] = [{"from": "input", "to": key}]
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    registry = _attr(_attr(agent, "tool_registry", None), "registry", None)
    for tool in _iter_tools(registry):
        entry = _tool_entry(tool)
        tools.setdefault(entry["key"], entry)
        _merge_server(servers, tool)
        nodes.append(
            {
                "id": entry["key"],
                "label": entry["name"],
                "kind": "tool",
                "ref": f"tool:{entry['key']}",
            }
        )
        edges.append({"from": key, "to": entry["key"], "kind": "tool"})
    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    edges.append({"from": key, "to": "output"})
    _merge_server_specs(servers, _declared_servers(agent))
    return "strands", nodes, edges, list(tools.values()), list(servers.values())


def _is_temporal(obj: Any) -> bool:
    return _module_name(obj).startswith("temporalio.worker") and (
        isinstance(_attr(obj, "_config", None), dict)
        or isinstance(_attr(obj, "_initial_config", None), dict)
    )


def _from_temporal(
    worker: Any,
) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    """Model one worker as its registered workflows reaching its activities."""
    config = _attr(worker, "_config", None)
    if not isinstance(config, dict):
        config = _attr(worker, "_initial_config", None)
    if not isinstance(config, dict):
        config = {}
    workflows = _iter_tools(config.get("workflows"))
    activities = _iter_tools(config.get("activities"))
    task_queue = str(config.get("task_queue") or "").strip()

    nodes: list[dict[str, Any]] = [
        {"id": "input", "label": "Input", "kind": "input"}
    ]
    edges: list[dict[str, Any]] = []
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    agent_keys: list[str] = []
    for workflow in workflows:
        name = _agent_name(workflow)
        key = _key(name)
        if not key:
            continue
        agent_keys.append(key)
        node = {
            "id": key,
            "label": name,
            "kind": "agent",
            # Temporal workflow definitions are application executables, not
            # merely labels in the graph. The gateway uses this closed marker
            # to require exact source coverage for every workflow node.
            "executable": True,
        }
        artifact = _source_artifact(
            workflow,
            subject_kind="agent",
            subject_key=key,
        )
        if artifact:
            node["_code_artifact"] = artifact
        nodes.append(node)
        edges.append({"from": "input", "to": key})
    if not agent_keys:
        key = _key(task_queue) or "temporal-worker"
        agent_keys.append(key)
        nodes.append(
            {
                "id": key,
                "label": task_queue or "Temporal Worker",
                "kind": "agent",
                "executable": False,
                "ref": "structural:temporal-worker",
            }
        )
        edges.append({"from": "input", "to": key})

    for activity in activities:
        entry = _tool_entry(activity)
        tools.setdefault(entry["key"], entry)
        nodes.append(
            {
                "id": entry["key"],
                "label": entry["name"],
                "kind": "tool",
                "ref": f"tool:{entry['key']}",
            }
        )
        for agent_key in agent_keys:
            edges.append(
                {"from": agent_key, "to": entry["key"], "kind": "activity"}
            )
    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    for agent_key in agent_keys:
        edges.append({"from": agent_key, "to": "output"})
    return "temporal", nodes, edges, list(tools.values()), list(servers.values())


def _framework_model(agent: Any) -> str:
    model = _attr(agent, "model", None)
    if isinstance(model, str) and model:
        return model
    for attr in ("model_id", "model_name", "name"):
        value = _attr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


# ── Generic ───────────────────────────────────────────────────────────────


def _from_generic(obj: Any) -> tuple[str, list[dict], list[dict], list[dict], list[dict]]:
    """Anything else: reuse the blueprint manifest (which already handles
    PydanticAI, the OpenAI Agents SDK, plain tool lists and single callables)
    and lift its flat tool surface into a one-agent network."""
    # A scalar has no tool surface at all. Without this, ``describe`` names the
    # value itself as a tool ("None", "42") and the registry grows a row for it.
    if obj is None or isinstance(obj, (bool, int, float, complex, bytes, str)):
        return "generic", [], [], [], []

    try:
        from .manifest import describe

        manifest = describe(obj)
        framework = manifest.framework
        names = [n for n in manifest.tools if isinstance(n, str) and not _is_repr(n)]
        raw_edges = [list(e) for e in manifest.edges]
        source_by_name = dict(manifest.tool_sources)
    except Exception:  # pragma: no cover - describe is already fail-soft
        framework, names, raw_edges, source_by_name = "generic", [], [], {}

    # ``manifest.describe`` uses "crewai" as its catch-all framework, which is
    # right for a bare list of CrewAI/LangChain tools but a lie for an object we
    # could not identify at all - the Registry would file it under a framework
    # the operator never used. Relabel only the genuinely unidentifiable case.
    if framework == "crewai" and not names and _attr(obj, "tools") is None:
        framework = "generic"

    raw_tools = obj if isinstance(obj, (list, tuple, set)) else _attr(obj, "tools", None)
    tool_items = _iter_tools(raw_tools)
    # A bare tool list has no agent of its own - name the holder after the
    # framework so the network still has the one agent node the graph needs.
    holder = "" if isinstance(obj, (list, tuple, set)) else _agent_name(obj)
    key = _key(holder) or f"{framework}-agent"
    agent_node: dict[str, Any] = {"id": key, "label": _label(key), "kind": "agent"}
    if callable(obj):
        agent_node["executable"] = True
        artifact = _source_artifact(
            obj,
            subject_kind="agent",
            subject_key=key,
        )
        if artifact:
            agent_node["_code_artifact"] = artifact
    elif tool_items:
        # This node is only a visual/topological holder for a bare collection
        # of tools. Its closed ref lets the gateway exempt it without turning
        # arbitrary unscanned agent nodes into a compatibility escape hatch.
        agent_node["executable"] = False
        agent_node["ref"] = "structural:generic-tool-holder"
    nodes: list[dict[str, Any]] = [
        {"id": "input", "label": "Input", "kind": "input"},
        agent_node,
    ]
    edges: list[dict[str, Any]] = [{"from": "input", "to": key}]
    tools: dict[str, dict[str, Any]] = {}
    servers: dict[str, dict[str, Any]] = {}
    for tool in tool_items:
        entry = _tool_entry(tool)
        tools.setdefault(entry["key"], entry)
        _merge_server(servers, tool)
    for name in names:
        tool_key = _key(name)
        if not tool_key:
            continue
        action_class = _action_class(name)
        fallback_entry: dict[str, Any] = {
            "key": tool_key,
            "name": name,
            "action_class": action_class,
            "actions": [_default_action(name, action_class=action_class)],
        }
        source = source_by_name.get(name, "")
        if source:
            artifact = _source_artifact(
                None,
                subject_kind="tool",
                subject_key=tool_key,
                source=source,
            )
            if artifact:
                fallback_entry["_code_artifact"] = artifact
        else:
            # ``manifest.describe`` identified an executable tool name but could
            # not recover the implementation. Make that absence explicit so the
            # gateway cannot approve a source-bearing subset as complete.
            fallback_entry["_code_artifact"] = _source_artifact(
                None,
                subject_kind="tool",
                subject_key=tool_key,
            )
        tools.setdefault(tool_key, fallback_entry)
        nodes.append({"id": tool_key, "label": name, "kind": "tool", "ref": f"tool:{tool_key}"})
        edges.append({"from": key, "to": tool_key, "kind": "tool"})
    for pair in raw_edges:
        if len(pair) == 2 and all(isinstance(p, str) for p in pair):
            edges.append({"from": _key(pair[0]), "to": _key(pair[1])})
    nodes.append({"id": "output", "label": "Output", "kind": "output"})
    edges.append({"from": key, "to": "output"})

    _merge_server_specs(servers, _declared_servers(obj))
    return framework, nodes, edges, list(tools.values()), list(servers.values())


# ── Plain dict manifest ───────────────────────────────────────────────────

# Keys this normaliser owns; anything else in the caller's dict is passed
# through untouched (``principal``, ``session_id``, custom server fields, …).
_MANIFEST_OWNED = frozenset(
    {
        "framework",
        "name",
        "network",
        "nodes",
        "edges",
        "agents",
        "tools",
        "servers",
        "code_artifacts",
        "code_artifact_coverage",
        "code_artifacts_incomplete",
        "self_attested_code_digest",
        "blueprint_digest",
    }
)


def _from_manifest(raw: dict[str, Any], *, name: str, framework: str) -> dict[str, Any]:
    """Normalise a hand-built (or cached) manifest dict into a valid payload.

    Every field is coerced to the shape ``/registry/discover`` expects and the
    **digest is always recomputed from the normalised topology** rather than
    trusted from the input - which is what makes ``discover(manifest=…)`` share
    the same idempotent-upsert / drift-detection semantics as a live graph, and
    what lets the 5-minute dedupe apply to it at all (a caller-supplied payload
    with no digest previously bypassed dedupe entirely and re-posted forever).

    Unknown top-level keys survive verbatim, so a payload that already carries
    ``principal`` / ``auto_provision`` is not silently stripped."""
    tools = [t for t in (_manifest_tool(t) for t in _as_seq(raw.get("tools"))) if t]
    embedded_artifacts = [
        artifact
        for tool in tools
        for artifact in (tool.pop("_code_artifact", None),)
        if isinstance(artifact, dict)
    ]
    # `tools` is authoritative for what a node IS: a node listed as a bare string
    # (or with no explicit kind) that also appears in `tools` is a tool, not an
    # agent - otherwise every tool named in both places would be counted as an
    # agent in the registry rollups.
    tool_keys = {t["key"] for t in tools}
    nodes = [n for n in (_manifest_node(n, tool_keys) for n in _as_seq(raw.get("nodes"))) if n]
    # A tool named only in `tools` still deserves a node, so the graph the
    # operator sees matches the tools the registry records.
    known = {n["id"] for n in nodes}
    for tool in tools:
        if tool["key"] not in known:
            nodes.append({
                "id": tool["key"], "label": tool["name"], "kind": "tool",
                "ref": f"tool:{tool['key']}",
            })
            known.add(tool["key"])
    edges = [e for e in (_manifest_edge(e) for e in _as_seq(raw.get("edges"))) if e]
    nodes, edges = _normalise_topology(nodes, edges)

    servers: dict[str, dict[str, Any]] = {}
    _merge_server_specs(servers, [s for s in (_server_spec(s) for s in _as_seq(raw.get("servers"))) if s])

    declared = [a for a in (_manifest_agent(a) for a in _as_seq(raw.get("agents"))) if a]
    agents = declared or _agents_from(nodes, edges)

    artifacts: list[dict[str, Any]] = []
    artifacts_incomplete = raw.get("code_artifacts_incomplete") is True
    used_artifact_bytes = 0
    known_agents = {str(agent.get("key") or "") for agent in agents}
    known_agents.update(
        str(node.get("id") or "")
        for node in nodes
        if node.get("kind") in {"agent", "router"}
    )
    known_tools = {str(tool.get("key") or "") for tool in tools}
    known_tools.update(
        str(node.get("id") or "")
        for node in nodes
        if node.get("kind") == "tool"
    )
    seen_artifacts: set[tuple[str, str, str]] = set()
    coverage: dict[tuple[str, str], dict[str, str]] = {}
    for raw_entry in _as_seq(raw.get("code_artifact_coverage")):
        entry = _coverage_entry(raw_entry)
        if not entry:
            artifacts_incomplete = True
            continue
        known_subjects = known_agents if entry["subject_kind"] == "agent" else known_tools
        if entry["subject_key"] not in known_subjects:
            artifacts_incomplete = True
            continue
        _merge_coverage(coverage, entry)

    for candidate in [*embedded_artifacts, *_as_seq(raw.get("code_artifacts"))]:
        artifact = _normalise_code_artifact(candidate)
        if not artifact:
            artifacts_incomplete = True
            _merge_coverage(coverage, _coverage_entry(candidate))
            continue
        if artifact.get("_incomplete") is True:
            artifacts_incomplete = True
        if "source" not in artifact:
            _merge_coverage(coverage, _coverage_entry(artifact))
            continue
        if artifact["subject_kind"] == "agent":
            if artifact["subject_key"] not in known_agents:
                artifacts_incomplete = True
                continue
        elif artifact["subject_key"] not in known_tools:
            artifacts_incomplete = True
            continue
        identity = (
            artifact["subject_kind"],
            artifact["subject_key"],
            artifact["digest"],
        )
        if identity in seen_artifacts:
            _merge_coverage(
                coverage,
                _coverage_entry(
                    artifact,
                    status=str(artifact.get("_coverage_status") or "captured"),
                ),
            )
            continue
        size = len(artifact["source"].encode("utf-8", "replace"))
        if (
            len(artifacts) >= _CODE_ARTIFACT_MAX_COUNT
            or used_artifact_bytes + size > _CODE_BUNDLE_MAX_BYTES
        ):
            artifacts_incomplete = True
            _merge_coverage(
                coverage,
                _coverage_entry({**artifact, "_coverage_status": "omitted"}),
            )
            continue
        status = str(artifact.get("_coverage_status") or "captured")
        artifact.pop("_incomplete", None)
        artifact.pop("_coverage_status", None)
        artifacts.append(artifact)
        seen_artifacts.add(identity)
        used_artifact_bytes += size
        _merge_coverage(coverage, _coverage_entry(artifact, status=status))

    accepted_subjects = {
        (artifact["subject_kind"], artifact["subject_key"]) for artifact in artifacts
    }
    # The server requires exact coverage for every declared local executable.
    # A remote MCP tool's implementation lives behind its server boundary and
    # is intentionally exempt from client-side Python source capture.
    for tool in tools:
        key = str(tool.get("key") or "")
        identity = ("tool", key)
        if key and not tool.get("server") and identity not in accepted_subjects:
            _merge_coverage(
                coverage,
                {"subject_kind": "tool", "subject_key": key, "status": "missing"},
            )
            artifacts_incomplete = True
    for identity, entry in list(coverage.items()):
        if entry["status"] == "captured" and identity not in accepted_subjects:
            coverage[identity] = {**entry, "status": "missing"}
    ledger = [coverage[identity] for identity in sorted(coverage)]
    artifacts_incomplete = artifacts_incomplete or any(
        entry["status"] != "captured" for entry in ledger
    )

    net = raw.get("network") if isinstance(raw.get("network"), dict) else {}
    fw = _slug(framework or net.get("framework") or raw.get("framework") or "") or "manifest"
    digest = _digest(nodes, edges)
    net_name = str(name or net.get("name") or raw.get("name") or "").strip()
    key = _slug(name or net.get("key") or net_name) or f"{fw}-{digest[7:19]}"

    payload: dict[str, Any] = {
        k: v for k, v in raw.items() if k not in _MANIFEST_OWNED
    }
    payload.update({
        "framework": fw,
        "network": {"key": key, "name": net_name or key, "digest": digest},
        "nodes": nodes,
        "edges": edges,
        "agents": agents,
        "tools": tools,
        "servers": list(servers.values()),
        "code_artifacts": artifacts,
        "blueprint_digest": _blueprint_digest(
            digest,
            artifacts,
            incomplete=artifacts_incomplete,
            coverage=ledger,
        ),
    })
    if ledger:
        payload["code_artifact_coverage"] = ledger
    attestation = _self_attested_code_digest(artifacts, ledger)
    if attestation:
        payload["self_attested_code_digest"] = attestation
    if artifacts_incomplete:
        payload["code_artifacts_incomplete"] = True
    return payload


def _as_seq(raw: Any) -> list[Any]:
    """A manifest collection: a list/tuple, the values of a keyed dict, or a
    single entry. ``None`` and scalars-that-aren't-entries yield nothing."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    return [raw]


def _manifest_node(raw: Any, tool_keys: frozenset[str] | set[str] = frozenset()) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"id": raw}
    if not isinstance(raw, dict):
        return {}
    node_id = _key(raw.get("id") or raw.get("key") or raw.get("name") or "")
    if not node_id:
        return {}
    declared_kind = str(raw.get("kind") or "").strip()
    kind = declared_kind or ("tool" if node_id in tool_keys else "agent")
    node: dict[str, Any] = {
        "id": node_id,
        "label": str(raw.get("label") or raw.get("name") or _label(node_id)),
        "kind": kind,
    }
    if kind == "tool":
        node["ref"] = f"tool:{node_id}"
    for optional in ("ref", "model"):
        if raw.get(optional):
            node[optional] = str(raw[optional])
    if isinstance(raw.get("executable"), bool):
        node["executable"] = raw["executable"]
    return node


def _manifest_edge(raw: Any) -> dict[str, Any]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        raw = {"from": raw[0], "to": raw[1]}
    if not isinstance(raw, dict):
        return {}
    src = _key(raw.get("from") or raw.get("source") or "")
    dst = _key(raw.get("to") or raw.get("target") or "")
    if not src or not dst:
        return {}
    edge: dict[str, Any] = {"from": src, "to": dst}
    for optional in ("kind", "label"):
        if raw.get(optional):
            edge[optional] = str(raw[optional])
    return edge


def _manifest_tool(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        key = _key(raw)
        if not key:
            return {}
        action_class = _action_class(raw)
        return {
            "key": key,
            "name": raw,
            "action_class": action_class,
            "actions": [_default_action(raw, action_class=action_class)],
        }
    if raw is None or isinstance(raw, (bool, int, float, complex, bytes)):
        return {}  # a scalar in `tools` is a typo, not a tool
    if not isinstance(raw, dict):
        return _tool_entry(raw)  # a live tool object mixed into a manifest
    name = str(raw.get("name") or raw.get("key") or "")
    key = _key(raw.get("key") or name)
    if not key:
        return {}
    action_class = str(
        raw.get("action_class") or _action_class(name or key)
    ).strip().lower()
    tool: dict[str, Any] = {
        "key": key,
        "name": name or key,
        "action_class": action_class,
    }
    for optional in ("description", "schema_digest"):
        if raw.get(optional):
            tool[optional] = str(raw[optional])[:500]
    # ``server`` is the Go DiscoverManifest wire key. Accept the SDK's former
    # ``server_key`` spelling when normalising cached/user-authored manifests,
    # but always emit the canonical contract.
    server = raw.get("server") or raw.get("server_key")
    if server:
        spec = _server_spec(server)
        tool["server"] = spec.get("key", "") if spec else _slug(server)
    action_source = raw.get("actions", _MISSING)
    if action_source is _MISSING:
        action_source = raw.get("operations", _MISSING)
    if action_source is _MISSING:
        tool["actions"] = [
            _default_action(
                name or key,
                action_class=action_class,
                schema_digest=str(tool.get("schema_digest") or ""),
                description=str(tool.get("description") or ""),
            )
        ]
    else:
        actions = [
            action
            for action in (
                _manifest_action(
                    item,
                    default_class=action_class,
                    default_schema=str(tool.get("schema_digest") or ""),
                )
                for item in _action_items(action_source)
            )
            if action
        ]
        tool["actions"] = actions or [
            _default_action(
                name or key,
                action_class=action_class,
                schema_digest=str(tool.get("schema_digest") or ""),
                description=str(tool.get("description") or ""),
            )
        ]
    return tool


def _manifest_action(
    raw: Any,
    *,
    default_class: str = "",
    default_schema: str = "",
) -> dict[str, Any]:
    """Normalise one SDK operation to Go's ``ManifestAction`` contract."""
    if isinstance(raw, str):
        raw = {"key": raw, "name": raw}
    elif not isinstance(raw, dict):
        raw = {
            "key": _attr(raw, "key", None) or _attr(raw, "name", None),
            "name": _attr(raw, "name", None),
            "description": _attr(raw, "description", None),
            "action_class": _attr(raw, "action_class", None),
            "risk": _attr(raw, "risk", None),
            "schema_digest": _attr(raw, "schema_digest", None),
        }
    name = str(raw.get("name") or raw.get("key") or "").strip()
    key = _registry_key(raw.get("key") or name)
    if not key:
        return {}
    action_class = str(
        raw.get("action_class")
        or (default_class if key == "invoke" else _action_class(name or key))
        or "write"
    ).strip().lower()
    action: dict[str, Any] = {
        "key": key,
        "name": name or _label(key),
        "action_class": action_class,
    }
    description = str(raw.get("description") or "").strip()
    if description:
        action["description"] = description[:500]
    risk = str(raw.get("risk") or "").strip().lower()
    if risk:
        action["risk"] = risk
    schema_digest = str(raw.get("schema_digest") or default_schema).strip()
    if schema_digest:
        action["schema_digest"] = schema_digest[:500]
    return action


def _action_items(raw: Any) -> list[Any]:
    """Preserve keys in ``{"read": {...}}`` action maps."""
    if isinstance(raw, dict):
        out: list[Any] = []
        for key, value in raw.items():
            if isinstance(value, dict):
                out.append({"key": key, **value})
            elif isinstance(value, str):
                out.append({"key": key, "name": value})
            else:
                out.append({"key": key})
        return out
    return _as_seq(raw)


def _default_action(
    tool_name: str,
    *,
    action_class: str,
    schema_digest: str = "",
    description: str = "",
) -> dict[str, Any]:
    """One callable tool is one discoverable operation by default."""
    raw: dict[str, Any] = {
        "key": "invoke",
        "name": f"Invoke {tool_name}",
        "action_class": action_class,
        "schema_digest": schema_digest,
        "description": description,
    }
    return _manifest_action(
        raw,
        default_class=action_class,
        default_schema=schema_digest,
    )


def _manifest_agent(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        key = _key(raw)
        return {"key": key, "name": raw, "capabilities": []} if key else {}
    if not isinstance(raw, dict):
        return {}
    name = str(raw.get("name") or raw.get("key") or "")
    key = _key(raw.get("key") or name)
    if not key:
        return {}
    caps = [c for c in (_key(c) for c in _as_seq(raw.get("capabilities"))) if c]
    agent: dict[str, Any] = {"key": key, "name": name or key, "capabilities": sorted(set(caps))}
    for optional in ("model", "description", "owner", "version"):
        if raw.get(optional):
            agent[optional] = str(raw[optional])
    return agent


# ── Shared extraction helpers ─────────────────────────────────────────────


def _iter_tools(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    return [raw]


def _tool_name(tool: Any) -> str:
    for attr in ("name", "tool_name"):
        name = _attr(tool, attr, None)
        if isinstance(name, str) and name:
            return name
    meta = _attr(tool, "metadata", None)
    meta_name = _attr(meta, "name", None)
    if isinstance(meta_name, str) and meta_name:
        return meta_name
    fn = _attr(tool, "func", None) or _attr(tool, "fn", None) or tool
    fn_name = _attr(fn, "__name__", None)
    if isinstance(fn_name, str) and fn_name and fn_name != "<lambda>":
        return fn_name
    return type(tool).__name__


def _tool_entry(tool: Any, *, name_override: str = "") -> dict[str, Any]:
    """A registry tool row: key + name + read/write class + a schema digest that
    changes when the tool's argument contract changes (drift signal)."""
    name = str(name_override or "").strip() or _tool_name(tool)
    metadata = _attr(tool, "metadata", None)
    declared_class = _attr(tool, "action_class", None)
    if not declared_class and isinstance(metadata, dict):
        declared_class = metadata.get("action_class")
    action_class = str(declared_class or _action_class(name)).strip().lower()
    entry: dict[str, Any] = {
        "key": _key(name) or "tool",
        "name": name,
        "action_class": action_class,
    }
    description = _attr(tool, "description", None)
    if not isinstance(description, str) or not description:
        doc = _attr(_attr(tool, "func", None) or tool, "__doc__", None)
        description = doc.strip().splitlines()[0] if isinstance(doc, str) and doc.strip() else ""
    if description:
        entry["description"] = description[:500]
    schema = _schema_digest(tool)
    if schema:
        entry["schema_digest"] = schema
    action_source = _attr(tool, "actions", _MISSING)
    if action_source is _MISSING:
        action_source = _attr(tool, "operations", _MISSING)
    if action_source is _MISSING and isinstance(metadata, dict):
        action_source = metadata.get("actions", metadata.get("operations", _MISSING))
    if action_source is _MISSING:
        entry["actions"] = [
            _default_action(
                name,
                action_class=action_class,
                schema_digest=schema,
                description=description,
            )
        ]
    else:
        actions = [
            action
            for action in (
                _manifest_action(
                    item,
                    default_class=action_class,
                    default_schema=schema,
                )
                for item in _action_items(action_source)
            )
            if action
        ]
        # An explicitly empty/malformed operation list cannot become an
        # actionless compatibility escape hatch. Treat the callable itself as
        # one conservative operation, exactly like a hand-authored manifest.
        entry["actions"] = actions or [
            _default_action(
                name,
                action_class=action_class,
                schema_digest=schema,
                description=description,
            )
        ]
    server = _tool_server_spec(tool)
    if server:
        entry["server"] = server["key"]
    fn = _tool_callable(tool)
    # Remote MCP tools have no client-side implementation to inspect. Every
    # other named tool is a local executable candidate and therefore receives
    # an explicit missing ledger entry when wrapper extraction fails.
    if fn is not None or not server:
        artifact = _source_artifact(
            fn,
            subject_kind="tool",
            subject_key=entry["key"],
        )
        if artifact:
            entry["_code_artifact"] = artifact
    return entry


def _tool_callable(tool: Any) -> Any:
    """Best-effort application callable behind common framework wrappers."""
    for holder in (tool, _attr(tool, "bound", None), _attr(tool, "runnable", None)):
        if holder is None:
            continue
        for attr in (
            "func",
            "fn",
            "_run",
            "_arun",
            "on_invoke_tool",
            "callable",
        ):
            candidate = _attr(holder, attr, None)
            if callable(candidate):
                return candidate
    return tool if callable(tool) else None


def _schema_digest(tool: Any) -> str:
    """``sha256[:16]`` over the complete canonical argument schema.

    Property names alone are insufficient: changing ``amount`` from integer to
    string, requiredness, a nested contract, or a default is real drift.  This
    hashes the full declaration while still never touching invocation values.
    """
    for attr in ("args_schema", "args", "input_schema", "parameters"):
        raw = _attr(tool, attr, None)
        if raw is None:
            continue
        try:
            if _has(raw, "model_json_schema"):
                schema = raw.model_json_schema()
            elif _has(raw, "json_schema"):
                schema = raw.json_schema()
            else:
                schema = raw
            text = json.dumps(
                schema,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=_schema_json_default,
            )
        except Exception:  # pragma: no cover - exotic schema object
            continue
        if text:
            return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    return ""


def _action_class(name: str) -> str:
    return infer_action_class(name)


def _schema_json_default(value: Any) -> str:
    """Stable fallback for exotic schema leaves (never include object repr ids)."""
    module = _attr(type(value), "__module__", "")
    qualname = _attr(type(value), "__qualname__", type(value).__name__)
    return f"{module}.{qualname}".strip(".")


def _tool_server_spec(tool: Any) -> dict[str, Any]:
    for attr in ("server_name", "mcp_server", "server"):
        raw = _attr(tool, attr, None)
        if raw is None:
            continue
        spec = _server_spec(raw)
        if spec:
            return spec
    return {}


def _merge_server(servers: dict[str, dict[str, Any]], tool: Any) -> str:
    """Record the MCP server a tool came from, when the adapter exposes one."""
    spec = _tool_server_spec(tool)
    if spec:
        servers.setdefault(spec["key"], spec)
        return str(spec["key"])
    return ""


def _merge_server_specs(servers: dict[str, dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        servers.setdefault(spec["key"], spec)


def _declared_servers(obj: Any) -> list[dict[str, Any]]:
    """MCP servers declared on the object itself (``mcp_servers`` on a graph /
    crew / agent, whatever shape the adapter used)."""
    out: list[dict[str, Any]] = []
    for attr in ("mcp_servers", "_mcp_servers", "servers"):
        raw = _attr(obj, attr, None)
        if raw is None:
            continue
        for item in _iter_tools(raw):
            spec = _server_spec(item)
            if spec:
                out.append(spec)
        if out:
            break
    return out


def _server_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            parsed = urlsplit(raw.strip())
        except Exception:
            parsed = None
        if parsed is not None and parsed.scheme and parsed.netloc:
            safe_url = _safe_server_url(raw)
            name = parsed.hostname or safe_url
            key = _slug(name)
            if key:
                return {
                    "key": key,
                    "name": name,
                    "transport": (
                        "http"
                        if parsed.scheme.lower() in ("http", "https")
                        else parsed.scheme.lower()
                    ),
                    "url": safe_url,
                }
        key = _slug(raw)
        return {"key": key, "name": raw, "transport": "http"} if key else {}
    if isinstance(raw, dict):
        name = str(raw.get("name") or raw.get("key") or "")
        key = _slug(raw.get("key") or name)
        if not key:
            return {}
        spec = {"key": key, "name": name or key, "transport": str(raw.get("transport") or "http")}
        if raw.get("url"):
            safe_url = _safe_server_url(raw["url"])
            if safe_url:
                spec["url"] = safe_url
        return spec
    name = _attr(raw, "name", None)
    if isinstance(name, str) and name:
        spec = {"key": _slug(name), "name": name, "transport": "http"}
        url = _attr(raw, "url", None)
        if isinstance(url, str) and url:
            safe_url = _safe_server_url(url)
            if safe_url:
                spec["url"] = safe_url
        transport = _attr(raw, "transport", None)
        if isinstance(transport, str) and transport:
            spec["transport"] = transport
        return spec
    return {}


def _safe_server_url(raw: Any) -> str:
    """Return connection metadata without credentials or query secrets.

    Registry discovery is structural telemetry, not a secret store. MCP URLs
    often carry bearer tokens in user-info, query parameters, or fragments; the
    host/path is enough to identify and render the server.
    """
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.netloc:
            return value.split("?", 1)[0].split("#", 1)[0]
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except Exception:
        return value.split("?", 1)[0].split("#", 1)[0]


def _is_repr(name: str) -> bool:
    """True for a Python default repr, which is never a real tool name and is
    never stable across processes."""
    return bool(_REPR_RE.search(name.strip()))


def _safe_items(mapping: Any) -> list[tuple[str, Any]]:
    try:
        return [(k, v) for k, v in mapping.items() if isinstance(k, str)]
    except Exception:  # pragma: no cover - exotic mapping
        return []


def _looks_like_router(key: str) -> bool:
    lowered = key.lower()
    return any(h in lowered for h in ("router", "route", "branch", "supervisor", "dispatch"))


def _network_name(obj: Any) -> str:
    """A human name for the network, ignoring the framework's generic default
    (``CompiledStateGraph.name`` is just "LangGraph")."""
    for attr in ("name", "graph_name"):
        value = _attr(obj, attr, None)
        if isinstance(value, str) and value.strip():
            if value.strip().lower() in ("langgraph", "stategraph", "crew", "agent"):
                continue
            return value.strip()
    return ""


def _label(key: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", key) if part) or key


def _slug(value: Any) -> str:
    """Dashed lowercase id, for names that are prose ("Customer Support" →
    "customer-support"): network keys and agent keys taken from a human role."""
    return _SLUG_RE.sub("-", str(value or "").strip().lower()).strip("-")


def _key(value: Any) -> str:
    """Collision-resistant registry id for tools and graph node identifiers."""
    return _registry_key(value)


def _registry_key(value: Any) -> str:
    """Match Go's collision-resistant ``AgenticNewSanitizeKey``."""
    source = str(value or "").strip()
    if not source:
        return ""
    if (
        len(source) <= 128
        and source[0] != "-"
        and source[-1] != "-"
        and re.fullmatch(r"[a-z0-9._-]+", source)
    ):
        return source

    raw = source.lower()
    out: list[str] = []
    last_dash = False
    for char in raw:
        if ("a" <= char <= "z") or ("0" <= char <= "9") or char in "._-":
            out.append(char)
            last_dash = False
        elif out and not last_dash:
            out.append("-")
            last_dash = True
    readable = "".join(out).strip("-") or "id"
    max_readable = 128 - len("v1") - 1 - len("--") - 32
    if len(readable) > max_readable:
        bounded = readable[:max_readable].rstrip("-._")
        readable = bounded or readable[:max_readable]

    payload = bytearray(b"deepintshield/registry-key/v1")
    encoded = source.encode("utf-8")
    payload.extend(len(encoded).to_bytes(8, "big"))
    payload.extend(encoded)
    digest = hashlib.sha256(payload).hexdigest()[:32]
    return f"v1-{readable}--{digest}"


# ── Topology assembly ─────────────────────────────────────────────────────


def _normalise_topology(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """De-duplicate nodes, add the implicit input/output endpoints referenced by
    edges, and drop edges pointing at nodes that don't exist."""
    by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id and node_id not in by_id:
            by_id[node_id] = node
    for endpoint, kind in (("input", "input"), ("output", "output")):
        referenced = any(e.get("from") == endpoint or e.get("to") == endpoint for e in edges)
        if referenced and endpoint not in by_id:
            by_id[endpoint] = {"id": endpoint, "label": endpoint.capitalize(), "kind": kind}
    seen: set[tuple[str, str]] = set()
    clean_edges: list[dict[str, Any]] = []
    for edge in edges:
        src, dst = edge.get("from"), edge.get("to")
        if src not in by_id or dst not in by_id or src == dst:
            continue
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        clean_edges.append(edge)
    return list(by_id.values()), clean_edges


def _agents_from(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every agent node becomes a registry agent. Its capabilities are the tools
    it can reach in one hop (what it is actually able to do); with no reachable
    tool we fall back to the meaningful token of its own name, which is what the
    Registry's "Avg Capabilities" tile counts."""
    tool_ids = {n["id"] for n in nodes if n.get("kind") == "tool"}
    downstream: dict[str, set[str]] = {}
    for edge in edges:
        src, dst = edge.get("from"), edge.get("to")
        if dst in tool_ids and isinstance(src, str):
            downstream.setdefault(src, set()).add(dst)
    agents: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("kind") != "agent":
            continue
        key = node["id"]
        caps = sorted(downstream.get(key, ()))
        if not caps:
            declared = node.get("_capabilities")
            if isinstance(declared, (list, tuple, set)):
                caps = sorted(
                    {
                        candidate
                        for candidate in (_key(value) for value in declared)
                        if candidate
                    }
                )
        if not caps:
            caps = [t for t in _SLUG_RE.split(key.lower()) if t and t not in _STOPWORDS][-1:]
        entry: dict[str, Any] = {
            "key": key,
            "name": node.get("label") or key,
            "capabilities": caps,
        }
        if node.get("model"):
            entry["model"] = node["model"]
        agents.append(entry)
    return agents


def _digest(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """Stable topology digest: sorted node keys + sorted edge keys. Order of
    discovery, labels and models are excluded, so re-running the same graph
    produces the same digest (idempotent upsert) while adding/removing a node or
    edge produces a new one (drift event)."""
    parts = sorted(
        f"n:{n.get('id', '')}:{n.get('kind', '')}:"
        f"{str(n.get('executable')).lower() if isinstance(n.get('executable'), bool) else 'unset'}"
        for n in nodes
    )
    parts += sorted(f"e:{e.get('from', '')}>{e.get('to', '')}" for e in edges)
    return "sha256:" + hashlib.sha256("\n".join(parts).encode("utf-8", "replace")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────


def _dedupe_key(engine: Any, payload: dict[str, Any]) -> str:
    """Hash the complete manifest inside one SDK-client scope.

    The old key was only ``network.digest`` (nodes + edges), which caused an
    identical graph from another tenant/client to disappear and delayed tool
    schema/action/server changes for five minutes.  The opaque per-engine scope
    prevents cross-client suppression without retaining the raw VK.
    """
    scope = str(_attr(engine, "registry_scope_id", "") or f"engine-{id(engine)}")
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_schema_json_default,
        )
    except Exception:
        canonical = repr(payload)
    return hashlib.sha256(f"{scope}\n{canonical}".encode("utf-8", "replace")).hexdigest()


def _wire_attested_payload(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Key raw-content equality attestations before they cross the process.

    Local discovery uses an ordinary SHA-256 change detector because it has no
    engine/credential dependency. At transport time, HMAC that digest with the
    workspace Virtual Key under a fixed domain. The gateway still receives a
    stable opaque change token, but an observer cannot perform offline guesses
    against a low-entropy redacted literal. This is confidentiality hardening,
    not provenance proof: the runtime reporter controls both source and token.
    """
    artifacts = [
        dict(item)
        for item in _as_seq(payload.get("code_artifacts"))
        if isinstance(item, dict)
    ]
    if not artifacts:
        return payload
    wire = dict(payload)
    wire["code_artifacts"] = artifacts
    virtual_key = str(_attr(engine, "virtual_key", "") or "").encode(
        "utf-8", "replace"
    )
    complete = bool(virtual_key)
    for artifact in artifacts:
        local_digest = str(
            artifact.get("self_attested_content_digest") or ""
        ).strip().lower()
        if not virtual_key or not _SHA256_DIGEST_RE.fullmatch(local_digest):
            complete = False
            artifact.pop("self_attested_content_digest", None)
            continue
        opaque = hmac.new(
            virtual_key,
            b"deepintshield-blueprint-content-v1\x00" + local_digest.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        artifact["self_attested_content_digest"] = "sha256:" + opaque
    coverage = [
        dict(item)
        for item in _as_seq(payload.get("code_artifact_coverage"))
        if isinstance(item, dict)
    ]
    if coverage:
        wire["code_artifact_coverage"] = coverage
    attestation = _self_attested_code_digest(artifacts, coverage)
    if complete and attestation:
        wire["self_attested_code_digest"] = attestation
    else:
        wire.pop("self_attested_code_digest", None)
        wire["code_artifacts_incomplete"] = True
    topology_digest = str((wire.get("network") or {}).get("digest") or "")
    wire["blueprint_digest"] = _blueprint_digest(
        topology_digest,
        artifacts,
        incomplete=wire.get("code_artifacts_incomplete") is True,
        coverage=coverage,
    )
    return wire


def discover(
    engine: Any,
    obj: Any = None,
    *,
    manifest: Optional[dict[str, Any]] = None,
    principal_email: Optional[str] = None,
    auto_provision: bool = True,
    sync: bool = False,
    name: str = "",
    framework: str = "",
    timeout: float = _TIMEOUT_SECONDS,
    _bootstrap_without_workload_proof: bool = False,
) -> dict[str, Any]:
    """Report ``obj``'s topology to ``/api/agentic-new/registry/discover``.

    Pass a compiled graph / uncompiled builder / crew / agent as ``obj``, or a
    pre-built payload as ``manifest`` (normalised + re-digested so it dedupes
    like any other topology). ``principal_email`` attaches the human the run is
    acting for (the server resolves it to a ``user:…`` subject);
    ``auto_provision`` lets the server write the ``agent → organization`` /
    ``tool → server`` tuples through the audited relationship store.

    ``framework`` overrides the auto-detected label. Detection reads an
    object's shape, so a bare list of callables is indistinguishable from a
    list of CrewAI tools and is reported as ``crewai``. Pass this when you know
    what the agent actually is, so the Registry does not file it under a
    framework the operator never used.

    Default is fire-and-forget on a **daemon** thread with a 3s timeout, so an
    explicit inventory report stays off the invocation path and never holds up
    interpreter exit; it returns ``{"dispatched": …, "digest": …}``
    immediately. Framework execution barriers use ``sync=True`` and require a
    durable server acknowledgement for each new or changed blueprint before
    application code runs.
    """
    if obj is None and not isinstance(manifest, dict):
        return {"dispatched": False, "error": "registry_discovery_empty"}
    # describe_network never raises, but keep the belt-and-braces guard: this is
    # called from inside a framework's compile()/kickoff() and must not surface
    # anything to the developer's code.
    try:
        source = manifest if isinstance(manifest, dict) else obj
        payload = describe_network(source, name=name, framework=framework)
    except Exception as exc:  # pragma: no cover - describe_network is fail-soft
        log.debug("registry discovery skipped (describe failed: %s)", exc)
        return {"dispatched": False, "error": "registry_discovery_invalid"}

    digest = str((payload.get("network") or {}).get("digest") or "")
    network_key = str((payload.get("network") or {}).get("key") or "")
    payload["auto_provision"] = bool(auto_provision)
    if principal_email:
        payload["principal"] = {"email": principal_email.strip().lower()}
    elif "principal" not in payload and not _bootstrap_without_workload_proof:
        # Carry the original email/username as well as the resolved subject.
        # VK ingest intentionally distrusts subject-only hints, so retaining the
        # source identifier is what makes automatic JIT binding work.
        try:
            from .identity import principal_manifest

            bound = principal_manifest(engine)
        except Exception:
            bound = {}
        if bound:
            payload["principal"] = bound

    dedupe_key = _dedupe_key(engine, payload)
    flight = None if sync else _ReportFlight()
    if not _should_report(dedupe_key, flight=flight):
        pending = _report_flight(dedupe_key)
        if pending is not None and not sync:
            # Do not let a second async reporter convert an unfinished mark
            # into a successful ``deduped`` result. Framework wrappers use the
            # error field to avoid permanently marking their object reported.
            return {
                "dispatched": False,
                "error": "registry_discovery_pending",
                "digest": digest,
                "network_key": network_key,
            }
        if pending is not None:
            # A digest marked by an async reporter is not yet a successful
            # report. Wait only for that request's bounded network budget; a
            # run must never race ahead merely because the dedupe mark exists.
            wait_seconds = max(0.0, float(timeout)) + 0.25
            if not pending.completed.wait(wait_seconds):
                return {
                    "dispatched": False,
                    "error": "registry_discovery_pending",
                    "digest": digest,
                    "network_key": network_key,
                }
            if not pending.succeeded:
                return {
                    "dispatched": False,
                    "error": "registry_discovery_unavailable",
                    "digest": digest,
                    "network_key": network_key,
                }
        # No pending flight means this exact payload completed successfully
        # inside the dedupe window. Seed the engine's per-tool capture set from
        # it so the shared gate does not issue a smaller duplicate report.
        _mark_registration_capture_ready(engine, payload)
        return {
            "dispatched": False,
            "deduped": True,
            "digest": digest,
            "network_key": network_key,
        }

    if sync:
        result = _post_discover(
            engine,
            payload,
            timeout,
            bootstrap_without_workload_proof=_bootstrap_without_workload_proof,
        )
        if result.get("error"):
            _log_enrolment_state(result)
            _unmark(dedupe_key)
        else:
            _mark_registration_capture_ready(engine, payload)
        return result

    # Never start a thread while the interpreter is tearing down: the post can't
    # complete and the attempt itself can raise.
    if _finalizing():
        _unmark(dedupe_key)
        _finish_report_flight(dedupe_key, flight, succeeded=False)
        return {"dispatched": False, "shutdown": True, "digest": digest, "network_key": network_key}
    if not _acquire_slot():
        # Too many posts already in flight. Un-mark so the *next* compile of this
        # topology tries again instead of being swallowed by the dedupe window.
        _unmark(dedupe_key)
        _finish_report_flight(dedupe_key, flight, succeeded=False)
        return {"dispatched": False, "throttled": True, "digest": digest, "network_key": network_key}
    try:
        threading.Thread(
            target=_post_and_release,
            args=(
                engine,
                payload,
                timeout,
                dedupe_key,
                _bootstrap_without_workload_proof,
                flight,
            ),
            name="deepintshield-discovery",
            daemon=True,
        ).start()
    except BaseException as exc:  # thread creation exhausted / shutting down
        _release_slot()
        _unmark(dedupe_key)
        _finish_report_flight(dedupe_key, flight, succeeded=False)
        log.debug("registry discovery thread not started: %s", exc)
        return {
            "dispatched": False,
            "error": "registry_discovery_unavailable",
            "digest": digest,
            "network_key": network_key,
        }
    return {"dispatched": True, "digest": digest, "network_key": network_key}


def reset_dedupe() -> None:
    """Forget every reported digest (tests / a worker that wants to force a
    re-report after an operator cleared the registry)."""
    with _seen_lock:
        _seen.clear()
        flights = list(_report_flights.values())
        _report_flights.clear()
    for flight in flights:
        flight.succeeded = False
        flight.completed.set()
    with _expanded_source_cache_lock:
        _expanded_source_cache.clear()


def _finalizing() -> bool:
    try:
        return bool(sys.is_finalizing())
    except Exception:  # pragma: no cover - pre-3.8 / exotic runtime
        return False


def _acquire_slot() -> bool:
    """Take one of the :data:`_INFLIGHT_MAX` background-post slots."""
    global _inflight
    with _seen_lock:
        if _inflight >= _INFLIGHT_MAX:
            return False
        _inflight += 1
        return True


def _release_slot() -> None:
    global _inflight
    with _seen_lock:
        if _inflight > 0:
            _inflight -= 1


def _unmark(dedupe_key: str) -> None:
    """Undo :func:`_should_report`'s mark when the post never left the process."""
    if not dedupe_key:
        return
    with _seen_lock:
        _seen.pop(dedupe_key, None)


# Remedies keyed by the enrolment lifecycle code the server returns. Each names
# the ONE thing the developer has to do; a code with no entry is left alone
# rather than guessed at.
_ENROLMENT_REMEDIES = {
    "agent_registration_pending": (
        "waiting for an administrator to approve its registration"
    ),
    "agent_registration_denied": (
        "its registration was denied; enrol under a different agent name"
    ),
    "agent_blueprint_review_pending": (
        "its code changed and the new blueprint is awaiting review"
    ),
    "agent_blueprint_review_denied": (
        "its code changed and the new blueprint was rejected"
    ),
    "blueprint_coverage_incomplete": (
        "discovery could not read the source of every declared tool; "
        "pass the tool callables rather than the agent instance"
    ),
    "agent_not_registered": (
        "no registration exists for this agent on this virtual key"
    ),
    "agent_quarantined": (
        "the agent is quarantined; review its blueprint scan to restore it"
    ),
    # The remedy is the whole point of this code: the name IS taken, by a row
    # this key does not own, and the fix is to pick a different one. A bare
    # 403 sent developers looking at the key instead.
    "agent_name_conflict": (
        "an agent of this name already exists on another key; choose a "
        "different agent_name (or DEEPINTSHIELD_AGENT_NAME)"
    ),
}

_enrolment_logged: set[str] = set()
_enrolment_log_lock = threading.Lock()


def _log_enrolment_state(result: Any) -> None:
    """Say plainly, once per process per state, that governed calls will fail.

    A developer's first signal that enrolment is pending, denied or quarantined
    used to be an unrelated-looking authorization failure on their first
    governed call, sometimes minutes later and with five very different causes
    behind one message. Discovery already learns the real state, so it is the
    cheapest honest place to say so - and because it runs off the invocation
    path, saying so costs the application nothing.

    Only enrolment LIFECYCLE codes are reported. Transport failures are
    ordinary, already logged at debug, and would otherwise turn a flaky network
    into a stream of alarming warnings.
    """
    if not isinstance(result, dict):
        return
    code = str(result.get("error") or "").strip()
    remedy = _ENROLMENT_REMEDIES.get(code)
    if remedy is None:
        return
    with _enrolment_log_lock:
        if code in _enrolment_logged:
            return
        _enrolment_logged.add(code)
    review_url = str(result.get("review_url") or "").strip()
    suffix = f" Review: {review_url}" if review_url else ""
    log.warning(
        "DeepintShield: this agent is not governed yet - %s (%s)."
        " Governed calls will be denied until this is resolved.%s",
        remedy,
        code,
        suffix,
    )


def _post_and_release(
    engine: Any,
    payload: dict[str, Any],
    timeout: float,
    dedupe_key: str,
    bootstrap_without_workload_proof: bool = False,
    flight: _ReportFlight | None = None,
) -> None:
    """Background-thread body. ``BaseException`` (not just ``Exception``) because
    a daemon thread can be unwound mid-flight at interpreter exit and an escaping
    exception there prints a scary traceback from an optional background call."""
    succeeded = False
    try:
        result = _post_discover(
            engine,
            payload,
            timeout,
            bootstrap_without_workload_proof=bootstrap_without_workload_proof,
        )
        if result.get("error"):
            _log_enrolment_state(result)
            _unmark(dedupe_key)
        else:
            succeeded = True
            _mark_registration_capture_ready(engine, payload)
    except BaseException:  # pragma: no cover - _post_discover already swallows
        _unmark(dedupe_key)
        pass
    finally:
        _finish_report_flight(dedupe_key, flight, succeeded=succeeded)
        _release_slot()


def _mark_registration_capture_ready(
    engine: Any,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    captured = _manifest_tool_keys(payload or {})
    attestations = _manifest_tool_attestations(payload or {})
    try:
        tools = getattr(engine, "_registration_captured_tools", None)
        if isinstance(tools, set):
            tools.update(captured)
            setattr(engine, "_registration_capture_ready", bool(tools))
        else:
            # Compatibility for custom engines that predate per-tool capture.
            setattr(engine, "_registration_capture_ready", bool(captured))
        known = getattr(engine, "_registration_captured_tool_attestations", None)
        if isinstance(known, dict):
            known.update(attestations)
        elif attestations:
            setattr(engine, "_registration_captured_tool_attestations", attestations)
    except Exception:
        pass


def _manifest_tool_keys(payload: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for raw in payload.get("tools", ()) or ():
        if not isinstance(raw, dict):
            continue
        key = _tool_registry_key(raw.get("key") or raw.get("name"))
        if key:
            keys.add(key)
    for raw in payload.get("nodes", ()) or ():
        if not isinstance(raw, dict) or raw.get("kind") != "tool":
            continue
        value = raw.get("ref") or raw.get("id")
        key = _tool_registry_key(value)
        if key:
            keys.add(key)
    return keys


def _manifest_tool_attestations(payload: dict[str, Any]) -> dict[str, str]:
    """Local raw-content attestations from an acknowledged discovery payload."""
    attestations: dict[str, str] = {}
    for raw in payload.get("code_artifacts", ()) or ():
        if not isinstance(raw, dict) or raw.get("subject_kind") != "tool":
            continue
        key = _tool_registry_key(raw.get("subject_key"))
        digest = str(raw.get("self_attested_content_digest") or "").strip().lower()
        if key and _SHA256_DIGEST_RE.fullmatch(digest):
            attestations[key] = digest
    for raw in payload.get("code_artifact_coverage", ()) or ():
        if not isinstance(raw, dict) or raw.get("subject_kind") != "tool":
            continue
        key = _tool_registry_key(raw.get("subject_key"))
        status = str(raw.get("status") or "").strip().lower()
        if key and status and status != "captured":
            attestations.setdefault(key, f"coverage:{status}")
    return attestations


def _tool_registry_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("tool:"):
        text = text.split(":", 1)[1]
    return _key(text)


def _callable_attestation(value: Any) -> str:
    """Exact raw-source digest used for the local zero-network fast path."""
    source, _ = _expanded_callable_source(value)
    if source:
        return _source_digest(source)
    token = _callable_change_token(value)
    return f"missing:{token}" if token else "coverage:missing"


def _capture_has_tool(
    engine: Any,
    tool_key: str,
    callable_attestation: str = "",
) -> bool:
    try:
        captured = getattr(engine, "_registration_captured_tools", None)
        captured_tool = (
            tool_key in captured
            if isinstance(captured, set)
            else bool(getattr(engine, "_registration_capture_ready", False))
        )
        if not captured_tool:
            return False
        if not callable_attestation:
            return True
        known = getattr(engine, "_registration_captured_tool_attestations", None)
        return isinstance(known, dict) and known.get(tool_key) == callable_attestation
    except Exception:
        return False


def _remember_tool_attestation(engine: Any, tool_key: str, attestation: str) -> None:
    if not tool_key or not attestation:
        return
    try:
        known = getattr(engine, "_registration_captured_tool_attestations", None)
        if not isinstance(known, dict):
            known = {}
            setattr(engine, "_registration_captured_tool_attestations", known)
        known[tool_key] = attestation
    except Exception:
        pass


def _capture_retry_at(engine: Any, tool_key: str) -> float:
    try:
        retries = getattr(engine, "_registration_capture_retry_by_tool", None)
        if isinstance(retries, dict):
            return float(retries.get(tool_key, 0.0) or 0.0)
        return float(getattr(engine, "_registration_capture_retry_at", 0.0) or 0.0)
    except Exception:
        return 0.0


def _defer_registration_capture(engine: Any, tool_key: str) -> None:
    retry_at = time.monotonic() + _REGISTRATION_CAPTURE_RETRY_SECONDS
    try:
        retries = getattr(engine, "_registration_capture_retry_by_tool", None)
        if isinstance(retries, dict):
            retries[tool_key] = retry_at
        # Retain the scalar as a compatibility/diagnostic view of the latest
        # failure without letting one tool suppress another on current engines.
        setattr(engine, "_registration_capture_retry_at", retry_at)
    except Exception:
        pass


def _set_registration_capture_error(engine: Any, code: object) -> None:
    normalized = normalize_agentic_error_code(code, "")
    # Preserve only the stable first-run admission codes here. Other discovery
    # failures keep the established blueprint_scan_unavailable fallback.
    if normalized not in {
        "agent_registration_quota_exceeded",
        "agent_name_required",
        "blueprint_coverage_incomplete",
    }:
        normalized = ""
    try:
        setattr(engine, "_registration_capture_error_code", normalized)
    except Exception:
        pass


def _minimal_registration_manifest(
    engine: Any,
    *,
    agent_key: str,
    tool_key: str,
    tool_label: str,
    tool_callable: Any = None,
) -> dict[str, Any]:
    try:
        prior = getattr(engine, "_registration_minimal_tools", set())
        minimal_tools = set(prior) if isinstance(prior, set) else set()
    except Exception:
        minimal_tools = set()
    minimal_tools.add(tool_key)
    try:
        prior_artifacts = getattr(engine, "_registration_minimal_code_artifacts", {})
        code_by_tool = dict(prior_artifacts) if isinstance(prior_artifacts, dict) else {}
    except Exception:
        code_by_tool = {}
    code_by_tool[tool_key] = _source_artifact(
        tool_callable,
        subject_kind="tool",
        subject_key=tool_key,
    )
    try:
        setattr(engine, "_registration_minimal_code_artifacts", code_by_tool)
    except Exception:
        pass
    tools = []
    nodes = [
        {
            "id": agent_key,
            "label": agent_key,
            "kind": "agent",
            "ref": f"agent:{agent_key}",
        }
    ]
    edges = []
    for key in sorted(minimal_tools):
        label = tool_label if key == tool_key else key
        tool: dict[str, Any] = {
            "key": key,
            "name": label,
            "actions": [
                {
                    "key": "invoke",
                    "name": "Invoke",
                    "action_class": infer_action_class(key),
                }
            ],
            "_code_artifact": code_by_tool.get(key)
            or {
                "subject_kind": "tool",
                "subject_key": key,
                "_incomplete": True,
            },
        }
        tools.append(tool)
        nodes.append(
            {
                "id": key,
                "label": label,
                "kind": "tool",
                "ref": f"tool:{key}",
            }
        )
        edges.append({"from": agent_key, "to": key})
    artifacts, artifacts_incomplete, coverage = _collect_code_artifacts(nodes, tools)
    manifest = {
        "framework": "agentic-sdk",
        "network": {
            "key": f"{agent_key}-first-run",
            "name": agent_key,
        },
        "agents": [{"key": agent_key, "name": agent_key}],
        "tools": tools,
        "nodes": nodes,
        "edges": edges,
        "code_artifacts": artifacts,
    }
    if coverage:
        manifest["code_artifact_coverage"] = coverage
    if artifacts_incomplete:
        manifest["code_artifacts_incomplete"] = True
    attestation = _self_attested_code_digest(artifacts, coverage)
    if attestation:
        manifest["self_attested_code_digest"] = attestation
    return manifest


def ensure_registration_capture(
    engine: Any,
    tool_name: str,
    tool_callable: Any = None,
) -> bool:
    """Synchronously submit one minimal first-run manifest when needed.

    Full native-framework reporters mark the same engine flag. This fallback is
    reached only when a supported/compatibility version exposes its final tool
    boundary without a discoverable outer run object. It is intentionally
    single-flight and bounded by the normal registry timeout.
    """
    tool_label = str(tool_name or "").strip()
    tool_key = _tool_registry_key(tool_label) or "invoke"
    callable_attestation = _callable_attestation(tool_callable)
    if _capture_has_tool(engine, tool_key, callable_attestation):
        return True
    if time.monotonic() < _capture_retry_at(engine, tool_key):
        return False
    lock = getattr(engine, "_registration_capture_lock", None)
    if lock is None:
        return False
    with lock:
        # Recompute after acquiring the single-flight lock: another thread may
        # have rebound the wrapper while this caller waited.
        callable_attestation = _callable_attestation(tool_callable)
        if _capture_has_tool(engine, tool_key, callable_attestation):
            return True
        if time.monotonic() < _capture_retry_at(engine, tool_key):
            return False
        selector = str(getattr(engine, "_agent_subject_selector", "") or "")
        if not selector:
            # No explicit agent_name was given. The gateway's own
            # credential-info carries the subject this virtual key is bound to,
            # and ``AgenticEngine.agent_subject`` documents that value as
            # authoritative - it is already what /decide is told the principal
            # is. Registering under it keeps one identity across the decision
            # and the registry instead of refusing a VK that the server can
            # name perfectly well.
            #
            # This is NOT the invented shared name warned about below: it is
            # server-issued and VK-bound, so it cannot collide across keys.
            try:
                selector = str(getattr(engine, "agent_subject", "") or "").strip()
            except Exception:
                # Credential discovery is allowed to fail here; the selector
                # check below then reports the missing name as before.
                selector = ""
        agent_key = _key(selector.removeprefix("agent:"))
        if not selector.startswith("agent:") or not agent_key:
            # Proof-less registration is deliberately limited to an explicit
            # selector. Inventing a shared ``sdk-agent`` would create ambiguous
            # registry rows and cannot pass the server's bootstrap middleware.
            #
            # Say WHICH thing is missing. This used to surface as
            # blueprint_scan_unavailable, which sent developers to look at the
            # scanner when the real answer is that the client was never given
            # an agent name to register under.
            _set_registration_capture_error(engine, "agent_name_required")
            _defer_registration_capture(engine, tool_key)
            return False
        manifest = _minimal_registration_manifest(
            engine,
            agent_key=agent_key,
            tool_key=tool_key,
            tool_label=tool_label or tool_key,
            tool_callable=tool_callable,
        )
        result = discover(
            engine,
            manifest=manifest,
            sync=True,
            # An unknown agent cannot obtain workload credentials until this
            # registration exists. Send the authenticated VK + explicit agent
            # selector first so capture happens before credential discovery.
            _bootstrap_without_workload_proof=True,
        )
        if (
            result.get("error")
            and result.get("status_code") in (401, 403)
            and result.get("error")
            not in {
                "agent_not_registered",
                "agent_registration_pending",
                "agent_registration_denied",
            }
        ):
            # Enabled identity-backed agents reject the proof-less bootstrap.
            # Retry this same manifest once through the normal workload-proof
            # path; unknown/pending/denied registrations never take this path.
            result = discover(engine, manifest=manifest, sync=True)
        succeeded = not bool(result.get("error"))
        if succeeded:
            _set_registration_capture_error(engine, "")
            _remember_tool_attestation(
                engine,
                tool_key,
                callable_attestation,
            )
            try:
                minimal = getattr(engine, "_registration_minimal_tools", None)
                if isinstance(minimal, set):
                    minimal.add(tool_key)
            except Exception:
                pass
        else:
            _set_registration_capture_error(engine, result.get("error"))
            _defer_registration_capture(engine, tool_key)
        return succeeded


def _report_flight(dedupe_key: str) -> _ReportFlight | None:
    with _seen_lock:
        return _report_flights.get(dedupe_key)


def _finish_report_flight(
    dedupe_key: str,
    flight: _ReportFlight | None,
    *,
    succeeded: bool,
) -> None:
    if flight is None:
        return
    flight.succeeded = bool(succeeded)
    flight.completed.set()
    with _seen_lock:
        if _report_flights.get(dedupe_key) is flight:
            _report_flights.pop(dedupe_key, None)


def _should_report(
    dedupe_key: str,
    *,
    flight: _ReportFlight | None = None,
) -> bool:
    """True once per scoped manifest key per :data:`REPOST_INTERVAL_SECONDS`. Marks
    before dispatch so concurrent compiles of the same graph post once; a failed
    post is unmarked by the dispatcher so a recovered gateway can be retried."""
    if not dedupe_key:
        return True
    now = time.monotonic()
    with _seen_lock:
        last = _seen.get(dedupe_key)
        if last is not None and now - last < REPOST_INTERVAL_SECONDS:
            return False
        if len(_seen) >= _SEEN_MAXSIZE:
            for stale, ts in list(_seen.items()):
                if (
                    now - ts >= REPOST_INTERVAL_SECONDS
                    and stale not in _report_flights
                ):
                    _seen.pop(stale, None)
            if len(_seen) >= _SEEN_MAXSIZE:
                evictable = next(
                    (key for key in _seen if key not in _report_flights),
                    None,
                )
                if evictable is not None:
                    _seen.pop(evictable, None)
        _seen[dedupe_key] = now
        if flight is not None:
            _report_flights[dedupe_key] = flight
        return True


def _post_discover(
    engine: Any,
    payload: dict[str, Any],
    timeout: float,
    *,
    bootstrap_without_workload_proof: bool = False,
) -> dict[str, Any]:
    """One POST to /registry/discover. It returns a stable error mapping rather
    than raising; callers decide whether the report is optional inventory or a
    required pre-execution security barrier.

    Existing agents send workload proof as usual. A never-before-seen agent
    cannot obtain that proof yet because credential discovery is intentionally
    gated on an approved Registry profile. In that one bootstrap situation,
    fall back to the authenticated virtual key plus the canonical agent
    selector; the server permits those proof-less credentials only on this
    discovery route and creates a disabled, pending-registration profile.
    """
    try:
        if bootstrap_without_workload_proof:
            headers = engine._headers(include_agent_token=False)
        else:
            try:
                headers = engine._headers(include_agent_token=True)
            except GovernanceConfigurationError as exc:
                if exc.code not in {
                    "agent_not_registered",
                    "agent_registration_pending",
                    "agent_registration_denied",
                }:
                    raise
                headers = engine._headers(include_agent_token=False)
        resp = engine._http.post(
            f"{engine.gateway_url}{_DISCOVER_PATH}",
            json=_wire_attested_payload(engine, payload),
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code >= 300:
            log.debug("registry discovery http %s", resp.status_code)
            try:
                body = resp.json()
            except Exception:
                body = {}
            return {
                "dispatched": True,
                "error": _discovery_error_code(
                    body,
                    "registry_discovery_rejected",
                ),
                "status_code": resp.status_code,
            }
        body = resp.json()
    except DeepIntShieldError as exc:
        log.debug("registry discovery unavailable: %s", exc.code)
        return {"dispatched": True, "error": exc.code}
    except Exception as exc:
        log.debug("registry discovery unavailable: %s", type(exc).__name__)
        return {"dispatched": True, "error": "registry_discovery_unavailable"}
    if not isinstance(body, dict):
        return {"dispatched": True}
    if body.get("error"):
        return {
            **body,
            "error": _discovery_error_code(
                body,
                "registry_discovery_rejected",
            ),
        }
    # A durable blueprint row is not an execution acknowledgement until every
    # configured scanner has finished.  In particular, do not seed the local
    # unchanged-code fast path from a transient model pending/failed response:
    # the next invocation must repost the source so the zero-retention server
    # can retry without retaining application code between attempts.
    registration_state = str(body.get("registration_state") or "").strip().lower()
    if registration_state:
        # Discovery may have atomically moved an unknown workload into pending
        # registration while another lifecycle worker was reading credentials.
        # Do not let that earlier short negative cache hide the newer precise
        # lifecycle state from the gated caller.
        clear_failure = getattr(engine, "_clear_credential_failure_cache", None)
        if callable(clear_failure):
            clear_failure()
    lifecycle_code = ""
    if registration_state == "pending":
        lifecycle_code = "agent_registration_pending"
    elif registration_state == "denied":
        lifecycle_code = "agent_registration_denied"

    scan_status = str(body.get("blueprint_scan_status") or "").strip().lower()
    if scan_status in {"model_pending", "model_running"}:
        return {
            **body,
            "error": lifecycle_code or "blueprint_model_scan_pending",
        }
    if scan_status == "model_failed":
        return {
            **body,
            "error": lifecycle_code or "blueprint_model_scan_failed",
        }
    # First-run registration has its own lifecycle and intentionally returns a
    # 202 capture acknowledgement so the dashboard can collect the remaining
    # fields.  For an already-registered agent, however, changed code creates a
    # separate review. Surface that precise gate before credential resolution
    # can collapse the quarantined profile into a generic registration error.
    review_status = str(body.get("blueprint_review_status") or "").strip().lower()
    if not registration_state and review_status == "pending":
        return {
            **body,
            "error": "agent_blueprint_review_pending",
        }
    if not registration_state and review_status == "denied":
        return {
            **body,
            "error": "agent_blueprint_review_denied",
        }
    return body


def _discovery_error_code(body: object, fallback: str) -> str:
    if not isinstance(body, dict):
        return fallback
    error = body.get("error")
    candidate = error.get("code") if isinstance(error, dict) else error
    return normalize_agentic_error_code(candidate, fallback)


__all__ = ["describe_network", "discover", "reset_dedupe", "REPOST_INTERVAL_SECONDS"]
