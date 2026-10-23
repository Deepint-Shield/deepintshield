"""Explicit server-managed handoff for standalone Python LangGraph scripts.

The gateway must advertise this mode. No environment, directory tree, private
runtime key, or arbitrary upload URL is collected. A submission exits before
the first graph execution; its eventual result belongs to the server job.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib.machinery
import importlib.metadata
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
from typing import Any
import uuid

from ..errors import get_error_definition
from .errors import GovernanceConfigurationError, normalize_agentic_error_code

log = logging.getLogger(__name__)
_PREFIX = "/api/agentic-new/managed-execution"
_MAX_BYTES = 2 * 1024 * 1024
_MAX_FILES = 64
_MAX_REQUIREMENTS = 64
_MAX_RESULT_BYTES = 64 * 1024
# A completed submission includes the bounded result plus job metadata.
_MAX_SUBMISSION_RESPONSE_BYTES = 128 * 1024
_LOCK = threading.RLock()
_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.!+_-]*$")
# One distribution can be written Foo_Bar.baz or foo-bar-baz: a run of these counts as a single "-".
_NAME_SEPARATORS = re.compile(r"[-_.]+")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_OUTPUT_SECRETS = re.compile(
    r"(?:bearer\s+[A-Za-z0-9._~-]+|sk-[A-Za-z0-9_-]{8,}|"
    r"virtual_key:[A-Za-z0-9_-]+|"
    r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|$))",
    re.IGNORECASE,
)
_TERMINAL_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")
# Hosts that call a program's code for as long as the process lives: servers,
# web frameworks, task queues, app runners and the notebook kernel. Named by
# top-level package and looked for on the call stack, not among the imports.
_SERVING_MODULES = frozenset({
    "uvicorn", "gunicorn", "hypercorn", "daphne", "waitress", "granian",
    "fastapi", "starlette", "flask", "django", "tornado", "sanic", "quart",
    "litestar", "falcon", "bottle", "cherrypy", "streamlit", "gradio",
    "chainlit", "celery", "langserve", "langgraph_api", "ipykernel",
})
# Packages of which only this part serves; the rest is a client that scripts
# use. A plain prefix, because aiohttp serves from aiohttp.web_app and its like.
_SERVING_PREFIXES = ("aiohttp.web", "fastmcp.server", "mcp.server")
_WORKER_FILES: dict[str, str] | None = None
_WORKER_APPROVAL_DEADLINE: float | None = None
_JOB_STATUS_ERRORS = {
    "queued": "managed_execution_pending",
    "preparing": "managed_execution_pending",
    "running": "managed_execution_pending",
    "waiting_identity": "managed_execution_identity_pending",
    "waiting_runtime": "managed_execution_runtime_pending",
    "waiting_approval": "agent_registration_pending",
    "failed": "managed_execution_failed",
    "cancelled": "managed_execution_cancelled",
    "expired": "managed_execution_expired",
    "reconcile_required": "managed_execution_reconcile_required",
}


def _identity_cannot_use_server_run(job: dict[str, Any]) -> bool:
    """Whether the server ended this run before the program ran because of its identity.

    A server run authenticates with gateway runtime exchange. The gateway ends
    the run as failed, in preparation, with ``managed_gateway_runtime_required``
    once the agent's identity is settled as one that cannot use it (approved as
    VK-trusted, or declared under a provider without gateway exchange). The
    program never started on the server, so running it locally repeats
    nothing. A run that reached its program is never read this way.
    """
    return (
        job.get("status") == "failed"
        and job.get("stage") == "prepare"
        and normalize_agentic_error_code(job.get("error_code"), "") == "managed_gateway_runtime_required"
    )


def _submission_error_code(job: dict[str, Any]) -> str | None:
    """Select trusted status guidance without reflecting server prose or IDs."""
    status = job.get("status")
    if status == "succeeded":
        return None
    if not isinstance(status, str) or status not in _JOB_STATUS_ERRORS:
        return "managed_execution_response_invalid"
    code = normalize_agentic_error_code(job.get("error_code"), "")
    if status == "failed" and code and get_error_definition(code) is not None:
        return code
    if status == "waiting_runtime" and code in {
        "managed_dependency_unavailable", "managed_python_version_unavailable",
    }:
        return code
    if status == "waiting_approval":
        if code in {"agent_access_setup_pending", "agent_access_plan_pending"}:
            return "managed_execution_access_pending"
        if code in {"agent_blueprint_review_pending", "agent_tool_identity_review_required", "managed_blueprint_changed"}:
            return code
    return _JOB_STATUS_ERRORS[status]


def augment_manifest(engine: Any, payload: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    """Bind all executable package source to an existing agent's scan evidence.

    The server independently constructs the same artifact from the uploaded
    files. Normal wire attestation subsequently HMACs the raw content digest.
    """
    agents = sorted(
        item["key"] for item in payload.get("agents", [])
        if isinstance(item, dict) and isinstance(item.get("key"), str) and item["key"]
    )
    if not agents:
        raise _error("managed_execution_bundle_unsupported")
    subject = agents[0]
    raw = "".join(f"# file: {path}\n{files[path]}\n" for path in sorted(files))
    if len(raw.encode("utf-8")) > 65536:
        raise _error("managed_execution_bundle_too_large")
    virtual_key = str(engine.virtual_key)
    source = raw.replace(virtual_key, "[REDACTED]") if virtual_key else raw
    artifact = {
        "subject_kind": "agent", "subject_key": subject, "language": "python",
        "source": source,
        "digest": "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "self_attested_content_digest": "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
    if source != raw:
        artifact["redacted"] = True
    artifacts = [
        item for item in payload.get("code_artifacts", [])
        if not (item.get("subject_kind") == "agent" and item.get("subject_key") == subject)
    ] + [artifact]
    if sum(len(str(item.get("source", "")).encode("utf-8")) for item in artifacts) > 512 * 1024:
        raise _error("managed_execution_bundle_too_large")
    coverage = [
        item for item in payload.get("code_artifact_coverage", [])
        if not (item.get("subject_kind") == "agent" and item.get("subject_key") == subject)
    ] + [{"subject_kind": "agent", "subject_key": subject, "status": "captured"}]
    return {
        **payload,
        "code_artifacts": sorted(artifacts, key=lambda item: (item["subject_kind"], item["subject_key"], item["digest"])),
        "code_artifact_coverage": sorted(coverage, key=lambda item: (item["subject_kind"], item["subject_key"])),
    }


def augment_worker_manifest(engine: Any, payload: dict[str, Any]) -> dict[str, Any]:
    # The env flag only prevents resubmission. Source attestation additionally
    # requires launcher-installed bytes from the isolated read-only bundle.
    return augment_manifest(engine, payload, _WORKER_FILES) if _WORKER_FILES is not None else payload


def program_bundle_manifest(engine: Any, described: dict[str, Any]) -> dict[str, Any] | None:
    """The discovery manifest a submission of this program reports, or ``None``.

    Only when the gateway's last capabilities answer asked for it
    (``report_program_bundle``): this agent's newest server run ended because
    its identity cannot use one, and its approval is pinned to the program it
    submitted. ``described`` is the program described once, as a submission
    describes it; the ``__main__`` script is packaged and its files bound by
    :func:`augment_manifest`, as a submission binds them. ``None`` without the
    flag, or when the program cannot be packaged or bound: the ordinary
    manifest is then reported, exactly as before.
    """
    if _WORKER_FILES is not None or getattr(engine, "_managed_execution_report_bundle", False) is not True:
        return None
    try:
        script = getattr(sys.modules.get("__main__"), "__file__", None)
        if not isinstance(script, str) or not script:
            return None
        # The "not offered" answer names no limits: a submission's defaults.
        files = package_script(Path(script), {})["files"]
        return augment_manifest(engine, copy.deepcopy(described), files)
    except Exception as error:  # discovery must never break the program
        log.debug("program bundle not reported (%s)", getattr(error, "code", type(error).__name__))
        return None


def _error(code: str) -> GovernanceConfigurationError:
    return GovernanceConfigurationError(framework="managed-execution", code=code)


def _response(response: Any, *, max_bytes: int = 65536) -> dict[str, Any]:
    if len(response.content) > max_bytes:
        raise _error("managed_execution_response_invalid")
    try:
        body = response.json()
    except (ValueError, TypeError):
        raise _error("managed_execution_response_invalid") from None
    if not isinstance(body, dict):
        raise _error("managed_execution_response_invalid")
    if response.status_code >= 400:
        raw = body.get("error") or body.get("code")
        if isinstance(raw, dict):
            raw = raw.get("code")
        raise _error(normalize_agentic_error_code(raw, "managed_execution_unavailable"))
    if not 200 <= response.status_code < 300:
        raise _error("managed_execution_response_invalid")
    return body


def _completed_output(engine: Any, job: dict[str, Any]) -> str:
    """Return captured program stdout, never job metadata or executable data."""
    output = job.get("result")
    if output is None:
        raise _error("managed_execution_output_unavailable")
    if not isinstance(output, str):
        raise _error("managed_execution_response_invalid")
    try:
        size = len(output.encode("utf-8"))
    except UnicodeError:
        raise _error("managed_execution_response_invalid") from None
    if size > _MAX_RESULT_BYTES:
        raise _error("managed_execution_response_invalid")
    # The gateway also sanitizes stored stdout. Repeat the boundary checks for
    # older gateways and strip terminal commands before writing returned text.
    credential = getattr(engine, "virtual_key", None)
    if isinstance(credential, str) and credential:
        output = output.replace(credential, "[REDACTED]")
    output = _OUTPUT_SECRETS.sub("[REDACTED]", output)
    output = _TERMINAL_ESCAPE.sub("", output)
    return "".join(char for char in output if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159))


def capabilities(engine: Any, *, refresh_disabled: bool = False) -> dict[str, Any]:
    """Cache enabled capability; recheck an optional compile-time opt-out at execution.

    A gateway restart or workspace configuration change must not leave a client
    permanently using local enrollment because its earlier compile probe saw
    managed execution disabled. No workload proof or redirects are involved.

    The recheck ends with the first execution that is told "not offered": it
    runs locally, and from then on so does the program.
    """
    if _WORKER_FILES is not None or os.environ.get("DEEPINTSHIELD_MANAGED_EXECUTION_WORKER") == "1":
        return {}
    # Custom framework engines without gateway transport cannot advertise a
    # managed worker; preserve their existing local integration contract.
    if not getattr(engine, "gateway_url", ""):
        return {}
    with _LOCK:
        cached = getattr(engine, "_managed_execution_capabilities", None)
        if isinstance(cached, dict) and (cached or not refresh_disabled):
            return cached
        if getattr(engine, "_managed_execution_local", False) is True:
            # Decided while this execution waited for the lock: it runs
            # locally with the one that asked, and is not told otherwise.
            return {}
        try:
            response = engine._http.get(
                f"{engine.gateway_url.rstrip('/')}{_PREFIX}/capabilities",
                headers=engine._headers(include_agent_token=False),
                timeout=5.0,
                follow_redirects=False,
            )
        except GovernanceConfigurationError:
            raise
        except Exception:
            raise _error("managed_execution_unavailable") from None
        report_bundle = False
        if response.status_code in (404, 405, 410, 501):
            body: dict[str, Any] = {}
        else:
            body = _response(response)
            if body.get("enabled") is not True:
                # Not offered because this agent's server run ended for its
                # identity: report the program bundle its approval is pinned to.
                report_bundle = body.get("report_program_bundle") is True
                body = {}
        engine._managed_execution_report_bundle = report_bundle
        engine._managed_execution_capabilities = body
        if refresh_disabled and not body:
            # This execution now runs locally. Handed over later, the script
            # would start again at its first line on the server and repeat it.
            # Set with the answer, under the lock, for executions waiting on it.
            engine._managed_execution_local = True
        return body


def _limit(settings: dict[str, Any], field: str, maximum: int) -> int:
    value = settings.get(field, maximum)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _error("managed_execution_response_invalid")
    return min(value, maximum)


def _read_python(path: Path, root: Path, remaining: int) -> str:
    try:
        relative = path.relative_to(root)
        if any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError()
        # Reject symlinks in every component, including the selected entrypoint.
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ValueError()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError()
            if info.st_size > remaining:
                raise _error("managed_execution_bundle_too_large")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(remaining + 1)
            if len(raw) > remaining:
                raise _error("managed_execution_bundle_too_large")
            return raw.decode("utf-8")
        finally:
            os.close(descriptor)
    except GovernanceConfigurationError:
        raise
    except (OSError, ValueError, UnicodeError):
        raise _error("managed_execution_bundle_unsupported") from None


def _packages_distributions() -> dict[str, list[str]]:
    """Map imports to installed distributions without importing their code.

    Python 3.10 only reads top_level.txt, which modern wheels need not include.
    Infer missing declarations from installed file records as newer Python
    versions do, including every distribution sharing a namespace package.
    """
    declared = importlib.metadata.packages_distributions()
    if sys.version_info >= (3, 11):
        return {module: list(names) for module, names in declared.items()}
    result = {
        module: set(names)
        for module, names in declared.items()
    }
    suffixes = tuple(importlib.machinery.all_suffixes())
    for distribution in importlib.metadata.distributions():
        if distribution.read_text("top_level.txt"):
            continue
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            continue
        for file in distribution.files or ():
            parts = file.parts
            if not parts or any(part in {"", ".", ".."} for part in parts):
                continue
            if not file.name.endswith(suffixes):
                continue
            module = parts[0] if len(parts) > 1 else parts[0].split(".", 1)[0]
            if module.isidentifier():
                result.setdefault(module, set()).add(name)
    return {module: sorted(names) for module, names in result.items()}


def package_script(entrypoint: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Package only the entry script and explicitly imported local Python files.

    Static literal dynamic imports are supported; computed imports, filesystem
    loaders, and code evaluation require an operator-built artifact instead.
    Installed distributions are recorded as pins, never installed by a worker.
    """
    entrypoint = Path(os.path.abspath(entrypoint))
    if entrypoint.suffix != ".py":
        raise _error("managed_execution_bundle_unsupported")
    root = entrypoint.parent
    byte_limit = _limit(settings, "max_bundle_bytes", _MAX_BYTES)
    file_limit = _limit(settings, "max_files", _MAX_FILES)
    files: dict[str, str] = {}
    queued: set[Path] = {entrypoint}
    external: set[str] = set()
    namespaces: set[str] = set()
    distributions = _packages_distributions()
    # Editable/source SDK checkouts may have no matching installed metadata.
    # Pin the running SDK's version as submission evidence only; the worker
    # runs its own installed SDK release and never needs to match it.
    distributions.setdefault("deepintshield", ["deepintshield"])
    used = 0

    def resolve(module: str, *, optional: bool = False) -> None:
        parts = module.split(".")
        if not module or any(not part.isidentifier() for part in parts):
            raise _error("managed_execution_bundle_unsupported")
        base = root.joinpath(*parts)
        target = base.with_suffix(".py")
        package = base / "__init__.py"
        if target.exists() or target.is_symlink():
            queued.add(target)
        elif package.exists() or package.is_symlink():
            queued.add(package)
        elif base.is_dir():
            if base.is_symlink():
                raise _error("managed_execution_bundle_unsupported")
            # A project directory without __init__.py must not shadow a real
            # installed package (e.g. test2.py beside the deepintshield repo).
            if parts[0] in distributions or parts[0] in sys.stdlib_module_names:
                external.add(parts[0])
                return
            namespaces.add(base.relative_to(root).as_posix())
        elif optional:
            return
        elif not (root / parts[0]).exists() and not (root / (parts[0] + ".py")).exists():
            external.add(parts[0])
            return
        else:
            raise _error("managed_execution_dependency_missing")
        for index in range(1, len(parts)):
            initializer = root.joinpath(*parts[:index], "__init__.py")
            if initializer.exists() or initializer.is_symlink():
                queued.add(initializer)

    while queued:
        path = min(queued, key=lambda item: str(item))
        queued.remove(path)
        relative = path.relative_to(root).as_posix()
        if relative in files:
            continue
        if len(files) >= file_limit:
            raise _error("managed_execution_bundle_too_large")
        source = _read_python(path, root, byte_limit - used)
        used += len(source.encode("utf-8"))
        try:
            tree = ast.parse(source, filename=relative)
        except (SyntaxError, ValueError, RecursionError):
            raise _error("managed_execution_bundle_unsupported") from None
        files[relative] = source
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    resolve(item.name)
                    aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parents = list(path.relative_to(root).parts[:-1])
                    if node.level > len(parents):
                        raise _error("managed_execution_bundle_unsupported")
                    module = ".".join(parents[:len(parents) - node.level + 1] + ([module] if module else []))
                resolve(module)
                for item in node.names:
                    if item.name != "*":
                        resolve(f"{module}.{item.name}", optional=True)
                        aliases[item.asname or item.name] = f"{module}.{item.name}"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            first, separator, rest = name.partition(".")
            name = aliases.get(first, first) + (separator + rest if separator else "")
            if name in {"__import__", "importlib.import_module"}:
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise _error("managed_execution_bundle_unsupported")
                resolve(node.args[0].value)
            elif name in {"exec", "eval", "runpy.run_path", "runpy.run_module", "importlib.util.spec_from_file_location", "importlib.machinery.SourceFileLoader", "importlib.machinery.SourcelessFileLoader"}:
                raise _error("managed_execution_bundle_unsupported")

    if any(not any(name.startswith(namespace + "/") for name in files) for namespace in namespaces):
        raise _error("managed_execution_dependency_missing")
    pins: set[str] = set()
    for module in sorted(external):
        if module in sys.stdlib_module_names:
            continue
        names = distributions.get(module, [])
        if not names:
            raise _error("managed_execution_dependency_missing")
        for name in names:
            try:
                if name.lower().replace("_", "-") == "deepintshield":
                    from ..version import __version__

                    version = __version__
                else:
                    version = importlib.metadata.version(name)
                pin = f"{name}=={version}"
            except importlib.metadata.PackageNotFoundError:
                raise _error("managed_execution_dependency_missing") from None
            if not _PIN.fullmatch(pin):
                raise _error("managed_execution_dependency_missing")
            pins.add(pin)
    if len(pins) > _MAX_REQUIREMENTS:
        raise _error("managed_execution_bundle_too_large")
    return {
        "entrypoint": entrypoint.name,
        "files": dict(sorted(files.items())),
        "requirements": sorted(pins),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


def handoff_if_enabled(engine: Any, app: Any) -> None:
    """Submit one invocation before any node runs and follow its server result.

    Returns, leaving the program to run locally, when the gateway does not offer
    server runs, or offers them with a local fall-back to a program it cannot take.
    A program that ran locally once stays local for the life of the process.
    """
    if getattr(engine, "_managed_execution_local", False) is True:
        # Not asked again: an offer made now would be for the whole script,
        # and the server would repeat every invocation that already ran here.
        return
    settings = capabilities(engine, refresh_disabled=True)
    if not settings:
        return
    # A gateway that offers server runs to every workspace lets a program it
    # cannot take keep running locally, as it did before the offer existed.
    fallback = settings.get("local_fallback") is True
    # ``fallback`` is narrowed below to "nothing was handed over yet"; the
    # offer itself also covers a run the server ended before its program ran.
    offered_fallback = fallback
    if not fallback:
        _require_autonomous_context(engine)
    with _LOCK:
        if getattr(engine, "_managed_execution_local", False) is True:
            # Decided by an invocation that held the lock before this one.
            return
        submitted = getattr(engine, "_managed_execution_submission", None)
        if fallback:
            # Once anything was handed to the server it owns the run; decide
            # under the lock so two first invocations cannot disagree.
            fallback = submitted is None and getattr(engine, "_managed_execution_wire", None) is None
            if fallback and getattr(engine, "_managed_execution_local", False) is True:
                return
            if fallback and _serves_requests():
                # Handing over a hosted program would end the request or test it
                # was called from and start the host itself on the worker.
                engine._managed_execution_local = True
                return
            try:
                _require_autonomous_context(engine)
            except GovernanceConfigurationError as error:
                if not _runs_locally(engine, error, fallback):
                    raise
                return
        if submitted is None:
            identified = settings.get("submission_ids") is True
            wire = getattr(engine, "_managed_execution_wire", None) if identified else None
            if wire is None:
                from .registry import _attach_registered_by, _wire_attested_payload, describe_network

                try:
                    script = getattr(sys.modules.get("__main__"), "__file__", None)
                    if not isinstance(script, str) or not script:
                        raise _error("managed_execution_bundle_unsupported")
                    bundle = package_script(Path(script), settings)
                    if fallback and _worker_lacks(settings, bundle["requirements"]):
                        # Sent, the job would wait for a package its worker never gets. Here the program runs.
                        raise _error("managed_execution_dependency_missing")
                    manifest = describe_network(app)
                    manifest["auto_provision"] = True
                    # The worker's sighting names the same registering developer
                    # as a local run would, with ``registered_by_source`` beside
                    # the address; ``registered_by`` is never a delegation
                    # context (see _require_autonomous_context). A delegated run
                    # is refused before this point, so the address a bundle
                    # carries is the explicit setting, never a requester hint.
                    _attach_registered_by(engine, manifest)
                    bundle["manifest"] = _wire_attested_payload(engine, augment_manifest(engine, manifest, bundle["files"]))
                    if identified:
                        # Invocation identity is separate from immutable package
                        # evidence. A fresh process may execute the same program.
                        bundle["submission_id"] = str(uuid.uuid4())
                    wire = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    if len(wire) > _limit(settings, "max_bundle_bytes", _MAX_BYTES):
                        raise _error("managed_execution_bundle_too_large")
                except GovernanceConfigurationError as error:
                    # Nothing has been sent yet, so nothing can run twice.
                    if not _runs_locally(engine, error, fallback):
                        raise
                    return
                if identified:
                    # Persist before I/O: a lost response must never cause a
                    # second invocation or upload changed source on retry.
                    engine._managed_execution_wire = wire
            submitted = _post_submission(engine, wire)
            engine._managed_execution_submission = submitted
        elif settings.get("submission_ids") is True and settings.get("result_polling") is True:
            wire = getattr(engine, "_managed_execution_wire", None)
            if not isinstance(wire, bytes):
                raise _error("managed_execution_response_invalid")
            update = _post_submission(engine, wire)
            if update["job"]["id"] != submitted["job"]["id"]:
                raise _error("managed_execution_response_invalid")
            submitted = update
            engine._managed_execution_submission = submitted
        if settings.get("submission_ids") is True and settings.get("result_polling") is True:
            submitted = _poll_submission(engine, settings, submitted)
        code = _submission_error_code(submitted["job"])
        if code is not None:
            if offered_fallback and _identity_cannot_use_server_run(submitted["job"]):
                # The server ended this run in preparation: the agent's
                # identity cannot use a server run, so the program never ran
                # there. It runs here, and stays here for the life of the
                # process, so nothing is submitted again. The gateway answers
                # the next start's probe with "not offered" as well.
                engine._managed_execution_local = True
                log.info(
                    "DeepintShield: this agent's identity cannot use a server run; "
                    "the program runs on this machine."
                )
                return
            # Keep the cached submission even when the caller catches this
            # error. The server owns continuation; never cancel or re-execute.
            raise _error(code)
        # The saved result is stdout from the complete script, including its
        # print(result) after graph.invoke. Continuing locally would execute
        # the graph or its subsequent side effects again.
        sys.stdout.write(_completed_output(engine, submitted["job"]))
        sys.stdout.flush()
        raise SystemExit(0)


def _serves_requests() -> bool:
    """Whether this invocation belongs to a hosted program rather than a script.

    A server run is for a standalone script that starts, runs its graph and
    ends. Three signs say this is something else: the call is on a worker
    thread, a host such as a web server, a task queue or a notebook kernel is
    on the call stack, or the process was started from installed code rather
    than the user's own file. A host that is only imported is no sign, because
    client libraries load them too. Any one sign is enough: the cost of a wrong
    guess is only that the program runs where it already runs.
    """
    if threading.current_thread() is not threading.main_thread():
        return True
    frame = sys._getframe()
    while frame is not None:
        name = frame.f_globals.get("__name__")
        if isinstance(name, str) and (name.partition(".")[0] in _SERVING_MODULES or name.startswith(_SERVING_PREFIXES)):
            return True
        frame = frame.f_back
    return _entry_is_installed_code()


def _entry_is_installed_code() -> bool:
    """Whether the process was started from installed code, not the user's script.

    True for ``python -m pytest`` or ``python -m uvicorn``: the entry file sits
    in a site-packages directory or in the interpreter's own library.
    """
    script = getattr(sys.modules.get("__main__"), "__file__", None)
    if not isinstance(script, str) or not script:
        return False
    # An installer or the interpreter may reach the same file through a link,
    # so look at the path as given and as resolved.
    given = os.path.normcase(os.path.abspath(script))
    resolved = os.path.normcase(os.path.realpath(script))
    if {"site-packages", "dist-packages"}.intersection(given.split(os.sep) + resolved.split(os.sep)):
        return True
    # Imported here so that it costs nothing where server runs are not offered.
    import sysconfig

    paths = sysconfig.get_paths()
    return resolved.startswith(tuple(
        os.path.normcase(os.path.realpath(paths[key])) + os.sep
        for key in ("stdlib", "platstdlib", "purelib", "platlib") if paths.get(key)
    ))


def _runs_locally(engine: Any, error: GovernanceConfigurationError, fallback: bool) -> bool:
    """Keep a program the server cannot take on its ordinary local path.

    Only a delegated context, a program that cannot be packaged or one that needs
    a package the worker lacks qualifies. The choice is remembered so later
    invocations in this process neither package again nor hand the same program
    to the server after all.
    """
    if not fallback or error.code not in {
        "managed_execution_context_unsupported", "managed_execution_bundle_unsupported",
        "managed_execution_dependency_missing", "managed_execution_bundle_too_large",
    }:
        return False
    engine._managed_execution_local = True
    return True


def _worker_lacks(settings: dict[str, Any], requirements: list[str]) -> bool:
    """Whether the worker image is known to lack a distribution the program imports.

    The gateway may name what its worker image has installed, each name in the
    normalized form: lower case, separators as one "-". Names alone are compared,
    never releases: the worker runs a program on whichever releases it has. A
    list that was not sent, or is not a list of names, says nothing.
    """
    installed = settings.get("worker_packages")
    if not isinstance(installed, list) or not all(isinstance(name, str) for name in installed):
        return False
    names = set(installed)
    return any(
        _NAME_SEPARATORS.sub("-", requirement.partition("==")[0]).lower() not in names
        for requirement in requirements
    )


def _post_submission(engine: Any, wire: bytes, *, timeout: float = 30.0) -> dict[str, Any]:
    try:
        response = engine._http.post(
            f"{engine.gateway_url.rstrip('/')}{_PREFIX}/submissions",
            headers={**engine._headers(include_agent_token=False), "Content-Type": "application/json"},
            content=wire,
            timeout=timeout,
            follow_redirects=False,
        )
    except GovernanceConfigurationError:
        raise
    except Exception:
        raise _error("managed_execution_submission_failed") from None
    submitted = _response(response, max_bytes=_MAX_SUBMISSION_RESPONSE_BYTES)
    job = submitted.get("job")
    if response.status_code != 202 or not isinstance(job, dict) or not isinstance(job.get("id"), str) or not _JOB_ID.fullmatch(job["id"]):
        raise _error("managed_execution_response_invalid")
    return submitted


def _poll_submission(engine: Any, settings: dict[str, Any], submitted: dict[str, Any]) -> dict[str, Any]:
    """Poll only this invocation; operator holds never keep a caller waiting."""
    active = ("queued", "preparing", "running")
    if submitted["job"].get("status") not in active:
        return submitted
    interval = settings.get("poll_interval_seconds", 1.0)
    timeout = settings.get("wait_timeout_seconds", 30.0)
    for value in (interval, timeout):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise _error("managed_execution_response_invalid")
    interval = min(max(interval, 0.05), 10.0)
    deadline = time.monotonic() + min(timeout, 60.0)
    wire = getattr(engine, "_managed_execution_wire", None)
    if not isinstance(wire, bytes):
        raise _error("managed_execution_response_invalid")
    job_id = submitted["job"]["id"]
    while submitted["job"].get("status") in active:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        update = _post_submission(engine, wire, timeout=min(30.0, remaining))
        if update["job"]["id"] != job_id:
            raise _error("managed_execution_response_invalid")
        submitted = update
        engine._managed_execution_submission = submitted
    return submitted


def wait_for_worker_approval(engine: Any, context: Any, decision: Any) -> Any:
    """Keep the exact blocked operation in its live managed-worker call frame.

    Only the controlled launcher sets _WORKER_FILES. Ordinary SDK callers keep
    the existing pending-error contract. Each poll rechecks current policy with
    the same immutable operands; neither graph state nor completed nodes replay.
    The worker's process timeout/cancellation still bounds the entire job.
    """
    from .types import Verdict

    if _WORKER_FILES is None or not decision.gaf_checked or decision.verdict != Verdict.REQUIRE_APPROVAL or not decision.approval_id:
        return decision
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # A synchronous gate cannot await a long approval without starving an
        # active event loop and delaying task cancellation. Keep async callers
        # on the existing fail-closed pending path until they have an async gate.
        return decision
    try:
        timeout = float(getattr(engine, "_approval_timeout", 240.0))
        interval = float(getattr(engine, "_approval_interval", 2.0))
    except (TypeError, ValueError):
        return decision
    if not math.isfinite(timeout) or not math.isfinite(interval) or timeout <= 0:
        return decision
    # Leave time for failure reporting before the default five-minute worker
    # deadline. A process cancellation interrupts this wait without any replay.
    deadline = time.monotonic() + min(timeout, 240.0)
    if _WORKER_APPROVAL_DEADLINE is not None:
        deadline = min(deadline, _WORKER_APPROVAL_DEADLINE)
    interval = min(max(interval, 0.05), 10.0)
    frozen = context.model_copy(deep=True)
    while decision.verdict == Verdict.REQUIRE_APPROVAL:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))
        if time.monotonic() >= deadline:
            break
        decision = engine.decide(frozen.model_copy(deep=True))
        if not decision.gaf_checked:
            raise _error("managed_execution_unavailable")
        if decision.verdict == Verdict.REQUIRE_APPROVAL and not decision.approval_id:
            break
    return decision


def _require_autonomous_context(engine: Any) -> None:
    """Never silently replace delegated execution with the worker's identity."""
    from .task_context import current_task_context

    # Check the hint first: resolving a principal may contact the gateway, and
    # submission must not resolve or upload a user's delegation credentials.
    if str(getattr(engine, "_requester_hint", "") or "").strip():
        raise _error("managed_execution_context_unsupported")
    if getattr(engine, "principal_binding", None) is not None or str(getattr(engine, "acting_user", "") or "").strip():
        raise _error("managed_execution_context_unsupported")
    if current_task_context() is not None:
        raise _error("managed_execution_context_unsupported")
