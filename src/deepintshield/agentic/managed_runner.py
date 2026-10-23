"""Launcher for an operator-configured, isolated managed-execution worker.

This adapter is not a sandbox. The server supplies an isolated worker, approved
installed dependencies, a read-only bundle/config, and bounded network and I/O.
No dependency installation or automatic retry of application code occurs here.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile
import time

_ALLOWED_ENV = frozenset({
    "DEEPINTSHIELD_BASE_URL", "DEEPINTSHIELD_GATEWAY_URL", "DEEPINTSHIELD_VIRTUAL_KEY",
    "DEEPINTSHIELD_AGENT_NAME", "DEEPINTSHIELD_REQUESTER", "DEEPINTSHIELD_REQUESTER_ROLE",
    "DEEPINTSHIELD_REGISTERED_BY",
    "DEEPINTSHIELD_APP_NAME", "DEEPINTSHIELD_TIMEOUT", "DEEPINTSHIELD_PERSIST",
    "DEEPINTSHIELD_RUNTIME_KEY_FILE", "DEEPINTSHIELD_RUNTIME_KIND",
    "DEEPINTSHIELD_RUNTIME_AUDIENCE", "DEEPINTSHIELD_RUNTIME_TOKEN_FILE",
    "DEEPINTSHIELD_RUNTIME_AZURE_CLIENT_ID", "DEEPINTSHIELD_RUNTIME_AWS_REGION",
})


def _fail(code: str) -> int:
    print(f"DEEPINTSHIELD_MANAGED_ERROR:{code}", file=sys.stderr)
    return 1


def _dependencies_available(requirements: list[str]) -> bool:
    """Require only that every distribution the program imports is installed.

    The worker always executes its own SDK, Python and framework releases, so
    one image serves clients on any SDK or Python release. The submitted pins
    stay in the bundle as review evidence; nothing is installed or rewritten
    here. Keep the gateway preflight in sync with this rule.
    """
    from .managed_execution import _PIN

    if len(requirements) > 64:
        return False
    for requirement in requirements:
        if not isinstance(requirement, str) or not _PIN.fullmatch(requirement):
            return False
        name = requirement.split("==", 1)[0]
        if name.lower().replace("_", "-") == "deepintshield":
            # This launcher is the installed SDK.
            continue
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    from deepintshield import DeepintShield
    from deepintshield.errors import get_exception_error_code
    from . import managed_execution

    parser = argparse.ArgumentParser(description="Run an approved managed script.")
    parser.add_argument("entrypoint")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    original_environment = dict(os.environ)
    key_path: str | None = None
    approval_wait_seconds = 240.0
    if args.config:
        try:
            with open(args.config, "rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError()
            configuration = json.loads(raw)
            overrides = configuration["environment"]
            private_key = configuration["runtime_private_key"]
            requirements = configuration.get("requirements", [])
            python_version = configuration.get("python_version", "")
            approval_wait_seconds = configuration.get("approval_wait_timeout_seconds", 240.0)
            if isinstance(approval_wait_seconds, bool) or not isinstance(approval_wait_seconds, (int, float)) or not math.isfinite(approval_wait_seconds) or not 0 <= approval_wait_seconds <= 240:
                raise ValueError()
            if not isinstance(overrides, dict) or not isinstance(private_key, str) or not 1 <= len(private_key) <= 16384 or not isinstance(requirements, list):
                raise ValueError()
            if any(key not in _ALLOWED_ENV or not isinstance(value, str) for key, value in overrides.items()):
                raise ValueError()
        except (OSError, ValueError, TypeError, KeyError):
            return _fail("managed_execution_configuration_invalid")
        if not _dependencies_available(requirements):
            return _fail("managed_dependency_unavailable")
        # The submitting interpreter version is evidence only: this worker's own
        # Python runs the program whichever release the client used.
        if not isinstance(python_version, str) or (python_version and not re.fullmatch(r"3\.[0-9]{1,2}(?:\.[0-9]{1,3})?", python_version)):
            return _fail("managed_execution_configuration_invalid")
        try:
            descriptor, key_path = tempfile.mkstemp(prefix="deepintshield-runtime-", suffix=".pem")
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(private_key)
            overrides = {**overrides, "DEEPINTSHIELD_RUNTIME_KEY_FILE": key_path}
        except (OSError, UnicodeError):
            if key_path:
                Path(key_path).unlink(missing_ok=True)
            return _fail("managed_execution_configuration_invalid")
    else:
        # An explicitly launched worker may use a mounted key and the same
        # allowlisted configuration supplied by its operator, never app env.
        overrides = {key: value for key, value in original_environment.items() if key in _ALLOWED_ENV}
    overrides["DEEPINTSHIELD_MANAGED_EXECUTION_WORKER"] = "1"
    required = ("DEEPINTSHIELD_BASE_URL", "DEEPINTSHIELD_VIRTUAL_KEY", "DEEPINTSHIELD_AGENT_NAME", "DEEPINTSHIELD_RUNTIME_KEY_FILE")
    script = Path(args.entrypoint).absolute()
    descriptor = DeepintShield.__dict__["from_env"]
    old_path, old_argv, old_directory = list(sys.path), list(sys.argv), os.getcwd()
    old_worker_files = managed_execution._WORKER_FILES
    old_approval_deadline = managed_execution._WORKER_APPROVAL_DEADLINE

    def apply_overrides() -> None:
        for key in list(os.environ):
            if key.startswith("DEEPINTSHIELD_"):
                del os.environ[key]
        os.environ.update(overrides)

    @classmethod
    def from_worker_env(cls):
        # Source may assign its original localhost/VK/name before this call.
        apply_overrides()
        return descriptor.__func__(cls)

    try:
        if any(not overrides.get(key) for key in required):
            return _fail("managed_execution_configuration_invalid")
        bundle = managed_execution.package_script(script, {})
        managed_execution._WORKER_FILES = bundle["files"]
        managed_execution._WORKER_APPROVAL_DEADLINE = time.monotonic() + approval_wait_seconds
        DeepintShield.from_env = from_worker_env
        apply_overrides()
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script)]
        os.chdir(script.parent)
        runpy.run_path(str(script), run_name="__main__")
        return 0
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    except BaseException as error:
        code = get_exception_error_code(error)
        print(f"DEEPINTSHIELD_MANAGED_ERROR:{code or 'managed_execution_failed'}", file=sys.stderr)
        return 20 if code else 1
    finally:
        DeepintShield.from_env = descriptor
        managed_execution._WORKER_FILES = old_worker_files
        managed_execution._WORKER_APPROVAL_DEADLINE = old_approval_deadline
        sys.path[:] = old_path
        sys.argv[:] = old_argv
        os.chdir(old_directory)
        os.environ.clear()
        os.environ.update(original_environment)
        if key_path:
            Path(key_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
