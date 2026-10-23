"""HTTP worker for the gateway's "remote" managed-execution runner.

Run as ``python -I -m deepintshield.agentic.managed_remote`` from the worker
image. The gateway POSTs one approved job to ``/run``; this process stages the
bundle, asks the gateway to grant that run its one start, and only then starts
the ordinary ``managed_runner`` launcher as a child with bounded output and a
hard deadline. The outcome is returned in the HTTP response.

The grant is how the gateway knows whether a program started. A start can be
asked for once and, until it has been, withdrawn. So after a run request that
ended without this worker's answer the gateway withdraws it: if that succeeds,
the program never started and never will.

A process accepts exactly one run and then exits, so the platform (a Cloud Run
service, Kubernetes, a compose service) gives every run a fresh instance and
nothing of one run can reach another. This is not a sandbox: isolation is the
platform's container. Only the standard library is imported here; the child
process loads the SDK and the program.
"""

from __future__ import annotations

import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import socketserver
import ssl
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, NamedTuple
import urllib.error
import urllib.request

_PORT_SETTING = "PORT"
_IMAGE_SETTING = "DEEPINTSHIELD_MANAGED_WORKER_IMAGE"
_AUTH_SETTING = "DEEPINTSHIELD_MANAGED_WORKER_AUTH"
_TOKEN_SETTING = "DEEPINTSHIELD_MANAGED_WORKER_TOKEN"
_DEFAULT_PORT = "8080"
# An immutable reference only: "<repository>@sha256:<digest>" or a bare image ID.
# The gateway accepts the same pattern for the image it pins jobs to.
_IMAGE = re.compile(r"(?:[A-Za-z0-9._:/-]+@)?sha256:[a-f0-9]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{32,512}")
_ROUTES = {"/healthz": "GET", "/inventory": "GET", "/run": "POST"}
# Sent with every answer. A gateway of another release expects another value
# and then leaves this worker alone, instead of sending it a request it would refuse.
_PROTOCOL = "2"
_RUN_FIELDS = frozenset({"image", "entrypoint", "files", "config", "timeout_seconds", "start"})
_START_FIELDS = frozenset({"url", "token"})
# Where a run asks the gateway for its start: one fixed route of an origin and
# nothing more. Matched in full, so that no URL parser has to agree with the
# client that sends the request.
_START_URL = re.compile(
    r"(?P<scheme>(?i:https?))://(?:(?P<name>[A-Za-z0-9._-]+)|\[(?P<literal>[0-9A-Fa-f:.]+)\])(?::(?P<port>[0-9]{0,5}))?"
    r"/api/agentic-new/managed-execution/start"
)
_START_TOKEN = re.compile(r"[\x21-\x7e]{16,4096}")
_MAX_URL_CHARACTERS = 2048
_GRANT_SECONDS = 15.0
_GRANT_ANSWER_BYTES = 4096
# A fixed name: urllib's own would tell the gateway which Python the worker runs.
_GRANT_AGENT = "deepintshield-managed-worker"
_MAX_BODY_BYTES = 4 * 1024 * 1024
_MAX_FILES = 64
_MAX_PATH_CHARACTERS = 512
_MAX_TIMEOUT_SECONDS = 3600
_MAX_PACKAGES = 1024
_STDOUT_LIMIT = 48 * 1024
_STDERR_LIMIT = 16 * 1024
_REQUEST_TIMEOUT_SECONDS = 30
_POLL_SECONDS = 0.25
_REAP_SECONDS = 5.0
_READER_WAIT_SECONDS = 2.0
_LINGER_SECONDS = 5.0
_LINGER_BYTES = 2 * _MAX_BODY_BYTES
# The unprivileged user of the worker image, used when the worker itself is root.
_PROGRAM_USER = 65532
_DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
_CHILD_PASSTHROUGH = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE")


class _InvalidSetting(Exception):
    """Names the offending setting; never carries its value."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _Order(NamedTuple):
    """One validated /run request, already encoded for staging."""

    image: str
    entrypoint: str
    files: dict[str, bytes]
    config: bytes
    timeout: int
    start_url: str
    start_token: str


def _settings(environ: Mapping[str, str]) -> tuple[int, str, str | None]:
    """Return (port, image, token); the token is None when the platform authenticates callers."""
    port = environ.get(_PORT_SETTING) or _DEFAULT_PORT
    if not re.fullmatch(r"[0-9]{1,5}", port) or not 1 <= int(port) <= 65535:
        raise _InvalidSetting(_PORT_SETTING)
    image = environ.get(_IMAGE_SETTING) or ""
    if not _IMAGE.fullmatch(image):
        raise _InvalidSetting(_IMAGE_SETTING)
    auth = environ.get(_AUTH_SETTING) or ""
    if auth not in ("platform", "token"):
        raise _InvalidSetting(_AUTH_SETTING)
    token = None
    if auth == "token":
        token = environ.get(_TOKEN_SETTING) or ""
        if not _TOKEN.fullmatch(token):
            raise _InvalidSetting(_TOKEN_SETTING)
    return int(port), image, token


def _refusal(code: str) -> dict[str, Any]:
    """The body of every answer that is not a run outcome: no program was started.

    "invalid_request" and "payload_too_large" are verdicts on the package, and
    the gateway ends its job for them. A request that could not be read as it
    was sent is "unreadable_request", which leaves the job to be tried again.
    """
    return {"started": False, "code": code}


def _not_json(_constant: str) -> Any:
    # NaN and Infinity are Python extensions, not JSON.
    raise ValueError()


def _relative_path(name: str) -> bool:
    # An empty component also covers "", a leading or trailing slash and "//".
    return (
        0 < len(name) <= _MAX_PATH_CHARACTERS
        and "\\" not in name
        and "\x00" not in name
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


def _start_url(url: str) -> bool:
    """Whether ``url`` is a gateway's start route, over https or over http to this machine only.

    The request that goes there carries a credential.
    """
    match = _START_URL.fullmatch(url) if len(url) <= _MAX_URL_CHARACTERS else None
    if match is None or int(match["port"] or 0) > 65535:
        return False
    name, literal = match["name"], match["literal"]
    try:
        address = ipaddress.IPv6Address(literal) if literal else ipaddress.IPv4Address(name)
    except ValueError:
        if literal:
            return False  # brackets hold an IPv6 address and nothing else
        address = None  # a name
    if match["scheme"].lower() == "https":
        return True
    if address is None:
        return name.lower() == "localhost"
    # ::ffff:127.0.0.1 is the same loopback interface.
    return (getattr(address, "ipv4_mapped", None) or address).is_loopback


def _parse_order(raw: bytes) -> _Order | None:
    """Return the run request, or None for anything the contract does not allow.

    Everything is checked here, before the worker is used up and before anything
    is written to disk.
    """
    try:
        body = json.loads(raw.decode("utf-8"), parse_constant=_not_json)
    except (ValueError, RecursionError):
        return None
    if not isinstance(body, dict) or body.keys() != _RUN_FIELDS:
        return None
    image, entrypoint, files, config, timeout, start = (
        body[field] for field in ("image", "entrypoint", "files", "config", "timeout_seconds", "start")
    )
    if not (isinstance(image, str) and isinstance(entrypoint, str) and isinstance(files, dict) and isinstance(config, dict)):
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= _MAX_TIMEOUT_SECONDS:
        return None
    if not isinstance(start, dict) or start.keys() != _START_FIELDS:
        return None
    url, token = start["url"], start["token"]
    if not (isinstance(url, str) and isinstance(token, str) and _start_url(url) and _START_TOKEN.fullmatch(token)):
        return None
    if not 1 <= len(files) <= _MAX_FILES or not entrypoint.endswith(".py") or entrypoint not in files:
        return None
    encoded: dict[str, bytes] = {}
    try:
        for name, source in files.items():
            if not _relative_path(name) or not isinstance(source, str):
                return None
            # A lone surrogate from a JSON escape cannot name or fill a file.
            name.encode("utf-8")
            encoded[name] = source.encode("utf-8")
        launcher_config = json.dumps(config).encode("ascii")
    except (ValueError, RecursionError):
        return None
    for name in encoded:
        parts = name.split("/")
        # A file cannot also be the directory of another file.
        if any("/".join(parts[:depth]) in encoded for depth in range(1, len(parts))):
            return None
    return _Order(image, entrypoint, encoded, launcher_config, timeout, url, token)


def _runtime_inventory() -> dict[str, Any]:
    """Mirror the gateway's Docker probe: the same fields, order and limit.

    Only installed distribution metadata is read; nothing is imported. Keep this
    in step with managedRuntimeInventoryScript in the gateway's preflight.
    """
    packages = []
    for distribution in importlib.metadata.distributions():
        packages.append({"name": distribution.metadata["Name"], "version": distribution.version})
        if len(packages) > _MAX_PACKAGES:
            raise ValueError()
    return {"python_version": ".".join(map(str, sys.version_info[:3])), "packages": packages}


def _child_command(config_path: str, entrypoint_path: str) -> list[str]:
    """The reviewed launcher, exactly as the Docker runner starts it."""
    return [sys.executable, "-I", "-m", "deepintshield.agentic.managed_runner", "--config", config_path, entrypoint_path]


def _child_environment(environ: Mapping[str, str], root: str) -> dict[str, str]:
    """Built from nothing: no worker token, DEEPINTSHIELD_* or cloud variable reaches the program."""
    environment = {
        "PATH": environ.get("PATH") or _DEFAULT_PATH,
        "HOME": os.path.join(root, "home"),
        "TMPDIR": os.path.join(root, "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
    }
    for name in _CHILD_PASSTHROUGH:
        value = environ.get(name)
        if value:
            environment[name] = value
    return environment


def _program_identity() -> tuple[int, int] | None:
    """A root worker runs the program as the image's unprivileged user and group."""
    return (_PROGRAM_USER, _PROGRAM_USER) if os.geteuid() == 0 else None


def _deny_inspection() -> None:
    """Stop a program that runs as the worker's own user from reading the worker.

    A non-dumpable Linux process cannot be traced, and its memory and environment
    (which hold the worker token) cannot be read through /proc, by that user.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE, off
    except Exception:
        pass


def _make_directory(path: str, mode: int) -> None:
    os.mkdir(path, mode)
    os.chmod(path, mode)  # exact modes whatever the process umask


def _write_file(path: str, data: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        os.fchmod(descriptor, mode)


def _stage(order: _Order, owner: tuple[int, int] | None) -> str:
    """Write the bundle and the launcher configuration into a fresh private directory."""
    # The launcher refuses a bundle reached through a symlinked component (macOS /var, /tmp).
    root = os.path.realpath(tempfile.mkdtemp(prefix="deepintshield-managed-"))
    try:
        bundle = os.path.join(root, "bundle")
        _make_directory(bundle, 0o755)
        created = [root, bundle]
        for name in sorted(order.files):
            *folders, leaf = name.split("/")
            directory = bundle
            for folder in folders:
                directory = os.path.join(directory, folder)
                if directory not in created:
                    _make_directory(directory, 0o755)
                    created.append(directory)
            target = os.path.join(directory, leaf)
            # Validation already refused such names; never write outside the bundle regardless.
            if os.path.commonpath((bundle, os.path.abspath(target))) != bundle:
                raise ValueError()
            _write_file(target, order.files[name], 0o444)
            created.append(target)
        # Keep the runtime key out of the program's own directory.
        config_path = os.path.join(root, "config.json")
        _write_file(config_path, order.config, 0o400)
        created.append(config_path)
        for name in ("home", "tmp"):
            _make_directory(os.path.join(root, name), 0o700)
            created.append(os.path.join(root, name))
        if owner is not None:
            for path in created:
                os.chown(path, *owner)
    except BaseException:
        _remove_tree(root)
        raise
    return root


def _remove_tree(root: str) -> None:
    """Best effort and never an error: the directory holds the bundle and the run's credentials."""
    try:
        shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass  # older Pythons raise for a tree deeper than the recursion limit, whatever was asked
    try:
        if os.path.lexists(root):
            _remove_remaining(root)
    except Exception:
        pass


def _remove_remaining(root: str) -> None:
    """Remove what ``shutil.rmtree`` left, without recursion and with one directory open at a time.

    A program can build a tree deeper than the recursion limit, than the
    descriptors a process may hold and than the longest path. So a directory is
    entered by name from its parent and left through "..". No link is followed,
    and the walk ends where ".." is not the directory it came from.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    # A program running as the worker's own user can have closed its directories
    # to that user. Root needs no such help, and must never change the mode of
    # something a program was able to point a link at.
    reopen = os.geteuid() != 0
    if reopen and stat.S_ISDIR(os.lstat(root).st_mode):
        os.chmod(root, 0o700)
    directory = os.open(root, flags)
    # For each directory the walk is inside: its name in the one above, which
    # directory that one is, and what is still to be entered there.
    trail: list[tuple[str, tuple[int, int], list[str]]] = []
    try:
        folders = _empty(directory)
        while folders or trail:
            if folders:
                name = folders.pop()
                here = os.fstat(directory)
                try:
                    if reopen:
                        os.chmod(name, 0o700, dir_fd=directory)
                    entered = os.open(name, flags, dir_fd=directory)
                except OSError:
                    continue  # it stays, and so does every directory above it
                trail.append((name, (here.st_dev, here.st_ino), folders))
                left, directory = directory, entered
                os.close(left)
                folders = _empty(directory)
            else:
                name, expected, folders = trail.pop()
                above = os.open("..", flags, dir_fd=directory)
                left, directory = directory, above
                os.close(left)
                found = os.fstat(directory)
                if (found.st_dev, found.st_ino) != expected:
                    return  # moved while the walk was inside it: what is above is not this run's
                try:
                    os.rmdir(name, dir_fd=directory)
                except OSError:
                    pass
    finally:
        os.close(directory)
    os.rmdir(root)


def _empty(directory: int) -> list[str]:
    """Unlink everything in an open directory except its directories, and return their names."""
    with os.scandir(directory) as entries:
        listing = list(entries)  # read to the end first: removing while reading can skip entries
    folders = []
    for entry in listing:
        try:
            if entry.is_dir(follow_symlinks=False):
                folders.append(entry.name)
            else:
                os.unlink(entry.name, dir_fd=directory)
        except OSError:
            pass
    return folders


def _grant_opener() -> urllib.request.OpenerDirector:
    """urllib's own handlers, without the one that follows a redirect.

    Proxy settings and the CA variables of the environment apply as they do to
    any urllib request, and certificates and host names are verified.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def _ask_gateway(url: str, token: str) -> bool:
    """Send the one request for a start, and return True only for the gateway's plain yes."""
    request = urllib.request.Request(
        url,
        data=b"{}",
        method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json", "User-Agent": _GRANT_AGENT},
    )
    try:
        response = _grant_opener().open(request, timeout=_GRANT_SECONDS)
    except urllib.error.HTTPError as refused:
        refused.close()  # every status that is not a success arrives this way, with its connection open
        return False
    with response:
        if response.status != HTTPStatus.OK:
            return False
        raw = response.read(_GRANT_ANSWER_BYTES + 1)
    if len(raw) > _GRANT_ANSWER_BYTES:
        return False
    answer = json.loads(raw.decode("utf-8"), parse_constant=_not_json)
    return isinstance(answer, dict) and answer.get("granted") is True


def _start_granted(url: str, token: str, stopping: threading.Event) -> bool:
    """Ask the gateway to grant this run its one start; anything but a yes in time is a no.

    The worker starts a program only with this grant. Whatever goes wrong on
    the way is no grant, and is never printed: a failure can quote the URL.
    """
    if stopping.is_set():
        # Asking uses the start up, and an instance that is being taken away
        # would start nothing with it. Unasked, the gateway can withdraw it.
        return False
    answered = threading.Event()
    verdict: list[bool] = []

    def ask() -> None:
        try:
            verdict.append(_ask_gateway(url, token))
        except Exception:
            pass
        finally:
            answered.set()

    # A thread of its own: no socket timeout bounds a name lookup, or an answer that trickles in.
    threading.Thread(target=ask, daemon=True).start()
    until = time.monotonic() + _GRANT_SECONDS
    while not answered.wait(min(_POLL_SECONDS, max(until - time.monotonic(), 0.0))):
        if stopping.is_set() or time.monotonic() >= until:
            return False
    # An instance that is being taken away starts nothing, whatever the gateway said.
    return verdict == [True] and not stopping.is_set()


def _report_abandoned() -> None:
    """Say once, in fixed words, that a run ended because its caller's connection closed.

    A hop that half-closes once its request is sent looks like a caller that
    left, and nothing else would tell an operator why runs end unanswered.
    """
    print(
        "managed worker: run abandoned: the caller closed its connection "
        "(a proxy that half-closes after sending the request looks the same)",
        file=sys.stderr,
    )


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass  # nothing is left in the group


def _watch(process: subprocess.Popen[bytes], exited: threading.Event) -> None:
    """Report the child's exit without reaping it where the platform can.

    An unreaped child keeps its process ID, so the group kill that follows cannot
    reach an unrelated group that was given the same number.
    """
    try:
        peeked = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    except (AttributeError, OSError):
        peeked = None  # no waitid here (macOS before Python 3.13)
    if peeked is None:
        process.wait()  # reaped as it is noticed instead
    exited.set()


def _bounded(text: str, limit: int) -> str:
    """Keep decoded output within ``limit`` bytes of UTF-8 and three times that as JSON.

    A replacement character is longer than the byte it stands for, and JSON
    writes a control character as six: unbounded, an answer could outgrow what
    the gateway accepts and leave a finished run looking uncertain. Ordinary
    text never reaches the JSON bound; the cut is always on a character.
    """
    encoded = text.encode("utf-8")
    if len(encoded) > limit:
        text = encoded[:limit].decode("utf-8", "ignore")
    budget = 3 * limit + 2  # the two quotes
    if len(json.dumps(text)) > budget:
        fits, over = 0, len(text)
        while over - fits > 1:
            middle = (fits + over) // 2
            if len(json.dumps(text[:middle])) <= budget:
                fits = middle
            else:
                over = middle
        text = text[:fits]
    return text


class _Capture:
    """Keep the first ``limit`` bytes of a pipe and discard the rest, so the program never blocks on output."""

    def __init__(self, pipe: Any, limit: int) -> None:
        self._pipe = pipe
        self._limit = limit
        self._kept = bytearray()
        self._guard = threading.Lock()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        # This thread owns the pipe: a process the program left behind can hold
        # it open long after the answer has been sent.
        with self._pipe:
            while True:
                try:
                    chunk = self._pipe.read(65536)
                except (OSError, ValueError):
                    return
                if not chunk:
                    return
                with self._guard:
                    room = self._limit - len(self._kept)
                    if room > 0:
                        self._kept += chunk[:room]

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def text(self) -> str:
        with self._guard:
            return _bounded(bytes(self._kept).decode("utf-8", "replace"), self._limit)


class _Worker(ThreadingHTTPServer):
    """The listener plus the state that makes this process single use."""

    daemon_threads = True  # /healthz answers while a run is in progress
    request_queue_size = 32

    def __init__(
        self,
        address: tuple[str, int],
        environ: Mapping[str, str],
        image: str,
        token: str | None,
        exit: Callable[[int], Any],
        child_command: Callable[[str, str], list[str]],
    ) -> None:
        self.environ = environ
        self.image = image
        # Only a digest of the expected header is kept, and digests are what get compared.
        self.credential = None if token is None else hashlib.sha256(b"Bearer " + token.encode("ascii")).digest()
        self.exit = exit
        self.child_command = child_command
        self.finished = threading.Event()
        self.stopping = threading.Event()
        self._claimed = threading.Lock()
        super().__init__(address, _Handler)

    def server_bind(self) -> None:
        # HTTPServer also resolves this host's own name here; a slow resolver would delay every cold start.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        # The default prints a traceback, which can quote request data. A caller
        # that drops its connection is not worth a line; anything else names its type only.
        error = sys.exc_info()[1]
        if not isinstance(error, OSError):
            print(f"managed worker: request failed ({type(error).__name__})", file=sys.stderr)

    def claim(self) -> bool:
        """Take this process's single run: the first caller wins and nobody ever wins again."""
        return self._claimed.acquire(blocking=False)

    def stop(self) -> None:
        """The platform is taking the instance away: end a running program now, or leave at once when idle."""
        self.stopping.set()
        # Idle means the single run could still be taken, so take it: a request
        # that arrives now is refused as busy, never accepted by a process that is leaving.
        if self.claim():
            self.finished.set()
        if self.finished.is_set():
            self.exit(0)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = _REQUEST_TIMEOUT_SECONDS
    disable_nagle_algorithm = True
    server: _Worker

    def log_message(self, format: str, *args: Any) -> None:
        # Request lines and parse errors carry caller-controlled text: write nothing per request.
        pass

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # Requests the base class rejects by itself (a malformed request line, an
        # unknown method) get the worker's own answer format as well.
        if code == HTTPStatus.NOT_IMPLEMENTED:
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, _refusal("method_not_allowed"))
        else:
            self._reply(code, _refusal("unreadable_request"))

    def _dispatch(self) -> None:
        self._accepted = False
        try:
            self._route()
        except Exception:
            if self._accepted:
                raise
            # Nothing was accepted by this request, so the program provably did not start.
            self.server.handle_error(self.request, self.client_address)
            self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, _refusal("launch_failed"))

    do_GET = do_POST = do_HEAD = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _dispatch

    def _route(self) -> None:
        path = self.path.split("?", 1)[0]
        method = _ROUTES.get(path)
        if method is None:
            self._reply(HTTPStatus.NOT_FOUND, _refusal("not_found"))
        elif method != self.command:
            self._reply(HTTPStatus.METHOD_NOT_ALLOWED, _refusal("method_not_allowed"))
        elif path == "/healthz":
            self._reply(HTTPStatus.OK, {"ok": True})
        elif not self._authorized():
            self._reply(HTTPStatus.UNAUTHORIZED, _refusal("worker_unauthorized"))
        elif path == "/inventory":
            self._serve_inventory()
        else:
            self._serve_run()

    def _authorized(self) -> bool:
        expected = self.server.credential
        if expected is None:
            # "platform": every caller was authenticated before the request reached the container.
            return True
        supplied = self.headers.get_all("Authorization") or []
        value = supplied[0] if len(supplied) == 1 else ""
        # Equal-length digests: the comparison reveals neither the token nor its length.
        return hmac.compare_digest(hashlib.sha256(value.encode("latin-1", "replace")).digest(), expected)

    def _serve_inventory(self) -> None:
        try:
            inventory = _runtime_inventory()
        except Exception:
            # Unreadable or oversized metadata fails the gateway's Docker probe in the same way.
            self._reply(HTTPStatus.INTERNAL_SERVER_ERROR, _refusal("inventory_unavailable"))
            return
        self._reply(HTTPStatus.OK, {"image": self.server.image, "inventory": inventory})

    def _serve_run(self) -> None:
        worker = self.server
        length = self._body_length()
        if length is None or self.headers.get_content_type() != "application/json":
            # About how the request was sent, not what it holds: a hop can send the same package chunked.
            self._reply(HTTPStatus.BAD_REQUEST, _refusal("unreadable_request"))
            return
        if length > _MAX_BODY_BYTES:
            self._reply(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, _refusal("payload_too_large"))
            return
        try:
            raw = self.rfile.read(length)
        except OSError:
            raw = b""  # the caller stalled or left before sending the whole body
        if len(raw) != length:
            # Nothing is wrong with the package: the same request may arrive whole next time.
            self._reply(HTTPStatus.BAD_REQUEST, _refusal("unreadable_request"))
            return
        order = _parse_order(raw)
        if order is None:
            self._reply(HTTPStatus.BAD_REQUEST, _refusal("invalid_request"))
        elif order.image != worker.image:
            self._reply(HTTPStatus.CONFLICT, _refusal("image_mismatch"))
        elif not worker.claim():
            self._reply(HTTPStatus.CONFLICT, _refusal("worker_busy"))
            if worker.finished.is_set():
                # The used-up process should already be gone; do not linger as an instance that only refuses.
                worker.exit(0)
        else:
            self._accepted = True
            self._execute(order)

    def _body_length(self) -> int | None:
        """The declared body size, or None when the request does not declare one usable size."""
        declared = self.headers.get_all("Content-Length") or []
        if len(declared) != 1 or self.headers.get("Transfer-Encoding") is not None:
            return None
        value = declared[0].strip()
        if not value.isascii() or not value.isdigit():
            return None
        # Every size of ten digits or more is far over the limit; do not convert a huge number.
        return int(value) if len(value) < 10 else _MAX_BODY_BYTES + 1

    def _execute(self, order: _Order) -> None:
        """The one accepted run. Whatever happens from here, this process ends afterwards."""
        worker = self.server
        try:
            answer = self._attempt(order)
            if answer is not None:
                self._reply(*answer)
        finally:
            worker.finished.set()
            try:
                self.connection.close()
            except OSError:
                pass
            # The worker is normally process 1 of its container, so this also ends
            # anything the program left behind.
            worker.exit(0)

    def _attempt(self, order: _Order) -> tuple[int, dict[str, Any]] | None:
        """Run the program and return the answer, or None when the caller is no longer there to hear it."""
        worker = self.server
        root = None
        try:
            try:
                if worker.stopping.is_set():
                    raise OSError()
                identity = _program_identity()
                root = _stage(order, identity)
                bundle = os.path.join(root, "bundle")
                if self._caller_gone():
                    # Nobody would hear the outcome. Starting nothing leaves the gateway free to try again.
                    _report_abandoned()
                    return None
                # Asked last, when only the launcher is left to create, and before the program exists to see it.
                if not _start_granted(order.start_url, order.start_token, worker.stopping):
                    return HTTPStatus.CONFLICT, _refusal("start_not_granted")
                process = subprocess.Popen(
                    worker.child_command(os.path.join(root, "config.json"), os.path.join(bundle, *order.entrypoint.split("/"))),
                    cwd=bundle,
                    env=_child_environment(worker.environ, root),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    start_new_session=True,
                    **({} if identity is None else {"user": identity[0], "group": identity[1], "extra_groups": []}),
                )
            except Exception:
                # Whatever failed, no launcher process exists: the program did not start.
                return HTTPStatus.INTERNAL_SERVER_ERROR, _refusal("launch_failed")
            outcome = self._supervise(process, order.timeout)
            return None if outcome is None else (HTTPStatus.OK, outcome)
        finally:
            # Before answering: once the gateway has the outcome nothing of the run is left on disk.
            if root is not None:
                _remove_tree(root)

    def _supervise(self, process: subprocess.Popen[bytes], timeout: int) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        exited = threading.Event()
        stopped = abandoned = False
        try:
            stdout = _Capture(process.stdout, _STDOUT_LIMIT)
            stderr = _Capture(process.stderr, _STDERR_LIMIT)
            threading.Thread(target=_watch, args=(process, exited), daemon=True).start()
            while not exited.wait(min(_POLL_SECONDS, max(deadline - time.monotonic(), 0.0))):
                if time.monotonic() >= deadline or self.server.stopping.is_set():
                    stopped = True
                    break
                if self._caller_gone():
                    abandoned = True
                    break
        finally:
            # At the deadline, or when the platform is taking the instance away,
            # this ends the program. After a normal exit it ends whatever the
            # program left in its group, as removing its container would.
            _kill_group(process.pid)
        exited.wait(_REAP_SECONDS)
        code = process.poll()
        if abandoned:
            _report_abandoned()
            return None
        # A process the program moved out of its group can hold the pipes open.
        readers_until = time.monotonic() + _READER_WAIT_SECONDS
        for capture in (stdout, stderr):
            capture.join(max(readers_until - time.monotonic(), 0.0))
        if code is None:
            code = -signal.SIGKILL  # still not reaped after the kill: nothing better is knowable
        return {
            "started": True,
            "exit_code": code if code >= 0 else 128 - code,
            # True for a program the worker stopped part-way, at the deadline or
            # for the platform: what it did is unknown. One that exited by
            # itself in that moment was not stopped.
            "timed_out": stopped and code == -signal.SIGKILL,
            "stdout": stdout.text(),
            "stderr": stderr.text(),
        }

    def _caller_gone(self) -> bool:
        """True once the caller has closed its connection: nobody is left to receive the outcome."""
        connection = self.connection
        try:
            connection.settimeout(0)
            try:
                return connection.recv(1, socket.MSG_PEEK) == b""
            finally:
                connection.settimeout(self.timeout)
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        status = HTTPStatus(status)
        payload = json.dumps(body, separators=(",", ":")).encode("ascii")
        head = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Cache-Control: no-store\r\n"
            f"X-Deepintshield-Managed-Worker: {_PROTOCOL}\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("ascii")
        self.close_connection = True
        try:
            self.wfile.write(head if self.command == "HEAD" else head + payload)
            self._linger()
        except OSError:
            pass  # the caller went away; there is nobody to tell

    def _linger(self) -> None:
        """Half-close, then wait briefly for the caller to finish reading.

        Closing while request bytes are still unread, or exiting right after the
        write, can destroy an answer that is still in transit.
        """
        connection = self.connection
        connection.shutdown(socket.SHUT_WR)
        until = time.monotonic() + _LINGER_SECONDS
        unread = _LINGER_BYTES
        while unread > 0:
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            connection.settimeout(remaining)
            chunk = connection.recv(min(65536, unread))
            if not chunk:
                return
            unread -= len(chunk)


def create_server(
    environ: Mapping[str, str],
    *,
    exit: Callable[[int], Any] = os._exit,
    child_command: Callable[[str, str], list[str]] | None = None,
    address: tuple[str, int] | None = None,
) -> _Worker:
    """Validate the settings and bind the listener; the caller runs ``serve_forever()``.

    ``exit`` ends the process after the single run, ``child_command`` builds the
    program's command line and ``address`` replaces 0.0.0.0:PORT. The defaults
    are the real behaviour; tests replace them to keep a worker in-process.
    """
    port, image, token = _settings(environ)
    return _Worker(address or ("0.0.0.0", port), environ, image, token, exit, child_command or _child_command)


def serve(
    environ: Mapping[str, str],
    *,
    exit: Callable[[int], Any] = os._exit,
    child_command: Callable[[str, str], list[str]] | None = None,
) -> int:
    """Serve until the single run has been answered and ``exit`` ends the process.

    Returns only when the worker could not start, or after a replaced ``exit``
    once the server has been shut down.
    """
    _deny_inspection()
    try:
        server = create_server(environ, exit=exit, child_command=child_command)
    except _InvalidSetting as error:
        print(f"managed worker: {error.name} is missing or invalid", file=sys.stderr)
        return 2
    except OSError:
        print(f"managed worker: cannot listen on the port given by {_PORT_SETTING}", file=sys.stderr)
        return 2
    for name in ("SIGTERM", "SIGINT"):
        try:
            # Process 1 of a container ignores these unless it handles them, which
            # would hold every shutdown for the platform's full grace period.
            signal.signal(getattr(signal, name), lambda *_: server.stop())
        except (AttributeError, OSError, ValueError):
            pass  # not the main thread, or no such signal on this platform
    with server:
        server.serve_forever()
    return 0


def main() -> int:
    return serve(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
