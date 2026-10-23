from __future__ import annotations

import concurrent.futures
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import socketserver
import ssl
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from deepintshield.agentic import managed_remote

if not hasattr(os, "killpg"):
    pytest.skip("the managed worker is a POSIX container process", allow_module_level=True)

IMAGE_SETTING = "DEEPINTSHIELD_MANAGED_WORKER_IMAGE"
AUTH_SETTING = "DEEPINTSHIELD_MANAGED_WORKER_AUTH"
TOKEN_SETTING = "DEEPINTSHIELD_MANAGED_WORKER_TOKEN"
IMAGE = "registry.example/deepintshield/managed-worker@sha256:" + "5a" * 32
# Every kind of character the contract allows in a token.
TOKEN = "Fixture.worker_token~+/=-0123456789abcdefghij"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
GATEWAY_PREFLIGHT = (
    Path(__file__).resolve().parents[2]
    / "deepintshield_server/transports/deepintshield-http/handlers/agenticnew_managed_execution_preflight.go"
)
# The gateway's Docker probe (managedRuntimeInventoryScript), which /inventory must reproduce.
PROBE = """import importlib.metadata, json, sys
packages = []
for distribution in importlib.metadata.distributions():
    packages.append({"name": distribution.metadata["Name"], "version": distribution.version})
    if len(packages) > 1024:
        raise SystemExit(1)
print(json.dumps({"python_version": ".".join(map(str, sys.version_info[:3])), "packages": packages}, separators=(",", ":")))
"""
# The shape the gateway sends as "config": what the Docker runner writes to config.json.
CONFIG = {
    "environment": {
        "DEEPINTSHIELD_BASE_URL": "https://gateway.example",
        "DEEPINTSHIELD_VIRTUAL_KEY": "sk-ds-fixture-virtual-key",
        "DEEPINTSHIELD_AGENT_NAME": "fixture-agent",
        "DEEPINTSHIELD_REQUESTER": "",
        "DEEPINTSHIELD_RUNTIME_KIND": "key",
    },
    "runtime_private_key": "-----BEGIN PRIVATE KEY-----\nfixture-private-key-material\n-----END PRIVATE KEY-----\n",
    "requirements": [],
    "approval_wait_timeout_seconds": 230,
}
START_PATH = "/api/agentic-new/managed-execution/start"
# Every visible ASCII character: the worker hands the token back and never reads it.
START_TOKEN = "".join(map(chr, range(0x21, 0x7F)))
# What a run asks the gateway before it starts its program. The ``granting``
# fixture points the URL at a stand-in that says yes; until then it leads nowhere.
START = {"url": "https://gateway.invalid" + START_PATH, "token": START_TOKEN}
ABANDONED = (
    "managed worker: run abandoned: the caller closed its connection "
    "(a proxy that half-closes after sending the request looks the same)\n"
)
ANSWER_HEADERS = {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
    "X-Deepintshield-Managed-Worker": "2",
    "Connection": "close",
}
# The interpreter running these tests may need its loader path to start at all;
# the worker rightly passes none of it on, so the stand-in launcher restores it.
LOADER = [f"{name}={os.environ[name]}" for name in ("LD_LIBRARY_PATH",) if os.environ.get(name)]


class Reply(NamedTuple):
    status: int
    body: dict | None


def refused(code):
    return {"started": False, "code": code}


def outcome(exit_code=0, stdout="", stderr="", timed_out=False):
    return {"started": True, "exit_code": exit_code, "timed_out": timed_out, "stdout": stdout, "stderr": stderr}


def order(source="print('ran')\n", **changes):
    return {
        "image": IMAGE, "entrypoint": "main.py", "files": {"main.py": source}, "config": CONFIG, "timeout_seconds": 60,
        "start": dict(START), **changes,
    }


def direct(config_path, entrypoint_path):
    """Stand-in for the launcher: run the bundle's entry script itself, with the config path as its argument."""
    command = [sys.executable, "-E", "-s", "-B", entrypoint_path, config_path]
    return ["/usr/bin/env", *LOADER, *command] if LOADER else command


def exchange(port, request, *, timeout=30.0, body_expected=True):
    """Send raw bytes on a fresh connection and check the fixed shape every answer has."""
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as connection:
        connection.sendall(request)
        return answer_on(connection, body_expected=body_expected)


def answer_on(connection, *, body_expected=True):
    """Read the answer on a connection to its end and check the fixed shape every answer has."""
    received = b""
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break  # the worker closes its side after every answer
        received += chunk
    head, separator, payload = received.partition(b"\r\n\r\n")
    assert separator, received[:200]
    status_line, *lines = head.decode("ascii").split("\r\n")
    version, status, _ = status_line.split(" ", 2)
    headers = dict(line.split(": ", 1) for line in lines)
    assert version == "HTTP/1.1"
    length = int(headers.pop("Content-Length"))
    assert headers == ANSWER_HEADERS
    if not body_expected:
        assert payload == b""
        return Reply(int(status), None)
    assert length == len(payload)
    body = json.loads(payload)
    assert isinstance(body, dict)
    return Reply(int(status), body)


def call(port, method="GET", path="/healthz", body=None, *, token=None, headers=(), timeout=30.0):
    """One request, written byte for byte so a test controls the framing."""
    if isinstance(body, dict):
        body = json.dumps(body).encode("ascii")
    lines = [f"{method} {path} HTTP/1.1", "Host: worker.test"]
    if token is not None:
        lines.append(f"Authorization: Bearer {token}")
    if body is not None:
        lines += ["Content-Type: application/json", f"Content-Length: {len(body)}"]
    lines += list(headers)
    request = "\r\n".join(lines).encode("latin-1") + b"\r\n\r\n" + (body or b"")
    return exchange(port, request, timeout=timeout, body_expected=method != "HEAD")


def wait_for(predicate, seconds=10.0):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def gone(pid):
    """True once the process no longer runs; a zombie nobody reaped has stopped running too."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            return stream.read().rpartition(b") ")[2][:1] == b"Z"
    except OSError:
        return False


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Beacon:
    """A loopback listener a test program reports to.

    The test learns that the program is running, and its PID, and decides when
    it may continue. A socket works whichever user the program runs as.
    """

    def __init__(self):
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.settimeout(30)
        self._connection = None
        port = self._listener.getsockname()[1]
        self.prelude = (
            "import os, socket\n"
            f"beacon = socket.create_connection(('127.0.0.1', {port}), timeout=120)\n"
            "beacon.sendall(b'%d\\n' % os.getpid())\n"
        )

    def started(self):
        self._connection, _ = self._listener.accept()
        self._connection.settimeout(30)
        line = b""
        while not line.endswith(b"\n"):
            chunk = self._connection.recv(64)
            assert chunk, "the program left before reporting"
            line += chunk
        return int(line)

    def release(self):
        self._connection.sendall(b"!")

    def close(self):
        for endpoint in (self._connection, self._listener):
            if endpoint is not None:
                endpoint.close()


class Asked(NamedTuple):
    """One request as the stand-in gateway received it; header names in lower case, sorted."""

    method: str
    target: str
    headers: list
    body: bytes


class Gateway:
    """A loopback stand-in for the gateway's start route.

    It records every request and answers with ``status``, ``headers`` and
    ``body``, which are the grant until a test sets something else. ``respond``
    replaces the answer altogether; ``hold`` is such a replacement, which
    answers nothing for as long as the test lasts.
    """

    def __init__(self, *, certificate=None, host="127.0.0.1"):
        gateway = self
        self.asked = []
        self.status, self.headers, self.body = 200, (("Content-Type", "application/json"),), b'{"granted":true}'
        self.respond = self.answer
        self._over = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *arguments):
                pass

            def receive(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                headers = sorted((name.lower(), value) for name, value in self.headers.items())
                gateway.asked.append(Asked(self.command, self.path, headers, body))
                self.close_connection = True
                # A worker that stopped listening is its own business, not an error here.
                with contextlib.suppress(OSError):
                    gateway.respond(self)

            do_GET = do_POST = receive

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

            def server_bind(self):
                socketserver.TCPServer.server_bind(self)  # without looking up this host's own name

        self._server = Server((host, 0), Handler)
        if certificate is not None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(*certificate)
            self._server.socket = context.wrap_socket(self._server.socket, server_side=True)
        self.port = self._server.server_address[1]
        self.authority = f"[{host}]:{self.port}" if ":" in host else f"{host}:{self.port}"
        self.url = f"{'http' if certificate is None else 'https'}://{self.authority}{START_PATH}"
        self.start = {"url": self.url, "token": START_TOKEN}
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    def answer(self, handler):
        handler.send_response(self.status)
        for name, value in (("Content-Length", str(len(self.body))), *self.headers):
            handler.send_header(name, value)
        handler.end_headers()
        handler.wfile.write(self.body)

    def hold(self, handler):
        self._over.wait(60)

    def close(self):
        self._over.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5)


@pytest.fixture(autouse=True)
def stay_in_process(monkeypatch):
    # These tests keep the worker inside the pytest process: never make that
    # process non-dumpable, and never switch user if the suite runs as root.
    monkeypatch.setattr(managed_remote, "_deny_inspection", lambda: None)
    monkeypatch.setattr(managed_remote, "_program_identity", lambda: None)
    # The stand-in gateways are on loopback: a proxy that the machine running
    # the tests is configured with must never be asked for them.
    monkeypatch.setenv("no_proxy", "*")


@pytest.fixture(scope="module", autouse=True)
def granting():
    """The gateway of every run that does not bring its own: it grants each start it is asked for."""
    standin = Gateway()
    START["url"] = standin.url
    yield standin
    standin.close()


@pytest.fixture
def gateway():
    """Makes stand-in gateways for one test, each to answer as that test sets it."""
    made = []

    def gateway(**options):
        made.append(Gateway(**options))
        return made[-1]

    yield gateway
    for standin in made:
        standin.close()


@pytest.fixture
def staged(monkeypatch):
    """The staging directories the worker creates during one test."""
    roots = []
    real = managed_remote._stage

    def stage(order, owner):
        roots.append(real(order, owner))
        return roots[-1]

    monkeypatch.setattr(managed_remote, "_stage", stage)
    return roots


@pytest.fixture
def start():
    running = []

    def start(*, auth="platform", child_command=direct, **environ):
        settings = {"PATH": os.environ.get("PATH", os.defpath), IMAGE_SETTING: IMAGE, AUTH_SETTING: auth}
        if auth == "token":
            settings[TOKEN_SETTING] = TOKEN
        settings.update(environ)
        exits, spawned = [], []

        def command(config_path, entrypoint_path):
            spawned.append((config_path, entrypoint_path))
            return child_command(config_path, entrypoint_path)

        server = managed_remote.create_server(settings, exit=exits.append, child_command=command, address=("127.0.0.1", 0))
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        running.append((server, thread))
        return SimpleNamespace(server=server, port=server.server_address[1], exits=exits, spawned=spawned)

    yield start
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def beacon():
    created = Beacon()
    yield created
    created.close()


@pytest.fixture
def background():
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        yield pool.submit


def probe_output():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(PROBE, "<gateway-probe>", "exec"), {"__name__": "gateway_probe"})
    return json.loads(stream.getvalue())


# --- /healthz and /inventory -------------------------------------------------


def test_healthz_is_never_authenticated_and_runs_nothing(start):
    worker = start(auth="token")
    for headers in ((), ("Authorization: Bearer not-the-token",), ("Authorization: garbage",)):
        assert call(worker.port, "GET", "/healthz", headers=headers) == (200, {"ok": True})
    assert call(worker.port, "GET", "/healthz?probe=startup") == (200, {"ok": True})
    assert worker.spawned == [] and worker.exits == []
    assert not worker.server._claimed.locked()


def test_inventory_is_exactly_what_the_gateway_probe_prints(start):
    worker = start()
    expected = probe_output()
    assert expected["python_version"] == ".".join(map(str, sys.version_info[:3]))
    assert all(package.keys() == {"name", "version"} for package in expected["packages"])
    # Serving it any number of times does not use up the worker.
    for _ in range(3):
        assert call(worker.port, "GET", "/inventory") == (200, {"image": IMAGE, "inventory": expected})
    assert worker.exits == []
    assert call(worker.port, "POST", "/run", order()) == (200, outcome(stdout="ran\n"))


def test_probe_copy_is_the_gateway_source():
    if not GATEWAY_PREFLIGHT.is_file():
        pytest.skip("the gateway source is not part of this checkout")
    script = re.search(r"const managedRuntimeInventoryScript = `(.*?)`", GATEWAY_PREFLIGHT.read_text(encoding="utf-8"), re.S)
    if script is None:
        pytest.skip("the gateway no longer embeds its probe in this file")
    assert script.group(1) == PROBE


@pytest.mark.parametrize(("installed", "status"), [(1024, 200), (1025, 500), (None, 500)])
def test_inventory_fails_where_the_probe_fails(start, monkeypatch, installed, status):
    def distributions():
        if installed is None:
            raise OSError("unreadable metadata")
        return [SimpleNamespace(metadata={"Name": f"demo-{index}"}, version="1.0") for index in range(installed)]

    monkeypatch.setattr(managed_remote.importlib.metadata, "distributions", distributions)
    reply = call(start().port, "GET", "/inventory")
    assert reply.status == status
    if status == 200:
        assert len(reply.body["inventory"]["packages"]) == 1024
        assert reply.body["inventory"]["packages"][0] == {"name": "demo-0", "version": "1.0"}
    else:
        assert reply.body == refused("inventory_unavailable")


# --- /run: the program and its outcome ---------------------------------------

STAGING_PROGRAM = """
import json, os, stat, sys
from pkg.util import VALUE

config_path = sys.argv[1]
root = os.path.dirname(config_path)
bundle = os.path.join(root, "bundle")
paths = {
    "root": root, "bundle": bundle, "pkg": os.path.join(bundle, "pkg"), "pkg/deep": os.path.join(bundle, "pkg", "deep"),
    "main.py": os.path.join(bundle, "main.py"), "pkg/util.py": os.path.join(bundle, "pkg", "util.py"),
    "pkg/deep/data.py": os.path.join(bundle, "pkg", "deep", "data.py"), "config.json": config_path,
    "home": os.environ["HOME"], "tmp": os.environ["TMPDIR"],
}
with open(config_path, encoding="utf-8") as stream:
    config = json.load(stream)
with open(paths["pkg/deep/data.py"], encoding="utf-8") as stream:
    data = stream.read()
tree = sorted(os.path.join(current, name) for current, folders, files in os.walk(root) for name in folders + files)
print(json.dumps({
    "value": VALUE, "data": data, "entry": sys.argv[0], "cwd": os.getcwd(), "root": root, "tree": tree,
    "modes": {name: stat.S_IMODE(os.stat(path).st_mode) for name, path in paths.items()},
    "config": config, "home": paths["home"], "tmp": paths["tmp"],
    "stdin_is_empty": os.read(0, 1) == b"", "session_leader": os.getsid(0) == os.getpid(),
}))
sys.stderr.write("diagnostic line\\n")
raise SystemExit(7)
"""
STAGING_FILES = {"main.py": STAGING_PROGRAM, "pkg/util.py": "VALUE = 42\n", "pkg/deep/data.py": "TEXT = 'r\u00e9sum\u00e9 \u7ed3\u679c'\n"}


def test_run_stages_the_bundle_runs_the_program_and_leaves_nothing_behind(start):
    worker = start()
    previous = os.umask(0o077)  # staging modes are exact whatever the worker's umask
    try:
        reply = call(worker.port, "POST", "/run", order(files=STAGING_FILES))
    finally:
        os.umask(previous)
    assert reply.status == 200
    assert reply.body == outcome(exit_code=7, stdout=reply.body["stdout"], stderr="diagnostic line\n")
    report = json.loads(reply.body["stdout"])
    root = report["root"]
    assert os.path.basename(root).startswith("deepintshield-managed-")
    # The launcher refuses a bundle reached through a symlinked component.
    assert os.path.realpath(root) == root
    assert report["value"] == 42 and report["data"] == STAGING_FILES["pkg/deep/data.py"]
    assert report["entry"] == os.path.join(root, "bundle", "main.py")
    assert report["cwd"] == os.path.join(root, "bundle")
    assert report["tree"] == sorted(os.path.join(root, *name.split("/")) for name in (
        "bundle", "bundle/main.py", "bundle/pkg", "bundle/pkg/deep", "bundle/pkg/deep/data.py", "bundle/pkg/util.py",
        "config.json", "home", "tmp",
    ))
    assert report["modes"] == {
        "root": 0o700, "bundle": 0o755, "pkg": 0o755, "pkg/deep": 0o755,
        "main.py": 0o444, "pkg/util.py": 0o444, "pkg/deep/data.py": 0o444,
        "config.json": 0o400, "home": 0o700, "tmp": 0o700,
    }
    assert report["config"] == CONFIG
    assert (report["home"], report["tmp"]) == (os.path.join(root, "home"), os.path.join(root, "tmp"))
    assert report["stdin_is_empty"] and report["session_leader"]
    assert worker.spawned == [(os.path.join(root, "config.json"), os.path.join(root, "bundle", "main.py"))]
    # Gone by the time the gateway has its answer, and the process then ends itself.
    assert not os.path.exists(root)
    assert wait_for(lambda: worker.exits == [0])


def test_staging_is_removed_even_when_the_program_closed_its_directories(start):
    program = (
        "import os, sys\n"
        "root, home, scratch = os.path.dirname(sys.argv[1]), os.environ['HOME'], os.environ['TMPDIR']\n"
        "os.makedirs(os.path.join(home, 'kept', 'deep'))\n"
        "open(os.path.join(home, 'kept', 'deep', 'file'), 'w').close()\n"
        # A link is removed as a link: nothing it points at may be touched.
        "os.symlink(os.path.dirname(root), os.path.join(scratch, 'elsewhere'))\n"
        "print(root, flush=True)\n"
        "for path in (os.path.join(home, 'kept', 'deep'), os.path.join(home, 'kept'), home, root):\n"
        "    os.chmod(path, 0)\n"
    )
    worker = start()
    reply = call(worker.port, "POST", "/run", order(program))
    assert reply == (200, outcome(stdout=reply.body["stdout"]))
    root = reply.body["stdout"].strip()
    assert os.path.basename(root).startswith("deepintshield-managed-")
    assert not os.path.lexists(root)
    assert os.path.isdir(os.path.dirname(root)) and os.access(os.path.dirname(root), os.R_OK | os.W_OK | os.X_OK)
    assert wait_for(lambda: worker.exits == [0])


@contextlib.contextmanager
def nested(root, depth):
    """``depth`` directories inside one another under ``root``, with a file in the deepest.

    Each is entered by descriptor, because no path is long enough to name the
    deepest. Whatever is left of them afterwards is removed the same way.
    """
    descriptor = os.open(root, os.O_RDONLY)
    try:
        for _ in range(depth):
            os.mkdir("d", dir_fd=descriptor)
            inner = os.open("d", os.O_RDONLY, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = inner
        os.close(os.open("deepest", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=descriptor))
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            descriptor, below = os.open(root, os.O_RDONLY), 0
            try:
                with contextlib.suppress(OSError):
                    while True:
                        inner = os.open("d", os.O_RDONLY, dir_fd=descriptor)
                        os.close(descriptor)
                        descriptor, below = inner, below + 1
                with contextlib.suppress(OSError):
                    os.unlink("deepest", dir_fd=descriptor)
                for _ in range(below):
                    above = os.open("..", os.O_RDONLY, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = above
                    os.rmdir("d", dir_fd=descriptor)
            finally:
                os.close(descriptor)


def test_tree_deeper_than_python_can_recurse_or_hold_open_is_removed(tmp_path):
    import resource  # not on every platform this module can be imported on

    root = tmp_path / "deepintshield-managed-fixture"
    (root / "home").mkdir(parents=True)
    (root / "config.json").write_text("{}")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # Room for a few dozen descriptors more than are open now: far fewer than the tree has levels.
    lowest = os.open(os.devnull, os.O_RDONLY)
    os.close(lowest)
    few = lowest + 64 if soft == resource.RLIM_INFINITY else min(soft, lowest + 64)
    with nested(str(root / "home"), max(sys.getrecursionlimit(), few) + 50):
        # shutil.rmtree fails on such a tree: older Pythons raise RecursionError, and
        # newer ones give up without a word once a descriptor is open for every level.
        resource.setrlimit(resource.RLIMIT_NOFILE, (few, hard))
        try:
            assert managed_remote._remove_tree(str(root)) is None
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        assert not os.path.lexists(root)


@pytest.mark.parametrize("error", [RecursionError, RuntimeError, OSError])
def test_answer_survives_a_removal_that_raises(start, staged, monkeypatch, capfd, error):
    def rmtree(path, *arguments, **options):
        raise error("removal failed")

    monkeypatch.setattr(managed_remote.shutil, "rmtree", rmtree)
    worker = start()
    assert call(worker.port, "POST", "/run", order()) == (200, outcome(stdout="ran\n"))
    # The staging directory, with the run's credentials in it, is gone all the same.
    assert len(staged) == 1 and not os.path.lexists(staged[0])
    assert wait_for(lambda: worker.exits == [0])
    assert capfd.readouterr() == ("", "")


def test_removal_never_raises(tmp_path, monkeypatch):
    def broken(*arguments, **options):
        raise RuntimeError("removal failed")

    monkeypatch.setattr(managed_remote.shutil, "rmtree", broken)
    monkeypatch.setattr(managed_remote, "_remove_remaining", broken)
    (tmp_path / "config.json").write_text("{}")
    assert managed_remote._remove_tree(str(tmp_path)) is None
    assert managed_remote._remove_tree(str(tmp_path / "absent")) is None


def test_removal_ends_where_a_directory_was_moved_from_under_it(tmp_path, monkeypatch):
    root, elsewhere = tmp_path / "staging", tmp_path / "elsewhere"
    (root / "a" / "b").mkdir(parents=True)
    (root / "z").mkdir()
    (elsewhere / "z").mkdir(parents=True)
    (elsewhere / "z" / "kept").write_text("another run's")
    inside = os.stat(root / "a" / "b")
    real = managed_remote._empty

    def empty(directory):
        if os.path.samestat(os.fstat(directory), inside):
            # What a process the program left behind can do while the walk is in here.
            os.rename(root / "a", elsewhere / "a")
        # "a" is entered before "z", whatever order the file system lists them in.
        return sorted(real(directory), reverse=True)

    monkeypatch.setattr(managed_remote, "_empty", empty)
    monkeypatch.setattr(managed_remote.shutil, "rmtree", lambda *arguments, **options: None)
    assert managed_remote._remove_tree(str(root)) is None
    # The walk came up into a directory it never entered, and touched nothing there.
    assert (elsewhere / "z" / "kept").read_text() == "another run's"
    assert sorted(os.listdir(elsewhere)) == ["a", "z"] and os.listdir(root) == ["z"]


def test_default_command_is_the_reviewed_launcher():
    assert managed_remote._child_command("/staging/config.json", "/staging/bundle/main.py") == [
        sys.executable, "-I", "-m", "deepintshield.agentic.managed_runner", "--config", "/staging/config.json", "/staging/bundle/main.py",
    ]


@pytest.mark.parametrize(("program", "exit_code"), [
    ("raise SystemExit(0)\n", 0),
    ("raise SystemExit(3)\n", 3),
    ("import os\nos._exit(255)\n", 255),
    ("raise RuntimeError('unhandled')\n", 1),
    ("import os, signal, time\nos.kill(os.getpid(), signal.SIGTERM)\ntime.sleep(60)\n", 128 + signal.SIGTERM),
    ("import os, signal, time\nos.kill(os.getpid(), signal.SIGUSR1)\ntime.sleep(60)\n", 128 + signal.SIGUSR1),
    ("import os, signal, time\nos.kill(os.getpid(), signal.SIGKILL)\ntime.sleep(60)\n", 128 + signal.SIGKILL),
])
def test_exit_code_is_the_status_or_128_plus_the_signal(start, program, exit_code):
    worker = start()
    reply = call(worker.port, "POST", "/run", order(program))
    assert reply.status == 200
    # A program that died from a signal by itself did not time out, SIGKILL included.
    assert (reply.body["started"], reply.body["exit_code"], reply.body["timed_out"]) == (True, exit_code, False)
    assert wait_for(lambda: worker.exits == [0])


def test_output_keeps_the_first_bytes_and_never_blocks_the_program(start):
    program = (
        "import sys\n"
        "sys.stdout.buffer.write(b''.join(b'%07d\\n' % line for line in range(131072)))\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b''.join(b'e%06d\\n' % line for line in range(131072)))\n"
        "sys.stderr.buffer.flush()\n"
    )
    # One mebibyte on each pipe: an undrained pipe would block the program until the deadline.
    reply = call(start().port, "POST", "/run", order(program, timeout_seconds=30))
    assert reply == (200, outcome(
        stdout="".join("%07d\n" % line for line in range(131072))[: 48 * 1024],
        stderr="".join("e%06d\n" % line for line in range(131072))[: 16 * 1024],
    ))


def test_output_that_is_not_utf8_is_replaced_not_dropped(start):
    program = "import os\nos.write(1, b'ok \\xff\\xfe\\x00 end\\n')\nos.write(2, b'caf\\xc3')\n"
    assert call(start().port, "POST", "/run", order(program)) == (200, outcome(stdout="ok \ufffd\ufffd\x00 end\n", stderr="caf\ufffd"))


@pytest.mark.parametrize(("emitted", "stdout", "stderr"), [
    # An undecodable byte becomes a replacement character three bytes long.
    (r"b'\xff' * (1 << 20)", "\ufffd" * 16384, "\ufffd" * 5461),
    # JSON writes a control character as six bytes.
    (r"b'\x00' * (1 << 20)", "\x00" * 24576, "\x00" * 8192),
    # A character cut in half by the byte limit is left out, not replaced.
    (r"b'a' + b'\xc3\xa9' * 40000", "a" + "\u00e9" * 24575, "a" + "\u00e9" * 8191),
    (r"'\U0001f600'.encode() * 40000", "\U0001f600" * 12288, "\U0001f600" * 4096),
])
def test_output_never_outgrows_what_the_gateway_accepts(start, emitted, stdout, stderr):
    program = f"import sys\ndata = {emitted}\nsys.stdout.buffer.write(data)\nsys.stdout.buffer.flush()\nsys.stderr.buffer.write(data)\n"
    reply = call(start().port, "POST", "/run", order(program))
    assert reply == (200, outcome(stdout=stdout, stderr=stderr))
    assert len(stdout.encode()) <= 48 * 1024 and len(stderr.encode()) <= 16 * 1024
    # The gateway reads at most 256 KiB of an answer; the worst case stays well inside it.
    assert len(json.dumps(reply.body, separators=(",", ":"))) <= 200 * 1024


# --- /run: the deadline and what a program leaves behind ------------------------

HELPER = "helper = subprocess.Popen([sys.executable, '-E', '-s', '-c', 'import time; time.sleep(120)']%s)\n"
REPORT = "print(json.dumps({'program': os.getpid(), 'helper': helper.pid}), flush=True)\n"


def test_deadline_kills_the_program_and_its_whole_group(start):
    worker = start()
    program = "import json, os, subprocess, sys, time\n" + HELPER % "" + REPORT + "time.sleep(120)\n"
    began = time.monotonic()
    reply = call(worker.port, "POST", "/run", order(program, timeout_seconds=1))
    elapsed = time.monotonic() - began
    assert reply == (200, outcome(exit_code=128 + signal.SIGKILL, stdout=reply.body["stdout"], timed_out=True))
    assert 1.0 <= elapsed < 20
    pids = json.loads(reply.body["stdout"])
    assert wait_for(lambda: gone(pids["program"]) and gone(pids["helper"]))
    assert wait_for(lambda: worker.exits == [0])


def test_exit_ends_what_the_program_left_in_its_group(start):
    program = "import json, os, subprocess, sys\n" + HELPER % "" + REPORT
    reply = call(start().port, "POST", "/run", order(program))
    assert reply == (200, outcome(stdout=reply.body["stdout"]))
    # As when the Docker runner removes the container once its main process has exited.
    assert wait_for(lambda: gone(json.loads(reply.body["stdout"])["helper"]))


@pytest.mark.parametrize("program_sleeps", [False, True])
def test_answer_never_waits_long_for_a_process_holding_the_pipes(start, monkeypatch, program_sleeps):
    # A process that moved to its own session survives the group kill and keeps
    # stdout and stderr open; only the instance ending removes it.
    assert managed_remote._READER_WAIT_SECONDS == 2.0
    monkeypatch.setattr(managed_remote, "_READER_WAIT_SECONDS", 0.4)
    program = "import json, os, subprocess, sys, time\n" + HELPER % ", start_new_session=True" + REPORT
    if program_sleeps:
        program += "time.sleep(120)\n"
    began = time.monotonic()
    reply = call(start().port, "POST", "/run", order(program, timeout_seconds=1))
    elapsed = time.monotonic() - began
    helper = json.loads(reply.body["stdout"])["helper"]
    try:
        expected = outcome(exit_code=128 + signal.SIGKILL, timed_out=True) if program_sleeps else outcome()
        assert reply == (200, {**expected, "stdout": reply.body["stdout"]})
        assert 0.4 <= elapsed < 20
        assert not gone(helper)
    finally:
        os.kill(helper, signal.SIGKILL)


@pytest.mark.parametrize("timed_out", [False, True])
def test_supervision_without_waitid(start, monkeypatch, timed_out):
    # macOS before Python 3.13 has no os.waitid: the exit is then reaped as it is noticed.
    monkeypatch.delattr(managed_remote.os, "waitid", raising=False)
    program = "import time\nprint('ran', flush=True)\n" + ("time.sleep(120)\n" if timed_out else "raise SystemExit(5)\n")
    reply = call(start().port, "POST", "/run", order(program, timeout_seconds=1 if timed_out else 60))
    assert reply == (200, outcome(exit_code=137 if timed_out else 5, stdout="ran\n", timed_out=timed_out))


# --- /run: single use -------------------------------------------------------------


def test_probes_answer_and_other_runs_are_refused_while_a_run_is_in_progress(start, beacon, background):
    worker = start()
    first = background(call, worker.port, "POST", "/run", order(beacon.prelude + "beacon.recv(1)\nprint('released')\n"))
    beacon.started()
    assert call(worker.port, "GET", "/healthz") == (200, {"ok": True})
    assert call(worker.port, "GET", "/inventory").status == 200
    assert call(worker.port, "POST", "/run", order()) == (409, refused("worker_busy"))
    # A busy answer while the run is still going does not end the process.
    assert worker.exits == [] and len(worker.spawned) == 1
    beacon.release()
    assert first.result(60) == (200, outcome(stdout="released\n"))
    assert wait_for(lambda: worker.exits == [0])
    # Should the process outlive its exit, every later run is refused and ends it again.
    assert call(worker.port, "POST", "/run", order()) == (409, refused("worker_busy"))
    assert wait_for(lambda: worker.exits == [0, 0])
    assert len(worker.spawned) == 1


def test_simultaneous_runs_accept_exactly_one(start, background):
    worker = start()
    contenders = 8
    barrier = threading.Barrier(contenders)

    def contend(index):
        barrier.wait(30)
        return call(worker.port, "POST", "/run", order(f"import time\ntime.sleep(0.3)\nprint({index})\n"))

    replies = [pending.result(60) for pending in [background(contend, index) for index in range(contenders)]]
    accepted = [index for index, reply in enumerate(replies) if reply.status == 200]
    assert len(accepted) == 1
    assert replies[accepted[0]] == (200, outcome(stdout=f"{accepted[0]}\n"))
    assert [reply for index, reply in enumerate(replies) if index != accepted[0]] == [(409, refused("worker_busy"))] * (contenders - 1)
    assert len(worker.spawned) == 1
    assert wait_for(lambda: bool(worker.exits)) and set(worker.exits) == {0}


@pytest.mark.parametrize("failure", ["missing", "not_executable", "raises"])
def test_launcher_that_cannot_be_created_still_uses_up_the_worker(start, tmp_path, failure):
    blocked = tmp_path / "launcher"
    blocked.write_text("#!/bin/sh\nexit 0\n")
    blocked.chmod(0o644)

    def command(config_path, entrypoint_path):
        if failure == "raises":
            raise RuntimeError("no command")
        return [str(tmp_path / "absent")] if failure == "missing" else [str(blocked)]

    worker = start(child_command=command)
    assert call(worker.port, "POST", "/run", order()) == (500, refused("launch_failed"))
    assert not os.path.exists(os.path.dirname(worker.spawned[0][0]))
    assert wait_for(lambda: worker.exits == [0])
    assert call(worker.port, "POST", "/run", order()) == (409, refused("worker_busy"))
    assert wait_for(lambda: worker.exits == [0, 0])


# --- /run: the gateway grants the start ---------------------------------------------

GRANT = b'{"granted":true}'
# Prints everything a program is given: its arguments, its environment and every staged file.
WITNESS = (
    "import json, os, sys\n"
    "texts = []\n"
    "for folder, _, names in os.walk(os.path.dirname(sys.argv[1])):\n"
    "    for name in names:\n"
    "        with open(os.path.join(folder, name), encoding='utf-8') as stream:\n"
    "            texts.append(stream.read())\n"
    "print(json.dumps({'arguments': sys.argv, 'environment': dict(os.environ), 'files': texts}))\n"
)


def not_started(worker, staged):
    """What a run without a grant leaves: no launcher, no staging directory, and a worker that is used up."""
    assert worker.spawned == []
    assert len(staged) == 1 and not os.path.lexists(staged[0])
    assert wait_for(lambda: worker.exits == [0])
    assert call(worker.port, "POST", "/run", order()) == (409, refused("worker_busy"))
    assert wait_for(lambda: worker.exits == [0, 0])


def test_start_is_asked_of_the_gateway_before_the_launcher_exists(start, gateway, staged):
    standin, worker, at_the_time = gateway(), start(auth="token"), []

    def answer(handler):
        at_the_time.append((sorted(os.listdir(staged[0])), list(worker.spawned)))
        standin.answer(handler)

    standin.respond = answer
    token = "a-start-token-no-program-may-see"
    reply = call(worker.port, "POST", "/run", order(WITNESS, start={"url": standin.url, "token": token}), token=TOKEN)
    assert reply == (200, outcome(stdout=reply.body["stdout"]))
    # One request, and nothing in it but what the protocol names and what any
    # HTTP client adds: the worker's own token, for one, stays with the worker.
    assert standin.asked == [Asked("POST", START_PATH, sorted({
        "host": standin.authority, "authorization": "Bearer " + token, "content-type": "application/json", "content-length": "2",
        "user-agent": "deepintshield-managed-worker", "accept-encoding": "identity", "connection": "close",
    }.items()), b"{}")]
    # Everything was staged by then, and no launcher had been asked for.
    assert at_the_time == [(["bundle", "config.json", "home", "tmp"], [])]
    witnessed = json.loads(reply.body["stdout"])
    assert witnessed["arguments"][1] == os.path.join(staged[0], "config.json") and len(witnessed["files"]) == 2
    for secret in (token, START_PATH, standin.authority):
        assert secret not in reply.body["stdout"]


def test_start_accepts_the_limits_of_the_contract(start, gateway):
    standin = gateway()
    for token in (START_TOKEN, "t" * 16, "T" * 4096):
        assert call(start().port, "POST", "/run", order(start={"url": standin.url, "token": token})) == (200, outcome(stdout="ran\n"))
        assert ("authorization", "Bearer " + token) in standin.asked[-1].headers
    assert len(standin.asked) == 3
    for origin in (
        "https://gateway.example", "https://gateway.example:8443", "HTTPS://Gateway.Example", "https://gateway.example:",
        "https://10.0.0.7", "https://[2001:db8::7]:8443", "https://managed_gateway.internal", "https://xn--gteway-cua.example",
        "https://" + "a" * (2048 - len("https://") - len(START_PATH)),
        "http://localhost", "http://LOCALHOST:8080", "Http://127.0.0.1", "http://127.255.255.254:65535",
        "http://[::1]", "http://[0:0:0:0:0:0:0:1]:8080", "http://[::ffff:127.0.0.1]:8080",
    ):
        accepted = managed_remote._parse_order(json.dumps(order(start={"url": origin + START_PATH, "token": START_TOKEN})).encode())
        assert accepted is not None and (accepted.start_url, accepted.start_token) == (origin + START_PATH, START_TOKEN), origin


@pytest.mark.parametrize("body", [
    GRANT,
    b' {\n  "granted" : true\n}\n',
    b'{"job":"7","granted":true,"expires_in":30}',
    b'{"granted":true}' + b" " * (4096 - len(GRANT)),
])
def test_start_is_granted_by_an_object_whose_granted_is_true(start, gateway, body):
    standin = gateway()
    # What the answer calls itself is not looked at, only what it is.
    standin.body, standin.headers = body, (("Content-Type", "text/plain"),)
    assert call(start().port, "POST", "/run", order(start=standin.start)) == (200, outcome(stdout="ran\n"))
    assert len(standin.asked) == 1


def test_start_is_asked_of_an_ipv6_loopback_address(start, gateway):
    try:
        standin = gateway(host="::1")
    except OSError:
        pytest.skip("this machine has no IPv6 loopback address")
    assert standin.url.startswith("http://[::1]:")
    assert call(start().port, "POST", "/run", order(start=standin.start)) == (200, outcome(stdout="ran\n"))
    assert [(asked.method, asked.target) for asked in standin.asked] == [("POST", START_PATH)]


def refusals():
    yield "granted false", 200, b'{"granted":false}'
    for status in (201, 202, 204, 206, 400, 401, 403, 404, 409, 429, 500, 502, 503):
        # The status decides: these say yes in a body that is never trusted.
        yield f"status {status}", status, b"" if status == 204 else GRANT
    for name, body in {
        "empty": b"",
        "not json": b"granted",
        "true alone": b"true",
        "string": b'"granted"',
        "null": b"null",
        "number": b"1",
        "array": b"[true]",
        "array of a grant": b"[" + GRANT + b"]",
        "empty object": b"{}",
        "granted null": b'{"granted":null}',
        "granted string": b'{"granted":"true"}',
        "granted one": b'{"granted":1}',
        "granted float": b'{"granted":1.0}',
        "granted array": b'{"granted":[true]}',
        "granted object": b'{"granted":{"granted":true}}',
        "another key": b'{"Granted":true}',
        "python literal": b'{"granted":True}',
        "cut short": GRANT[:-1],
        "two objects": GRANT + GRANT,
        "byte order mark": b"\xef\xbb\xbf" + GRANT,
        "utf-16": GRANT.decode().encode("utf-16"),
        "longer than is read": b'{"granted":true,"padding":"' + b"x" * 4096 + b'"}',
        "one byte longer than is read": GRANT + b" " * (4097 - len(GRANT)),
    }.items():
        yield f"body {name}", 200, body


@pytest.mark.parametrize(("status", "body"), [pytest.param(status, body, id=name) for name, status, body in refusals()])
def test_program_is_not_started_without_the_plain_yes_of_the_gateway(start, gateway, staged, capfd, status, body):
    standin, worker = gateway(), start()
    standin.status, standin.body = status, body
    assert call(worker.port, "POST", "/run", order(start=standin.start)) == (409, refused("start_not_granted"))
    # Asked once, and never again.
    assert len(standin.asked) == 1
    not_started(worker, staged)
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_from_the_start_route_is_never_followed(start, gateway, staged, capfd, status):
    first, second, worker = gateway(), gateway(), start()
    first.status, first.headers = status, (("Location", second.url),)
    assert call(worker.port, "POST", "/run", order(start=first.start)) == (409, refused("start_not_granted"))
    # The token went to the one URL the gateway named, although the next one would have granted.
    assert len(first.asked) == 1 and second.asked == []
    not_started(worker, staged)
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("answer", ["never", "late"])
def test_start_that_is_not_answered_in_time_is_not_granted(start, gateway, staged, monkeypatch, capfd, answer):
    # Fifteen seconds in all; shortened here so the test does not wait for them.
    assert managed_remote._GRANT_SECONDS == 15.0
    monkeypatch.setattr(managed_remote, "_GRANT_SECONDS", 0.5)
    standin, worker, complete = gateway(), start(), threading.Event()

    def late(handler):
        # A yes that keeps arriving, a byte at a time, and is complete only after the limit.
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(GRANT)))
        handler.end_headers()
        try:
            for index in range(len(GRANT)):
                time.sleep(0.2)
                handler.wfile.write(GRANT[index:index + 1])
        finally:
            complete.set()

    standin.respond = late if answer == "late" else standin.hold
    began = time.monotonic()
    assert call(worker.port, "POST", "/run", order(start=standin.start)) == (409, refused("start_not_granted"))
    assert 0.5 <= time.monotonic() - began < 10
    assert len(standin.asked) == 1
    if answer == "late":
        # The limit is on the whole exchange, not on each wait for the next byte:
        # the worker gave up while the yes was arriving, and nothing starts once it is complete.
        assert not complete.is_set()
        assert complete.wait(30)
        time.sleep(0.2)
    not_started(worker, staged)
    assert capfd.readouterr() == ("", "")


def test_start_that_cannot_reach_the_gateway_is_not_granted(start, staged, capfd):
    worker = start()
    nowhere = {"url": f"http://127.0.0.1:{free_port()}{START_PATH}", "token": START_TOKEN}
    began = time.monotonic()
    assert call(worker.port, "POST", "/run", order(start=nowhere)) == (409, refused("start_not_granted"))
    assert time.monotonic() - began < 10
    not_started(worker, staged)
    # Not even the kind of failure is written: it could name the gateway.
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("gateway_answers", [True, False])
def test_worker_told_to_stop_while_it_asks_starts_nothing(start, gateway, staged, gateway_answers):
    standin, worker = gateway(), start()

    def answer(handler):
        worker.server.stop()
        # Whatever the gateway says now, and however long it takes to say it.
        (standin.answer if gateway_answers else standin.hold)(handler)

    standin.respond = answer
    began = time.monotonic()
    assert call(worker.port, "POST", "/run", order(start=standin.start)) == (409, refused("start_not_granted"))
    assert time.monotonic() - began < 10
    assert worker.spawned == [] and not os.path.lexists(staged[0])
    assert wait_for(lambda: worker.exits == [0])


def test_worker_told_to_stop_before_it_asks_uses_no_start(start, gateway, staged, monkeypatch):
    standin, worker = gateway(), start()
    stage = managed_remote._stage

    def stage_then_stop(*arguments):
        root = stage(*arguments)
        # The platform takes the instance away while the bundle is being written.
        worker.server.stop()
        return root

    monkeypatch.setattr(managed_remote, "_stage", stage_then_stop)
    assert call(worker.port, "POST", "/run", order(start=standin.start)) == (409, refused("start_not_granted"))
    # A start that was asked for is used up. This one the gateway can still
    # withdraw, and so it knows that the program never began.
    assert standin.asked == [] and worker.spawned == []
    assert len(staged) == 1 and not os.path.lexists(staged[0])
    assert wait_for(lambda: worker.exits == [0])


def test_start_goes_through_the_proxy_the_environment_names(start, gateway, monkeypatch):
    proxy, target = gateway(), gateway()
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("http_proxy", f"http://{proxy.authority}")
    assert call(start().port, "POST", "/run", order(start=target.start)) == (200, outcome(stdout="ran\n"))
    # A proxy is asked for the whole URL; the gateway itself was not connected to.
    assert [(asked.method, asked.target, asked.body) for asked in proxy.asked] == [("POST", target.url, b"{}")]
    assert target.asked == []


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    """A throwaway self-signed certificate for 127.0.0.1, where the openssl command can make one."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("needs the openssl command to make a throwaway certificate")
    folder = tmp_path_factory.mktemp("certificate")
    paths = {name: str(folder / name) for name in ("openssl.cnf", "certificate.pem", "key.pem")}
    Path(paths["openssl.cnf"]).write_text(
        "[req]\ndistinguished_name = name\nx509_extensions = extensions\nprompt = no\n"
        "[name]\nCN = managed-worker-test\n"
        "[extensions]\nsubjectAltName = IP:127.0.0.1\nbasicConstraints = critical, CA:TRUE\n"
        "keyUsage = critical, digitalSignature, keyCertSign\nsubjectKeyIdentifier = hash\n",
        encoding="ascii",
    )
    made = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-config", paths["openssl.cnf"],
         "-keyout", paths["key.pem"], "-out", paths["certificate.pem"]],
        capture_output=True, timeout=120, check=False,
    )
    if made.returncode != 0:
        pytest.skip("the openssl command here cannot make the throwaway certificate")
    return paths["certificate.pem"], paths["key.pem"]


def test_start_over_https_trusts_what_the_environment_names(start, gateway, certificate, monkeypatch):
    standin = gateway(certificate=certificate)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", certificate[0])
    assert standin.url.startswith("https://127.0.0.1:")
    assert call(start().port, "POST", "/run", order(start=standin.start)) == (200, outcome(stdout="ran\n"))
    assert [(asked.method, asked.target, asked.body) for asked in standin.asked] == [("POST", START_PATH, b"{}")]


@pytest.mark.parametrize("failure", ["untrusted", "another name"])
def test_start_over_https_needs_a_certificate_that_verifies(start, gateway, certificate, staged, monkeypatch, capfd, failure):
    standin, worker = gateway(certificate=certificate), start()
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    url = standin.url
    if failure == "untrusted":
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    else:
        # Trusted, but issued to the address and not to this name.
        monkeypatch.setenv("SSL_CERT_FILE", certificate[0])
        url = url.replace("127.0.0.1", "localhost")
    assert call(worker.port, "POST", "/run", order(start={"url": url, "token": START_TOKEN})) == (409, refused("start_not_granted"))
    # The token is not sent to a gateway that has not proved who it is.
    assert standin.asked == []
    not_started(worker, staged)
    assert capfd.readouterr() == ("", "")


# --- /run: what is refused before anything is accepted ------------------------------


def invalid_orders():
    def changed(**changes):
        return order(**changes)

    def without(field):
        body = order()
        del body[field]
        return body

    def with_file(name, source="pass\n"):
        return order(files={"main.py": "pass\n", name: source})

    yield "empty body", b""
    yield "truncated json", b'{"image": '
    yield "invalid utf-8", b'{"image": "\xff"}'
    yield "byte order mark", b"\xef\xbb\xbf" + json.dumps(order()).encode()
    yield "not a number", json.dumps(changed(config={"budget": float("nan")})).encode()
    for name, body in (("null", b"null"), ("array", b"[]"), ("string", b'"run"'), ("number", b"42")):
        yield f"top-level {name}", body
    for field in ("image", "entrypoint", "files", "config", "timeout_seconds", "start"):
        yield f"missing {field}", without(field)
    yield "extra field", changed(mode="run")
    yield "image not a string", changed(image=7)
    yield "image null", changed(image=None)
    yield "entrypoint not a string", changed(entrypoint=7)
    yield "entrypoint not in files", changed(entrypoint="other.py")
    yield "entrypoint not python", order(entrypoint="main.txt", files={"main.txt": "pass\n"})
    yield "files not an object", changed(files=[["main.py", "pass\n"]])
    yield "files empty", order(entrypoint="main.py", files={})
    yield "too many files", changed(files={"main.py": "pass\n", **{f"module_{index}.py": "" for index in range(64)}})
    for name, source in (("number", 7), ("null", None), ("array", ["pass"]), ("object", {"source": "pass"})):
        yield f"file source {name}", changed(files={"main.py": source})
    yield "file source lone surrogate", changed(files={"main.py": "\ud800"})
    for name in (
        "/absolute.py", "pkg//double.py", "./here.py", "pkg/./here.py", "../up.py", "pkg/../up.py", "pkg/..", "pkg/",
        "back\\slash.py", "nul\x00byte.py", "", ".", "x" * 510 + ".py", "\ud800.py",
    ):
        yield f"file name {name[:24]!r}", with_file(name)
    yield "file that is also a directory", changed(files={"main.py": "pass\n", "pkg": "", "pkg/util.py": ""})
    for name, config in (("array", []), ("string", "{}"), ("null", None), ("number", 1)):
        yield f"config {name}", changed(config=config)
    for value in (0, -1, 3601, True, 1.5, "30", None, [30]):
        yield f"timeout_seconds {value!r}", changed(timeout_seconds=value)
    for name, value in (("null", None), ("string", START["url"]), ("array", [START["url"], START_TOKEN]), ("number", 1), ("empty", {})):
        yield f"start {name}", changed(start=value)
    yield "start with an extra field", changed(start={**START, "retry": 1})
    yield "start without url", changed(start={"token": START_TOKEN})
    yield "start without token", changed(start={"url": START["url"]})
    for field in ("url", "token"):
        for name, value in (("number", 7), ("null", None), ("array", [START[field]]), ("object", {field: START[field]}), ("boolean", True)):
            yield f"start {field} {name}", changed(start={**START, field: value})
    for name, url in invalid_start_urls():
        yield f"start url {name}", changed(start={**START, "url": url})
    for name, token in (
        ("empty", ""), ("too short", "t" * 15), ("too long", "t" * 4097), ("with a space", "t" * 16 + " " + "t" * 16),
        ("with a tab", "t" * 16 + "\t"), ("with a line end", "t" * 32 + "\n"), ("with a header after it", "t" * 16 + "\r\nX-Injected: 1"),
        ("with a nul", "t" * 16 + "\x00"), ("with a delete", "t" * 16 + "\x7f"), ("not ascii", "\xe9" * 32),
        ("lone surrogate", "t" * 16 + "\ud800"),
    ):
        yield f"start token {name}", changed(start={**START, "token": token})


def invalid_start_urls():
    """URLs that are not exactly an https origin, or an http one on this machine, followed by the start route."""
    https, loopback = "https://gateway.example", "http://127.0.0.1:8080"
    for name, origin in {
        "http to another host": "http://gateway.example",
        "http to a private address": "http://10.0.0.7:8080",
        "http to every interface": "http://0.0.0.0:8080",
        "http to a name that only starts like loopback": "http://127.0.0.1.gateway.example",
        "http to a name that only ends in localhost": "http://gateway.localhost",
        "http to a number that only resolves to loopback": "http://2130706433",
        "http to a short loopback": "http://127.1",
        "http to an unspecified ipv6 address": "http://[::]:8080",
        "http to another ipv6 address": "http://[fe80::1]:8080",
        "http to a mapped address that is not loopback": "http://[::ffff:10.0.0.7]",
        "ftp": "ftp://gateway.example",
        "websocket": "wss://gateway.example",
        "file": "file://",
        "no scheme": "//gateway.example",
        "no scheme at all": "gateway.example",
        "no host": "https://",
        "no host before the port": "https://:443",
        "user": "https://worker@gateway.example",
        "user and password": "https://worker:secret@gateway.example",
        "empty user": "https://@gateway.example",
        "user before loopback": "http://gateway.example@127.0.0.1",
        "port too large": "https://gateway.example:65536",
        "port too long": "https://gateway.example:000443",
        "port not a number": "https://gateway.example:https",
        "negative port": "https://gateway.example:-1",
        "two ports": "https://gateway.example:443:443",
        "brackets around a name": "https://[gateway.example]",
        "brackets around an ipv4 address": "http://[127.0.0.1]",
        "bracket left open": "https://[::1",
        "ipv6 zone": "http://[::1%25lo]",
        "encoded host": "https://gateway%2eexample",
        "not ascii host": "https://g\xe4teway.example",
        "backslash in host": "https://gateway.example\\@127.0.0.1",
        "space in host": "https://gateway .example",
        "leading space": " https://gateway.example",
        "tab in scheme": "ht\ttps://gateway.example",
        "path under the origin": "https://gateway.example/prefix",
        "trailing slash on the origin": "https://gateway.example/",
    }.items():
        yield name, origin + START_PATH
    for name, path in {
        "no path": "",
        "root path": "/",
        "another route": "/api/agentic-new/managed-execution/drive",
        "trailing slash": START_PATH + "/",
        "longer path": START_PATH + "/now",
        "shorter path": START_PATH[:-1],
        "doubled slash": "/" + START_PATH,
        "upper case": START_PATH.upper(),
        "dot segment": START_PATH.replace("/start", "/./start"),
        "parent segment": START_PATH.replace("/start", "/x/../start"),
        "encoded letter": START_PATH.replace("/start", "/%73tart"),
        "parameters": START_PATH + ";v=1",
        "query": START_PATH + "?job=1",
        "empty query": START_PATH + "?",
        "fragment": START_PATH + "#start",
        "empty fragment": START_PATH + "#",
        "trailing space": START_PATH + " ",
        "trailing line end": START_PATH + "\n",
        "nul": START_PATH + "\x00",
    }.items():
        yield name, https + path
        yield name + " on loopback", loopback + path
    yield "too long", "https://" + "a" * (2049 - len("https://") - len(START_PATH)) + START_PATH


@pytest.mark.parametrize("body", [pytest.param(body, id=name) for name, body in invalid_orders()])
def test_invalid_run_request_is_refused_before_anything_is_accepted(start, body):
    worker = start()
    assert call(worker.port, "POST", "/run", body) == (400, refused("invalid_request"))
    assert worker.spawned == [] and worker.exits == []
    assert not worker.server._claimed.locked()


def test_run_accepts_the_limits_of_the_contract(start):
    files = {"main.py": "import os\nprint(len(os.listdir('.')))\n", **{f"module_{index}.py": "" for index in range(62)}}
    # 512 characters, each component short enough for a real file system.
    longest = "a" * 200 + "/" + "b" * 200 + "/" + "c" * 107 + ".py"
    files[longest] = ""
    assert len(files) == 64 and len(longest) == 512
    for timeout in (1, 3600):
        reply = call(start().port, "POST", "/run", order(files=files, config={}, timeout_seconds=timeout))
        assert reply == (200, outcome(stdout="64\n"))
    names = {"main.py": "print('named')\n", "donn\u00e9es/\u6a21\u5757 one.py": "", ".hidden/..py": ""}
    assert call(start().port, "POST", "/run", order(files=names)) == (200, outcome(stdout="named\n"))


def test_run_request_framing(start):
    worker = start()
    body = json.dumps(order()).encode()
    head = b"POST /run HTTP/1.1\r\nHost: worker.test\r\n"
    json_type = b"Content-Type: application/json\r\n"
    length = b"Content-Length: %d\r\n" % len(body)
    for name, request in {
        "no length": head + json_type + b"\r\n" + body,
        "chunked": head + json_type + b"Transfer-Encoding: chunked\r\n\r\n%x\r\n%s\r\n0\r\n\r\n" % (len(body), body),
        "length and chunked": head + json_type + length + b"Transfer-Encoding: chunked\r\n\r\n" + body,
        "two lengths": head + json_type + length + length + b"\r\n" + body,
        "length not a number": head + json_type + b"Content-Length: many\r\n\r\n" + body,
        "negative length": head + json_type + b"Content-Length: -1\r\n\r\n" + body,
        "no content type": head + length + b"\r\n" + body,
        "text content type": head + b"Content-Type: text/plain\r\n" + length + b"\r\n" + body,
        "form content type": head + b"Content-Type: application/x-www-form-urlencoded\r\n" + length + b"\r\n" + body,
        "body shorter than declared": head + json_type + b"Content-Length: %d\r\n\r\n" % (len(body) + 5) + body,
    }.items():
        with contextlib.ExitStack() as stack:
            if name == "body shorter than declared":
                # The worker waits for the missing bytes; shorten its 30 second patience for this one case.
                stack.enter_context(pytest.MonkeyPatch.context()).setattr(managed_remote._Handler, "timeout", 0.3)
            # Not "invalid_request", which ends the job: the package may be fine,
            # and a hop that sent it this way can send it readable next time.
            assert exchange(worker.port, request) == (400, refused("unreadable_request")), name
    assert worker.spawned == [] and not worker.server._claimed.locked()
    # A charset parameter does not make JSON anything else.
    request = head + b"Content-Type: application/json; charset=utf-8\r\n" + length + b"\r\n" + body
    assert exchange(worker.port, request) == (200, outcome(stdout="ran\n"))


def test_expect_continue_is_answered_before_the_body(start):
    # What curl sends for a large body when an operator tries the worker by hand.
    body = json.dumps(order()).encode()
    head = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: %d\r\nExpect: 100-continue\r\n\r\n"
    with socket.create_connection(("127.0.0.1", start().port), timeout=30) as connection:
        connection.sendall(head % len(body))
        interim = b""
        while not interim.endswith(b"\r\n\r\n"):
            chunk = connection.recv(65536)
            assert chunk
            interim += chunk
        connection.sendall(body)
        final = b""
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            final += chunk
    assert interim == b"HTTP/1.1 100 Continue\r\n\r\n"
    assert final.startswith(b"HTTP/1.1 200 OK\r\n")
    assert json.loads(final.partition(b"\r\n\r\n")[2]) == outcome(stdout="ran\n")


def test_body_limit_is_four_mebibytes_inclusive(start):
    worker = start()
    limit = 4 * 1024 * 1024
    # Refused on the declared size alone: no body follows, so a worker that waited for one would stall here.
    for declared in (limit + 1, 10**9, 10**30):
        request = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % declared
        assert exchange(worker.port, request, timeout=10) == (413, refused("payload_too_large"))
    assert not worker.server._claimed.locked()
    frame = order("print('fits')\n")
    frame["files"]["padding.py"] = ""
    frame["files"]["padding.py"] = "#" * (limit - len(json.dumps(frame).encode()))
    body = json.dumps(frame).encode()
    assert len(body) == limit
    assert call(worker.port, "POST", "/run", body) == (200, outcome(stdout="fits\n"))


def test_image_mismatch_is_refused_without_using_up_the_worker(start):
    worker = start()
    for image in ("sha256:" + "5a" * 32, IMAGE.replace("5a", "5b"), IMAGE.upper(), ""):
        assert call(worker.port, "POST", "/run", order(image=image)) == (409, refused("image_mismatch"))
    assert worker.spawned == [] and worker.exits == []
    assert call(worker.port, "POST", "/run", order()) == (200, outcome(stdout="ran\n"))


def test_bare_image_id_is_compared_exactly(start):
    bare = "sha256:" + "0f" * 32
    worker = start(**{IMAGE_SETTING: bare})
    assert call(worker.port, "GET", "/inventory").body["image"] == bare
    assert call(worker.port, "POST", "/run", order()) == (409, refused("image_mismatch"))
    assert call(worker.port, "POST", "/run", order(image=bare)) == (200, outcome(stdout="ran\n"))


def test_unknown_path_and_wrong_method(start):
    worker = start(auth="token")
    for path in ("/", "/run/", "/runs", "/healthz/live", "/inventory.json", "/api/agentic-new/managed-execution/worker/claim"):
        for method in ("GET", "POST"):
            assert call(worker.port, method, path) == (404, refused("not_found"))
    for method, path in (
        ("POST", "/healthz"), ("POST", "/inventory"), ("GET", "/run"), ("PUT", "/run"), ("DELETE", "/run"),
        ("PATCH", "/inventory"), ("OPTIONS", "/run"), ("BREW", "/run"), ("BREW", "/unknown"),
    ):
        assert call(worker.port, method, path, token=TOKEN) == (405, refused("method_not_allowed"))
    assert call(worker.port, "HEAD", "/healthz") == (405, None)
    assert worker.spawned == [] and worker.exits == []


def test_malformed_http_is_answered_in_the_worker_format(start):
    worker = start()
    # Never "invalid_request": none of these got as far as a package.
    assert exchange(worker.port, b"this is not http\r\n\r\n") == (400, refused("unreadable_request"))
    assert exchange(worker.port, b"GET /healthz HTTP/9.9\r\n\r\n") == (505, refused("unreadable_request"))
    assert exchange(worker.port, b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\n\r\n") == (414, refused("unreadable_request"))
    assert exchange(worker.port, b"GET /healthz HTTP/1.1\r\n" + b"X-Filler: 1\r\n" * 200 + b"\r\n") == (431, refused("unreadable_request"))
    # The same for a run request that http.server gives up on before the worker sees it.
    crowded = b"POST /run HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n" + b"X-Filler: 1\r\n" * 200 + b"\r\n{}"
    assert exchange(worker.port, crowded) == (431, refused("unreadable_request"))
    assert worker.spawned == [] and not worker.server._claimed.locked()
    assert call(worker.port, "GET", "/healthz") == (200, {"ok": True})


def test_failure_before_acceptance_says_the_program_did_not_start(start, monkeypatch, capfd):
    worker = start()

    def broken(_raw):
        raise RuntimeError("defect carrying sk-ds-fixture-virtual-key")

    with monkeypatch.context() as patched:
        patched.setattr(managed_remote, "_parse_order", broken)
        assert call(worker.port, "POST", "/run", order()) == (500, refused("launch_failed"))
    # Only the kind of failure is written, never its text.
    assert capfd.readouterr() == ("", "managed worker: request failed (RuntimeError)\n")
    assert not worker.server._claimed.locked() and worker.exits == []
    assert call(worker.port, "POST", "/run", order()) == (200, outcome(stdout="ran\n"))


# --- authentication ---------------------------------------------------------------


def test_token_is_required_and_compared_in_constant_time(start, monkeypatch):
    compared = []
    real = managed_remote.hmac.compare_digest

    def compare(supplied, expected):
        compared.append((len(supplied), len(expected)))
        return real(supplied, expected)

    monkeypatch.setattr(managed_remote.hmac, "compare_digest", compare)
    worker = start(auth="token")
    wrong = [
        (),
        ("Authorization: Bearer " + "x" * len(TOKEN),),
        ("Authorization: Bearer " + TOKEN[:-1],),
        ("Authorization: Bearer " + TOKEN + "x",),
        ("Authorization: bearer " + TOKEN,),
        ("Authorization: Basic " + TOKEN,),
        ("Authorization: " + TOKEN,),
        ("Authorization: Bearer  " + TOKEN,),
        ("Authorization: Bearer " + TOKEN, "Authorization: Bearer " + TOKEN),
        ("X-Authorization: Bearer " + TOKEN,),
    ]
    for headers in wrong:
        for method, path, body in (("GET", "/inventory", None), ("POST", "/run", order())):
            assert call(worker.port, method, path, body, headers=headers) == (401, refused("worker_unauthorized"))
    # Always one comparison of two equal-length digests, whatever was supplied.
    assert compared == [(32, 32)] * (2 * len(wrong))
    assert worker.spawned == [] and not worker.server._claimed.locked()
    assert call(worker.port, "GET", "/inventory", token=TOKEN).status == 200
    assert call(worker.port, "POST", "/run", order(), token=TOKEN) == (200, outcome(stdout="ran\n"))
    assert compared[-2:] == [(32, 32)] * 2


def test_authentication_comes_before_any_look_at_the_request(start):
    worker = start(auth="token")
    assert call(worker.port, "POST", "/run", b"not json") == (401, refused("worker_unauthorized"))
    assert call(worker.port, "POST", "/run", order(image="sha256:" + "00" * 32)) == (401, refused("worker_unauthorized"))
    oversized = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: 99999999\r\n\r\n"
    assert exchange(worker.port, oversized, timeout=10) == (401, refused("worker_unauthorized"))
    # A caller still sending megabytes of body reads the refusal instead of a connection reset.
    large = order("#" * (3 * 1024 * 1024))
    assert call(worker.port, "POST", "/run", large, token="not-the-token-" + "x" * 32) == (401, refused("worker_unauthorized"))
    assert worker.spawned == []


def test_platform_mode_checks_no_header(start):
    # A token left in the environment is not used: the platform authenticated the caller.
    worker = start(auth="platform", **{TOKEN_SETTING: TOKEN})
    assert worker.server.credential is None
    assert call(worker.port, "GET", "/inventory", headers=("Authorization: Bearer anything",)).status == 200
    assert call(worker.port, "POST", "/run", order()) == (200, outcome(stdout="ran\n"))


# --- what reaches the program -----------------------------------------------------


@pytest.mark.skipif(not os.path.exists("/usr/bin/env"), reason="needs /usr/bin/env to print an environment exactly as passed")
@pytest.mark.parametrize("worker_has_them", [True, False])
def test_program_environment_is_built_from_nothing(start, monkeypatch, worker_has_them):
    monkeypatch.setenv("DEEPINTSHIELD_PROCESS_SECRET", "process-environment-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "process-cloud-secret")
    environ = {
        "DEEPINTSHIELD_VIRTUAL_KEY": "sk-ds-worker-environment", "DEEPINTSHIELD_BASE_URL": "https://gateway.example",
        "AWS_SECRET_ACCESS_KEY": "cloud-secret", "GOOGLE_APPLICATION_CREDENTIALS": "/var/run/secret.json",
        "K_SERVICE": "managed-worker", "KUBERNETES_SERVICE_HOST": "10.0.0.1", "PORT": "8080",
        "PYTHONPATH": "/injected", "LD_PRELOAD": "/injected.so", "HOME": "/root", "TMPDIR": "/worker-tmp",
        "LANG": "en_US.UTF-8", "USER": "root",
    }
    passed = {
        "PATH": "/opt/worker/bin:/usr/bin:/bin", "SSL_CERT_FILE": "/etc/ssl/custom.pem",
        "SSL_CERT_DIR": "/etc/ssl/custom", "REQUESTS_CA_BUNDLE": "/etc/ssl/bundle.pem",
    }
    environ.update(passed if worker_has_them else {"PATH": "", "SSL_CERT_FILE": ""})
    worker = start(auth="token", child_command=lambda config_path, entrypoint_path: ["/usr/bin/env"], **environ)
    reply = call(worker.port, "POST", "/run", order(), token=TOKEN)
    assert reply == (200, outcome(stdout=reply.body["stdout"]))
    environment = dict(line.split("=", 1) for line in reply.body["stdout"].splitlines())
    root = os.path.dirname(environment["HOME"])
    assert os.path.basename(root).startswith("deepintshield-managed-") and not os.path.exists(root)
    assert environment == {
        "HOME": os.path.join(root, "home"),
        "TMPDIR": os.path.join(root, "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        **(passed if worker_has_them else {"PATH": "/usr/local/bin:/usr/bin:/bin"}),
    }


def test_program_inherits_no_descriptor_of_the_worker(start):
    # Hold a second connection open so the worker owns more than its listener
    # while the program starts: neither may be reachable from the program.
    worker = start()
    listing = (
        "import os\n"
        "inherited = []\n"
        "for descriptor in range(3, 256):\n"
        "    try:\n"
        "        os.fstat(descriptor)\n"
        "    except OSError:\n"
        "        continue\n"
        "    inherited.append(descriptor)\n"
        "print(inherited)\n"
    )
    with socket.create_connection(("127.0.0.1", worker.port), timeout=30):
        assert call(worker.port, "POST", "/run", order(listing)) == (200, outcome(stdout="[]\n"))


def test_root_worker_runs_the_program_as_the_unprivileged_user(start, monkeypatch):
    owned, identity = [], {}
    real = subprocess.Popen

    def popen(*arguments, **options):
        # Becoming another user needs root; record the request and run as ourselves.
        identity.update({name: options.pop(name, None) for name in ("user", "group", "extra_groups")})
        return real(*arguments, **options)

    monkeypatch.setattr(managed_remote, "_program_identity", lambda: (65532, 65532))
    monkeypatch.setattr(managed_remote.os, "chown", lambda path, user, group: owned.append((path, user, group)))
    monkeypatch.setattr(managed_remote.subprocess, "Popen", popen)
    reply = call(start().port, "POST", "/run", order(files=STAGING_FILES))
    assert reply.status == 200 and reply.body["exit_code"] == 7
    report = json.loads(reply.body["stdout"])
    assert identity == {"user": 65532, "group": 65532, "extra_groups": []}
    # That user owns the whole staging directory before the program starts.
    assert sorted(owned) == sorted((path, 65532, 65532) for path in [report["root"], *report["tree"]])


def test_only_a_root_worker_switches_user(monkeypatch):
    monkeypatch.undo()
    with monkeypatch.context() as patched:
        patched.setattr(managed_remote.os, "geteuid", lambda: 0)
        assert managed_remote._program_identity() == (65532, 65532)
        patched.setattr(managed_remote.os, "geteuid", lambda: 65532)
        assert managed_remote._program_identity() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_DUMPABLE is a Linux control")
def test_worker_makes_itself_unreadable_to_its_own_user():
    script = (
        "import ctypes\n"
        "from deepintshield.agentic import managed_remote\n"
        "libc = ctypes.CDLL(None)\n"
        "before = libc.prctl(3, 0, 0, 0, 0)\n"  # PR_GET_DUMPABLE
        "managed_remote._deny_inspection()\n"
        "print(before, libc.prctl(3, 0, 0, 0, 0))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert (result.returncode, result.stdout) == (0, "1 0\n"), result.stderr


def test_denying_inspection_never_fails(monkeypatch):
    monkeypatch.undo()
    if sys.platform.startswith("linux"):
        pytest.skip("would change the test process itself; covered in a separate process")
    assert managed_remote._deny_inspection() is None


# --- the caller or the platform goes away ---------------------------------------------


def test_caller_leaving_stops_the_program(start, beacon, capfd):
    worker = start()
    body = json.dumps(order(beacon.prelude + "import time\ntime.sleep(120)\n")).encode()
    request = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body)
    with socket.create_connection(("127.0.0.1", worker.port), timeout=30) as connection:
        connection.sendall(request + body)
        pid = beacon.started()
        assert worker.exits == []
    # Nobody is left to hear the outcome, as when the Docker runner's caller is cancelled.
    assert wait_for(lambda: worker.exits == [0])
    assert wait_for(lambda: gone(pid))
    assert not os.path.exists(os.path.dirname(worker.spawned[0][0]))
    # The operator is told once, in words that carry nothing of the request.
    assert capfd.readouterr() == ("", ABANDONED)


def test_caller_that_half_closes_is_taken_for_one_that_left(start, gateway, capfd):
    worker, standin = start(), gateway()
    body = json.dumps(order("import time\ntime.sleep(120)\n", start=standin.start)).encode()
    request = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body)
    with socket.create_connection(("127.0.0.1", worker.port), timeout=30) as connection:
        connection.sendall(request + body)
        # What some proxies do once a request is sent. This caller goes on reading, and is answered nothing.
        connection.shutdown(socket.SHUT_WR)
        assert connection.recv(65536) == b""
    assert wait_for(lambda: worker.exits == [0])
    # Already gone when the run was accepted: no start was asked for and nothing ran.
    assert standin.asked == [] and worker.spawned == []
    assert capfd.readouterr() == ("", ABANDONED)


def test_caller_that_left_before_its_run_was_staged_uses_no_start(start, gateway, monkeypatch, capfd):
    worker, standin = start(), gateway()
    staged = []
    stage = managed_remote._stage

    def slow_stage(*arguments):
        # Long enough for the caller's close to arrive before the worker looks.
        staged.append(stage(*arguments))
        time.sleep(0.3)
        return staged[-1]

    monkeypatch.setattr(managed_remote, "_stage", slow_stage)
    body = json.dumps(order(start=standin.start)).encode()
    request = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body)
    with socket.create_connection(("127.0.0.1", worker.port), timeout=30) as connection:
        connection.sendall(request + body)
    assert wait_for(lambda: worker.exits == [0])
    # The gateway can withdraw a start nobody asked for, and so knows the program never began.
    assert standin.asked == [] and worker.spawned == []
    assert len(staged) == 1 and not os.path.exists(staged[0])
    assert capfd.readouterr() == ("", ABANDONED)


def test_stop_ends_a_running_program_and_still_answers(start, beacon, background):
    worker = start()
    pending = background(call, worker.port, "POST", "/run", order(beacon.prelude + "import time\ntime.sleep(120)\n"))
    pid = beacon.started()
    worker.server.stop()
    # Stopped part-way, exactly like a run stopped at its deadline: not a program that failed.
    assert pending.result(60) == (200, outcome(exit_code=128 + signal.SIGKILL, timed_out=True))
    assert wait_for(lambda: worker.exits == [0]) and wait_for(lambda: gone(pid))


def test_program_that_ends_by_itself_as_the_worker_is_stopped_was_not_stopped(start, beacon, background, monkeypatch):
    real = managed_remote._kill_group

    def kill_group(pid):
        # The program gets to end by itself between the worker's decision and its kill.
        beacon.release()
        time.sleep(0.5)
        real(pid)

    monkeypatch.setattr(managed_remote, "_kill_group", kill_group)
    worker = start()
    pending = background(call, worker.port, "POST", "/run", order(beacon.prelude + "beacon.recv(1)\nraise SystemExit(3)\n"))
    beacon.started()
    worker.server.stop()
    assert pending.result(60) == (200, outcome(exit_code=3))


def test_stop_ends_an_idle_worker_at_once(start):
    worker = start()
    worker.server.stop()
    assert worker.exits == [0]
    # A run that arrives during shutdown is refused as busy: the gateway asks again, and reaches a fresh instance.
    assert call(worker.port, "POST", "/run", order()) == (409, refused("worker_busy"))
    assert worker.spawned == []
    assert wait_for(lambda: worker.exits == [0, 0])


def test_stop_takes_the_single_run_before_it_decides_the_worker_is_idle():
    held = []
    server = managed_remote.create_server(VALID, exit=lambda code: held.append((code, server._claimed.locked())), address=("127.0.0.1", 0))
    try:
        server.stop()
        assert held == [(0, True)]
        assert not server.claim()
    finally:
        server.server_close()


def test_run_still_arriving_when_the_worker_is_stopped_is_refused_as_busy(start):
    worker = start()
    body = json.dumps(order()).encode()
    head = b"POST /run HTTP/1.1\r\nHost: worker.test\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body)
    with socket.create_connection(("127.0.0.1", worker.port), timeout=30) as connection:
        # All but the last byte: the worker cannot have accepted this run yet.
        connection.sendall(head + body[:-1])
        worker.server.stop()
        assert worker.exits == [0]
        connection.sendall(body[-1:])
        assert answer_on(connection) == (409, refused("worker_busy"))
    assert worker.spawned == []
    assert wait_for(lambda: worker.exits == [0, 0])


# --- settings -----------------------------------------------------------------------

VALID = {IMAGE_SETTING: IMAGE, AUTH_SETTING: "platform"}
DIGEST = "sha256:" + "ab" * 32


@pytest.mark.parametrize(("changes", "setting"), [
    ({IMAGE_SETTING: None}, IMAGE_SETTING),
    ({IMAGE_SETTING: ""}, IMAGE_SETTING),
    ({IMAGE_SETTING: "registry.example/worker:latest"}, IMAGE_SETTING),
    ({IMAGE_SETTING: "registry.example/worker@sha256:" + "AB" * 32}, IMAGE_SETTING),
    ({IMAGE_SETTING: "registry.example/worker@sha256:" + "ab" * 31}, IMAGE_SETTING),
    ({IMAGE_SETTING: "registry.example/worker@sha512:" + "ab" * 32}, IMAGE_SETTING),
    ({IMAGE_SETTING: "registry.example/wor ker@" + DIGEST}, IMAGE_SETTING),
    ({IMAGE_SETTING: "@" + DIGEST}, IMAGE_SETTING),
    ({IMAGE_SETTING: DIGEST + "\n"}, IMAGE_SETTING),
    ({IMAGE_SETTING: DIGEST + "0"}, IMAGE_SETTING),
    ({AUTH_SETTING: None}, AUTH_SETTING),
    ({AUTH_SETTING: ""}, AUTH_SETTING),
    ({AUTH_SETTING: "none"}, AUTH_SETTING),
    ({AUTH_SETTING: "google"}, AUTH_SETTING),
    ({AUTH_SETTING: "Token"}, AUTH_SETTING),
    ({AUTH_SETTING: "token "}, AUTH_SETTING),
    ({AUTH_SETTING: "token"}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: ""}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: "t" * 31}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: "t" * 513}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: "t" * 40 + " "}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: "t" * 40 + "\n"}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: "t" * 40 + "!"}, TOKEN_SETTING),
    ({AUTH_SETTING: "token", TOKEN_SETTING: "\u00e9" * 40}, TOKEN_SETTING),
    ({"PORT": "0"}, "PORT"),
    ({"PORT": "65536"}, "PORT"),
    ({"PORT": "-1"}, "PORT"),
    ({"PORT": "http"}, "PORT"),
    ({"PORT": "80 80"}, "PORT"),
    ({"PORT": "\uff18\uff10\uff18\uff10"}, "PORT"),
])
def test_invalid_setting_stops_start_up_and_names_only_the_setting(capsys, changes, setting):
    environ = {name: value for name, value in {**VALID, **changes}.items() if value is not None}
    with pytest.raises(managed_remote._InvalidSetting) as refusal:
        managed_remote.create_server(environ, address=("127.0.0.1", 0))
    assert refusal.value.name == setting and str(refusal.value) == setting
    exits = []
    assert managed_remote.serve(environ, exit=exits.append) == 2
    assert exits == []
    # One line that names the setting and never shows what it was set to.
    assert capsys.readouterr() == ("", f"managed worker: {setting} is missing or invalid\n")


@pytest.mark.parametrize("changes", [
    {},
    {IMAGE_SETTING: DIGEST},
    {IMAGE_SETTING: "localhost:5000/team/worker:2026.10@" + DIGEST},
    {IMAGE_SETTING: "REGION-docker.pkg.dev/project/repo/worker@" + DIGEST},
    {AUTH_SETTING: "token", TOKEN_SETTING: "t" * 32},
    {AUTH_SETTING: "token", TOKEN_SETTING: "t" * 512},
    {AUTH_SETTING: "token", TOKEN_SETTING: TOKEN},
    {TOKEN_SETTING: "ignored in platform mode"},
    {"PORT": ""},
    {"PORT": "1"},
    {"PORT": "65535"},
])
def test_valid_settings(changes):
    environ = {**VALID, **changes}
    port, image, token = managed_remote._settings(environ)
    assert port == int(environ.get("PORT") or 8080)
    assert image == environ[IMAGE_SETTING]
    assert token == (environ[TOKEN_SETTING] if environ[AUTH_SETTING] == "token" else None)


def test_listens_on_every_interface_at_the_configured_port():
    port = free_port()
    server = managed_remote.create_server({**VALID, "PORT": str(port)}, exit=lambda code: None)
    try:
        assert server.server_address == ("0.0.0.0", port)
    finally:
        server.server_close()


def test_port_that_cannot_be_bound_stops_start_up(capsys):
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", 0))
        taken.listen(1)
        environ = {**VALID, "PORT": str(taken.getsockname()[1])}
        assert managed_remote.serve(environ, exit=lambda code: None) == 2
        assert capsys.readouterr() == ("", "managed worker: cannot listen on the port given by PORT\n")


def test_main_reads_the_process_environment(monkeypatch, capsys):
    for name in (IMAGE_SETTING, AUTH_SETTING, TOKEN_SETTING):
        monkeypatch.delenv(name, raising=False)
    assert managed_remote.main() == 2
    assert capsys.readouterr().err == f"managed worker: {IMAGE_SETTING} is missing or invalid\n"


# --- nothing sensitive is written ---------------------------------------------------


def test_worker_writes_nothing_of_its_own(start, capfd):
    worker = start(auth="token")
    leaky = "import sys\nprint(open(sys.argv[1]).read())\nsys.stderr.write('sk-ds-fixture-virtual-key\\n')\n"
    assert call(worker.port, "GET", "/healthz").status == 200
    assert call(worker.port, "GET", "/inventory", token=TOKEN).status == 200
    assert call(worker.port, "GET", "/inventory", token="wrong-" + TOKEN).status == 401
    assert call(worker.port, "GET", "/missing?token=" + TOKEN).status == 404
    assert call(worker.port, "POST", "/run", {"config": CONFIG, "unexpected": TOKEN}, token=TOKEN).status == 400
    assert exchange(worker.port, b"not http " + TOKEN.encode() + b"\r\n\r\n").status == 400
    reply = call(worker.port, "POST", "/run", order(leaky), token=TOKEN)
    # The program's own output goes to the caller, who is entitled to it, and nowhere else.
    assert reply.status == 200 and "fixture-private-key-material" in reply.body["stdout"]
    assert wait_for(lambda: worker.exits == [0])
    assert capfd.readouterr() == ("", "")


# --- the real module, as its own process ----------------------------------------------


@contextlib.contextmanager
def worker_process(arguments, **settings):
    """Start the module for real: it listens on 0.0.0.0, hardens itself and ends with os._exit."""
    for attempt in range(3):
        port = free_port()
        environ = {
            **os.environ, "PYTHONPATH": str(SOURCE_ROOT), "PORT": str(port),
            IMAGE_SETTING: IMAGE, AUTH_SETTING: "token", TOKEN_SETTING: TOKEN, **settings,
        }
        process = subprocess.Popen([sys.executable, *arguments], env=environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def ready():
            if process.poll() is not None:
                return True
            try:
                return call(port, "GET", "/healthz", timeout=5) == (200, {"ok": True})
            except OSError:
                return False

        if wait_for(ready, 120) and process.poll() is None:
            break
        process.kill()
        output = process.communicate()
        # Another process can take the probed port first; that is the one start worth repeating.
        if attempt == 2 or "cannot listen" not in output[1]:
            pytest.fail(f"the worker did not start: {output}")
    try:
        yield SimpleNamespace(process=process, port=port)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def finish(process, seconds=60):
    """Wait for the worker to end by itself and return (exit code, stdout, stderr)."""
    stdout, stderr = process.communicate(timeout=seconds)
    return process.returncode, stdout, stderr


def isolated_launcher_available():
    """Whether the real launcher can start here: ``python -I`` needs the SDK installed in the interpreter."""
    environment = managed_remote._child_environment(os.environ, "/nonexistent")
    probe = subprocess.run(
        [sys.executable, "-I", "-c", "import deepintshield.agentic.managed_runner"],
        env=environment, capture_output=True, timeout=120, check=False,
    )
    return probe.returncode == 0


def test_real_worker_process_serves_one_run_and_exits_by_itself():
    with worker_process(["-m", "deepintshield.agentic.managed_remote"]) as worker:
        assert call(worker.port, "GET", "/inventory").status == 401
        inventory = call(worker.port, "GET", "/inventory", token=TOKEN)
        assert inventory.status == 200 and inventory.body["image"] == IMAGE
        # The real launcher refuses this configuration, or cannot be imported by
        # ``python -I`` at all: either way a launcher process was started and failed.
        refused_by_launcher = {"environment": {}, "runtime_private_key": "fixture-private-key-material", "requirements": []}
        reply = call(worker.port, "POST", "/run", order(config=refused_by_launcher), token=TOKEN, timeout=120)
        assert reply.status == 200
        assert reply.body["started"] is True and reply.body["timed_out"] is False
        assert reply.body["exit_code"] != 0 and reply.body["stdout"] == ""
        assert finish(worker.process) == (0, "", "")


def test_real_worker_process_runs_a_program_through_the_real_launcher():
    if not isolated_launcher_available():
        pytest.skip("the SDK is not installed in this interpreter, so ``python -I`` cannot import the launcher")
    program = (
        "import json, os\n"
        "from deepintshield.agentic import managed_execution\n"
        "print(json.dumps({'worker': managed_execution._WORKER_FILES is not None, 'names': sorted(os.environ)}))\n"
    )
    with worker_process(["-m", "deepintshield.agentic.managed_remote"], AWS_SECRET_ACCESS_KEY="cloud-secret") as worker:
        reply = call(worker.port, "POST", "/run", order(program), token=TOKEN, timeout=120)
        assert reply.status == 200, reply
        assert (reply.body["exit_code"], reply.body["timed_out"], reply.body["stderr"]) == (0, False, ""), reply
        report = json.loads(reply.body["stdout"])
        assert report["worker"] is True
        # Only what the launcher itself sets from the approved configuration, never the worker's settings.
        managed = {name for name in report["names"] if name.startswith("DEEPINTSHIELD_")}
        assert managed == {*CONFIG["environment"], "DEEPINTSHIELD_RUNTIME_KEY_FILE", "DEEPINTSHIELD_MANAGED_EXECUTION_WORKER"}
        assert "AWS_SECRET_ACCESS_KEY" not in report["names"] and "PYTHONPATH" not in report["names"]
        assert finish(worker.process) == (0, "", "")


DRIVER = (
    "import os, sys\n"
    "from deepintshield.agentic import managed_remote\n"
    "command = lambda config, entry: [*sys.argv[1:], sys.executable, '-E', '-s', '-B', entry, config]\n"
    "raise SystemExit(managed_remote.serve(os.environ, child_command=command))\n"
)
DRIVER_ARGUMENTS = ["-c", DRIVER, *(["/usr/bin/env", *LOADER] if LOADER else [])]


def test_terminated_worker_process_ends_its_program_and_answers(beacon, background):
    with worker_process(DRIVER_ARGUMENTS) as worker:
        pending = background(call, worker.port, "POST", "/run", order(beacon.prelude + "import time\ntime.sleep(120)\n"), token=TOKEN)
        pid = beacon.started()
        worker.process.send_signal(signal.SIGTERM)
        assert pending.result(60) == (200, outcome(exit_code=128 + signal.SIGKILL, timed_out=True))
        assert finish(worker.process) == (0, "", "")
        assert wait_for(lambda: gone(pid))


def test_terminated_idle_worker_process_leaves_at_once():
    with worker_process(DRIVER_ARGUMENTS) as worker:
        began = time.monotonic()
        worker.process.send_signal(signal.SIGTERM)
        assert finish(worker.process) == (0, "", "")
        assert time.monotonic() - began < 30


def test_worker_process_without_a_grant_starts_nothing_and_exits_by_itself():
    with worker_process(DRIVER_ARGUMENTS) as worker:
        nowhere = {"url": f"http://127.0.0.1:{free_port()}{START_PATH}", "token": START_TOKEN}
        reply = call(worker.port, "POST", "/run", order(start=nowhere), token=TOKEN, timeout=120)
        assert reply == (409, refused("start_not_granted"))
        assert finish(worker.process) == (0, "", "")


def test_abandoned_worker_process_says_so_before_it_ends(beacon):
    with worker_process(DRIVER_ARGUMENTS) as worker:
        body = json.dumps(order(beacon.prelude + "import time\ntime.sleep(120)\n")).encode()
        head = (
            b"POST /run HTTP/1.1\r\nHost: worker.test\r\nAuthorization: Bearer %s\r\n"
            b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (TOKEN.encode(), len(body))
        )
        with socket.create_connection(("127.0.0.1", worker.port), timeout=30) as connection:
            connection.sendall(head + body)
            pid = beacon.started()
        # Written although the process ends without flushing anything.
        assert finish(worker.process) == (0, "", ABANDONED)
        assert wait_for(lambda: gone(pid))
