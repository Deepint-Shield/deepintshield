from __future__ import annotations

import asyncio
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys
import threading

import httpx
import pytest

pytest.importorskip("cryptography")

from deepintshield.agentic import registry
from deepintshield.agentic.credentials import runtime_key
from deepintshield.agentic.credentials import gateway_broker
from deepintshield.agentic.credentials.gateway_broker import GatewayBrokerCredential
from deepintshield.agentic.dpop import DPoPKey, process_dpop_key
from deepintshield.agentic.errors import GovernanceConfigurationError
from deepintshield.agentic.types import RuntimeBrokerInfo
from deepintshield.mcp._workload import gateway_workload_hook
from tests.test_agentic_dpop import _ath, _decode, _verify


ORIGIN = "https://gateway.example"
PREFIX = "/api/agentic-new/runtime-auth/"
TOKEN = "dsrs_" + "a" * 43


def config(**changes):
    return RuntimeBrokerInfo(**{
        "enrollment_path": PREFIX + "enroll",
        "challenge_path": PREFIX + "challenge",
        "session_path": PREFIX + "session",
        "tenant_id": "platform-tenant", "workspace_id": "workspace-one",
        "agent_subject": "agent:math-agent", "runtime_kind": "key",
        **changes,
    })


def thumbprint(proof):
    jwk = json.loads(_decode(proof.split(".")[0]))["jwk"]
    raw = json.dumps(jwk, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()


@pytest.fixture(autouse=True)
def isolated_keys(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPINTSHIELD_RUNTIME_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KEY_DIR", str(tmp_path / "keys"))
    monkeypatch.setattr(runtime_key, "_secure_keyring", lambda: None)


class Broker:
    def __init__(self, *, approved=True, broker=None):
        self.approved = approved
        self.config = broker or config()
        self.calls = []
        self.jkt = ""
        self.session_changes = {}
        self.enrollment_changes = {}
        self.challenge_changes = {}

    def __call__(self, request):
        self.calls.append(request.url.path)
        if request.url.path == "/api/agentic-new/credential-info":
            return httpx.Response(200, json={
                "provider_type": "entra_agent_id", "agent_configured": True,
                "agent_subject": self.config.agent_subject,
                "require_proof_of_possession": True,
                "runtime_auth_mode": "gateway_broker",
                "runtime_broker": self.config.model_dump(),
            })
        assert request.headers["Authorization"] == "Bearer sk-ds-test"
        assert request.headers["X-Agent-Subject"] == self.config.agent_subject
        proof = _verify(request.headers["DPoP"])
        assert proof["htu"] == str(request.url)
        assert proof["htm"] == request.method
        jkt = thumbprint(request.headers["DPoP"])
        body = json.loads(request.content or b"{}")
        if request.url.path == PREFIX + "enroll":
            assert "d" not in body["public_jwk"]
            assert proof["ath"] == _ath("sk-ds-test")
            self.jkt = jkt
            return httpx.Response(200, json={
                "binding_id": "binding-one", "key_thumbprint": jkt,
                "status": "approved" if self.approved else "pending",
                "enabled": self.approved,
                **self.enrollment_changes,
            })
        assert jkt == self.jkt
        if request.url.path == PREFIX + "challenge":
            assert proof["ath"] == _ath("sk-ds-test")
            assert body == {"key_thumbprint": jkt}
            return httpx.Response(200, json={
                "binding_id": "binding-one", "nonce": "one-use-nonce", "expires_in": 60,
                **self.challenge_changes,
            })
        if request.url.path == PREFIX + "session":
            assert body["binding_id"] == "binding-one"
            assert proof["nonce"] == body["nonce"] == "one-use-nonce"
            assert proof["ath"] == _ath("one-use-nonce")
            return httpx.Response(200, json={
                "binding_id": "binding-one", "access_token": TOKEN,
                "token_type": "DPoP", "expires_in": 300, "cnf": {"jkt": jkt},
                **self.session_changes,
            })
        assert request.headers["X-Agent-Token"] == TOKEN
        assert proof["ath"] == _ath(TOKEN)
        return httpx.Response(200, json={})


def credential(broker, http=None):
    return GatewayBrokerCredential(
        http=http or httpx.Client(transport=httpx.MockTransport(broker)),
        gateway_url=ORIGIN, virtual_key="sk-ds-test",
        agent_subject=broker.config.agent_subject, broker=broker.config,
    )


def test_pending_enrollment_stops_execution_and_reuses_key_after_approval(shield_factory, caplog, monkeypatch):
    # The single pending line is printed once per process by the engine/registry.
    monkeypatch.setattr(registry, "_enrolment_logged", set())
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")
    broker = Broker(approved=False)
    shield = shield_factory(broker, base_url=ORIGIN, agent_name="math-agent")
    engine = shield.agentic.engine
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_enrollment_pending") as error:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")
    assert broker.calls == ["/api/agentic-new/credential-info", PREFIX + "enroll"]
    fingerprint = engine.agent_credential.key_thumbprint
    assert error.value.payload["key_thumbprint"] == fingerprint
    assert f"SHA256:{fingerprint}" in caplog.text
    assert "approve" not in caplog.text.lower()
    assert "sk-ds-test" not in caplog.text
    broker.approved = True
    headers = engine._headers(include_agent_token=True, path="/api/agentic-new/decide")
    assert headers["X-Agent-Token"] == TOKEN
    assert thumbprint(headers["DPoP"]) == fingerprint
    assert thumbprint(headers["DPoP"]) != process_dpop_key().thumbprint
    assert _verify(headers["DPoP"])["ath"] == _ath(TOKEN)


def test_restart_reuses_scoped_key_and_never_reuses_other_agent_or_workspace_key():
    first = credential(Broker())
    assert first.key_thumbprint == credential(Broker()).key_thumbprint
    second = credential(Broker(broker=config(agent_subject="agent:another")))
    third = credential(Broker(broker=config(workspace_id="workspace-two")))
    fourth = credential(Broker(broker=config(tenant_id="another-tenant")))
    assert len({first.key_thumbprint, second.key_thumbprint, third.key_thumbprint, fourth.key_thumbprint}) == 4
    assert len(list(Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).glob("*.json"))) == 4


def test_session_refresh_is_singleflight_and_reuses_enrollment():
    broker = Broker()
    cred = credential(broker)
    barrier = threading.Barrier(12)

    def get():
        barrier.wait(timeout=5)
        return cred.get_token()

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert list(pool.map(lambda _: get(), range(12))) == [TOKEN] * 12
    assert broker.calls.count(PREFIX + "enroll") == 1
    assert broker.calls.count(PREFIX + "challenge") == 1
    assert broker.calls.count(PREFIX + "session") == 1
    cred._cache._refresh_at = 0
    assert cred.get_token() == TOKEN
    assert broker.calls.count(PREFIX + "enroll") == 1
    assert broker.calls.count(PREFIX + "session") == 2


@pytest.mark.parametrize("change", [
    {"cnf": {"jkt": "another-key"}}, {"binding_id": "another-binding"},
    {"token_type": "Bearer"}, {"access_token": "some-external-jwt"},
    {"access_token": "dsrs_" + "a" * 42}, {"access_token": "dsrs_" + "a" * 44},
    {"expires_in": 0}, {"expires_in": True}, {"expires_in": 301},
    {"expires_in": "300"}, {"cnf": None},
])
def test_session_binding_and_lifetime_fail_closed_without_fallback(change, monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_DEV_AGENT_TOKEN", "never-use-this-fallback")
    broker = Broker()
    broker.session_changes = change
    cred = credential(broker)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_response_invalid"):
        cred.get_token()
    assert cred._cache.get() is None
    broker.session_changes = {}
    assert cred.get_token() == TOKEN


def test_broker_redirect_never_forwards_virtual_key_even_with_follow_redirects_client():
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://attacker.example/collect"})

    client = httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        credential(Broker(), client).get_token()
    assert calls == [ORIGIN + PREFIX + "enroll"]


@pytest.mark.parametrize("field,value", [
    ("enrollment_path", "https://other.example/enroll"),
    ("session_path", "//other.example/session"),
    ("challenge_path", PREFIX + "../challenge"),
    ("session_path", PREFIX + "session?redirect=evil"),
    ("tenant_id", ""), ("workspace_id", ""), ("runtime_kind", "automatic"),
])
def test_metadata_cannot_change_destinations_or_omit_scope(field, value):
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        credential(Broker(broker=config(**{field: value})))


def test_broker_fails_closed_for_http_cloud_gateway():
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        GatewayBrokerCredential(
            http=httpx.Client(), gateway_url="http://cloud.example", virtual_key="test",
            agent_subject="agent:math-agent", broker=config(),
        )


def test_pending_broker_does_not_become_optional_identity(shield_factory):
    shield = shield_factory(Broker(approved=False), base_url=ORIGIN, agent_name="math-agent")
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_enrollment_pending"):
        shield._agent_token()


def test_unknown_runtime_mode_cannot_fall_back_to_local_entra_or_dev_token(shield_factory, monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_DEV_AGENT_TOKEN", "never-use-this-fallback")
    shield = shield_factory(lambda _: httpx.Response(200, json={
        "runtime_auth_mode": "unknown-broker", "provider_type": "entra_agent_id", "agent_configured": False,
    }), base_url=ORIGIN, agent_name="math-agent")
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        shield.agentic.engine.agent_token(fail_closed=True)


def test_credential_discovery_redirect_never_forwards_workspace_key(shield_factory):
    destinations = []

    def handle(request):
        destinations.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://attacker.example/credential-info"})

    shield = shield_factory(handle, base_url=ORIGIN, agent_name="math-agent")
    shield._client.follow_redirects = True
    with pytest.raises(Exception) as error:
        shield.agentic.engine.agent_token(fail_closed=True)
    assert getattr(error.value, "code", "") == "invalid_gateway_response"
    assert destinations == [ORIGIN + "/api/agentic-new/credential-info"]


def test_explicit_broker_session_cannot_use_process_global_signer(shield_factory):
    engine = shield_factory(lambda _: httpx.Response(200), base_url=ORIGIN).agentic.engine
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        engine.workload_headers("POST", ORIGIN + "/mcp", token=TOKEN)


def test_request_proof_rejects_other_gateway_and_normalizes_https_default_port():
    cred = credential(Broker())
    cred.get_token()
    with pytest.raises(GovernanceConfigurationError, match="agent_workload_destination_invalid"):
        cred.proof_headers("POST", "https://attacker.example/mcp", TOKEN)
    proof = cred.proof_headers("POST", "https://gateway.example:443/mcp?session=abc", TOKEN)["DPoP"]
    assert _verify(proof)["htu"] == ORIGIN + "/mcp"


def test_session_capacity_retry_gets_new_challenge_nonce_and_proof(monkeypatch):
    broker = Broker()
    proofs = []
    nonces = []
    sleeps = []
    sessions = []
    monkeypatch.setattr(gateway_broker, "_sleep", sleeps.append)

    def handle(request):
        proof = _verify(request.headers["DPoP"])
        proofs.append(proof["jti"])
        if request.url.path.endswith("/session"):
            body = json.loads(request.content)
            assert proof["nonce"] == body["nonce"] == nonces[-1]
            assert proof["ath"] == _ath(nonces[-1])
            sessions.append(body)
            if len(sessions) == 1:
                return httpx.Response(429, headers={"Retry-After": "5"})
            return httpx.Response(200, json={
                "binding_id": "binding-one", "access_token": TOKEN,
                "token_type": "DPoP", "expires_in": 300, "cnf": {"jkt": broker.jkt},
            })
        response = broker(request)
        if request.url.path.endswith("/challenge"):
            data = response.json()
            data["nonce"] = f"fresh-nonce-{len(nonces)}"
            nonces.append(data["nonce"])
            return httpx.Response(200, json=data)
        return response

    cred = credential(broker, httpx.Client(transport=httpx.MockTransport(handle), timeout=30))
    assert cred.get_token() == TOKEN
    assert len(sessions) == 2 and len(set(nonces)) == 2
    assert len(proofs) == len(set(proofs))
    assert len(sleeps) == 1 and 5 <= sleeps[0] <= 5.25
    assert broker.calls.count(PREFIX + "enroll") == 1


def test_capacity_retry_respects_remaining_client_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(gateway_broker, "_sleep", lambda _: pytest.fail("retry exceeded HTTP timeout budget"))

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(429, headers={"Retry-After": "60"})

    cred = credential(Broker(), httpx.Client(transport=httpx.MockTransport(handle), timeout=0.1))
    with pytest.raises(Exception) as error:
        cred.get_token()
    assert getattr(error.value, "code", "") == "agent_runtime_capacity_exceeded"
    assert calls == [PREFIX + "enroll"]


def test_static_identity_dictionary_refuses_bound_session(shield_factory):
    shield = shield_factory(Broker(), base_url=ORIGIN, agent_name="math-agent")
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_request_proof_required"):
        shield.create_headers(identity=True)


def test_native_mcp_uses_the_session_credential_signer(shield_factory):
    broker = Broker()
    shield = shield_factory(broker, base_url=ORIGIN, agent_name="math-agent")
    hook = gateway_workload_hook(shield, ORIGIN + "/mcp", identity=True, headers={})
    request = httpx.Request("POST", ORIGIN + "/mcp")
    asyncio.run(hook(request))
    assert thumbprint(request.headers["DPoP"]) == shield.agentic.engine.agent_credential.key_thumbprint
    assert _verify(request.headers["DPoP"])["ath"] == _ath(TOKEN)


def test_two_agents_in_one_process_never_share_signer(shield_factory):
    one = shield_factory(Broker(), base_url=ORIGIN, agent_name="math-agent").agentic.engine
    two = shield_factory(Broker(broker=config(agent_subject="agent:other")), base_url=ORIGIN, agent_name="other").agentic.engine
    first = one._headers(include_agent_token=True, path="/api/agentic-new/decide")
    second = two._headers(include_agent_token=True, path="/api/agentic-new/decide")
    assert thumbprint(first["DPoP"]) != thumbprint(second["DPoP"])
    assert thumbprint(first["DPoP"]) == one.agent_credential.key_thumbprint
    assert thumbprint(second["DPoP"]) == two.agent_credential.key_thumbprint


@pytest.mark.parametrize("contents", [b"null", b"{}", b"not json", b'{"version":1,"storage":"file","private_key_pem":"corrupt"}'])
def test_corrupt_key_never_regenerates_or_overwrites(contents):
    credential(Broker())
    stored = next(Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).glob("*.json"))
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    stored.write_bytes(contents)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        credential(Broker())
    assert stored.read_bytes() == contents


def test_world_readable_key_refused():
    credential(Broker())
    stored = next(Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).glob("*.json"))
    stored.chmod(0o644)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        credential(Broker())


def test_managed_store_symlink_refused(tmp_path):
    credential(Broker())
    stored = next(Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).glob("*.json"))
    actual = tmp_path / "actual-key"
    stored.rename(actual)
    stored.symlink_to(actual)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        credential(Broker())


def test_explicit_mounted_key_is_read_only_and_supports_projected_symlinks(tmp_path, monkeypatch):
    key = DPoPKey()
    actual = tmp_path / "mounted.pem"
    actual.write_bytes(runtime_key._private_bytes(key))
    actual.chmod(0o400)
    link = tmp_path / "projected.pem"
    link.symlink_to(actual)
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KEY_FILE", str(link))
    assert credential(Broker()).key_thumbprint == key.thumbprint
    assert not Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).exists()
    actual.unlink()
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        credential(Broker())
    assert not actual.exists()


def test_existing_keyring_key_never_falls_back_to_new_file(monkeypatch):
    class Ring:
        value = None

        def get_password(self, *args):
            return self.value

        def set_password(self, service, name, value):
            self.value = value

    ring = Ring()
    monkeypatch.setattr(runtime_key, "_secure_keyring", lambda: ring)
    key = credential(Broker()).key_thumbprint
    assert credential(Broker()).key_thumbprint == key
    path = next(Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).glob("*.json"))
    before = path.read_bytes()
    assert b"PRIVATE KEY" not in before
    monkeypatch.setattr(runtime_key, "_secure_keyring", lambda: None)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_key_unavailable"):
        credential(Broker())
    assert path.read_bytes() == before


def test_cloud_configuration_can_be_resolved_by_enrollment_but_cannot_silently_switch(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "gcp")
    broker = Broker()
    broker.enrollment_changes = {"runtime_kind": "gcp", "runtime_audience": "runtime-audience"}
    broker.challenge_changes = {"runtime_kind": "aws", "runtime_audience": "runtime-audience"}
    cred = credential(broker)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        cred.get_token()
    assert PREFIX + "session" not in broker.calls


def test_original_math_agent_runs_unchanged_after_runtime_key_approval(tmp_path, shield_factory, monkeypatch, capsys):
    """Execute the user's actual graph after replacing only credential values.

    Secrets are removed with AST before writing/executing a temporary source
    file, so framework source scanning cannot read the original credentials.
    This does not edit or execute the original test2.py file.
    """
    pytest.importorskip("langgraph")
    from deepintshield import DeepintShield
    from deepintshield.agentic import registry
    from deepintshield.errors import get_exception_error_code

    source = Path(__file__).resolve().parents[2] / "test2.py"
    if not source.is_file():
        pytest.skip("workspace test2.py is not part of the published SDK")
    tree = ast.parse(source.read_text())
    safe_values = {
        "DEEPINTSHIELD_AGENT_NAME": "math-agent",
        "DEEPINTSHIELD_BASE_URL": ORIGIN,
        "DEEPINTSHIELD_VIRTUAL_KEY": "sk-ds-test",
        "DEEPINTSHIELD_REQUESTER": "",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os" and target.value.attr == "environ"
            ):
                name = target.slice.value if isinstance(target.slice, ast.Constant) else ""
                # Do not include unknown assignment values in assertion/error
                # output. Every environment assignment is sanitized first.
                node.value = ast.Constant(safe_values.get(name, ""))
                if name:
                    # runpy executes the assignment; restore the environment
                    # after this test without ever printing its original value.
                    monkeypatch.setenv(name, os.environ.get(name, ""))
    ast.fix_missing_locations(tree)
    sanitized = tmp_path / "math_agent_runtime_check.py"
    sanitized.write_text(ast.unparse(tree))
    broker = Broker(approved=False)
    executed = []
    clients = []

    def handle(request):
        path = request.url.path
        if path == "/api/agentic-security/blueprints":
            return httpx.Response(404)
        if path == "/api/agentic-new/credential-info" or path.startswith(PREFIX):
            return broker(request)
        if path.endswith("/registry/discover") and "X-Agent-Token" not in request.headers:
            if broker.approved:
                return httpx.Response(401, json={"error": "workload_proof_required"})
            return httpx.Response(202, json={
                "registration_state": "pending", "blueprint_scan_status": "complete",
            })
        broker(request)  # validate actual token, DPoP signature, key and URI
        if path.endswith("/registry/discover"):
            return httpx.Response(200, json={
                "network_id": "math-network", "agents_upserted": 1,
                "blueprint_scan_status": "complete", "registration_state": "active",
            })
        if path.endswith("/decide"):
            return httpx.Response(200, json={
                "verdict": "ALLOW", "allow": True, "proceed": True,
                "mode": "enforce", "would_block": False,
            })
        return httpx.Response(200, json={})

    def from_env(cls):
        client = shield_factory(handle, base_url=ORIGIN, agent_name="math-agent", requester="")
        clients.append(client)
        return client

    monkeypatch.setattr(DeepintShield, "from_env", classmethod(from_env))

    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == str(sanitized) and frame.f_code.co_name in {"add_two", "multiply_by_three"}:
            executed.append(frame.f_code.co_name)

    old_trace = sys.getprofile()
    try:
        sys.setprofile(trace)
        registry.reset_dedupe()
        with pytest.raises((RuntimeError, PermissionError)) as error:
            runpy.run_path(str(sanitized), run_name="__main__")
        assert get_exception_error_code(error.value) == "agent_runtime_enrollment_pending"
        assert executed == []
        pending_key = broker.jkt
        broker.approved = True
        registry.reset_dedupe()
        # A real rerun starts a new process. runpy retains the first script's
        # client through shield_factory, so close it before simulating that
        # restart; multiple live unbound clients correctly fail closed.
        for client in clients:
            client.close()
        result = runpy.run_path(str(sanitized), run_name="__main__")
        assert broker.jkt == pending_key
        assert executed == ["add_two", "multiply_by_three"]
        assert result["result"] == {"number": 21}
        assert "21" in capsys.readouterr().out
    finally:
        sys.setprofile(old_trace)
        registry.reset_dedupe()


def test_processes_share_one_durable_key_without_creation_races():
    script = (
        "from deepintshield.agentic.credentials import runtime_key; "
        "runtime_key._secure_keyring=lambda:None; "
        "print(runtime_key.load_runtime_key('https://gateway.example','tenant','workspace','agent:math').thumbprint)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    processes = [subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    ) for _ in range(4)]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        outputs.append(stdout.strip())
    assert len(set(outputs)) == 1
    assert len(outputs[0]) == 43
    assert len(list(Path(os.environ["DEEPINTSHIELD_RUNTIME_KEY_DIR"]).glob("*.json"))) == 1
