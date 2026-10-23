"""SDK-first agent registration: the registering developer, the pending message
and the proof-less re-sighting of a registration the gateway already holds.

Covers SDK 3.2.0's ``registered_by`` setting (kwarg / ``DEEPINTSHIELD_REGISTERED_BY``
/ e-mail-shaped ``requester`` fallback) and its ``registered_by_email`` discover
field with the ``registered_by_source`` beside it (``explicit`` for the setting,
``requester`` for the derived address, neither when there is no e-mail); the
gated VK-only fallback for a runtime key or host identity that is not ready
while the registration is known pending; the single developer-facing line
``Registration pending. Reference <id> · runtime key SHA256:<thumbprint>. Contact
your administrator. You will be notified at <email>.``, whose last sentence is
printed only for an address the gateway echoed; ``status()`` without review
links; and the marker-gated Entra host chain that fails fast with
``credential_host_identity_missing``.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
import types
from urllib.parse import parse_qs

import httpx
import pytest

from deepintshield import DeepintShield, ShieldConfig
from deepintshield.agentic import engine as engine_module
from deepintshield.agentic import enforcement, gate
from deepintshield.agentic import managed_execution as managed
from deepintshield.agentic import managed_runner, registry
from deepintshield.agentic.credentials import entra, runtime_key
from deepintshield.agentic.credentials.gateway_broker import GatewayBrokerCredential
from deepintshield.agentic.errors import (
    GatewayUnavailable,
    GovernanceConfigurationError,
    format_registration_message,
    public_agentic_error,
    public_agentic_error_details,
)
from deepintshield.agentic.types import VKCredentialInfo
from deepintshield.errors import DeepintShieldError, get_error_definition, get_exception_error_code

ORIGIN = "https://gateway.example"
PREFIX = "/api/agentic-new/runtime-auth/"
REGISTRATION_ID = "agnew-rr-0123456789abcdef"
# The one refusal over a registering e-mail, as the catalogue words it. The
# documentation quotes this sentence byte for byte.
REQUESTER_MISMATCH = (
    "Registration refused: this agent is already being registered under a different email. "
    "Contact your administrator."
)
# What the gateway itself says in that 403. The SDK never prints it.
GATEWAY_MISMATCH_PROSE = (
    "This agent is already being registered under a different email. Use the email it was first "
    "registered with, or ask your administrator to deny the pending registration."
)
JKT = base64.urlsafe_b64encode(hashlib.sha256(b"runtime-key").digest()).rstrip(b"=").decode()
AZURE_ENV = (
    "AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET", "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_USERNAME", "AZURE_PASSWORD", "AZURE_FEDERATED_TOKEN_FILE", "IDENTITY_ENDPOINT",
    "MSI_ENDPOINT", "IMDS_ENDPOINT", "AZURE_POD_IDENTITY_AUTHORITY_HOST",
)


@pytest.fixture(autouse=True)
def isolated_process_state(tmp_path, monkeypatch):
    """Once-per-process guards and runtime-key storage never leak between tests."""
    monkeypatch.setattr(registry, "_enrolment_logged", set())
    monkeypatch.setattr(engine_module, "_credential_failure_logged", set())
    for name in list(os.environ):
        if name.startswith("DEEPINTSHIELD_RUNTIME_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("DEEPINTSHIELD_REGISTERED_BY", raising=False)
    monkeypatch.delenv("DEEPINTSHIELD_REQUESTER", raising=False)
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KEY_DIR", str(tmp_path / "keys"))
    monkeypatch.setattr(runtime_key, "_secure_keyring", lambda: None)


def _json(request: httpx.Request) -> dict:
    return json.loads(request.content or b"{}")


def _jwk_thumbprint(jwk: dict) -> str:
    raw = json.dumps(jwk, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()


def _broker_config(agent_subject: str) -> dict:
    return {
        "enrollment_path": PREFIX + "enroll",
        "challenge_path": PREFIX + "challenge",
        "session_path": PREFIX + "session",
        "tenant_id": "platform-tenant",
        "workspace_id": "workspace-one",
        "agent_subject": agent_subject,
        "runtime_kind": "key",
    }


class ProvisionalGateway:
    """A gateway holding a pending registration with a provisional runtime key.

    credential-info answers 200 with key-only broker settings (no
    ``agent_identity_client_id``); enroll keeps the key pending; discover accepts
    only proof-less sightings and returns the 202 registration envelope.
    """

    def __init__(self, agent_subject: str = "agent:math-agent") -> None:
        self.agent_subject = agent_subject
        self.calls: list[tuple[str, str]] = []
        self.jkt = ""
        self.discover_bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((path, request.headers.get("X-Agent-Token", "")))
        if path == "/api/agentic-new/credential-info":
            return httpx.Response(200, headers={"Cache-Control": "no-store"}, json={
                "runtime_auth_mode": "gateway_broker",
                "runtime_broker": _broker_config(self.agent_subject),
                "provider_id": "idp-entra", "provider_type": "entra_agent_id",
                "tenant_id": "directory-tenant", "blueprint_client_id": "blueprint-client",
                "authority": "https://login.microsoftonline.com/directory-tenant/v2.0",
                "gateway_audience": "api://deepintshield", "scopes": ["tools:invoke"],
                "require_proof_of_possession": True, "agent_subject": self.agent_subject,
                "agent_configured": True, "credential_bootstrap": True, "provisional": True,
                "registration_id": REGISTRATION_ID, "registration_state": "pending",
            })
        if path == PREFIX + "enroll":
            body = _json(request)
            assert "d" not in body["public_jwk"]
            self.jkt = _jwk_thumbprint(body["public_jwk"])
            return httpx.Response(200, json={
                "binding_id": "binding-provisional", "key_thumbprint": self.jkt,
                "status": "pending", "enabled": False, "provisional": True,
                "registration_id": REGISTRATION_ID,
            })
        if path == "/api/agentic-new/registry/discover":
            assert "X-Agent-Token" not in request.headers, "a pending agent has no workload proof"
            self.discover_bodies.append(_json(request))
            return httpx.Response(202, json={
                "network_id": "network:math", "registration_state": "pending",
                "blueprint_scan_status": "complete",
                "registration": {
                    "id": REGISTRATION_ID, "agent_subject": self.agent_subject,
                    "status": "pending", "requester_email": "alice@corp.com",
                },
            })
        return httpx.Response(404)


MANIFEST = {
    "network": {"key": "math-agent", "name": "math-agent"},
    "agents": [{"key": "math-agent", "name": "math-agent"}],
    "tools": [],
    "edges": [],
}


# ── registered_by precedence and validation ───────────────────────────────


def test_registered_by_kwarg_wins_over_environment_and_requester(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_REGISTERED_BY", "env@corp.com")
    monkeypatch.setenv("DEEPINTSHIELD_REQUESTER", "requester@corp.com")
    monkeypatch.setenv("DEEPINTSHIELD_AGENT_NAME", "math-agent")

    with DeepintShield.from_env() as from_env:
        assert from_env.registered_by == "env@corp.com"
        assert from_env.agentic.engine.registered_by_email == "env@corp.com"
        assert from_env.agentic.engine.registered_by_source == "explicit"
    with DeepintShield(
        virtual_key="sk-ds-test", agent_name="math-agent",
        requester="requester@corp.com", registered_by=" Dev@Corp.com ",
    ) as explicit:
        assert explicit.registered_by == "dev@corp.com"
        assert explicit.agentic.engine.registered_by_email == "dev@corp.com"
        assert explicit.agentic.engine.registered_by_source == "explicit"
    with DeepintShield.from_config(
        ShieldConfig(virtual_key="sk-ds-test", agent_name="math-agent", registered_by="Cfg@Corp.com")
    ) as from_config:
        assert from_config.registered_by == "cfg@corp.com"


def test_registered_by_derives_from_an_email_shaped_requester():
    with DeepintShield(virtual_key="sk-ds-test", agent_name="math-agent", requester="Alice@Corp.com") as shield:
        assert shield.registered_by is None
        assert shield.agentic.engine.registered_by_email == "alice@corp.com"
        # Derived, so the gateway reads it as a hint rather than a claim.
        assert shield.agentic.engine.registered_by_source == "requester"
        # The acting-for identity is untouched by the registrant derivation.
        assert shield.agentic.engine._requester_hint == "Alice@Corp.com"


@pytest.mark.parametrize("requester", ["sdk-user", "alice", "service-account-1"])
def test_registered_by_is_empty_without_an_email_shaped_requester(requester):
    with DeepintShield(virtual_key="sk-ds-test", agent_name="math-agent", requester=requester) as shield:
        assert shield.registered_by is None
        assert shield.agentic.engine.registered_by_email == ""
        assert shield.agentic.engine.registered_by_source == ""


def test_registered_by_empty_string_disables_the_requester_fallback():
    with DeepintShield(
        virtual_key="sk-ds-test", agent_name="math-agent", requester="alice@corp.com", registered_by="",
    ) as shield:
        assert shield.registered_by == ""
        assert shield.agentic.engine.registered_by_email == ""
        assert shield.agentic.engine.registered_by_source == ""


def test_registered_by_naming_the_requester_is_still_explicit():
    # Stating the address is what makes it the registrant the gateway maps
    # to a directory identity and notifies, even when the registrant and the
    # acting-for identity are the same person.
    with DeepintShield(
        virtual_key="sk-ds-test", agent_name="math-agent",
        requester="alice@corp.com", registered_by="alice@corp.com",
    ) as shield:
        engine = shield.agentic.engine
        assert (engine.registered_by_email, engine.registered_by_source) == ("alice@corp.com", "explicit")
        # Both are a snapshot: a later change of the client cannot turn the
        # registrant of an already built engine into a different claim.
        shield.registered_by = None
        shield.requester = "bob@corp.com"
        assert (engine.registered_by_email, engine.registered_by_source) == ("alice@corp.com", "explicit")


@pytest.mark.parametrize("value", ["not-an-email", "a@b", "@corp.com", "two words@corp.com", "x" * 250 + "@c.io"])
def test_invalid_registered_by_is_a_configuration_error_at_construction(value):
    with pytest.raises(DeepintShieldError) as caught:
        DeepintShield(virtual_key="sk-ds-test", agent_name="math-agent", registered_by=value)
    assert caught.value.code == "configuration_error"
    assert caught.value.details == {"setting": "registered_by"}


def test_invalid_registered_by_environment_is_annotated(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_REGISTERED_BY", "bad")
    with pytest.raises(ValueError) as caught:
        ShieldConfig.from_env()
    assert get_exception_error_code(caught.value) == "configuration_error"
    assert caught.value.details == {"environment_variable": "DEEPINTSHIELD_REGISTERED_BY"}
    with pytest.raises(ValueError):
        DeepintShield.from_env()


# ── discover payload ──────────────────────────────────────────────────────


def _discovery_handler(bodies: list[dict], agent_subject: str = "agent:math-agent"):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/credential-info"):
            return httpx.Response(200, json={
                "provider_type": "", "agent_configured": False, "agent_subject": agent_subject,
            })
        if path.endswith("/identity/resolve"):
            return httpx.Response(200, json={"subject": "user:alice-corp-com"})
        if path.endswith("/registry/discover"):
            bodies.append(_json(request))
            return httpx.Response(200, json={"network_id": "network:math", "blueprint_scan_status": "complete"})
        return httpx.Response(404)
    return handler


def test_discover_payload_carries_registered_by_email_beside_principal(shield_factory):
    bodies: list[dict] = []
    engine = shield_factory(
        _discovery_handler(bodies), agent_name="math-agent", requester="alice@corp.com",
    ).agentic.engine

    result = registry.discover(engine, manifest=dict(MANIFEST), sync=True)

    assert "error" not in result
    assert len(bodies) == 1
    assert bodies[0]["registered_by_email"] == "alice@corp.com"
    # Derived from the requester: the gateway takes it as a hint, not a claim.
    assert bodies[0]["registered_by_source"] == "requester"
    assert bodies[0]["auto_provision"] is True
    assert bodies[0]["principal"]["kind"] == "user"
    assert "registered_by_email" not in bodies[0]["principal"]
    assert "registered_by_source" not in bodies[0]["principal"]


def test_principal_email_still_only_fills_principal(shield_factory):
    bodies: list[dict] = []
    engine = shield_factory(
        _discovery_handler(bodies), agent_name="math-agent", registered_by="dev@corp.com",
    ).agentic.engine

    registry.discover(engine, manifest=dict(MANIFEST), principal_email="End.User@Corp.com", sync=True)

    assert bodies[0]["principal"] == {"email": "end.user@corp.com"}
    assert bodies[0]["registered_by_email"] == "dev@corp.com"
    assert bodies[0]["registered_by_source"] == "explicit"


def test_discover_payload_omits_registered_by_email_when_unconfigured(shield_factory):
    bodies: list[dict] = []
    engine = shield_factory(_discovery_handler(bodies), agent_name="math-agent").agentic.engine

    registry.discover(engine, manifest=dict(MANIFEST), sync=True)

    assert len(bodies) == 1
    assert "registered_by_email" not in bodies[0]
    assert "registered_by_source" not in bodies[0]
    assert "registered_by" not in json.dumps(bodies[0])


def test_discover_payload_names_no_registrant_when_the_requester_fallback_is_disabled(shield_factory):
    bodies: list[dict] = []
    engine = shield_factory(
        _discovery_handler(bodies), agent_name="math-agent", requester="alice@corp.com", registered_by="",
    ).agentic.engine

    registry.discover(engine, manifest=dict(MANIFEST), sync=True)

    assert len(bodies) == 1
    # The acting-for identity still travels as the principal; nothing names a
    # registrant, so there is neither an address nor a source.
    assert bodies[0]["principal"]["kind"] == "user"
    assert "registered_by" not in json.dumps(bodies[0])


@pytest.mark.parametrize(
    ("carried", "source"),
    [
        # A source is sent only beside an address.
        ({"registered_by_source": "requester"}, None),
        ({"registered_by_source": "explicit"}, None),
        ({"registered_by_email": "  ", "registered_by_source": "explicit"}, None),
        ({"registered_by_email": None, "registered_by_source": "requester"}, None),
        # A caller-built manifest keeps its own pair, and only a known source.
        ({"registered_by_email": "relay@corp.com", "registered_by_source": "requester"}, "requester"),
        ({"registered_by_email": "relay@corp.com", "registered_by_source": "explicit"}, "explicit"),
        ({"registered_by_email": "relay@corp.com"}, None),
        ({"registered_by_email": "relay@corp.com", "registered_by_source": "directory"}, None),
        ({"registered_by_email": "relay@corp.com", "registered_by_source": ["requester"]}, None),
    ],
)
def test_caller_built_manifest_never_sends_a_source_without_an_address(shield_factory, carried, source):
    bodies: list[dict] = []
    engine = shield_factory(_discovery_handler(bodies), agent_name="math-agent").agentic.engine
    assert (engine.registered_by_email, engine.registered_by_source) == ("", "")

    registry.discover(engine, manifest={**MANIFEST, **carried}, sync=True)

    assert len(bodies) == 1
    sent = bodies[0]
    assert sent.get("registered_by_source") == source
    assert ("registered_by_source" in sent) is (source is not None)
    if source is not None:
        assert sent["registered_by_email"] == carried["registered_by_email"]


def test_client_registrant_replaces_a_caller_built_pair_as_a_pair(shield_factory):
    bodies: list[dict] = []
    engine = shield_factory(
        _discovery_handler(bodies), agent_name="math-agent", registered_by="dev@corp.com",
    ).agentic.engine

    registry.discover(
        engine,
        manifest={**MANIFEST, "registered_by_email": "relay@corp.com", "registered_by_source": "requester"},
        sync=True,
    )

    # The source always describes the address that is actually sent.
    assert bodies[0]["registered_by_email"] == "dev@corp.com"
    assert bodies[0]["registered_by_source"] == "explicit"


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (
            types.SimpleNamespace(registered_by_email="dev@corp.com", registered_by_source="explicit"),
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
        ),
        (
            types.SimpleNamespace(registered_by_email="alice@corp.com", registered_by_source="requester"),
            {"registered_by_email": "alice@corp.com", "registered_by_source": "requester"},
        ),
        # An engine that predates the source names its address deliberately;
        # the gateway reads an absent source the same way.
        (
            types.SimpleNamespace(registered_by_email="dev@corp.com"),
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
        ),
        (
            types.SimpleNamespace(registered_by_email="dev@corp.com", registered_by_source="directory"),
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
        ),
        # No address, no source - whatever the engine claims about the source.
        (types.SimpleNamespace(registered_by_email="", registered_by_source="explicit"), {}),
        (types.SimpleNamespace(registered_by_source="requester"), {}),
        (types.SimpleNamespace(), {}),
    ],
)
def test_attach_registered_by_sends_the_source_only_with_the_address(engine, expected):
    assert registry._attach_registered_by(engine, {}) == expected


@pytest.mark.parametrize(
    ("client_kwargs", "expected"),
    [
        (
            {"registered_by": "dev@corp.com"},
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
        ),
        (
            {"requester": "Alice@Corp.com"},
            {"registered_by_email": "alice@corp.com", "registered_by_source": "requester"},
        ),
        ({"requester": "alice@corp.com", "registered_by": ""}, {}),
        ({"requester": "service-account-1"}, {}),
        ({}, {}),
    ],
)
def test_minimal_bootstrap_manifest_names_the_registrant_and_its_source(shield_factory, client_kwargs, expected):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(403, json={"error": {"code": "agent_not_registered"}})
        assert request.url.path.endswith("/registry/discover")
        assert "X-Agent-Token" not in request.headers
        bodies.append(_json(request))
        return httpx.Response(202, json={
            "registration_state": "pending", "blueprint_scan_status": "complete",
            "registration": {"id": REGISTRATION_ID, "status": "pending"},
        })

    engine = shield_factory(handler, agent_name="math-agent", **client_kwargs).agentic.engine

    assert registry.ensure_registration_capture(engine, "read_account") is True
    assert len(bodies) == 1
    assert {key: value for key, value in bodies[0].items() if key.startswith("registered_by")} == expected
    assert "principal" not in bodies[0]
    assert engine._registration_reference["registration_id"] == REGISTRATION_ID
    assert engine._registration_known_pending() is True


# ── gated VK-only re-sighting ─────────────────────────────────────────────


def test_pending_registration_enrols_provisional_key_then_files_vk_only_sighting(shield_factory):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    engine = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", requester="alice@corp.com").agentic.engine
    # discover() names the registrant before it posts; send the same payload.
    payload = registry._attach_registered_by(engine, dict(MANIFEST))

    result = registry._post_discover(engine, payload, timeout=1.0)

    assert result["registration_state"] == "pending"
    assert "error" not in result
    assert [path for path, _token in gateway.calls] == [
        "/api/agentic-new/credential-info", PREFIX + "enroll", "/api/agentic-new/registry/discover",
    ]
    assert all(token == "" for _path, token in gateway.calls)
    assert isinstance(engine.agent_credential, GatewayBrokerCredential)
    assert engine._runtime_key_thumbprint == gateway.jkt == engine.agent_credential.key_thumbprint
    assert engine._registration_reference["registration_id"] == REGISTRATION_ID
    assert engine._registration_reference["state"] == "pending"
    assert engine._registration_reference["notify_email"] == "alice@corp.com"
    assert gateway.discover_bodies[0]["registered_by_email"] == "alice@corp.com"
    assert gateway.discover_bodies[0]["registered_by_source"] == "requester"


def test_known_pending_registration_falls_back_when_credential_exchange_fails(shield_factory):
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(200, json={
                "provider_type": "entra_agent_id", "agent_configured": True,
                "agent_subject": "agent:math-agent", "blueprint_client_id": "blueprint-client",
                "agent_identity_client_id": "child-agent-client",
            })
        assert request.url.path.endswith("/registry/discover")
        posts.append(request.headers.get("X-Agent-Token", ""))
        return httpx.Response(202, json={"registration_state": "pending", "blueprint_scan_status": "complete"})

    class Broken:
        def get_token(self):
            raise GatewayUnavailable(reason="private host detail", code="credential_exchange_failed")

    engine = shield_factory(handler, agent_name="math-agent").agentic.engine
    engine.set_agent_credential(Broken())
    engine._note_registration({"registration_state": "pending", "registration": {"id": REGISTRATION_ID}})

    result = registry._post_discover(engine, dict(MANIFEST), timeout=1.0)

    assert result["registration_state"] == "pending"
    assert posts == [""]


def test_unknown_registration_state_never_widens_the_fallback(shield_factory):
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(200, json={
                "provider_type": "entra_agent_id", "agent_configured": True,
                "agent_subject": "agent:math-agent", "blueprint_client_id": "blueprint-client",
                "agent_identity_client_id": "child-agent-client",
            })
        if request.url.path.endswith("/registry/discover"):
            posts.append(request.headers.get("X-Agent-Token", ""))
        return httpx.Response(202, json={"registration_state": "pending"})

    class Broken:
        def get_token(self):
            raise GatewayUnavailable(reason="private host detail", code="credential_exchange_failed")

    engine = shield_factory(handler, agent_name="math-agent").agentic.engine
    engine.set_agent_credential(Broken())

    result = registry._post_discover(engine, dict(MANIFEST), timeout=1.0)

    assert result == {"dispatched": True, "error": "credential_exchange_failed"}
    assert posts == []
    assert engine._registration_known_pending() is False


def test_bootstrap_retry_keeps_the_filed_sighting_when_the_runtime_key_is_pending(shield_factory):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    engine = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent").agentic.engine

    result = registry._post_discover(
        engine, dict(MANIFEST), timeout=1.0, bootstrap_without_workload_proof=True,
    )

    assert "error" not in result
    assert result["registration_state"] == "pending"
    assert result["registration"]["id"] == REGISTRATION_ID
    discover_posts = [path for path, _token in gateway.calls if path.endswith("/registry/discover")]
    assert discover_posts == ["/api/agentic-new/registry/discover"]
    assert [path for path, _token in gateway.calls] == [
        "/api/agentic-new/registry/discover", "/api/agentic-new/credential-info", PREFIX + "enroll",
    ]
    assert engine._runtime_key_thumbprint == gateway.jkt


def test_gateway_unavailable_during_proof_acquisition_is_not_widened(shield_factory):
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(409, json={
                "error": {"code": "agent_registration_pending"},
                "registration": {"id": REGISTRATION_ID, "status": "pending"},
            })
        posts += 1
        return httpx.Response(202, json={"registration_state": "pending"})

    class Outage:
        def get_token(self):
            raise GatewayUnavailable(reason="connect timeout")

    engine = shield_factory(handler, agent_name="math-agent").agentic.engine
    engine.set_agent_credential(Outage())
    engine._note_registration({"registration_state": "pending"})

    result = registry._post_discover(engine, dict(MANIFEST), timeout=1.0)

    assert result == {"dispatched": True, "error": "gateway_unavailable"}
    assert posts == 0


# ── the developer-facing message ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"reference": REGISTRATION_ID, "thumbprint": JKT, "email": "Alice@Corp.com"},
            f"Registration pending. Reference {REGISTRATION_ID} · runtime key SHA256:{JKT}. "
            "Contact your administrator. You will be notified at alice@corp.com.",
        ),
        (
            {"reference": REGISTRATION_ID, "email": "alice@corp.com"},
            f"Registration pending. Reference {REGISTRATION_ID}. Contact your administrator. "
            "You will be notified at alice@corp.com.",
        ),
        (
            {"reference": REGISTRATION_ID, "thumbprint": JKT},
            f"Registration pending. Reference {REGISTRATION_ID} · runtime key SHA256:{JKT}. "
            "Contact your administrator.",
        ),
        (
            {"thumbprint": JKT},
            f"Registration pending. Runtime key SHA256:{JKT}. Contact your administrator.",
        ),
        (
            {"reference": "bad reference/with spaces", "thumbprint": "short", "email": "not-an-email"},
            "Registration pending. Contact your administrator.",
        ),
        ({}, "Registration pending. Contact your administrator."),
    ],
)
def test_format_registration_message_renders_only_validated_fragments(kwargs, expected):
    assert format_registration_message("agent_registration_pending", **kwargs) == expected
    assert format_registration_message("agent_runtime_enrollment_pending", **kwargs) == expected


def test_format_registration_message_for_requester_mismatch_cites_only_the_reference():
    # The refusal names the registration when the gateway named one; it never
    # carries the runtime-key clause or the notification sentence of a pending
    # message: the recorded address belongs to whoever registered first.
    assert format_registration_message(
        "registration_requester_mismatch", reference=REGISTRATION_ID, thumbprint=JKT, email="alice@corp.com",
    ) == (
        "Registration refused: this agent is already being registered under a different email. "
        f"Reference {REGISTRATION_ID}. Contact your administrator."
    )
    assert format_registration_message("registration_requester_mismatch", thumbprint=JKT) == REQUESTER_MISMATCH
    assert format_registration_message("registration_requester_mismatch", email="alice@corp.com") == REQUESTER_MISMATCH
    assert format_registration_message("agent_not_registered", reference=REGISTRATION_ID) == (
        public_agentic_error_details("agent_not_registered")["message"]
    )


def test_requester_mismatch_catalogue_text_has_one_meaning():
    # The gateway maps a registering e-mail to an identity in the workspace
    # directory and notifies it. The address is not verified, so the one
    # refusal left is a registration already filed under another address, and
    # the sentence says that and nothing about keys or people.
    definition = get_error_definition("registration_requester_mismatch")
    assert definition is not None
    assert definition.description == REQUESTER_MISMATCH
    assert definition.action == "Contact your administrator."
    assert definition.retryable is False
    assert public_agentic_error_details("registration_requester_mismatch")["message"] == REQUESTER_MISMATCH
    assert str(public_agentic_error(GovernanceConfigurationError(code="registration_requester_mismatch"))) == (
        "registration_requester_mismatch: Registration refused: this agent is already being registered "
        "under a different email. Contact your administrator."
    )
    for retired in ("key", "owner", "creator", "member", "workspace", "issued", "match"):
        assert retired not in REQUESTER_MISMATCH.lower()


def test_format_registration_message_drops_an_echoed_address_with_control_characters():
    # An address-shaped echo may not smuggle terminal control sequences into
    # the developer's console; it is dropped like any other invalid fragment.
    assert format_registration_message(
        "agent_registration_pending", reference=REGISTRATION_ID, email="ali\x1b[31mce@corp.com",
    ) == f"Registration pending. Reference {REGISTRATION_ID}. Contact your administrator."


@pytest.mark.parametrize(
    ("client_kwargs", "echo", "notified"),
    [
        # No confirmation from the gateway: an older gateway, or a derived
        # address it set aside, sends no e-mail. The local setting, explicit
        # or derived, never fills the sentence.
        ({"requester": "alice@corp.com"}, {}, ""),
        ({"registered_by": "dev@corp.com"}, {}, ""),
        ({}, {}, ""),
        # The gateway names the address it recorded for this registration.
        ({"requester": "alice@corp.com"}, {"notify_email": "Alice@Corp.com"}, "alice@corp.com"),
        ({"registered_by": "dev@corp.com"}, {"notify_email": "dev@corp.com"}, "dev@corp.com"),
        ({}, {"notify_email": "owner@corp.com"}, "owner@corp.com"),
        # An echo that is not an address confirms nothing.
        ({"registered_by": "dev@corp.com"}, {"notify_email": "see the console"}, ""),
        ({"registered_by": "dev@corp.com"}, {"notify_email": True}, ""),
    ],
)
def test_public_error_from_pending_credential_info_notifies_only_a_confirmed_address(
    shield_factory, client_kwargs, echo, notified,
):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/credential-info")
        return httpx.Response(409, json={
            "error": {
                "code": "agent_registration_pending",
                "message": "PRIVATE: review the registration at /workspace/private",
            },
            "registration": {"id": "registration:first-run", "agent_subject": "agent:math-agent", "status": "pending"},
            "action": {"path": "/workspace/private", "surface": "console", "method": "GET"},
            "review_url": "/workspace/private?review_registration=1",
            **echo,
        })

    engine = shield_factory(handler, agent_name="math-agent", **client_kwargs).agentic.engine
    engine._note_runtime_key(JKT)

    with pytest.raises(GovernanceConfigurationError) as caught:
        engine.credential_info
    public = public_agentic_error(caught.value)

    # The reference and the runtime key are cited either way.
    pending = (
        "agent_registration_pending: Registration pending. Reference registration:first-run · "
        f"runtime key SHA256:{JKT}. Contact your administrator."
    )
    assert str(public) == (f"{pending} You will be notified at {notified}." if notified else pending)
    assert "review" not in str(public).lower()
    assert "PRIVATE" not in str(public)
    assert caught.value.payload["registration_context"] == {
        "registration_id": "registration:first-run",
        "key_thumbprint": JKT,
        "notify_email": notified,
        "state": "pending",
    }
    # The short negative cache re-raises with the same context.
    with pytest.raises(GovernanceConfigurationError) as cached:
        engine.credential_info
    assert str(public_agentic_error(cached.value)) == str(public)


def test_server_echo_wins_over_local_registration_context(shield_factory):
    server_jkt = base64.urlsafe_b64encode(hashlib.sha256(b"server-key").digest()).rstrip(b"=").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={
            "error": {"code": "agent_registration_pending"},
            "registration": {"id": "registration:local", "status": "pending"},
            "reference": "registration:echo",
            "runtime_key_thumbprint": server_jkt,
            "notify_email": "Owner@Corp.com",
            "contact_administrator": True,
            "decision_status": "recorded",
        })

    engine = shield_factory(handler, agent_name="math-agent", requester="alice@corp.com").agentic.engine
    engine._note_runtime_key(JKT)

    with pytest.raises(GovernanceConfigurationError) as caught:
        engine.credential_info

    assert str(public_agentic_error(caught.value)) == (
        "agent_registration_pending: Registration pending. Reference registration:echo · "
        f"runtime key SHA256:{server_jkt}. Contact your administrator. You will be notified at owner@corp.com."
    )
    # The address printed is the one the gateway will write to, not the local one.
    assert caught.value.payload["registration_context"]["notify_email"] == "owner@corp.com"
    assert engine.registered_by_email == "alice@corp.com"


def test_a_confirmed_address_is_remembered_for_its_registration_only(shield_factory):
    engine = shield_factory(
        lambda request: httpx.Response(404), agent_name="math-agent", registered_by="dev@corp.com",
    ).agentic.engine

    # Nothing confirmed yet: the configured address is not a notification target.
    assert engine._registration_context() == {
        "registration_id": "", "key_thumbprint": "", "notify_email": "", "state": "",
    }

    engine._note_registration({
        "registration_state": "pending",
        "registration": {"id": REGISTRATION_ID, "status": "pending", "requester_email": "Dev@Corp.com"},
    })
    assert engine._registration_context()["notify_email"] == "dev@corp.com"
    # A later body about the same registration that repeats no address keeps it.
    engine._note_registration({"registration": {"id": REGISTRATION_ID, "status": "pending"}})
    assert engine._registration_context({"registration_id": REGISTRATION_ID})["notify_email"] == "dev@corp.com"

    # A payload about another registration does not borrow the remembered address.
    other = "agnew-rr-fedcba9876543210"
    borrowed = engine._registration_context({"reference": other})
    assert (borrowed["registration_id"], borrowed["notify_email"]) == (other, "")
    # Once the gateway reports that other registration, the address is forgotten
    engine._note_registration({"registration": {"id": other, "status": "pending"}})
    assert engine._registration_reference == {"registration_id": other, "state": "pending"}
    assert engine._registration_context()["notify_email"] == ""
    # until the gateway confirms one for it.
    engine._note_registration({"reference": other, "notify_email": "owner@corp.com"})
    assert engine._registration_context()["notify_email"] == "owner@corp.com"
    assert engine.registered_by_email == "dev@corp.com"


def test_a_confirmed_address_follows_the_row_id_not_the_displayed_reference(shield_factory):
    engine = shield_factory(lambda request: httpx.Response(404), agent_name="math-agent").agentic.engine
    row = {"id": "registration:row", "status": "pending"}

    engine._note_registration({"registration": row, "reference": "REF-2026-0042", "notify_email": "dev@corp.com"})

    # A later body about the same row shows the gateway's reference and keeps
    # the address confirmed for that row, although it repeats no address.
    assert engine._registration_context({"registration": row, "reference": "REF-2026-0042"}) == {
        "registration_id": "REF-2026-0042", "key_thumbprint": "", "notify_email": "dev@corp.com", "state": "pending",
    }
    # The same reference text on another row confirms nothing for that row.
    assert engine._registration_context(
        {"registration": {"id": "registration:other", "status": "pending"}, "reference": "REF-2026-0042"},
    )["notify_email"] == ""


def test_one_registration_cites_one_reference_in_every_message(shield_factory):
    row = {"id": "registration:row", "agent_subject": "agent:math-agent", "status": "pending"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/credential-info")
        return httpx.Response(409, json={
            "error": {"code": "agent_registration_pending"},
            "registration": row, "reference": "REF-2026-0042", "notify_email": "dev@corp.com",
        })

    engine = shield_factory(handler, agent_name="math-agent").agentic.engine
    pending = (
        "agent_registration_pending: Registration pending. Reference REF-2026-0042. "
        "Contact your administrator. You will be notified at dev@corp.com."
    )

    with pytest.raises(GovernanceConfigurationError) as first:
        engine.credential_info
    assert str(public_agentic_error(first.value)) == pending
    # The short negative cache re-raises without a body. It cites the reference
    # the developer was just given, not the row id behind it.
    with pytest.raises(GovernanceConfigurationError) as cached:
        engine.credential_info
    assert str(public_agentic_error(cached.value)) == pending
    # So does any later message about the same row that carries no reference:
    # no body at all (the gate), the 202 sighting, the runtime broker's payload.
    for payload in (None, {"registration": row}, {"registration_id": "registration:row"}):
        assert engine._registration_context(payload)["registration_id"] == "REF-2026-0042"

    # Another registration never borrows it, and reporting one forgets it.
    other = {"id": "registration:other", "status": "pending"}
    borrowed = engine._registration_context({"registration": other})
    assert (borrowed["registration_id"], borrowed["notify_email"]) == ("registration:other", "")
    engine._note_registration({"registration": other})
    assert engine._registration_context()["registration_id"] == "registration:other"
    assert engine._registration_reference == {
        "registration_id": "registration:other", "state": "pending", "agent_subject": "agent:math-agent",
    }


def test_a_reference_equal_to_the_row_id_adds_nothing_to_the_registration_memory(shield_factory):
    engine = shield_factory(lambda request: httpx.Response(404), agent_name="math-agent").agentic.engine

    # The contract: the reference is the registration id. Nothing extra is kept.
    engine._note_registration({
        "registration": {"id": REGISTRATION_ID, "status": "pending"}, "reference": REGISTRATION_ID,
    })
    assert engine._registration_reference == {"registration_id": REGISTRATION_ID, "state": "pending"}
    assert engine._registration_context()["registration_id"] == REGISTRATION_ID

    # A gateway that shows another reference for the row, then the row id again.
    engine._note_registration({"registration": {"id": REGISTRATION_ID}, "reference": "REF-2026-0042"})
    assert engine._registration_context()["registration_id"] == "REF-2026-0042"
    engine._note_registration({"registration": {"id": REGISTRATION_ID}, "reference": REGISTRATION_ID})
    assert engine._registration_reference == {"registration_id": REGISTRATION_ID, "state": "pending"}
    assert engine._registration_context()["registration_id"] == REGISTRATION_ID


@pytest.mark.parametrize(
    ("client_kwargs", "source", "recorded", "notified"),
    [
        # An explicit registrant the gateway recorded.
        ({"registered_by": "dev@corp.com"}, "explicit", "dev@corp.com", " You will be notified at dev@corp.com."),
        # A requester the gateway accepted as the registrant.
        ({"requester": "alice@corp.com"}, "requester", "alice@corp.com", " You will be notified at alice@corp.com."),
        # A requester the gateway set aside - an end user the organization has
        # not placed in the workspace. The registration is filed without a
        # registrant: no refusal, no e-mail.
        ({"requester": "end.user@customer.example"}, "requester", "", ""),
        # A gateway that predates the field records nothing and sends nothing.
        ({"registered_by": "dev@corp.com"}, "explicit", "", ""),
    ],
)
def test_first_run_notifies_only_the_registrant_the_gateway_recorded(
    shield_factory, caplog, client_kwargs, source, recorded, notified,
):
    sightings: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        registration = {"id": REGISTRATION_ID, "agent_subject": "agent:math-agent", "status": "pending"}
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(409, json={
                "error": {"code": "agent_registration_pending"}, "registration": registration,
            })
        assert request.url.path.endswith("/registry/discover")
        assert "X-Agent-Token" not in request.headers
        sightings.append(_json(request))
        if recorded:
            registration["requester_email"] = recorded
        return httpx.Response(202, json={
            "registration_state": "pending", "blueprint_scan_status": "complete", "registration": registration,
        })

    engine = shield_factory(handler, agent_name="math-agent", **client_kwargs).agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # The first gated call files the sighting, then stops at credential lookup.
    gate.preflight(engine, "read_account")
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")

    assert len(sightings) == 1
    assert sightings[0]["registered_by_email"] == engine.registered_by_email
    assert sightings[0]["registered_by_source"] == source
    pending = (
        f"agent_registration_pending: Registration pending. Reference {REGISTRATION_ID}. "
        f"Contact your administrator.{notified}"
    )
    assert caught.value.code == "agent_registration_pending"
    assert str(public_agentic_error(caught.value)) == pending
    # The exception text and the single logged line agree.
    assert caplog.messages == [f"DeepintShield: {pending}"]


def test_pending_runtime_key_public_error_cites_the_provisional_registration(shield_factory):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    engine = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com").agentic.engine

    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")

    assert caught.value.code == "agent_runtime_enrollment_pending"
    assert caught.value.payload["key_thumbprint"] == gateway.jkt
    assert caught.value.payload["binding_id"] == "binding-provisional"
    assert caught.value.payload["registration_id"] == REGISTRATION_ID
    assert caught.value.payload["provisional"] is True
    # The enrolment response names the registration and the key. It confirms
    # no address, so the configured one is not promised a notification.
    assert str(public_agentic_error(caught.value)) == (
        f"agent_runtime_enrollment_pending: Registration pending. Reference {REGISTRATION_ID} · "
        f"runtime key SHA256:{gateway.jkt}. Contact your administrator."
    )


def test_pending_runtime_key_public_error_notifies_the_address_the_sighting_confirmed(shield_factory):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    engine = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com").agentic.engine
    payload = registry._attach_registered_by(engine, dict(MANIFEST))

    # First-run order: the proof-less sighting is filed and its 202 body names
    # the requester the gateway recorded; only then does the gated call stop.
    filed = registry._post_discover(engine, payload, timeout=1.0, bootstrap_without_workload_proof=True)
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")

    assert "error" not in filed
    assert gateway.discover_bodies[0]["registered_by_email"] == "dev@corp.com"
    assert gateway.discover_bodies[0]["registered_by_source"] == "explicit"
    assert caught.value.code == "agent_runtime_enrollment_pending"
    # The address is the gateway's (this fake records alice), not the local one.
    assert str(public_agentic_error(caught.value)) == (
        f"agent_runtime_enrollment_pending: Registration pending. Reference {REGISTRATION_ID} · "
        f"runtime key SHA256:{gateway.jkt}. Contact your administrator. You will be notified at alice@corp.com."
    )


def test_rerun_with_a_pending_runtime_key_logs_the_line_the_error_carries(shield_factory, caplog):
    pytest.importorskip("cryptography")
    # A later run of a registration that is still pending: the report asks for
    # workload proof first, the key is still pending, and only the proof-less
    # sighting that follows names the address the gateway will notify.
    gateway = ProvisionalGateway()
    engine = shield_factory(gateway, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com").agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    filed = registry.discover(engine, manifest=dict(MANIFEST), sync=True)
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")

    assert "error" not in filed
    assert [path for path, _token in gateway.calls][:3] == [
        "/api/agentic-new/credential-info", PREFIX + "enroll", "/api/agentic-new/registry/discover",
    ]
    pending = (
        f"agent_runtime_enrollment_pending: Registration pending. Reference {REGISTRATION_ID} · "
        f"runtime key SHA256:{gateway.jkt}. Contact your administrator. You will be notified at alice@corp.com."
    )
    assert str(public_agentic_error(caught.value)) == pending
    # One line for the process, and it says what the error says.
    assert caplog.messages == [f"DeepintShield: {pending}"]
    assert registry._enrolment_log_deferred.get() is False


def test_rerun_of_a_pending_registration_logs_one_line_after_its_sighting(shield_factory, caplog):
    calls: list[str] = []
    registration = {"id": REGISTRATION_ID, "agent_subject": "agent:math-agent", "status": "pending"}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path.rsplit("/", 1)[-1])
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(409, json={
                "error": {"code": "agent_registration_pending"}, "registration": registration,
            })
        assert request.url.path.endswith("/registry/discover")
        assert "X-Agent-Token" not in request.headers
        return httpx.Response(202, json={
            "registration_state": "pending", "blueprint_scan_status": "complete",
            "registration": {**registration, "requester_email": "dev@corp.com"},
        })

    engine = shield_factory(handler, agent_name="math-agent", registered_by="dev@corp.com").agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    filed = registry.discover(engine, manifest=dict(MANIFEST), sync=True)

    assert "error" not in filed
    assert calls == ["credential-info", "discover"]
    pending = (
        f"agent_registration_pending: Registration pending. Reference {REGISTRATION_ID}. "
        "Contact your administrator. You will be notified at dev@corp.com."
    )
    assert caplog.messages == [f"DeepintShield: {pending}"]
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")
    assert str(public_agentic_error(caught.value)) == pending
    assert caplog.messages == [f"DeepintShield: {pending}"]


@pytest.mark.parametrize("outcome", ["no_sighting", "sighting_refused", "sighting_unreachable"])
def test_a_registration_state_is_logged_once_whatever_happens_to_the_sighting(shield_factory, caplog, outcome):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/agentic-new/credential-info" and outcome == "no_sighting":
            # An approved, identity-backed agent whose new key awaits
            # authorization: nothing says the registration is pending, so the
            # report stops instead of falling back to a proof-less sighting.
            body = gateway(request).json()
            for key in ("credential_bootstrap", "provisional", "registration_id", "registration_state"):
                body.pop(key)
            return httpx.Response(200, json=body)
        if path == PREFIX + "enroll" and outcome == "no_sighting":
            body = gateway(request).json()
            body.pop("registration_id")
            body["provisional"] = False
            return httpx.Response(200, json=body)
        if path == "/api/agentic-new/registry/discover":
            posts.append(path)
            if outcome == "sighting_refused":
                return httpx.Response(503, json={"error": {"code": "registry_unavailable"}})
            if outcome == "sighting_unreachable":
                raise httpx.ConnectError("connection refused")
        return gateway(request)

    engine = shield_factory(handler, base_url=ORIGIN, agent_name="math-agent", registered_by="dev@corp.com").agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    result = registry._post_discover(engine, dict(MANIFEST), timeout=1.0)

    assert result["dispatched"] is True
    assert result["error"] == {
        "no_sighting": "agent_runtime_enrollment_pending",
        "sighting_refused": "registry_unavailable",
        "sighting_unreachable": "registry_discovery_unavailable",
    }[outcome]
    assert len(posts) == (0 if outcome == "no_sighting" else 1)
    reference = "" if outcome == "no_sighting" else f"Reference {REGISTRATION_ID} · runtime key "
    assert caplog.messages == [
        "DeepintShield: agent_runtime_enrollment_pending: Registration pending. "
        f"{reference or 'Runtime key '}SHA256:{gateway.jkt}. Contact your administrator."
    ]
    # The hold lasts for the proof request only.
    assert registry._enrolment_log_deferred.get() is False
    registry._log_enrolment_state({"error": "agent_registration_pending", "reference": REGISTRATION_ID})
    assert len(caplog.messages) == 2


def test_broker_pending_payload_drops_an_unbounded_registration_id(shield_factory):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    original = gateway.__call__

    def handler(request: httpx.Request) -> httpx.Response:
        response = original(request)
        if request.url.path == PREFIX + "enroll":
            body = response.json()
            body["registration_id"] = "x" * 65
            return httpx.Response(200, json=body)
        return response

    engine = shield_factory(handler, base_url=ORIGIN, agent_name="math-agent").agentic.engine
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")
    assert caught.value.payload["registration_id"] == ""
    assert caught.value.payload["key_thumbprint"] == gateway.jkt


def test_requester_mismatch_stops_the_first_gated_call_with_its_own_code(shield_factory, caplog):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        assert request.url.path.endswith("/registry/discover"), "a refused registrant triggers no credential lookup"
        assert "X-Agent-Token" not in request.headers
        assert _json(request)["registered_by_email"] == "someone.else@corp.com"
        # The agent is already being registered under another address. Only
        # an explicit registrant is refused for that; the gateway sets a
        # requester-derived one aside instead.
        assert _json(request)["registered_by_source"] == "explicit"
        return httpx.Response(403, json={"error": {
            "code": "registration_requester_mismatch",
            "message": f"PRIVATE: {GATEWAY_MISMATCH_PROSE}",
        }})

    engine = shield_factory(
        handler, agent_name="math-agent", registered_by="someone.else@corp.com",
    ).agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    with pytest.raises(GovernanceConfigurationError) as caught:
        gate.preflight(engine, "read_account")

    refusal = (
        "registration_requester_mismatch: Registration refused: this agent is already being registered "
        "under a different email. Contact your administrator."
    )
    assert refusal == f"registration_requester_mismatch: {REQUESTER_MISMATCH}"
    assert caught.value.code == "registration_requester_mismatch"
    assert str(public_agentic_error(caught.value)) == refusal
    assert "PRIVATE" not in str(public_agentic_error(caught.value))
    # The gateway's own sentence is never printed, however close it is.
    assert "first registered with" not in str(public_agentic_error(caught.value))
    # Refused once: no retry through the workload-proof path.
    assert requests == ["/api/agentic-new/registry/discover"]
    assert caplog.messages == [f"DeepintShield: {refusal}"]
    assert engine._registration_known_pending() is False


def test_requester_mismatch_on_a_known_registration_cites_its_reference(shield_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/credential-info"):
            return httpx.Response(409, json={
                "error": {"code": "agent_registration_pending"},
                "registration": {"id": REGISTRATION_ID, "status": "pending"},
            })
        assert request.url.path.endswith("/registry/discover")
        return httpx.Response(403, json={"error": {"code": "registration_requester_mismatch"}})

    engine = shield_factory(
        handler, agent_name="math-agent", registered_by="someone.else@corp.com",
    ).agentic.engine

    with pytest.raises(GovernanceConfigurationError) as caught:
        enforcement.report_topology(engine, dict(MANIFEST), sync=True, required=True)

    assert caught.value.code == "registration_requester_mismatch"
    assert str(public_agentic_error(caught.value)) == (
        "registration_requester_mismatch: Registration refused: this agent is already being registered "
        f"under a different email. Reference {REGISTRATION_ID}. Contact your administrator."
    )


def test_a_refused_second_address_is_not_told_who_registered_first(shield_factory, caplog):
    # One registration keeps its first address. Whatever the gateway puts in
    # its refusal, the developer gets the SDK's own sentence: no gateway prose,
    # no review link, and not the address recorded for whoever registered
    # first. The sighting is sent once, as configured, and is not repeated
    # without the address to get past the refusal.
    sightings: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/registry/discover")
        sightings.append(_json(request))
        return httpx.Response(403, json={
            "error": {"code": "registration_requester_mismatch", "message": GATEWAY_MISMATCH_PROSE},
            "notify_email": "first@corp.com",
            "review_url": "/workspace/agentic-new/registry?review_registration=private",
        })

    engine = shield_factory(handler, agent_name="math-agent", registered_by="second@corp.com").agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    with pytest.raises(GovernanceConfigurationError) as caught:
        gate.preflight(engine, "read_account")

    refusal = f"registration_requester_mismatch: {REQUESTER_MISMATCH}"
    assert str(public_agentic_error(caught.value)) == refusal
    assert caplog.messages == [f"DeepintShield: {refusal}"]
    assert [(sent["registered_by_email"], sent["registered_by_source"]) for sent in sightings] == [
        ("second@corp.com", "explicit"),
    ]
    assert "first@corp.com" not in caplog.text
    assert "review" not in caplog.text.lower()
    # Nothing from the refusal is kept for a later message either.
    assert engine._registration_context()["notify_email"] == ""


def test_runtime_binding_identity_missing_is_surfaced_from_the_broker(shield_factory):
    pytest.importorskip("cryptography")
    gateway = ProvisionalGateway()
    original = gateway.__call__

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == PREFIX + "enroll":
            body = _json(request)
            return httpx.Response(200, json={
                "binding_id": "binding-approved", "key_thumbprint": _jwk_thumbprint(body["public_jwk"]),
                "status": "approved", "enabled": True,
            })
        if request.url.path == PREFIX + "challenge":
            return httpx.Response(409, json={"error": {"code": "runtime_binding_identity_missing"}})
        return original(request)

    engine = shield_factory(handler, base_url=ORIGIN, agent_name="math-agent").agentic.engine
    with pytest.raises(GovernanceConfigurationError) as caught:
        engine._headers(include_agent_token=True, path="/api/agentic-new/decide")
    assert caught.value.code == "runtime_binding_identity_missing"
    assert str(public_agentic_error(caught.value)) == (
        "runtime_binding_identity_missing: The runtime key is bound to an agent whose identity "
        "has not been declared. Contact your administrator."
    )


# ── status() ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", ["live", "pending", "approved", "denied", "quarantined", "not_registered", "future_state"])
def test_status_strips_review_links_and_console_paths_for_every_state(shield_factory, state):
    body = {
        "state": state,
        "registration_id": "registration:test",
        "agent_subject": "agent:test",
        "reason": "private provider instructions",
        "action": {"path": "/workspace/private", "surface": "console", "method": "GET", "text": "Registration pending."},
        "review_url": "/workspace/agentic-new/registry?review_registration=private",
    }
    shield = shield_factory(lambda request: httpx.Response(200, json=body), requester="alice@corp.com")

    result = shield.agentic.status()

    assert result["state"] == state
    assert result["registration_id"] == "registration:test"
    assert "review_url" not in result
    assert "review" not in json.dumps(result).lower()
    assert "/workspace/" not in json.dumps(result)
    if state == "live":
        assert result["action"] == {"surface": "console", "method": "GET", "text": "Registration pending."}
    elif state == "pending":
        # The body confirms no address, so the requester-derived one is not
        # promised a notification.
        assert result["reason"] == "Registration pending. Reference registration:test. Contact your administrator."
        assert result["action"] == "Contact your administrator."
    else:
        assert isinstance(result["action"], str)
        assert "private provider instructions" not in result["reason"]


@pytest.mark.parametrize(
    ("client_kwargs", "echo", "notified"),
    [
        ({"requester": "alice@corp.com"}, {"notify_email": "alice@corp.com"}, "alice@corp.com"),
        ({"registered_by": "dev@corp.com"}, {"notify_email": "Dev@Corp.com"}, "dev@corp.com"),
        # The recorded address may belong to a colleague who filed the registration.
        ({"registered_by": ""}, {"notify_email": "owner@corp.com"}, "owner@corp.com"),
        ({"registered_by": "dev@corp.com"}, {}, ""),
        ({"requester": "alice@corp.com"}, {"notify_email": ""}, ""),
    ],
)
def test_status_pending_reason_notifies_only_the_address_the_gateway_names(
    shield_factory, client_kwargs, echo, notified,
):
    body = {"state": "pending", "registration_id": "registration:test", "agent_subject": "agent:test", **echo}
    shield = shield_factory(lambda request: httpx.Response(200, json=body), **client_kwargs)

    result = shield.agentic.status()

    pending = "Registration pending. Reference registration:test. Contact your administrator."
    assert result["reason"] == (f"{pending} You will be notified at {notified}." if notified else pending)
    assert result["action"] == "Contact your administrator."
    assert result["registration_id"] == "registration:test"


def test_status_pending_reason_includes_a_remembered_runtime_key(shield_factory):
    body = {"state": "pending", "registration_id": "registration:test", "agent_subject": "agent:test"}
    shield = shield_factory(lambda request: httpx.Response(200, json=body), registered_by="")
    shield.agentic.engine._note_runtime_key(JKT)

    result = shield.agentic.status()

    assert result["reason"] == (
        f"Registration pending. Reference registration:test · runtime key SHA256:{JKT}. Contact your administrator."
    )
    assert shield.agentic.engine._registration_reference == {
        "registration_id": "registration:test", "state": "pending", "agent_subject": "agent:test",
    }


# ── once-per-process logging ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("registration", "notified"),
    [
        # The 202 body names the requester the gateway recorded.
        ({"id": REGISTRATION_ID, "requester_email": "alice@corp.com"}, " You will be notified at alice@corp.com."),
        # No confirmation: the requester-derived address is not promised one.
        ({"id": REGISTRATION_ID}, ""),
    ],
)
def test_enrolment_line_is_logged_once_per_code_with_thumbprint_and_no_approval_hint(
    shield_factory, caplog, registration, notified,
):
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")
    engine = shield_factory(lambda request: httpx.Response(404), agent_name="math-agent", requester="alice@corp.com").agentic.engine
    engine._note_runtime_key(JKT)
    result = {
        "error": "agent_runtime_enrollment_pending",
        "registration": registration,
        "review_url": "/workspace/PRIVATE",
        "message": "PRIVATE: approve the fingerprint",
    }

    registry._log_enrolment_state(result, engine)
    registry._log_enrolment_state(result, engine)
    registry._log_enrolment_state({**result, "error": "agent_registration_pending"}, engine)

    assert caplog.messages == [
        "DeepintShield: agent_runtime_enrollment_pending: Registration pending. "
        f"Reference {REGISTRATION_ID} · runtime key SHA256:{JKT}. Contact your administrator.{notified}",
        "DeepintShield: agent_registration_pending: Registration pending. "
        f"Reference {REGISTRATION_ID} · runtime key SHA256:{JKT}. Contact your administrator.{notified}",
    ]
    assert "SHA256:" in caplog.text
    assert "approve" not in caplog.text.lower()
    assert "Review:" not in caplog.text
    assert "PRIVATE" not in caplog.text


def test_enrolment_line_without_an_engine_reads_only_the_gateway_body(caplog):
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    registry._log_enrolment_state({
        "error": "agent_registration_pending",
        "reference": REGISTRATION_ID,
        "runtime_key_thumbprint": JKT,
        "notify_email": "alice@corp.com",
    })
    registry._log_enrolment_state({
        "error": "agent_runtime_enrollment_pending",
        "registration_id": REGISTRATION_ID,
        "key_thumbprint": JKT,
    })

    assert caplog.messages == [
        "DeepintShield: agent_registration_pending: Registration pending. "
        f"Reference {REGISTRATION_ID} · runtime key SHA256:{JKT}. Contact your administrator. "
        "You will be notified at alice@corp.com.",
        "DeepintShield: agent_runtime_enrollment_pending: Registration pending. "
        f"Reference {REGISTRATION_ID} · runtime key SHA256:{JKT}. Contact your administrator.",
    ]


@pytest.mark.parametrize(
    ("echo", "notified"),
    [
        ({"notify_email": "dev@corp.com"}, " You will be notified at dev@corp.com."),
        # A gateway that predates the field confirms nothing and sends nothing.
        ({}, ""),
    ],
)
def test_best_effort_transport_reports_a_pending_registration_with_one_line(shield_factory, caplog, echo, notified):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(409, json={
            "error": {"code": "agent_registration_pending", "message": "PRIVATE: approve it in the console"},
            "registration": {"id": REGISTRATION_ID, "status": "pending"},
            **echo,
        })

    engine = shield_factory(handler, agent_name="math-agent", registered_by="dev@corp.com").agentic.engine
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")

    # Transparent traffic asks for the agent token on every request.
    assert engine.agent_token() is None
    assert engine.agent_token() is None

    assert caplog.messages == [
        "DeepintShield: agent_registration_pending: Registration pending. "
        f"Reference {REGISTRATION_ID}. Contact your administrator.{notified}"
    ]
    assert "PRIVATE" not in caplog.text
    assert set(requests) == {"/api/agentic-new/credential-info"}


def test_requester_mismatch_is_logged_with_the_refusal_lead(caplog):
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic")
    registry._log_enrolment_state({
        "error": "registration_requester_mismatch",
        "registration": {"id": REGISTRATION_ID},
        "notify_email": "alice@corp.com",
    })
    assert caplog.messages == [
        "DeepintShield: registration_requester_mismatch: Registration refused: this agent is already being "
        f"registered under a different email. Reference {REGISTRATION_ID}. Contact your administrator."
    ]


# ── Entra host identity chain ─────────────────────────────────────────────


class _FakeToken:
    def __init__(self, token: str) -> None:
        self.token = token


@pytest.fixture
def azure_stub(monkeypatch):
    for name in AZURE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPINTSHIELD_ENTRA_SKIP_IMDS_PROBE", "1")
    monkeypatch.setattr(entra, "_imds_probe_result", None)
    record = types.SimpleNamespace(constructed=[], chain_calls=[], chain_error=None, members=[])

    class Member:
        def __init__(self, *args, **kwargs):
            record.constructed.append((type(self).__name__, kwargs))

    class ChainedTokenCredential:
        def __init__(self, *members):
            record.constructed.append(("ChainedTokenCredential", {}))
            record.members = [type(member).__name__ for member in members]

        def get_token(self, scope):
            record.chain_calls.append(scope)
            if record.chain_error is not None:
                raise record.chain_error
            return _FakeToken("managed-identity-assertion")

    module = types.ModuleType("azure.identity")
    module.ChainedTokenCredential = ChainedTokenCredential
    module.EnvironmentCredential = type("EnvironmentCredential", (Member,), {})
    module.WorkloadIdentityCredential = type("WorkloadIdentityCredential", (Member,), {})
    module.ManagedIdentityCredential = type("ManagedIdentityCredential", (Member,), {})
    monkeypatch.setitem(sys.modules, "azure.identity", module)
    return record


def _exchange_handler(exchanges: list[dict]):
    def exchange(request: httpx.Request) -> httpx.Response:
        form = {key: values[0] for key, values in parse_qs(request.content.decode("utf-8")).items()}
        exchanges.append(form)
        if len(exchanges) == 1:
            assert form["client_assertion"] == "managed-identity-assertion"
            assert form["fmi_path"] == "child-agent-client"
            return httpx.Response(200, json={"access_token": "intermediate-t1"})
        assert form["client_assertion"] == "intermediate-t1"
        return httpx.Response(200, json={"access_token": "final-child-resource-token", "expires_in": 3600})
    return exchange


def _entra_credential(exchange) -> entra.EntraAgentCredential:
    return entra.EntraAgentCredential(
        authority="https://login.microsoftonline.com/tenant/v2.0",
        blueprint_client_id="blueprint-client",
        agent_identity_client_id="child-agent-client",
        gateway_audience="api://deepintshield-gateway",
        scopes=["tools:invoke"],
        http_client=httpx.Client(transport=httpx.MockTransport(exchange)),
    )


def test_entra_without_host_markers_fails_fast_once_and_logs_once(azure_stub, shield_factory, caplog):
    exchanges: list[dict] = []
    credential = _entra_credential(_exchange_handler(exchanges))
    engine = shield_factory(lambda request: httpx.Response(404), agent_name="math-agent").agentic.engine
    engine.set_agent_credential(credential)
    caplog.set_level(logging.WARNING, logger="deepintshield.agentic.engine")

    with pytest.raises(GovernanceConfigurationError) as first:
        credential.get_token()
    assert first.value.code == "credential_host_identity_missing"
    assert azure_stub.constructed == []
    # Memoised for the credential's lifetime: no chain, no exchange.
    with pytest.raises(GovernanceConfigurationError) as second:
        credential.get_token()
    assert second.value.code == "credential_host_identity_missing"
    assert azure_stub.constructed == []
    assert exchanges == []

    assert engine.agent_token() is None
    assert engine.agent_token() is None
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "credential_host_identity_missing" in messages[0]
    assert "No Azure host identity is available to this process." in messages[0]


def test_entra_workload_identity_markers_build_only_their_chain_members(azure_stub, monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("federated")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("AZURE_CLIENT_ID", "workload-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    exchanges: list[dict] = []
    credential = _entra_credential(_exchange_handler(exchanges))

    assert entra._host_identity_markers() == ("workload_identity",)
    assert credential.get_token() == "final-child-resource-token"
    assert credential.get_token() == "final-child-resource-token"
    assert azure_stub.members == ["WorkloadIdentityCredential"]
    assert azure_stub.chain_calls == ["api://AzureADTokenExchange/.default"]
    assert len(exchanges) == 2

    monkeypatch.setenv("IDENTITY_ENDPOINT", "http://169.254.169.254/metadata/identity")
    assert entra._host_identity_markers() == ("workload_identity", "managed_identity")
    azure_stub.constructed.clear()
    fresh = _entra_credential(_exchange_handler([]))
    fresh.get_token()
    assert azure_stub.members == ["WorkloadIdentityCredential", "ManagedIdentityCredential"]
    assert ("ManagedIdentityCredential", {"client_id": "workload-client"}) in azure_stub.constructed


def test_entra_environment_markers_require_a_secret_or_certificate(azure_stub, monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "sp-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    assert entra._host_identity_markers() == ()
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret")
    assert entra._host_identity_markers() == ("environment",)


def test_entra_chain_failure_is_memoised_for_the_window(azure_stub, monkeypatch):
    monkeypatch.setenv("MSI_ENDPOINT", "http://169.254.169.254/metadata/identity")
    azure_stub.chain_error = RuntimeError("PRIVATE Azure chain diagnostics")
    exchanges: list[dict] = []
    credential = _entra_credential(_exchange_handler(exchanges))
    clock = types.SimpleNamespace(now=1000.0)
    # Replace the module's clock only; the process-wide monotonic clock stays real.
    monkeypatch.setattr(entra, "time", types.SimpleNamespace(monotonic=lambda: clock.now))

    with pytest.raises(GatewayUnavailable) as first:
        credential.get_token()
    assert first.value.code == "credential_exchange_failed"
    assert "PRIVATE" not in repr(first.value)
    assert azure_stub.chain_calls == ["api://AzureADTokenExchange/.default"]

    with pytest.raises(GatewayUnavailable) as second:
        credential.get_token()
    assert second.value.code == "credential_exchange_failed"
    assert len(azure_stub.chain_calls) == 1, "memoised failure must not call the chain again"

    clock.now += entra._HOST_FAILURE_TTL_SECONDS + 1
    azure_stub.chain_error = None
    assert credential.get_token() == "final-child-resource-token"
    assert len(azure_stub.chain_calls) == 2
    assert exchanges and len(exchanges) == 2


def test_entra_mi_credential_override_bypasses_host_detection(azure_stub):
    class ManagedIdentity:
        def __init__(self):
            self.scopes = []

        def get_token(self, scope):
            self.scopes.append(scope)
            return _FakeToken("managed-identity-assertion")

    exchanges: list[dict] = []
    managed_identity = ManagedIdentity()
    credential = entra.EntraAgentCredential(
        authority="https://login.microsoftonline.com/tenant/v2.0",
        blueprint_client_id="blueprint-client",
        agent_identity_client_id="child-agent-client",
        gateway_audience="api://deepintshield-gateway",
        scopes=["tools:invoke"],
        mi_credential=managed_identity,
        http_client=httpx.Client(transport=httpx.MockTransport(_exchange_handler(exchanges))),
    )

    assert credential.get_token() == "final-child-resource-token"
    assert managed_identity.scopes == ["api://AzureADTokenExchange/.default"]
    assert azure_stub.constructed == []


def test_imds_probe_is_skipped_by_environment_and_memoised(monkeypatch):
    monkeypatch.setattr(entra, "_imds_probe_result", None)
    monkeypatch.setenv("DEEPINTSHIELD_ENTRA_SKIP_IMDS_PROBE", "1")
    assert entra._imds_reachable() is False
    assert entra._imds_probe_result is None

    monkeypatch.delenv("DEEPINTSHIELD_ENTRA_SKIP_IMDS_PROBE")
    probes = []

    class Refused:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def create_connection(address, timeout):
        probes.append((address, timeout))
        raise OSError("unreachable")

    monkeypatch.setattr(entra.socket, "create_connection", create_connection)
    assert entra._imds_reachable() is False
    assert entra._imds_reachable() is False
    assert probes == [(("169.254.169.254", 80), 1.0)]


# ── credential-info model ─────────────────────────────────────────────────


def test_credential_info_without_agent_identity_client_id_parses_for_gateway_broker():
    info = VKCredentialInfo.model_validate({
        "runtime_auth_mode": "gateway_broker",
        "runtime_broker": _broker_config("agent:math-agent"),
        "provider_id": "idp-entra", "provider_type": "entra_agent_id",
        "agent_subject": "agent:math-agent", "agent_configured": True,
        "credential_bootstrap": True, "provisional": True,
        "registration_id": REGISTRATION_ID, "registration_state": "pending",
    })
    assert info.agent_identity_client_id == ""
    assert info.credential_bootstrap is True
    assert info.provisional is True
    assert info.registration_state == "pending"
    assert info.registration_id == REGISTRATION_ID
    legacy = VKCredentialInfo.model_validate({"provider_type": "", "agent_configured": False})
    assert (legacy.credential_bootstrap, legacy.provisional, legacy.registration_state, legacy.registration_id) == (
        False, False, "", "",
    )


# ── managed execution ─────────────────────────────────────────────────────


def test_registered_by_is_an_allowed_worker_environment_variable():
    assert "DEEPINTSHIELD_REGISTERED_BY" in managed_runner._ALLOWED_ENV


def test_worker_configuration_passes_registered_by_through_to_the_program(tmp_path, capsys):
    script = tmp_path / "entry.py"
    script.write_text("import os\nprint(os.environ['DEEPINTSHIELD_REGISTERED_BY'])\n")
    config = tmp_path / "worker.json"
    config.write_text(json.dumps({
        "environment": {
            "DEEPINTSHIELD_BASE_URL": "https://gateway.example", "DEEPINTSHIELD_VIRTUAL_KEY": "sk-ds-test",
            "DEEPINTSHIELD_AGENT_NAME": "math-agent", "DEEPINTSHIELD_REGISTERED_BY": "dev@corp.com",
        },
        "runtime_private_key": "not-a-real-key",
        "requirements": [],
    }))

    assert managed_runner.main(["--config", str(config), str(script)]) == 0
    assert capsys.readouterr().out == "dev@corp.com\n"
    assert "DEEPINTSHIELD_REGISTERED_BY" not in os.environ

    config.write_text(json.dumps({
        "environment": {"DEEPINTSHIELD_BASE_URL": "https://gateway.example", "DEEPINTSHIELD_REGISTERED_BY_X": "x"},
        "runtime_private_key": "not-a-real-key",
    }))
    assert managed_runner.main(["--config", str(config), str(script)]) == 1
    assert "managed_execution_configuration_invalid" in capsys.readouterr().err


def test_registered_by_is_not_a_delegation_context(shield_factory):
    engine = shield_factory(
        lambda request: httpx.Response(404), agent_name="math-agent", registered_by="dev@corp.com",
    ).agentic.engine
    assert engine.registered_by_email == "dev@corp.com"
    assert engine.registered_by_source == "explicit"
    managed._require_autonomous_context(engine)  # must not raise

    delegated = shield_factory(
        lambda request: httpx.Response(404), agent_name="math-agent", requester="alice@corp.com",
    ).agentic.engine
    with pytest.raises(GovernanceConfigurationError, match="managed_execution_context_unsupported"):
        managed._require_autonomous_context(delegated)


@pytest.mark.parametrize(
    ("registrant", "expected"),
    [
        # A server run is never a delegated run, so its registrant is the
        # explicit setting; the bundle manifest says so beside the address.
        (
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
        ),
        # An engine object that predates the source still sends the pair.
        (
            {"registered_by_email": "dev@corp.com"},
            {"registered_by_email": "dev@corp.com", "registered_by_source": "explicit"},
        ),
        # No registering e-mail: neither field reaches the bundle.
        ({"registered_by_email": "", "registered_by_source": ""}, {}),
        ({}, {}),
    ],
)
def test_handoff_bundle_manifest_names_the_registering_developer(tmp_path, monkeypatch, registrant, expected):
    from types import SimpleNamespace

    script = tmp_path / "entry.py"
    script.write_text("raise AssertionError('must not execute locally')\n")
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script))
    monkeypatch.setattr(managed, "package_script", lambda *_: {
        "entrypoint": "entry.py", "files": {"entry.py": script.read_text()}, "requirements": [],
    })
    monkeypatch.setattr(registry, "describe_network", lambda _: {"agents": [{"key": "math"}]})
    submissions: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/submissions"):
            submissions.append(_json(request))
            # Not a 202 with a job: the handoff stops after the one upload.
            return httpx.Response(200, json={"accepted": False})
        return httpx.Response(200, json={"enabled": True})

    engine = SimpleNamespace(
        gateway_url="https://gateway.example", virtual_key="sk-ds-managed-test",
        **registrant,
        _http=httpx.Client(transport=httpx.MockTransport(handler)),
        _headers=lambda **_: {"Authorization": "Bearer sk-ds-managed-test", "X-Agent-Subject": "agent:math"},
        _managed_execution_capabilities={"enabled": True},
    )
    try:
        with pytest.raises(GovernanceConfigurationError, match="managed_execution_response_invalid"):
            managed.handoff_if_enabled(engine, object())
    finally:
        engine._http.close()

    assert len(submissions) == 1
    manifest = submissions[0]["manifest"]
    assert {key: value for key, value in manifest.items() if key.startswith("registered_by")} == expected
    assert manifest["auto_provision"] is True
    assert "principal" not in manifest
