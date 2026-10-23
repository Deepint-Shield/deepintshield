from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from deepintshield.agentic import dpop
from deepintshield.agentic import execution, identity, registry
from deepintshield.agentic.credentials import StaticAgentCredential


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _verify(proof: str) -> dict:
    header_part, payload_part, signature_part = proof.split(".")
    header = json.loads(_decode(header_part))
    assert header["typ"] == "dpop+jwt"
    assert header["alg"] == "ES256"
    jwk = header["jwk"]
    assert "d" not in jwk
    public_key = ec.EllipticCurvePublicNumbers(
        int.from_bytes(_decode(jwk["x"]), "big"),
        int.from_bytes(_decode(jwk["y"]), "big"),
        ec.SECP256R1(),
    ).public_key()
    signature = _decode(signature_part)
    public_key.verify(
        utils.encode_dss_signature(
            int.from_bytes(signature[:32], "big"),
            int.from_bytes(signature[32:], "big"),
        ),
        f"{header_part}.{payload_part}".encode("ascii"),
        ec.ECDSA(hashes.SHA256()),
    )
    return json.loads(_decode(payload_part))


def _ath(token: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(token.encode("ascii")).digest()).rstrip(b"=").decode()


def test_dpop_signature_exact_token_hash_and_request_binding():
    key = dpop.DPoPKey()
    token = "exact.agent.token"
    first = _verify(key.proof("POST", "https://gw.example/Case/Path/?q=1#x", token))
    second = _verify(key.proof("POST", "https://gw.example/Case/Path/?q=1", token))
    assert first["ath"] == _ath(token)
    assert first["htm"] == "POST"
    assert first["htu"] == "https://gw.example/Case/Path/"
    assert first["jti"] != second["jti"]
    assert _verify(key.proof("post", "https://gw.example/x", token))["htm"] == "post"
    assert _verify(key.proof("POST", "https://gw.example/x", " " + token))["ath"] != first["ath"]


def test_dpop_token_endpoint_helper_remains_callable_without_access_token():
    assert "ath" not in _verify(dpop.DPoPKey().proof("POST", "https://issuer.example/token"))


def test_engine_generates_ath_for_actual_agent_token_without_public_config_changes(shield_factory):
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    engine.set_agent_credential(StaticAgentCredential("exact.workload.token"))
    headers = engine._headers(include_agent_token=True, method="POST", path="/api/agentic-new/decide")
    proof = _verify(headers["DPoP"])
    assert headers["Authorization"] == "Bearer sk-ds-test"
    assert proof["ath"] == _ath(headers["X-Agent-Token"])
    assert proof["ath"] != _ath("sk-ds-test")
    assert proof["htu"] == f"{engine.gateway_url.rstrip('/')}/api/agentic-new/decide"


def test_plain_bearer_headers_survive_unavailable_optional_crypto(shield_factory, monkeypatch):
    monkeypatch.setattr(dpop, "_CRYPTOGRAPHY_AVAILABLE", False)
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    engine.set_agent_credential(StaticAgentCredential("plain.bearer.token"))
    headers = engine._headers(include_agent_token=True)
    assert headers["X-Agent-Token"] == "plain.bearer.token"
    assert headers["Authorization"] == "Bearer sk-ds-test"
    assert "DPoP" not in headers


def test_control_plane_headers_do_not_attach_agent_proofs(shield_factory):
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    headers = engine._headers(include_agent_token=False)
    assert "X-Agent-Token" not in headers
    assert "DPoP" not in headers


def test_discovery_identity_and_execution_proofs_bind_the_actual_request(shield_factory):
    proofs = []

    def handler(request):
        claims = _verify(request.headers["DPoP"])
        assert claims["htm"] == request.method
        assert claims["htu"] == str(request.url)
        assert claims["ath"] == _ath(request.headers["X-Agent-Token"])
        proofs.append(claims)
        if request.url.path.endswith("/identity/resolve"):
            return httpx.Response(200, json={"subject": "user:alice"})
        return httpx.Response(200, json={})

    engine = shield_factory(handler).agentic.engine
    engine.set_agent_credential(StaticAgentCredential("exact.workload.token"))
    assert identity._post_resolve(
        engine, email="alice@example.com", username=None, subject=None,
        display_name=None, kind="user",
    ) == "user:alice"
    assert "error" not in registry._post_discover(engine, {}, timeout=1)
    execution._post_event(engine, {"event": "complete"})
    assert len(proofs) == 3
    assert len({claim["jti"] for claim in proofs}) == 3
