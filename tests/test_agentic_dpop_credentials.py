from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs

import httpx
import pytest

pytest.importorskip("cryptography")

from deepintshield.agentic.credentials import dpop_oauth
from deepintshield.agentic.credentials.base import _CachedToken
from deepintshield.agentic.credentials.oidc import OIDCCredential
from deepintshield.agentic.dpop import process_dpop_key
from deepintshield.agentic.errors import GatewayUnavailable, GovernanceConfigurationError
from deepintshield.agentic.types import VKCredentialInfo
from tests.test_agentic_dpop import _verify

ISSUER = "https://issuer.example/realm"
ENDPOINT = ISSUER + "/token"


def _metadata(**changes):
    return dict({
        "issuer": ISSUER, "token_endpoint": ENDPOINT,
        "dpop_signing_alg_values_supported": ["ES256"],
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"],
    }, **changes)


def _response(**changes):
    # Token-response unit fixture; real issuer signature is exercised separately.
    claims = {"iss": ISSUER, "exp": time.time() + 300,
              "cnf": {"jkt": process_dpop_key().thumbprint}}
    claims.update(changes)
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return {"access_token": "header." + payload + ".signature",
            "token_type": "DPoP", "expires_in": 300}


def _credential(monkeypatch, handler, **kwargs):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "private-test-secret")
    return OIDCCredential(
        ENDPOINT, "agent-client", issuer=ISSUER, require_proof_of_possession=True,
        http_client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
        **kwargs,
    )


def test_nonce_retry_signature_cache_and_renewal_use_one_process_key(monkeypatch):
    calls = []
    responses = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_metadata())
        proof = _verify(request.headers["DPoP"])
        calls.append(proof)
        assert proof["htu"] == ENDPOINT and proof["htm"] == "POST"
        assert "ath" not in proof
        assert "scope" not in parse_qs(request.content.decode(), keep_blank_values=True)
        if len(calls) == 1:
            assert "nonce" not in proof
            return httpx.Response(400, json={"error": "use_dpop_nonce"}, headers={"DPoP-Nonce": "challenge"})
        if len(calls) == 2:
            assert proof["nonce"] == "challenge"
        else:
            assert proof["nonce"] == "next-request"
        body = _response()
        responses.append(body["access_token"])
        return httpx.Response(200, json=body, headers={"DPoP-Nonce": "next-request"})

    credential = _credential(monkeypatch, handler)
    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = list(pool.map(lambda _: credential.get_token(), range(8)))
    assert len(set(tokens)) == 1 and len(calls) == 2
    assert credential._dpop_exchange._key is process_dpop_key()
    credential._cache._refresh_at = 0
    assert credential.get_token() == responses[-1]
    assert len(calls) == 3
    assert len({proof["jti"] for proof in calls}) == 3


@pytest.mark.parametrize("metadata", [
    _metadata(issuer="https://other.example"),
    _metadata(dpop_signing_alg_values_supported=[]),
    _metadata(grant_types_supported=["authorization_code"]),
    _metadata(token_endpoint="https://other.example/token"),
    _metadata(token_endpoint_auth_methods_supported=["private_key_jwt"]),
])
def test_unsupported_capabilities_never_send_client_credentials(monkeypatch, metadata):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "GET" and b"private-test-secret" not in request.content
        return httpx.Response(200, json=metadata)

    with pytest.raises(GovernanceConfigurationError, match="credential_dpop_configuration_error"):
        _credential(monkeypatch, handler).get_token()
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["bearer", "unbound", "wrong_key", "opaque", "expired", "nan", "boolean_ttl", "wrong_issuer"])
def test_invalid_binding_and_downgrade_are_not_cached(monkeypatch, change):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_metadata())
        body = _response()
        if change == "bearer": body["token_type"] = "Bearer"
        elif change == "unbound": body = _response(cnf={})
        elif change == "wrong_key": body = _response(cnf={"jkt": "different"})
        elif change == "opaque": body["access_token"] = "opaque"
        elif change == "expired": body = _response(exp=time.time() - 1)
        elif change == "nan": body = _response(exp=float("nan"))
        elif change == "boolean_ttl": body["expires_in"] = True
        elif change == "wrong_issuer": body = _response(iss="https://other.example")
        return httpx.Response(200, json=body)

    credential = _credential(monkeypatch, handler)
    with pytest.raises(GovernanceConfigurationError):
        credential.get_token()
    assert credential._cache.get() is None


def test_repeated_nonce_challenge_is_bounded_and_secrets_are_not_exposed(monkeypatch):
    calls = []

    def handler(request):
        if request.method == "GET": return httpx.Response(200, json=_metadata())
        calls.append(request)
        return httpx.Response(400, json={"error": "use_dpop_nonce", "error_description": "private detail"}, headers={"DPoP-Nonce": "challenge"})

    with pytest.raises(GatewayUnavailable) as error:
        _credential(monkeypatch, handler).get_token()
    assert str(error.value) == "credential_dpop_nonce_failed"
    assert len(calls) == 2
    assert "private" not in repr(error.value)


def test_redirect_is_not_followed_even_with_redirecting_http_client(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET": return httpx.Response(200, json=_metadata())
        return httpx.Response(307, json={}, headers={"Location": "https://other.example/token"})

    with pytest.raises(GatewayUnavailable):
        _credential(monkeypatch, handler).get_token()
    assert len(calls) == 2 and all(request.url.host == "issuer.example" for request in calls)


def test_discovery_client_secret_basic_authentication(monkeypatch):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_metadata(token_endpoint_auth_methods_supported=["client_secret_basic"]))
        assert request.headers["Authorization"].startswith("Basic ")
        assert "client_secret" not in parse_qs(request.content.decode())
        return httpx.Response(200, json=_response())

    assert _credential(monkeypatch, handler).get_token()


def test_crypto_dependency_and_plain_http_fail_before_network(monkeypatch):
    monkeypatch.setattr(dpop_oauth, "process_dpop_key", lambda: None)
    with pytest.raises(GovernanceConfigurationError):
        _credential(monkeypatch, lambda _: pytest.fail("unexpected request"))
    with pytest.raises(GovernanceConfigurationError):
        OIDCCredential("http://issuer.example/token", "agent", issuer="http://issuer.example", require_proof_of_possession=True)


@pytest.mark.parametrize("provider", ["entra_agent_id", "zeroid", "unknown"])
def test_required_unsupported_adapter_has_no_bearer_fallback(shield_factory, provider):
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    with pytest.raises(GovernanceConfigurationError, match="credential_dpop_provider_unsupported"):
        engine._build_credential_for(VKCredentialInfo(provider_type=provider, require_proof_of_possession=True))


def test_discovered_gateway_requirement_automatically_selects_bound_adapter(shield_factory, monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_ID", "workload-client")
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    credential = engine._build_credential_for(VKCredentialInfo(
        provider_type="generic_oidc", issuer=ISSUER, require_proof_of_possession=True,
    ))
    assert credential._dpop_exchange._issuer == ISSUER
    assert credential._dpop_exchange._key is process_dpop_key()
    assert credential._client_id == "workload-client"


def test_short_lived_cache_renews_before_expiry_without_refreshing_every_call(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    cache = _CachedToken()
    cache.set("short-lived", 60)
    now += 53
    assert cache.get() == "short-lived"
    now += 2
    assert cache.get() is None
