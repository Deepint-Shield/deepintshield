from __future__ import annotations

import base64
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import ssl
import time
from urllib.parse import parse_qs

import httpx
import pytest

from deepintshield.agentic.credentials import oidc_auth
from deepintshield.agentic.credentials.oidc import OIDCCredential
from deepintshield.agentic.errors import GatewayUnavailable, GovernanceConfigurationError
from deepintshield.agentic.types import VKCredentialInfo


ISSUER = "https://issuer.example/realm"
ENDPOINT = ISSUER + "/token"


def credential(handler, **kwargs):
    return OIDCCredential(
        kwargs.pop("exchange_endpoint", ENDPOINT), "workload:client+id", issuer=ISSUER,
        http_client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
        **kwargs,
    )


@pytest.mark.parametrize("method", ["client_secret_post", "client_secret_basic"])
def test_secret_methods_resource_and_cache(monkeypatch, method):
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", method)
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "secret:with+reserved chars")
    monkeypatch.setenv("OIDC_AUDIENCE", "")
    monkeypatch.setenv("OIDC_RESOURCE", "https://resource.example/api")
    calls = []

    def handler(request):
        calls.append(request)
        form = parse_qs(request.content.decode())
        assert form["grant_type"] == ["client_credentials"]
        assert form["resource"] == ["https://resource.example/api"]
        assert form["scope"] == ["read write"]
        assert "audience" not in form
        if method == "client_secret_basic":
            assert base64.b64decode(request.headers["Authorization"].split()[1]).decode() == (
                "workload%3Aclient%2Bid:secret%3Awith%2Breserved+chars"
            )
            assert "client_secret" not in form and "client_id" not in form
        else:
            assert "Authorization" not in request.headers
            assert form["client_secret"] == ["secret:with+reserved chars"]
        return httpx.Response(200, json={"access_token": "warm-token", "expires_in": 60})

    client = credential(handler, gateway_audience="legacy-audience", scopes=["read", "write"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: client.get_token(), range(16))) == ["warm-token"] * 16
    assert len(calls) == 1


def test_issuer_discovery_once_and_explicit_endpoint_authority(monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "test-secret")
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.method == "GET":
            assert "Authorization" not in request.headers and not request.content
            return httpx.Response(200, json={
                "issuer": ISSUER, "token_endpoint": ENDPOINT,
                "token_endpoint_auth_methods_supported": ["client_secret_basic"],
            })
        assert "client_secret" not in parse_qs(request.content.decode())
        return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})

    client = credential(handler, exchange_endpoint="")
    assert client.get_token() == "token"
    client._cache._refresh_at = 0
    assert client.get_token() == "token"
    assert paths == ["/realm/.well-known/openid-configuration", "/realm/token", "/realm/token"]
    paths.clear()
    explicit = credential(handler, client_auth_method="client_secret_basic")
    assert explicit.get_token() == "token"
    assert paths == ["/realm/token"]


@pytest.mark.parametrize("metadata", [
    {"issuer": "https://other.example", "token_endpoint": ENDPOINT},
    {"issuer": ISSUER, "token_endpoint": "http://issuer.example/token"},
    {"issuer": ISSUER, "token_endpoint": ENDPOINT,
     "token_endpoint_auth_methods_supported": ["private_key_jwt"]},
])
def test_invalid_discovery_never_sends_credentials(monkeypatch, metadata):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "must-not-leak")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert b"must-not-leak" not in request.content
        return httpx.Response(200, json=metadata)

    with pytest.raises(GovernanceConfigurationError):
        credential(handler, exchange_endpoint="").get_token()
    assert len(requests) == 1


@pytest.mark.parametrize("endpoint", ["http://issuer.example/token", "http://localhost/token", "http://127.0.0.1.example/token", "http://192.168.1.1/token", "ftp://127.0.0.1/token", "https://user:pass@issuer.example/token", "https://issuer.example/token#secret"])
def test_insecure_endpoint_never_sends_credentials(monkeypatch, endpoint):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "must-not-leak")
    with pytest.raises(GovernanceConfigurationError):
        credential(lambda _: pytest.fail("unexpected HTTP"), exchange_endpoint=endpoint).get_token()


@pytest.mark.parametrize("host", ["127.0.0.1:9000", "[::1]:9000"])
@pytest.mark.parametrize("require_pop", [False, True])
def test_numeric_loopback_local_issuer_preserves_authentication(monkeypatch, host, require_pop):
    if require_pop:
        pytest.importorskip("cryptography")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "local-test-secret")
    issuer = "http://" + host + "/realm"
    endpoint = issuer + "/token"
    body = {"access_token": "local-token", "expires_in": 60, "token_type": "Bearer"}
    if require_pop:
        from tests.test_agentic_dpop_credentials import _response
        body = _response(iss=issuer)
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={
                "issuer": issuer, "token_endpoint": endpoint,
                "token_endpoint_auth_methods_supported": ["client_secret_post"],
                "grant_types_supported": ["client_credentials"],
                "dpop_signing_alg_values_supported": ["ES256"],
            })
        assert str(request.url) == endpoint
        assert parse_qs(request.content.decode())["client_secret"] == ["local-test-secret"]
        assert bool(request.headers.get("DPoP")) == require_pop
        return httpx.Response(200, json=body)

    client = OIDCCredential("", "local-client", issuer=issuer, require_proof_of_possession=require_pop,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.get_token() == body["access_token"]
    assert client.get_token() == body["access_token"]
    assert len(calls) == 2


def test_redirect_cannot_forward_client_secret(monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "private-secret")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://other.example/token"})

    with pytest.raises(GatewayUnavailable, match="credential_exchange_failed"):
        credential(handler).get_token()
    assert len(requests) == 1


def test_token_endpoint_query_is_preserved(monkeypatch):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "secret")

    def handler(request):
        assert str(request.url) == ENDPOINT + "?tenant=workload"
        return httpx.Response(200, json={"access_token": "token"})

    assert credential(handler, exchange_endpoint=ENDPOINT + "?tenant=workload").get_token() == "token"


@pytest.mark.parametrize("method", ["private_key_jwt", "tls_client_auth", "unknown"])
def test_missing_or_unknown_authentication_fails_before_network(monkeypatch, method):
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", method)
    monkeypatch.delenv("OIDC_PRIVATE_KEY_FILE", raising=False)
    monkeypatch.delenv("OIDC_CLIENT_CERT_FILE", raising=False)
    monkeypatch.delenv("OIDC_CLIENT_KEY_FILE", raising=False)
    with pytest.raises(GovernanceConfigurationError):
        credential(lambda _: pytest.fail("unexpected HTTP")).get_token()


@pytest.mark.parametrize("body", [
    {"access_token": ""}, {"access_token": "  "}, {"access_token": 42},
    {"access_token": "token", "expires_in": 0},
    {"access_token": "token", "expires_in": -1},
    {"access_token": "token", "expires_in": True},
    {"access_token": "token", "expires_in": "NaN"},
    {"access_token": "token", "expires_in": "Infinity"},
    {"access_token": "token", "token_type": "DPoP"}, [],
])
def test_malformed_responses_are_not_cached(monkeypatch, body):
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "secret")
    client = credential(lambda _: httpx.Response(200, json=body))
    with pytest.raises(GatewayUnavailable, match="credential_exchange_failed"):
        client.get_token()
    assert client._cache.get() is None


def _b64decode(value):
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _key(algorithm):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import ec, rsa
    return ec.generate_private_key(ec.SECP256R1()) if algorithm == "ES256" else rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _write_key(filename, key):
    from cryptography.hazmat.primitives import serialization
    filename.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))


def _verify_assertion(token, public_key, algorithm):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, utils
    header, body, signature = token.split(".")
    raw = _b64decode(signature)
    unsigned = f"{header}.{body}".encode()
    if algorithm == "ES256":
        signature_der = utils.encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        public_key.verify(signature_der, unsigned, ec.ECDSA(hashes.SHA256()))
    else:
        pad = padding.PKCS1v15() if algorithm == "RS256" else padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)
        public_key.verify(raw, unsigned, pad, hashes.SHA256())
    return json.loads(_b64decode(header)), json.loads(_b64decode(body))


@pytest.mark.parametrize("algorithm", ["RS256", "PS256", "ES256"])
def test_private_key_assertions_signature_rotation_and_cached_tokens(monkeypatch, tmp_path, algorithm):
    key = _key(algorithm)
    filename = tmp_path / "client-key.pem"
    _write_key(filename, key)
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", "private_key_jwt")
    monkeypatch.setenv("OIDC_PRIVATE_KEY_ALG", algorithm)
    monkeypatch.setenv("OIDC_PRIVATE_KEY_FILE", str(filename))
    monkeypatch.setenv("OIDC_PRIVATE_KEY_KID", "key-1")
    assertions = []

    def handler(request):
        form = parse_qs(request.content.decode())
        assert "client_secret" not in form and "Authorization" not in request.headers
        assert form["client_assertion_type"] == ["urn:ietf:params:oauth:client-assertion-type:jwt-bearer"]
        header, claims = _verify_assertion(form["client_assertion"][0], key.public_key(), algorithm)
        assert header["kid"] == ("key-1" if not assertions else "key-2")
        assert claims["iss"] == claims["sub"] == "workload:client+id"
        assert claims["aud"] == ENDPOINT
        assert 0 < claims["exp"] - time.time() <= 60
        assert claims["exp"] - claims["iat"] == 60
        assertions.append(claims)
        return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})

    client = credential(handler)
    assert client.get_token() == client.get_token() == "token"
    assert len(assertions) == 1
    key = _key(algorithm)
    _write_key(filename, key)
    monkeypatch.setenv("OIDC_PRIVATE_KEY_KID", "key-2")
    client._cache._refresh_at = 0
    assert client.get_token() == "token"
    assert len(assertions) == 2 and assertions[0]["jti"] != assertions[1]["jti"]


def test_private_key_dpop_nonce_retry_uses_fresh_signed_assertion(monkeypatch, tmp_path):
    from tests.test_agentic_dpop_credentials import _metadata, _response
    key = _key("ES256")
    filename = tmp_path / "client-key.pem"
    _write_key(filename, key)
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", "private_key_jwt")
    monkeypatch.setenv("OIDC_PRIVATE_KEY_ALG", "ES256")
    monkeypatch.setenv("OIDC_PRIVATE_KEY_FILE", str(filename))
    assertions = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_metadata(token_endpoint_auth_methods_supported=["private_key_jwt"]))
        form = parse_qs(request.content.decode())
        _, claims = _verify_assertion(form["client_assertion"][0], key.public_key(), "ES256")
        assertions.append(claims)
        if len(assertions) == 1:
            return httpx.Response(400, json={"error": "use_dpop_nonce"}, headers={"DPoP-Nonce": "nonce"})
        return httpx.Response(200, json=_response())

    client = credential(handler, require_proof_of_possession=True)
    assert client.get_token() == client.get_token()
    assert len(assertions) == 2 and assertions[0]["jti"] != assertions[1]["jti"]


def test_dpop_token_endpoint_query_is_excluded_only_from_proof(monkeypatch):
    from tests.test_agentic_dpop import _verify
    from tests.test_agentic_dpop_credentials import _metadata, _response
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "secret")
    endpoint = ENDPOINT + "?tenant=workload"

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_metadata(token_endpoint=endpoint))
        assert str(request.url) == endpoint
        assert _verify(request.headers["DPoP"])["htu"] == ENDPOINT
        return httpx.Response(200, json=_response())

    assert credential(handler, exchange_endpoint=endpoint, require_proof_of_possession=True).get_token()


@pytest.mark.parametrize("method", ["tls_client_auth", "self_signed_tls_client_auth"])
def test_mtls_uses_verified_ssl_context_and_reloads_cert_on_refresh(monkeypatch, tmp_path, method):
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.oid import NameOID
    keyfile, certfile = tmp_path / "key.pem", tmp_path / "cert.pem"

    def rotate():
        key = _key("ES256")
        _write_key(keyfile, key)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-client")])
        now = datetime.now(timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
                .public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
                .sign(key, hashes.SHA256()))
        certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    rotate()
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", method)
    monkeypatch.setenv("OIDC_CLIENT_CERT_FILE", str(certfile))
    monkeypatch.setenv("OIDC_CLIENT_KEY_FILE", str(keyfile))
    contexts = []
    calls = []

    def handler(request):
        calls.append(request)
        form = parse_qs(request.content.decode())
        assert form["client_id"] == ["workload:client+id"]
        assert "client_secret" not in form and "client_assertion" not in form
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})

    client = credential(lambda _: pytest.fail("token request must use TLS-authenticated client"))
    real_client = httpx.Client

    def tls_client(**kwargs):
        context = kwargs["verify"]
        assert isinstance(context, ssl.SSLContext)
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
        assert kwargs["follow_redirects"] is False
        contexts.append(context)
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(oidc_auth.httpx, "Client", tls_client)
    assert client.get_token() == client.get_token() == "token"
    assert len(contexts) == len(calls) == 1
    rotate()
    client._cache._refresh_at = 0
    assert client.get_token() == "token"
    assert len(contexts) == len(calls) == 2 and contexts[0] is not contexts[1]


def test_explicit_oidc_and_zeroid_environment_endpoints_take_precedence(shield_factory, monkeypatch):
    monkeypatch.setenv("OIDC_TOKEN_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("ZEROID_EXCHANGE_ENDPOINT", "https://zero.example/exchange")
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    oidc = engine._build_credential_for(VKCredentialInfo(provider_type="generic_oidc", exchange_endpoint="https://old.example/token"))
    zeroid = engine._build_credential_for(VKCredentialInfo(provider_type="zeroid", exchange_endpoint=""))
    assert oidc._endpoint == ENDPOINT
    assert zeroid._endpoint == "https://zero.example/exchange"


@pytest.mark.parametrize("configured, expected", [
    (None, "provider.read provider.write"),
    ("  workload.read\tworkload.call\n", "workload.read workload.call"),
    ("", None),
    (" \t\n", None),
])
def test_default_oidc_credential_uses_workload_scopes(shield_factory, monkeypatch, configured, expected):
    monkeypatch.setenv("OIDC_TOKEN_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("OIDC_CLIENT_ID", "provisioned-client")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "workload-secret")
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", "client_secret_post")
    monkeypatch.setenv("OIDC_AUDIENCE", "")
    monkeypatch.delenv("OIDC_RESOURCE", raising=False)
    if configured is None:
        monkeypatch.delenv("OIDC_SCOPES", raising=False)
    else:
        monkeypatch.setenv("OIDC_SCOPES", configured)
    info = VKCredentialInfo(provider_type="generic_oidc", scopes=["provider.read", "provider.write"])
    engine = shield_factory(lambda _: httpx.Response(200)).agentic.engine
    client = engine._build_credential_for(info)

    def exchange(request):
        form = parse_qs(request.content.decode())
        assert form["client_id"] == ["provisioned-client"]
        if expected is None:
            assert "scope" not in form
        else:
            assert form["scope"] == [expected]
        return httpx.Response(200, json={"access_token": "workload-token", "expires_in": 60})

    client._http.close()
    with httpx.Client(transport=httpx.MockTransport(exchange)) as transport:
        client._http = transport
        assert client.get_token() == "workload-token"
    assert info.scopes == ["provider.read", "provider.write"]


@pytest.mark.parametrize("require_dpop", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_mtls_advertised_endpoint_alias_and_dpop_nonce(monkeypatch, require_dpop, explicit):
    from tests.test_agentic_dpop import _verify
    from tests.test_agentic_dpop_credentials import _metadata, _response
    monkeypatch.setenv("OIDC_CLIENT_AUTH_METHOD", "tls_client_auth")
    endpoint = "https://mtls.issuer.example/token?client=workload"
    calls = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_metadata(
                token_endpoint_auth_methods_supported=["tls_client_auth"],
                mtls_endpoint_aliases={"token_endpoint": endpoint},
            ))
        assert str(request.url) == endpoint
        assert parse_qs(request.content.decode())["client_id"] == ["workload:client+id"]
        assert "Authorization" not in request.headers
        calls.append(request)
        if require_dpop:
            proof = _verify(request.headers["DPoP"])
            assert proof["htu"] == "https://mtls.issuer.example/token"
            if len(calls) == 1:
                return httpx.Response(400, json={"error": "use_dpop_nonce"}, headers={"DPoP-Nonce": "challenge"})
            assert proof["nonce"] == "challenge"
            return httpx.Response(200, json=_response())
        return httpx.Response(200, json={"access_token": "token"})

    client = credential(handler, exchange_endpoint=endpoint if explicit else "", require_proof_of_possession=require_dpop)
    # Certificate loading and refresh are separately tested with real PEMs;
    # keep this negotiation/nonce test on the mocked HTTP transport.
    monkeypatch.setattr(client._client_auth, "http_client", lambda default: nullcontext(default))
    assert client.get_token() == client.get_token()
    assert len(calls) == (2 if require_dpop else 1)
