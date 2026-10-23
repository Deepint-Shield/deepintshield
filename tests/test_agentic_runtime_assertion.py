from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import httpx
import pytest

from deepintshield.agentic.credentials.runtime_assertion import RuntimeAssertion
from deepintshield.agentic.errors import GovernanceConfigurationError


JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJydW50aW1lIn0.c2lnbmF0dXJl"


@pytest.fixture(autouse=True)
def no_ambient_runtime(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPINTSHIELD_RUNTIME_"):
            monkeypatch.delenv(name)


@pytest.mark.parametrize("kind", ["gcp", "aws", "azure", "kubernetes"])
def test_discovery_cannot_select_ambient_native_credentials(kind, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("metadata accessed without opt-in"))
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        RuntimeAssertion(kind, "runtime-audience")


def metadata_client(monkeypatch, handler):
    real = httpx.Client

    def create(**kwargs):
        assert kwargs == {"timeout": 5, "follow_redirects": False, "trust_env": False}
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", create)


def test_gcp_adapter_requests_only_fixed_metadata_identity_audience(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "gcp")

    def handle(request):
        assert request.url.host == "169.254.169.254"
        assert request.url.path == "/computeMetadata/v1/instance/service-accounts/default/identity"
        assert dict(request.url.params) == {"audience": "https://gateway.example/runtime", "format": "full"}
        assert request.headers["Metadata-Flavor"] == "Google"
        assert "Authorization" not in request.headers
        return httpx.Response(200, text=JWT, headers={"Metadata-Flavor": "Google"})

    metadata_client(monkeypatch, handle)
    assert RuntimeAssertion("gcp", "https://gateway.example/runtime").get_token() == JWT


@pytest.mark.parametrize("client_id", ["", "explicit-managed-client"])
def test_azure_vm_adapter_uses_only_explicit_managed_identity(monkeypatch, client_id):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "azure")
    monkeypatch.setenv("AZURE_CLIENT_ID", "unrelated-ambient-sdk-client")
    if client_id:
        monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_AZURE_CLIENT_ID", client_id)

    def handle(request):
        assert request.url.host == "169.254.169.254"
        assert request.url.path == "/metadata/identity/oauth2/token"
        assert request.url.params["api-version"] == "2018-02-01"
        assert request.url.params["resource"] == "api://runtime-api"
        assert request.headers["Metadata"] == "true"
        assert request.url.params.get("client_id", "") == client_id
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"access_token": JWT})

    metadata_client(monkeypatch, handle)
    assert RuntimeAssertion("azure", "api://runtime-api").get_token() == JWT


@pytest.mark.parametrize("kind", ["gcp", "azure"])
def test_metadata_redirect_is_not_followed(kind, monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", kind)
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://attacker.example"})

    metadata_client(monkeypatch, handle)
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_assertion_invalid"):
        RuntimeAssertion(kind, "audience").get_token()
    assert len(calls) == 1


def test_kubernetes_requires_explicit_projected_token_and_rereads_rotated_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "kubernetes")
    assertion = RuntimeAssertion("kubernetes", "gateway-runtime")
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        assertion.get_token()
    token = tmp_path / "projected-token"
    token.write_text(JWT)
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_TOKEN_FILE", str(token))
    assert assertion.get_token() == JWT
    token.write_text("new.header.signature")
    assert assertion.get_token() == "new.header.signature"
    token.write_text("not-a-jwt")
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_assertion_invalid"):
        assertion.get_token()


def test_aws_uses_regional_outbound_identity_jwt_and_exact_audience(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "aws")
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_AWS_REGION", "us-west-2")
    calls = []

    class Client:
        def get_web_identity_token(self, **kwargs):
            assert kwargs == {"Audience": ["gateway-runtime"], "DurationSeconds": 300, "SigningAlgorithm": "RS256"}
            calls.append("token")
            return {"WebIdentityToken": JWT}

        def close(self):
            calls.append("close")

    class Session:
        def __init__(self, **kwargs):
            assert kwargs == {"region_name": "us-west-2"}

        def get_partition_for_region(self, region):
            assert region == "us-west-2"
            return "aws"

        def client(self, name, **kwargs):
            assert name == "sts"
            assert kwargs["endpoint_url"] == "https://sts.us-west-2.amazonaws.com"
            assert kwargs["region_name"] == "us-west-2"
            return Client()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(Session=Session))
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=lambda **kwargs: kwargs))
    assert RuntimeAssertion("aws", "gateway-runtime").get_token() == JWT
    assert calls == ["token", "close"]


def test_native_audience_can_be_locally_pinned(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "gcp")
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_AUDIENCE", "expected")
    with pytest.raises(GovernanceConfigurationError, match="agent_runtime_configuration_error"):
        RuntimeAssertion("gcp", "unexpected")


@pytest.mark.parametrize("endpoint_variable", ["IDENTITY_ENDPOINT", "MSI_ENDPOINT"])
def test_azure_hosted_apps_use_managed_identity_without_developer_fallback(monkeypatch, endpoint_variable):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "azure")
    monkeypatch.setenv(endpoint_variable, "http://127.0.0.1/managed-identity")
    monkeypatch.setenv("AZURE_CLIENT_ID", "unrelated-developer-client")
    calls = []

    class ManagedIdentity:
        def __init__(self, **kwargs):
            assert kwargs == {"client_id": None}

        def get_token(self, scope):
            assert scope == "api://runtime/.default"
            calls.append("managed-identity")
            return SimpleNamespace(token=JWT)

        def close(self):
            calls.append("close")

    monkeypatch.setitem(sys.modules, "azure.identity", SimpleNamespace(ManagedIdentityCredential=ManagedIdentity))
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("VM IMDS used in hosted Azure app"))
    assert RuntimeAssertion("azure", "api://runtime").get_token() == JWT
    assert calls == ["managed-identity", "close"]


def test_native_provider_errors_never_leak_cloud_credentials(monkeypatch):
    monkeypatch.setenv("DEEPINTSHIELD_RUNTIME_KIND", "azure")

    def fail(request):
        raise RuntimeError("sensitive-cloud-secret")

    metadata_client(monkeypatch, fail)
    with pytest.raises(GovernanceConfigurationError) as error:
        RuntimeAssertion("azure", "audience").get_token()
    assert "sensitive-cloud-secret" not in str(error.value)
    assert "sensitive-cloud-secret" not in repr(error.value)


def test_key_mode_never_touches_cloud_credentials(monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("metadata requested for key-only runtime"))
    assert RuntimeAssertion("key", "").get_token() is None
