from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs

import httpx
import pytest

from deepintshield.agentic.credentials.base import _CachedToken
from deepintshield.agentic.credentials.entra import EntraAgentCredential
from deepintshield.agentic.credentials.oidc import OIDCCredential
from deepintshield.agentic.credentials.zeroid import ZeroIDCredential
from deepintshield.agentic.errors import (
    GatewayUnavailable,
    GovernanceConfigurationError,
)


@pytest.mark.parametrize(
    "credential_type",
    [EntraAgentCredential, OIDCCredential, ZeroIDCredential],
)
def test_first_token_refresh_is_single_flight_for_every_provider(credential_type):
    """Regression: get_token held a Lock then called cache.set(), which tried to
    acquire the same non-reentrant Lock and deadlocked the first caller."""
    credential = credential_type.__new__(credential_type)
    credential._cache = _CachedToken()
    calls = 0
    calls_lock = threading.Lock()
    barrier = threading.Barrier(16)

    def exchange():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.03)  # keep callers overlapped inside the refresh
        return "fresh-token", 3600.0

    credential._exchange = exchange

    def get():
        barrier.wait(5)
        return credential.get_token()

    with ThreadPoolExecutor(max_workers=16) as pool:
        tokens = list(pool.map(lambda _i: get(), range(16)))

    assert tokens == ["fresh-token"] * 16
    assert calls == 1


def test_failed_refresh_does_not_poison_the_cache_or_lock():
    cache = _CachedToken()
    calls = 0

    def refresh():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("issuer unavailable")
        return "recovered", 3600.0

    with pytest.raises(RuntimeError, match="issuer unavailable"):
        cache.get_or_refresh(refresh)

    assert cache.get_or_refresh(refresh) == "recovered"
    assert cache.get_or_refresh(refresh) == "recovered"
    assert calls == 2


def test_oidc_missing_secret_uses_code_only_configuration_error(monkeypatch):
    monkeypatch.delenv("TEST_OIDC_SECRET", raising=False)
    credential = OIDCCredential(
        exchange_endpoint="https://issuer.example/token",
        client_id="agent",
        client_secret_env="TEST_OIDC_SECRET",
    )

    with pytest.raises(GovernanceConfigurationError) as caught:
        credential.get_token()

    assert str(caught.value) == "credential_configuration_error"
    assert "TEST_OIDC_SECRET" not in repr(caught.value)


def test_credential_exchange_failure_suppresses_provider_response(monkeypatch):
    monkeypatch.setenv("TEST_OIDC_SECRET", "secret")

    def exchange(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="private identity-provider detail")

    credential = OIDCCredential(
        exchange_endpoint="https://issuer.example/token",
        client_id="agent",
        client_secret_env="TEST_OIDC_SECRET",
        http_client=httpx.Client(transport=httpx.MockTransport(exchange)),
    )

    with pytest.raises(GatewayUnavailable) as caught:
        credential.get_token()

    assert str(caught.value) == "credential_exchange_failed"
    assert caught.value.__cause__ is None
    assert "private identity-provider detail" not in repr(caught.value)


def test_entra_credential_uses_blueprint_t1_then_child_identity_t2():
    class ManagedIdentity:
        def __init__(self):
            self.scopes = []

        def get_token(self, scope):
            self.scopes.append(scope)
            return type("AccessToken", (), {"token": "managed-identity-assertion"})()

    exchanges = []

    def exchange(request: httpx.Request) -> httpx.Response:
        form = {
            key: values[0]
            for key, values in parse_qs(request.content.decode("utf-8")).items()
        }
        exchanges.append(form)
        if len(exchanges) == 1:
            assert form == {
                "client_id": "blueprint-client",
                "fmi_path": "child-agent-client",
                "client_assertion_type": (
                    "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
                ),
                "client_assertion": "managed-identity-assertion",
                "grant_type": "client_credentials",
                "scope": "api://AzureADTokenExchange/.default",
            }
            return httpx.Response(200, json={"access_token": "intermediate-t1"})

        assert form == {
            "client_id": "child-agent-client",
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": "intermediate-t1",
            "grant_type": "client_credentials",
            "scope": "api://deepintshield-gateway/.default",
        }
        return httpx.Response(
            200,
            json={"access_token": "final-child-resource-token", "expires_in": 3600},
        )

    managed_identity = ManagedIdentity()
    http_client = httpx.Client(transport=httpx.MockTransport(exchange))
    credential = EntraAgentCredential(
        authority="https://login.microsoftonline.com/tenant/v2.0",
        blueprint_client_id="blueprint-client",
        agent_identity_client_id="child-agent-client",
        gateway_audience="api://deepintshield-gateway",
        scopes=["tools:invoke"],
        fic_audience="api://custom-managed-identity-audience",
        mi_credential=managed_identity,
        http_client=http_client,
    )

    assert credential.get_token() == "final-child-resource-token"
    assert credential.get_token() == "final-child-resource-token"
    assert managed_identity.scopes == [
        "api://custom-managed-identity-audience/.default"
    ]
    assert len(exchanges) == 2
