"""Entra Agent ID credential - Federated Identity Credential exchange.

Flow per call (only when the cached token is missing or near expiry):

    1. Get a self-token from the host identity for audience
       ``api://AzureADTokenExchange``. The host identity is selected from
       what the process can actually see: environment service-principal
       variables, a workload-identity federated token file, or a managed
       identity endpoint (AKS/App Service/VM IMDS). No marker means no host
       identity, which fails fast with ``credential_host_identity_missing``
       instead of letting the Azure SDK probe every source on every call.

    2. POST that as the ``client_assertion`` for the blueprint, with
       ``fmi_path`` naming the child Agent Identity. Entra validates the FIC
       and returns an intermediate Agent ID assertion (T1).

    3. Exchange T1 as the child identity for the gateway resource token (T2).
       Cache T2 and refresh ~5 min before expiry.

``azure-identity`` is an optional install (``pip install
deepintshield[azure]``). If it is absent we surface a clear error pointing at
the install command rather than letting Python die with a generic
ImportError on first use. Failures of the host chain are memoised so the Azure
error block is produced at most once per window.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import socket
import sys
import threading
import time
from typing import List, Mapping, Optional

import httpx

from ..errors import (
    DeepIntShieldError,
    GatewayUnavailable,
    GovernanceConfigurationError,
)
from .base import _CachedToken

log = logging.getLogger(__name__)

_ENTRA_TOKEN_EXCHANGE_SCOPE = "api://AzureADTokenExchange/.default"
# A failed host chain (bad federation, unreachable IMDS, revoked identity) is
# remembered for this long so retries do not re-run the Azure chain, and the
# SDK prints its one warning per process instead of one per call.
_HOST_FAILURE_TTL_SECONDS = 60.0
_IMDS_ENDPOINT = ("169.254.169.254", 80)
_IMDS_PROBE_TIMEOUT_SECONDS = 1.0
_SKIP_IMDS_PROBE_ENV = "DEEPINTSHIELD_ENTRA_SKIP_IMDS_PROBE"
_MANAGED_IDENTITY_ENDPOINT_ENV = (
    "IDENTITY_ENDPOINT",
    "MSI_ENDPOINT",
    "IMDS_ENDPOINT",
    "AZURE_POD_IDENTITY_AUTHORITY_HOST",
)


def _azure_identity_available() -> bool:
    """Whether ``azure.identity`` can be imported, without importing it."""
    if "azure.identity" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("azure.identity") is not None
    except Exception:  # pragma: no cover - a broken parent package behaves as absent
        return False


_AZURE_AVAILABLE = _azure_identity_available()

_imds_probe_lock = threading.Lock()
_imds_probe_result: Optional[bool] = None


def _env_value(environ: Mapping[str, str], name: str) -> str:
    return str(environ.get(name) or "").strip()


def _imds_reachable(environ: Optional[Mapping[str, str]] = None) -> bool:
    """One process-wide, memoised TCP probe of the Azure IMDS endpoint.

    Consulted only when no environment marker names a host identity. Never
    raises; ``DEEPINTSHIELD_ENTRA_SKIP_IMDS_PROBE=1`` skips it (CI, laptops).
    """
    global _imds_probe_result
    env = os.environ if environ is None else environ
    if _env_value(env, _SKIP_IMDS_PROBE_ENV).lower() in {"1", "true", "yes"}:
        return False
    with _imds_probe_lock:
        if _imds_probe_result is None:
            try:
                with socket.create_connection(
                    _IMDS_ENDPOINT, timeout=_IMDS_PROBE_TIMEOUT_SECONDS
                ):
                    pass
                _imds_probe_result = True
            except Exception:
                _imds_probe_result = False
        return _imds_probe_result


def _host_identity_markers(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[str, ...]:
    """Name the host identity sources this process can use, in chain order."""
    env = os.environ if environ is None else environ

    def present(*names: str) -> bool:
        return all(_env_value(env, name) for name in names)

    markers: list[str] = []
    if present("AZURE_CLIENT_ID", "AZURE_TENANT_ID") and (
        present("AZURE_CLIENT_SECRET")
        or present("AZURE_CLIENT_CERTIFICATE_PATH")
        or present("AZURE_USERNAME", "AZURE_PASSWORD")
    ):
        markers.append("environment")
    if present("AZURE_FEDERATED_TOKEN_FILE", "AZURE_CLIENT_ID", "AZURE_TENANT_ID"):
        markers.append("workload_identity")
    if any(present(name) for name in _MANAGED_IDENTITY_ENDPOINT_ENV):
        markers.append("managed_identity")
    elif not markers and _imds_reachable(env):
        markers.append("managed_identity")
    return tuple(markers)


def _build_host_credential(markers: tuple[str, ...]) -> object:
    """Build the lazy ``ChainedTokenCredential`` from the matched markers only.

    ``WorkloadIdentityCredential()`` raises without its variables and
    ``EnvironmentCredential`` logs warnings without its own, so members are
    included only when their marker matched.
    """
    from azure.identity import (
        ChainedTokenCredential,
        EnvironmentCredential,
        ManagedIdentityCredential,
        WorkloadIdentityCredential,
    )

    members: list[object] = []
    if "environment" in markers:
        members.append(EnvironmentCredential())
    if "workload_identity" in markers:
        members.append(WorkloadIdentityCredential())
    if "managed_identity" in markers:
        members.append(
            ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID") or None)
        )
    return ChainedTokenCredential(*members)


class EntraAgentCredential:
    """Concrete AgentCredential for Microsoft Entra Agent ID blueprints.

    Constructed via :class:`AgenticEngine` from the values discovered at
    ``GET /api/agentic-new/credential-info`` - developers should not instantiate
    this directly except for overrides. ``mi_credential`` bypasses host
    detection and failure memoisation (tests, custom hosts).
    """

    def __init__(
        self,
        authority: str,
        blueprint_client_id: str,
        agent_identity_client_id: str,
        gateway_audience: str,
        scopes: List[str],
        fic_audience: str = "api://AzureADTokenExchange",
        exchange_endpoint: str = "",
        mi_credential: Optional[object] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if mi_credential is None and not _azure_identity_available():
            raise GovernanceConfigurationError(
                framework="agent-credential",
                reason="azure-identity dependency is unavailable",
                code="credential_dependency_missing",
            ) from None
        self._authority = authority.rstrip("/")
        self._blueprint_client_id = blueprint_client_id
        self._agent_identity_client_id = agent_identity_client_id
        self._gateway_audience = gateway_audience
        self._scopes = list(scopes) or ["tools:invoke"]
        self._fic_audience = fic_audience
        self._exchange_endpoint = (
            exchange_endpoint or f"{self._authority}/oauth2/v2.0/token"
        )
        # The host chain is built lazily, on the first token request, from the
        # identity sources the process can see. Construction never touches
        # the network, and nothing is built when an override is supplied.
        self._mi_credential = mi_credential
        self._host_credential: Optional[object] = None
        self._host_failure_code = ""
        self._host_failure_until = 0.0
        self._host_lock = threading.Lock()
        self._http = http_client or httpx.Client(timeout=10.0)
        self._cache = _CachedToken()

    @property
    def provider_type(self) -> str:
        return "entra_agent_id"

    def get_token(self) -> str:
        try:
            return self._cache.get_or_refresh(self._exchange)
        except DeepIntShieldError:
            raise
        except Exception as exc:
            raise GatewayUnavailable(
                reason=type(exc).__name__,
                code="credential_exchange_failed",
            ) from None

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    def _exchange(self) -> tuple[str, float]:
        """Perform the autonomous Agent ID T1→T2 exchange."""
        if not self._blueprint_client_id or not self._agent_identity_client_id:
            raise GovernanceConfigurationError(
                framework="agent-credential",
                reason="Entra Agent ID identifiers are incomplete",
                code="credential_configuration_error",
            )
        log.debug("entra FIC exchange to %s", self._exchange_endpoint)
        # 1. Get the host self-token used as the blueprint assertion.
        mi_token = self._get_mi_token()

        # 2. Blueprint + fmi_path -> intermediate child-agent assertion (T1).
        t1_data = {
            "client_id": self._blueprint_client_id,
            "fmi_path": self._agent_identity_client_id,
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": mi_token,
            "grant_type": "client_credentials",
            # This is the Entra Agent ID token-exchange resource, not a
            # provider/downstream scope and therefore is never caller-defined.
            "scope": _ENTRA_TOKEN_EXCHANGE_SCOPE,
        }
        t1_resp = self._http.post(
            self._exchange_endpoint,
            data=t1_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if t1_resp.status_code >= 400:
            raise RuntimeError(
                f"Entra Agent ID T1 exchange failed (HTTP {t1_resp.status_code}): "
                f"{t1_resp.text[:300]}"
            )
        t1 = str(t1_resp.json().get("access_token") or "").strip()
        if not t1:
            raise RuntimeError("Entra Agent ID T1 exchange returned no access_token")

        # 3. Child identity + T1 -> final gateway resource token (T2).
        t2_data = {
            "client_id": self._agent_identity_client_id,
            "client_assertion_type": (
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            ),
            "client_assertion": t1,
            "grant_type": "client_credentials",
            "scope": self._default_scope(self._gateway_audience),
        }
        t2_resp = self._http.post(
            self._exchange_endpoint,
            data=t2_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if t2_resp.status_code >= 400:
            raise RuntimeError(
                f"Entra Agent ID T2 exchange failed (HTTP {t2_resp.status_code}): "
                f"{t2_resp.text[:300]}"
            )
        body = t2_resp.json()
        token = str(body.get("access_token") or "").strip()
        if not token:
            raise RuntimeError("Entra Agent ID T2 exchange returned no access_token")
        return token, float(body.get("expires_in", 3600))

    @staticmethod
    def _default_scope(resource: str) -> str:
        resource = str(resource or "").strip().rstrip("/")
        if resource.endswith("/.default"):
            return resource
        return f"{resource}/.default"

    def _raise_memoised_host_failure(self) -> None:
        if not self._host_failure_code:
            return
        if time.monotonic() >= self._host_failure_until:
            self._host_failure_code = ""
            self._host_failure_until = 0.0
            return
        if self._host_failure_code == "credential_host_identity_missing":
            raise GovernanceConfigurationError(
                framework="agent-credential", code=self._host_failure_code,
            )
        raise GatewayUnavailable(
            reason="host credential chain failed recently",
            code=self._host_failure_code,
        )

    def _memoise_host_failure(self, code: str, ttl: float) -> None:
        self._host_failure_code = code
        self._host_failure_until = time.monotonic() + ttl

    def _get_mi_token(self) -> str:
        """Ask the host identity for a self-token to the FIC audience.

        An explicit ``mi_credential`` bypasses host detection and memoisation.
        Otherwise the chain is built once from the host markers; no marker
        fails fast for the lifetime of this credential, and a failing chain is
        remembered for ``_HOST_FAILURE_TTL_SECONDS``. The Azure SDK handles the
        IMDS roundtrip + caching inside the chain.
        """
        scope = self._default_scope(self._fic_audience)
        if self._mi_credential is not None:
            return self._mi_credential.get_token(scope).token
        with self._host_lock:
            self._raise_memoised_host_failure()
            if self._host_credential is None:
                markers = _host_identity_markers()
                if not markers:
                    self._memoise_host_failure("credential_host_identity_missing", math.inf)
                    raise GovernanceConfigurationError(
                        framework="agent-credential",
                        reason="no Azure host identity marker is present",
                        code="credential_host_identity_missing",
                    )
                try:
                    self._host_credential = _build_host_credential(markers)
                except ImportError:
                    raise GovernanceConfigurationError(
                        framework="agent-credential",
                        reason="azure-identity dependency is unavailable",
                        code="credential_dependency_missing",
                    ) from None
                except Exception as exc:
                    self._memoise_host_failure("credential_exchange_failed", _HOST_FAILURE_TTL_SECONDS)
                    raise GatewayUnavailable(
                        reason=type(exc).__name__,
                        code="credential_exchange_failed",
                    ) from None
            try:
                result = self._host_credential.get_token(scope)
            except DeepIntShieldError:
                raise
            except Exception as exc:
                self._memoise_host_failure("credential_exchange_failed", _HOST_FAILURE_TTL_SECONDS)
                raise GatewayUnavailable(
                    reason=type(exc).__name__,
                    code="credential_exchange_failed",
                ) from None
            return result.token
