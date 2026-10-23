"""Explicit cloud identity proofs for gateway runtime sessions.

Cloud selection is a deployment decision: discovery alone can never trigger
ambient credential access. The gateway validates issuer, subject, audience and
signature against the per-agent approved runtime binding. The SDK also keeps
the enrolled key in every flow; a cloud bearer assertion never replaces DPoP.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import httpx

from ..errors import GovernanceConfigurationError

_JWT = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


def _error(code: str = "agent_runtime_configuration_error") -> GovernanceConfigurationError:
    return GovernanceConfigurationError(framework="gateway-broker", code=code)


def _jwt(value: object) -> str:
    # Do not decode and mistake unverified claims for proof. All cryptographic
    # identity validation remains the gateway's responsibility.
    if not isinstance(value, str) or len(value) > 65536 or not _JWT.fullmatch(value):
        raise _error("agent_runtime_assertion_invalid")
    return value


class RuntimeAssertion:
    def __init__(self, kind: str, audience: str) -> None:
        self.kind = "key" if kind == "pinned_key" else kind
        self.audience = audience
        configured = os.environ.get("DEEPINTSHIELD_RUNTIME_KIND", "").strip().lower()
        if configured == "pinned_key":
            configured = "key"
        if self.kind == "key":
            if configured not in ("", "key"):
                raise _error()
            return
        if (
            self.kind not in {"gcp", "azure", "aws", "kubernetes"}
            or configured != self.kind or not audience or len(audience) > 2048
        ):
            raise _error()
        # Optional locally pinned audience further constrains a managed
        # deployment. It is configuration, never a secret.
        selected_audience = os.environ.get("DEEPINTSHIELD_RUNTIME_AUDIENCE", "").strip()
        if selected_audience and selected_audience != audience:
            raise _error()

    def get_token(self) -> str | None:
        if self.kind == "key":
            return None
        try:
            if self.kind == "kubernetes":
                # Require an explicitly projected audience-specific token. Do
                # not reuse the default Kubernetes API service-account token.
                selected = os.environ.get("DEEPINTSHIELD_RUNTIME_TOKEN_FILE", "").strip()
                if not selected:
                    raise _error()
                with Path(selected).expanduser().open("rb") as stream:
                    raw = stream.read(65537)
                if len(raw) > 65536:
                    raise _error("agent_runtime_assertion_invalid")
                return _jwt(raw.decode("ascii").strip())
            if self.kind == "aws":
                return self._aws_token()
            if self.kind == "azure" and (
                os.environ.get("IDENTITY_ENDPOINT") or os.environ.get("MSI_ENDPOINT")
            ):
                return self._azure_hosted_token()
            # Fixed link-local metadata endpoint, no proxy or redirects, no
            # gateway virtual key in these requests. These two native adapters
            # cover GCP metadata identity tokens and Azure VM managed identity.
            with httpx.Client(timeout=5, follow_redirects=False, trust_env=False) as client:
                if self.kind == "gcp":
                    response = client.get(
                        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/identity",
                        params={"audience": self.audience, "format": "full"},
                        headers={"Metadata-Flavor": "Google"},
                    )
                    if (
                        response.status_code != 200 or len(response.content) > 65536
                        or response.headers.get("Metadata-Flavor") != "Google"
                    ):
                        raise _error("agent_runtime_assertion_invalid")
                    return _jwt(response.text.strip())
                params = {"api-version": "2018-02-01", "resource": self.audience}
                # Omitted means the VM's system-assigned identity; never
                # inherit an unrelated AZURE_CLIENT_ID from another SDK.
                client_id = os.environ.get("DEEPINTSHIELD_RUNTIME_AZURE_CLIENT_ID", "").strip()
                if client_id:
                    params["client_id"] = client_id
                response = client.get(
                    "http://169.254.169.254/metadata/identity/oauth2/token",
                    params=params, headers={"Metadata": "true"},
                )
                if response.status_code != 200 or len(response.content) > 65536:
                    raise _error("agent_runtime_assertion_invalid")
                return _jwt(response.json().get("access_token"))
        except GovernanceConfigurationError:
            raise
        except ImportError:
            raise _error("credential_dependency_missing") from None
        except Exception:
            # Provider exceptions often embed credentials or entire HTTP
            # bodies; retain only the trusted stable code.
            raise _error("agent_runtime_assertion_invalid") from None

    def _azure_hosted_token(self) -> str:
        # App Service, Container Apps and other Azure-managed hosts expose
        # host-specific managed-identity endpoints. Use Microsoft's supported
        # host-aware credential, never DefaultAzureCredential (which could
        # select a developer login or an unrelated application secret).
        from azure.identity import ManagedIdentityCredential

        client_id = os.environ.get("DEEPINTSHIELD_RUNTIME_AZURE_CLIENT_ID", "").strip()
        credential = ManagedIdentityCredential(client_id=client_id or None)
        scope = self.audience if self.audience.endswith("/.default") else self.audience.rstrip("/") + "/.default"
        try:
            return _jwt(credential.get_token(scope).token)
        finally:
            credential.close()

    def _aws_token(self) -> str:
        # AWS outbound identity federation is explicitly enabled by the
        # account administrator. It returns a publicly verifiable JWT and is
        # available only from REGIONAL STS endpoints (not GetCallerIdentity).
        import boto3
        from botocore.config import Config

        region = os.environ.get("DEEPINTSHIELD_RUNTIME_AWS_REGION", "").strip()
        if not region or not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d", region):
            raise _error()
        session = boto3.Session(region_name=region)
        partition = session.get_partition_for_region(region)
        if partition not in {"aws", "aws-cn", "aws-us-gov"}:
            raise _error()
        suffix = "amazonaws.com.cn" if partition == "aws-cn" else "amazonaws.com"
        client = session.client(
            "sts", region_name=region,
            endpoint_url=f"https://sts.{region}.{suffix}",
            config=Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}),
        )
        try:
            method = getattr(client, "get_web_identity_token", None)
            if not callable(method):
                raise _error("credential_dependency_missing")
            result = method(Audience=[self.audience], DurationSeconds=300, SigningAlgorithm="RS256")
            return _jwt(result.get("WebIdentityToken"))
        finally:
            client.close()
