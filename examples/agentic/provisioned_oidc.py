# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepIntShield contributors
"""Start a secret-based OIDC workload from an already-mounted Vault ready record.

Vault Agent or the customer's secret integration must render the inner KV v2
data.data object using the workload's own Vault authentication. This launcher
does not fetch secrets, provision identities, or approve agent execution.

Example (keep virtual key, gateway, and agent name in the normal environment):
    python examples/agentic/provisioned_oidc.py \
        --provider keycloak --record /run/secrets/agent-oidc.json \
        -- python examples/langgraph/transparent.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def workload_environment(record: Any, provider: str) -> dict[str, str]:
    """Validate a ready record without including any of its values in errors."""
    if provider not in {"keycloak", "auth0", "okta"}:
        raise ValueError("Unsupported secret-based OIDC provider.")
    if not isinstance(record, dict) or record.get("ready") is not True:
        raise ValueError("The workload credential record is not ready.")
    required = ("client_id", "client_secret", "token_endpoint")
    if any(
        not isinstance(record.get(key), str)
        or not record[key].strip()
        or "\x00" in record[key]
        for key in required
    ):
        raise ValueError("The ready record is missing valid workload credentials.")
    try:
        endpoint = urlsplit(record["token_endpoint"])
        valid_endpoint = (
            endpoint.scheme == "https"
            and bool(endpoint.hostname)
            and endpoint.port in {None, 443}
            and endpoint.username is None
            and endpoint.password is None
            and not endpoint.query
            and not endpoint.fragment
            and not any(c.isspace() for c in record["token_endpoint"])
        )
    except ValueError:
        valid_endpoint = False
    if not valid_endpoint:
        raise ValueError("The ready record has an invalid HTTPS token endpoint.")

    # A connection with no scopes may be serialized as a null slice by Go.
    scopes = record.get("scopes")
    if scopes is None:
        scopes = []
    if not isinstance(scopes, list) or any(
        not isinstance(scope, str)
        or not scope
        or any(ord(c) < 0x21 or ord(c) > 0x7E or c in {'"', "\\"} for c in scope)
        for scope in scopes
    ):
        raise ValueError("The ready record has invalid OAuth scopes.")

    audience = record.get("audience", "") if provider == "auth0" else ""
    if not isinstance(audience, str) or "\x00" in audience:
        raise ValueError("The ready record has an invalid audience.")
    if provider == "auth0" and not audience.strip():
        raise ValueError("The Auth0 ready record is missing the API audience.")

    environment = dict(os.environ)
    environment.update(
        OIDC_CLIENT_ID=record["client_id"],
        OIDC_CLIENT_SECRET=record["client_secret"],
        OIDC_TOKEN_ENDPOINT=record["token_endpoint"],
        OIDC_CLIENT_AUTH_METHOD="client_secret_post",
        OIDC_SCOPES=" ".join(scopes),
        OIDC_AUDIENCE=audience,
        OIDC_RESOURCE="",
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("keycloak", "auth0", "okta"), required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a workload command after --.")

    try:
        with args.record.open("rb") as stream:
            payload = stream.read(128 * 1024 + 1)
        if len(payload) > 128 * 1024:
            raise ValueError("The workload credential record is too large.")
        try:
            record = json.loads(payload)
        except (ValueError, UnicodeError):
            raise ValueError("The workload credential record is not valid JSON.") from None
        environment = workload_environment(record, args.provider)
    except OSError:
        parser.error("The workload credential record could not be read.")
    except ValueError as exc:
        parser.error(str(exc))

    # exec passes values through the process environment, never through a shell
    # or command arguments. No credential record is printed, even on failure.
    try:
        os.execvpe(command[0], command, environment)
    except OSError:
        parser.error("The workload command could not be started.")


if __name__ == "__main__":
    main()
