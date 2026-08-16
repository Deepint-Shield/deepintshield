<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 DeepIntShield contributors -->

# OpenClaw integration

OpenClaw is a Node/TypeScript runtime, so its model and local-tool boundaries
use two small integrations. Both feed the canonical Agentic workspace; there is
no legacy Agentic API or second policy model.

## 1. Route model traffic through the gateway

Generate the provider block from a Python client:

```python
import json
from deepintshield import DeepintShield

shield = DeepintShield.from_env()
cfg = shield.agentic.openclaw_config(
    models=[
        {
            "id": "gpt-4o-mini",
            "contextWindow": 128000,
            "maxTokens": 16384,
            "cost": {"input": 0.15, "output": 0.6},
        }
    ],
)
print(json.dumps(cfg, indent=2))
```

Merge that output into `openclaw.json` and select the model:

```json
{
  "models": {
    "providers": {
      "deepintshield": {
        "baseUrl": "https://app.deepintshield.com/v1",
        "apiKey": "${DIS_VK}",
        "api": "openai-completions"
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "deepintshield/gpt-4o-mini"
      }
    }
  }
}
```

## 2. Gate every local tool with the plugin

This directory is a loadable OpenClaw plugin, not just a pseudocode snippet. It
uses OpenClaw's typed `before_tool_call` hook and the canonical
`POST /api/agentic-new/decide` contract.

```bash
openclaw plugins install ./deepintshield/examples/openclaw
openclaw plugins enable deepintshield-agentic

export DIS_GATEWAY="https://app.deepintshield.com"
export DIS_VK="<virtual-key>"
# Required when the Agent Registry binds this VK to an Entra workload identity:
export DIS_AGENT_TOKEN="the-workload-assertion"

openclaw gateway restart
openclaw doctor
```

Optional acting identities use the same directory resolver as the Python SDK:

```bash
# Human acting through the agent (JIT-provisioned in Organization → Users):
export DIS_USER_EMAIL="alice@corp.example"

# Or a service account:
export DIS_USERNAME="billing-bot"
export DIS_PRINCIPAL_KIND="service_account"

# A pre-resolved directory subject may be supplied instead:
# export DIS_USER_SUBJECT="user:alice"
```

The agent identity is never accepted from OpenClaw context. The server derives
it from the authenticated virtual key and, when configured, verifies
`X-Agent-Token` against the workspace's Entra identity provider. The optional
email/username is resolved once at gateway startup and then served from memory.

For every tool call the plugin:

- hashes canonicalized parameters locally and sends only a `sha256:` digest;
- forwards run/session identifiers so the execution opens as one correlated
  Agentic network with clickable agent/tool/action decisions;
- uses the server-owned registry action class and authoritative `proceed`
  result (including shadow mode);
- blocks on denial, pending DeepIntShield approval, malformed response, timeout,
  gateway outage or missing credentials.

An unregistered tool is recorded by the gateway with conservative write/high
risk metadata. Observation never creates OpenFGA grants, so an operator still
must approve the tool/action and assign the appropriate workspace role or
permission before it can execute.

`DIS_AUTHZ_TIMEOUT_MS` controls the fail-closed decision timeout (default 5000,
bounded to 100–30000 ms). Network authorization necessarily has a decision
round trip; identity resolution is warmed at startup and the server can use its
bounded decision cache where immediate-revocation policy permits it.
