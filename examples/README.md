<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 DeepIntShield contributors -->

# deepintshield - examples

These examples target SDK **3.2.3** and DeepIntShield Server **3.2.3**.
Install the matching release with `pip install "deepintshield==3.2.3"`
and add the extras required by your example.

SDK and Server 3.2.3 include Entra gateway-runtime authentication and managed
LangGraph execution. For a complete local
`test2.py` / `math-agent` run, follow
[Microsoft and local gateway setup](../../deepintshield_docs/src/content/docs/enterprise/entra-local-gateway.mdx).
It installs `runtime,langgraph`, keeps Entra credentials in the gateway,
authorizes the SDK runtime key, and uses an empty requester for the autonomous
arithmetic example before normal code/access approval.

Every example is runnable and uses the real provider/framework SDK. Constructing
`DeepintShield` automatically arms supported agent frameworks; a binder is only
needed when model traffic must also be routed through the gateway. Set your VK
first:

```bash
export DEEPINTSHIELD_VIRTUAL_KEY="<virtual-key>"
# REQUIRED for every agentic example. A stable name selects this logical agent
# in the Registry and has no default: two workloads sharing a name
# share one registration, and an unnamed client stops with
# `agent_name_required` rather than resolving to whichever agent the key
# happens to be bound to.
export DEEPINTSHIELD_AGENT_NAME="my-agent"
# optional: point at a self-hosted / staging gateway
export DEEPINTSHIELD_BASE_URL="https://gateway.example.com"
# optional: the acting user or service account
export DEEPINTSHIELD_REQUESTER="user@example.com"
# for a new agent: the developer who registers it (your work e-mail)
export DEEPINTSHIELD_REGISTERED_BY="developer@example.com"

python examples/agentic/status.py        # is this agent governed yet?
python examples/openai/chat.py
python examples/openai/async_chat.py     # native AsyncOpenAI
python examples/openai/responses.py     # GPT-6 Astra via Responses
python examples/openai/responses.py --stream
python examples/agentic/decorator.py
python examples/langgraph/transparent.py
```

Install the matching extra per example, e.g. `pip install 'deepintshield[langgraph]==3.2.3'`.

The primary inference path is `shield.openai()` / `shield.async_openai()`, or
`OpenAI(**shield.openai_config())` when constructing a native client yourself.
Select a supported provider-qualified model through `DEEPINTSHIELD_MODEL` in
the generic examples. Framework binders return native model objects; native
graphs, agents, runners and workers still execute your application.

Install framework examples in the separate tested dependency combinations in
the [SDK README](../README.md#install). Current LlamaIndex requires
OpenAI 2; current PydanticAI/OpenAI Agents require OpenAI 3. The legacy `[all]`
extra does not guarantee all latest upstream releases can coexist.

Select model IDs enabled for your gateway and provider keys. Provider examples
accept `DEEPINTSHIELD_ANTHROPIC_MODEL` (default `claude-sonnet-4-6`) and
`DEEPINTSHIELD_GENAI_MODEL` (default `gemini-2.5-flash`), including their
passthrough examples. Override these values when your configured catalog or
provider access differs; a sample default does not grant access to a model.

### GPT-6 Astra: use Responses

Run `openai/responses.py` for Astra. It reads the gateway and virtual key from
the environment above and uses `shield.openai().responses.create(...)` with
`input`, `reasoning={"effort": "low"}`, and `response.output_text`.
`--stream` prints `response.output_text.delta` events. Set
`DEEPINTSHIELD_MODEL` to another Responses-compatible model if needed; the
general Chat examples retain their existing defaults.

The gateway can add authorized MCP tools even when a call omits `tools`.
Astra tool calling requires Responses, so changing only the model in
`shield.chat()` or `client.chat.completions.create()` can produce a 400.
Astra supports `low`, `medium`, `high`, `xhigh`, and `max` reasoning;
`none` is unsupported. Omit `temperature` and `top_p`; use
`max_output_tokens` if you need an output budget. See the
[official OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model).

For a new `DEEPINTSHIELD_AGENT_NAME`, a successful first native execution
capture raises a marked `RuntimeError` with code `agent_registration_pending`
(or `agent_runtime_enrollment_pending` when the run enrolled a runtime key)
and trusted summary/action fields. Its message is
`Registration pending. Reference <registration id> · runtime key SHA256:<thumbprint>. Contact your administrator. You will be notified at <email>.`
with unknown clauses omitted. The last sentence appears only when the gateway
confirms the address it will notify. Raw gateway detail and review links are
never rendered, and the SDK does not wait for approval.
An administrator opens **Agentic → Agents → Needs attention**, selects **Review** there or under **Agents →
Agent inventory**, and completes the review: choose an active human owner, configure workload credentials and
verify matching proof after discovery, review the completed code scan, and
select the exact actions/resource permissions. Complete **Runtime access** and **Validate changes**, then read the exact
proposal in **Review & decide** and select **Confirm approval and access**. This confirms the
registration, exact blueprint, and selected access plan together. Check the saved approval and selected grants, then inspect **Runtime access**
for server readiness and execution state. Use **Resume saved approval** for
incomplete access and resolve remaining requirements before rerunning. The interface owns the human-readable
explanation; examples do not catch normal governance outcomes to reformat them.

An expected identity declaration is not proof. Use **Approve as VK-trusted
(development only)** only when the deployment deliberately accepts unattested Virtual-Key trust,
not as a fix for missing credentials required by a provider-backed deployment.

Provision a workload identity once for a logical agent and environment, then
reuse it across runs and replicas. One human can own many agents. Use separate
identities for independently controlled permissions or revocation. In
**Agentic → Identities → Provisioning**, customer-approved management connections
can create or link Entra Agent ID, Keycloak, Auth0, or Okta identities. This is
separate from runtime **Identity providers**. Managed identities can be disabled
or retired; linked identities remain read-only and keep their existing
credentials. Provisioning never replaces workload proof, code review, or access
approval. See [the workload setup guide](https://aidocs.deepintshield.com/enterprise/agent-workload-identity/#provision-identities-from-deepintshield).

For shared Entra setup, save and validate an explicit empty identity list before
creating the first child. It admits nobody until provisioning adds a child ID.
**Authorize provisioning** reuses the selected verifier's settings; do not create
another verifier or invent a placeholder client ID for each agent.

Approval has no fixed run count or registration expiry, but credentials, the
accepted blueprint, grants, and policy must still allow every protected action.
**Block runtime drift** defaults off under **Policy & Access → Runtime policy →
Blueprint protection**. On requires review for every newly observed digest,
including clean changes and rollback; off retains mandatory risk-based scans
and does not release existing quarantine. Existing SDK change reporting is
sufficient; arbitrary host and unreported remote changes are outside coverage.
See the [simple-math-agent walkthrough](https://aidocs.deepintshield.com/sdk/agents-agentic/#first-run-simple-math-agent).

## Provisioned OIDC credentials

`agentic/provisioned_oidc.py` is a standard-library launcher for secret-based
Keycloak, Auth0, and Okta clients created by the provisioning connectors. Use
Vault Agent or your deployment's secret integration to render the selected
KV v2 record's inner `data.data` object to a file readable only by the workload.
Authenticate to Vault as the workload; do not give it the gateway's provisioning
Vault token or an identity-provider management credential.

The file has this shape after delivery (illustrative values only):

```json
{
  "ready": true,
  "client_id": "provisioned-client-id",
  "client_secret": "DELIVERED_BY_THE_SECRET_MANAGER",
  "issuer": "https://sso.example.com/realms/agents",
  "token_endpoint": "https://sso.example.com/realms/agents/protocol/openid-connect/token",
  "audience": "deepintshield",
  "scopes": ["workload"]
}
```

Keep the gateway, virtual key, and friendly agent name in the ordinary
environment, then launch the example without a shell or secrets in arguments:

```bash
python examples/agentic/provisioned_oidc.py \
  --provider keycloak --record /run/secrets/agent-oidc.json \
  -- python examples/langgraph/transparent.py
```

Select `auth0` or `okta` for the corresponding connection. The launcher refuses
staged records (`ready: false`), missing credentials, and invalid token endpoints.
It does not print the record or retrieve it from Vault. Restrict access to both
the file and the child process environment. The record's `ready` flag confirms
credential delivery, not registry approval or permission to call tools.

The launcher sets the following values for the child process:

| Environment variable | Source or behavior |
| --- | --- |
| `OIDC_CLIENT_ID` | Ready record's `client_id` |
| `OIDC_CLIENT_SECRET` | Ready record's `client_secret` |
| `OIDC_TOKEN_ENDPOINT` | Ready record's `token_endpoint` |
| `OIDC_CLIENT_AUTH_METHOD` | `client_secret_post`, matching these provisioned clients |
| `OIDC_SCOPES` | Space-separated record scopes; an empty list requests no scopes |
| `OIDC_AUDIENCE` | Record audience for Auth0; empty for Keycloak/Okta, whose configured issuers set token audience |
| `OIDC_RESOURCE` | Empty; these connector configurations do not use a token-request resource parameter |

SDK 3.2.1 includes the `OIDC_SCOPES` override. Outside this
launcher, leaving that variable unset uses the verification provider's default
scopes; setting it to empty requests no scopes. The gateway continues to verify
issuer, audience, and the exact expected identity. Configure its shared Generic
OIDC provider without a single-client default, as described in the workload guide.

For Auth0/Okta public-key clients, register only the public RSA JWKS and keep
the private key with the workload. Configure `OIDC_CLIENT_AUTH_METHOD=private_key_jwt`,
`OIDC_PRIVATE_KEY_FILE`, and the matching `OIDC_PRIVATE_KEY_KID` separately;
this secret-based launcher does not support that flow. Okta rotation adds
public keys while retaining existing ones. The other connectors do not offer
rotation, and none offers automatic secret rotation. Entra uses an existing
blueprint with customer-configured host federation and the Azure SDK extra,
so it also does not use this launcher.

---

## Use cases (framework-agnostic)

| Folder | What it shows |
| --- | --- |
| [multimodal/](multimodal/README.md) | Native OpenAI text and media examples for all 29 provider identities, explicit model selection, synthetic image/PDF fixtures and offline request previews |
| `agentic/status.py` | Startup healthcheck - refuse to serve traffic this agent is not enrolled for |
| `agentic/obligations.py` | What each MASK obligation redacts, positional arguments included |
| `agentic/decorator.py` | Advanced explicit per-function `@shield.agentic.tool` API |
| `agentic/decide.py` | Advanced direct PDP probe returning the raw verdict |
| `agentic/identity.py` | Advanced diagnostics for the discovered Entra/ZeroID/OIDC binding |
| `agentic/provisioned_oidc.py` | Start a secret-based OIDC workload from a mounted Vault record only after delivery is ready |
| `agentic/guard.py` | Native LangChain tool execution with automatic enforcement |
| `transparent/connection.py` | Point any OpenAI-compatible client at the gateway via `shield.connection()` |
| `transparent/create_headers.py` | Gateway `shield.create_headers()` injector |
| `transparent/universal_openai.py` | Raw-HTTP pattern (n8n / Flowise / Dify / any platform) |
| `rag/guard_retriever.py` | Synchronous and asynchronous post-retrieval safety filtering, once per retrieval |
| `rag/guard_embedder.py` | Pre-embedding input screening (PII / injection) |

## Per provider - chat & RAG (transparent model traffic)

For private corpora, authenticate the user and enforce document permissions in
your application or retriever before sending content to DeepintShield. The RAG
examples apply safety policies to already authorized content. A safety `allow`
decision does not grant document access.

Before running a provider's `rag.py`, register a RAG source in the virtual
key's workspace and select it using either its ID or its exact registered name:

```bash
export DEEPINTSHIELD_RAG_SOURCE_ID="qa-support"
python examples/openai/rag.py
```

The parameter remains `source_id` for both forms. Name lookup requires the
gateway's ID-or-name resolution support; no source-list permission or SDK
lookup is required. The examples default to `kb-public`, which must be a
registered ID or name in your workspace. Unknown references return HTTP 404
and ambiguous names return HTTP 409; use the canonical ID to disambiguate.
See the [RAG contract](../README.md#rag) for per-chunk overrides, renames and
workspace defaults.

These folders demonstrate the SDK's native client facades. They are not the
gateway provider inventory. The `openai/` and `transparent/` patterns can use
provider-qualified models for all 29 built-in gateway provider identities,
including [DeepSeek](../../deepintshield_server/docs/providers/supported-providers/deepseek.mdx),
[Amazon Bedrock Mantle](../../deepintshield_server/docs/providers/supported-providers/bedrock-mantle.mdx),
[Sarvam AI](../../deepintshield_server/docs/providers/supported-providers/sarvam.mdx),
and [Wafer](../../deepintshield_server/docs/providers/supported-providers/wafer.mdx).
No provider-specific SDK wrapper is implied by the gateway registration.

For vision, PDFs, generated images, speech, transcription, video and uploaded
file references, use the [multimodal runner](multimodal/README.md). Its
provider/operation table identifies each example's input and endpoint, and its
`--list` and `--dry-run` modes need no credentials.

| Folder | Files |
| --- | --- |
| `openai/` | `responses.py` (Astra, with `--stream`), `chat.py`, `async_chat.py`, `rag.py`, `mcp.py`, `agent.py` |
| `anthropic/` | `chat.py`, `rag.py`, `mcp.py` |
| `bedrock/` | `chat.py`, `rag.py` |
| `genai/` | `chat.py`, `rag.py` |
| `litellm/` | `chat.py`, `rag.py` |
| `passthrough/` | `openai_chat.py`, `anthropic_chat.py`, `genai_chat.py` |

## Per agent framework - transparent binder + automatic tool enforcement

Each folder has `transparent.py` (native model via the gateway) and
`tool_gating.py` (the framework's ordinary execution path through the PDP).

| Folder | Transparent binder | Tool enforcement |
| --- | --- | --- |
| `langgraph/` | `transparent.py`, `agent.py`, `custom_nodes.py` | `tool_gating.py` |
| `langchain/` | `chat.py`, `rag.py`, `mcp.py` | (use `agentic/decorator.py`) |
| `openai_agents/` | `transparent.py` | `tool_gating.py` |
| `llamaindex/` | `transparent.py`, `rag.py` | `tool_gating.py` |
| `autogen/` | `transparent.py` | `tool_gating.py` |
| `pydanticai/` | `transparent.py`, `chat.py` | `tool_gating.py` |

## Native-hook and durable frameworks

Temporal, Strands and Google ADK are armed automatically at their final native
execution boundary. The explicit hook/plugin objects remain supported for older
applications, but new code does not attach anything per worker or agent.

| Folder | Required application change | Governed boundary |
| --- | --- | --- |
| `temporal/worker.py` | none after client construction | injected `ActivityInboundInterceptor` |
| `temporal/inference.py` | `shield.async_openai()` inside a native activity | injected `ActivityInboundInterceptor`; workflow remains deterministic |
| `strands/agent.py` | `shield.strands().model(id)` routes inference | final `ToolExecutor` dispatch after callbacks and middleware |
| `google_adk/agent.py` | `shield.google_adk().model(id)` uses ADK's native LiteLLM connector | final normal/live/threaded dispatch |
| `hermes/plugin.py` | load the host bootstrap plugin | central `model_tools` dispatcher |
| `openclaw/README.md` | `shield.agentic.openclaw_config()` + TS plugin | `models.providers` + `before_tool_call` |

## MCP

| Example | Install extra | Integration |
| --- | --- | --- |
| `openai/mcp.py` | `deepintshield[openai-agents]` | OpenAI Agents owns its MCP session and agent loop; result/error callbacks preserve coded authorization failures. |
| `anthropic/mcp.py` | `deepintshield[anthropic-mcp]` | `shield.mcp.connect()` opens an official MCP session; Anthropic's helper converts tools and explicit dispatch preserves exceptions. |
| `langchain/mcp.py` | `deepintshield[langchain-mcp]` | The maintained LangChain adapter uses `shield.mcp.connection()` and an interceptor for governed tool errors. |

These samples use native MCP/framework types. Configure an MCP server and the
calling key's permissions before running; the model-loop samples expect the
DeepWiki tool mentioned in their prompt. Discovery permission does not imply
execution permission. The samples close their owned clients on exit.

For delegated calls, pass the request-scoped subject token in `extra_headers`
to `connect()` or `connection()` and create a separate session for each caller.
Keep credentials in transport headers. New applications should use these
native paths instead of the deprecated 2.x `Tool` conversion helpers.

---

**Two ways to protect an agent:**

1. **Transparent** - optionally get your model/embeddings from a binder
   (`shield.bind("llamaindex").llm(...)`, `shield.bind("langgraph").model(...)`,
   `shield.connection()`). Guardrails,
   observability and identity run server-side with no changes to your agent code.
2. **Enforcement** - supported agent frameworks are gated automatically.
   Keep native `compile()`, `invoke()`, and `run()` calls.
   `@shield.agentic.tool(...)` and `shield.rag.guard_retriever(...)` are advanced
   framework-independent boundaries.
