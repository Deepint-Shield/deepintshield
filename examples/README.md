<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 DeepIntShield contributors -->

# deepintshield - examples

These examples target SDK **2.7.2** and DeepIntShield Server **2.7.2**.
Install the matching release with `pip install "deepintshield==2.7.2"`
and add the extras required by your example.

Every example is runnable and uses the real provider/framework SDK. Constructing
`DeepintShield` automatically arms supported agent frameworks; a binder is only
needed when model traffic must also be routed through the gateway. Set your VK
first:

```bash
export DEEPINTSHIELD_VIRTUAL_KEY="<virtual-key>"
# REQUIRED for every agentic example. The agent name is this workload's
# identity in the Registry and has no default: two workloads sharing a name
# share one registration, and an unnamed client stops with
# `agent_name_required` rather than resolving to whichever agent the key
# happens to be bound to.
export DEEPINTSHIELD_AGENT_NAME="my-agent"
# optional: point at a self-hosted / staging gateway
export DEEPINTSHIELD_BASE_URL="https://gateway.example.com"
# optional: the acting user or service account
export DEEPINTSHIELD_REQUESTER="user@example.com"

python examples/agentic/status.py        # is this agent governed yet?
python examples/openai/chat.py
python examples/agentic/decorator.py
python examples/crewai/transparent.py
```

Install the matching extra per example, e.g. `pip install 'deepintshield[crewai]'`.

Select model IDs enabled for your gateway and provider keys. Provider examples
accept `DEEPINTSHIELD_ANTHROPIC_MODEL` (default `claude-sonnet-4-6`) and
`DEEPINTSHIELD_GENAI_MODEL` (default `gemini-2.5-flash`), including their
passthrough examples. Override these values when your configured catalog or
provider access differs; a sample default does not grant access to a model.

For a new `DEEPINTSHIELD_AGENT_NAME`, a successful first native execution
capture raises a marked `RuntimeError` with code `agent_registration_pending`
and trusted summary/action fields. Raw gateway detail is never rendered.
Open **Agentic → Work Queue**, select **Review** there or under **Assets →
Agents**, complete the single registration form, and select **Approve &
activate**. Then rerun the same example. The interface owns the human-readable
explanation; examples do not catch normal governance outcomes to reformat them.

---

## Use cases (framework-agnostic)

| Folder | What it shows |
| --- | --- |
| `agentic/status.py` | Startup healthcheck - refuse to serve traffic this agent is not enrolled for |
| `agentic/obligations.py` | What each MASK obligation redacts, positional arguments included |
| `agentic/decorator.py` | Advanced explicit per-function `@shield.agentic.tool` API |
| `agentic/decide.py` | Advanced direct PDP probe returning the raw verdict |
| `agentic/identity.py` | Advanced diagnostics for the discovered Entra/ZeroID/OIDC binding |
| `agentic/guard.py` | Native LangChain tool execution with automatic enforcement |
| `transparent/connection.py` | Point any OpenAI-compatible client at the gateway via `shield.connection()` |
| `transparent/create_headers.py` | Portkey-style `shield.create_headers()` injector |
| `transparent/universal_openai.py` | Raw-HTTP pattern (n8n / Flowise / Dify / any platform) |
| `rag/guard_retriever.py` | Synchronous and asynchronous post-retrieval filtering (ACL / provenance / injection), once per retrieval |
| `rag/guard_embedder.py` | Pre-embedding input screening (PII / injection) |

## Per provider - chat & RAG (transparent model traffic)

These folders demonstrate the SDK's native client facades. They are not the
gateway provider inventory. The `openai/` and `transparent/` patterns can use
provider-qualified models for all 29 built-in gateway provider identities,
including [DeepSeek](../../deepintshield_server/docs/providers/supported-providers/deepseek.mdx),
[Amazon Bedrock Mantle](../../deepintshield_server/docs/providers/supported-providers/bedrock-mantle.mdx),
[Sarvam AI](../../deepintshield_server/docs/providers/supported-providers/sarvam.mdx),
and [Wafer](../../deepintshield_server/docs/providers/supported-providers/wafer.mdx).
No provider-specific SDK wrapper is implied by the gateway registration.

| Folder | Files |
| --- | --- |
| `openai/` | `chat.py`, `rag.py`, `mcp.py`, `agent.py` |
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
| `crewai/` | `transparent.py` | `tool_gating.py` |
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
| `strands/agent.py` | none after client construction | final `ToolExecutor` dispatch |
| `google_adk/agent.py` | none after client construction | final normal/live/threaded dispatch |
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
   (`shield.bind("crewai").llm(...)`, `shield.bind("langgraph").model(...)`,
   `shield.connection()`). Guardrails,
   observability and identity run server-side with no changes to your agent code.
2. **Enforcement** - supported agent frameworks are gated automatically.
   Keep native `compile()`, `invoke()`, `run()`, and `kickoff()` calls.
   `@shield.agentic.tool(...)` and `shield.rag.guard_retriever(...)` are advanced
   framework-independent boundaries.
