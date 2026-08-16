<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 DeepIntShield contributors -->

# deepintshield - examples

Every example is runnable and uses the real provider/framework SDK. Constructing
`DeepintShield` automatically arms supported agent frameworks; a binder is only
needed when model traffic must also be routed through the gateway. Set your VK
first:

```bash
export DEEPINTSHIELD_VIRTUAL_KEY="<virtual-key>"
# optional: point at a self-hosted / staging gateway
export DEEPINTSHIELD_BASE_URL="https://gateway.example.com"
# stable Agentic identity and acting principal
export DEEPINTSHIELD_AGENT_NAME="my-agent"
export DEEPINTSHIELD_REQUESTER="user@example.com"

python examples/openai/chat.py
python examples/agentic/decorator.py
python examples/crewai/transparent.py
```

Install the matching extra per example, e.g. `pip install 'deepintshield[crewai]'`.

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
| `agentic/decorator.py` | Advanced explicit per-function `@shield.agentic.tool` API |
| `agentic/decide.py` | Advanced direct PDP probe returning the raw verdict |
| `agentic/identity.py` | Advanced diagnostics for the discovered Entra/ZeroID/OIDC binding |
| `agentic/guard.py` | Native LangChain tool execution with automatic enforcement |
| `transparent/connection.py` | Point any OpenAI-compatible client at the gateway via `shield.connection()` |
| `transparent/create_headers.py` | Portkey-style `shield.create_headers()` injector |
| `transparent/universal_openai.py` | Raw-HTTP pattern (n8n / Flowise / Dify / any platform) |
| `rag/guard_retriever.py` | Post-retrieval chunk filtering (ACL / provenance / injection) |
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

`openai/mcp.py`, `anthropic/mcp.py`, `langchain/mcp.py` - the same `Tool` /
`MCPClient` API drives any MCP server connected to your gateway.

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
