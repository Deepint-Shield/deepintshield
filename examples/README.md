# deepintshield - examples

Every example is runnable and uses the real provider/framework SDK. The only
DeepintShield-specific lines are `shield = DeepintShield.from_env()` plus one
binder or wrapper call. Set your VK first:

```bash
export DEEPINTSHIELD_VIRTUAL_KEY="sk-..."
# optional: point at a self-hosted / staging gateway
export DEEPINTSHIELD_BASE_URL="https://gateway.example.com"

python examples/openai/chat.py
python examples/agentic/decorator.py
python examples/crewai/transparent.py
```

Install the matching extra per example, e.g. `pip install 'deepintshield[crewai]'`.

---

## Use cases (framework-agnostic)

| Folder | What it shows |
| --- | --- |
| `agentic/decorator.py` | Gate any function with `@shield.agentic.tool` (ALLOW/DENY/MASK/approval) |
| `agentic/decide.py` | Direct PDP probe - raw verdict without raising |
| `agentic/identity.py` | Inspect the auto-discovered Entra/ZeroID/OIDC binding |
| `transparent/connection.py` | Point any OpenAI-compatible client at the gateway via `shield.connection()` |
| `transparent/create_headers.py` | Portkey-style `shield.create_headers()` injector |
| `transparent/universal_openai.py` | Raw-HTTP pattern (n8n / Flowise / Dify / any platform) |
| `rag/guard_retriever.py` | Post-retrieval chunk filtering (ACL / provenance / injection) |
| `rag/guard_embedder.py` | Pre-embedding input screening (PII / injection) |

## Per provider - chat & RAG (transparent model traffic)

| Folder | Files |
| --- | --- |
| `openai/` | `chat.py`, `rag.py`, `mcp.py`, `agent.py` |
| `anthropic/` | `chat.py`, `rag.py`, `mcp.py` |
| `bedrock/` | `chat.py`, `rag.py` |
| `genai/` | `chat.py`, `rag.py` |
| `litellm/` | `chat.py`, `rag.py` |
| `passthrough/` | `openai_chat.py`, `anthropic_chat.py`, `genai_chat.py` |

## Per agent framework - transparent binder + tool enforcement

Each folder has `transparent.py` (native model via the gateway) and
`tool_gating.py` (wrap the framework's tools through the PDP).

| Folder | Transparent binder | Tool enforcement |
| --- | --- | --- |
| `langgraph/` | `transparent.py`, `agent.py`, `custom_nodes.py` | `tool_gating.py` |
| `langchain/` | `chat.py`, `rag.py`, `mcp.py` | (use `agentic/decorator.py`) |
| `crewai/` | `transparent.py` | `tool_gating.py` |
| `openai_agents/` | `transparent.py` | `tool_gating.py` |
| `llamaindex/` | `transparent.py`, `rag.py` | `tool_gating.py` |
| `autogen/` | `transparent.py` | `tool_gating.py` |
| `pydanticai/` | `transparent.py`, `chat.py` | `tool_gating.py` |

## Native-hook & durable frameworks

These attach a **single object** at construction — the adapter rides the
framework's own interceptor / hook / plugin system (no monkey-patch).

| Folder | Attach point | Native hook |
| --- | --- | --- |
| `temporal/worker.py` | `Worker(interceptors=[shield.agentic.temporal()])` | `ActivityInboundInterceptor` |
| `strands/agent.py` | `Agent(hooks=[shield.agentic.strands()])` | `BeforeToolInvocationEvent` |
| `google_adk/agent.py` | `InMemoryRunner(plugins=[shield.agentic.google_adk()])` | `BasePlugin.before_tool_callback` |
| `hermes/plugin.py` | `register(ctx): shield.agentic.hermes(ctx)` | plugin `pre_tool_call` |
| `openclaw/README.md` | `shield.agentic.openclaw_config()` + TS plugin | `models.providers` + `before_tool_call` |

## MCP

`openai/mcp.py`, `anthropic/mcp.py`, `langchain/mcp.py` - the same `Tool` /
`MCPClient` API drives any MCP server connected to your gateway.

---

**Two ways to protect an agent:**

1. **Transparent** - get your model/embeddings from a binder (`shield.crewai().llm(...)`,
   `shield.langgraph().model(...)`, `shield.connection()`). Guardrails,
   observability and identity run server-side with no changes to your agent code.
2. **Enforcement** - one wrapper adds local tool gating and chunk filtering:
   `shield.agentic.<framework>(...)`, `@shield.agentic.tool(...)`,
   `shield.rag.guard_retriever(...)`.
