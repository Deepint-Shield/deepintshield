# OpenClaw integration

OpenClaw is a Node/TypeScript agent runtime, so it integrates in two layers.

## Layer 1 — LLM gateway (zero code, Python-generated config)

Route every OpenClaw model call through the DeepintShield gateway — VK auth,
semantic/exact cache, request coalescing, guardrails, budgets, cost analytics,
catalog fallbacks — with **no code change**. Generate the provider block from a
live client:

```python
# gen_openclaw_config.py
import json
from deepintshield import DeepintShield

shield = DeepintShield.from_env()
cfg = shield.agentic.openclaw_config(
    models=[{"id": "gpt-4o-mini", "contextWindow": 128000, "maxTokens": 16384,
             "cost": {"input": 0.15, "output": 0.6}}],
)
print(json.dumps(cfg, indent=2))
```

Merge the output into `openclaw.json` and set the default model:

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
  "agents": { "defaults": { "model": { "primary": "deepintshield/gpt-4o-mini" } } }
}
```

## Layer 2 — in-process tool governance (TypeScript plugin)

OpenClaw plugins run in-process in the Node gateway, so the Python SDK cannot be
embedded. Ship governance as a thin TS plugin that calls the gateway's REST
`/decide` + approvals + code-threat-scan APIs on OpenClaw's native hooks:

```ts
// deepintshield.plugin.ts  (openclaw.plugin.json + definePluginEntry)
import { definePluginEntry } from "openclaw/plugin";

const GW = process.env.DIS_GATEWAY!;        // e.g. https://app.deepintshield.com
const VK = process.env.DIS_VK!;

async function decide(tool: string, args: unknown, ctx: any) {
  const r = await fetch(`${GW}/api/agentic-security/decide`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${VK}`, "Content-Type": "application/json" },
    body: JSON.stringify({ tool, args_digest: args, principal: `agent:${ctx.agentId}` }),
  });
  return r.json();                          // { verdict, reason, decision_id, ... }
}

export default definePluginEntry({
  register(api) {
    api.registerHook("before_tool_call", async (call, ctx) => {
      const d = await decide(call.toolName, call.args, ctx);
      if (d.verdict === "DENY") return { block: true, blockReason: d.reason };
      if (d.verdict === "REQUIRE_APPROVAL")
        return { requireApproval: { title: call.toolName, severity: "high", timeout: 300 } };
      return {};                            // ALLOW
    });
    api.registerHook("after_tool_call", async (call, res, ctx) => {
      // observed-behavior event for fingerprinting / drift / audit
    });
    api.registerHook("before_install", async (staged) => {
      // POST staged skill/plugin source to /api/.../scan for code-threat findings
    });
  },
});
```

> The Python SDK ships Layer 1 today (`shield.agentic.openclaw_config()`). Layer 2
> is the thin TypeScript plugin above — it reuses the same REST decide/approval/scan
> APIs the Python adapters call, so policy, grants, and audit are identical.
