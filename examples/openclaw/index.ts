import { createHash } from "node:crypto";

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

type JsonRecord = Record<string, unknown>;

type ToolEvent = {
  toolName: string;
  params: JsonRecord;
  runId?: string;
  toolCallId?: string;
};

type ToolContext = {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  toolCallId?: string;
};

type Decision = {
  proceed?: boolean;
  verdict?: string;
  reason?: string;
  approval_id?: string;
  decision_id?: string;
};

const gateway = (process.env.DIS_GATEWAY ?? "").replace(/\/+$/, "");
const virtualKey = process.env.DIS_VK ?? "";
const agentToken = process.env.DIS_AGENT_TOKEN ?? "";
const configuredSubject = process.env.DIS_USER_SUBJECT ?? "";
const configuredEmail = process.env.DIS_USER_EMAIL ?? "";
const configuredUsername = process.env.DIS_USERNAME ?? "";
const principalKind =
  process.env.DIS_PRINCIPAL_KIND === "service_account"
    ? "service_account"
    : "user";
const timeoutMs = Math.min(
  30_000,
  Math.max(100, Number(process.env.DIS_AUTHZ_TIMEOUT_MS ?? "5000") || 5000),
);

let principalPromise: Promise<string> | undefined;

function hashHex(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function lengthPrefixed(value: string): Buffer {
  const encoded = Buffer.from(value, "utf8");
  const length = Buffer.alloc(8);
  length.writeBigUInt64BE(BigInt(encoded.length));
  return Buffer.concat([length, encoded]);
}

// Byte-compatible with AgenticNewSanitizeKey. Existing canonical identifiers
// stay readable; lossy identifiers retain 128 digest bits to avoid collisions.
function registryKey(value: string): string {
  const source = value.trim();
  if (
    source.length <= 128 &&
    !source.startsWith("-") &&
    !source.endsWith("-") &&
    /^[a-z0-9._-]+$/.test(source)
  ) {
    return source;
  }
  let readable = source
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!readable) readable = "id";
  readable = readable.slice(0, 91).replace(/[-._]+$/g, "") || readable.slice(0, 91);
  const digest = hashHex(
    Buffer.concat([
      Buffer.from("deepintshield/registry-key/v1", "utf8"),
      lengthPrefixed(source),
    ]),
  ).slice(0, 32);
  return `v1-${readable}--${digest}`;
}

function inferActionClass(name: string): "read" | "write" {
  const tokens = new Set(
    name
      .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
      .toLowerCase()
      .match(/[a-z0-9]+/g) ?? [],
  );
  const mutations = [
    "append",
    "approve",
    "archive",
    "assign",
    "cancel",
    "charge",
    "commit",
    "create",
    "delete",
    "deploy",
    "disable",
    "drop",
    "edit",
    "enable",
    "grant",
    "insert",
    "invite",
    "merge",
    "modify",
    "move",
    "patch",
    "pay",
    "post",
    "publish",
    "put",
    "remove",
    "rename",
    "revoke",
    "send",
    "set",
    "submit",
    "transfer",
    "update",
    "upload",
    "upsert",
    "write",
  ];
  if (mutations.some((token) => tokens.has(token))) return "write";
  const reads = [
    "check",
    "count",
    "describe",
    "download",
    "fetch",
    "find",
    "get",
    "health",
    "inspect",
    "list",
    "lookup",
    "preview",
    "query",
    "read",
    "retrieve",
    "search",
    "select",
    "status",
    "view",
  ];
  return reads.some((token) => tokens.has(token)) ? "read" : "write";
}

function permissionSlug(value: string): string {
  let result = "";
  let lastDash = false;
  for (const char of value.trim().toLowerCase()) {
    if (/^[a-z0-9]$/.test(char)) {
      result += char;
      lastDash = false;
    } else if ("_-.:".includes(char)) {
      result += "-";
      lastDash = true;
    } else if (" \t/\\".includes(char) && !lastDash && result) {
      result += "-";
      lastDash = true;
    }
  }
  return result.replace(/^-+|-+$/g, "");
}

// Byte-compatible with the Go PermissionSubject helper used by custom roles.
function permissionSubject(key: string): string {
  const canonical = key.trim().toLowerCase();
  const readable =
    permissionSlug(canonical).slice(0, 64).replace(/[-._]+$/g, "") || "tool-write";
  const digest = hashHex(
    Buffer.concat([
      Buffer.from("deepintshield/openfga/object-id/v1", "utf8"),
      lengthPrefixed("permission"),
      lengthPrefixed(canonical),
    ]),
  ).slice(0, 32);
  return `permission:v1-${readable}--${digest}`;
}

function stableJson(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? JSON.stringify(value) : JSON.stringify(String(value));
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value as JsonRecord;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(String(value));
}

function argsDigest(params: JsonRecord): string {
  return `sha256:${hashHex(stableJson({ args: [], kwargs: params }))}`;
}

function headers(): Record<string, string> {
  const result: Record<string, string> = {
    Authorization: `Bearer ${virtualKey}`,
    "Content-Type": "application/json",
  };
  if (agentToken) result["X-Agent-Token"] = agentToken;
  return result;
}

async function post(path: string, body: JsonRecord): Promise<Response> {
  if (!gateway || !virtualKey) {
    throw new Error("DIS_GATEWAY and DIS_VK are required");
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${gateway}${path}`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

function localSubject(value: string, kind: "user" | "service_account"): string {
  const normalized = value.trim().toLowerCase();
  const typed = normalized.match(/^(user|service_account):(.*)$/);
  const effectiveKind = (typed?.[1] ?? kind) as "user" | "service_account";
  const identifier = typed?.[2]?.trim() ?? normalized;
  let slug = identifier.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  const digest = hashHex(identifier).slice(0, 32);
  if (!slug) {
    slug = `id-${digest}`;
  } else if (slug !== identifier) {
    slug = `${slug.slice(0, 63).replace(/-+$/g, "")}-${digest}`;
  }
  if (slug.length > 96) {
    slug = `${slug.slice(0, 63).replace(/-+$/g, "")}-${digest}`;
  }
  return `${effectiveKind}:${slug}`;
}

async function resolvePrincipal(): Promise<string> {
  if (configuredSubject) return localSubject(configuredSubject, principalKind);
  const identifier = configuredEmail || configuredUsername;
  if (!identifier) return "";
  const fallback = localSubject(identifier, principalKind);
  try {
    const response = await post("/api/agentic-new/identity/resolve", {
      ...(configuredEmail ? { email: configuredEmail } : {}),
      ...(configuredUsername ? { username: configuredUsername } : {}),
      kind: principalKind,
    });
    if (!response.ok) return fallback;
    const payload = (await response.json()) as { subject?: unknown };
    return typeof payload.subject === "string" && payload.subject
      ? payload.subject
      : fallback;
  } catch {
    // Match the Python SDK: keep a deterministic subject and retry after this
    // process restarts. Authorization still fails closed if tuples are absent.
    return fallback;
  }
}

function actingPrincipal(): Promise<string> {
  principalPromise ??= resolvePrincipal();
  return principalPromise;
}

async function decide(event: ToolEvent, ctx: ToolContext): Promise<Decision> {
  const toolKey = registryKey(event.toolName);
  const actionClass = inferActionClass(event.toolName);
  const executionId =
    event.runId ?? ctx.runId ?? event.toolCallId ?? ctx.toolCallId ?? "";
  const response = await post("/api/agentic-new/decide", {
    // The gateway fills this from the authenticated VK and rejects conflicting
    // client claims. Never trust OpenClaw's display agent id as a GAF identity.
    agent: "",
    user: await actingPrincipal(),
    permission: "holder",
    object: permissionSubject(`${toolKey}:${actionClass}`),
    tool: `tool:${toolKey}`,
    action: "invoke",
    action_class: actionClass,
    args_digest: argsDigest(event.params),
    execution_id: executionId,
    session_id: ctx.sessionId ?? ctx.sessionKey ?? "",
  });
  if (!response.ok) {
    throw new Error(`authorization endpoint returned HTTP ${response.status}`);
  }
  return (await response.json()) as Decision;
}

export default definePluginEntry({
  id: "deepintshield-agentic",
  name: "DeepIntShield Agentic",
  description:
    "Fail-closed Agentic authorization and execution correlation for every OpenClaw tool call",
  register(api) {
    // Warm the optional Entra/service-account directory binding outside the
    // first tool's critical path. The hook below remains correct if startup
    // hooks are unavailable or the warm-up has not finished.
    api.on("gateway_start", async () => {
      await actingPrincipal();
    });

    api.on("before_tool_call", async (event: ToolEvent, ctx: ToolContext) => {
      try {
        const decision = await decide(event, ctx);
        if (decision.proceed === true) return {};

        const approval = decision.approval_id
          ? ` Approval ${decision.approval_id} is queued in DeepIntShield; retry after it is resolved.`
          : "";
        return {
          block: true,
          blockReason: `DeepIntShield ${decision.verdict ?? "DENY"}: ${
            decision.reason ?? "authorization did not permit execution"
          }.${approval}`,
        };
      } catch (error) {
        const summary = error instanceof Error ? error.message : "unknown error";
        api.logger.error(`DeepIntShield blocked ${event.toolName}: ${summary}`);
        return {
          block: true,
          blockReason:
            "DeepIntShield authorization is unavailable; the tool was blocked",
        };
      }
    });
  },
});
