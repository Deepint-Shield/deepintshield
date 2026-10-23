# DeepIntShield Python SDK changelog

## 3.2.2 - October 23, 2026

- A LangGraph node built from Runnables (a `RunnableParallel`, a chain such
  as `itemgetter("messages") | model`, a subgraph) no longer quarantines its
  agent at the first execution after its blueprint was approved. 3.2.1 reported
  such a node's code as the composition it dispatches, but the node's
  execution gate still attested the framework method it wraps (the bound
  runnable's `invoke`), which never matched the acknowledged artifact; the
  first execution therefore filed a second, "first-run" report built from
  langgraph's own source, an incomplete artifact the gateway refuses as
  `blueprint_coverage_incomplete` and quarantines the reporting agent for.
  The gate now attests the node by the same composition text and digest the
  discovery report carried (the decision fingerprint too), so an approved
  blueprint is recognised and the run proceeds to the authorization decision.
  A composition that cannot be read keeps failing closed exactly as before.

## 3.2.1 - October 16, 2026

This release aligns the Python SDK with DeepIntShield Server 3.2.1. The SDK's
own behaviour is unchanged apart from the item below; the server release makes
a key that carries MCP bindings usable for plain inference again (see the
3.2.1 server notes).

- A program whose server run ended because its identity cannot use one runs locally with the fingerprint the administrator approved: when the gateway's capabilities answer carries `report_program_bundle`, local discovery reports the program bundle the submission reported (no second review); without that answer nothing changes.
- The managed worker's tests that start a real worker process run on a root
  checkout whose interpreter the program user (uid 65532) can reach; nothing in
  the worker changed.
- `shield.mcp.connect(identity=True)` names the reason when the gateway
  refuses to issue workload credentials (for example `agent_not_registered`).
  The refusal happens inside the MCP transport's request hook, which the
  transport answered with a bare `CancelledError: Cancelled via cancel scope`;
  the session now raises the recorded refusal as its connection error (and as
  the operation's error when a credential renewal fails mid-session). A real
  cancellation without such a refusal still propagates unchanged.
- A LangGraph node built from Runnables - a `RunnableParallel`, a chain such
  as `itemgetter("messages") | model`, a `RunnableBranch`, a `ToolNode` or a
  compiled subgraph - is now covered in the blueprint. The SDK already gated
  such nodes at run time, but it described them as a local tool with no
  source, so the gateway refused every graph of that shape as
  `blueprint_coverage_incomplete` and quarantined the reporting agent. The
  node's source is now the composition it dispatches (class names and keys
  only, never arguments, keys or configuration) together with the source of
  every Python callable and application class inside it, which is what the
  gateway scans; its fingerprint is that text's digest. The discovery report
  the gateway's coverage ledger reads carries the same text: the node becomes
  a governed tool keyed by its node name (the identity the enforcement adapter
  gates it under) with a complete code artifact, instead of the incomplete
  artifact that quarantined the reporting agent. A callable inside the
  composition whose source cannot be read still leaves the node uncovered,
  exactly as a plain function without source does.
- A program that holds no application code of its own - an agent made of
  instructions, a model and an output type, with no tools, hooks or
  callbacks - reports its definition as the code to scan. The gateway treats
  the blueprint envelope as an inventory requirement: a discovery report with
  no code artifact at all is refused as `blueprint_coverage_incomplete` and
  quarantines the reporting agent, so every tool-less program (the OpenAI
  Agents and PydanticAI programs of the QA catalogue among them) was refused
  on every gateway. The definition artifact names the framework, the agent,
  its model, its instructions (bounded, redacted like any source), its output
  type and its handoffs; a program whose only tools are remote MCP tools is
  unchanged, since the gateway verifies those through its own inventory.
- An `autogen_agentchat` (or AG2) agent run on its own, without a team, is
  discovered as its own one-participant AutoGen program. Left to the generic
  path it had no tool surface and its `run` method is framework code (wrapped
  by the runtime guard), so the discovery report carried no artifact, read as
  `blueprint_coverage_incomplete`, quarantined the reporting agent, and its
  digest collided with any other artifact-less report of the same shape. The
  agent is now reported with its model (read from the client's create
  arguments too) and, with no tools, its definition - the system message
  ahead of the framework's default description - as the code to scan.

## 3.2.0 - October 11, 2026

This release aligns the Python SDK with DeepIntShield Server 3.2.0 and adds the
administrator-review registration route: a developer runs the agent, an
administrator reviews it, and the unchanged script runs.

### Agent registration

- New code `managed_gateway_runtime_required`: in a workspace listed for server
  runs (no local fall-back), an agent whose identity cannot use gateway runtime
  exchange is told so instead of the generic `managed_execution_failed`. Where
  server runs reach every workspace, such a program runs on the developer's
  machine instead.
- `DeepintShield(registered_by=...)`, `ShieldConfig.registered_by` and
  `DEEPINTSHIELD_REGISTERED_BY` name the developer who files an agent
  registration. The gateway maps the address to an identity in the workspace
  directory and notifies it. The address is not verified and grants nothing:
  the administrator confirms the accountable owner during review. One
  registration keeps its first address. An invalid address raises
  `configuration_error` before any request is made.
- When `registered_by` is unset, an e-mail-shaped `requester` is sent in its
  place as a fallback hint only. `requester` names whoever the agent acts
  for, which is often an end user, so the gateway ignores the hint for people
  the organization has not placed in the workspace and files the registration
  as if no address had been sent. It does not refuse a registration because of
  the hint. `registered_by=""` sends no address at all.
- The address travels only in registry discovery payloads (first-run, attested
  and server-run bundle manifests), as the top-level `registered_by_email`
  field with `registered_by_source` beside it: `explicit` for the setting,
  `requester` for the hint. A client with no registering address sends neither
  field, the source is never sent without an address, neither field is a
  header, and `requester` keeps its acting-for meaning.
- A pending registration stops the first governed call with one message.
  Clauses whose value is unknown are omitted:

  ```text
  Registration pending. Reference <registration id> · runtime key SHA256:<thumbprint>. Contact your administrator. You will be notified at <email>.
  ```

  The last sentence appears only when the gateway confirms the address it will
  notify, and it names that address. When the gateway has recorded no
  registering e-mail it has no developer address to notify, and the sentence is
  omitted: that is the case for a gateway older than 3.2.0 and for a
  `requester` hint the gateway ignored. The configured value alone never
  produces the sentence.

  The SDK logs this line once per process for each code. It never prints a
  review link or gateway prose, and it never waits for approval.
- `agent_registration_pending` and `agent_runtime_enrollment_pending` now
  describe themselves as `Registration pending. Contact your administrator.`
  The instruction to approve a runtime-key fingerprint is removed; the key's
  thumbprint appears in the message above. Branch on the code, not on the
  description text.
- New codes: `registration_requester_mismatch`, when a pending registration
  is already recorded under a different `registered_by`, and
  `runtime_binding_identity_missing`, when a runtime key is bound to an agent
  whose identity has not been declared. Both ask the caller to contact an
  administrator. The first reads
  `Registration refused: this agent is already being registered under a different email. Contact your administrator.`
  A `requester` hint never raises `registration_requester_mismatch`.
- Activation takes two runs when the gateway supports deferred approval: the
  administrator records the decision after the first run, and the next run of
  the unchanged script executes. Gateways without deferred approval keep the
  three-run sequence unchanged.
- The two-run route needs the runtime extra, `pip install 'deepintshield[runtime]'`; without it the SDK stays on the earlier flow and reports the plain pending line, `Registration pending. Reference <registration id>. Contact your administrator.`
- Credential discovery for a pending registration may now return key-only
  gateway runtime settings (`credential_bootstrap`, `provisional`,
  `registration_state`, `registration_id`), so the first run already enrols
  its runtime key.
- A registration the gateway already holds as pending is sighted again with
  the virtual key alone when its runtime key is not yet authorized
  (`agent_runtime_enrollment_pending`) or the workload credential cannot be
  obtained (`credential_exchange_failed`, `credential_host_identity_missing`).
  The same failures still stop an agent that is not known to be pending.
- An agent approved as VK-trusted (no identity provider) runs on its virtual
  key, also when an identity is still declared for it under a provider with
  gateway exchange. The gateway marks that credential-info answer
  `credential_optional`. A failure to build that credential or to obtain its
  token then sends no token instead of stopping the call, is not tried again
  for 60 seconds, and is reported once per process:

  ```text
  DeepintShield: this agent runs on its virtual key; its declared identity is not active yet (runtime key SHA256:<thumbprint>).
  ```

  The thumbprint appears when a runtime key was enrolled. Only the flag has
  this effect: every other answer, including an agent bound to its provider,
  still stops the call when its credential fails, and a gateway that does not
  send the flag keeps the earlier behaviour.
- A provisional runtime key that the gateway refuses at enrolment or challenge
  with `agent_runtime_authentication_failed` (what a VK-trusted approval or a
  denial leaves behind) is set aside, and credential-info is read once more in
  the same call. The new answer replaces the hour-long cached one, so a process
  that keeps running continues with the administrator's decision without a
  restart. If the gateway repeats its provisional offer, the call reports the
  plain pending line without a runtime-key clause. Every other enrolment or
  session failure, such as an outage, a 429, a conflict or a key still waiting
  for authorization, is reported unchanged.
- `shield.agentic.status()` no longer returns `review_url` or a console path.
  A pending state reports the message above as `reason` and
  `Contact your administrator.` as `action`, next to `registration_id`.

### Workload credentials

- Entra direct mode selects its host credential from what the process exposes:
  environment service-principal variables, a workload-identity federated token
  file, or a managed identity endpoint. A host with none of them fails at once
  with the new code `credential_host_identity_missing`. A failing chain reports
  `credential_exchange_failed` and is not tried again for 60 seconds.
  Developer sign-ins such as the Azure CLI are no longer consulted; use gateway
  runtime authentication on a laptop.
- `DEEPINTSHIELD_ENTRA_SKIP_IMDS_PROBE=1` skips the one-second managed identity
  endpoint probe, which otherwise runs once per process when no Azure identity
  variable is set.
- Best-effort credential warnings are logged once per process for each code.

### Managed execution

- Managed execution workers run submissions on their own installed SDK, Python
  and package releases. The launcher now requires only that every package the
  program imports is installed; the client's SDK, Python and package versions
  are recorded as review evidence and no longer block execution.
- `DEEPINTSHIELD_REGISTERED_BY` is an accepted worker setting, and a submitted
  bundle manifest carries `registered_by_email` with `registered_by_source`.
  `registered_by` is not a delegation context, so an autonomous script stays
  eligible for a server run.
- Where the gateway offers server runs with a local fall-back, a run it ended
  before the program started because the agent's identity cannot use a server
  run (`managed_gateway_runtime_required`, for example an agent approved as
  VK-trusted) returns the program to the developer's machine. It runs there,
  stays local for the rest of the process, and nothing is submitted again; the
  gateway no longer offers that program a server run on its next start.
  Without the fall-back, as in a workspace listed by name, the run is reported
  as `managed_execution_failed`, as before.

### Upgrading

```bash
python -m pip install --upgrade 'deepintshield[runtime,langgraph]==3.2.0'
python -m pip check
```

Set `DEEPINTSHIELD_REGISTERED_BY`, or pass `registered_by=`, to the work e-mail
of the developer who registers agents, on laptops and CI workers alike. That
gives the registration a proposed owner, whom the administrator confirms, and
an address to notify. The address is not verified and grants nothing.

An application that sets `requester` to an end user's e-mail keeps registering
agents without a change: the address is sent as a fallback hint, and the
gateway ignores the hint for people the organization has not placed in the
workspace without refusing the registration. A hint the gateway does accept is
recorded as the registrant, so pass `registered_by` explicitly to name the
right developer, or `registered_by=""` to send no hint.

Gateways older than 3.2.0 ignore both discovery fields, keep their existing
registration sequence and send no registration e-mail. Against them the pending
message ends at `Contact your administrator.`

## 3.1.0 - October 8, 2026

This release aligns the Python SDK with DeepIntShield Server 3.1.0 and the
current Agentic interface.

### Agentic runtime and execution

- Gateway-managed Entra authentication uses an approved durable P-256 runtime
  key, short-lived broker sessions, and signed per-request proofs. The `runtime`
  extra supplies secure key storage; `runtime-aws` adds AWS identity support.
- Optional platform assertions support configured GCP, Azure, AWS and Kubernetes
  runtimes. Provider management credentials remain separate from workload
  credentials and runtime keys.
- Standalone autonomous LangGraph scripts can hand execution to a configured
  gateway before local graph execution. The SDK submits a bounded source bundle,
  pinned dependencies, Python requirements, and matching blueprint evidence.
- Managed execution exposes stable pending, compatibility, failure,
  cancellation, expiry and reconciliation outcomes. Completed jobs expose
  bounded, sanitized saved output; unresolved work does not execute locally.
- Authenticated Agentic Security configuration is evaluated for each native
  execution. An explicit disabled setting preserves native execution; malformed
  configuration and transport failures retain fail-closed behavior.
- Registration, blueprint scans, workload proof, human ownership and selected
  access grants use the current Agentic review workflow. Approval does not
  replace runtime readiness or per-action authorization.
- Provisioned Generic OIDC clients honor explicit `OIDC_SCOPES`, including an
  empty override. Native MCP sessions preserve request-scoped identity proofs
  and coded tool authorization errors.

### Upgrading

```bash
python -m pip install --upgrade 'deepintshield[runtime,langgraph]==3.1.0'
python -m pip check
```

Choose only the extras your application needs. CPython 3.10 through 3.14 remain
supported; SDK 3.1.0 is a package version, not a Python version requirement.
The `all` extra selects compatible aggregate dependency versions and does not
promise the newest version of every upstream framework.

Server workers must have every package the submitted program imports installed;
they run their own SDK, Python and package releases, and the client's versions
are recorded as evidence only. Use Server 3.1.0 for the managed runtime and
execution flows; installing the SDK does not provision server credentials or
worker packages.

Applications migrating from 2.x must still remove RAG `acl_tags` and enforce
document authorization before invoking safety screening. Those 3.0 migration
requirements are unchanged. Existing ordinary inference and native framework
entry points remain supported.
