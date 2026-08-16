"""AgenticSurface - the ``shield.agentic`` accessor.

Bundles the PDP engine, the ``tool`` decorator, a direct ``decide`` probe,
and the per-framework enforcement adapters. All of them funnel through the
single ``gate.enforce`` core, so verdict handling is identical everywhere.
"""

from __future__ import annotations

from contextlib import contextmanager
import weakref
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from .decorators import shield_tool
from .engine import AgenticEngine
from .errors import (
    DeepIntShieldError,
    GovernanceConfigurationError,
    public_agentic_boundary,
)
from .identity import PrincipalBinding, obo_actor_chain, resolve_principal
from .obligations import digest
from .types import ContextBag, Decision, DelegationContext, VKCredentialInfo

if TYPE_CHECKING:
    from ..client import DeepintShield


class AgenticSurface:
    """``shield.agentic`` - agentic (PDP) tool gating across frameworks."""

    def __init__(self, parent: "DeepintShield") -> None:
        self._parent_ref = weakref.ref(parent)
        self.engine = AgenticEngine(parent)
        # Non-bypassable enforcement: patch the build/execute boundary of any
        # framework already imported so tools/graphs can't run ungoverned without
        # the developer remembering ``govern()``. (Also installed at client
        # construction; re-run here in case the surface was built first.)
        self.enforce()

    @public_agentic_boundary
    def enforce(self) -> list[str]:
        """Install enforcement guards so every framework's tools/graphs are gated
        without an explicit ``govern()`` - ``compile()``/tool execution always
        passes through the PDP. Auto-called for any framework already imported;
        late imports are watched and armed automatically. Execution guards are
        fail-closed; optional discovery stays fail-open. Calling this method
        explicitly is idempotent and returns the newly guarded frameworks. The
        gateway remains the hard boundary."""
        from .enforcement import install_all

        parent = self._parent_ref()
        return install_all(client=parent) if parent is not None else []

    # ── discovery ────────────────────────────────────────────────────────

    @property
    @public_agentic_boundary
    def credential_info(self) -> VKCredentialInfo:
        """What the gateway knows about the selected Registry identity profile.
        Handy for ops diagnostics ("which Entra profile did this client select?")."""
        return self.engine.credential_info

    # ── identity (GAF directory) ──────────────────────────────────────────

    @public_agentic_boundary
    def identity(
        self,
        *,
        email: Optional[str] = None,
        username: Optional[str] = None,
        subject: Optional[str] = None,
        display_name: Optional[str] = None,
        kind: str = "user",
    ) -> str:
        """Resolve a human identifier to its canonical GAF subject
        (``shield.agentic.identity(email="alice@corp.com")`` →
        ``"user:<collision-resistant-id>"``).

        The gateway creates the directory principal on first sight, so this is
        also how a user appears under Organization → Users without a manual
        import. Cached per process and fail-soft: an unreachable gateway still
        returns the same locally-derived subject."""
        return resolve_principal(
            self.engine,
            email=email,
            username=username,
            subject=subject,
            display_name=display_name,
            kind=kind,
        )

    @public_agentic_boundary
    def as_user(
        self,
        email: Optional[str] = None,
        *,
        username: Optional[str] = None,
        subject: Optional[str] = None,
        display_name: Optional[str] = None,
        kind: str = "user",
    ) -> str:
        """Bind this client to the human the run is acting for and return their
        subject.

        One call at the top of a request handler
        (``shield.agentic.as_user("alice@corp.com")``) and every subsequent
        ``decide()`` / gated tool call carries the on-behalf-of leg
        (``actor_chain = ["user:<derived-id>", "<selected-registry-agent>"]``)
        automatically, so the authorization intersection evaluates the user's
        permission as well as the agent's inside the single canonical GAF
        decision. Pass no identifier to clear the
        binding."""
        if not (email or username or subject):
            self.engine.bind_principal(None)
            return ""
        resolved = self.identity(
            email=email, username=username, subject=subject,
            display_name=display_name, kind=kind,
        )
        self.engine.bind_principal(
            PrincipalBinding(
                subject=resolved,
                email=str(email or "").strip().lower(),
                username=str(username or "").strip(),
                display_name=str(display_name or "").strip(),
                kind=kind if kind in ("user", "service_account") else "user",
            )
        )
        return resolved

    def start_run(self, session_id: str = "") -> str:
        """Bind a new run id in the current request/task context.

        A shared client can therefore serve concurrent agent runs without their
        decisions being grouped into the same execution.
        """
        self.engine.session_id = session_id or self.engine.new_session_id()
        return self.engine.session_id

    def end_run(self) -> None:
        """Clear the request-local run override and restore the client default."""
        self.engine.session_id = ""

    @contextmanager
    def run(
        self,
        *,
        email: Optional[str] = None,
        username: Optional[str] = None,
        subject: Optional[str] = None,
        display_name: Optional[str] = None,
        kind: str = "user",
        session_id: str = "",
    ) -> Iterator[str]:
        """Scope identity and execution grouping to one agent run.

        ``with shield.agentic.run(email=request.user.email):`` is safe with a
        singleton client in async web applications: nested scopes restore the
        previous user/session and concurrent tasks cannot overwrite each other.
        The yielded value is the run/session id.
        """
        from .enforcement import engine_scope

        session_token = self.engine.bind_session(session_id)
        principal_token = None
        with engine_scope(self.engine):
            try:
                if email or username or subject:
                    resolved = self.identity(
                        email=email,
                        username=username,
                        subject=subject,
                        display_name=display_name,
                        kind=kind,
                    )
                    principal_token = self.engine.bind_principal(
                        PrincipalBinding(
                            subject=resolved,
                            email=str(email or "").strip().lower(),
                            username=str(username or "").strip(),
                            display_name=str(display_name or "").strip(),
                            kind=kind if kind in ("user", "service_account") else "user",
                        )
                    )
                yield self.engine.session_id
            finally:
                if principal_token is not None:
                    self.engine.reset_principal(principal_token)
                self.engine.reset_session(session_token)

    # ── registry discovery ────────────────────────────────────────────────

    def discover(
        self,
        target: Any = None,
        *,
        manifest: Optional[dict] = None,
        principal_email: Optional[str] = None,
        auto_provision: bool = True,
        sync: bool = False,
        name: str = "",
    ) -> dict:
        """Report an agent network's topology to the GAF registry.

        Called automatically after any graph compiles / ``govern()`` runs, so
        Registry → Agents | Tools | Networks fills itself in. Call it explicitly
        to attach the acting human (``principal_email=…``), to name the network,
        or to block on the result (``sync=True``) in a test. Fire-and-forget and
        never raises."""
        from .registry import discover as _discover

        return _discover(
            self.engine,
            target,
            manifest=manifest,
            principal_email=principal_email,
            auto_provision=auto_provision,
            sync=sync,
            name=name,
        )

    # ── direct decide ──────────────────────────────────────────────────────

    @public_agentic_boundary
    def decide(
        self,
        dc: Optional[DelegationContext] = None,
        *,
        tool: Optional[str] = None,
        args: Any = None,
        recovery_cost: str = "",
        rag_provenance: str = "",
        prompt: str = "",
        agent: str = "",
        user: str = "",
        permission: str = "",
        object: str = "",
        delegation_id: str = "",
        action: str = "",
        action_class: str = "",
    ) -> Decision:
        """Call the PDP and return the raw :class:`Decision` (does not raise on
        DENY - use :meth:`tool` or a framework adapter for that).

        Either pass a fully-formed ``DelegationContext`` or the convenience
        ``tool=…, args=…`` form. Pass ``prompt=…`` to have the agent's current
        instruction scanned by the prompt guardrail (injection / PII) at the PDP
        boundary; the text is scan-only and never stored (zero-data-retention).
        """
        if dc is None:
            if tool is None:
                raise GovernanceConfigurationError(
                    framework="agentic",
                    reason="decision context or tool is required",
                    code="agent_decision_context_missing",
                ) from None
            dc = DelegationContext(
                tool=tool,
                args_digest=digest((), {"args": args}),
                virtual_key=self.engine.virtual_key,
                prompt=prompt,
                # The shorthand caller does not claim an arbitrary agent. Use
                # only the authenticated Registry association; on older
                # gateways leave it empty so the data plane resolves it.
                principal=self.engine.agent_subject,
                actor_chain=obo_actor_chain(self.engine),
                identity_type="application",
                context=ContextBag(recovery_cost=recovery_cost, rag_provenance=rag_provenance),
                agent=agent,
                user=user,
                permission=permission,
                object=object,
                delegation_id=delegation_id,
                action=action,
                action_class=action_class,
            )
        return self.engine.decide(dc)

    # ── decorator ────────────────────────────────────────────────────────

    def tool(
        self,
        tool: str,
        *,
        recovery_cost: str = "",
        rag_provenance: str = "",
        agent: str = "",
        permission: str = "",
        object: str = "",
        delegation_id: str = "",
        action: str = "",
        action_class: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator binding ``@shield.agentic.tool("db.write")`` to this
        client's engine."""
        return shield_tool(
            tool=tool,
            client=self.engine,
            recovery_cost=recovery_cost,
            rag_provenance=rag_provenance,
            agent=agent,
            permission=permission,
            object=object,
            delegation_id=delegation_id,
            action=action,
            action_class=action_class,
        )

    # ── one-line front door ───────────────────────────────────────────────

    @public_agentic_boundary
    def guard(self, target: Any = None) -> Any:
        """Optional compatibility entry point for agentic tool enforcement.

        * ``shield.agentic.guard()`` - no argument - returns a native
          LangChain/LangGraph callback handler. Attach it once via
          ``config={"callbacks": [shield.agentic.guard()]}`` and *every* tool the
          agent calls is gated by the PDP. No per-tool code, no parameters: the
          framework supplies the tool name and the gateway resolves the tier,
          policy and identity server-side. Supported frameworks already install
          this enforcement at their native build/run boundary, so new
          applications do not need to add this callback.
        * ``shield.agentic.guard(target)`` - instrument a framework object in
          place (a compiled LangGraph, a CrewAI tool / list of tools, an OpenAI
          Agents ``Agent`` or a PydanticAI ``Agent``) and return it. Equivalent
          to calling the matching adapter method, but auto-detected so callers
          don't have to name their framework.
        """
        if target is None:
            return self.callback()
        return self.govern(target)

    @public_agentic_boundary
    def govern(self, target: Any) -> Any:
        """The full server-driven entry point: **register + instrument** a
        framework agent/graph in one call.

        1. **Describe** - auto-discover the agent's declared tool surface
           (nodes / tools / edges) from the compiled object, framework-agnostic.
        2. **Register** - make a best-effort early POST of that blueprint so the
           server can prepare topology, policy validation, and drift evidence.
           The installed execution guard still requires a durable current
           blueprint acknowledgement before application code runs.
        3. **Instrument** - gate every tool/node through the PDP (same as
           :meth:`guard`).

        This method remains for older applications that already call it. New
        applications keep their tools/graph in plain third-party shape:
        supported framework ``compile()``/``run()``/``kickoff()`` boundaries
        perform discovery and enforcement automatically. For MCP tools routed
        through the gateway no client instrumentation is needed.
        """
        try:
            from .manifest import describe

            self.engine.register_blueprint(describe(target))
        except DeepIntShieldError:
            raise
        except Exception:  # unsupported legacy description remains optional
            pass
        try:
            # Same surface, richer shape: the GAF registry wants the typed
            # node/edge network so Registry + the authorization tuples populate
            # themselves. This early report is background + de-duplicated; the
            # execution guard retries synchronously if it was not acknowledged.
            self.discover(target)
        except Exception:  # required execution barrier handles this later
            pass
        return self._dispatch(target)

    @public_agentic_boundary
    def callback(self) -> Any:
        """Native LangChain ``BaseCallbackHandler`` bound to this client.

        Identical to ``guard()`` with no argument; named for readers who think
        in LangChain terms ("give me a callback handler")."""
        from .integrations.langchain import make_handler

        return make_handler(self.engine)

    def _dispatch(self, target: Any) -> Any:
        """Auto-route a framework object to its in-place adapter."""
        # Compiled LangGraph - dict-shaped `.nodes` is the reliable marker.
        if isinstance(getattr(target, "nodes", None), dict) and hasattr(target, "invoke"):
            return self.langgraph(target)
        # PydanticAI agent - keeps tools in an internal `_function_tools(et)` registry.
        if any(hasattr(target, a) for a in ("_function_tools", "_function_toolset")):
            return self.pydanticai(target)
        # OpenAI Agents SDK - tools expose an async `on_invoke_tool` callable.
        sample = target[0] if isinstance(target, (list, tuple)) and target else target
        oa_tools = getattr(target, "tools", None)
        if hasattr(sample, "on_invoke_tool") or (
            isinstance(oa_tools, (list, tuple)) and oa_tools and hasattr(oa_tools[0], "on_invoke_tool")
        ):
            return self.openai_agents(target)
        # Everything else that looks like a tool / list of tools (CrewAI
        # BaseTool, a LangChain StructuredTool, …) - gate the tool callable.
        return self.crewai(target)

    # ── framework enforcement adapters (L2) ───────────────────────────────

    @public_agentic_boundary
    def langgraph(self, graph: Any) -> Any:
        from .integrations.langgraph import shield_graph

        return shield_graph(graph, engine=self.engine)

    @public_agentic_boundary
    def crewai(self, tools: Any) -> Any:
        from .integrations.crewai import shield_tools

        return shield_tools(tools, engine=self.engine)

    @public_agentic_boundary
    def openai_agents(self, target: Any) -> Any:
        from .integrations.openai_agents import shield_agent

        return shield_agent(target, engine=self.engine)

    @public_agentic_boundary
    def llamaindex(self, tools: Any) -> Any:
        from .integrations.llamaindex import shield_tools

        return shield_tools(tools, engine=self.engine)

    @public_agentic_boundary
    def autogen(self, target: Any) -> Any:
        from .integrations.autogen import shield_tools

        return shield_tools(target, engine=self.engine)

    @public_agentic_boundary
    def pydanticai(self, agent: Any) -> Any:
        from .integrations.pydanticai import shield_agent

        return shield_agent(agent, engine=self.engine)

    @public_agentic_boundary
    def temporal(self) -> Any:
        """Return a Temporal ``Interceptor`` that gates every activity through
        the PDP. Attach it once: ``Worker(..., interceptors=[shield.agentic.temporal()])``."""
        from .integrations.temporal import interceptor

        return interceptor(self.engine)

    @public_agentic_boundary
    def strands(self) -> Any:
        """Return an AWS Strands ``HookProvider`` that gates every tool
        invocation through the PDP. ``Agent(..., hooks=[shield.agentic.strands()])``."""
        from .integrations.strands import hook_provider

        return hook_provider(self.engine)

    @public_agentic_boundary
    def google_adk(self) -> Any:
        """Return a Google ADK ``BasePlugin`` that gates every tool call through
        the PDP app-wide. ``InMemoryRunner(..., plugins=[shield.agentic.google_adk()])``."""
        from .integrations.google_adk import plugin

        return plugin(self.engine)

    @public_agentic_boundary
    def hermes(self, ctx: Any) -> bool:
        """Install the PDP hooks on a Hermes plugin context. Call from your
        Hermes plugin's ``register(ctx)``: ``shield.agentic.hermes(ctx)``."""
        from .integrations.hermes import install

        return install(ctx, self.engine)

    @public_agentic_boundary
    def openclaw_config(self, *, gateway_url: str = "", models: Any = None) -> dict:
        """Return the OpenClaw ``models.providers`` config block that routes all
        model traffic through the gateway (Layer-1, zero-code governance:
        semantic cache, coalescing, guardrails, budgets, cost analytics). Merge
        it into ``openclaw.json`` and set ``agents.defaults.model.primary`` to a
        ``deepintshield/<model>`` id. In-process tool governance (Layer 2) needs
        the TypeScript plugin - Python can't be embedded in OpenClaw's Node
        gateway; see docs/examples for the plugin stub."""
        from .integrations.openclaw import provider_config

        return provider_config(self.engine, gateway_url=gateway_url, models=models)


__all__ = ["AgenticSurface"]
