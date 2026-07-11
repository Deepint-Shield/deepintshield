"""AgenticSurface - the ``shield.agentic`` accessor.

Bundles the PDP engine, the ``tool`` decorator, a direct ``decide`` probe,
and the per-framework enforcement adapters. All of them funnel through the
single ``gate.enforce`` core, so verdict handling is identical everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from .decorators import set_default_client, shield_tool
from .engine import AgenticEngine
from .obligations import digest
from .types import ContextBag, Decision, DelegationContext, VKCredentialInfo

if TYPE_CHECKING:
    from ..client import DeepintShield


class AgenticSurface:
    """``shield.agentic`` - agentic (PDP) tool gating across frameworks."""

    def __init__(self, parent: "DeepintShield") -> None:
        self._parent = parent
        self.engine = AgenticEngine(parent)
        # Make the most-recently-built shield the implicit default for bare
        # ``@shield_tool(...)`` declarations. Explicit client=… always wins.
        set_default_client(parent)
        # Non-bypassable enforcement: patch the build/execute boundary of any
        # framework already imported so tools/graphs can't run ungoverned without
        # the developer remembering ``govern()``. (Also installed at client
        # construction; re-run here in case the surface was built first.)
        self.enforce()

    def enforce(self) -> list[str]:
        """Install enforcement guards so every framework's tools/graphs are gated
        without an explicit ``govern()`` - ``compile()``/tool execution always
        passes through the PDP. Auto-called for any framework already imported;
        call it again after importing a framework later
        (e.g. ``import crewai; shield.agentic.enforce()``). Best-effort + fail-open;
        returns the frameworks guarded. The gateway stays the hard boundary."""
        try:
            from .enforcement import install_all

            return install_all(lambda: self.engine)
        except Exception:  # never let enforcement setup break client init
            return []

    # ── discovery ────────────────────────────────────────────────────────

    @property
    def credential_info(self) -> VKCredentialInfo:
        """What the gateway knows about this VK's identity binding. Handy for
        ops diagnostics ("which Entra blueprint is this VK bound to?")."""
        return self.engine.credential_info

    # ── direct decide ──────────────────────────────────────────────────────

    def decide(
        self,
        dc: Optional[DelegationContext] = None,
        *,
        tool: Optional[str] = None,
        args: Any = None,
        recovery_cost: str = "",
        rag_provenance: str = "",
        prompt: str = "",
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
                raise ValueError("decide requires either a DelegationContext or tool=…")
            dc = DelegationContext(
                tool=tool,
                args_digest=digest((), {"args": args}),
                virtual_key=self.engine.virtual_key,
                prompt=prompt,
                # The shorthand caller doesn't spell out an identity, but the
                # PDP's subject matchers are authored against an agent role
                # (any_role: ["agent", ...]). Carry a default agent principal +
                # actor_chain so role-scoped policies match the SDK path the
                # same way they match an explicitly-populated DelegationContext.
                # The server only synthesises this itself for agent-bound VKs;
                # LLM-only VKs (the common SDK case) need the SDK to supply it.
                principal="agent:sdk",
                actor_chain=["agent:sdk"],
                identity_type="application",
                context=ContextBag(recovery_cost=recovery_cost, rag_provenance=rag_provenance),
            )
        return self.engine.decide(dc)

    # ── decorator ────────────────────────────────────────────────────────

    def tool(
        self,
        tool: str,
        *,
        recovery_cost: str = "",
        rag_provenance: str = "",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator binding ``@shield.agentic.tool("db.write")`` to this
        client's engine."""
        return shield_tool(
            tool=tool,
            client=self._parent,
            recovery_cost=recovery_cost,
            rag_provenance=rag_provenance,
        )

    # ── one-line front door ───────────────────────────────────────────────

    def guard(self, target: Any = None) -> Any:
        """The single entry point for agentic tool enforcement.

        * ``shield.agentic.guard()`` - no argument - returns a native
          LangChain/LangGraph callback handler. Attach it once via
          ``config={"callbacks": [shield.agentic.guard()]}`` and *every* tool the
          agent calls is gated by the PDP. No per-tool code, no parameters: the
          framework supplies the tool name and the gateway resolves the tier,
          policy and identity server-side. This is the recommended path for
          anything built on LangChain (chains, agents, LangGraph, prebuilt
          ReAct agents).
        * ``shield.agentic.guard(target)`` - instrument a framework object in
          place (a compiled LangGraph, a CrewAI tool / list of tools, an OpenAI
          Agents ``Agent`` or a PydanticAI ``Agent``) and return it. Equivalent
          to calling the matching adapter method, but auto-detected so callers
          don't have to name their framework.
        """
        if target is None:
            return self.callback()
        return self.govern(target)

    def govern(self, target: Any) -> Any:
        """The full server-driven entry point: **register + instrument** a
        framework agent/graph in one call.

        1. **Describe** - auto-discover the agent's declared tool surface
           (nodes / tools / edges) from the compiled object, framework-agnostic.
        2. **Register** - POST that blueprint to the server BEFORE the run so the
           server holds the declared topology (full-graph viz, policy
           pre-validation, declared-vs-observed drift). Best-effort, non-fatal.
        3. **Instrument** - gate every tool/node through the PDP (same as
           :meth:`guard`).

        The developer keeps their tools/graph in plain third-party shape and adds
        exactly one line: ``app = shield.agentic.govern(app)``. Discovery, policy
        and the decision all live server-side. (For MCP tools routed through the
        gateway no client code is needed at all - they are governed in transit.)
        """
        try:
            from .manifest import describe

            self.engine.register_blueprint(describe(target))
        except Exception:  # describe/register is best-effort - never block govern
            pass
        return self._dispatch(target)

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

    def langgraph(self, graph: Any) -> Any:
        from .integrations.langgraph import shield_graph

        return shield_graph(graph, engine=self.engine)

    def crewai(self, tools: Any) -> Any:
        from .integrations.crewai import shield_tools

        return shield_tools(tools, engine=self.engine)

    def openai_agents(self, target: Any) -> Any:
        from .integrations.openai_agents import shield_agent

        return shield_agent(target, engine=self.engine)

    def llamaindex(self, tools: Any) -> Any:
        from .integrations.llamaindex import shield_tools

        return shield_tools(tools, engine=self.engine)

    def autogen(self, target: Any) -> Any:
        from .integrations.autogen import shield_tools

        return shield_tools(target, engine=self.engine)

    def pydanticai(self, agent: Any) -> Any:
        from .integrations.pydanticai import shield_agent

        return shield_agent(agent, engine=self.engine)

    def temporal(self) -> Any:
        """Return a Temporal ``Interceptor`` that gates every activity through
        the PDP. Attach it once: ``Worker(..., interceptors=[shield.agentic.temporal()])``."""
        from .integrations.temporal import interceptor

        return interceptor(self.engine)

    def strands(self) -> Any:
        """Return an AWS Strands ``HookProvider`` that gates every tool
        invocation through the PDP. ``Agent(..., hooks=[shield.agentic.strands()])``."""
        from .integrations.strands import hook_provider

        return hook_provider(self.engine)

    def google_adk(self) -> Any:
        """Return a Google ADK ``BasePlugin`` that gates every tool call through
        the PDP app-wide. ``InMemoryRunner(..., plugins=[shield.agentic.google_adk()])``."""
        from .integrations.google_adk import plugin

        return plugin(self.engine)

    def hermes(self, ctx: Any) -> bool:
        """Install the PDP hooks on a Hermes plugin context. Call from your
        Hermes plugin's ``register(ctx)``: ``shield.agentic.hermes(ctx)``."""
        from .integrations.hermes import install

        return install(ctx, self.engine)

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
