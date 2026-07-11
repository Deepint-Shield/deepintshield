"""Agent blueprint manifest - a framework-agnostic description of an agent's
declared tool surface (nodes / tools / edges / MCP servers), extracted from the
compiled graph or agent object so it can be registered with the server BEFORE
the run.

The live framework object can't be shipped to the server, but this small,
structure-only manifest can: it carries names + edges, never arguments, secrets
or data (zero-data-retention preserved). The server stores it as the declared
topology for full-graph visualization, policy pre-validation, and
declared-vs-observed drift (ASI04 supply-chain).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from .integrations._common import callable_tool_name, source_fingerprint

log = logging.getLogger(__name__)


@dataclass
class AgentManifest:
    framework: str
    version: str = ""
    nodes: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    edges: list[list[str]] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    # name → "src:<sha256[:16]>" (persisted, declares the code identity for drift)
    tool_fingerprints: dict[str, str] = field(default_factory=dict)
    # name → raw source, sent ONCE at govern() so the gateway can threat-scan it;
    # the server scans then discards it (never persisted - ZDR preserved).
    tool_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _langgraph_node_fn(node: Any) -> Any:
    """The user function backing a compiled LangGraph node (PregelNode.bound.func
    / .afunc), or None for ToolNodes / custom runnables with no plain function."""
    for holder in (getattr(node, "bound", None), getattr(node, "runnable", None), node):
        if holder is None:
            continue
        for attr in ("func", "afunc"):
            fn = getattr(holder, attr, None)
            if callable(fn):
                return fn
    return None


def _safe_source(fn: Any) -> str:
    try:
        return inspect.getsource(inspect.unwrap(fn))
    except (OSError, TypeError, ValueError):
        return ""


def describe(target: Any) -> AgentManifest:
    """Auto-detect the framework of ``target`` and extract its declared tool
    surface. Mirrors ``AgenticSurface._dispatch`` so ``govern`` and ``guard``
    agree on the framework."""
    # Compiled LangGraph - dict-shaped `.nodes` + `.invoke`.
    if isinstance(getattr(target, "nodes", None), dict) and hasattr(target, "invoke"):
        return _describe_langgraph(target)
    # PydanticAI - internal function-tool registry.
    if any(hasattr(target, a) for a in ("_function_tools", "_function_toolset")):
        return _describe_tools(target, "pydanticai")
    # OpenAI Agents SDK - tools expose async `on_invoke_tool`.
    sample = target[0] if isinstance(target, (list, tuple)) and target else target
    oa_tools = getattr(target, "tools", None)
    if hasattr(sample, "on_invoke_tool") or (
        isinstance(oa_tools, (list, tuple)) and oa_tools and hasattr(oa_tools[0], "on_invoke_tool")
    ):
        return _describe_tools(target, "openai_agents")
    # Everything else that looks like a tool / list of tools (CrewAI, LangChain
    # StructuredTool, LlamaIndex, AutoGen FunctionTool, …).
    return _describe_tools(target, "crewai")


def _describe_langgraph(app: Any) -> AgentManifest:
    reserved = ("__start__", "__end__")
    # Govern the IMPLEMENTATION: the declared tool is each node's FUNCTION name
    # (matching what the interceptor gates), not the node label. Build a
    # key→tool map so the topology edges line up with the governed identities.
    key_to_tool: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    sources: dict[str, str] = {}
    tools: list[str] = []
    for key, node in app.nodes.items():
        if key in reserved:
            continue
        fn = _langgraph_node_fn(node)
        tname = callable_tool_name(fn, key) if fn is not None else key
        key_to_tool[key] = tname
        tools.append(tname)
        if fn is not None:
            fp = source_fingerprint(fn)
            if fp:
                fingerprints[tname] = fp
            src = _safe_source(fn)
            if src:
                sources[tname] = src
    edges: list[list[str]] = []
    try:
        gr = app.get_graph()
        edges = [
            [key_to_tool.get(e.source, e.source), key_to_tool.get(e.target, e.target)]
            for e in gr.edges
            if e.source not in reserved and e.target not in reserved
        ]
    except Exception:  # pragma: no cover - topology is best-effort
        pass
    return AgentManifest(
        framework="langgraph", nodes=tools, tools=tools, edges=edges,
        tool_fingerprints=fingerprints, tool_sources=sources,
    )


def _tool_fn(tool: Any) -> Any:
    """The underlying callable of a tool object (CrewAI/LangChain/LlamaIndex/…)."""
    return getattr(tool, "func", None) or getattr(tool, "_run", None) or (tool if callable(tool) else None)


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    meta = getattr(tool, "metadata", None)
    if meta is not None and getattr(meta, "name", None):
        return meta.name
    fn = _tool_fn(tool) or tool
    return getattr(fn, "__name__", None) or str(tool)


def _describe_tools(target: Any, framework: str) -> AgentManifest:
    tools_obj = getattr(target, "tools", None)
    if tools_obj is None:
        for attr in ("_function_tools", "_function_toolset"):
            reg = getattr(target, attr, None)
            if reg is not None:
                inner = getattr(reg, "tools", reg)
                tools_obj = list(inner.values()) if isinstance(inner, dict) else list(inner)
                break
    if tools_obj is None:
        tools_obj = list(target) if isinstance(target, (list, tuple, set)) else [target]
    names: list[str] = []
    fingerprints: dict[str, str] = {}
    sources: dict[str, str] = {}
    for t in tools_obj:
        try:
            n = _tool_name(t)
        except Exception:  # pragma: no cover
            continue
        names.append(n)
        fn = _tool_fn(t)
        if fn is not None:
            fp = source_fingerprint(fn)
            if fp:
                fingerprints[n] = fp
            src = _safe_source(fn)
            if src:
                sources[n] = src
    return AgentManifest(
        framework=framework, nodes=names, tools=names,
        tool_fingerprints=fingerprints, tool_sources=sources,
    )


__all__ = ["AgentManifest", "describe"]
