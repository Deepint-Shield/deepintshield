from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from ..errors import DeepintShieldBlockedError, ErrorCode, _dependency_error
from ..types import NON_BLOCKING_DECISIONS, GuardrailResult, ToolInvocation

if TYPE_CHECKING:
    from ..client import DeepintShield


class LangGraphShield:
    """
    Node factories for wrapping a LangGraph with DeepintShield guards.

    >>> shield = DeepintShield.from_env()
    >>> lg = shield.langgraph()
    >>> graph = StateGraph(State)
    >>> graph.add_node("input_guard", lg.input_guard)
    >>> graph.add_node("tool_guard", lg.tool_guard)
    >>> graph.add_node("output_guard", lg.output_guard)
    """

    STATE_BLOCKED = "shield_blocked"
    STATE_REASON = "shield_reason"

    def __init__(self, client: "DeepintShield") -> None:
        self._client = client

    # ── node factories ──────────────────────────────────────────────────────

    def input_guard(self, state: dict) -> dict:
        messages = state.get("messages", [])
        text = getattr(messages[-1], "content", "") if messages else ""
        result = self._client.guard(stage="input", input=text, raise_on_block=False)
        return self._state_from_result(result, block_message="Blocked by DeepintShield before agent execution.")

    def output_guard(self, state: dict) -> dict:
        messages = state.get("messages", [])
        text = getattr(messages[-1], "content", "") if messages else ""
        result = self._client.guard(stage="output", output=text, raise_on_block=False)
        return self._state_from_result(result, block_message="Final response blocked by DeepintShield.")

    def tool_guard(self, state: dict) -> dict:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        for call in getattr(last, "tool_calls", []) or []:
            invocation = ToolInvocation(
                tool_name=call.get("name") or call.get("tool_name", "unknown"),
                tool_input=call.get("args") or call.get("tool_input") or {},
                server_label=call.get("server_label", ""),
                action_class=call.get("action_class", "read"),
            )
            result = self._client.guard(
                stage="mcp" if invocation.server_label else "action",
                actor_type="agent",
                tool_input=(
                    invocation.tool_input
                    if isinstance(invocation.tool_input, str)
                    else json.dumps(invocation.tool_input, default=str, sort_keys=True)
                ),
                server_label=invocation.server_label or None,
                tool_name=invocation.tool_name,
                action_class=invocation.action_class,
                raise_on_block=False,
            )
            if result.blocked:
                return self._state_from_result(result, block_message="Tool call blocked by DeepintShield.")
        return {self.STATE_BLOCKED: False, self.STATE_REASON: ""}

    # ── full graph wrapper ──────────────────────────────────────────────────

    def wrap(
        self,
        graph: Any,
        *,
        agent_node: str = "agent",
        tool_node: str = "tools",
    ) -> Any:
        """
        Wrap a StateGraph with input/tool/output guards.

        Assumes ``agent_node`` and ``tool_node`` already exist. Inserts
        input_guard before agent, tool_guard between agent and tools,
        output_guard after tools.
        """
        try:
            from langgraph.graph import END
        except ImportError as exc:  # pragma: no cover
            raise _dependency_error(
                "Install langgraph: pip install 'deepintshield[langgraph]'",
                code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
                component="langgraph",
            ) from exc

        blocked = self.STATE_BLOCKED

        graph.add_node("input_guard", self.input_guard)
        graph.add_node("tool_guard", self.tool_guard)
        graph.add_node("output_guard", self.output_guard)
        graph.set_entry_point("input_guard")
        graph.add_conditional_edges("input_guard", lambda s: END if s.get(blocked) else agent_node)
        graph.add_edge(agent_node, "tool_guard")
        graph.add_conditional_edges("tool_guard", lambda s: END if s.get(blocked) else tool_node)
        graph.add_edge(tool_node, "output_guard")
        graph.add_conditional_edges("output_guard", lambda _s: END)
        return graph

    # ── internal ────────────────────────────────────────────────────────────

    def _state_from_result(self, result: GuardrailResult, *, block_message: str) -> dict:
        try:
            from langchain_core.messages import AIMessage
        except ImportError:  # pragma: no cover
            AIMessage = None  # type: ignore[assignment]

        if result.blocked:
            block = {self.STATE_BLOCKED: True, self.STATE_REASON: result.reason or result.decision}
            if AIMessage is not None:
                block["messages"] = [AIMessage(content=block_message)]
            return block
        return {self.STATE_BLOCKED: False, self.STATE_REASON: ""}
