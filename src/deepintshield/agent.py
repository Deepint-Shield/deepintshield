from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any, Callable, Mapping

from .errors import DeepintShieldBlockedError, ErrorCode, _annotate_error
from .types import GuardrailResult, ToolInvocation

if TYPE_CHECKING:
    from .client import DeepintShield


class AgentSurface:
    """
    Agentic guardrails: input/output scanning plus tool/MCP evaluation.

    >>> shield = DeepintShield.from_env()
    >>>
    >>> @shield.agent.tool
    >>> def read_file(path: str) -> str: ...
    >>>
    >>> shield.agent.check_input("user message")
    >>> shield.agent.check_output("assistant reply")
    >>> shield.agent.evaluate_tool(name="read_file", args={"path": "/tmp"})
    """

    def __init__(self, client: "DeepintShield") -> None:
        self._client = client

    # ── stage helpers ────────────────────────────────────────────────────────

    def check_input(self, text: str, **kwargs: Any) -> GuardrailResult:
        return self._client.guard(stage="input", input=text, **kwargs)

    def check_output(self, text: str, **kwargs: Any) -> GuardrailResult:
        return self._client.guard(stage="output", output=text, **kwargs)

    # ── tool invocation ─────────────────────────────────────────────────────

    def evaluate_tool(
        self,
        invocation: ToolInvocation | Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        args: Any = None,
        server_label: str = "",
        action_class: str = "read",
        domains: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        actor_type: str = "agent",
        raise_on_block: bool = True,
        **kwargs: Any,
    ) -> GuardrailResult:
        if invocation is None:
            if name is None:
                raise _annotate_error(
                    ValueError("evaluate_tool requires either invocation or name=..."),
                    ErrorCode.AGENT_INVOCATION_INVALID,
                )
            invocation = ToolInvocation(
                tool_name=name,
                tool_input=args if args is not None else {},
                server_label=server_label,
                action_class=action_class,
                domains=list(domains or []),
                metadata=dict(metadata or {}),
            )
        invocation_failed = False
        try:
            tool = (
                invocation
                if isinstance(invocation, ToolInvocation)
                else ToolInvocation(**dict(invocation))
            )
            tool_input = (
                tool.tool_input
                if isinstance(tool.tool_input, str)
                else json.dumps(tool.tool_input, default=str, sort_keys=True)
            )
        except Exception:
            # Invalid mappings, circular structures, and failing custom string
            # conversions are all invocation-validation failures. Translate
            # only after leaving the handler so private object state is not
            # retained as chained exception context.
            tool = None
            tool_input = ""
            invocation_failed = True
        if invocation_failed or tool is None:
            raise _annotate_error(
                ValueError("evaluate_tool received an invalid tool invocation"),
                ErrorCode.AGENT_INVOCATION_INVALID,
            ) from None
        return self._client.guard(
            stage="mcp" if tool.server_label else "action",
            actor_type=actor_type,
            tool_input=tool_input,
            server_label=tool.server_label or None,
            tool_name=tool.tool_name,
            action_class=tool.action_class,
            domains=tool.domains,
            metadata=tool.metadata,
            raise_on_block=raise_on_block,
            **kwargs,
        )

    # ── decorator ───────────────────────────────────────────────────────────

    def tool(
        self,
        func: Callable | None = None,
        *,
        action_class: str = "read",
        server_label: str = "",
        name: str | None = None,
    ) -> Callable:
        """
        Decorator that guards a function call through DeepintShield before execution.

        >>> @shield.agent.tool(action_class="write")
        ... def write_file(path: str, content: str) -> None: ...
        """

        def decorate(fn: Callable) -> Callable:
            tool_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.evaluate_tool(
                    name=tool_name,
                    args={"args": list(args), "kwargs": dict(kwargs)},
                    action_class=action_class,
                    server_label=server_label,
                )
                return fn(*args, **kwargs)

            return wrapper

        if func is not None and callable(func):
            return decorate(func)
        return decorate

    # ── full loop helper ────────────────────────────────────────────────────

    def guard_turn(
        self,
        *,
        user_input: str,
        model_output: str | None = None,
        tool_calls: list[Mapping[str, Any]] | None = None,
        raise_on_block: bool = True,
    ) -> dict[str, GuardrailResult]:
        """
        Evaluate a single agent turn: input, tool calls (if any), output (if any).
        Returns a dict of stage -> result.
        """
        results: dict[str, GuardrailResult] = {}
        results["input"] = self.check_input(user_input, **{"raise_on_block": raise_on_block})  # type: ignore[arg-type]
        for call in tool_calls or []:
            key = f"tool:{call.get('name') or call.get('tool_name')}"
            results[key] = self.evaluate_tool(
                name=call.get("name") or call.get("tool_name", "unknown"),
                args=call.get("args") or call.get("tool_input"),
                action_class=call.get("action_class", "read"),
                server_label=call.get("server_label", ""),
                raise_on_block=raise_on_block,
            )
        if model_output is not None:
            results["output"] = self.check_output(model_output, raise_on_block=raise_on_block)
        return results
