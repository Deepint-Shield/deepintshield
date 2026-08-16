"""AutoGen (AG2 / autogen-core) enforcement - gate AutoGen ``FunctionTool``
objects or bare callables registered as tools.

AutoGen's tool surface varies across versions, so this adapter probes the
known shapes:
    * ``autogen_core.tools.FunctionTool`` → wrap its underlying ``_func`` /
      ``func`` callable (``run``/``run_json`` delegate to it).
    * an ``AssistantAgent`` exposing ``.tools`` → wrap each tool.
    * a bare callable registered with ``register_function`` → wrap directly.
Accepts a single target or a list; returns the same object(s).
"""

from __future__ import annotations

import logging
import inspect
from types import SimpleNamespace
from typing import Any

from ..enforcement import report_topology_for
from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ._common import (
    CallbackInventory,
    as_list,
    explicit_callback_inventory,
    install_method_guard,
    merge_callback_inventories,
    set_attr,
    wrap_callable,
)

log = logging.getLogger(__name__)

# A team/agent's first run - by then every participant and their tools exist, so
# the group chat can be reported to the GAF registry as a network.
_TOPOLOGY_BOUNDARIES = (
    ("autogen_agentchat.teams", "BaseGroupChat", ("run", "run_stream")),
    ("autogen_agentchat.agents", "AssistantAgent", ("run", "run_stream")),
    ("autogen.agentchat", "GroupChatManager", ("run_chat", "a_run_chat")),
    ("autogen.agentchat", "ConversableAgent", ("initiate_chat", "a_initiate_chat")),
)

_CALLBACK_FIELDS = (
    # AG2's explicit hook registry. Deliberately exclude _reply_func_list: it
    # is predominantly a framework dispatcher containing library internals.
    "hook_lists",
    "_hook_lists",
    # autogen-agentchat public application selectors / termination callbacks.
    "termination_condition",
    "_termination_condition",
    "selector_func",
    "_selector_func",
    "candidate_func",
    "_candidate_func",
    "handoff_func",
    "_handoff_func",
)


def executable_callbacks(owner: Any, *, prefix: str = "autogen") -> CallbackInventory:
    """Inventory explicit AutoGen hooks without traversing reply internals."""
    framework_modules = ("autogen", "autogen_core", "autogen_agentchat")
    inventories = [
        explicit_callback_inventory(
            owner,
            _CALLBACK_FIELDS,
            framework_modules=framework_modules,
            prefix=prefix,
        )
    ]
    wrapped: list[Any] = []
    incomplete = False
    # FunctionalTermination and Handoff are framework-owned declarative
    # wrappers whose explicit function members contain the application code.
    # Read only those documented members, never the wrapper's other state.
    for field in ("termination_condition", "_termination_condition"):
        try:
            condition = getattr(owner, field)
        except AttributeError:
            continue
        except Exception:
            incomplete = True
            continue
        for callback_field in ("func", "_func"):
            try:
                callback = getattr(condition, callback_field)
            except AttributeError:
                continue
            except Exception:
                incomplete = True
                continue
            if callback is not None:
                wrapped.append(callback)
    for handoffs_field in ("handoffs", "_handoffs"):
        try:
            handoffs = getattr(owner, handoffs_field)
        except AttributeError:
            continue
        except Exception:
            incomplete = True
            continue
        if handoffs is not None and not isinstance(handoffs, (list, tuple)):
            incomplete = True
            continue
        for handoff in handoffs or ():
            try:
                callback = getattr(handoff, "on_handoff")
            except AttributeError:
                continue
            except Exception:
                incomplete = True
                continue
            if callback is not None:
                wrapped.append(callback)
    inventories.append(
        explicit_callback_inventory(
            SimpleNamespace(values=wrapped),
            ("values",),
            framework_modules=framework_modules,
            prefix=f"{prefix}.wrapped_callbacks",
        )
    )
    if incomplete:
        inventories.append(CallbackInventory(incomplete=True))
    return merge_callback_inventories(*inventories)


def enforce() -> bool:
    """Non-bypassable AutoGen enforcement: patch ``FunctionTool.run``/``run_json``
    so every tool is gated at execution. Idempotent and fail-closed.

    Also reports the team's topology to the GAF registry on its first run."""
    report_topology_for(_TOPOLOGY_BOUNDARIES)
    name_fn = lambda self: getattr(self, "name", None) or type(self).__name__
    impl_fn = lambda self: getattr(self, "_func", None) or getattr(self, "func", None) or self
    installed = False
    for mod, cls in (
        ("autogen_core.tools", "FunctionTool"),
        ("autogen_core.tools", "BaseTool"),
        ("autogen.tools", "FunctionTool"),
    ):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            continue
        # Current AutoGen's run_json validates/deserializes and delegates to
        # run. Patching both therefore performs two network decisions per tool.
        # Guard the lowest executable boundary, with run_json only as a
        # compatibility fallback for versions that expose no run method.
        for attr in ("run", "run_json"):
            method = getattr(base, attr, None)
            if install_method_guard(
                base,
                attr,
                name_fn,
                is_async=inspect.iscoroutinefunction(method),
                impl_fn=impl_fn,
            ):
                installed = True
                break
    return installed


@public_agentic_boundary
def shield_tools(target: Any, *, engine: Any) -> Any:
    tools = getattr(target, "tools", None)
    if tools is not None:
        for tool in tools:
            _wrap_tool(tool, engine)
        return target

    single = not isinstance(target, (list, tuple, set))
    wrapped = [_wrap_tool(t, engine) for t in as_list(target)]
    return wrapped[0] if single else wrapped


def _wrap_tool(tool: Any, engine: Any) -> Any:
    # A bare callable registered as a tool.
    if callable(tool) and not hasattr(tool, "run") and not hasattr(tool, "_func"):
        name = getattr(tool, "__name__", "unknown")
        return wrap_callable(engine, name, tool)

    name = getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")
    for attr in ("_func", "func", "run_json", "run"):
        fn = getattr(tool, attr, None)
        if callable(fn):
            if set_attr(tool, attr, wrap_callable(engine, name, fn)):
                return tool
    raise GovernanceConfigurationError(
        framework="autogen",
        reason=f"tool {name!r} exposes no enforceable callable",
    )


__all__ = ["shield_tools", "executable_callbacks"]
