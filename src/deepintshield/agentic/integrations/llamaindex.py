"""LlamaIndex enforcement - gate each ``FunctionTool`` / ``BaseTool`` by
wrapping its ``call`` (and async ``acall``) method. Accepts a single tool or
a list; mutates in place and returns the same object(s).
"""

from __future__ import annotations

import logging
from typing import Any

from ..enforcement import report_topology_for
from ..errors import GovernanceConfigurationError, public_agentic_boundary
from ._common import (
    CallbackInventory,
    as_list,
    callback_method_inventory,
    explicit_callback_inventory,
    install_method_guard,
    merge_callback_inventories,
    set_attr,
    wrap_callable,
)

log = logging.getLogger(__name__)

# An agent's first chat/run - its tool retriever and worker are populated by
# then, which is what the registry needs to record the agent + its tools.
_TOPOLOGY_BOUNDARIES = (
    ("llama_index.core.agent", "AgentRunner", ("chat", "achat", "run", "arun")),
    ("llama_index.core.agent.workflow", "AgentWorkflow", ("run",)),
    ("llama_index.core.agent.workflow", "FunctionAgent", ("run",)),
)

_CALLBACK_FIELDS = ("callback", "callbacks", "step_callback")
_HANDLER_METHODS = (
    "on_event_start",
    "on_event_end",
    "start_trace",
    "end_trace",
)


def executable_callbacks(owner: Any, *, prefix: str = "llamaindex") -> CallbackInventory:
    """Inventory application callbacks and explicit callback-manager handlers.

    Only the manager's public handler list and callback protocol methods are
    considered.  The adapter never walks LlamaIndex workflow/event internals;
    inherited built-in handler methods are filtered by module ownership.
    """
    inventories = [
        explicit_callback_inventory(
            owner,
            _CALLBACK_FIELDS,
            framework_modules=("llama_index",),
            prefix=prefix,
        )
    ]
    for manager_field in ("callback_manager", "_callback_manager"):
        try:
            manager = getattr(owner, manager_field)
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        try:
            handlers = getattr(manager, "handlers")
        except AttributeError:
            continue
        except Exception:
            inventories.append(CallbackInventory(incomplete=True))
            continue
        inventories.append(
            callback_method_inventory(
                handlers,
                _HANDLER_METHODS,
                framework_modules=("llama_index",),
                prefix=f"{prefix}.{manager_field}.handlers",
            )
        )
    return merge_callback_inventories(*inventories)


def enforce() -> bool:
    """Non-bypassable LlamaIndex enforcement: patch the tool base ``call``/``acall``
    so every tool is gated at execution. Idempotent and fail-closed.

    Also reports the agent's topology to the GAF registry on its first run."""
    report_topology_for(_TOPOLOGY_BOUNDARIES)
    name_fn = lambda self: (
        getattr(getattr(self, "metadata", None), "name", None)
        or getattr(self, "name", None)
        or type(self).__name__
    )
    impl_fn = lambda self: getattr(self, "fn", None) or self
    installed = False
    for mod, cls in (
        ("llama_index.core.tools", "FunctionTool"),
        ("llama_index.core.tools.function_tool", "FunctionTool"),
        ("llama_index.core.tools", "BaseTool"),
    ):
        try:
            base = getattr(__import__(mod, fromlist=[cls]), cls)
        except Exception:
            continue
        if install_method_guard(base, "call", name_fn, impl_fn=impl_fn):
            installed = True
        install_method_guard(base, "acall", name_fn, is_async=True, impl_fn=impl_fn)
    return installed


@public_agentic_boundary
def shield_tools(tools: Any, *, engine: Any) -> Any:
    single = not isinstance(tools, (list, tuple, set))
    wrapped = [_wrap_tool(t, engine) for t in as_list(tools)]
    return wrapped[0] if single else wrapped


def _tool_name(tool: Any) -> str:
    meta = getattr(tool, "metadata", None)
    if meta is not None and getattr(meta, "name", None):
        return meta.name
    return getattr(tool, "name", None) or getattr(tool, "__name__", "unknown")


def _wrap_tool(tool: Any, engine: Any) -> Any:
    name = _tool_name(tool)
    hooked = False
    for attr in ("call", "acall", "__call__", "fn"):
        fn = getattr(tool, attr, None)
        if callable(fn):
            if set_attr(tool, attr, wrap_callable(engine, name, fn)):
                hooked = True
                if attr in ("call", "acall"):
                    continue  # wrap both sync + async entry points if present
                break
    if not hooked:
        raise GovernanceConfigurationError(
            framework="llama-index",
            reason=f"tool {name!r} exposes no enforceable callable",
        )
    return tool


__all__ = ["shield_tools", "executable_callbacks"]
