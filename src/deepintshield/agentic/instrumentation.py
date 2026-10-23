"""Recognize SDK instrumentation without trusting copied wrapper markers.

Source capture follows application decorators and helpers, but SDK guards are
not application code even when an editable checkout lives inside the project.
The allowlist below identifies exact code objects owned by our loaded wrapper
factories and the binding each guard actually calls. It grants no exemption to
third-party functions or to functions merely claiming an SDK module name.
"""

from __future__ import annotations

import inspect
import sys
from functools import lru_cache
from typing import Any


# Factory path -> executed target binding(s). A leading '*' denotes a keyword
# default used to freeze the original method while patching a loop of methods.
# A factory may emit several variants, each exposing one of these bindings.
_FACTORIES: dict[str, dict[str, tuple[str, ...]]] = {
    "deepintshield.agent": {"AgentSurface.tool": ("fn",)},
    "deepintshield.rag": {
        "RAGSurface.guard_retriever": ("original",),
        "RAGSurface._wrap_embed": ("original",),
    },
    "deepintshield.agentic.decorators": {"shield_tool": ("fn",)},
    "deepintshield.agentic.errors": {"public_agentic_boundary": ("function",)},
    "deepintshield.agentic.enforcement": {
        "_watch_imports": ("original",),
        "_make_topology_reporter": ("orig",),
    },
    "deepintshield.agentic.integrations._common": {
        "wrap_callable": ("fn",),
        "install_method_guard": ("orig",),
    },
    "deepintshield.agentic.integrations.langgraph": {
        "_wrap_fn": ("fn",),
        "_install_execution_guards": ("*_orig",),
        "enforce": ("orig_compile",),
    },
    "deepintshield.agentic.integrations.openai_agents": {
        "enforce": ("*_orig",),
        "_wrap_tool": ("original",),
    },
    "deepintshield.agentic.integrations.langchain": {
        "enforce": ("*_original",),
    },
    "deepintshield.agentic.integrations.google_adk": {
        "_patch_final_dispatch": ("*_original", "original_live"),
        "_patch_plugin_manager_fallback": ("original",),
    },
    "deepintshield.agentic.integrations.google_genai": {
        "_guard_afc_callable": ("guarded",),
        "_guard_generation": ("original",),
    },
    "deepintshield.agentic.integrations.pydanticai": {
        "_install_tool_manager_guard": ("original",),
    },
    "deepintshield.agentic.integrations.strands": {
        "_enforce_middleware_terminal": ("original",),
        "enforce": ("original",),
    },
    "deepintshield.agentic.integrations.temporal": {"enforce": ("original",)},
    "deepintshield.agentic.integrations.hermes": {"enforce": ("original",)},
    "deepintshield.agentic.integrations.litellm": {"enforce": ("orig", "aorig")},
}
_MAX_WRAPPER_CHAIN = 8


@lru_cache(maxsize=128)
def _nested_codes(code: Any) -> tuple[Any, ...]:
    """Keep exact code objects alive; membership below always uses identity."""
    found: list[Any] = []
    pending = [code]
    while pending and len(found) < 256:
        current = pending.pop()
        for nested in current.co_consts:
            if inspect.iscode(nested):
                found.append(nested)
                pending.append(nested)
    return tuple(found)


def _factory(module: Any, path: str) -> Any:
    current = module
    for component in path.split("."):
        if not (inspect.ismodule(current) or inspect.isclass(current)):
            return None
        current = vars(current).get(component)
        if isinstance(current, (staticmethod, classmethod)):
            current = current.__func__
    # Several factories themselves expose the SDK's public error boundary.
    # Follow their inert function dictionaries to the module-owned factory.
    seen: set[int] = set()
    for _ in range(_MAX_WRAPPER_CHAIN):
        if not inspect.isfunction(current) or id(current) in seen:
            return None
        if current.__globals__ is vars(module):
            return current
        seen.add(id(current))
        current = vars(current).get("__wrapped__")
    return None


def _executed_target(fn: Any) -> Any:
    if inspect.ismethod(fn):
        fn = fn.__func__
    if not inspect.isfunction(fn):
        return None
    module_name = fn.__globals__.get("__name__")
    factories = _FACTORIES.get(module_name)
    module = sys.modules.get(module_name) if isinstance(module_name, str) else None
    if not factories or not inspect.ismodule(module) or fn.__globals__ is not vars(module):
        return None
    for path, bindings in factories.items():
        factory = _factory(module, path)
        if factory is None or not any(fn.__code__ is code for code in _nested_codes(factory.__code__)):
            continue
        targets: list[Any] = []
        for binding in bindings:
            if binding.startswith("*"):
                target = (fn.__kwdefaults__ or {}).get(binding[1:])
            else:
                try:
                    target = fn.__closure__[fn.__code__.co_freevars.index(binding)].cell_contents
                except (AttributeError, IndexError, TypeError, ValueError):
                    continue
            if callable(target):
                targets.append(target)
        if targets and all(target is targets[0] for target in targets):
            return targets[0]
    return None


def is_factory_function(fn: Any, module: Any, path: str) -> bool:
    """Whether a callable is exact nested implementation code of a factory.

    Framework-specific adapters can use this to decode a known native wrapper
    without trusting its public attributes or discarding arbitrary decorators.
    The caller supplies an explicitly supported, already-loaded module.
    """
    if not inspect.isfunction(fn) or not inspect.ismodule(module):
        return False
    if fn.__globals__ is not vars(module):
        return False
    factory = _factory(module, path)
    return factory is not None and any(
        fn.__code__ is code for code in _nested_codes(factory.__code__)
    )


def is_sdk_instrumentation_wrapper(fn: Any) -> bool:
    """Whether the exact SDK implementation reaches its advertised target.

    ``@wraps(original) @public_agentic_boundary`` hides one SDK guard behind
    the boundary's ``__wrapped__`` metadata. Verify each actual SDK call in
    that chain; an application wrapper, altered globals, code replacement, or
    mismatched target prevents the exemption and remains subject to capture.
    """
    if not inspect.isfunction(fn):
        return False
    advertised = vars(fn).get("__wrapped__")
    if not callable(advertised):
        return False
    current = fn
    seen: set[int] = set()
    for _ in range(_MAX_WRAPPER_CHAIN):
        if id(current) in seen:
            return False
        seen.add(id(current))
        target = _executed_target(current)
        if target is advertised:
            return True
        if target is None:
            return False
        current = target
    return False
