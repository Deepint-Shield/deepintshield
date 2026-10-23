"""Small interpreter compatibility helpers for supported Python versions."""

from __future__ import annotations

import asyncio
import inspect
import sys
from typing import Any

if sys.version_info >= (3, 14):
    from annotationlib import Format

    _SIGNATURE_OPTIONS: dict[str, Any] = {"annotation_format": Format.STRING}
else:
    _SIGNATURE_OPTIONS = {"eval_str": False}


def call_signature(function: Any) -> inspect.Signature:
    """Read parameter shape without requiring resolved annotation values.

    Python 3.14 evaluates deferred annotations by default when inspecting a
    signature. Binding tool arguments needs only names and parameter kinds;
    resolving a TYPE_CHECKING-only name can otherwise break enforcement.
    """
    return inspect.signature(function, **_SIGNATURE_OPTIONS)


def is_coroutine_function(function: Any) -> bool:
    """Recognize async tools without the deprecated asyncio predicate."""
    if inspect.iscoroutinefunction(function):
        return True
    # Preserve asyncio's legacy marker for frameworks supporting Python
    # 3.10/3.11, before inspect.markcoroutinefunction became available.
    marker = getattr(asyncio.coroutines, "_is_coroutine", None)
    return marker is not None and getattr(function, "_is_coroutine", None) is marker
