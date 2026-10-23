"""Exercise interpreter-sensitive enforcement behavior on every supported Python."""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import warnings

import pytest

from deepintshield._compat import call_signature, is_coroutine_function
from deepintshield.agentic.obligations import apply_obligations_to_call


def _annotated_tool():
    namespace = {}
    source = (
        "def tool(email: missing.Email, /, *, phone: missing.Phone = '') -> missing.Result:\n"
        "    return email, phone\n"
    )
    if sys.version_info < (3, 14):
        source = "from __future__ import annotations\n" + source
    # Do not inherit this test module's future flag: Python 3.14 must exercise
    # real deferred annotations, including unresolved type-only imports.
    exec(compile(source, "<annotation-fixture>", "exec", dont_inherit=True), namespace)
    return namespace["tool"]


def test_type_only_annotations_do_not_prevent_positional_masking():
    tool = _annotated_tool()
    args, kwargs = apply_obligations_to_call(
        tool, ("ada@example.com",), {"phone": "+1 202 555 0123"}, ["mask:pii"]
    )
    assert tool(*args, **kwargs) == ("***", "***")


def test_deferred_annotations_preserve_binding_for_wrappers_and_partials():
    tool = _annotated_tool()

    @functools.wraps(tool)
    def wrapped(*args, **kwargs):
        return tool(*args, **kwargs)

    for candidate in (tool, wrapped, functools.partial(tool, phone="***")):
        signature = call_signature(candidate)
        bound = signature.bind("ada@example.com")
        assert bound.arguments == {"email": "ada@example.com"}
        assert signature.parameters["email"].kind is inspect.Parameter.POSITIONAL_ONLY
        assert signature.parameters["phone"].kind is inspect.Parameter.KEYWORD_ONLY


def test_async_detection_preserves_partial_and_legacy_framework_markers():
    async def tool(value):
        return value

    def legacy(value):
        return tool(value)

    marker = getattr(asyncio.coroutines, "_is_coroutine", None)
    if marker is not None:
        legacy._is_coroutine = marker
        assert is_coroutine_function(legacy)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert is_coroutine_function(tool)
        assert is_coroutine_function(functools.partial(tool, 1))
        assert not is_coroutine_function(lambda: 1)


@pytest.mark.parametrize("allowed", [False, True])
def test_async_tool_authorization_runs_when_awaited(shield_factory, allowed):
    import httpx

    calls = []
    executed = []

    def handler(request):
        if request.url.path.endswith("/vk-credential-info"):
            return httpx.Response(200, json={"provider_type": "", "agent_configured": False})
        if request.url.path.endswith("/agentic-new/decide"):
            calls.append("decide")
            return httpx.Response(200, json={
                "verdict": "ALLOW" if allowed else "DENY",
                "allow": allowed, "proceed": allowed, "mode": "enforce",
            })
        return httpx.Response(404)

    async def tool(value):
        executed.append(value)
        return value * 2

    from deepintshield.agentic.integrations._common import wrap_callable

    shield = shield_factory(handler)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        guarded = wrap_callable(shield.agentic.engine, "tool", tool)
    assert inspect.iscoroutinefunction(guarded)
    pending = guarded(21)
    assert calls == []
    assert executed == []
    if allowed:
        assert asyncio.run(pending) == 42
        assert executed == [21]
    else:
        with pytest.raises(PermissionError):
            asyncio.run(pending)
        assert executed == []
    assert calls == ["decide"]
