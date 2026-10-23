"""Govern native Google GenAI automatic Python function calls.

Only callable tools on a client returned by DeepintShield are adapted. Native
model requests and declarative tool schemas retain the provider path. Copies
of request configuration preserve the caller's reusable tool list and native
function signatures, including chat and streaming APIs.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import Any, Iterator

from ..errors import _raise_detached_error, public_agentic_error_boundary
from ..gate import preflight
from ._common import callable_tool_name, wrap_callable


class _AFCGovernanceStop(BaseException):
    """Carry a safe governance failure past GenAI's tool-error-to-model catch.

    GenAI intentionally catches ordinary tool Exceptions and sends them back
    to the model. A governance denial must instead stop the generation call.
    This private signal is always unwrapped at our native API boundary.
    """

    def __init__(self, error: Exception) -> None:
        self.error = error


@contextmanager
def _governance_boundary() -> Iterator[None]:
    failure: Exception | None = None
    try:
        with public_agentic_error_boundary():
            yield
    except _AFCGovernanceStop as stop:
        failure = stop.error
    if failure is not None:
        _raise_detached_error(failure)


def _guard_afc_callable(engine: Any, fn: Any) -> Any:
    guarded = wrap_callable(engine, callable_tool_name(fn, "unknown"), fn)

    if inspect.iscoroutinefunction(guarded):
        @functools.wraps(guarded)
        async def asynchronous(*args: Any, **kwargs: Any) -> Any:
            try:
                return await guarded(*args, **kwargs)
            except Exception as error:
                if getattr(error, "_deepintshield_error_code", ""):
                    raise _AFCGovernanceStop(error) from None
                raise

        return asynchronous

    @functools.wraps(guarded)
    def synchronous(*args: Any, **kwargs: Any) -> Any:
        try:
            return guarded(*args, **kwargs)
        except Exception as error:
            if getattr(error, "_deepintshield_error_code", ""):
                raise _AFCGovernanceStop(error) from None
            raise

    return synchronous


def _prepare_config(config: Any, engine: Any) -> Any:
    tools = config.get("tools") if isinstance(config, dict) else getattr(config, "tools", None)
    if not isinstance(tools, (list, tuple)) or not any(callable(tool) for tool in tools):
        return config
    afc = config.get("automatic_function_calling") if isinstance(config, dict) else getattr(config, "automatic_function_calling", None)
    if (afc.get("disable") if isinstance(afc, dict) else getattr(afc, "disable", False)) is True:
        return config
    # The gateway must receive code before an initial model request can be
    # admitted for a newly discovered agent. Per-action authorization remains
    # at actual function dispatch, where the model's arguments are known.
    for tool in tools:
        if callable(tool):
            preflight(engine, callable_tool_name(tool, "unknown"), tool)
    wrapped = [_guard_afc_callable(engine, tool) if callable(tool) else tool for tool in tools]
    if isinstance(config, dict):
        return {**config, "tools": wrapped}
    return config.model_copy(update={"tools": wrapped})


def _guard_generation(original: Any, engine: Any, *, asynchronous: bool, stream: bool) -> Any:
    def options(kwargs: dict[str, Any]) -> dict[str, Any]:
        config = kwargs.get("config") or getattr(getattr(original, "__self__", None), "_config", None)
        prepared = _prepare_config(config, engine)
        return {**kwargs, "config": prepared} if prepared is not config else kwargs

    if asynchronous and stream:
        @functools.wraps(original)
        async def async_stream(*args: Any, **kwargs: Any) -> Any:
            with _governance_boundary():
                response = await original(*args, **options(kwargs))

            async def iterate() -> Any:
                with _governance_boundary():
                    try:
                        async for item in response:
                            yield item
                    finally:
                        close = getattr(response, "aclose", None)
                        if callable(close):
                            await close()

            return iterate()

        return async_stream
    if asynchronous:
        @functools.wraps(original)
        async def async_call(*args: Any, **kwargs: Any) -> Any:
            with _governance_boundary():
                return await original(*args, **options(kwargs))

        return async_call
    if stream:
        @functools.wraps(original)
        def sync_stream(*args: Any, **kwargs: Any) -> Any:
            with _governance_boundary():
                yield from original(*args, **options(kwargs))

        return sync_stream

    @functools.wraps(original)
    def sync_call(*args: Any, **kwargs: Any) -> Any:
        with _governance_boundary():
            return original(*args, **options(kwargs))

    return sync_call


def _bind_chat_factory(factory: Any, *, asynchronous: bool) -> None:
    # GenAI 2.x executes AFC in Chat; earlier releases delegate it to Models.
    # Preserve both native paths and bind only chats using our model instance.
    cls = type(factory)
    original = cls.create
    if getattr(original, "_deepintshield_afc_factory", False):
        return

    @functools.wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        chat = original(self, *args, **kwargs)
        engine = getattr(self._modules, "_deepintshield_afc_engine", None)
        if engine is not None:
            for name, stream in (("send_message", False), ("send_message_stream", True)):
                setattr(chat, name, _guard_generation(getattr(chat, name), engine, asynchronous=asynchronous, stream=stream))
        return chat

    create._deepintshield_afc_factory = True
    cls.create = create


def bind_client(client: Any, *, engine: Any) -> Any:
    """Bind native surfaces without replacing their clients or AFC loops."""
    for models, asynchronous in ((client.models, False), (client.aio.models, True)):
        models._deepintshield_afc_engine = engine
        for name, stream in (("generate_content", False), ("generate_content_stream", True)):
            original = getattr(models, name)
            setattr(models, name, _guard_generation(original, engine, asynchronous=asynchronous, stream=stream))
    _bind_chat_factory(client.chats, asynchronous=False)
    _bind_chat_factory(client.aio.chats, asynchronous=True)
    return client
