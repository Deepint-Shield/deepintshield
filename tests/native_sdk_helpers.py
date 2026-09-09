"""Select mock transports from the installed native SDK's HTTP library."""

import importlib

import pytest


def openai_backend():
    sdk = pytest.importorskip("openai")
    backend = next(
        cls.__module__.partition(".")[0]
        for cls in sdk.DefaultHttpxClient.__mro__ if cls.__name__ == "Client"
    )
    return sdk, importlib.import_module(backend)
