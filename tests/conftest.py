from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest

from deepintshield import DeepintShield


@pytest.fixture
def mock_transport() -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.MockTransport]:
    def make(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
        return httpx.MockTransport(handler)

    return make


@pytest.fixture
def shield_factory(mock_transport):
    created: list[DeepintShield] = []

    def make(
        handler: Callable[[httpx.Request], httpx.Response],
        **client_kwargs: Any,
    ) -> DeepintShield:
        def with_blueprint_ack(request: httpx.Request) -> httpx.Response:
            response = handler(request)
            # Current Agentic execution requires a durable discovery/blueprint
            # acknowledgement before a callable can run. Most focused PDP
            # tests predate that endpoint and use 404 as their generic fallback;
            # supply the unrelated successful preflight while preserving any
            # explicit discovery response/failure a test defines.
            if (
                request.url.path.endswith("/agentic-new/registry/discover")
                and response.status_code == 404
            ):
                return httpx.Response(200, json={"blueprint_scan_status": "complete"})
            return response

        shield = DeepintShield(virtual_key="sk-ds-test", **client_kwargs)
        shield._client.close()
        shield._client = httpx.Client(transport=mock_transport(with_blueprint_ack))
        created.append(shield)
        return shield

    yield make
    for shield in created:
        shield.close()


@pytest.fixture
def allow_response() -> dict[str, Any]:
    return {"result": {"decision": "allow", "reason": ""}}


@pytest.fixture
def block_response() -> dict[str, Any]:
    return {"result": {"decision": "block", "reason": "policy violation"}}
