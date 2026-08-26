"""The SDK side of the obligation contract the gateway's templates rely on.

`framework/agentic/template_obligations_test.go` maintains an
`enforcedObligations` map documented as "a transformation the SDK's handler
table actually performs on the call's arguments (obligations.py)". A Go test
cannot see a Python dict, so that assertion passed while `redact:value` had no
handler here - a shipped secrets template promising redaction that nothing
performed, which is the exact class that file exists to bound.

This closes the loop from the other side: the Go map is parsed and every
obligation it claims is enforced must resolve to a handler.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from deepintshield.agentic.obligations import _HANDLERS, apply_obligations

_GO_TEST = (
    pathlib.Path(__file__).resolve().parents[2]
    / "deepintshield_server"
    / "framework"
    / "agentic"
    / "template_obligations_test.go"
)


def _server_enforced_obligations() -> set[str]:
    source = _GO_TEST.read_text()
    block = re.search(
        r"var enforcedObligations = map\[string\]bool\{(.*?)\n\}", source, re.S
    )
    assert block, "enforcedObligations map not found; the Go test was restructured"
    return set(re.findall(r'"([a-z:\-]+)":\s*true', block.group(1)))


@pytest.mark.skipif(not _GO_TEST.exists(), reason="server tree not present")
def test_every_server_enforced_obligation_has_an_sdk_handler() -> None:
    missing = sorted(_server_enforced_obligations() - set(_HANDLERS))
    assert not missing, (
        f"the gateway's templates promise {missing} as enforced redaction, but "
        f"obligations.py has no handler. Add one, or drop it from "
        f"enforcedObligations in template_obligations_test.go."
    )


def test_every_handler_redacts_something() -> None:
    """A handler that silently passes its input through is not a control."""
    probe = {
        "email": "ada@example.com",
        "api_key": "sk-live-9f2c41d7a8b3e5061c4d",
        "diagnosis": "E11.9",
        "card": "4111 1111 1111 1111",
        "iban": "GB33BUKB20201555555555",
        "value": "hunter2",
    }
    for obligation in _HANDLERS:
        assert apply_obligations(probe, [obligation]) != probe, (
            f"{obligation} changed nothing on a payload built to trigger every "
            f"handler; it is registered but inert"
        )


def test_unknown_obligation_is_a_local_no_op() -> None:
    payload = {"amount": 12}
    assert apply_obligations(payload, ["throttle:per-minute=5"]) == payload
