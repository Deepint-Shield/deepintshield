"""Discovery must describe the exact resource checked at execution time."""
from types import SimpleNamespace

import pytest

from deepintshield.agentic.engine import AgenticEngine, _permission_subject
from deepintshield.agentic.registry import _default_action, _manifest_tool
from deepintshield.agentic.types import DelegationContext


@pytest.mark.parametrize("name,kind", [("add_two", "write"), ("multiply_by_three", "write"), ("get_record", "read")])
def test_discovered_default_permission_matches_runtime_object(name, kind):
    action = _default_action(name, action_class=kind)
    engine = SimpleNamespace(agent_subject="agent:math", acting_user="", execution_id="", session_id="")
    payload = AgenticEngine._gaf_payload(engine, DelegationContext(tool=name, args_digest="sha256:" + "0" * 64))
    assert action["required_permission_key"] == f"{name}:{kind}"
    assert payload["object"] == _permission_subject(action["required_permission_key"])
    assert payload["permission"] == "holder"
    if name == "add_two":
        # Shared with the Go access-plan test: the client and server must use
        # exactly the same authorization resource, including canonical hashing.
        assert payload["object"] == "permission:v1-add-two-write--30c58ac6d0baee22722f4cfccfd1edcb"


def test_required_permission_survives_manifest_normalization():
    declaration = _manifest_tool("add_two")
    assert _manifest_tool(declaration)["actions"][0]["required_permission_key"] == "add_two:write"
