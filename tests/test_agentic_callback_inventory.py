"""Executable callback inventories stay explicit, bounded, and framework-local."""

from __future__ import annotations

from types import SimpleNamespace

from deepintshield.agentic.integrations import (
    autogen,
    crewai,
    google_adk,
    llamaindex,
    strands,
)
from deepintshield.agentic.registry import describe_network


def application_callback(value=None):
    return value


def second_application_callback(value=None):
    return value


def malicious_application_callback(payload=None):
    return eval(payload)  # noqa: S307 - scanner fixture


def _framework_callback(value=None):
    return value


_framework_callback.__module__ = "crewai.internal.events"


def _callbacks(inventory):
    return [entry.callback for entry in inventory.callbacks]


def _slots(inventory):
    return [entry.slot for entry in inventory.callbacks]


def test_crewai_inventory_covers_only_documented_callback_fields():
    crew = SimpleNamespace(
        before_kickoff_callbacks=[application_callback, _framework_callback],
        after_kickoff_callbacks=[],
        task_callback=second_application_callback,
        step_callback=None,
        callbacks=[],
        callback=None,
        # An arbitrary callable attribute is not a framework execution surface.
        internal_dispatch=application_callback,
    )

    inventory = crewai.executable_callbacks(crew, prefix="crew")

    assert _callbacks(inventory) == [
        application_callback,
        second_application_callback,
    ]
    assert _slots(inventory) == [
        "crew.before_kickoff_callbacks[0]",
        "crew.task_callback",
    ]
    assert inventory.incomplete is False


def test_autogen_inventory_uses_explicit_hooks_not_reply_dispatch_internals():
    agent = SimpleNamespace(
        hook_lists={"process_message_before_send": [application_callback]},
        _reply_func_list=[{"reply_func": second_application_callback}],
        termination_condition=None,
    )

    inventory = autogen.executable_callbacks(agent, prefix="participant")

    assert _callbacks(inventory) == [application_callback]
    assert _slots(inventory) == [
        "participant.hook_lists.process_message_before_send[0]"
    ]


def test_autogen_inventory_unwraps_termination_and_private_handoffs():
    termination = SimpleNamespace(_func=application_callback)
    handoff = SimpleNamespace(on_handoff=second_application_callback)
    team = SimpleNamespace(
        hook_lists={},
        termination_condition=termination,
        _handoffs=[handoff],
    )

    inventory = autogen.executable_callbacks(team, prefix="team")

    assert application_callback in _callbacks(inventory)
    assert second_application_callback in _callbacks(inventory)


class ApplicationLlamaHandler:
    def on_event_start(self, *args, **kwargs):
        return None

    def on_event_end(self, *args, **kwargs):
        return None


def test_llamaindex_inventory_reads_only_explicit_callback_manager_handlers():
    handler = ApplicationLlamaHandler()
    agent = SimpleNamespace(
        callback=None,
        callbacks=[],
        step_callback=application_callback,
        callback_manager=SimpleNamespace(handlers=[handler]),
    )

    inventory = llamaindex.executable_callbacks(agent, prefix="agent")

    callbacks = _callbacks(inventory)
    assert application_callback in callbacks
    assert handler.on_event_start in callbacks
    assert handler.on_event_end in callbacks
    assert all("callback_manager.handlers" in slot for slot in _slots(inventory)[1:])


class ApplicationADKPlugin:
    async def before_tool_callback(self, **kwargs):
        return None


class SecondApplicationADKPlugin:
    async def before_model_callback(self, **kwargs):
        return None


def test_google_adk_inventory_covers_agent_callbacks_and_plugin_overrides():
    callback = application_callback
    plugin = ApplicationADKPlugin()
    agent = SimpleNamespace(
        before_agent_callback=callback,
        after_agent_callback=None,
        before_model_callback=None,
        after_model_callback=None,
        before_tool_callback=None,
        after_tool_callback=None,
        plugins=[plugin],
    )

    inventory = google_adk.executable_callbacks(agent, prefix="root")

    assert callback in _callbacks(inventory)
    assert plugin.before_tool_callback in _callbacks(inventory)


def test_google_adk_inventory_covers_runner_plugin_manager():
    plugin = ApplicationADKPlugin()
    runner = SimpleNamespace(
        plugin_manager=SimpleNamespace(plugins=[plugin]),
    )

    inventory = google_adk.executable_callbacks(runner, prefix="runner")

    assert _callbacks(inventory) == [plugin.before_tool_callback]


def test_google_adk_unordered_plugin_collection_is_stable():
    first_plugin = ApplicationADKPlugin()
    second_plugin = SecondApplicationADKPlugin()
    first = SimpleNamespace(plugins={first_plugin, second_plugin})
    second = SimpleNamespace(plugins={second_plugin, first_plugin})

    first_inventory = google_adk.executable_callbacks(first, prefix="runner")
    second_inventory = google_adk.executable_callbacks(second, prefix="runner")

    assert [
        callback.__func__ for callback in _callbacks(first_inventory)
    ] == [callback.__func__ for callback in _callbacks(second_inventory)]
    assert _slots(first_inventory) == _slots(second_inventory)


def test_strands_inventory_covers_registered_hook_callbacks_only():
    registry = SimpleNamespace(
        _callbacks={"BeforeToolInvocationEvent": [application_callback]}
    )
    agent = SimpleNamespace(
        hook_registry=registry,
        hooks=[],
        callbacks=[],
        callback_handler=None,
        before_tool_callback=None,
        after_tool_callback=None,
        executor=SimpleNamespace(callback=second_application_callback),
    )

    inventory = strands.executable_callbacks(agent, prefix="agent")

    assert _callbacks(inventory) == [application_callback]
    assert "executor" not in " ".join(_slots(inventory))


def test_callback_inventory_fails_closed_when_explicit_collection_is_too_large():
    crew = SimpleNamespace(
        before_kickoff_callbacks=[application_callback] * 65,
        after_kickoff_callbacks=[],
        task_callback=None,
        step_callback=None,
        callbacks=[],
        callback=None,
    )

    inventory = crewai.executable_callbacks(crew)

    assert len(inventory.callbacks) == 64
    assert inventory.incomplete is True


def test_unordered_callback_inventory_is_stable():
    first = SimpleNamespace(
        before_kickoff_callbacks={
            second_application_callback,
            application_callback,
        },
        after_kickoff_callbacks=[],
        task_callback=None,
        step_callback=None,
        callbacks=[],
        callback=None,
    )
    second = SimpleNamespace(
        before_kickoff_callbacks={
            application_callback,
            second_application_callback,
        },
        after_kickoff_callbacks=[],
        task_callback=None,
        step_callback=None,
        callbacks=[],
        callback=None,
    )

    first_inventory = crewai.executable_callbacks(first)
    second_inventory = crewai.executable_callbacks(second)

    assert _callbacks(first_inventory) == _callbacks(second_inventory)
    assert _slots(first_inventory) == _slots(second_inventory)


class RaisingAutoGenTeam:
    hook_lists = {}

    @property
    def termination_condition(self):
        raise RuntimeError("must-not-escape")


def test_raising_explicit_callback_field_is_fail_closed_metadata():
    inventory = autogen.executable_callbacks(RaisingAutoGenTeam())

    assert inventory.callbacks == ()
    assert inventory.incomplete is True


def _crew(callback):
    agent = SimpleNamespace(
        role="researcher",
        tools=[],
        callbacks=[],
        step_callback=callback,
        callback=None,
    )
    return SimpleNamespace(
        agents=[agent],
        tasks=[],
        tools=[],
        before_kickoff_callbacks=[],
        after_kickoff_callbacks=[],
        task_callback=None,
        step_callback=None,
        callbacks=[],
        callback=None,
    )


def _autogen_team(callback):
    participant = SimpleNamespace(name="planner", tools=[], hook_lists={})
    return SimpleNamespace(
        _participants=[participant],
        selector_func=callback,
        candidate_func=None,
        termination_condition=None,
    )


def _llama_agent(callback):
    return SimpleNamespace(
        name="rag-agent",
        tools=[],
        agent_worker=SimpleNamespace(tools=[]),
        callback=callback,
        callbacks=[],
        step_callback=None,
    )


class FakeADKAgent:
    def __init__(self, callback):
        self.name = "adk-agent"
        self.tools = []
        self.sub_agents = []
        self.before_agent_callback = None
        self.after_agent_callback = None
        self.before_model_callback = None
        self.after_model_callback = None
        self.before_tool_callback = callback
        self.after_tool_callback = None
        self.plugins = []


FakeADKAgent.__module__ = "google.adk.agents"


class FakeStrandsAgent:
    def __init__(self, callback):
        entries = [] if callback is None else [SimpleNamespace(callback=callback)]
        self.name = "strands-agent"
        self.tool_registry = SimpleNamespace(registry={})
        self.hooks = SimpleNamespace(_registered_callbacks={"before": entries})


FakeStrandsAgent.__module__ = "strands.agent"


def test_every_adapter_declares_and_captures_application_callbacks():
    manifests = [
        describe_network(_crew(malicious_application_callback)),
        describe_network(_autogen_team(malicious_application_callback)),
        describe_network(_llama_agent(malicious_application_callback)),
        describe_network(FakeADKAgent(malicious_application_callback)),
        describe_network(FakeStrandsAgent(malicious_application_callback)),
    ]

    assert [manifest["framework"] for manifest in manifests] == [
        "crewai",
        "autogen",
        "llamaindex",
        "google_adk",
        "strands",
    ]
    for manifest in manifests:
        executable = [node for node in manifest["nodes"] if node.get("executable")]
        assert executable, manifest["framework"]
        executable_keys = {node["id"] for node in executable}
        assert {
            artifact["subject_key"] for artifact in manifest["code_artifacts"]
        } == executable_keys
        assert {
            entry["subject_key"]
            for entry in manifest["code_artifact_coverage"]
            if entry["status"] == "captured"
        } == executable_keys
        assert all(
            "eval(payload)" in artifact["source"]
            for artifact in manifest["code_artifacts"]
        )
        assert manifest.get("code_artifacts_incomplete") is not True


def test_declarative_agents_do_not_claim_executable_callback_coverage():
    manifests = [
        describe_network(_crew(None)),
        describe_network(_autogen_team(None)),
        describe_network(_llama_agent(None)),
        describe_network(FakeADKAgent(None)),
        describe_network(FakeStrandsAgent(None)),
    ]

    for manifest in manifests:
        assert not any(node.get("executable") for node in manifest["nodes"])
        assert manifest["code_artifacts"] == []
        assert "code_artifact_coverage" not in manifest
        assert manifest.get("code_artifacts_incomplete") is not True


def test_callback_slot_changes_blueprint_digest_without_exposing_slot_name():
    before = _crew(None)
    before.before_kickoff_callbacks = [application_callback]
    after = _crew(None)
    after.after_kickoff_callbacks = [application_callback]

    first = describe_network(before)
    second = describe_network(after)

    assert first["blueprint_digest"] != second["blueprint_digest"]
    assert "before_kickoff_callbacks" not in first["code_artifacts"][0]["source"]
