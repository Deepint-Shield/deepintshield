"""Executable callback inventories stay explicit, bounded, and framework-local."""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import subprocess
import sys
import types
import weakref
from types import SimpleNamespace

import pytest

from deepintshield.agentic.integrations import (
    autogen,
    google_adk,
    llamaindex,
    openai_agents,
    pydanticai,
    strands,
)
from deepintshield.agentic.registry import describe_network


def application_callback(value=None):
    return value


def second_application_callback(value=None):
    return value


def malicious_application_callback(payload=None):
    return eval(payload)  # noqa: S307 - scanner fixture


@pytest.fixture
def framework_callback(monkeypatch):
    module = types.ModuleType("llama_index.internal.events")
    exec("def callback(value=None):\n    return value\n", vars(module))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.callback


def _callbacks(inventory):
    return [entry.callback for entry in inventory.callbacks]


def _slots(inventory):
    return [entry.slot for entry in inventory.callbacks]


def test_callback_inventory_covers_only_documented_application_fields(framework_callback):
    agent = SimpleNamespace(
        callbacks=[application_callback, framework_callback],
        callback=None,
        step_callback=second_application_callback,
        # An arbitrary callable attribute is not a framework execution surface.
        internal_dispatch=application_callback,
    )
    inventory = llamaindex.executable_callbacks(agent, prefix="agent")
    assert _callbacks(inventory) == [application_callback, second_application_callback]
    assert _slots(inventory) == ["agent.callbacks[0]", "agent.step_callback"]
    assert inventory.incomplete is False


@pytest.mark.parametrize("copy_with_wraps", (False, True))
def test_copied_framework_module_metadata_cannot_hide_application_callback(
    framework_callback, copy_with_wraps,
):
    def application_wrapper(payload):
        return eval(payload)

    if copy_with_wraps:
        application_wrapper = functools.wraps(framework_callback)(application_wrapper)
    else:
        application_wrapper.__module__ = framework_callback.__module__
    agent = SimpleNamespace(callback=application_wrapper)

    inventory = llamaindex.executable_callbacks(agent)

    assert _callbacks(inventory) == [application_wrapper]
    assert inventory.incomplete is False


def test_attested_sdk_boundary_preserves_callback_implementation_ownership(framework_callback):
    from deepintshield.agentic.errors import public_agentic_boundary

    native = public_agentic_boundary(framework_callback)
    application = public_agentic_boundary(malicious_application_callback)
    agent = SimpleNamespace(callbacks=[native, application])

    inventory = llamaindex.executable_callbacks(agent)

    assert _callbacks(inventory) == [application]


def test_native_adk_plugin_sdk_callback_is_excluded():
    try:
        importlib.metadata.version("google-adk")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("google-adk is not installed")
    # Native framework imports participate in global automatic instrumentation.
    # Keep this real-package check isolated from tests of older mocked APIs.
    result = subprocess.run(
        [sys.executable, "-c", "\n".join((
            "from types import SimpleNamespace",
            "from deepintshield.agentic.integrations import google_adk",
            "plugin = google_adk.plugin(object())",
            "inventory = google_adk.executable_callbacks(SimpleNamespace(plugins=[plugin]))",
            "assert inventory.callbacks == ()",
            "assert inventory.incomplete is False",
        ))],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_native_inherited_callback_is_excluded_but_application_override_is_retained(
    framework_callback,
):
    module = sys.modules[framework_callback.__module__]
    exec("class Handler:\n    def on_event_start(self, *args):\n        return None\n", vars(module))

    class ApplicationHandler(module.Handler):
        @functools.wraps(module.Handler.on_event_start)
        def on_event_start(self, *args):
            return args

    native = module.Handler()
    application = ApplicationHandler()
    agent = SimpleNamespace(callback_manager=SimpleNamespace(handlers=[native, application]))

    inventory = llamaindex.executable_callbacks(agent)

    assert _callbacks(inventory) == [application.on_event_start]


def test_application_wrapper_outside_native_package_remains_in_blueprint(monkeypatch, tmp_path):
    native_path = tmp_path / "llama_index" / "internal" / "events.py"
    native_path.parent.mkdir(parents=True)
    native_path.write_text("def callback(payload):\n    return payload\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("llama_index.internal.events", native_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def application_helper(payload):
        return payload

    helper = application_helper

    @functools.wraps(module.callback)
    def application_wrapper(payload):
        return eval(helper(payload))

    manifest = describe_network(_llama_agent(application_wrapper))

    assert manifest.get("code_artifacts_incomplete") is not True
    source = "\n".join(artifact["source"] for artifact in manifest["code_artifacts"])
    assert "def application_wrapper" in source
    assert "eval(helper(payload))" in source
    assert "def application_helper" in source

    def replacement_helper(payload):
        return payload[::-1]

    helper = replacement_helper
    changed = describe_network(_llama_agent(application_wrapper))
    assert changed["blueprint_digest"] != manifest["blueprint_digest"]
    assert any("def replacement_helper" in artifact["source"] for artifact in changed["code_artifacts"])


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
    agent = SimpleNamespace(callbacks=[application_callback] * 65)
    inventory = llamaindex.executable_callbacks(agent)
    assert len(inventory.callbacks) == 64
    assert inventory.incomplete is True


def test_openai_inventory_covers_agent_and_function_tool_callbacks():
    class Hooks:
        def on_start(self, *args):
            return None

    hooks = Hooks()
    guardrail = SimpleNamespace(guardrail_function=malicious_application_callback)
    tool = SimpleNamespace(
        is_enabled=second_application_callback,
        needs_approval=application_callback,
        _failure_error_function=malicious_application_callback,
        tool_input_guardrails=[guardrail],
    )
    agent = SimpleNamespace(
        instructions=application_callback,
        prompt="declarative-prompt",
        tool_use_behavior=second_application_callback,
        input_guardrails=[guardrail],
        output_guardrails=[guardrail],
        tools=[tool],
        hooks=hooks,
        handoffs=[SimpleNamespace(input_filter=application_callback)],
        internal_dispatch=lambda: None,
    )

    inventory = openai_agents.executable_callbacks(agent, prefix="root")

    assert inventory.incomplete is False
    assert hooks.on_start in _callbacks(inventory)
    assert _slots(inventory) == [
        "root.instructions", "root.tool_use_behavior",
        "root.input_guardrails[0].guardrail_function",
        "root.output_guardrails[0].guardrail_function",
        "root.tools[0].is_enabled", "root.tools[0].needs_approval",
        "root.tools[0]._failure_error_function",
        "root.tools[0].tool_input_guardrails[0].guardrail_function",
        "root.hooks[0].on_start", "root.handoffs[0].input_filter",
    ]


def test_openai_inventory_extracts_only_native_handoff_factory_bindings(monkeypatch):
    module = types.ModuleType("agents.handoffs")
    exec(
        "def handoff(agent, on_handoff):\n"
        "    async def invoke(ctx):\n"
        "        on_handoff(ctx)\n"
        "        return agent\n"
        "    return invoke\n"
        "def _invoke_handoff_with_redaction(fn, *args):\n"
        "    return fn(*args)\n",
        vars(module),
    )
    monkeypatch.setitem(sys.modules, "agents.handoffs", module)
    child = SimpleNamespace(name="child", tools=[], handoffs=[])
    native = functools.partial(
        module._invoke_handoff_with_redaction,
        module.handoff(child, malicious_application_callback),
    )
    handoff = SimpleNamespace(on_invoke_handoff=native)
    root = SimpleNamespace(handoffs=[handoff])

    inventory = openai_agents.executable_callbacks(root)
    children, incomplete = openai_agents.handoff_agents(root)

    assert _callbacks(inventory) == [malicious_application_callback]
    assert children == (child,)
    assert incomplete is False

    # A replacement with copied function metadata is not the factory code.
    @functools.wraps(module.handoff(child, malicious_application_callback))
    def replacement(ctx):
        return malicious_application_callback(ctx)

    assert openai_agents._factory_handoff_binding(replacement, "on_handoff") is None


@pytest.mark.parametrize("field", ("handoffs", "input_guardrails", "tools"))
def test_openai_callback_collections_are_bounded_and_fail_closed(field):
    agent = SimpleNamespace(**{field: [SimpleNamespace()] * 65})

    assert openai_agents.executable_callbacks(agent).incomplete is True


def test_openai_handoff_discovery_handles_direct_agents_refs_and_cycles():
    class Agent:
        def __init__(self, name):
            self.name = name
            self.tools = []
            self.handoffs = []

    root, child = Agent("root"), Agent("child")
    root.handoffs = [child, SimpleNamespace(_agent_ref=weakref.ref(child))]
    child.handoffs = [root]

    assert openai_agents.handoff_agents(root) == ((child,), False)
    assert openai_agents.handoff_agents(child) == ((root,), False)
    assert openai_agents.shield_agent(root, engine=object()) is root


def test_openai_shields_declared_handoff_tools_before_execution(monkeypatch):
    child_tool = SimpleNamespace(name="child_tool")
    child = SimpleNamespace(name="child", tools=[child_tool], handoffs=[])
    root = SimpleNamespace(name="root", tools=[], handoffs=[child])
    wrapped = []
    monkeypatch.setattr(openai_agents, "_wrap_tool", lambda tool, _engine: wrapped.append(tool))

    openai_agents.shield_agent(root, engine=object())

    assert wrapped == [child_tool]


@pytest.mark.parametrize("instruction_record", (False, True))
def test_pydantic_inventory_covers_registered_prompts_validators_and_preparation(instruction_record):
    instruction = (
        SimpleNamespace(instruction=application_callback)
        if instruction_record else application_callback
    )
    agent = SimpleNamespace(
        _instructions=[instruction],
        _system_prompt_functions=[SimpleNamespace(function=second_application_callback)],
        _system_prompt_dynamic_functions={},
        _output_validators=[SimpleNamespace(function=malicious_application_callback)],
        _function_toolset=SimpleNamespace(tools={
            "read": SimpleNamespace(prepare=application_callback, args_validator=second_application_callback),
        }),
        internal_dispatch=lambda: None,
    )

    inventory = pydanticai.executable_callbacks(agent, prefix="agent")

    assert inventory.incomplete is False
    assert _callbacks(inventory) == [
        application_callback, second_application_callback, malicious_application_callback,
        application_callback, second_application_callback,
    ]
    assert all("internal_dispatch" not in slot for slot in _slots(inventory))


def test_pydantic_callback_inventory_rejects_truncated_or_raising_registries():
    oversized = SimpleNamespace(_instructions=[SimpleNamespace(instruction=application_callback)] * 65)
    assert pydanticai.executable_callbacks(oversized).incomplete is True

    class RaisingAgent:
        @property
        def _output_validators(self):
            raise RuntimeError("callback registry unavailable")

    assert pydanticai.executable_callbacks(RaisingAgent()).incomplete is True


def test_unordered_callback_inventory_is_stable():
    first = SimpleNamespace(callbacks={second_application_callback, application_callback})
    second = SimpleNamespace(callbacks={application_callback, second_application_callback})
    first_inventory = llamaindex.executable_callbacks(first)
    second_inventory = llamaindex.executable_callbacks(second)
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
        describe_network(_autogen_team(malicious_application_callback)),
        describe_network(_llama_agent(malicious_application_callback)),
        describe_network(FakeADKAgent(malicious_application_callback)),
        describe_network(FakeStrandsAgent(malicious_application_callback)),
    ]

    assert [manifest["framework"] for manifest in manifests] == [
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
    """A declarative agent claims no executable coverage: no executable node, no
    coverage ledger entry, nothing incomplete. Since 3.2.1 it still reports one
    artifact - its definition - because the gateway refuses a report with no
    artifact at all as incomplete coverage and quarantines the reporter."""
    manifests = [
        describe_network(_autogen_team(None)),
        describe_network(_llama_agent(None)),
        describe_network(FakeADKAgent(None)),
        describe_network(FakeStrandsAgent(None)),
    ]

    for manifest in manifests:
        assert not any(node.get("executable") for node in manifest["nodes"])
        (definition,) = manifest["code_artifacts"]
        assert definition["subject_kind"] == "agent"
        assert definition["source"].startswith("# deepintshield program definition")
        assert "_incomplete" not in definition
        assert "code_artifact_coverage" not in manifest
        assert manifest.get("code_artifacts_incomplete") is not True
        assert manifest["self_attested_code_digest"].startswith("sha256:")


def test_callback_slot_changes_blueprint_digest_without_exposing_slot_name():
    before = FakeADKAgent(None)
    before.before_agent_callback = application_callback
    after = FakeADKAgent(None)
    after.after_agent_callback = application_callback

    first = describe_network(before)
    second = describe_network(after)

    assert first["blueprint_digest"] != second["blueprint_digest"]
    assert "before_agent_callback" not in first["code_artifacts"][0]["source"]
