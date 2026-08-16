"""Stable, centralized error codes for the DeepintShield Python SDK.

The catalogue is static and immutable.  Looking up a definition is an in-memory
``dict`` lookup; no error API performs network or filesystem I/O.  Error-code
values are part of the public compatibility contract: new values may be added,
but an existing value must not be repurposed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class ErrorCategory(str, Enum):
    """Feature area that owns an SDK error code."""

    CLIENT = "client"
    CONFIGURATION = "configuration"
    TRANSPORT = "transport"
    CHAT = "chat"
    GUARDRAIL = "guardrail"
    RAG = "rag"
    AGENT = "agent"
    AGENTIC = "agentic"
    MCP = "mcp"
    PROVIDER = "provider"
    FRAMEWORK = "framework"
    VALIDATION = "validation"


class ErrorCode(str, Enum):
    """Stable machine-readable SDK error codes.

    Members serialize to their lower-case values.  The enum member names and
    values are intentionally explicit so application code can use either
    ``ErrorCode.RATE_LIMITED`` or the wire-safe string ``"rate_limited"``.
    """

    # Client, configuration, validation, and HTTP/transport.
    SDK_ERROR = "sdk_error"
    CONFIGURATION_ERROR = "configuration_error"
    VIRTUAL_KEY_MISSING = "virtual_key_missing"
    CLIENT_CLOSED = "client_closed"
    INVALID_ARGUMENT = "invalid_argument"
    VALIDATION_ERROR = "validation_error"
    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    TRANSPORT_ERROR = "transport_error"
    TRANSPORT_TIMEOUT = "transport_timeout"
    HTTP_ERROR = "http_error"
    INVALID_RESPONSE = "invalid_response"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_NOT_FOUND = "resource_not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    FEATURE_LOCKED = "feature_locked"
    SERVER_ERROR = "server_error"
    INTERNAL_ERROR = "internal_error"

    # Primary SDK surfaces.
    CHAT_REQUEST_FAILED = "chat_request_failed"
    CHAT_STREAM_INVALID_EVENT = "chat_stream_invalid_event"
    GUARDRAIL_EVALUATION_FAILED = "guardrail_evaluation_failed"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    RAG_EVALUATION_FAILED = "rag_evaluation_failed"
    RAG_RETRIEVER_UNSUPPORTED = "rag_retriever_unsupported"
    RAG_EMBEDDER_UNSUPPORTED = "rag_embedder_unsupported"
    AGENT_INVOCATION_INVALID = "agent_invocation_invalid"
    MCP_DEPENDENCY_MISSING = "mcp_dependency_missing"
    MCP_CONNECTION_FAILED = "mcp_connection_failed"
    MCP_PROTOCOL_ERROR = "mcp_protocol_error"
    MCP_EXECUTION_FAILED = "mcp_execution_failed"
    MCP_DISCOVERY_FAILED = "mcp_discovery_failed"
    MCP_TOOL_NAME_INVALID = "mcp_tool_name_invalid"
    MCP_ARGUMENTS_INVALID = "mcp_arguments_invalid"
    MCP_TOOL_AUTHORIZATION_DENIED = "mcp_tool_authorization_denied"
    MCP_TOOL_AUTHORIZATION_UNAVAILABLE = "mcp_tool_authorization_unavailable"
    MCP_TOOL_APPROVAL_REQUIRED = "mcp_tool_approval_required"
    PROVIDER_DEPENDENCY_MISSING = "provider_dependency_missing"
    PROVIDER_INITIALIZATION_FAILED = "provider_initialization_failed"
    FRAMEWORK_BINDER_NOT_FOUND = "framework_binder_not_found"
    FRAMEWORK_BINDER_ATTRIBUTE_MISSING = "framework_binder_attribute_missing"

    # Agentic/PDP public boundary and gateway lifecycle.
    AGENTIC_ERROR = "agentic_error"
    GOVERNANCE_CONFIGURATION_ERROR = "governance_configuration_error"
    GUARDRAIL_DENIED = "guardrail_denied"
    REQUIRE_APPROVAL = "require_approval"
    MASK_OBLIGATION_UNSUPPORTED = "mask_obligation_unsupported"
    GATEWAY_UNAVAILABLE = "gateway_unavailable"
    INVALID_GATEWAY_RESPONSE = "invalid_gateway_response"
    AGENT_REGISTRATION_PENDING = "agent_registration_pending"
    AGENT_NOT_REGISTERED = "agent_not_registered"
    AGENT_REGISTRATION_DENIED = "agent_registration_denied"
    AGENT_REGISTRATION_NOT_READY = "agent_registration_not_ready"
    AGENT_REGISTRATION_APPROVAL_REQUIRED = "agent_registration_approval_required"
    AGENT_REGISTRATION_QUOTA_EXCEEDED = "agent_registration_quota_exceeded"
    AGENT_REGISTRATION_REVIEW_STALE = "agent_registration_review_stale"
    AGENT_REGISTRATION_REVIEW_CONFLICT = "agent_registration_review_conflict"
    AGENT_APPROVAL_PENDING = "agent_approval_pending"
    GUARDRAIL_APPROVAL_PENDING = "guardrail_approval_pending"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_ACCESS_DENIED = "approval_access_denied"
    APPROVAL_STORE_UNAVAILABLE = "approval_store_unavailable"
    AUTHZ_STORE_UNAVAILABLE = "authz_store_unavailable"
    LEGACY_PDP_UNAVAILABLE = "legacy_pdp_unavailable"
    AGENT_BLUEPRINT_REVIEW_PENDING = "agent_blueprint_review_pending"
    AGENT_BLUEPRINT_REVIEW_DENIED = "agent_blueprint_review_denied"
    BLUEPRINT_SCAN_UNAVAILABLE = "blueprint_scan_unavailable"
    BLUEPRINT_SCAN_REQUIRED = "blueprint_scan_required"
    BLUEPRINT_SCANNING_REQUIRED = "blueprint_scanning_required"
    BLUEPRINT_REGISTRATION_FAILED = "blueprint_registration_failed"
    BLUEPRINT_COVERAGE_INCOMPLETE = "blueprint_coverage_incomplete"
    BLUEPRINT_MANIFEST_TOO_LARGE = "blueprint_manifest_too_large"
    BLUEPRINT_REMOTE_TOOL_UNVERIFIED = "blueprint_remote_tool_unverified"
    BLUEPRINT_MCP_INVENTORY_UNAVAILABLE = "blueprint_mcp_inventory_unavailable"
    BLUEPRINT_MODEL_SCAN_PENDING = "blueprint_model_scan_pending"
    BLUEPRINT_MODEL_SCAN_FAILED = "blueprint_model_scan_failed"
    BLUEPRINT_MODEL_UNAVAILABLE = "blueprint_model_unavailable"
    INVALID_BLUEPRINT_MANIFEST = "invalid_blueprint_manifest"
    CREDENTIAL_CONFIGURATION_ERROR = "credential_configuration_error"
    CREDENTIAL_PROVIDER_UNSUPPORTED = "credential_provider_unsupported"
    CREDENTIAL_DEPENDENCY_MISSING = "credential_dependency_missing"
    CREDENTIAL_EXCHANGE_FAILED = "credential_exchange_failed"
    FRAMEWORK_DEPENDENCY_MISSING = "framework_dependency_missing"
    FRAMEWORK_INTEGRATION_UNSUPPORTED = "framework_integration_unsupported"
    AGENT_DECISION_CONTEXT_MISSING = "agent_decision_context_missing"
    PRINCIPAL_IDENTIFIER_MISSING = "principal_identifier_missing"
    REGISTRY_DISCOVERY_EMPTY = "registry_discovery_empty"
    REGISTRY_DISCOVERY_INVALID = "registry_discovery_invalid"
    REGISTRY_DISCOVERY_PENDING = "registry_discovery_pending"
    REGISTRY_DISCOVERY_REJECTED = "registry_discovery_rejected"
    REGISTRY_DISCOVERY_UNAVAILABLE = "registry_discovery_unavailable"
    REGISTRY_UNAVAILABLE = "registry_unavailable"
    WORKLOAD_PROOF_REQUIRED = "workload_proof_required"

    # Known authorization denial codes emitted by the Agentic gateway.
    AGENT_ACCESS_DENIED = "agent_access_denied"
    AUTHORIZATION_DENIED = "authorization_denied"
    AUTHZ_ENGINE_ERROR = "authz_engine_error"
    CONTEXT_DENY = "context_deny"
    NO_STORE = "no_store"
    OBO_ACTION_NOT_ALLOWED = "obo_action_not_allowed"
    OBO_DELEGATION_REQUIRED = "obo_delegation_required"
    OBO_NO_ACTS_FOR = "obo_no_acts_for"
    OBO_SCOPE_MISMATCH = "obo_scope_mismatch"
    OBO_TOOL_NOT_ALLOWED = "obo_tool_not_allowed"
    OBO_USER_LACKS_PERM = "obo_user_lacks_perm"
    OBO_USER_MISMATCH = "obo_user_mismatch"


@dataclass(frozen=True, slots=True)
class ErrorDefinition:
    """Trusted metadata associated with one stable error code."""

    code: str
    category: ErrorCategory
    description: str
    retryable: bool = False
    action: str = ""
    dashboard_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of this immutable definition."""
        return {
            "code": self.code,
            "category": self.category.value,
            "description": self.description,
            "retryable": self.retryable,
            "action": self.action,
            "dashboard_path": self.dashboard_path,
        }


_REGISTRATIONS = "/workspace/agentic-new/registry?tab=agents"
_APPROVALS = "/workspace/agentic-new/authorization?tab=approvals"
_ACTIVITY = "/workspace/agentic-new/activity?tab=decisions"
_BLUEPRINTS = "/workspace/agentic-new/blueprint-scans"
_IDENTITIES = "/workspace/agentic-new/identities?tab=identity-providers"
_MCP = "/workspace/mcp-registry"
_OVERVIEW = "/workspace/agentic-new/overview"
_VIRTUAL_KEYS = "/workspace/access/virtual-keys"


def _definition(
    code: ErrorCode,
    category: ErrorCategory,
    description: str,
    *,
    retryable: bool = False,
    action: str = "",
    dashboard_path: str = "",
) -> ErrorDefinition:
    return ErrorDefinition(
        code=code.value,
        category=category,
        description=description,
        retryable=retryable,
        action=action,
        dashboard_path=dashboard_path,
    )


_DEFINITIONS = (
    _definition(ErrorCode.SDK_ERROR, ErrorCategory.CLIENT, "The DeepIntShield SDK operation failed."),
    _definition(ErrorCode.CONFIGURATION_ERROR, ErrorCategory.CONFIGURATION, "The SDK configuration is invalid or incomplete."),
    _definition(ErrorCode.VIRTUAL_KEY_MISSING, ErrorCategory.CONFIGURATION, "An active workspace Virtual Key is required.", action="Configure DEEPINTSHIELD_VIRTUAL_KEY and retry.", dashboard_path=_VIRTUAL_KEYS),
    _definition(ErrorCode.CLIENT_CLOSED, ErrorCategory.CLIENT, "The DeepIntShield client has already been closed."),
    _definition(ErrorCode.INVALID_ARGUMENT, ErrorCategory.VALIDATION, "A request argument is invalid."),
    _definition(ErrorCode.VALIDATION_ERROR, ErrorCategory.VALIDATION, "SDK input validation failed."),
    _definition(ErrorCode.OPTIONAL_DEPENDENCY_MISSING, ErrorCategory.CONFIGURATION, "A required optional dependency is not installed."),
    _definition(ErrorCode.TRANSPORT_ERROR, ErrorCategory.TRANSPORT, "The gateway could not be reached.", retryable=True),
    _definition(ErrorCode.TRANSPORT_TIMEOUT, ErrorCategory.TRANSPORT, "The gateway request timed out.", retryable=True),
    _definition(ErrorCode.HTTP_ERROR, ErrorCategory.TRANSPORT, "The gateway rejected the request."),
    _definition(ErrorCode.INVALID_RESPONSE, ErrorCategory.TRANSPORT, "The gateway returned an invalid response.", retryable=True),
    _definition(ErrorCode.AUTHENTICATION_FAILED, ErrorCategory.TRANSPORT, "Gateway authentication failed."),
    _definition(ErrorCode.PERMISSION_DENIED, ErrorCategory.TRANSPORT, "The requested gateway operation is not permitted."),
    _definition(ErrorCode.RESOURCE_NOT_FOUND, ErrorCategory.TRANSPORT, "The requested gateway resource was not found."),
    _definition(ErrorCode.CONFLICT, ErrorCategory.TRANSPORT, "The request conflicts with the current gateway state."),
    _definition(ErrorCode.RATE_LIMITED, ErrorCategory.TRANSPORT, "The gateway rate limit was exceeded.", retryable=True),
    _definition(ErrorCode.QUOTA_EXCEEDED, ErrorCategory.TRANSPORT, "The workspace quota was exceeded."),
    _definition(ErrorCode.FEATURE_LOCKED, ErrorCategory.CONFIGURATION, "This feature is not enabled for the workspace."),
    _definition(ErrorCode.SERVER_ERROR, ErrorCategory.TRANSPORT, "The gateway failed to process the request.", retryable=True),
    _definition(ErrorCode.INTERNAL_ERROR, ErrorCategory.TRANSPORT, "The gateway encountered an internal error.", retryable=True),
    _definition(ErrorCode.CHAT_REQUEST_FAILED, ErrorCategory.CHAT, "The chat completion request failed."),
    _definition(ErrorCode.CHAT_STREAM_INVALID_EVENT, ErrorCategory.CHAT, "The chat stream returned a malformed event."),
    _definition(ErrorCode.GUARDRAIL_EVALUATION_FAILED, ErrorCategory.GUARDRAIL, "The guardrail evaluation failed."),
    _definition(ErrorCode.GUARDRAIL_BLOCKED, ErrorCategory.GUARDRAIL, "A guardrail blocked the operation."),
    _definition(ErrorCode.RAG_EVALUATION_FAILED, ErrorCategory.RAG, "The RAG security evaluation failed."),
    _definition(ErrorCode.RAG_RETRIEVER_UNSUPPORTED, ErrorCategory.RAG, "The retriever exposes no supported retrieval method."),
    _definition(ErrorCode.RAG_EMBEDDER_UNSUPPORTED, ErrorCategory.RAG, "The embedder exposes no supported embedding method."),
    _definition(ErrorCode.AGENT_INVOCATION_INVALID, ErrorCategory.AGENT, "The tool invocation is incomplete or invalid."),
    _definition(ErrorCode.MCP_DEPENDENCY_MISSING, ErrorCategory.MCP, "A supported official MCP Python SDK is not available.", action="Install the DeepintShield MCP extra and retry."),
    _definition(ErrorCode.MCP_CONNECTION_FAILED, ErrorCategory.MCP, "The MCP server connection failed."),
    _definition(ErrorCode.MCP_PROTOCOL_ERROR, ErrorCategory.MCP, "The MCP server returned an invalid protocol response."),
    _definition(ErrorCode.MCP_EXECUTION_FAILED, ErrorCategory.MCP, "The MCP tool execution failed."),
    _definition(ErrorCode.MCP_DISCOVERY_FAILED, ErrorCategory.MCP, "MCP tool discovery failed."),
    _definition(ErrorCode.MCP_TOOL_NAME_INVALID, ErrorCategory.MCP, "The MCP tool name is not qualified with a server prefix."),
    _definition(ErrorCode.MCP_ARGUMENTS_INVALID, ErrorCategory.MCP, "The MCP tool arguments are not valid JSON."),
    _definition(ErrorCode.MCP_TOOL_AUTHORIZATION_DENIED, ErrorCategory.MCP, "Canonical Agentic authorization denied the MCP tool execution.", action="Review the recorded decision and least-privilege access in Agentic.", dashboard_path=_ACTIVITY),
    _definition(ErrorCode.MCP_TOOL_AUTHORIZATION_UNAVAILABLE, ErrorCategory.MCP, "Canonical Agentic authorization for the MCP tool is unavailable.", retryable=True, action="Restore the authorization service before retrying; execution remains blocked.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.MCP_TOOL_APPROVAL_REQUIRED, ErrorCategory.MCP, "The MCP tool execution is waiting for approval.", action="Approve or deny the pending MCP action in Agentic.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.PROVIDER_DEPENDENCY_MISSING, ErrorCategory.PROVIDER, "The selected provider dependency is not installed."),
    _definition(ErrorCode.PROVIDER_INITIALIZATION_FAILED, ErrorCategory.PROVIDER, "The selected provider could not be initialized."),
    _definition(ErrorCode.FRAMEWORK_BINDER_NOT_FOUND, ErrorCategory.FRAMEWORK, "The requested framework binder is not supported."),
    _definition(ErrorCode.FRAMEWORK_BINDER_ATTRIBUTE_MISSING, ErrorCategory.FRAMEWORK, "The framework does not provide the requested binder operation."),
    _definition(ErrorCode.AGENTIC_ERROR, ErrorCategory.AGENTIC, "The governed operation stopped safely.", action="Review the recorded Agentic decision before retrying.", dashboard_path=_ACTIVITY),
    _definition(ErrorCode.GOVERNANCE_CONFIGURATION_ERROR, ErrorCategory.AGENTIC, "Agent governance is not fully configured.", action="Complete the Agentic configuration and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.GUARDRAIL_DENIED, ErrorCategory.AGENTIC, "Agentic authorization denied this operation.", action="Review the decision and least-privilege access in Agentic.", dashboard_path=_ACTIVITY),
    _definition(ErrorCode.REQUIRE_APPROVAL, ErrorCategory.AGENTIC, "This operation is waiting for approval.", action="Approve or deny the pending action in Agentic.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.MASK_OBLIGATION_UNSUPPORTED, ErrorCategory.AGENTIC, "A required data-protection obligation could not be applied safely.", action="Review the decision obligations before retrying.", dashboard_path=_ACTIVITY),
    _definition(ErrorCode.GATEWAY_UNAVAILABLE, ErrorCategory.AGENTIC, "The Agentic gateway is unavailable.", retryable=True, action="Restore gateway connectivity and retry.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.INVALID_GATEWAY_RESPONSE, ErrorCategory.AGENTIC, "The Agentic gateway returned an invalid response.", retryable=True, action="Check gateway health and retry; the operation was blocked safely.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.AGENT_REGISTRATION_PENDING, ErrorCategory.AGENTIC, "This agent is waiting for registration approval.", action="Review and approve the captured agent registration in Agentic.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_NOT_REGISTERED, ErrorCategory.AGENTIC, "This agent is not registered.", action="Run discovery, then review and approve the captured registration.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_REGISTRATION_DENIED, ErrorCategory.AGENTIC, "This agent's registration was denied.", action="Review the registration decision before submitting corrected evidence.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_REGISTRATION_NOT_READY, ErrorCategory.AGENTIC, "This agent's registration review is incomplete.", action="Complete the verified identity, code, and access review.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_REGISTRATION_APPROVAL_REQUIRED, ErrorCategory.AGENTIC, "This agent requires registration approval.", action="Review and approve the captured registration in Agentic.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_REGISTRATION_QUOTA_EXCEEDED, ErrorCategory.AGENTIC, "This reporting key has reached its pending-registration limit.", action="Approve or deny pending registrations, then rerun discovery.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.AGENT_REGISTRATION_REVIEW_STALE, ErrorCategory.AGENTIC, "The agent registration changed during review.", action="Reload the latest registration evidence before deciding.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_REGISTRATION_REVIEW_CONFLICT, ErrorCategory.AGENTIC, "The agent registration changed during review.", action="Reload the latest registration evidence before deciding.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.AGENT_APPROVAL_PENDING, ErrorCategory.AGENTIC, "This operation is waiting for approval.", action="Approve or deny the pending action in Agentic.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.GUARDRAIL_APPROVAL_PENDING, ErrorCategory.AGENTIC, "This operation is waiting for approval.", action="Approve or deny the pending action in Agentic.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.APPROVAL_REQUIRED, ErrorCategory.AGENTIC, "This operation is waiting for approval.", action="Approve or deny the pending action in Agentic.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.APPROVAL_ACCESS_DENIED, ErrorCategory.AGENTIC, "You are not allowed to review this approval.", action="Ask a workspace administrator to grant approval-review access.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.APPROVAL_STORE_UNAVAILABLE, ErrorCategory.AGENTIC, "The durable approval service is unavailable.", retryable=True, action="Restore the approval service and retry; the operation remains blocked.", dashboard_path=_APPROVALS),
    _definition(ErrorCode.AUTHZ_STORE_UNAVAILABLE, ErrorCategory.AGENTIC, "The authorization service is unavailable.", retryable=True, action="Restore the authorization service and retry; the operation remains blocked.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.LEGACY_PDP_UNAVAILABLE, ErrorCategory.AGENTIC, "The configured policy decision service is unavailable.", retryable=True, action="Restore the policy decision service and retry.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.AGENT_BLUEPRINT_REVIEW_PENDING, ErrorCategory.AGENTIC, "This code blueprint is waiting for security review.", action="Review the exact code digest in Agentic.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.AGENT_BLUEPRINT_REVIEW_DENIED, ErrorCategory.AGENTIC, "This code blueprint was denied.", action="Review the findings and submit a corrected code digest.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_SCAN_UNAVAILABLE, ErrorCategory.AGENTIC, "The code blueprint could not be scanned safely.", retryable=True, action="Restore the scanner and rerun source-bearing discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_SCAN_REQUIRED, ErrorCategory.AGENTIC, "An approved code blueprint is required.", action="Publish and review a complete source-bearing blueprint.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_SCANNING_REQUIRED, ErrorCategory.AGENTIC, "Static code-blueprint scanning must remain enabled.", action="Keep static scanning enabled and review the blueprint policy.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_REGISTRATION_FAILED, ErrorCategory.AGENTIC, "The code blueprint could not be registered safely.", retryable=True, action="Restore the registration service and rerun source-bearing discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_COVERAGE_INCOMPLETE, ErrorCategory.AGENTIC, "The executable code evidence is incomplete.", action="Correct blueprint capture and rerun discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_MANIFEST_TOO_LARGE, ErrorCategory.AGENTIC, "The executable code blueprint exceeds the safe size limit.", action="Reduce the captured executable bundle and rerun discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_REMOTE_TOOL_UNVERIFIED, ErrorCategory.AGENTIC, "A remote tool could not be verified against an MCP connection.", action="Verify the tool inventory in MCP Connections and rerun discovery.", dashboard_path=_MCP),
    _definition(ErrorCode.BLUEPRINT_MCP_INVENTORY_UNAVAILABLE, ErrorCategory.AGENTIC, "The MCP tool inventory is unavailable.", retryable=True, action="Restore the MCP connection and rerun discovery.", dashboard_path=_MCP),
    _definition(ErrorCode.BLUEPRINT_MODEL_SCAN_PENDING, ErrorCategory.AGENTIC, "Code model analysis is still running.", retryable=True, action="Wait for the scan or review its status in Agentic.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_MODEL_SCAN_FAILED, ErrorCategory.AGENTIC, "Code model analysis failed safely.", retryable=True, action="Verify scanner settings and rerun source-bearing discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.BLUEPRINT_MODEL_UNAVAILABLE, ErrorCategory.AGENTIC, "The configured code-analysis model is unavailable.", retryable=True, action="Restore the model route and rerun source-bearing discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.INVALID_BLUEPRINT_MANIFEST, ErrorCategory.AGENTIC, "The code blueprint evidence is invalid.", action="Correct the source evidence and rerun discovery.", dashboard_path=_BLUEPRINTS),
    _definition(ErrorCode.CREDENTIAL_CONFIGURATION_ERROR, ErrorCategory.AGENTIC, "The workload identity is not fully configured.", action="Complete the agent identity configuration and retry.", dashboard_path=_IDENTITIES),
    _definition(ErrorCode.CREDENTIAL_PROVIDER_UNSUPPORTED, ErrorCategory.AGENTIC, "The workload identity provider is unsupported.", action="Select and validate a supported identity provider.", dashboard_path=_IDENTITIES),
    _definition(ErrorCode.CREDENTIAL_DEPENDENCY_MISSING, ErrorCategory.AGENTIC, "A workload identity dependency is missing.", action="Install the configured provider dependency and retry.", dashboard_path=_IDENTITIES),
    _definition(ErrorCode.CREDENTIAL_EXCHANGE_FAILED, ErrorCategory.AGENTIC, "The workload identity exchange failed.", retryable=True, action="Verify the selected identity provider and retry.", dashboard_path=_IDENTITIES),
    _definition(ErrorCode.FRAMEWORK_DEPENDENCY_MISSING, ErrorCategory.FRAMEWORK, "The selected agent framework dependency is missing.", action="Install the framework integration dependency and retry.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.FRAMEWORK_INTEGRATION_UNSUPPORTED, ErrorCategory.FRAMEWORK, "The installed agent framework version is not supported.", action="Install a supported framework version and retry.", dashboard_path=_OVERVIEW),
    _definition(ErrorCode.AGENT_DECISION_CONTEXT_MISSING, ErrorCategory.AGENTIC, "The Agentic decision context is incomplete.", action="Provide the required tool or decision context and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.PRINCIPAL_IDENTIFIER_MISSING, ErrorCategory.AGENTIC, "A stable principal identity is required.", action="Configure the requester identity and retry.", dashboard_path=_IDENTITIES),
    _definition(ErrorCode.REGISTRY_DISCOVERY_EMPTY, ErrorCategory.AGENTIC, "No discoverable agent topology was provided.", action="Provide an agent, workflow, or explicit manifest and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.REGISTRY_DISCOVERY_INVALID, ErrorCategory.AGENTIC, "The agent discovery payload is invalid.", action="Correct the discovery input and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.REGISTRY_DISCOVERY_PENDING, ErrorCategory.AGENTIC, "Agent discovery is already in progress.", retryable=True, action="Wait for discovery to complete and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.REGISTRY_DISCOVERY_REJECTED, ErrorCategory.AGENTIC, "The agent registry rejected the discovery report.", action="Review the registry response and correct the discovery evidence.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.REGISTRY_DISCOVERY_UNAVAILABLE, ErrorCategory.AGENTIC, "The agent discovery service is unavailable.", retryable=True, action="Restore the discovery service and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.REGISTRY_UNAVAILABLE, ErrorCategory.AGENTIC, "The agent registry is unavailable.", retryable=True, action="Restore the registry service and retry.", dashboard_path=_REGISTRATIONS),
    _definition(ErrorCode.WORKLOAD_PROOF_REQUIRED, ErrorCategory.AGENTIC, "Verified workload identity proof is required.", action="Configure workload identity proof and retry.", dashboard_path=_IDENTITIES),
)


_DENIAL_CODE_VALUES = (
    ErrorCode.AGENT_ACCESS_DENIED,
    ErrorCode.AUTHORIZATION_DENIED,
    ErrorCode.AUTHZ_ENGINE_ERROR,
    ErrorCode.CONTEXT_DENY,
    ErrorCode.NO_STORE,
    ErrorCode.OBO_ACTION_NOT_ALLOWED,
    ErrorCode.OBO_DELEGATION_REQUIRED,
    ErrorCode.OBO_NO_ACTS_FOR,
    ErrorCode.OBO_SCOPE_MISMATCH,
    ErrorCode.OBO_TOOL_NOT_ALLOWED,
    ErrorCode.OBO_USER_LACKS_PERM,
    ErrorCode.OBO_USER_MISMATCH,
)
_DENIAL_DEFINITION = next(
    item for item in _DEFINITIONS if item.code == ErrorCode.GUARDRAIL_DENIED.value
)
_DEFINITIONS += tuple(
    ErrorDefinition(
        code=code.value,
        category=ErrorCategory.AGENTIC,
        description=_DENIAL_DEFINITION.description,
        retryable=False,
        action=_DENIAL_DEFINITION.action,
        dashboard_path=_DENIAL_DEFINITION.dashboard_path,
    )
    for code in _DENIAL_CODE_VALUES
)

_catalog = {definition.code: definition for definition in _DEFINITIONS}
if len(_catalog) != len(_DEFINITIONS):  # pragma: no cover - import-time invariant
    raise RuntimeError("duplicate DeepintShield error code")
if set(_catalog) != {code.value for code in ErrorCode}:  # pragma: no cover
    raise RuntimeError("DeepintShield error catalogue and ErrorCode are out of sync")

# MappingProxyType prevents application code from mutating process-wide error
# semantics.  Individual definitions are frozen dataclasses as well.
ERROR_CATALOG: Mapping[str, ErrorDefinition] = MappingProxyType(_catalog)


def _code_value(code: str | ErrorCode | object) -> str:
    if isinstance(code, ErrorCode):
        return code.value
    if not isinstance(code, str):
        return ""
    return code.strip().lower().replace("-", "_")


def get_error_definition(code: str | ErrorCode) -> ErrorDefinition | None:
    """Return trusted metadata for ``code``, or ``None`` when it is unknown."""
    return ERROR_CATALOG.get(_code_value(code))


def iter_error_definitions(
    category: ErrorCategory | str | None = None,
) -> tuple[ErrorDefinition, ...]:
    """Return the deterministic catalogue sequence, optionally by category."""
    if category is None:
        return _DEFINITIONS
    try:
        selected = category if isinstance(category, ErrorCategory) else ErrorCategory(category)
    except (TypeError, ValueError):
        return ()
    return tuple(item for item in _DEFINITIONS if item.category is selected)


_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_ERROR_CODE_MARKER = "_deepintshield_error_code"
_SERVER_CODE_ALIASES = MappingProxyType(
    {
        "unauthenticated": ErrorCode.AUTHENTICATION_FAILED.value,
        "authentication_error": ErrorCode.AUTHENTICATION_FAILED.value,
        "unauthorized": ErrorCode.AUTHENTICATION_FAILED.value,
        "forbidden": ErrorCode.PERMISSION_DENIED.value,
        "not_found": ErrorCode.RESOURCE_NOT_FOUND.value,
        "rate_limit_exceeded": ErrorCode.RATE_LIMITED.value,
        "too_many_requests": ErrorCode.RATE_LIMITED.value,
    }
)
_STATUS_CODES = MappingProxyType(
    {
        401: ErrorCode.AUTHENTICATION_FAILED.value,
        402: ErrorCode.FEATURE_LOCKED.value,
        403: ErrorCode.PERMISSION_DENIED.value,
        404: ErrorCode.RESOURCE_NOT_FOUND.value,
        408: ErrorCode.TRANSPORT_TIMEOUT.value,
        409: ErrorCode.CONFLICT.value,
        429: ErrorCode.RATE_LIMITED.value,
    }
)


def _normalize_catalog_code(value: object) -> str:
    candidate = _code_value(value)
    if not _ERROR_CODE.fullmatch(candidate):
        return ""
    candidate = _SERVER_CODE_ALIASES.get(candidate, candidate)
    return candidate if candidate in ERROR_CATALOG else ""


def get_exception_error_code(error: BaseException) -> str:
    """Return the bounded code carried by an SDK or Agentic boundary error.

    Agentic framework boundaries deliberately translate internal exceptions to
    standard Python exception types.  This helper is the public way to read
    their marker without depending on a private attribute name.  Unknown
    future Agentic codes are retained only when they satisfy the bounded code
    grammar; arbitrary exception prose is never returned.
    """
    marked = getattr(error, _ERROR_CODE_MARKER, "")
    raw = marked or getattr(error, "code", "")
    candidate = _code_value(raw)
    if not _ERROR_CODE.fullmatch(candidate):
        return ""
    candidate = _SERVER_CODE_ALIASES.get(candidate, candidate)
    # Marked Agentic/native exceptions and typed SDK exceptions may carry a
    # bounded future gateway code. Do not treat an arbitrary third-party
    # ``.code`` string as a DeepintShield code merely because it happens to
    # match the grammar or a current catalogue entry.
    if marked or isinstance(error, DeepintShieldError):
        return candidate
    return ""


def _response_error_code(payload: Mapping[str, Any] | None) -> tuple[str, str]:
    """Return ``(recognized_code, bounded_raw_code)`` from public code fields."""
    body = payload or {}
    error = body.get("error")
    candidates: Iterable[object] = (
        error.get("code") if isinstance(error, Mapping) else error,
        body.get("code"),
        body.get("error_code"),
    )
    bounded_raw = ""
    for value in candidates:
        candidate = _code_value(value)
        if _ERROR_CODE.fullmatch(candidate):
            bounded_raw = bounded_raw or candidate
            normalized = _normalize_catalog_code(candidate)
            if normalized:
                return normalized, candidate
    return "", bounded_raw


def _annotate_error(
    error: BaseException,
    code: str | ErrorCode,
    *,
    details: Mapping[str, Any] | None = None,
) -> BaseException:
    """Attach the public error contract while preserving a native error type."""
    normalized = _normalize_catalog_code(code) or ErrorCode.SDK_ERROR.value
    definition = ERROR_CATALOG[normalized]
    setattr(error, _ERROR_CODE_MARKER, normalized)
    setattr(error, "code", normalized)
    setattr(error, "description", definition.description)
    setattr(error, "retryable", definition.retryable)
    setattr(error, "details", dict(details or {}))
    return error


def _dependency_error(
    message: str,
    *,
    code: ErrorCode,
    component: str,
) -> ImportError:
    """Build a coded ``ImportError`` while preserving legacy catch behavior."""
    error = ImportError(message)
    _annotate_error(error, code, details={"component": component})
    return error


class DeepintShieldError(Exception):
    """Base SDK exception with a stable code and trusted description.

    The original constructor remains compatible.  ``message`` and ``payload``
    retain gateway diagnostics, while ``code``/``description`` are safe fields
    for programmatic branching and user-facing summaries.
    """

    default_code = ErrorCode.SDK_ERROR.value

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        payload: dict | None = None,
        *,
        code: str | ErrorCode | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
        normalized = _normalize_catalog_code(code or self.default_code)
        self.code = normalized or ErrorCode.SDK_ERROR.value
        definition = ERROR_CATALOG[self.code]
        self.description = definition.description
        self.retryable = definition.retryable
        self.details = dict(details or {})

    @classmethod
    def from_response(
        cls,
        status_code: int,
        payload: dict | None,
        *,
        code: str | ErrorCode | None = None,
        fallback_code: str | ErrorCode | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> "DeepintShieldError":
        body = payload or {}
        error = body.get("error", {})
        # ``error`` may be nested ({"message": ...}) or a plain string.  The
        # latter is used by feature-gate responses, and must never crash parsing.
        error_message = error.get("message") if isinstance(error, Mapping) else error
        message = (
            error_message
            or body.get("message")
            or f"DeepintShield request failed with status {status_code}"
        )
        response_code, raw_response_code = _response_error_code(body)
        resolved_code = (
            _normalize_catalog_code(code)
            or response_code
            or _STATUS_CODES.get(status_code, "")
            or _normalize_catalog_code(fallback_code)
            or (ErrorCode.SERVER_ERROR.value if status_code >= 500 else ErrorCode.HTTP_ERROR.value)
        )
        error_details = dict(details or {})
        error_details.setdefault("status_code", status_code)
        if raw_response_code:
            error_details.setdefault("response_code", raw_response_code)
        # Instantiate through the original three-argument constructor contract
        # before attaching the new metadata. This keeps ``from_response``
        # compatible with user subclasses whose ``__init__`` predates the
        # keyword-only ``code`` and ``details`` additions.
        instance = cls(
            message=str(message),
            status_code=status_code,
            payload=body,
        )
        instance.code = resolved_code
        definition = ERROR_CATALOG[resolved_code]
        instance.description = definition.description
        instance.retryable = definition.retryable
        instance.details = error_details
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Return structured metadata without raw gateway prose or payloads.

        Both the raw payload and diagnostic ``message`` are omitted because a
        gateway may supply untrusted or sensitive prose.  Use ``description``
        for display. ``details`` is caller/request diagnostic data and must be
        reviewed before display or logging; only catalogue-derived fields are
        inherently trusted.
        """
        return {
            "code": self.code,
            "description": self.description,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


class DeepintShieldBlockedError(DeepintShieldError):
    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        decision: str | None = None,
        reason: str | None = None,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
        code: str | ErrorCode = ErrorCode.GUARDRAIL_BLOCKED,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        structured = dict(details or {})
        if stage is not None:
            structured.setdefault("stage", stage)
        if decision is not None:
            structured.setdefault("decision", decision)
        super().__init__(
            message=message,
            status_code=status_code,
            payload=payload,
            code=code,
            details=structured,
        )
        self.stage = stage
        self.decision = decision
        self.reason = reason


__all__ = [
    "ERROR_CATALOG",
    "ErrorCategory",
    "ErrorCode",
    "ErrorDefinition",
    "DeepintShieldBlockedError",
    "DeepintShieldError",
    "get_error_definition",
    "get_exception_error_code",
    "iter_error_definitions",
]
