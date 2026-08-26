from __future__ import annotations

import json
import os
import threading
from typing import Any, Iterable, Literal, Mapping, overload

import httpx

from .config import DEFAULT_BASE_URL, ShieldConfig, _normalize_base_url
from .errors import (
    DeepintShieldBlockedError,
    DeepintShieldError,
    ErrorCode,
    _annotate_error,
)
from .streaming import ChatCompletionStream, _bounded_response_payload
from .types import GuardrailResult, RetrievedChunk, ToolInvocation


class DeepintShield:
    """
    Unified DeepintShield client.

    >>> shield = DeepintShield(virtual_key="sk-ds-your-virtual-key")
    >>> openai = shield.openai()               # native openai.OpenAI pointed at the gateway
    >>> resp = shield.chat(model="gpt-4o-mini", messages=[...])
    >>> shield.rag.filter(query="...", chunks=[...])
    >>> shield.agent.evaluate_tool(name="read_file", args={"path": "/tmp"})

    Pass ``base_url`` to target a self-hosted or staging gateway. The default
    is ``https://app.deepintshield.com``; the environment variable
    ``DEEPINTSHIELD_BASE_URL`` is honoured by ``DeepintShield.from_env()``.
    """

    def __init__(
        self,
        virtual_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        default_headers: Mapping[str, str] | None = None,
        app_name: str = "deepintshield",
        agent_name: str = "",
        requester: str = "sdk-user",
        requester_role: str = "member",
        persist: bool = True,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.virtual_key = (virtual_key or "").strip() or None
        self.timeout = timeout
        self.default_headers = dict(default_headers or {})
        self.app_name = app_name
        self.agent_name = agent_name
        self.requester = requester
        self.requester_role = requester_role
        self.persist = persist
        self._deepintshield_closed = False
        self._client = httpx.Client(timeout=timeout)

        from .rag import RAGSurface
        from .agent import AgentSurface
        from .providers import ProviderRegistry
        from .mcp import MCPClient

        self.rag = RAGSurface(self)
        self.agent = AgentSurface(self)
        self.providers = ProviderRegistry(self)
        self.mcp = MCPClient(self)
        # Agentic (PDP) surface is built lazily on first access so that
        # constructing a DeepintShield never performs network I/O and never
        # imports pydantic/azure unless agentic features are actually used.
        self._agentic = None
        self._agentic_lock = threading.Lock()
        # Install non-bypassable enforcement guards for any agent framework that
        # is ALREADY imported (langgraph, crewai, litellm, …) so a compiled graph
        # / built tool can't run ungoverned. Lazy engine ⇒ this does NOT build the
        # agentic surface; it only patches frameworks the app actually uses. Safe
        # no-op when none are present; the import guard arms supported frameworks
        # that are imported later.
        try:
            self._install_agentic_guards()
        except Exception:
            # Constructor failure must not leave a half-initialized client in
            # the weak ownership broker or leak its connection pool.
            self.close()
            raise

    # ─────────────────────────── constructors / context ──────────────────────

    @classmethod
    def from_env(cls) -> "DeepintShield":
        cfg = ShieldConfig.from_env()
        return cls(
            virtual_key=cfg.virtual_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            app_name=cfg.app_name,
            agent_name=cfg.agent_name,
            requester=cfg.requester,
            requester_role=cfg.requester_role,
            persist=cfg.persist,
        )

    @classmethod
    def from_config(cls, config: ShieldConfig) -> "DeepintShield":
        return cls(
            virtual_key=config.virtual_key,
            base_url=config.base_url,
            timeout=config.timeout,
            default_headers=config.default_headers,
            app_name=config.app_name,
            agent_name=config.agent_name,
            requester=config.requester,
            requester_role=config.requester_role,
            persist=config.persist,
        )

    def __enter__(self) -> "DeepintShield":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._deepintshield_closed:
            return
        self._deepintshield_closed = True
        try:
            from .agentic.enforcement import unregister_client

            unregister_client(self)
        finally:
            self._client.close()

    # ─────────────────────────── keys and headers ────────────────────────────

    def virtual_key_or_raise(self) -> str:
        if not self.virtual_key:
            raise DeepintShieldError(
                "DEEPINTSHIELD_VIRTUAL_KEY is required",
                code=ErrorCode.VIRTUAL_KEY_MISSING,
            )
        return self.virtual_key

    def api_key(self) -> str:
        return self.virtual_key_or_raise()

    def headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        out = {"content-type": "application/json", **self.default_headers}
        if self.virtual_key:
            out["x-deepintshield-vk"] = self.virtual_key
        if extra:
            out.update(dict(extra))
        return out

    # ─────────────────────────── provider endpoints ──────────────────────────

    def endpoint(self, provider: str) -> str:
        return f"{self.base_url}/{provider.strip('/')}"

    def openai_base_url(self) -> str:
        return self.endpoint("openai")

    def anthropic_base_url(self) -> str:
        return self.endpoint("anthropic")

    def bedrock_endpoint_url(self) -> str:
        return self.endpoint("bedrock")

    def genai_base_url(self) -> str:
        return self.endpoint("genai")

    def langchain_base_url(self) -> str:
        return self.endpoint("langchain")

    def litellm_base_url(self) -> str:
        return self.endpoint("litellm")

    def pydanticai_base_url(self) -> str:
        return self.endpoint("pydanticai")

    def openai_passthrough_base_url(self) -> str:
        return f"{self.base_url}/openai_passthrough/v1"

    def anthropic_passthrough_base_url(self) -> str:
        return f"{self.base_url}/anthropic_passthrough"

    def genai_passthrough_base_url(self) -> str:
        return f"{self.base_url}/genai_passthrough"

    # ─────────────────────────── provider shortcuts ──────────────────────────

    def openai(self, *, passthrough: bool = False, **kwargs: Any):
        from .providers.openai import build_client
        return build_client(self, passthrough=passthrough, **kwargs)

    def anthropic(self, *, passthrough: bool = False, **kwargs: Any):
        from .providers.anthropic import build_client
        return build_client(self, passthrough=passthrough, **kwargs)

    def bedrock(self, **kwargs: Any):
        from .providers.bedrock import build_client
        return build_client(self, **kwargs)

    def genai(self, *, passthrough: bool = False, **kwargs: Any):
        from .providers.genai import build_client
        return build_client(self, passthrough=passthrough, **kwargs)

    def genai_cached(
        self,
        *,
        passthrough: bool = False,
        ttl_seconds: int | None = None,
        min_prefix_tokens: int | None = None,
        **kwargs: Any,
    ):
        """Return a Gemini client that auto-manages Google's ``cachedContents``
        resource lifecycle so repeat calls with the same static prefix reuse
        the cache (≈75% input-token discount, plus a small storage fee).

        Drop-in for ``shield.genai()``. The first call with a new prefix runs
        at normal latency and asynchronously creates the cache resource; the
        next call (and every subsequent one within the TTL window) uses the
        cached prefix. The workspace switch in Caching settings governs
        whether the resource reference reaches Google.
        """
        from .providers.genai import build_cached_client
        return build_cached_client(
            self,
            passthrough=passthrough,
            ttl_seconds=ttl_seconds,
            min_prefix_tokens=min_prefix_tokens,
            **kwargs,
        )

    def langchain(self, model: str = "gpt-4o-mini", **kwargs: Any):
        from .providers.langchain import build_client
        return build_client(self, model=model, **kwargs)

    def langgraph(self):
        from .providers.langgraph import LangGraphShield
        return LangGraphShield(self)

    def litellm(self):
        from .providers.litellm import LiteLLMShield
        return LiteLLMShield(self)

    def pydanticai(self, model: str = "gpt-4o-mini", **kwargs: Any):
        from .providers.pydanticai import build_agent
        return build_agent(self, model=model, **kwargs)

    # ─────────────────────────── agentic (PDP) surface ───────────────────────

    @property
    def agentic(self):
        """``shield.agentic`` - tool-call enforcement (decide / @tool / per-
        framework wrappers). All Entra/identity/policy detail is auto-
        discovered from the gateway; the user passes only VK + base_url."""
        if self._agentic is None:
            with self._agentic_lock:
                if self._agentic is None:
                    from .agentic.surface import AgenticSurface

                    self._agentic = AgenticSurface(self)
        return self._agentic

    def _install_agentic_guards(self) -> None:
        """Install non-bypassable framework guards (compile/tool patches). Lazy
        engine via ``self.agentic.engine`` so this never builds the surface here.
        Imported supported frameworks must be instrumented successfully."""
        from .agentic.enforcement import install_all

        install_all(client=self)

    def _agent_token(self) -> str | None:
        """Best-effort agent identity token for transparent (L1) traffic.
        Never raises - identity is a strengthening signal, not required."""
        try:
            return self.agentic.engine.agent_token()
        except Exception:
            return None

    # ───────────────────── transparent transport (L1) ────────────────────────

    def connection(
        self,
        *,
        provider: str = "openai",
        identity: bool = False,
        extra: Mapping[str, str] | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Return ``(base_url, headers)`` to point any native framework client
        at the gateway. ``provider`` picks the route ("openai", "anthropic",
        "genai", …); ``identity=True`` also attaches an agent token."""
        from .transport import connection
        return connection(self, provider=provider, identity=identity, extra=extra)

    def create_headers(
        self,
        *,
        provider: str = "openai",
        identity: bool = False,
        extra: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return just the gateway header set (VK + attribution [+ identity]).

        Parity helper for the Portkey-style ``createHeaders`` ergonomic - drop
        it into any client's ``default_headers``/``extra_headers`` and point
        ``base_url`` at ``shield.endpoint(provider)``::

            OpenAI(base_url=shield.endpoint("openai"),
                   api_key=shield.api_key(),
                   default_headers=shield.create_headers())
        """
        from .transport import connection_headers
        return connection_headers(self, identity=identity, extra=extra)

    def http_client(
        self,
        *,
        provider: str = "openai",
        identity: bool = False,
        base_url: str | None = None,
        extra: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ):
        """Return an ``httpx.Client`` pre-wired to the gateway (base URL +
        VK + attribution headers). Hand it to any SDK that accepts one."""
        from .transport import http_client
        return http_client(
            self, provider=provider, identity=identity, base_url=base_url, extra=extra, timeout=timeout
        )

    def bind(self, framework: str):
        """Return a :class:`FrameworkBinder` for ``framework`` (e.g.
        ``shield.bind("langgraph").model("gpt-4o-mini")``)."""
        from .frameworks import get_binder
        return get_binder(self, framework)

    def crewai(self):
        return self.bind("crewai")

    def openai_agents(self):
        return self.bind("openai_agents")

    def llamaindex(self):
        return self.bind("llamaindex")

    def autogen(self):
        return self.bind("autogen")

    # ─────────────────────────── HTTP + guardrails ───────────────────────────

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        error_code: str | ErrorCode | None = None,
        require_object: bool = False,
    ) -> Any:
        request_details = {"method": method.upper(), "path": path}
        if self._deepintshield_closed:
            raise DeepintShieldError(
                "DeepintShield client is closed",
                code=ErrorCode.CLIENT_CLOSED,
                details=request_details,
            )
        try:
            response = self._client.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=self.headers(extra_headers),
                json=json_body,
            )
        except httpx.TimeoutException as exc:
            raise _annotate_error(
                exc,
                ErrorCode.TRANSPORT_TIMEOUT,
                details=request_details,
            )
        except httpx.HTTPError as exc:
            raise _annotate_error(
                exc,
                ErrorCode.TRANSPORT_ERROR,
                details=request_details,
            )
        invalid_json = False
        try:
            payload = response.json()
        except (ValueError, RecursionError):
            payload = {"raw": response.text}
            invalid_json = True
        if response.status_code < 400 and require_object and invalid_json:
            raise DeepintShieldError(
                "DeepintShield gateway returned a non-JSON response",
                status_code=response.status_code,
                payload={"raw": response.text},
                code=ErrorCode.INVALID_RESPONSE,
                details=request_details,
            ) from None
        if response.status_code >= 400:
            raise DeepintShieldError.from_response(
                response.status_code,
                payload if isinstance(payload, dict) else {"data": payload},
                fallback_code=error_code,
                details=request_details,
            )
        if require_object and not isinstance(payload, dict):
            raise DeepintShieldError(
                "DeepintShield gateway returned a non-object JSON response",
                status_code=response.status_code,
                payload={"data": payload},
                code=ErrorCode.INVALID_RESPONSE,
                details=request_details,
            )
        return payload

    @overload
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: Literal[False] = False,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @overload
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: Literal[True],
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionStream: ...

    @overload
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | ChatCompletionStream: ...

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | ChatCompletionStream:
        """Unified chat completion via the gateway OpenAI-compatible endpoint.

        ``stream=False`` returns the decoded response object. ``stream=True``
        returns a lazy :class:`~deepintshield.streaming.ChatCompletionStream`;
        use it as a context manager or close it when stopping early.
        """
        body = {"model": model, "messages": messages, "stream": stream, **kwargs}
        if stream:
            return self._open_chat_stream(body, extra_headers=extra_headers)
        return self.request(
            "POST",
            "/v1/chat/completions",
            json_body=body,
            extra_headers=extra_headers,
            error_code=ErrorCode.CHAT_REQUEST_FAILED,
            require_object=True,
        )

    def _open_chat_stream(
        self,
        body: Mapping[str, Any],
        *,
        extra_headers: Mapping[str, str] | None,
    ) -> ChatCompletionStream:
        path = "/v1/chat/completions"
        request_details: dict[str, Any] = {"method": "POST", "path": path}
        if self._deepintshield_closed:
            raise DeepintShieldError(
                "DeepintShield client is closed",
                code=ErrorCode.CLIENT_CLOSED,
                details=request_details,
            )
        try:
            request = self._client.build_request(
                "POST",
                f"{self.base_url}{path}",
                headers=self.headers(extra_headers),
                json=body,
            )
            response = self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise _annotate_error(
                exc,
                ErrorCode.TRANSPORT_TIMEOUT,
                details=request_details,
            )
        except httpx.HTTPError as exc:
            raise _annotate_error(
                exc,
                ErrorCode.TRANSPORT_ERROR,
                details=request_details,
            )

        if response.status_code >= 400:
            payload, truncated = _bounded_response_payload(response)
            if truncated:
                request_details["response_truncated"] = True
            raise DeepintShieldError.from_response(
                response.status_code,
                payload,
                fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
                details=request_details,
            )

        content_type = response.headers.get("content-type", "").lower()
        media_type = content_type.partition(";")[0].strip()
        if media_type != "text/event-stream":
            payload, truncated = _bounded_response_payload(response)
            if truncated:
                request_details["response_truncated"] = True
            request_details["content_type"] = content_type
            if isinstance(payload, dict) and any(
                payload.get(key) is not None
                for key in ("error", "code", "error_code")
            ):
                raise DeepintShieldError.from_response(
                    response.status_code,
                    payload,
                    fallback_code=ErrorCode.CHAT_REQUEST_FAILED,
                    details=request_details,
                )
            raise DeepintShieldError(
                "Chat streaming response is not Server-Sent Events",
                status_code=response.status_code,
                payload=payload,
                code=ErrorCode.CHAT_STREAM_INVALID_EVENT,
                details=request_details,
            )

        if self._deepintshield_closed:
            response.close()
            raise DeepintShieldError(
                "DeepintShield client is closed",
                code=ErrorCode.CLIENT_CLOSED,
                details=request_details,
            )
        return ChatCompletionStream(response, self, details=request_details)

    def evaluate_guardrail(
        self,
        *,
        stage: str,
        actor_type: str = "sdk_user",
        actor_id: str | None = None,
        actor_role: str | None = None,
        actor_customer_id: str | None = None,
        actor_team_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        input: str | None = None,
        output: str | None = None,
        tool_input: str | None = None,
        server_label: str | None = None,
        tool_name: str | None = None,
        action_class: str | None = None,
        domains: list[str] | None = None,
        app_name: str | None = None,
        agent_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        persist: bool | None = None,
    ) -> GuardrailResult:
        body: dict[str, Any] = {
            "stage": stage,
            "actor_type": actor_type,
            "actor_id": actor_id or self.requester,
            "actor_role": actor_role or self.requester_role,
            "model": model,
            "provider": provider,
            "input": input,
            "output": output,
            "tool_input": tool_input,
            "server_label": server_label,
            "tool_name": tool_name,
            "action_class": action_class,
            "domains": domains or [],
            "app_name": app_name or self.app_name,
            "agent_name": agent_name or self.agent_name,
            "persist": self.persist if persist is None else persist,
        }
        if actor_customer_id:
            body["actor_customer_id"] = actor_customer_id
        if actor_team_id:
            body["actor_team_id"] = actor_team_id
        if metadata:
            body["metadata"] = dict(metadata)
        payload = self.request(
            "POST",
            "/api/guardrails/evaluate",
            json_body=body,
            error_code=ErrorCode.GUARDRAIL_EVALUATION_FAILED,
            require_object=True,
        )
        return GuardrailResult.from_response(stage, payload)

    def guard(
        self,
        *,
        stage: str,
        raise_on_block: bool = True,
        **kwargs: Any,
    ) -> GuardrailResult:
        """Evaluate a guardrail and optionally raise on block."""
        result = self.evaluate_guardrail(stage=stage, **kwargs)
        if raise_on_block and result.blocked:
            raise DeepintShieldBlockedError(
                f"DeepintShield blocked at stage={stage}: {result.reason or result.decision}",
                stage=stage,
                decision=result.decision,
                reason=result.reason,
                payload=result.raw,
            )
        return result
