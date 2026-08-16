"""LangChain / LangGraph adapter for MCP tools.

Wraps each MCP tool as a LangChain ``BaseTool`` whose ``_run`` posts to
DeepintShield's MCP executor. Drops straight into LangChain agents and
LangGraph ``ToolNode``s.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from ...errors import ErrorCode, _dependency_error
from ..tool import Tool

if TYPE_CHECKING:
    from ..client import MCPClient


def to_langchain(client: "MCPClient", tools: Iterable[Tool]) -> list[Any]:
    try:
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel, ConfigDict, Field, create_model
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "to_langchain requires `langchain-core` and `pydantic`. "
            "Install with: pip install 'deepintshield[langchain]'.",
            code=ErrorCode.FRAMEWORK_DEPENDENCY_MISSING,
            component="langchain_mcp",
        ) from exc

    out: list[Any] = []
    for tool in tools:
        args_model = _schema_to_model(tool, create_model, BaseModel, Field, ConfigDict)
        out.append(_make_tool(client, tool, BaseTool, args_model))
    return out


def _make_tool(
    client: "MCPClient",
    tool: Tool,
    base_tool_cls: Any,
    args_model: Any,
) -> Any:
    """Construct a single MCPTool subclass instance bound to ``tool``."""

    class _MCPTool(base_tool_cls):  # type: ignore[misc, valid-type]
        # LangChain ≥0.3 uses Pydantic v2; field defaults below are
        # set on the subclass itself so each instance is bound to one MCP tool.
        name: str = tool.qualified_name
        description: str = tool.description or f"MCP tool {tool.qualified_name}"
        args_schema: Any = args_model  # may be None when the schema is missing

        def _run(self_inner, *args: Any, **kwargs: Any) -> str:
            # When ``args_schema`` is None, LangChain passes the raw input
            # positionally (string) or as the only positional. Coerce that
            # into kwargs the MCP executor expects.
            if args and not kwargs:
                first = args[0]
                if isinstance(first, dict):
                    kwargs = dict(first)
                elif isinstance(first, str):
                    import json as _json
                    try:
                        parsed = _json.loads(first)
                        if isinstance(parsed, dict):
                            kwargs = parsed
                    except Exception:
                        pass
            result = client.call(server=tool.server, tool=tool.name, arguments=kwargs)
            return result.text or "(empty result)"

        async def _arun(self_inner, *args: Any, **kwargs: Any) -> str:
            return self_inner._run(*args, **kwargs)

    _MCPTool.__name__ = f"MCPTool_{tool.server}_{tool.name}"
    return _MCPTool()


# JSON Schema → Pydantic v2 model. Best-effort: handles flat objects with
# primitive properties, which covers the vast majority of MCP tool schemas in
# the wild. Falls back to a permissive ``Any`` model when the schema is
# missing or non-trivially nested.
def _schema_to_model(
    tool: Tool,
    create_model: Any,
    base_model_cls: Any,
    field_factory: Any,
    config_dict: Any,
) -> Any:
    """Convert a JSON-Schema object into a Pydantic v2 model, or return
    ``None`` to signal "no schema; pass kwargs through raw" to the caller."""
    schema = tool.schema or {}
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return None

    properties: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        py_type = _json_type_to_python(prop)
        default: Any
        if name in required:
            default = field_factory(..., description=prop.get("description"))
        else:
            default = field_factory(default=None, description=prop.get("description"))
        fields[name] = (py_type, default)

    if not fields:
        return None

    return create_model(
        f"MCPArgs_{tool.server}_{tool.name}",
        __base__=base_model_cls,
        **fields,
    )


def _json_type_to_python(prop: dict[str, Any]) -> Any:
    """Map a single JSON Schema property to a Python type for Pydantic.

    Lossy on purpose - frameworks vary in how they consume nested schemas.
    Falls back to ``Any`` when in doubt.
    """
    from typing import Any as _Any
    from typing import List, Optional

    t = prop.get("type")
    if isinstance(t, list):
        # union - pick the first non-null type as a hint
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else "string"
    if t == "string":
        return Optional[str]
    if t == "integer":
        return Optional[int]
    if t == "number":
        return Optional[float]
    if t == "boolean":
        return Optional[bool]
    if t == "array":
        return Optional[List[_Any]]
    if t == "object":
        return Optional[dict]
    return Optional[_Any]
