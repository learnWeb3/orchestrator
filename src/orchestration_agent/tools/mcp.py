"""Adapts tools discovered from an MCP server into `BaseTool` instances.

Wraps `fastmcp.Client` so remote or local MCP tools can be handed straight to
`Agent(tools=...)` without hand-writing a `BaseTool` subclass per tool. This
module is opt-in: `fastmcp` is only imported here, not from `tools/__init__.py`,
so the base package works without it installed. Install with the `mcp` extra
(`pip install orchestration-agent[mcp]`).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Type

from fastmcp import Client
from fastmcp.exceptions import McpError
from fastmcp.exceptions import ToolError as FastMCPToolError
from pydantic import BaseModel, ConfigDict, create_model

from .base import BaseTool

_JSON_SCHEMA_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class MCPConnectionError(Exception):
    """Raised when the MCP client cannot connect, or is used outside a connected session."""


def _pascal_case(value: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", value)
    return "".join(part.capitalize() for part in parts if part) or "Tool"


def _json_schema_to_model(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Build a lenient Pydantic model from an MCP tool's raw JSON input schema.

    Validation is intentionally loose (best-effort field types, extra keys
    allowed) -- the model's job is to let `Agent.invoke_tool` construct and
    pass through arguments, not to re-implement JSON Schema validation.
    `model_json_schema()` is overridden to return the original schema verbatim
    so providers (e.g. `OpenAIProvider`) see the server's real schema rather
    than a lossy, regenerated one.
    """
    properties: Dict[str, Any] = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    fields: Dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        raw_type = field_schema.get("type") if isinstance(field_schema, dict) else None
        field_type = _JSON_SCHEMA_TYPE_MAP.get(raw_type, Any) if isinstance(raw_type, str) else Any
        if field_name in required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (Optional[field_type], None)

    class_name = _pascal_case(name) + "Input"
    base_model = create_model(
        class_name,
        __config__=ConfigDict(extra="allow"),
        **fields,
    )

    class _Model(base_model):  # type: ignore[misc, valid-type]
        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            return schema

    _Model.__name__ = class_name
    _Model.__qualname__ = class_name
    return _Model


class MCPTool(BaseTool):
    """A single MCP server tool, adapted to the `BaseTool` interface.

    Instances are created by `MCPToolProvider.discover_tools`; there is no
    reason to construct one directly.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Type[BaseModel],
        provider: "MCPToolProvider",
        remote_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._provider = provider
        self._remote_name = remote_name or name

    async def execute(self, input: BaseModel) -> Any:
        """Call the underlying MCP tool with the validated arguments.

        Raises on failure; `Agent.invoke_tool` catches and formats the error
        using `error_schema`.
        """
        arguments = input.model_dump(exclude_none=True)
        try:
            result = await self._provider.call_tool(self._remote_name, arguments)
        except MCPConnectionError:
            raise
        except (FastMCPToolError, McpError) as exc:
            raise RuntimeError(f"MCP tool '{self.name}' failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Unexpected error calling MCP tool '{self.name}': {exc}"
            ) from exc

        if result.data is not None:
            return result.data
        if result.structured_content is not None:
            return result.structured_content
        return "\n".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )


class MCPToolProvider:
    """Async lifecycle wrapper around `fastmcp.Client` that discovers an MCP
    server's tools and adapts them into `BaseTool` instances.

    Usable as a short-lived async context manager::

        async with MCPToolProvider("https://example.com/mcp") as provider:
            agent = Agent(provider=llm_provider, tools=provider.tools)
            ...

    or with an explicit connect/close lifecycle for a long-running agent
    process::

        mcp_provider = MCPToolProvider("https://example.com/mcp")
        await mcp_provider.connect()
        agent = Agent(provider=llm_provider, tools=mcp_provider.tools)
        ...
        await mcp_provider.aclose()
    """

    def __init__(
        self,
        transport: Any,
        *,
        client_kwargs: Optional[Dict[str, Any]] = None,
        name_prefix: Optional[str] = None,
    ) -> None:
        """
        Args:
            transport: A URL string, filesystem path, in-memory `FastMCP`
                server, or `fastmcp.client.transports.ClientTransport` --
                anything `fastmcp.Client` itself accepts.
            client_kwargs: Extra keyword arguments forwarded to `Client`
                (auth, headers, timeouts, ...).
            name_prefix: Optional namespace prepended to each discovered
                tool's name (`"{prefix}.{tool_name}"`), to avoid collisions
                when wiring multiple MCP servers into one `Agent`.
        """
        self._client = Client(transport, **(client_kwargs or {}))
        self._name_prefix = name_prefix
        self._tools: Dict[str, MCPTool] = {}
        self._connected = False
        self._lock = asyncio.Lock()

    @property
    def tools(self) -> List[BaseTool]:
        """Currently discovered tools. Empty until `connect()` has run."""
        return list(self._tools.values())

    async def connect(self) -> None:
        """Open the MCP connection and discover tools. Idempotent."""
        async with self._lock:
            if self._connected:
                return
            try:
                await self._client.__aenter__()
            except Exception as exc:  # noqa: BLE001
                raise MCPConnectionError(f"Failed to connect to MCP server: {exc}") from exc
            self._connected = True
        await self.discover_tools()

    async def aclose(self) -> None:
        """Close the MCP connection. Safe to call even if never connected."""
        async with self._lock:
            if not self._connected:
                return
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._connected = False
                self._tools = {}

    async def __aenter__(self) -> "MCPToolProvider":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def discover_tools(self) -> List[BaseTool]:
        """Fetch the server's tool catalog and adapt it into `MCPTool` instances."""
        if not self._connected:
            raise MCPConnectionError("Cannot discover tools before connect()")
        try:
            mcp_tools = await self._client.list_tools()
        except Exception as exc:  # noqa: BLE001
            raise MCPConnectionError(f"Failed to list tools from MCP server: {exc}") from exc

        discovered: Dict[str, MCPTool] = {}
        for mcp_tool in mcp_tools:
            raw_schema = (
                getattr(mcp_tool, "input_schema", None)
                or getattr(mcp_tool, "inputSchema", None)
                or {}
            )
            exposed_name = (
                f"{self._name_prefix}.{mcp_tool.name}" if self._name_prefix else mcp_tool.name
            )
            input_model = _json_schema_to_model(exposed_name, raw_schema)
            discovered[exposed_name] = MCPTool(
                name=exposed_name,
                description=mcp_tool.description or "",
                input_schema=input_model,
                provider=self,
                remote_name=mcp_tool.name,
            )

        self._tools = discovered
        return list(self._tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Low-level passthrough to `client.call_tool`, used by `MCPTool.execute`."""
        if not self._connected or not self._client.is_connected():
            raise MCPConnectionError(f"MCP client is not connected; cannot call tool '{name}'")
        return await self._client.call_tool(name, arguments)
