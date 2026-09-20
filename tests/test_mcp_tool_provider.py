from typing import Any, Dict

import pytest
from fastmcp import FastMCP

from orchestration_agent import Agent
from orchestration_agent.tools.mcp import MCPConnectionError, MCPTool, MCPToolProvider

from .fakes import FakeProvider, text_response, tool_call_response


def make_server() -> FastMCP:
    server = FastMCP("TestServer")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def boom() -> str:
        """Always fails."""
        raise ValueError("kaboom")

    return server


async def test_discover_tools_exposes_base_tool_interface():
    async with MCPToolProvider(make_server()) as provider:
        tools: Dict[str, Any] = {t.name: t for t in provider.tools}

        assert set(tools) == {"add", "boom"}
        add_tool = tools["add"]
        assert isinstance(add_tool, MCPTool)
        assert add_tool.description == "Add two numbers."

        schema = add_tool.input_schema.model_json_schema()
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["b"]["type"] == "integer"
        assert set(schema["required"]) == {"a", "b"}


async def test_execute_calls_through_to_server():
    async with MCPToolProvider(make_server()) as provider:
        add_tool = next(t for t in provider.tools if t.name == "add")
        validated = add_tool.input_schema(a=2, b=3)

        result = await add_tool.execute(validated)

        assert result == {"result": 5}


async def test_execute_raises_on_server_error():
    async with MCPToolProvider(make_server()) as provider:
        boom_tool = next(t for t in provider.tools if t.name == "boom")
        validated = boom_tool.input_schema()

        with pytest.raises(Exception):
            await boom_tool.execute(validated)


async def test_call_tool_before_connect_raises():
    provider = MCPToolProvider(make_server())

    with pytest.raises(MCPConnectionError):
        await provider.call_tool("add", {"a": 1, "b": 1})


async def test_name_prefix_namespaces_tools():
    async with MCPToolProvider(make_server(), name_prefix="srv1") as provider:
        names = {t.name for t in provider.tools}
        assert names == {"srv1.add", "srv1.boom"}


async def test_agent_invoke_tool_round_trip_through_mcp():
    async with MCPToolProvider(make_server()) as provider:
        script = [
            tool_call_response("add", {"a": 4, "b": 5}),
            text_response("The sum is 9."),
        ]
        agent = Agent(
            provider=FakeProvider(script),
            system_prompt="You are a test agent. {{skills_catalog}}",
            tools=provider.tools,
            backoff_strategy=lambda attempt: 0,
        )

        response = await agent.run("what is 4 + 5?")

        assert response.status == "success"
        assert response.response == "The sum is 9."
