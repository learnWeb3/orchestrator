"""Live integration test against a real, trusted, public MCP server.

Open-Meteo (the weather API this project would plausibly wire in) does not
itself run an official MCP endpoint -- only unofficial third-party mirrors
exist, which aren't something to depend on in a test suite. DeepWiki
(https://mcp.deepwiki.com/mcp), run by Cognition (Devin), is the closest
equivalent: a free, no-auth-required, officially documented public MCP
server, widely used as the reference "real remote MCP server" for testing.

Skipped automatically when the endpoint isn't reachable (offline dev, CI
without egress), so it never blocks unrelated work.
"""

import pytest

from orchestration_agent import Agent
from orchestration_agent.tools.mcp import MCPToolProvider

from .fakes import FakeProvider, text_response, tool_call_response

DEEPWIKI_MCP_URL = "https://mcp.deepwiki.com/mcp"


@pytest.fixture
async def deepwiki_provider():
    provider = MCPToolProvider(DEEPWIKI_MCP_URL)
    try:
        await provider.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DeepWiki MCP server unreachable: {exc}")
    yield provider
    await provider.aclose()


async def test_discovers_real_deepwiki_tools(deepwiki_provider):
    names = {t.name for t in deepwiki_provider.tools}

    assert {"read_wiki_structure", "read_wiki_contents", "ask_question"} <= names

    structure_tool = next(t for t in deepwiki_provider.tools if t.name == "read_wiki_structure")
    schema = structure_tool.input_schema.model_json_schema()
    assert schema["required"] == ["repoName"]


async def test_calls_real_deepwiki_tool_for_a_known_repo(deepwiki_provider):
    tool = next(t for t in deepwiki_provider.tools if t.name == "read_wiki_structure")
    validated = tool.input_schema(repoName="jlowin/fastmcp")

    result = await tool.execute(validated)

    assert isinstance(result, str)
    assert "fastmcp" in result.lower()


async def test_agent_round_trip_through_real_mcp_server(deepwiki_provider):
    """Proves the discovered MCPTool works unmodified inside Agent.invoke_tool,
    the same path a live LLM tool call would take -- only the LLM side is
    faked, the MCP call itself is real.
    """
    script = [
        tool_call_response("read_wiki_structure", {"repoName": "jlowin/fastmcp"}),
        text_response("Here is the documentation structure."),
    ]
    agent = Agent(
        provider=FakeProvider(script),
        system_prompt="You are a test agent. {{skills_catalog}}",
        tools=deepwiki_provider.tools,
        backoff_strategy=lambda attempt: 0,
    )

    response = await agent.run("What docs exist for jlowin/fastmcp?")

    assert response.success is True
    assert response.output == "Here is the documentation structure."
