from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from orchestration_agent import Agent, BaseTool
from orchestration_agent.models.errors import TemporaryProviderError

from .fakes import FakeProvider, text_response, tool_call_response

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def make_agent(script, **kwargs) -> Agent:
    return Agent(
        provider=FakeProvider(script),
        system_prompt="You are a test agent. {{skills_catalog}}",
        skills_path=str(SKILLS_DIR),
        backoff_strategy=lambda attempt: 0,  # no real sleeping in tests
        **kwargs,
    )


async def test_simple_text_response():
    agent = make_agent([text_response("Hello there")])
    response = await agent.run("hi")

    assert response.success is True
    assert response.output == "Hello there"
    assert response.total_tokens == 15
    assert len(response.steps) == 1


async def test_skill_invocation_round_trip():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "text-summarizer"}),
        text_response("Here is my summary, using the skill instructions."),
    ]
    agent = make_agent(script)

    response = await agent.run("summarize my text")

    assert response.success is True
    assert "summary" in response.output.lower()
    assert len(response.steps) == 2

    # The second call must include the skill's markdown content as a tool message,
    # and must replay the assistant's tool_calls verbatim (fix #4).
    second_call_messages = agent.provider.calls[1]["messages"]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles

    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "Text Summarization Instructions" in tool_msg["content"]

    assistant_msg = next(
        m for m in second_call_messages if m["role"] == "assistant" and m.get("tool_calls")
    )
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"


async def test_unknown_skill_name_reports_error_and_continues():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "does-not-exist"}),
        text_response("Sorry, could not find that skill."),
    ]
    agent = make_agent(script)

    response = await agent.run("use a skill")

    assert response.success is True
    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert "not found" in tool_msg["content"]


class _EchoInput(BaseModel):
    text: str


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes back the given text"
    input_schema = _EchoInput

    async def execute(self, input: _EchoInput) -> Dict[str, Any]:
        return {"echoed": input.text}


async def test_external_tool_invocation_round_trip():
    script = [
        tool_call_response("echo", {"text": "hi"}, call_id="call_9"),
        text_response("Done."),
    ]
    agent = make_agent(script, tools=[_EchoTool()])

    response = await agent.run("echo hi")

    assert response.success is True
    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_9"
    assert "echoed" in tool_msg["content"]


async def test_unregistered_tool_name_reports_error_and_continues():
    script = [
        tool_call_response("mystery_tool", {}, call_id="call_x"),
        text_response("Fallback response."),
    ]
    agent = make_agent(script)

    response = await agent.run("do something")

    assert response.success is True
    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert "Unknown tool" in tool_msg["content"]


async def test_max_steps_exceeded():
    def always_tool_call(_index):
        return tool_call_response("invoke_skill", {"skill_name": "text-summarizer"})

    agent = make_agent([always_tool_call], max_steps=3)

    response = await agent.run("loop forever")

    assert response.success is False
    assert response.error == "max_steps_exceeded"
    assert len(response.steps) == 3


async def test_retries_then_succeeds():
    script = [
        TemporaryProviderError("temporary blip"),
        TemporaryProviderError("temporary blip"),
        text_response("Recovered"),
    ]
    agent = make_agent(script, max_retries=3)

    response = await agent.run("hi")

    assert response.success is True
    assert response.output == "Recovered"
    # All 3 attempts happened within a single logical step.
    assert len(response.steps) == 1


async def test_retry_exhaustion_fails_run():
    script = [TemporaryProviderError("down") for _ in range(10)]
    agent = make_agent(script, max_retries=2)

    response = await agent.run("hi")

    assert response.success is False
    assert "down" in response.error


class _Person(BaseModel):
    name: str
    age: int


async def test_structured_output_success():
    agent = make_agent(
        [text_response('{"name": "Jane", "age": 54}')],
        output_schema=_Person,
    )

    response = await agent.run("who is this")

    assert response.success is True
    assert isinstance(response.output, _Person)
    assert response.output.name == "Jane"


async def test_structured_output_failure_marks_run_unsuccessful():
    agent = make_agent(
        [text_response("not json at all")],
        output_schema=_Person,
    )

    response = await agent.run("who is this")

    assert response.success is False
    assert response.error is not None
