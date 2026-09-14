"""Live integration tests against the real OpenAI API.

Skipped automatically when OPENAI_API_KEY isn't set. These are the tests that
actually prove the provider (and the fixes made vs. the spec's pseudocode) work
against the real API, not just a mock.

Prompts here are deliberately mundane (weather, time, recipes) to avoid
tripping OpenAI's moderation layer, which is unrelated to what these tests
are meant to exercise.
"""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import BaseModel

from orchestration_agent import Agent
from orchestration_agent.tools.base import BaseTool

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@pytest.fixture
def model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")


@pytest.fixture
def provider(openai_api_key_available, model):
    if not openai_api_key_available:
        pytest.skip("OPENAI_API_KEY not set")

    from orchestration_agent.provider.openai_provider import OpenAIProvider

    return OpenAIProvider(model=model, api_key=os.environ["OPENAI_API_KEY"])


async def test_live_basic_completion_returns_content_and_usage(provider, model):
    response = await provider.complete(
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        system_prompt="You are a terse test assistant.",
        model=model,
        max_tokens=64,
    )

    assert response.content.strip() != ""
    assert response.token_usage.input_tokens > 0
    assert response.token_usage.output_tokens > 0
    assert response.stop_reason in {"stop", "length"}


class _GetTimeInput(BaseModel):
    timezone: str


class _GetTimeTool(BaseTool):
    name = "get_current_time"
    description = "Returns the current time for a given IANA timezone name"
    input_schema = _GetTimeInput

    async def execute(self, input: _GetTimeInput) -> Dict[str, Any]:
        return {"timezone": input.timezone, "time": "12:00:00"}


async def test_live_tool_call_round_trip(provider, model):
    """Forces a real tool call, then completes the round trip, exercising the
    tool-call-capture fix (#3) and the raw-message-replay fix (#4) against the
    actual API — a request built the spec's original way (dropping tool_calls
    off the replayed assistant message) gets rejected by OpenAI on this second
    call.
    """
    tool = _GetTimeTool()
    messages = [
        {
            "role": "user",
            "content": "Call get_current_time for the 'UTC' timezone, then tell me the time.",
        }
    ]

    first = await provider.complete(
        messages=messages,
        system_prompt="You must use the available tool to answer.",
        model=model,
        max_tokens=256,
        tools=[tool],
        tool_choice="required",
    )

    assert first.tool_calls, "model did not request the tool despite tool_choice='required'"
    call = next((c for c in first.tool_calls if c.name == "get_current_time"), first.tool_calls[0])
    assert call.arguments.get("timezone")

    messages.append(first.raw_message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": '{"timezone": "UTC", "time": "12:00:00"}',
        }
    )

    second = await provider.complete(
        messages=messages,
        system_prompt="You must use the available tool to answer.",
        model=model,
        max_tokens=256,
        tools=[tool],
    )

    assert second.content.strip() != ""


async def test_live_agent_skill_invocation_end_to_end(provider):
    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a helpful writing assistant with access to specialized skills.\n\n"
            "{{skills_catalog}}\n\n"
            "For any text summarization request, you MUST first call invoke_skill with "
            "skill_name='text-summarizer' before answering."
        ),
        skills_path=str(SKILLS_DIR),
        max_steps=5,
    )

    response = await agent.run(
        "Summarize this in two sentences:\n\n"
        "The weather this week has been mild and mostly sunny, with a brief "
        "rain shower expected on Wednesday afternoon. Temperatures are "
        "forecast to stay in the mid-60s (Fahrenheit) through the weekend, "
        "with light winds and clear skies returning by Friday."
    )

    assert response.success is True
    assert response.output
    assert response.total_tokens > 0
