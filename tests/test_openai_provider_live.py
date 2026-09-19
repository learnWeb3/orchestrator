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
from typing import Any, Dict, List

import jsonschema
import pytest
import yaml
from pydantic import BaseModel, Field

from orchestration_agent import Agent, Budget
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
        budget=Budget(max_turns=5),
    )

    response = await agent.run(
        "Summarize this in two sentences:\n\n"
        "The weather this week has been mild and mostly sunny, with a brief "
        "rain shower expected on Wednesday afternoon. Temperatures are "
        "forecast to stay in the mid-60s (Fahrenheit) through the weekend, "
        "with light winds and clear skies returning by Friday."
    )

    assert response.status == "success"
    assert response.response
    assert response.total_tokens > 0
    # text-summarizer declares `output`, so the round trip includes both the
    # dispatch and the schema-bound completion (spec section 6).
    assert any(m.type == "text-summarizer" for m in response.messages)


# -- Mock external tool result flowing through a skill's `output` schema ---------
#
# Everything here is a real call to the OpenAI API — model, skill dispatch, and
# the schema-bound completion. The only fake is `_OrderStatusTool.execute`: it
# never hits a real network, it just returns a fixed payload, standing in for
# whatever real API a deployment would wire in.

_ORDER_STATUS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "description": "The order ID that was looked up"},
        "status": {
            "type": "string",
            "description": "Shipment status, copied verbatim from the get_order_status tool result",
        },
        "eta_days": {
            "type": "integer",
            "description": "Estimated delivery days, copied verbatim from the tool result",
        },
    },
    "required": ["order_id", "status", "eta_days"],
    "additionalProperties": False,
}

_ORDER_STATUS_SKILL_BODY = """## Order Status Reporting Instructions

Earlier in this conversation, the `get_order_status` tool was already called
for a specific order, and its result (`order_id`, `status`, `eta_days`) is
present in the tool output above.

Read that tool result and report it back as structured data:
- `order_id`: the order ID that was looked up
- `status`: the status string from the tool result, verbatim
- `eta_days`: the estimated delivery days from the tool result, verbatim

Do not invent values. Use exactly what the tool returned.
"""


def _write_order_status_skill(skills_dir: Path) -> None:
    skill_dir = skills_dir / "order-status-reporter"
    skill_dir.mkdir()
    frontmatter = yaml.dump(
        {
            "name": "order-status-reporter",
            "description": (
                "Report the current status of a customer order, using a prior "
                "get_order_status tool result"
            ),
            "compatibility": "python>=3.10, openai",
            "output": _ORDER_STATUS_OUTPUT_SCHEMA,
        },
        sort_keys=False,
    )
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}---\n\n{_ORDER_STATUS_SKILL_BODY}\n")


class _OrderStatusInput(BaseModel):
    order_id: str = Field(..., description="The order ID to look up")


class _OrderStatusTool(BaseTool):
    """A fake/mock external API tool: no real network call, fixed payload."""

    name = "get_order_status"
    description = "Look up the current shipment status for a given order_id"
    input_schema = _OrderStatusInput

    def __init__(self) -> None:
        self.calls: List[str] = []

    async def execute(self, input: _OrderStatusInput) -> Dict[str, Any]:
        self.calls.append(input.order_id)
        return {"order_id": input.order_id, "status": "shipped", "eta_days": 3}


async def test_live_mock_tool_result_flows_through_skill_output_schema(provider, tmp_path):
    """A real model call chain: the model calls the mock `get_order_status`
    tool, then invokes the `order-status-reporter` skill, whose bound
    completion (spec section 6) must report the tool's payload back as
    schema-validated structured data — proving a tool's result actually
    reaches, and survives, a skill's `output` contract end to end.
    """
    _write_order_status_skill(tmp_path)
    tool = _OrderStatusTool()

    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a customer support assistant with access to specialized "
            "skills and tools.\n\n"
            "{{skills_catalog}}\n\n"
            "When asked about an order's status, first call get_order_status "
            "with the order's id to look up its current status. Once you have "
            "that result, call invoke_skill with "
            "skill_name='order-status-reporter' and follow its instructions "
            "to report the status."
        ),
        skills_path=str(tmp_path),
        tools=[tool],
        budget=Budget(max_turns=6),
    )

    response = await agent.run("What is the status of order ORD-42?")

    assert response.status == "success"
    assert tool.calls, "the mock get_order_status tool was never invoked"

    dispatch = next(
        (m for m in response.messages if m.type == "order-status-reporter" and m.input is not None),
        None,
    )
    assert dispatch is not None, "order-status-reporter was never dispatched"
    assert dispatch.status == "success"

    bound = next(
        (m for m in response.messages if m.type == "order-status-reporter" and m.input is None),
        None,
    )
    assert bound is not None, "no schema-bound completion recorded for order-status-reporter"
    assert bound.status == "success", bound.error

    # Explicitly re-validate against the declared schema (belt-and-braces on
    # top of the agent's own boundary validation, spec section 6).
    jsonschema.validate(bound.data, _ORDER_STATUS_OUTPUT_SCHEMA)

    # The tool's fixed payload must survive unchanged into the skill's output.
    assert bound.data["order_id"] == tool.calls[0]
    assert bound.data["status"] == "shipped"
    assert bound.data["eta_days"] == 3
