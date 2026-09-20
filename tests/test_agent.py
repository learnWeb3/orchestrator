from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from orchestration_agent import Agent, BaseTool, Budget
from orchestration_agent.models.errors import (
    ModelRefusalError,
    SchemaCompilationError,
    TemporaryProviderError,
)
from orchestration_agent.models.rate_limiter import RateLimiterContext

from orchestration_agent.models.provider import ToolCall

from .fakes import FakeProvider, json_response, text_response, tool_call_response

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def make_agent(script, **kwargs) -> Agent:
    return Agent(
        provider=FakeProvider(script),
        system_prompt="You are a test agent. {{skills_catalog}}",
        skills_path=str(SKILLS_DIR),
        backoff_strategy=lambda attempt: 0,  # no real sleeping in tests
        **kwargs,
    )


# -- Ordinary turns / final response --------------------------------------------


async def test_simple_text_response():
    agent = make_agent([text_response("Hello there")])
    response = await agent.run("hi")

    assert response.status == "success"
    assert response.response == "Hello there"
    assert response.total_tokens == 15
    assert len(response.steps) == 1
    assert response.messages == []


async def test_retries_then_succeeds():
    script = [
        TemporaryProviderError("temporary blip"),
        TemporaryProviderError("temporary blip"),
        text_response("Recovered"),
    ]
    agent = make_agent(script, max_retries=3)

    response = await agent.run("hi")

    assert response.status == "success"
    assert response.response == "Recovered"
    # All 3 attempts happened within a single logical step.
    assert len(response.steps) == 1


async def test_retry_exhaustion_fails_run():
    script = [TemporaryProviderError("down") for _ in range(10)]
    agent = make_agent(script, max_retries=2)

    response = await agent.run("hi")

    assert response.status == "error"
    assert response.error.code == "AGENT_ERROR"
    assert "down" in response.error.message


# -- Skill dispatch (no `output`) -------------------------------------------------


async def test_skill_invocation_round_trip_no_output_schema():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "recipe-helper"}),
        text_response("Here is your recipe."),
    ]
    agent = make_agent(script)

    response = await agent.run("suggest a recipe")

    assert response.status == "success"
    assert response.response == "Here is your recipe."
    assert len(response.messages) == 1

    dispatch = response.messages[0]
    assert dispatch.type == "recipe-helper"
    assert dispatch.status == "success"
    assert dispatch.input == {"skill_name": "recipe-helper"}
    assert "Recipe Assistant Instructions" in dispatch.data
    assert dispatch.duration_ms is not None

    # The second call must include the skill's markdown content as a tool message,
    # and must replay the assistant's tool_calls verbatim.
    second_call_messages = agent.provider.calls[1]["messages"]
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "Recipe Assistant Instructions" in tool_msg["content"]

    assistant_msg = next(
        m for m in second_call_messages if m["role"] == "assistant" and m.get("tool_calls")
    )
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"


async def test_missing_skill_name_reports_skill_not_found():
    script = [
        tool_call_response("invoke_skill", {}),
        text_response("Sorry, I need a skill name."),
    ]
    agent = make_agent(script)

    response = await agent.run("use a skill")

    assert response.status == "success"
    assert len(response.messages) == 1
    assert response.messages[0].type == "invoke_skill"
    assert response.messages[0].status == "error"
    assert response.messages[0].error.code == "SKILL_NOT_FOUND"
    assert response.messages[0].data is None


async def test_unknown_skill_name_reports_error_and_continues():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "does-not-exist"}),
        text_response("Sorry, could not find that skill."),
    ]
    agent = make_agent(script)

    response = await agent.run("use a skill")

    assert response.status == "success"
    assert response.messages[0].type == "does-not-exist"
    assert response.messages[0].error.code == "SKILL_NOT_FOUND"

    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert "not found" in tool_msg["content"]


# -- Skill dispatch with `output` (bound completion) ------------------------------


_SUMMARY_PAYLOAD = {
    "summary": "A short summary.",
    "key_points": ["point one", "point two"],
}


async def test_skill_with_output_binds_next_completion():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "text-summarizer"}),
        json_response(_SUMMARY_PAYLOAD),
        text_response("Done — summarized using text-summarizer."),
    ]
    agent = make_agent(script)

    response = await agent.run("summarize my text")

    assert response.status == "success"
    assert len(response.messages) == 2

    dispatch, bound = response.messages
    assert dispatch.type == "text-summarizer"
    assert dispatch.status == "success"
    assert dispatch.input == {"skill_name": "text-summarizer"}

    assert bound.type == "text-summarizer"
    assert bound.status == "success"
    assert bound.input is None  # did not originate from a tool call
    assert bound.data == _SUMMARY_PAYLOAD

    # The bound completion must be issued with no tools and the skill's schema bound.
    bound_call = agent.provider.calls[1]
    assert bound_call["tools"] is None
    assert bound_call["structured_output"] is not None
    assert bound_call["structured_output_name"] == "text-summarizer"

    # The following turn resumes as an ordinary step (no external tools registered
    # in this agent, so `tools` is None here — invoke_skill is always offered by
    # the provider itself).
    final_call = agent.provider.calls[2]
    assert final_call["tools"] is None


async def test_skill_output_validation_repairs_once_then_succeeds():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "text-summarizer"}),
        text_response("not valid json"),
        json_response(_SUMMARY_PAYLOAD),
        text_response("All done."),
    ]
    agent = make_agent(script)

    response = await agent.run("summarize")

    assert response.status == "success"
    bound = response.messages[1]
    assert bound.status == "success"
    assert bound.data == _SUMMARY_PAYLOAD
    # Both the failed attempt and the repair count toward one duration.
    assert bound.duration_ms is not None


async def test_skill_output_validation_fails_after_one_repair():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "text-summarizer"}),
        text_response("still not json"),
        text_response("also not json"),
        text_response("Sorry, that skill failed."),
    ]
    agent = make_agent(script)

    response = await agent.run("summarize")

    assert response.status == "success"  # the run itself recovers
    bound = response.messages[1]
    assert bound.status == "error"
    assert bound.data is None
    assert bound.error.code == "OUTPUT_VALIDATION_ERROR"
    assert "violations" in bound.error.details


async def test_skill_output_model_refusal():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "text-summarizer"}),
        ModelRefusalError("I can't help with that."),
        text_response("Understood, moving on."),
    ]
    agent = make_agent(script)

    response = await agent.run("summarize")

    assert response.status == "success"
    bound = response.messages[1]
    assert bound.status == "error"
    assert bound.error.code == "MODEL_REFUSAL"
    assert bound.data is None


async def test_skill_output_schema_compilation_falls_back_invisibly():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "text-summarizer"}),
        SchemaCompilationError("provider rejected the schema"),
        json_response(_SUMMARY_PAYLOAD),
        text_response("All done via fallback."),
    ]
    agent = make_agent(script)

    response = await agent.run("summarize")

    assert response.status == "success"
    bound = response.messages[1]
    assert bound.status == "success"
    assert bound.data == _SUMMARY_PAYLOAD

    # The fallback call has no tools and no native structured_output — the
    # schema was injected into the prompt instead.
    fallback_call = agent.provider.calls[2]
    assert fallback_call["structured_output"] is None
    assert fallback_call["tools"] is None
    fallback_msg = fallback_call["messages"][-1]
    assert "JSON Schema" in fallback_msg["content"]

    # The substitution is invisible to the model: only one SkillOutput for
    # this bound completion, not a separate error entry.
    assert len(response.messages) == 2


async def test_only_last_output_declaring_skill_in_round_binds():
    # Three invoke_skill calls in a single round: text-summarizer (has output),
    # then recipe-helper (no output), then text-summarizer again (has output).
    # Only the last output-declaring dispatch should bind the next completion.
    multi_call = tool_call_response("invoke_skill", {"skill_name": "recipe-helper"}, call_id="call_a")
    multi_call.tool_calls = [
        ToolCall(name="invoke_skill", arguments={"skill_name": "text-summarizer"}, id="call_a"),
        ToolCall(name="invoke_skill", arguments={"skill_name": "recipe-helper"}, id="call_b"),
        ToolCall(name="invoke_skill", arguments={"skill_name": "text-summarizer"}, id="call_c"),
    ]
    multi_call.raw_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "invoke_skill", "arguments": "{}"}}
            for cid in ("call_a", "call_b", "call_c")
        ],
    }

    script = [multi_call, json_response(_SUMMARY_PAYLOAD), text_response("done")]
    agent = make_agent(script)

    response = await agent.run("go")

    # 3 dispatches + 1 bound completion (for the last text-summarizer dispatch).
    assert len(response.messages) == 4
    assert [m.type for m in response.messages] == [
        "text-summarizer",
        "recipe-helper",
        "text-summarizer",
        "text-summarizer",
    ]
    assert response.messages[3].input is None  # the bound completion


# -- External tools (unaffected by SkillOutput trace) ------------------------------


class _EchoInput(BaseModel):
    text: str


class _EchoOutput(BaseModel):
    echoed: str


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes back the given text"
    input_schema = _EchoInput
    output_schema = _EchoOutput

    async def execute(self, input: _EchoInput) -> Dict[str, Any]:
        return {"echoed": input.text}


async def test_external_tool_invocation_round_trip():
    script = [
        tool_call_response("echo", {"text": "hi"}, call_id="call_9"),
        text_response("Done."),
    ]
    agent = make_agent(script, tools=[_EchoTool()])

    response = await agent.run("echo hi")

    assert response.status == "success"
    # Tools are out of scope for the SkillOutput trace (spec section 8).
    assert response.messages == []

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

    assert response.status == "success"
    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert "Unknown tool" in tool_msg["content"]


class _BadOutputInput(BaseModel):
    text: str


class _BadOutputOutput(BaseModel):
    required_field: str


class _BadOutputTool(BaseTool):
    """Deliberately returns a dict that violates its own `output_schema`."""

    name = "bad_output"
    description = "Always returns a shape that doesn't match output_schema"
    input_schema = _BadOutputInput
    output_schema = _BadOutputOutput

    async def execute(self, input: _BadOutputInput) -> Dict[str, Any]:
        return {"wrong_field": input.text}


async def test_tool_output_violating_output_schema_reports_error_and_continues():
    script = [
        tool_call_response("bad_output", {"text": "hi"}, call_id="call_y"),
        text_response("Fallback response."),
    ]
    agent = make_agent(script, tools=[_BadOutputTool()])

    response = await agent.run("do something")

    assert response.status == "success"
    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert "Invalid tool output" in tool_msg["content"]


# -- Budget (spec section 9) -----------------------------------------------------


async def test_budget_turns_exhausted_gives_reserved_final_turn():
    script = [text_response("Only one turn allowed.")]
    agent = make_agent(script, budget=Budget(max_turns=1, max_invocations=25))

    response = await agent.run("hi")

    assert response.status == "partial"
    assert response.response == "Only one turn allowed."


async def test_budget_turns_exhausted_without_valid_response_is_error():
    class _Person(BaseModel):
        name: str
        age: int

    script = [text_response("not json at all")]
    agent = make_agent(script, budget=Budget(max_turns=1), output_schema=_Person)

    response = await agent.run("who is this")

    assert response.status == "error"
    assert response.error.code == "BUDGET_EXHAUSTED"


async def test_budget_invocations_exhausted():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "recipe-helper"}),
        text_response("Hit the invocation cap."),
    ]
    agent = make_agent(script, budget=Budget(max_invocations=1, max_turns=15))

    response = await agent.run("go")

    assert response.status == "partial"
    assert len(response.messages) == 1  # only the one allowed dispatch


# -- Final response schema (response_schema / output_schema) ----------------------


class _Person(BaseModel):
    name: str
    age: int


async def test_structured_output_success():
    agent = make_agent(
        [text_response('{"name": "Jane", "age": 54}')],
        output_schema=_Person,
    )

    response = await agent.run("who is this")

    assert response.status == "success"
    assert isinstance(response.response, _Person)
    assert response.response.name == "Jane"


async def test_structured_output_repairs_once_then_succeeds():
    script = [
        text_response("not json at all"),
        json_response({"name": "Jane", "age": 54}),
    ]
    agent = make_agent(script, output_schema=_Person)

    response = await agent.run("who is this")

    assert response.status == "success"
    assert response.response.name == "Jane"


async def test_structured_output_failure_after_repair_marks_run_unsuccessful():
    script = [
        text_response("not json at all"),
        text_response("still not json"),
    ]
    agent = make_agent(script, output_schema=_Person)

    response = await agent.run("who is this")

    assert response.status == "error"
    assert response.error.code == "RESPONSE_VALIDATION_ERROR"
    assert response.response is None


# -- Rate limiting: delta accounting, not cumulative (spec section 10) ------------


class RecordingRateLimiter(RateLimiterContext):
    def __init__(self):
        self.consumed: List[int] = []
        self.released: List[int] = []

    def check_and_consume(self, tokens: int) -> bool:
        self.consumed.append(tokens)
        return True

    def get_current_bucket(self):
        return {"consumed": sum(self.consumed) - sum(self.released)}

    def reset(self) -> None:
        self.consumed.clear()
        self.released.clear()

    def release(self, tokens: int) -> None:
        self.released.append(tokens)


async def test_rate_limiter_receives_step_delta_not_cumulative_total():
    script = [
        tool_call_response("invoke_skill", {"skill_name": "recipe-helper"}),
        tool_call_response("invoke_skill", {"skill_name": "recipe-helper"}),
        text_response("Done."),
    ]
    limiter = RecordingRateLimiter()
    agent = make_agent(script, rate_limiter_context=limiter, budget=Budget(max_turns=10))

    response = await agent.run("go")

    assert response.status == "success"
    # Reserve-then-reconcile nets out to exactly the actual per-call usage, so
    # cumulative net consumption equals the run's total tokens — never an
    # inflated multiple of it, which is what passing the running total on
    # every step would produce.
    net_consumed = sum(limiter.consumed) - sum(limiter.released)
    assert net_consumed == response.total_tokens
    assert len(limiter.consumed) == len(response.steps)
