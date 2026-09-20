from typing import List

import pytest
from pydantic import ValidationError

from orchestration_agent.models.errors import ToolError
from orchestration_agent.tools import (
    AskUserQuestionInput,
    AskUserQuestionOutput,
    AskUserQuestionTool,
    Question,
    QuestionOption,
)

from .fakes import FakeProvider, text_response, tool_call_response
from .test_agent import make_agent

_OPTIONS = [
    QuestionOption(label="Option A", description="First choice"),
    QuestionOption(label="Option B", description="Second choice"),
    QuestionOption(label="Option C", description="Third choice"),
]


def _question(multiple: bool, header: str = "Choice") -> Question:
    return Question(question="Which one?", header=header, options=_OPTIONS, multiple=multiple)


class _FakeHandler:
    def __init__(self, answers: List[List[str]]) -> None:
        self.answers = answers
        self.calls: List[List[Question]] = []

    async def __call__(self, questions: List[Question]) -> List[List[str]]:
        self.calls.append(questions)
        return self.answers


async def test_single_select_returns_one_answer_per_question():
    handler = _FakeHandler([["Option A"]])
    tool = AskUserQuestionTool(handler=handler)

    result = await tool.execute(AskUserQuestionInput(questions=[_question(multiple=False)]))

    assert result["metadata"]["answers"] == [["Option A"]]
    assert "Choice" in result["output"]
    assert handler.calls == [[_question(multiple=False)]]


async def test_multi_select_preserves_order():
    handler = _FakeHandler([["Option A", "Option C"]])
    tool = AskUserQuestionTool(handler=handler)

    result = await tool.execute(AskUserQuestionInput(questions=[_question(multiple=True)]))

    assert result["metadata"]["answers"] == [["Option A", "Option C"]]


async def test_multiple_questions_preserve_order():
    questions = [_question(multiple=False, header="Q1"), _question(multiple=True, header="Q2")]
    handler = _FakeHandler([["Option A"], ["Option B", "Option C"]])
    tool = AskUserQuestionTool(handler=handler)

    result = await tool.execute(AskUserQuestionInput(questions=questions))

    assert result["metadata"]["answers"] == [["Option A"], ["Option B", "Option C"]]
    assert result["title"] == "User answered 2 question(s)"


async def test_mismatched_answer_count_raises_tool_error():
    handler = _FakeHandler([["Option A"]])  # only one answer for two questions
    tool = AskUserQuestionTool(handler=handler)
    questions = [_question(multiple=False, header="Q1"), _question(multiple=False, header="Q2")]

    with pytest.raises(ToolError):
        await tool.execute(AskUserQuestionInput(questions=questions))


def test_empty_questions_rejected():
    with pytest.raises(ValidationError):
        AskUserQuestionInput(questions=[])


def test_empty_options_rejected():
    with pytest.raises(ValidationError):
        Question(question="Which?", header="H", options=[], multiple=False)


def test_output_schema_requires_all_fields():
    with pytest.raises(ValidationError):
        AskUserQuestionOutput(title="t", output="o")  # missing metadata


async def test_agent_round_trip_through_question_tool():
    handler = _FakeHandler([["Option A"]])
    tool = AskUserQuestionTool(handler=handler)
    script = [
        tool_call_response(
            "question",
            {
                "questions": [
                    {
                        "question": "Which one?",
                        "header": "Choice",
                        "options": [
                            {"label": "Option A", "description": "First choice"},
                            {"label": "Option B", "description": "Second choice"},
                        ],
                        "multiple": False,
                    }
                ]
            },
            call_id="call_q1",
        ),
        text_response("Done."),
    ]
    agent = make_agent(script, tools=[tool])

    response = await agent.run("ask the user")

    assert response.status == "success"
    tool_msg = next(m for m in agent.provider.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_q1"
    assert "Option A" in tool_msg["content"]
