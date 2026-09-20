"""Built-in tool letting the LLM ask the human one or more multiple-choice
questions mid-run.

No transport/UI is implemented here -- how the question actually reaches the
human (terminal, web socket, queue, ...) is entirely up to the embedder,
supplied as the required `handler` callback.
"""

from typing import Any, Awaitable, Callable, Dict, List

from pydantic import BaseModel, Field

from ..models.errors import ToolError
from .base import BaseTool


class QuestionOption(BaseModel):
    label: str
    description: str


class Question(BaseModel):
    question: str
    header: str
    options: List[QuestionOption] = Field(min_length=1)
    multiple: bool


class AskUserQuestionInput(BaseModel):
    questions: List[Question] = Field(min_length=1)


class AskUserQuestionMetadata(BaseModel):
    answers: List[List[str]]


class AskUserQuestionOutput(BaseModel):
    title: str
    output: str
    metadata: AskUserQuestionMetadata


QuestionHandler = Callable[[List[Question]], Awaitable[List[List[str]]]]


class AskUserQuestionTool(BaseTool):
    name = "question"
    description = (
        "Use this tool when you need to ask the user questions during execution. "
        "This allows you to gather user preferences or requirements, clarify "
        "ambiguous instructions, get decisions on implementation choices, or offer "
        "choices to the user about what direction to take."
    )
    input_schema = AskUserQuestionInput
    output_schema = AskUserQuestionOutput

    def __init__(self, handler: QuestionHandler) -> None:
        self.handler = handler

    async def execute(self, input: AskUserQuestionInput) -> Dict[str, Any]:
        answers = await self.handler(input.questions)
        if len(answers) != len(input.questions):
            raise ToolError("handler returned a mismatched number of answers")

        output = "\n".join(
            f"{q.header}: {q.question}\n  -> {', '.join(ans)}"
            for q, ans in zip(input.questions, answers)
        )
        return {
            "title": f"User answered {len(input.questions)} question(s)",
            "output": output,
            "metadata": {"answers": answers},
        }
