"""Multi-turn conversation: a laptop-search skill with a discriminated-union
`output` (needs_more_filters -> narrowed_results -> final_pick), narrowed via
repeated `search_laptops` calls and `AskUserQuestionTool` clarifications.

Each `invoke_skill` dispatch's bound completion happens on the very next
provider call with tools disabled (agent.py), so the skill is re-invoked once
per structured checkpoint; the search/question tool calls happen in the
ordinary turns between those re-invocations.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from orchestration_agent.tools import AskUserQuestionTool, BaseTool, Question

from .fakes import json_response, text_response, tool_call_response
from .test_agent import make_agent

_CATALOG = [
    {"id": "lt-01", "brand": "Acme", "model": "Aria 14", "price_usd": 899.0,
     "screen_size_in": 14.0, "ram_gb": 16, "use_case": "general"},
    {"id": "lt-02", "brand": "Acme", "model": "Vantage 16", "price_usd": 2199.0,
     "screen_size_in": 16.0, "ram_gb": 32, "use_case": "gaming"},
    {"id": "lt-03", "brand": "Nova", "model": "Sketch 15", "price_usd": 1299.0,
     "screen_size_in": 15.0, "ram_gb": 16, "use_case": "creative"},
    {"id": "lt-04", "brand": "Nova", "model": "Sketch 15 Pro", "price_usd": 1899.0,
     "screen_size_in": 15.0, "ram_gb": 32, "use_case": "creative"},
    {"id": "lt-05", "brand": "Vertex", "model": "Flow 13", "price_usd": 749.0,
     "screen_size_in": 13.0, "ram_gb": 8, "use_case": "general"},
    {"id": "lt-06", "brand": "Zenith", "model": "Studio 15", "price_usd": 1450.0,
     "screen_size_in": 15.0, "ram_gb": 16, "use_case": "creative"},
]


class SearchLaptopsInput(BaseModel):
    brand: Optional[str] = None
    use_case: Optional[str] = None
    min_price_usd: Optional[float] = None
    max_price_usd: Optional[float] = None
    min_ram_gb: Optional[int] = None
    id: Optional[str] = None


class LaptopSummary(BaseModel):
    id: str
    brand: str
    model: str
    price_usd: float
    screen_size_in: float
    ram_gb: int
    use_case: str


class SearchLaptopsOutput(BaseModel):
    count: int
    candidates: List[LaptopSummary]
    note: str


class SearchLaptopsTool(BaseTool):
    """A fake/mock catalog search tool: no real network call, filters an
    in-memory list based on whichever parameters are present."""

    name = "search_laptops"
    description = "Search the laptop catalog with the given filters"
    input_schema = SearchLaptopsInput
    output_schema = SearchLaptopsOutput

    def __init__(self) -> None:
        self.calls: List[SearchLaptopsInput] = []

    async def execute(self, input: SearchLaptopsInput) -> Dict[str, Any]:
        self.calls.append(input)
        matches = [
            laptop
            for laptop in _CATALOG
            if (input.brand is None or laptop["brand"].lower() in input.brand.lower())
            and (input.use_case is None or laptop["use_case"].lower() in input.use_case.lower())
            and (input.min_price_usd is None or laptop["price_usd"] >= input.min_price_usd)
            and (input.max_price_usd is None or laptop["price_usd"] <= input.max_price_usd)
            and (input.min_ram_gb is None or laptop["ram_gb"] >= input.min_ram_gb)
            and (input.id is None or laptop["id"] == input.id)
        ]
        if len(matches) > 2:
            note = "Too many matches; narrow by budget or use case."
        elif len(matches) == 1:
            note = "Single match found."
        else:
            note = "Candidates ready for the user to choose from."
        return {
            "count": len(matches),
            "candidates": [
                {
                    "id": m["id"],
                    "brand": m["brand"],
                    "model": m["model"],
                    "price_usd": m["price_usd"],
                    "screen_size_in": m["screen_size_in"],
                    "ram_gb": m["ram_gb"],
                    "use_case": m["use_case"],
                }
                for m in matches
            ],
            "note": note,
        }


class _ScriptedHandler:
    """A `QuestionHandler` returning a different scripted answer on each call."""

    def __init__(self, answers_per_call: List[List[List[str]]]) -> None:
        self._answers_per_call = answers_per_call
        self.calls: List[List[Question]] = []

    async def __call__(self, questions: List[Question]) -> List[List[str]]:
        self.calls.append(questions)
        return self._answers_per_call[len(self.calls) - 1]


async def test_laptop_search_multi_turn_conversation():
    search_tool = SearchLaptopsTool()
    handler = _ScriptedHandler(
        [
            [["creative"], ["1500"]],  # answers to stage-1 clarifying questions
            [["Zenith Studio 15"]],  # answer to stage-2 candidate pick
        ]
    )
    question_tool = AskUserQuestionTool(handler=handler)

    script = [
        # -- Stage 1: gather filters --------------------------------------
        tool_call_response("invoke_skill", {"skill_name": "laptop-search"}, call_id="skill_1"),
        json_response(
            {
                "result": {
                    "type": "needs_more_filters",
                    "missing_filters": ["use_case", "budget"],
                    "message": "What will you mainly use it for, and what's your budget?",
                }
            }
        ),
        tool_call_response(
            "question",
            {
                "questions": [
                    {
                        "question": "What will you mainly use the laptop for?",
                        "header": "Use case",
                        "options": [
                            {"label": "creative", "description": "Design, photo/video editing"},
                            {"label": "gaming", "description": "Gaming"},
                            {"label": "general", "description": "Browsing, office work"},
                        ],
                        "multiple": False,
                    },
                    {
                        "question": "What's your maximum budget in USD?",
                        "header": "Budget",
                        "options": [
                            {"label": "1500", "description": "Up to $1500"},
                            {"label": "2500", "description": "Up to $2500"},
                        ],
                        "multiple": False,
                    },
                ]
            },
            call_id="q_1",
        ),
        # -- Stage 2: narrow candidates -------------------------------------
        tool_call_response(
            "search_laptops",
            {"use_case": "creative", "max_price_usd": 1500},
            call_id="s_1",
        ),
        tool_call_response("invoke_skill", {"skill_name": "laptop-search"}, call_id="skill_2"),
        json_response(
            {
                "result": {
                    "type": "narrowed_results",
                    "candidates": [
                        {"id": "lt-03", "brand": "Nova", "model": "Sketch 15", "price_usd": 1299.0,
                         "screen_size_in": 15.0, "ram_gb": 16, "use_case": "creative"},
                        {"id": "lt-06", "brand": "Zenith", "model": "Studio 15", "price_usd": 1450.0,
                         "screen_size_in": 15.0, "ram_gb": 16, "use_case": "creative"},
                    ],
                    "message": "Here are two creative laptops under $1500 -- which one?",
                }
            }
        ),
        tool_call_response(
            "question",
            {
                "questions": [
                    {
                        "question": "Which laptop would you like?",
                        "header": "Pick one",
                        "options": [
                            {"label": "Nova Sketch 15", "description": "$1299, 16GB RAM"},
                            {"label": "Zenith Studio 15", "description": "$1450, 16GB RAM"},
                        ],
                        "multiple": False,
                    }
                ]
            },
            call_id="q_2",
        ),
        # -- Stage 3: confirm final pick -------------------------------------
        tool_call_response("search_laptops", {"id": "lt-06"}, call_id="s_2"),
        tool_call_response("invoke_skill", {"skill_name": "laptop-search"}, call_id="skill_3"),
        json_response(
            {
                "result": {
                    "type": "final_pick",
                    "laptop": {"id": "lt-06", "brand": "Zenith", "model": "Studio 15",
                               "price_usd": 1450.0, "screen_size_in": 15.0, "ram_gb": 16,
                               "use_case": "creative"},
                    "message": "The Zenith Studio 15 fits your creative use case and budget.",
                }
            }
        ),
        text_response("Great, I've found your laptop: the Zenith Studio 15."),
    ]

    agent = make_agent(script, tools=[search_tool, question_tool])

    response = await agent.run("Help me find a laptop.")

    assert response.status == "success"

    assert len(response.messages) == 6
    assert all(m.type == "laptop-search" for m in response.messages)

    bound = [m for m in response.messages if m.input is None]
    assert len(bound) == 3
    assert [m.data["result"]["type"] for m in bound] == [
        "needs_more_filters",
        "narrowed_results",
        "final_pick",
    ]

    assert len(handler.calls) == 2
    assert [q.header for q in handler.calls[0]] == ["Use case", "Budget"]
    assert [q.header for q in handler.calls[1]] == ["Pick one"]

    assert len(search_tool.calls) == 2
    assert search_tool.calls[0].use_case == "creative"
    assert search_tool.calls[1].id == "lt-06"

    assert bound[-1].data["result"]["laptop"]["id"] == "lt-06"
