"""End-to-end live test: a real `Agent` + real OpenAI model drive the
`laptop-search` skill (discriminated-union `output`) through `search_laptops`
(mocked catalog) and `AskUserQuestionTool`.

The only thing mocked here is the human: `SimulatedUserHandler` answers
whatever questions the model asks, by matching option labels/prices against
a fixed set of preferences. Everything else -- the model's turn-by-turn
decisions, how many times it calls `search_laptops`/`question`/`invoke_skill`,
the exact wording of its questions -- is real, so assertions are necessarily
looser than the scripted `FakeProvider` version in
`tests/test_laptop_search_conversation.py`.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from orchestration_agent import Agent, Budget
from orchestration_agent.logging import StdoutLogger
from orchestration_agent.tools import AskUserQuestionTool, Question

from .test_laptop_search_conversation import _CATALOG, SearchLaptopsInput, SearchLaptopsTool

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _log(tag: str, message: Any) -> None:
    if not isinstance(message, str):
        message = json.dumps(message, indent=2, default=str)
    indented = "\n".join(f"  {line}" for line in message.splitlines())
    print(f"\n--- {tag} ---\n{indented}")


@pytest.fixture
def model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")


@pytest.fixture
def provider(openai_api_key_available, model):
    if not openai_api_key_available:
        pytest.skip("OPENAI_API_KEY not set")

    from orchestration_agent.provider.openai_provider import OpenAIProvider

    return OpenAIProvider(model=model, api_key=os.environ["OPENAI_API_KEY"])


def _first_price(text: str) -> Optional[float]:
    match = re.search(r"\$?\s?(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


class SimulatedUserHandler:
    """Stands in for the human shopper: matches the use-case preference by
    keyword, and otherwise picks the cheapest affordable option -- for a
    budget question ("Up to $1500") and for a candidate-pick question alike,
    since both list a price per option. Falls back to the first option."""

    def __init__(self, use_case: str, max_budget_usd: float) -> None:
        self.use_case = use_case.lower()
        self.max_budget_usd = max_budget_usd
        self.calls: List[List[Question]] = []

    async def __call__(self, questions: List[Question]) -> List[List[str]]:
        self.calls.append(questions)
        answers: List[List[str]] = []
        for q in questions:
            _log(
                "AGENT -> USER (question tool)",
                f"[{q.header}] {q.question}\n"
                + "\n".join(f"  ({i + 1}) {o.label} -- {o.description}" for i, o in enumerate(q.options)),
            )
            answer = self._answer(q)
            _log("USER -> AGENT (simulated answer)", answer)
            answers.append([answer])
        return answers

    def _answer(self, q: Question) -> str:
        for option in q.options:
            if self.use_case in option.label.lower():
                return option.label

        priced = [
            (option, _first_price(f"{option.label} {option.description}"))
            for option in q.options
        ]
        priced = [(o, p) for o, p in priced if p is not None]
        if priced:
            affordable = [(o, p) for o, p in priced if p <= self.max_budget_usd]
            pool = affordable or priced
            return min(pool, key=lambda pair: pair[1])[0].label

        return q.options[0].label


class LoggingSearchLaptopsTool(SearchLaptopsTool):
    """`SearchLaptopsTool` with the request/response logged, so the search
    round trip shows up in the same turn-by-turn log as the question tool."""

    async def execute(self, input: SearchLaptopsInput) -> Dict[str, Any]:
        _log("AGENT -> search_laptops (call)", input.model_dump(exclude_none=True))
        result = await super().execute(input)
        _log("search_laptops -> AGENT (result)", result)
        return result


def _log_skill_message(m: Any) -> None:
    if m.input is not None:
        _log(f"AGENT -> invoke_skill({m.input['skill_name']})", "instructions reloaded")
    elif m.status == "success":
        _log(f"{m.type} -> AGENT (bound completion)", m.data)
    else:
        _log(f"{m.type} -> AGENT (bound completion FAILED)", m.error.message if m.error else "unknown error")


async def test_live_laptop_search_end_to_end(provider):
    search_tool = LoggingSearchLaptopsTool()
    handler = SimulatedUserHandler(use_case="creative", max_budget_usd=1500)
    question_tool = AskUserQuestionTool(handler=handler)

    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a shopping assistant with access to specialized skills "
            "and tools.\n\n{{skills_catalog}}\n\n"
            "When asked to help find a laptop, use the laptop-search skill "
            "and follow its instructions exactly -- use the question tool to "
            "ask the user rather than guessing, and never invent a laptop "
            "search_laptops did not return."
        ),
        skills_path=str(SKILLS_DIR),
        tools=[search_tool, question_tool],
        budget=Budget(max_turns=20, max_invocations=25),
        logger=StdoutLogger(),
    )

    user_prompt = "I'm looking for a new laptop. Can you help me find one?"
    _log("USER -> AGENT (initial request)", user_prompt)

    response = await agent.run(user_prompt)

    # search_laptops/question calls above were logged live, as they happened;
    # this is the skill's own dispatch/bound-completion trace, in the same
    # chronological (execution) order, recapped now that the run is done.
    print("\n=== laptop-search skill trace (in execution order) ===")
    for m in response.messages:
        _log_skill_message(m)
    _log("AGENT -> USER (final response)", response.response or "")

    assert response.status == "success"
    assert response.response

    assert search_tool.calls, "search_laptops was never called"
    assert handler.calls, "the question tool was never used to ask the (simulated) user"

    laptop_search_messages = [m for m in response.messages if m.type == "laptop-search"]
    assert laptop_search_messages, "the laptop-search skill was never dispatched"

    bound = [m for m in laptop_search_messages if m.input is None]
    assert bound, "no schema-bound completion was ever produced for laptop-search"

    # A bound completion may occasionally fail schema validation and get
    # retried by a later `invoke_skill` round (the agent's own repair path,
    # agent.py) -- that's expected, real-model resilience, not a test bug.
    # Only the ones that succeeded need to report a known union variant.
    succeeded = [m for m in bound if m.status == "success"]
    assert succeeded, f"every bound completion failed: {[m.error for m in bound]}"
    for m in succeeded:
        assert m.data["result"]["type"] in {"needs_more_filters", "narrowed_results", "final_pick"}

    final = succeeded[-1]
    assert final.data["result"]["type"] == "final_pick", (
        "conversation did not converge to a final pick within the turn budget: "
        f"{[m.data['result']['type'] for m in succeeded]}"
    )
    # The core "never invent a laptop" guarantee: the final pick must be a
    # real, unaltered catalog entry, not something the model made up. This
    # holds regardless of exactly which budget band the model and the
    # simulated user converged on along the way.
    laptop = final.data["result"]["laptop"]
    catalog_entry = next((l for l in _CATALOG if l["id"] == laptop["id"]), None)
    assert catalog_entry is not None, f"final pick {laptop!r} is not in the catalog"
    assert laptop == catalog_entry
    assert laptop["use_case"] == "creative"
