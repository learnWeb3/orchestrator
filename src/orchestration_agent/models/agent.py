"""Agent-facing result types: the run's own trace/step records, budget, and
`AgentResponse` (spec section 8)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .provider import TokenUsage
from .skill_output import ErrorDetail, SkillOutput


@dataclass
class StepRecord:
    """Record of a single provider completion, for observability/debugging.

    Not part of the spec's fixed envelope — implementation detail carried on
    `AgentResponse.steps` (section 8).
    """

    step_number: int
    lm_call: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class Budget:
    """Run-level caps (spec section 9): max skill invocations and max model
    turns. Both trigger the same reserved-final-turn termination path
    (section 9, "Termination")."""

    max_invocations: int = 25
    max_turns: int = 15


@dataclass
class AgentResponse:
    """Final result of `Agent.run()`.

    `messages` is the evidence (every skill dispatch and bound completion, in
    execution order); `response` is the answer, validated against the
    caller's `response_schema` where supplied. `steps`, `total_tokens`, and
    `token_usage` are implementation detail beyond the envelope this
    specification governs (section 8) — a conformant consumer reads
    `messages`, `response`, `status`, `error`, and `duration_ms`.
    """

    messages: List[SkillOutput]
    response: Any
    status: str  # "success" | "error" | "partial"
    error: Optional[ErrorDetail] = None
    duration_ms: Optional[float] = None

    steps: List[StepRecord] = field(default_factory=list)
    total_tokens: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
