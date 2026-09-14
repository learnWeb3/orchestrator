"""Agent-facing result types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .provider import TokenUsage


@dataclass
class StepRecord:
    """Record of a single agent step, for observability/debugging."""

    step_number: int
    lm_call: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class AgentResponse:
    """Final result of Agent.run()."""

    output: Any
    steps: List[StepRecord]
    total_tokens: int
    token_usage: TokenUsage
    success: bool
    error: Optional[str] = None
