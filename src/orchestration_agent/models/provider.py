"""Provider-facing data structures: token accounting and completion results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """Token consumption tracking for a single call or accumulated across steps."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None

    def total(self) -> int:
        """Total tokens consumed (excluding cache), the primary figure for rate limiting."""
        return self.input_tokens + self.output_tokens

    def total_with_cache(self) -> int:
        """Total tokens including cache creation/read operations."""
        total = self.input_tokens + self.output_tokens
        if self.cache_creation_tokens:
            total += self.cache_creation_tokens
        if self.cache_read_tokens:
            total += self.cache_read_tokens
        return total


@dataclass
class ToolCall:
    """Represents a tool/skill invocation requested by the LLM."""

    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class CompletionResponse:
    """Response from provider.complete()."""

    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str = "end_turn"
    raw_message: Optional[Dict[str, Any]] = None
    """Provider-native assistant message (e.g. OpenAI's message dict, including its
    own `tool_calls` shape). Agents must replay this verbatim when appending the
    assistant turn to history so any follow-up tool-result messages remain valid
    against the provider's API (see spec deviation #4)."""
