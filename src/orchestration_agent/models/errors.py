"""Custom exception hierarchy shared across the orchestration agent.

Three distinct recovery mechanisms exist (spec section 10) and must not be
conflated:

1. Transport-level step retry: `RetryableError` subclasses (`RateLimitError`,
   `TemporaryProviderError`). Retried by `Agent._step_with_retry` with backoff,
   invisibly to the model.
2. Output-validation internal repair: `StructuredOutputValidationError` is
   *not* retryable at the transport level — a deterministic content mismatch
   reproduces itself against an unchanged prompt. It is recovered via one
   orchestrator-driven repair (violations fed back into the prompt), never a
   bare backoff-and-retry.
3. Model-level recovery: everything else (`SkillNotFoundError`,
   `ModelRefusalError`, a `TIMEOUT`) is surfaced to the model as an ordinary
   `SkillOutput`; the model decides whether to retry by invoking again.
"""

from typing import Any, Dict, List, Optional


class RetryableError(Exception):
    """Base class for errors that trigger a transport-level step retry."""


class RateLimitError(RetryableError):
    """Raised when the provider signals a rate limit (HTTP 429)."""


class TemporaryProviderError(RetryableError):
    """Raised on temporary provider errors (HTTP 500, 503, ...)."""


class StructuredOutputValidationError(Exception):
    """Raised when generated content doesn't validate against a JSON Schema.

    Deliberately *not* a `RetryableError`: a malformed/invalid model response
    must never be treated as a transport failure (spec section 10, mechanism
    1 vs. 2). Recovery is the orchestrator's own one-shot internal repair
    (section 6), not step-level backoff.
    """

    def __init__(self, message: str, violations: Optional[Any] = None) -> None:
        super().__init__(message)
        self.violations = violations


class ModelRefusalError(Exception):
    """Raised when the provider declines to generate on a schema-bound completion.

    A content refusal, distinct from the provider rejecting the schema itself
    (`SchemaCompilationError`). Maps to `error.code: "MODEL_REFUSAL"`; must
    never be parsed as if it were the payload.
    """


class SchemaCompilationError(Exception):
    """Raised when the provider rejects a structured-output schema itself.

    A compilation/structural error, typically raised before any generation
    happens — distinct from a content refusal. Triggers the fallback
    (prompt-injection) path for that same completion, invisibly to the model.
    """


class RateLimitExceededError(Exception):
    """Raised when the external RateLimiterContext rejects a call."""


class SkillNotFoundError(Exception):
    """Raised when invoke_skill references an unknown or missing skill_name."""


class ToolNotFoundError(Exception):
    """Raised when invoke_tool references an unregistered tool."""


class ToolError(Exception):
    """Raised by a BaseTool implementation on execution failure."""


class ConversationHistoryError(Exception):
    """Base exception for conversation history errors."""


class ResponseValidationError(Exception):
    """Run-level: the final `response` failed `response_schema` validation
    after the one permitted internal repair attempt. Maps to
    `error.code: "RESPONSE_VALIDATION_ERROR"`.
    """

    def __init__(self, message: str, violations: Optional[Any] = None) -> None:
        super().__init__(message)
        self.violations = violations


class BudgetExhaustedError(Exception):
    """Run-level: the invocation/turn budget (or the caller's overall timeout)
    was exhausted and the final, reserved, tools-disabled turn did not
    produce a `response_schema`-valid answer. Maps to
    `error.code: "BUDGET_EXHAUSTED"`.
    """


class AgentError(Exception):
    """Raised on unrecoverable agent failures."""


# Stable, machine-readable error codes (spec section 10 taxonomy). Add to this
# set rather than repurposing an existing code.
ERROR_CODES = frozenset(
    {
        "SKILL_NOT_FOUND",
        "OUTPUT_VALIDATION_ERROR",
        "MODEL_REFUSAL",
        "TIMEOUT",
        "RESPONSE_VALIDATION_ERROR",
        "BUDGET_EXHAUSTED",
        "RATE_LIMIT_EXCEEDED",
        "AGENT_ERROR",
    }
)


def violations_to_list(violations: Any) -> List[Dict[str, Any]]:
    """Best-effort normalisation of a validation failure into the `violations`
    array shape used throughout `SkillOutput.error.details` (`pointer`,
    `keyword`, `expected`, `actual`, `message`)."""
    if violations is None:
        return []
    if isinstance(violations, list):
        return violations
    return [{"pointer": "", "keyword": "", "expected": None, "actual": None, "message": str(violations)}]
