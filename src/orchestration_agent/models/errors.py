"""Custom exception hierarchy shared across the orchestration agent."""


class RetryableError(Exception):
    """Base class for errors that trigger a step-level retry."""


class RateLimitError(RetryableError):
    """Raised when the provider signals a rate limit (HTTP 429)."""


class TemporaryProviderError(RetryableError):
    """Raised on temporary provider errors (HTTP 500, 503, ...)."""


class StructuredOutputValidationError(RetryableError):
    """Raised when the model output doesn't match the requested schema."""


class RateLimitExceededError(Exception):
    """Raised when the external RateLimiterContext rejects a call."""


class SkillNotFoundError(Exception):
    """Raised when invoke_skill references an unknown skill."""


class ToolNotFoundError(Exception):
    """Raised when invoke_tool references an unregistered tool."""


class ToolError(Exception):
    """Raised by a BaseTool implementation on execution failure."""


class ConversationHistoryError(Exception):
    """Base exception for conversation history errors."""


class AgentError(Exception):
    """Raised on unrecoverable agent failures."""
