"""Re-exported for spec-shape compatibility (`orchestration_agent.provider.errors`).

The actual definitions live in `orchestration_agent.models.errors` so the whole
codebase shares one exception hierarchy.
"""

from ..models.errors import (
    RateLimitError,
    RetryableError,
    StructuredOutputValidationError,
    TemporaryProviderError,
)

__all__ = [
    "RateLimitError",
    "RetryableError",
    "StructuredOutputValidationError",
    "TemporaryProviderError",
]
