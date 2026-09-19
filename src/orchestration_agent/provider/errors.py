"""Re-exported for spec-shape compatibility (`orchestration_agent.provider.errors`).

The actual definitions live in `orchestration_agent.models.errors` so the whole
codebase shares one exception hierarchy.
"""

from ..models.errors import (
    ModelRefusalError,
    RateLimitError,
    RetryableError,
    SchemaCompilationError,
    StructuredOutputValidationError,
    TemporaryProviderError,
)

__all__ = [
    "ModelRefusalError",
    "RateLimitError",
    "RetryableError",
    "SchemaCompilationError",
    "StructuredOutputValidationError",
    "TemporaryProviderError",
]
