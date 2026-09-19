"""Provider abstraction: the contract every LLM backend must fulfill."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..models.provider import CompletionResponse

if TYPE_CHECKING:
    from ..tools.base import BaseTool


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, api_key: str, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
        structured_output_name: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List["BaseTool"]] = None,
        **kwargs: Any,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Generate a completion from the LLM.

        `structured_output` binds the provider's native structured-output
        mechanism, and may be either a Pydantic model (the caller's
        `response_schema`) or a raw JSON Schema `dict` (a skill's declared
        `output`, section 6) — `structured_output_name` names the schema in
        the latter case. The orchestrator, not the provider, validates the
        result against the schema (spec section 6, "Validation at the
        boundary").

        Raises:
            RateLimitError: if rate limited (429).
            TemporaryProviderError: on a temporary error (500, 503).
            ModelRefusalError: on a content refusal on a schema-bound
                completion — never a malformed-output condition.
            SchemaCompilationError: if the provider rejects `structured_output`
                itself (a compilation/structural error), distinct from a
                content refusal.
        """
