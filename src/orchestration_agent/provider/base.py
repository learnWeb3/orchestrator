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
        structured_output: Optional[Type[BaseModel]] = None,
        stream: bool = False,
        tools: Optional[List["BaseTool"]] = None,
        **kwargs: Any,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Generate a completion from the LLM.

        Raises:
            RateLimitError: if rate limited (429).
            TemporaryProviderError: on a temporary error (500, 503).
            StructuredOutputValidationError: if output doesn't match `structured_output`.
        """
