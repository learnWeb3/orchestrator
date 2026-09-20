"""External, executable tool interface (fastMCP-compatible shape)."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from pydantic import BaseModel


class BaseError(BaseModel):
    """Default error response schema for tool execution failures."""

    error: str
    details: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for executable external tools.

    Tools accept structured input (a Pydantic model), execute deterministically,
    and return structured output. Distinct from skills: tools execute code, skills
    hand the LLM markdown instructions.
    """

    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    error_schema: Type[BaseModel] = BaseError

    @abstractmethod
    async def execute(self, input: BaseModel) -> Any:
        """Execute the tool with validated parameters.

        Raise on failure; the Agent catches and formats the error using
        `error_schema`. The return value (a dict or an `output_schema`
        instance) must validate against `output_schema`; the Agent enforces
        this and formats a mismatch using `error_schema` as well.
        """
