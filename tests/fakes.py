"""Test doubles: a scripted BaseProvider that never hits the network."""

import json
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from orchestration_agent.models.provider import CompletionResponse, TokenUsage, ToolCall
from orchestration_agent.provider.base import BaseProvider


class FakeProvider(BaseProvider):
    """Returns a scripted sequence of CompletionResponse (or raises), one per call.

    `script` items may be a `CompletionResponse`, an `Exception` instance to raise,
    or a callable `(call_index) -> CompletionResponse` for stateful scenarios.
    """

    def __init__(self, script: List[Union[CompletionResponse, Exception, Callable]]):
        super().__init__(model="fake-model", api_key="fake-key")
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []

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
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "structured_output": structured_output,
                "structured_output_name": structured_output_name,
                "tools": tools,
            }
        )
        index = len(self.calls) - 1
        item = self.script[min(index, len(self.script) - 1)]

        if callable(item) and not isinstance(item, CompletionResponse):
            item = item(index)

        if isinstance(item, Exception):
            raise item
        return item


def text_response(content: str, usage: Optional[TokenUsage] = None) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        tool_calls=[],
        token_usage=usage or TokenUsage(input_tokens=10, output_tokens=5),
        stop_reason="stop",
        raw_message={"role": "assistant", "content": content},
    )


def json_response(payload: Dict[str, Any], usage: Optional[TokenUsage] = None) -> CompletionResponse:
    """A CompletionResponse whose content is a JSON-serialized payload, as a
    schema-bound completion would return."""
    return text_response(json.dumps(payload), usage=usage)


def tool_call_response(
    name: str, arguments: Dict[str, Any], call_id: str = "call_1"
) -> CompletionResponse:
    """A CompletionResponse requesting a single tool/skill call, with a realistic
    OpenAI-shaped `raw_message` (assistant message carrying `tool_calls`)."""
    return CompletionResponse(
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments, id=call_id)],
        token_usage=TokenUsage(input_tokens=20, output_tokens=8),
        stop_reason="tool_calls",
        raw_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
    )
