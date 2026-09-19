"""OpenAI provider implementation.

Implements `BaseProvider.complete()` against OpenAI's Responses API
(`https://api.openai.com/v1/responses`, `client.responses.create`), with the
following corrections versus the spec's illustrative Chat Completions
pseudocode (see the implementation plan for the full rationale):

1. Token usage is read from the Responses API's `usage` shape
   (`input_tokens`/`output_tokens`/`input_tokens_details.cached_tokens`), not
   Anthropic's or Chat Completions' field names.
2. `gpt-5*`/o-series "reasoning" models reject a custom `temperature` —
   detected and handled here. `reasoning_effort` is sent as the nested
   `reasoning.effort` param the Responses API expects, not a top-level field.
3. All `function_call` output items are captured (not just ones named
   `invoke_skill`), so the dual skill/tool dispatch in `Agent` actually
   receives external tool calls.
4. The raw, provider-native output items are returned on
   `CompletionResponse.raw_message` (wrapped as `{"type": "response_output",
   "output": [...]}`) so `Agent` can replay them verbatim on the next turn —
   the Responses API accepts its own output items back as input items
   directly, which is what makes follow-up `function_call_output` items valid
   against the API (see spec deviation #4).
5. `structured_output` may be a Pydantic model (the caller's `response_schema`)
   or a raw JSON Schema `dict` (a skill's declared `output`, orchestration
   spec section 6) — `structured_output_name` names the schema in the latter
   case. The provider does *not* validate the result against the schema
   itself; that's the orchestrator's job at the boundary (section 6).
6. A content refusal on a schema-bound completion is raised as
   `ModelRefusalError`, distinct from the provider rejecting the schema
   itself before any generation happens (`SchemaCompilationError`) — the
   orchestrator falls back to prompt injection only for the latter.
"""

import json
import re
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
)
from openai import RateLimitError as OpenAIRateLimitError
from pydantic import BaseModel

from ..models.errors import (
    ModelRefusalError,
    RateLimitError,
    SchemaCompilationError,
    TemporaryProviderError,
)
from ..models.provider import CompletionResponse, TokenUsage, ToolCall
from .base import BaseProvider

# gpt-5* and o-series ("o1", "o3", "o4", ...) are reasoning models: they reject a
# non-default `temperature`.
_REASONING_MODEL_RE = re.compile(r"^(gpt-5|o1|o3|o4)")

# "luna" reasoning models reject a request that sets both `reasoning.effort` and
# function tools ("Function tools are not supported with reasoning effort for
# luna models"). We always send at least the `invoke_skill` tool, so for these
# models `reasoning.effort` can never be sent.
_LUNA_MODEL_RE = re.compile(r"luna", re.IGNORECASE)

# Marker `type` used on our own `CompletionResponse.raw_message` wrapper (see
# module docstring #4) — never a real Responses API item type, so it's safe to
# use as a discriminator when converting agent history back into `input` items.
_RAW_MESSAGE_TYPE = "response_output"

# response.status -> the finish_reason vocabulary the rest of this codebase
# (Agent._TERMINAL_STOP_REASONS) already expects from the Chat Completions days.
_STATUS_TO_STOP_REASON = {
    "completed": "stop",
    "cancelled": "stop",
    "failed": "stop",
}
_INCOMPLETE_REASON_TO_STOP_REASON = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


def _is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.match(model))


def _is_luna_model(model: str) -> bool:
    return bool(_LUNA_MODEL_RE.search(model))


def _make_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively rewrite a JSON Schema document for OpenAI's `strict` mode.

    Strict mode requires every object to set `additionalProperties: false` and
    list *all* of its properties (including optional ones) in `required` —
    optional fields stay expressible via a nullable type. Applied unchanged to
    both a Pydantic-derived schema and a skill's declared `output` schema
    (orchestration spec section 4): "The same helper is used, unchanged, for
    a skill's `output` schema."
    """
    node = dict(schema)

    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        node["properties"] = {
            key: _make_strict_json_schema(value) for key, value in node["properties"].items()
        }
        node["required"] = list(node["properties"].keys())

    if "items" in node:
        node["items"] = _make_strict_json_schema(node["items"])

    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            node[key] = [_make_strict_json_schema(sub) for sub in node[key]]

    if "$defs" in node:
        node["$defs"] = {k: _make_strict_json_schema(v) for k, v in node["$defs"].items()}

    return node


def _messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert the agent's Chat-Completions-shaped history into Responses `input` items.

    `messages` is a heterogeneous list built up by `Agent`: plain
    `{"role": ..., "content": ...}` turns, our own `raw_message` wrapper
    (marker type `_RAW_MESSAGE_TYPE`) replayed verbatim from a prior turn, and
    `{"role": "tool", "tool_call_id": ..., "content": ...}` tool replies.
    """
    input_items: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        if message.get("type") == _RAW_MESSAGE_TYPE:
            input_items.extend(message.get("output", []))
        elif message.get("role") == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": message.get("content", ""),
                }
            )
        elif "role" in message:
            input_items.append({"role": message["role"], "content": message.get("content", "")})
        else:
            # Already a Responses-native input item (e.g. hand-built by a caller) —
            # pass it through unchanged.
            input_items.append(message)

    return input_items


def _map_stop_reason(status: Optional[str], incomplete_reason: Optional[str]) -> str:
    if status == "incomplete" and incomplete_reason:
        return _INCOMPLETE_REASON_TO_STOP_REASON.get(incomplete_reason, "incomplete")
    return _STATUS_TO_STOP_REASON.get(status or "", status or "stop")


def _extract_refusal(output_items: List[Dict[str, Any]]) -> Optional[str]:
    """Find a content refusal among `response.output` items, if any.

    Distinct from the provider rejecting the schema itself (that surfaces as
    a `BadRequestError` before any output exists at all, handled separately
    as `SchemaCompilationError`).
    """
    for item in output_items:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "refusal":
                return part.get("refusal") or "The model declined to generate a response."
    return None


class OpenAIProvider(BaseProvider):
    """OpenAI API provider (gpt-4o, gpt-5.x, o-series, ...) via the Responses API."""

    def __init__(self, model: str, api_key: str, **kwargs: Any) -> None:
        super().__init__(model, api_key, **kwargs)
        client_kwargs = dict(kwargs)
        client_kwargs.setdefault("max_retries", 0)  # Agent owns retry/backoff.
        base_url = client_kwargs.pop("base_url", None)
        client_kwargs.pop("api_key", None)
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, **_only_client_kwargs(client_kwargs)
        )

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
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        resolved_model = model or self.model

        tool_list: List[Dict[str, Any]] = [_invoke_skill_tool_schema()]
        if tools:
            tool_list.extend(self._convert_tool_to_openai_schema(t) for t in tools)

        completion_params: Dict[str, Any] = {
            "model": resolved_model,
            "instructions": system_prompt,
            "input": _messages_to_responses_input(messages),
            "tools": tool_list,
            "max_output_tokens": max_tokens,
        }

        if _is_reasoning_model(resolved_model):
            reasoning_effort = kwargs.pop("reasoning_effort", None)
            if reasoning_effort and not (tool_list and _is_luna_model(resolved_model)):
                completion_params["reasoning"] = {"effort": reasoning_effort}
        else:
            completion_params["temperature"] = temperature

        if structured_output is not None:
            if isinstance(structured_output, dict):
                schema_dict = _make_strict_json_schema(structured_output)
                schema_name = structured_output_name or "output"
            else:
                schema_dict = _make_strict_json_schema(structured_output.model_json_schema())
                schema_name = structured_output_name or structured_output.__name__
            completion_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema_dict,
                    "strict": True,
                }
            }

        completion_params.update(kwargs)

        try:
            if stream:
                return self._stream_completion(completion_params)
            return await self._non_stream_completion(completion_params, structured_output)
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit: {e}") from e
        except (InternalServerError, APIConnectionError, APITimeoutError) as e:
            raise TemporaryProviderError(f"OpenAI temporary error: {e}") from e

    async def _non_stream_completion(
        self,
        params: Dict[str, Any],
        structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]],
    ) -> CompletionResponse:
        try:
            response = await self.client.responses.create(**params)
        except BadRequestError as e:
            # A schema-compilation/structural rejection happens before any
            # generation and is distinct from a content refusal below.
            if structured_output is not None:
                raise SchemaCompilationError(f"Schema rejected by provider: {e}") from e
            raise

        content = response.output_text or ""

        tool_calls: List[ToolCall] = []
        output_items: List[Dict[str, Any]] = []
        for item in response.output:
            output_items.append(item.model_dump(exclude_none=True))
            if item.type == "function_call":
                try:
                    arguments = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(ToolCall(name=item.name, arguments=arguments, id=item.call_id))

        if structured_output is not None:
            refusal = _extract_refusal(output_items)
            if refusal is not None:
                raise ModelRefusalError(refusal)

        usage = response.usage
        cache_read_tokens = None
        if usage is not None and usage.input_tokens_details is not None:
            cache_read_tokens = usage.input_tokens_details.cached_tokens

        token_usage = TokenUsage(
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cache_creation_tokens=None,  # Not applicable to OpenAI's Responses API.
            cache_read_tokens=cache_read_tokens,
        )

        incomplete_reason = (
            response.incomplete_details.reason if response.incomplete_details else None
        )

        stop_reason = _map_stop_reason(response.status, incomplete_reason)
        if tool_calls and stop_reason == "stop":
            # The Responses API reports "completed" status even when the model's
            # turn ended on a function call, with no separate "requires action"
            # status. Report it as "tool_calls" so callers (e.g. Agent.run's
            # step loop) don't mistake a pending tool call for a final answer.
            stop_reason = "tool_calls"

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            stop_reason=stop_reason,
            raw_message={"type": _RAW_MESSAGE_TYPE, "output": output_items},
        )

    async def _stream_completion(
        self,
        params: Dict[str, Any],
    ) -> AsyncIterator[CompletionResponse]:
        stream = await self.client.responses.create(**params, stream=True)

        accumulated_content = ""
        token_usage = TokenUsage()

        async for event in stream:
            if event.type == "response.output_text.delta":
                accumulated_content += event.delta
                yield CompletionResponse(
                    content=accumulated_content,
                    token_usage=token_usage,
                    stop_reason="streaming",
                )
            elif event.type == "response.completed":
                usage = event.response.usage
                if usage is not None:
                    token_usage = TokenUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )

    def _convert_tool_to_openai_schema(self, tool: Any) -> Dict[str, Any]:
        """Convert a BaseTool to a Responses API function-tool schema (provider-as-gateway)."""
        input_schema = tool.input_schema.model_json_schema()
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": input_schema.get("properties", {}),
                "required": input_schema.get("required", []),
            },
        }


def _invoke_skill_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "invoke_skill",
        "description": "Invoke a skill from the available skill catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to invoke",
                }
            },
            "required": ["skill_name"],
        },
    }


_CLIENT_KWARG_ALLOWLIST = {
    "max_retries",
    "timeout",
    "organization",
    "project",
    "default_headers",
    "default_query",
    "http_client",
}


def _only_client_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop provider-config kwargs that aren't valid `AsyncOpenAI(...)` constructor args."""
    return {k: v for k, v in kwargs.items() if k in _CLIENT_KWARG_ALLOWLIST}
