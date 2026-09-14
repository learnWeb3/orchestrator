"""BaseAgent / Agent: the orchestration loop.

Two corrections vs. the spec's pseudocode, on top of the provider-level fixes
documented in `provider/openai_provider.py`:

- **Dispatch by real function name.** The spec's `Agent.run()` (history-integrated
  variant) checks `tool_call.name == "invoke_tool"` and expects a wrapped
  `{"tool_name": ..., "tool_parameters": {...}}` payload. But
  `OpenAIProvider._convert_tool_to_openai_schema` registers each `BaseTool` as its
  *own* named function (matching the spec's own "provider acts as gateway" design
  and its `invoke_tool(tool_name="fetch_user", ...)` usage examples). So dispatch
  here is: `invoke_skill` -> skill lookup, any other registered tool name -> that
  tool directly. This is what actually round-trips against the real API.
- **Every tool_call gets a reply.** OpenAI rejects the next turn if any
  `tool_calls` entry from the assistant message lacks a matching `tool` role
  reply, so unknown/failed calls still get an error `tool` message instead of
  being silently skipped.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from pydantic import BaseModel

from .conversation.base import ConversationHistory
from .conversation.in_memory import InMemoryHistory
from .logging.base import Logger
from .logging.implementations import NoOpLogger
from .models.agent import AgentResponse, StepRecord
from .models.errors import (
    RateLimitExceededError,
    RetryableError,
    SkillNotFoundError,
    StructuredOutputValidationError,
    ToolNotFoundError,
)
from .models.provider import CompletionResponse, TokenUsage, ToolCall
from .models.rate_limiter import RateLimiterContext
from .provider.base import BaseProvider
from .skills.loader import SkillLoader
from .skills.registry import SkillRegistry
from .tools.base import BaseTool
from .utils.prompt import inject_skills_into_prompt
from .utils.retry import default_exponential_backoff

# finish_reason values that mean "the model is done, no more tool calls to process".
_TERMINAL_STOP_REASONS = {"stop", "length", "content_filter"}


@dataclass
class StepResult:
    content: str
    tool_calls: List[ToolCall]
    token_usage: TokenUsage
    stop_reason: str
    step_record: StepRecord
    raw_message: Optional[Dict[str, Any]] = None


class BaseAgent:
    """Base class holding shared construction/loading logic for orchestration agents."""

    def __init__(
        self,
        provider: BaseProvider,
        system_prompt: str,
        skills_path: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        max_steps: int = 10,
        max_retries: int = 3,
        output_schema: Optional[Type[BaseModel]] = None,
        streaming: bool = False,
        backoff_strategy: Optional[Callable[[int], float]] = None,
        logger: Optional[Logger] = None,
        rate_limiter_context: Optional[RateLimiterContext] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.skills_path = skills_path
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in (tools or [])}
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.output_schema = output_schema
        self.streaming = streaming
        self.backoff_strategy = backoff_strategy or default_exponential_backoff
        self.logger = logger or NoOpLogger()
        self.rate_limiter_context = rate_limiter_context
        self.session_id = session_id or str(uuid4())

        self.skills_registry: SkillRegistry = SkillRegistry()
        self._load_skills()

    def _load_skills(self) -> None:
        if not self.skills_path:
            return

        loader = SkillLoader()
        for skill_metadata in loader.load_skills_from_path(self.skills_path):
            self.skills_registry.register(skill_metadata)

        self.system_prompt = inject_skills_into_prompt(self.system_prompt, self.skills_registry)


class Agent(BaseAgent):
    """Production agent implementation: one-shot `run()` over a step loop."""

    def __init__(
        self,
        provider: BaseProvider,
        system_prompt: str,
        conversation_history: Optional[ConversationHistory] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        on_message_added: Optional[Callable[..., Any]] = None,
        on_history_cleared: Optional[Callable[[], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, system_prompt, **kwargs)
        self.conversation_history = conversation_history or InMemoryHistory()
        self.history_formatter = history_formatter
        self.on_message_added = on_message_added
        self.on_history_cleared = on_history_cleared

    # -- Public API ---------------------------------------------------------

    async def run(self, user_input: str) -> AgentResponse:
        self.logger.info(f"Agent.run() starting with input: {user_input[:100]}...")

        history_messages = await self.conversation_history.get_all(session_id=self.session_id)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *history_messages,
            {"role": "user", "content": user_input},
        ]
        await self._add_to_history("user", user_input)

        cumulative_token_usage = TokenUsage()
        steps: List[StepRecord] = []
        step_count = 0

        try:
            while step_count < self.max_steps:
                self.logger.debug(f"Step {step_count + 1}/{self.max_steps}")

                step_result = await self._step_with_retry(messages, step_count)
                step_count += 1

                cumulative_token_usage.input_tokens += step_result.token_usage.input_tokens
                cumulative_token_usage.output_tokens += step_result.token_usage.output_tokens
                if step_result.token_usage.cache_creation_tokens:
                    cumulative_token_usage.cache_creation_tokens = (
                        step_result.token_usage.cache_creation_tokens
                    )
                if step_result.token_usage.cache_read_tokens:
                    cumulative_token_usage.cache_read_tokens = (
                        step_result.token_usage.cache_read_tokens
                    )

                steps.append(step_result.step_record)

                if self.rate_limiter_context:
                    allowed = self.rate_limiter_context.check_and_consume(
                        tokens=cumulative_token_usage.total()
                    )
                    if not allowed:
                        self.logger.warning("Rate limit exceeded")
                        raise RateLimitExceededError(cumulative_token_usage.total())

                if not step_result.tool_calls or step_result.stop_reason in _TERMINAL_STOP_REASONS:
                    return await self._finalize(step_result, steps, cumulative_token_usage)

                await self._dispatch_tool_calls(messages, step_result)

            return AgentResponse(
                output="Max steps reached without final response",
                steps=steps,
                total_tokens=cumulative_token_usage.total(),
                token_usage=cumulative_token_usage,
                success=False,
                error="max_steps_exceeded",
            )

        except Exception as e:  # noqa: BLE001 - top-level guard, per spec
            self.logger.error(f"Agent.run() failed: {e}", exception=e)
            return AgentResponse(
                output=None,
                steps=steps,
                total_tokens=cumulative_token_usage.total(),
                token_usage=cumulative_token_usage,
                success=False,
                error=str(e),
            )

    async def invoke_skill(self, skill_name: str) -> str:
        """Return the full skill markdown (instructions) for `skill_name`."""
        skill_metadata = self.skills_registry.get_by_name(skill_name)
        if not skill_metadata:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")

        self.logger.info(f"Invoking skill: {skill_name}")
        return skill_metadata.content

    async def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Validate parameters, execute the tool, and format errors via its error_schema."""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")

        self.logger.info(f"Invoking tool: {tool_name} with parameters: {parameters}")

        try:
            validated_input = tool.input_schema(**parameters)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Tool {tool_name} parameter validation failed: {e}")
            return tool.error_schema(error="Invalid parameters", details=str(e)).model_dump()

        try:
            result = await tool.execute(validated_input)
            return result.model_dump() if isinstance(result, BaseModel) else result
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Tool {tool_name} execution failed: {e}")
            return tool.error_schema(
                error=str(e), details=f"Execution failed in {tool_name}"
            ).model_dump()

    # -- Internals ------------------------------------------------------------

    async def _finalize(
        self,
        step_result: "StepResult",
        steps: List[StepRecord],
        cumulative_token_usage: TokenUsage,
    ) -> AgentResponse:
        output: Any = step_result.content
        await self._add_to_history("assistant", step_result.content)

        if self.output_schema:
            try:
                output = self.output_schema.model_validate_json(step_result.content)
            except Exception as e:
                self.logger.error(f"Output validation failed: {e}")
                raise StructuredOutputValidationError(str(e)) from e

        return AgentResponse(
            output=output,
            steps=steps,
            total_tokens=cumulative_token_usage.total(),
            token_usage=cumulative_token_usage,
            success=True,
        )

    async def _dispatch_tool_calls(
        self, messages: List[Dict[str, Any]], step_result: "StepResult"
    ) -> None:
        # Replay the provider's own assistant message (with its tool_calls) verbatim —
        # required so the `tool` messages below are valid on the next API call.
        messages.append(
            step_result.raw_message or {"role": "assistant", "content": step_result.content}
        )

        skills_invoked: List[Dict[str, Any]] = []
        tools_invoked: List[Dict[str, Any]] = []

        for tool_call in step_result.tool_calls:
            call_id = tool_call.id or tool_call.name

            if tool_call.name == "invoke_skill":
                skill_name = tool_call.arguments.get("skill_name")
                if not skill_name:
                    messages.append(
                        _tool_message(call_id, {"error": "invoke_skill requires skill_name"})
                    )
                    continue
                try:
                    skill_content = await self.invoke_skill(skill_name)
                    messages.append(_tool_message(call_id, skill_content, raw=True))
                    skills_invoked.append({"name": skill_name})
                except SkillNotFoundError as e:
                    self.logger.error(f"Skill not found: {skill_name}")
                    messages.append(_tool_message(call_id, {"error": str(e)}))
                except Exception as e:  # noqa: BLE001
                    self.logger.error(f"Skill invocation failed: {e}")
                    messages.append(
                        _tool_message(call_id, {"error": f"Error invoking skill: {e}"})
                    )

            elif tool_call.name in self.tools:
                try:
                    result = await self.invoke_tool(tool_call.name, tool_call.arguments)
                    messages.append(_tool_message(call_id, result))
                    tools_invoked.append(
                        {
                            "name": tool_call.name,
                            "parameters": tool_call.arguments,
                            "result": result,
                        }
                    )
                except ToolNotFoundError as e:
                    self.logger.error(f"Tool not found: {tool_call.name}")
                    messages.append(_tool_message(call_id, {"error": str(e)}))
                except Exception as e:  # noqa: BLE001
                    self.logger.error(f"Tool invocation failed: {e}")
                    messages.append(
                        _tool_message(call_id, {"error": f"Error invoking tool: {e}"})
                    )

            else:
                self.logger.warning(f"Unknown tool: {tool_call.name}")
                messages.append(
                    _tool_message(call_id, {"error": f"Unknown tool '{tool_call.name}'"})
                )

        await self._add_to_history(
            "assistant",
            step_result.content,
            skills_invoked=skills_invoked or None,
            tools_invoked=tools_invoked or None,
        )

    async def _step_with_retry(
        self, messages: List[Dict[str, Any]], step_count: int
    ) -> "StepResult":
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Step attempt {attempt + 1}/{self.max_retries + 1}")
                return await self._step(messages, step_count)
            except RetryableError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.backoff_strategy(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
                    raise

        assert last_error is not None
        raise last_error

    async def _step(self, messages: List[Dict[str, Any]], step_count: int = 0) -> "StepResult":
        completion: CompletionResponse = await self.provider.complete(
            messages=messages,
            system_prompt=self.system_prompt,
            model=self.provider.model,
            temperature=0.7,
            max_tokens=2048,
            structured_output=self.output_schema,
            stream=False,  # Tool-call parsing over a stream isn't supported in v1.
            tools=list(self.tools.values()) if self.tools else None,
        )

        step_record = StepRecord(
            step_number=step_count,
            lm_call={
                "model": self.provider.model,
                "message_count": len(messages),
                "stop_reason": completion.stop_reason,
            },
            tool_calls=[{"name": tc.name, "arguments": tc.arguments} for tc in completion.tool_calls],
            token_usage=completion.token_usage,
        )

        return StepResult(
            content=completion.content,
            tool_calls=completion.tool_calls,
            token_usage=completion.token_usage,
            stop_reason=completion.stop_reason,
            step_record=step_record,
            raw_message=completion.raw_message,
        )

    async def _add_to_history(
        self,
        role: str,
        content: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
        tools_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if tools_invoked:
            self.logger.debug(f"Tools invoked this turn: {tools_invoked}")

        await self.conversation_history.add_message(
            role=role,
            content=content,
            session_id=self.session_id,
            skills_invoked=skills_invoked,
        )

        if self.on_message_added:
            result = self.on_message_added(role, content, skills_invoked)
            if asyncio.iscoroutine(result):
                await result

    async def _clear_history(self) -> None:
        await self.conversation_history.clear(session_id=self.session_id)
        if self.on_history_cleared:
            result = self.on_history_cleared()
            if asyncio.iscoroutine(result):
                await result


def _tool_message(tool_call_id: str, content: Any, raw: bool = False) -> Dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content if raw else json.dumps(content),
    }
