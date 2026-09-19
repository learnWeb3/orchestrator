"""BaseAgent / Agent: the model-driven orchestration loop (orchestration-agent-spec.md).

Every skill invocation goes through the single `invoke_skill(skill_name)` tool
and is recorded as one `SkillOutput` (dispatch). Where a skill declares
`output`, the completion immediately following a successful dispatch is bound
to that schema with tools disabled, producing a second `SkillOutput` (the
bound completion) — section 6. The final answer, `response`, is written by
the model and validated against the caller's `output_schema` (the
`response_schema`) before `AgentResponse` is returned — section 8.

Two corrections vs. the spec's own illustrative pseudocode, on top of the
provider-level fixes documented in `provider/openai_provider.py`:

- **Dispatch by real function name.** `OpenAIProvider._convert_tool_to_openai_schema`
  registers each `BaseTool` as its *own* named function. So dispatch here is:
  `invoke_skill` -> skill lookup, any other registered tool name -> that tool
  directly.
- **Every tool_call gets a reply.** OpenAI rejects the next turn if any
  `tool_calls` entry from the assistant message lacks a matching `tool` role
  reply, so unknown/failed calls still get an error `tool` message instead of
  being silently skipped (spec section 9, "Tool-call protocol").
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from uuid import uuid4

import jsonschema
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from .conversation.base import ConversationHistory
from .conversation.in_memory import InMemoryHistory
from .logging.base import Logger
from .logging.implementations import NoOpLogger
from .models.agent import AgentResponse, Budget, StepRecord
from .models.errors import (
    ModelRefusalError,
    RateLimitExceededError,
    RetryableError,
    SchemaCompilationError,
    SkillNotFoundError,
    ToolNotFoundError,
    violations_to_list,
)
from .models.provider import CompletionResponse, TokenUsage, ToolCall
from .models.rate_limiter import RateLimiterContext
from .models.skill_output import ErrorDetail, SkillOutput
from .provider.base import BaseProvider
from .skills.loader import SkillLoader
from .skills.registry import SkillRegistry
from .tools.base import BaseTool
from .utils.prompt import inject_skills_into_prompt
from .utils.retry import default_exponential_backoff

# finish_reason values that mean "the model is done, no more tool calls to process".
_TERMINAL_STOP_REASONS = {"stop", "length", "content_filter"}

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


@dataclass
class _RunState:
    """One `Agent.run()` call's in-flight state.

    Scoped to a single run and discarded once it returns (spec section 8,
    "Run state versus conversation history") — never stored on `self`, so it
    can't leak between runs or bleed into `ConversationHistory`, the
    persistent, longer-lived store.
    """

    run_start: float
    provider_messages: List[Dict[str, Any]]
    trace: List[SkillOutput] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


class BaseAgent:
    """Base class holding shared construction/loading logic for orchestration agents."""

    def __init__(
        self,
        provider: BaseProvider,
        system_prompt: str,
        skills_path: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        budget: Optional[Budget] = None,
        max_retries: int = 3,
        output_schema: Optional[Type[BaseModel]] = None,
        backoff_strategy: Optional[Callable[[int], float]] = None,
        logger: Optional[Logger] = None,
        rate_limiter_context: Optional[RateLimiterContext] = None,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.skills_path = skills_path
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in (tools or [])}
        self.budget = budget or Budget()
        self.max_retries = max_retries
        self.output_schema = output_schema
        self.backoff_strategy = backoff_strategy or default_exponential_backoff
        self.logger = logger or NoOpLogger()
        self.rate_limiter_context = rate_limiter_context
        self.session_id = session_id or str(uuid4())
        self.timeout_seconds = timeout_seconds

        self.skills_registry: SkillRegistry = SkillRegistry()
        self._load_skills()

    def _load_skills(self) -> None:
        if not self.skills_path:
            return

        loader = SkillLoader()
        for skill_metadata in loader.load_skills_from_path(self.skills_path):
            self.skills_registry.register(skill_metadata)

        # `output` is never rendered into the catalogue or system prompt — only
        # `name`/`description` are published (spec sections 1, 9).
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
        provider_messages: List[Dict[str, Any]] = [
            *history_messages,
            {"role": "user", "content": user_input},
        ]
        await self._add_to_history("user", user_input)

        run_state = _RunState(run_start=time.monotonic(), provider_messages=provider_messages)
        turn_count = 0
        invocation_count = 0
        # Reserve one turn for the final, tools-disabled answer (spec section 10,
        # "Never spend the whole budget on skill invocations and leave nothing to
        # answer with").
        effective_max_turns = max(self.budget.max_turns - 1, 0)

        try:
            while turn_count < effective_max_turns and invocation_count < self.budget.max_invocations:
                completion = await self._step_with_retry(run_state, turn_count)
                turn_count += 1

                if not completion.tool_calls or completion.stop_reason in _TERMINAL_STOP_REASONS:
                    return await self._finalize(run_state, completion)

                invoked, pending_contract = await self._dispatch_round(
                    run_state, completion, turn_count
                )
                invocation_count += invoked

                if pending_contract:
                    bound_output = await self._bind_skill_output(
                        run_state, pending_contract[0], pending_contract[1], f"turn:{turn_count}:bound"
                    )
                    run_state.trace.append(bound_output)

            # Invocation/turn budget exhausted: same termination path as an
            # overall timeout (spec section 9, "Termination").
            return await self._final_turn_and_finish(run_state)

        except asyncio.TimeoutError:
            return await self._final_turn_and_finish(run_state)
        except RateLimitExceededError as e:
            self.logger.warning(f"Rate limit exceeded: {e}")
            return self._response(
                run_state,
                response=None,
                status="error",
                error=ErrorDetail(code="RATE_LIMIT_EXCEEDED", message=str(e)),
            )
        except Exception as e:  # noqa: BLE001 - top-level guard, per spec
            self.logger.error(f"Agent.run() failed: {e}", exception=e)
            return self._response(
                run_state,
                response=None,
                status="error",
                error=ErrorDetail(code="AGENT_ERROR", message=str(e)),
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

    # -- Skill dispatch (spec section 5) --------------------------------------

    async def _dispatch_skill(self, tool_call: ToolCall) -> Tuple[SkillOutput, Any]:
        """Resolve and dispatch one `invoke_skill` call.

        Returns the dispatch `SkillOutput` plus the content to hand back to
        the provider as this call's tool result (raw markdown on success, a
        JSON error object otherwise).
        """
        start = time.monotonic()
        skill_name = (tool_call.arguments or {}).get("skill_name")

        if not skill_name:
            message = "invoke_skill requires skill_name"
            out = SkillOutput(
                type="invoke_skill",
                status="error",
                data=None,
                error=ErrorDetail(code="SKILL_NOT_FOUND", message=message),
                duration_ms=_elapsed_ms(start),
            )
            return out, json.dumps({"error": message})

        try:
            content = await self.invoke_skill(skill_name)
        except SkillNotFoundError as e:
            out = SkillOutput(
                type=skill_name,
                status="error",
                data=None,
                input={"skill_name": skill_name},
                error=ErrorDetail(code="SKILL_NOT_FOUND", message=str(e)),
                duration_ms=_elapsed_ms(start),
            )
            return out, json.dumps({"error": str(e)})

        out = SkillOutput(
            type=skill_name,
            status="success",
            data=content,
            input={"skill_name": skill_name},
            duration_ms=_elapsed_ms(start),
        )
        return out, content

    async def _dispatch_round(
        self, run_state: _RunState, completion: CompletionResponse, turn_count: int
    ) -> Tuple[int, Optional[Tuple[str, Any]]]:
        """Process every `tool_calls` entry from one model turn.

        Replays the provider's own assistant message once for the whole
        round (never once per call), addresses each reply to its own call
        ID, and gives every entry a reply — none silently dropped (spec
        section 9, "Tool-call protocol").

        Returns the number of `invoke_skill` calls processed (for the
        invocation budget) and the last output-declaring skill dispatched
        this round, if any (only the last binds the next completion, spec
        section 6).
        """
        run_state.provider_messages.append(
            completion.raw_message or {"role": "assistant", "content": completion.content}
        )

        invoked = 0
        pending_contract: Optional[Tuple[str, Any]] = None
        skills_invoked_for_history: List[Dict[str, Any]] = []
        tools_invoked_for_history: List[Dict[str, Any]] = []

        for tool_call in completion.tool_calls:
            call_id = tool_call.id or tool_call.name

            if tool_call.name == "invoke_skill":
                invoked += 1
                dispatch_output, tool_content = await self._dispatch_skill(tool_call)
                run_state.trace.append(dispatch_output)
                raw = dispatch_output.status == "success"
                run_state.provider_messages.append(_tool_message(call_id, tool_content, raw=raw))

                if dispatch_output.status == "success":
                    skills_invoked_for_history.append({"name": dispatch_output.type})
                    skill = self.skills_registry.get_by_name(dispatch_output.type)
                    if skill is not None and skill.output is not None:
                        pending_contract = (skill.name, skill.output)  # last dispatch wins

            elif tool_call.name in self.tools:
                try:
                    result = await self.invoke_tool(tool_call.name, tool_call.arguments)
                    run_state.provider_messages.append(_tool_message(call_id, result))
                    tools_invoked_for_history.append(
                        {
                            "name": tool_call.name,
                            "parameters": tool_call.arguments,
                            "result": result,
                        }
                    )
                except ToolNotFoundError as e:
                    self.logger.error(f"Tool not found: {tool_call.name}")
                    run_state.provider_messages.append(_tool_message(call_id, {"error": str(e)}))
                except Exception as e:  # noqa: BLE001
                    self.logger.error(f"Tool invocation failed: {e}")
                    run_state.provider_messages.append(
                        _tool_message(call_id, {"error": f"Error invoking tool: {e}"})
                    )

            else:
                self.logger.warning(f"Unknown tool: {tool_call.name}")
                run_state.provider_messages.append(
                    _tool_message(call_id, {"error": f"Unknown tool '{tool_call.name}'"})
                )

        await self._add_to_history(
            "assistant",
            completion.content,
            skills_invoked=skills_invoked_for_history or None,
            tools_invoked=tools_invoked_for_history or None,
        )

        return invoked, pending_contract

    # -- Output enforcement (spec section 6) -----------------------------------

    async def _bind_skill_output(
        self, run_state: _RunState, skill_name: str, schema: Union[bool, Dict[str, Any]], step_label: str
    ) -> SkillOutput:
        """Issue the schema-bound, tools-disabled completion following a skill
        dispatch, with at most one internal repair on validation failure. Both
        attempts count toward this single `SkillOutput`'s `duration_ms`."""
        start = time.monotonic()

        try:
            payload, violations = await self._bound_completion(
                run_state, run_state.provider_messages, schema, skill_name, step_label
            )
        except ModelRefusalError as e:
            return SkillOutput(
                type=skill_name,
                status="error",
                data=None,
                error=ErrorDetail(code="MODEL_REFUSAL", message=str(e)),
                duration_ms=_elapsed_ms(start),
            )
        except asyncio.TimeoutError:
            return SkillOutput(
                type=skill_name,
                status="error",
                data=None,
                error=ErrorDetail(code="TIMEOUT", message="Bound completion timed out"),
                duration_ms=_elapsed_ms(start),
            )

        if violations is None:
            return SkillOutput(
                type=skill_name, status="success", data=payload, duration_ms=_elapsed_ms(start)
            )

        repair_messages = run_state.provider_messages + [_violation_feedback_message(violations)]
        try:
            payload2, violations2 = await self._bound_completion(
                run_state, repair_messages, schema, skill_name, f"{step_label}:repair"
            )
        except ModelRefusalError as e:
            return SkillOutput(
                type=skill_name,
                status="error",
                data=None,
                error=ErrorDetail(code="MODEL_REFUSAL", message=str(e)),
                duration_ms=_elapsed_ms(start),
            )
        except asyncio.TimeoutError:
            return SkillOutput(
                type=skill_name,
                status="error",
                data=None,
                error=ErrorDetail(code="TIMEOUT", message="Bound completion timed out"),
                duration_ms=_elapsed_ms(start),
            )

        if violations2 is None:
            return SkillOutput(
                type=skill_name, status="success", data=payload2, duration_ms=_elapsed_ms(start)
            )

        return SkillOutput(
            type=skill_name,
            status="error",
            data=None,
            error=ErrorDetail(
                code="OUTPUT_VALIDATION_ERROR",
                message="Output failed schema validation after one repair attempt",
                details={"violations": violations_to_list(violations2)},
            ),
            duration_ms=_elapsed_ms(start),
        )

    async def _bound_completion(
        self,
        run_state: _RunState,
        messages: List[Dict[str, Any]],
        schema: Union[bool, Dict[str, Any], Type[BaseModel]],
        schema_name: str,
        step_label: str,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """One tools-disabled, schema-bound completion: native path first,
        falling back to prompt injection only if the provider rejects the
        schema itself (spec section 6). Returns `(payload, violations)`."""
        try:
            completion = await self._call_provider(
                run_state,
                messages,
                tools=None,
                structured_output=schema,
                structured_output_name=schema_name,
                step_label=step_label,
            )
        except SchemaCompilationError:
            completion = await self._fallback_completion(run_state, messages, schema, step_label)

        return _validate_payload(completion.content, schema)

    async def _fallback_completion(
        self,
        run_state: _RunState,
        messages: List[Dict[str, Any]],
        schema: Union[bool, Dict[str, Any], Type[BaseModel]],
        step_label: str,
    ) -> CompletionResponse:
        schema_json = schema if isinstance(schema, (dict, bool)) else schema.model_json_schema()
        instruction = {
            "role": "user",
            "content": (
                "Respond with only a single JSON object conforming exactly to this "
                f"JSON Schema (no prose, no markdown fences):\n{json.dumps(schema_json)}"
            ),
        }
        completion = await self._call_provider(
            run_state,
            messages + [instruction],
            tools=None,
            structured_output=None,
            step_label=f"{step_label}:fallback",
        )
        completion.content = _strip_fenced_code_block(completion.content)
        return completion

    # -- Final response (spec section 8) ---------------------------------------

    async def _finalize(self, run_state: _RunState, completion: CompletionResponse) -> AgentResponse:
        await self._add_to_history("assistant", completion.content)

        if not self.output_schema:
            return self._response(run_state, response=completion.content, status="success")

        payload, violations = _validate_payload(completion.content, self.output_schema)
        if violations is None:
            return self._response(run_state, response=payload, status="success")

        repair_messages = run_state.provider_messages + [
            {"role": "assistant", "content": completion.content},
            _violation_feedback_message(violations),
        ]
        try:
            payload2, violations2 = await self._bound_completion(
                run_state, repair_messages, self.output_schema, "response", "response:repair"
            )
        except (ModelRefusalError, SchemaCompilationError, asyncio.TimeoutError) as e:
            return self._response(
                run_state,
                response=None,
                status="error",
                error=ErrorDetail(code="RESPONSE_VALIDATION_ERROR", message=str(e)),
            )

        if violations2 is None:
            return self._response(run_state, response=payload2, status="success")

        return self._response(
            run_state,
            response=None,
            status="error",
            error=ErrorDetail(
                code="RESPONSE_VALIDATION_ERROR",
                message="Final response failed schema validation after one repair attempt",
                details={"violations": violations_to_list(violations2)},
            ),
        )

    async def _final_turn_and_finish(self, run_state: _RunState) -> AgentResponse:
        """The one reserved, tools-disabled turn given when the budget or the
        caller's overall timeout is exhausted (spec section 9, "Termination")."""
        try:
            completion = await self._call_provider(
                run_state,
                run_state.provider_messages,
                tools=None,
                structured_output=self.output_schema,
                step_label="final",
            )
        except Exception as e:  # noqa: BLE001 - any failure here is budget exhaustion
            return self._response(
                run_state,
                response=None,
                status="error",
                error=ErrorDetail(
                    code="BUDGET_EXHAUSTED",
                    message=f"Budget/timeout exhausted before a response could be produced: {e}",
                ),
            )

        if self.output_schema:
            payload, violations = _validate_payload(completion.content, self.output_schema)
            if violations is not None:
                return self._response(
                    run_state,
                    response=None,
                    status="error",
                    error=ErrorDetail(
                        code="BUDGET_EXHAUSTED",
                        message="Final reserved turn did not produce a schema-valid response",
                        details={"violations": violations_to_list(violations)},
                    ),
                )
        else:
            payload = completion.content

        await self._add_to_history("assistant", completion.content)
        return self._response(run_state, response=payload, status="partial")

    def _response(
        self,
        run_state: _RunState,
        response: Any,
        status: str,
        error: Optional[ErrorDetail] = None,
    ) -> AgentResponse:
        return AgentResponse(
            messages=run_state.trace,
            response=response,
            status=status,
            error=error,
            duration_ms=_elapsed_ms(run_state.run_start),
            steps=run_state.steps,
            total_tokens=run_state.token_usage.total(),
            token_usage=run_state.token_usage,
        )

    # -- Provider call plumbing -------------------------------------------------

    async def _step_with_retry(self, run_state: _RunState, turn_count: int) -> CompletionResponse:
        """Transport-level retry only (rate limit / transient 5xx) — never for a
        malformed or invalid model response (spec section 10, mechanism 1)."""
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Step attempt {attempt + 1}/{self.max_retries + 1}")
                return await self._call_provider(
                    run_state,
                    run_state.provider_messages,
                    tools=list(self.tools.values()) if self.tools else None,
                    structured_output=self.output_schema,
                    step_label=f"turn:{turn_count}",
                )
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

    def _remaining_timeout(self, run_start: float) -> Optional[float]:
        if self.timeout_seconds is None:
            return None
        return self.timeout_seconds - (time.monotonic() - run_start)

    async def _call_provider(
        self,
        run_state: _RunState,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[BaseTool]],
        structured_output: Optional[Union[Type[BaseModel], Dict[str, Any], bool]] = None,
        structured_output_name: Optional[str] = None,
        step_label: str = "step",
    ) -> CompletionResponse:
        """The single choke point for every provider call this agent makes:
        overall-timeout enforcement, reserve-then-reconcile rate limiting
        (spec section 10), cumulative token accounting, and step recording."""
        remaining = self._remaining_timeout(run_state.run_start)
        if remaining is not None and remaining <= 0:
            raise asyncio.TimeoutError("Agent run timeout exceeded")

        reserved: Optional[int] = None
        if self.rate_limiter_context:
            # Reserve an upper-bound estimate *before* dispatching, rather than
            # checking only after the spend has already landed (spec section 10,
            # "Rate limiting").
            estimate = _estimate_tokens(messages) + 2048
            if not self.rate_limiter_context.check_and_consume(tokens=estimate):
                raise RateLimitExceededError(estimate)
            reserved = estimate

        coro = self.provider.complete(
            messages=messages,
            system_prompt=self.system_prompt,
            model=self.provider.model,
            temperature=0.7,
            max_tokens=2048,
            structured_output=structured_output,
            structured_output_name=structured_output_name,
            stream=False,
            tools=tools,
        )

        try:
            completion = await (
                asyncio.wait_for(coro, timeout=remaining) if remaining is not None else coro
            )
        except Exception:
            if reserved is not None:
                self.rate_limiter_context.release(reserved)
            raise

        # Record what actually happened before any reconciliation decision below —
        # the call already landed and spent real tokens regardless of how the
        # reservation nets out, so observability must not silently drop it.
        # Cumulative accounting is a separate, read-only running total — never
        # the value passed to `check_and_consume` (spec section 10, "Token
        # accounting").
        usage = completion.token_usage
        run_state.token_usage.input_tokens += usage.input_tokens
        run_state.token_usage.output_tokens += usage.output_tokens
        if usage.cache_creation_tokens:
            run_state.token_usage.cache_creation_tokens = usage.cache_creation_tokens
        if usage.cache_read_tokens:
            run_state.token_usage.cache_read_tokens = usage.cache_read_tokens

        run_state.steps.append(
            StepRecord(
                step_number=len(run_state.steps),
                lm_call={
                    "model": self.provider.model,
                    "message_count": len(messages),
                    "stop_reason": completion.stop_reason,
                    "label": step_label,
                },
                tool_calls=[
                    {"name": tc.name, "arguments": tc.arguments} for tc in completion.tool_calls
                ],
                token_usage=usage,
            )
        )

        if reserved is not None:
            actual = usage.total()
            if actual < reserved:
                self.rate_limiter_context.release(reserved - actual)
            elif actual > reserved:
                if not self.rate_limiter_context.check_and_consume(tokens=actual - reserved):
                    raise RateLimitExceededError(actual)

        return completion

    # -- History -----------------------------------------------------------

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


def _violation_feedback_message(violations: str) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Your previous output did not match the required schema:\n{violations}\n"
            "Please produce a corrected response that conforms exactly to the schema."
        ),
    }


def _validate_payload(
    content: str, schema: Union[bool, Dict[str, Any], Type[BaseModel]]
) -> Tuple[Optional[Any], Optional[str]]:
    """Validate a completion's raw text content against either a Pydantic
    model (the caller's `response_schema`) or a raw JSON Schema `dict`/`bool`
    (a skill's declared `output`). Returns `(payload, violations)` — exactly
    one is `None`."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"Response was not valid JSON: {e}"

    try:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(raw), None
        jsonschema.validate(raw, schema)
        return raw, None
    except (jsonschema.ValidationError, PydanticValidationError) as e:
        return None, str(e)


def _strip_fenced_code_block(content: str) -> str:
    stripped = content.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Crude pre-call token estimate (~4 chars/token) for rate-limiter
    reservation (spec section 10, "reserve an estimate of input tokens").
    Not a real tokenizer — a deployment with one should override via a
    custom `RateLimiterContext`/`Agent` subclass."""
    try:
        size = len(json.dumps(messages, default=str))
    except (TypeError, ValueError):
        size = sum(len(str(m)) for m in messages)
    return max(size // 4, 1)


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
