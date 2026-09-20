# Orchestration Agent Specification

**Status**: Reflects the shipped implementation in `src/orchestration_agent/`.
**Scope**: Skill system, schema-driven skill output, external tools (including MCP), the OpenAI provider, token/rate-limit accounting, conversation history (with summarization), and observability.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Architecture & Component Hierarchy](#architecture--component-hierarchy)
4. [Skill System](#skill-system)
5. [System Prompt Injection & Skill Catalogue](#system-prompt-injection--skill-catalogue)
6. [External Tools System](#external-tools-system)
7. [Agent Interface & Lifecycle](#agent-interface--lifecycle)
8. [Skill Dispatch & Output Enforcement](#skill-dispatch--output-enforcement)
9. [`SkillOutput` and `AgentResponse`](#skilloutput-and-agentresponse)
10. [Orchestration Flow (`Agent.run()`)](#orchestration-flow-agentrun)
11. [Provider Abstraction Layer](#provider-abstraction-layer)
12. [Token Tracking & Rate Limiting](#token-tracking--rate-limiting)
13. [Retry & Error Handling](#retry--error-handling)
14. [Conversation History](#conversation-history)
15. [Logging & Observability](#logging--observability)
16. [File Structure](#file-structure)
17. [pyproject.toml](#pyprojecttoml)
18. [Usage](#usage)
19. [Conformance Checklist](#conformance-checklist)

---

## Overview

The orchestration agent loads skills from a local skill directory, publishes them to a model, lets the model decide which skills to invoke and in what order, and returns both the answer and a full trace of what happened.

It:

- Loads skills from a skill directory (`SKILL.md` files with minimal YAML frontmatter).
- Injects a skill catalogue — each skill's `name` and `description` only — into the system prompt via a `{{skills_catalog}}` placeholder.
- Dispatches every skill invocation through a single `invoke_skill(skill_name)` tool.
- Optionally binds a skill's declared JSON Schema `output` to the completion that follows a successful dispatch, so that skill's result is schema-validated rather than free-form prose.
- Separately supports **external tools** (`BaseTool`), each exposed to the model as its own named function — including tools auto-discovered from an MCP server via `MCPToolProvider`.
- Tracks token consumption per completion and reconciles it against an external, synchronous `RateLimiterContext` using a reserve-then-release pattern.
- Optionally validates the agent's final answer against a caller-supplied `output_schema` (a Pydantic model), applied as structured output on every ordinary turn — not only a dedicated final turn.
- Retries transient provider failures with exponential backoff, independent of any model-level recovery.
- Enforces an overall `timeout_seconds` and a `Budget` (max skill invocations, max model turns) per run, reserving one turn for a final answer.
- Keeps conversation history (`ConversationHistory`) as a separate, longer-lived concern from a single run's in-flight state, and lets that history summarize itself through any `BaseProvider`.
- Reports every skill invocation, in order, in the returned `AgentResponse.messages` trace.

The only provider implemented today is `OpenAIProvider`, built on OpenAI's **Responses API** (`client.responses.create`), not the Chat Completions API. `BaseProvider` is written generically enough for other backends, but none currently ship.

### Out of scope

- Bespoke, per-skill input schemas. `invoke_skill` takes exactly one argument, `skill_name: string`.
- Declarative workflow definitions or orchestrator-side rules for composing skill outputs. Composition is the model's job.
- Runtime validation of a skill's `output` schema — that happens only in the standalone `orchestration_agent.scripts.validate_skill` CLI, never in `SkillLoader` or `Agent.__init__`.
- Any orchestrator-side transformation of a skill's `data`.
- Skill-level permissions, versioning enforcement, or side-effect declarations.
- Deciding *when* to call `ConversationHistory.summarize()` — that is caller policy; the primitive itself is in scope.

---

## Design Principles

1. **Decoupling.** Provider, skill loader, tools, conversation history, logger, and rate limiter are each independently usable.
2. **Opt-in.** Skills, external tools, `output_schema`, conversation history, and rate limiting are all optional.
3. **One contract language for skill output.** A skill's `output` is JSON Schema, unchanged, the same artifact the provider's structured-output mechanism consumes.
4. **No translation layer**, aside from the strict-mode rewrite (`_make_strict_json_schema`) applied identically to a skill's `output` and to the caller's `output_schema`.
5. **Enforcement at the boundary.** The boundary is the completion following a skill's dispatch, not the skill's markdown.
6. **Fail before deployment, not at call.** A malformed `output` schema is caught by `validate-skill`, run offline, never by the agent process.
7. **The trace is part of the contract.** Every skill invocation is recorded with its input, status, duration, and error.
8. **Generic at the agent level.** The agent has no opinion on what a skill's `data` means.
9. **The model proposes; the orchestrator authorizes.** The model names a `skill_name`; the orchestrator resolves it, decides whether a completion gets bound and to what schema, and is the only thing that touches the provider's structured-output parameter.
10. **Provider abstraction.** `BaseProvider` is interface-driven, even though only one implementation ships.
11. **Token transparency.** Every completion's usage is both accumulated (`AgentResponse.total_tokens`) and reconciled against a rate limiter as a step-level delta, never the running total.

---

## Architecture & Component Hierarchy

```
BaseProvider (abstract)
└── OpenAIProvider          # the only implementation that ships; built on the Responses API

BaseAgent                    # plain class: construction + skill loading, NOT an ABC
└── Agent                    # the concrete, only runnable agent; adds conversation history

BaseTool (abstract)
├── (caller-defined subclasses)
├── MCPTool                  # adapts one MCP server tool; produced by MCPToolProvider
└── AskUserQuestionTool      # built-in; asks the human one or more multiple-choice questions

Dataclasses (models/)
├── SkillOutput, ErrorDetail             (models/skill_output.py)
├── AgentResponse, Budget, StepRecord    (models/agent.py)
├── TokenUsage, ToolCall, CompletionResponse  (models/provider.py)
├── RateLimiterContext, RedisRateLimiter (models/rate_limiter.py)
└── the shared exception hierarchy, ERROR_CODES  (models/errors.py)

Pydantic models (skills/, tools/)
├── SkillMetadata     (skills/models.py)
├── BaseError         (tools/base.py)
└── Question, QuestionOption, AskUserQuestionInput/Output/Metadata  (tools/ask_user_question.py)

SkillLoader / SkillRegistry (skills/)

ConversationHistory (abstract)
├── InMemoryHistory
└── MongoDBConversationHistory   # imported only if `motor` is installed

Logger (abstract)
├── NoOpLogger
└── StdoutLogger
```

`BaseAgent` is a concrete Python class, not an `abc.ABC` — it holds `__init__` and `_load_skills()` only. `Agent` is the sole implementation that defines `run()`, `invoke_skill()`, and `invoke_tool()`; there is no separate abstract contract enforced at import time.

---

## Skill System

### `SKILL.md` format

Each skill is one file, `SKILL.md`, with YAML frontmatter plus a markdown body. The body becomes the tool-result content delivered to the model on dispatch.

```markdown
---
name: text-summarizer
description: Summarize a piece of text into a short, clear summary
compatibility: "python>=3.10, openai, anthropic, ollama"

output:
  type: object
  properties:
    summary:     { type: string, description: 3 to 5 sentence summary in plain, neutral language }
    key_points:  { type: array, items: { type: string }, description: 3 to 5 bullet-point facts or claims from the source }
  required: [summary, key_points]
  additionalProperties: false
---

## Text Summarization Instructions

You are an expert editor. When asked to summarize text:
1. Read the full source carefully.
2. Produce a neutral, plain-language summary of 3 to 5 sentences.
3. Extract 3 to 5 key points as short factual bullets.
```

This exact skill, plus a second one (`recipe-helper`, no `output`), ship under `skills/` in this repository and are what the test suite exercises against.

Frontmatter fields:

| Key | Required | Enforced by `SkillLoader` |
| --- | --- | --- |
| `name` | Yes | Must be present and a string |
| `description` | Yes | Must be present and a string |
| `compatibility` | Yes | Must be present and a string; otherwise free-form, uninterpreted |
| `output` | No | Read verbatim into `SkillMetadata.output` (a `dict`, `bool`, or `None`) — **not validated at all** by the loader |

### `SkillMetadata`

```python
# src/orchestration_agent/skills/models.py
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class SkillMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    compatibility: str
    content: str
    output: Optional[Any] = None
```

### `SkillLoader`

```python
# src/orchestration_agent/skills/loader.py
REQUIRED_FIELDS = ("name", "description", "compatibility")

class SkillLoader:
    def load_skills_from_path(self, skills_path: str) -> List[SkillMetadata]:
        """Recursively globs `SKILL.md` under skills_path, in sorted order.
        A skill that fails to parse is skipped (printed to stdout/stderr);
        loading continues with the rest."""

    def parse_skill_file(self, file_path: Path) -> SkillMetadata:
        """Splits the file on '---' (must start with it and have a closing
        delimiter), yaml.safe_load()s the frontmatter, requires name/
        description/compatibility to be present strings, and carries
        `output` through via `yaml_data.get("output")` with no further
        checks of any kind."""
```

Discovery order is deterministic (`sorted(skills_dir.rglob("SKILL.md"))`). If `skills_path` doesn't exist, `load_skills_from_path` raises `ValueError` immediately (this is *not* caught per-file — it's a hard failure for a missing directory).

### `SkillRegistry`

```python
# src/orchestration_agent/skills/registry.py
class SkillRegistry:
    def register(self, metadata: SkillMetadata) -> None: ...   # raises ValueError on duplicate name
    def get_by_name(self, name: str) -> Optional[SkillMetadata]: ...
    def get_all(self) -> List[SkillMetadata]: ...
    def serialize_to_prompt(self) -> str: ...  # delegates to utils.prompt.serialize_skills_to_prompt
```

### The `output` schema contract

`output` is optional. Where present its value is a native YAML mapping (or `True`/`False`), read as-is — no translation. It is a JSON Schema document:

- Every JSON Schema keyword is available (`type`, `properties`, `required`, `items`, `enum`, `$defs`, `$ref`, `anyOf`, `oneOf`, `allOf`, etc.).
- `$schema` pins a dialect; the only dialect the offline validator currently recognizes is Draft 2020-12 (`https://json-schema.org/draft/2020-12/schema`). Anything else fails validation.
- Local `#/$defs/...` references are supported; a `$ref` that does not start with `#` (i.e. a remote URL or file path) is rejected by the offline validator.

`output` is never published to the model — not in the skill catalogue, not in the `invoke_skill` tool definition. The model discovers a skill's output shape only via the schema-bound completion issued immediately after a successful dispatch of that skill.

### Strict-mode schema subset

Native structured output compiles against a constrained grammar. `_make_strict_json_schema` (duplicated, byte-for-byte identical logic, in `OpenAIProvider` — there is no separate shared module for it) recursively rewrites any object-typed schema node:

```python
# src/orchestration_agent/provider/openai_provider.py
def _make_strict_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    node = dict(schema)
    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        node["properties"] = {k: _make_strict_json_schema(v) for k, v in node["properties"].items()}
        node["required"] = list(node["properties"].keys())
    if "items" in node:
        node["items"] = _make_strict_json_schema(node["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            node[key] = [_make_strict_json_schema(sub) for sub in node[key]]
    if "$defs" in node:
        node["$defs"] = {k: _make_strict_json_schema(v) for k, v in node["$defs"].items()}
    return node
```

This same function is applied to both a caller's `output_schema` and a skill's declared `output` before either is sent to OpenAI — always, unconditionally, on every structured-output call, whether or not the schema was already strict-shaped.

The **offline validator** (below) separately *classifies* a schema as `strict` or `permissive` for authoring feedback, using a stricter and more literal notion of "strict" than the rewrite performs — the rewrite mechanically forces compliance (folding every key into `required`, forcing `additionalProperties: false`); the classifier instead flags what the *author* wrote as non-compliant, as a diagnostic, and does not fix it.

### Offline validation: `validate-skill`

```
.venv/bin/validate-skill /abs/path/to/a/skill
.venv/bin/validate-skill /abs/path/to/a/skill/SKILL.md
python -m orchestration_agent.scripts.validate_skill /abs/path/to/a/skill
```

Implemented in `src/orchestration_agent/scripts/validate_skill.py`, installed as the `validate-skill` console script. It:

1. Resolves the skill file (accepts a directory or a direct `SKILL.md` path).
2. Parses it with the same `SkillLoader.parse_skill_file` used at runtime — a frontmatter error is reported as a `ValidationFailure` with pointer `frontmatter`.
3. If `output` is absent, prints `OK <name>: no output declared (dispatch-only)` and exits 0.
4. If `output` is present but not a `dict`/`bool`, fails.
5. If it's a `bool`, it's trivially valid (no strict/permissive classification is meaningful for a boolean schema).
6. Otherwise validates against the JSON Schema Draft 2020-12 meta-schema (`jsonschema.Draft202012Validator.check_schema`) — the only supported dialect; any other `$schema` value fails with pointer `/output/$schema`.
7. Rejects any `$ref` that is not a local `#`-anchored reference, anywhere in the tree.
8. Classifies the schema as `strict` or `permissive` and prints every reason found. The unsupported-keyword set checked is:

   ```
   minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf,
   minLength, maxLength, pattern, format,
   if, then, else, not, patternProperties, dependentSchemas,
   minItems, maxItems, uniqueItems, minProperties, maxProperties
   ```

   Root type must be `object`; every object must set `additionalProperties: false` and list every property in `required`; `oneOf` is flagged with a "support varies by provider" reason even though it is still recursed into (`anyOf`/`allOf` are not flagged).

9. On success: exits 0, prints `OK <name>: output=strict` or `output=permissive` plus each permissive reason. On failure: exits 1, printing the skill's resolved path, a JSON Pointer, the offending keyword, and the message.

The runtime (`SkillLoader`, `Agent`) never calls this script and never reads its output — a skill with an invalid `output` still loads and dispatches; only the bound completion that follows would fail (as `SchemaCompilationError`, silently falling back — see [Skill Dispatch & Output Enforcement](#skill-dispatch--output-enforcement)) or produce validation errors at the boundary.

### Schema versioning

`output` is versioned with the skill, not separately — there is no schema registry or version negotiation in this codebase. When changing a skill's `output`, classify the change before shipping it:

| Change | Compatibility | Safe to ship as-is |
| --- | --- | --- |
| Add an optional field | Backward-compatible for readers | Yes, but see note below on what "optional" costs under strict mode |
| Remove a field | Breaking for anything reading it | No — treat as a new skill or a coordinated rollout |
| Narrow a type or add a constraint | Breaking | No |
| Widen a type, relax a constraint | Compatible | Yes |
| Edit a `description` | Behavioural, not structural | Yes, but re-run whatever evals exist — `description` text is what steers the model, and the strict-mode rewrite sends it through unchanged |

Because `_make_strict_json_schema` forces every property into `required`, "optional" under this system means a nullable union (`type: ["string", "null"]`), never an omitted key — adding a field that way is still a required key from the grammar's point of view, even though a consumer reading `data` can treat a `null` as absent. There is no naming convention enforced for a breaking skill version (e.g. `text-summarizer-v2`) — that is left to whoever ships the change.

---

## System Prompt Injection & Skill Catalogue

```python
# src/orchestration_agent/utils/prompt.py
PLACEHOLDER_RE = re.compile(r"\{\{\s*skills_catalog\s*\}\}")

def serialize_skills_to_prompt(registry: SkillRegistry) -> str:
    if not registry.skills:
        return "[No skills available]"
    lines = ["## Available Skills\n"]
    for skill in registry.get_all():
        lines.append(f"- **{skill.name}**: {skill.description}")
    return "\n".join(lines)

def inject_skills_into_prompt(system_prompt: str, skills_registry: SkillRegistry) -> str:
    if not PLACEHOLDER_RE.search(system_prompt):
        return system_prompt
    return PLACEHOLDER_RE.sub(lambda _: serialize_skills_to_prompt(skills_registry), system_prompt)
```

The placeholder regex matches both `{{skills_catalog}}` and `{{ skills_catalog }}`. A system prompt with no placeholder at all is left untouched — skills still load and dispatch normally; the model just never sees the catalogue unless the caller put the placeholder in.

Only `name` and `description` are ever rendered. `output` never appears here, and there is no separate "how to use skills vs. tools" boilerplate injected automatically — that explanatory text, if wanted, is the caller's own system prompt content (see the real example in [Usage](#usage)).

---

## External Tools System

A tool is a `BaseTool` subclass with its own Pydantic input schema, executed by the agent and exposed to the model **as its own named function** — there is no generic `invoke_tool(tool_name, tool_parameters)` wrapper. The model calls a tool exactly the way it would call any native function.

### `BaseTool`

```python
# src/orchestration_agent/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel


class BaseError(BaseModel):
    error: str
    details: Optional[str] = None


class BaseTool(ABC):
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    error_schema: Type[BaseModel] = BaseError

    @abstractmethod
    async def execute(self, input: BaseModel) -> Any:
        """Raise on failure; Agent.invoke_tool catches and formats with error_schema."""
```

`output_schema` is required with no default (unlike `error_schema`, which every tool inherits from `BaseTool`) — every concrete tool, hand-written or MCP-adapted, must declare what shape its result takes, since `Agent.invoke_tool` validates the return value against it before handing it back to the provider.

`ToolError` and `ToolNotFoundError` are **not** defined in `tools/base.py` — they live in `models/errors.py`, alongside every other exception in this codebase, so the whole project shares one hierarchy.

### Registration and dispatch

Tools are passed to `Agent(tools=[...])` and stored as `{tool.name: tool}`. `OpenAIProvider._convert_tool_to_openai_schema` turns each one into a Responses-API function-tool schema (`{"type": "function", "name": tool.name, ...}`) and adds it to the tool list alongside the always-present `invoke_skill` function. When the model calls a function by that name, `Agent._dispatch_round` recognizes it via `elif tool_call.name in self.tools:` and calls `Agent.invoke_tool(tool_call.name, tool_call.arguments)`.

```python
# src/orchestration_agent/agent.py — Agent.invoke_tool
async def invoke_tool(self, tool_name, parameters):
    tool = self.tools.get(tool_name)
    if not tool:
        raise ToolNotFoundError(f"Tool '{tool_name}' not found")

    try:
        validated_input = tool.input_schema(**parameters)
    except Exception as e:
        return tool.error_schema(error="Invalid parameters", details=str(e)).model_dump()

    try:
        result = await tool.execute(validated_input)
    except Exception as e:
        return tool.error_schema(error=str(e), details=f"Execution failed in {tool_name}").model_dump()

    try:
        validated_output = (
            result if isinstance(result, tool.output_schema)
            else tool.output_schema(**result) if isinstance(result, dict)
            else tool.output_schema.model_validate(result)
        )
    except Exception as e:
        return tool.error_schema(error="Invalid tool output", details=str(e)).model_dump()

    return validated_output.model_dump()
```

Three independent failure points, each formatted the same way via `error_schema.model_dump()`: bad input, a raised execution error, or a result that doesn't validate against `output_schema` (whether returned as an `output_schema` instance, a plain `dict`, or anything else `model_validate` can coerce). Only a fully validated result is ever handed back to the provider.

A lookup failure (`ToolNotFoundError`) and an unrecognized function name that is neither `invoke_skill` nor a registered tool are both caught in `_dispatch_round` and turned into an error `tool`-role reply — the run is not aborted. **External tool calls never appear in `AgentResponse.messages`** — only in `AgentResponse.steps` (`StepRecord.tool_calls`), which just records `{"name": ..., "arguments": ...}` for every call the provider returned that turn, skill or tool alike. A tool call never sets or clears a skill's pending output contract.

### MCP tools

`src/orchestration_agent/tools/mcp.py` adapts an MCP server's tools into `BaseTool` instances, so they can be handed straight to `Agent(tools=...)` without a hand-written subclass per tool. This module is opt-in: it imports `fastmcp`, which is only pulled in by the `mcp` extra (`pip install orchestration-agent[mcp]`) — the base package works without it, and nothing else in the codebase imports this module.

```python
from orchestration_agent.tools.mcp import MCPToolProvider

# Short-lived, as an async context manager:
async with MCPToolProvider("https://example.com/mcp") as mcp:
    agent = Agent(provider=llm_provider, system_prompt="...", tools=mcp.tools)
    ...

# Or an explicit connect/close lifecycle for a long-running process:
mcp = MCPToolProvider("https://example.com/mcp")
await mcp.connect()
agent = Agent(provider=llm_provider, system_prompt="...", tools=mcp.tools)
...
await mcp.aclose()
```

- `transport`: anything `fastmcp.Client` itself accepts — a URL, a filesystem path, an in-memory `FastMCP` server, or a `ClientTransport`.
- `name_prefix`: optional, namespaces every discovered tool as `"{prefix}.{tool_name}"` to avoid collisions when wiring multiple MCP servers into one agent.
- `connect()` opens the connection and calls `discover_tools()`; `.tools` is empty until then.
- Each discovered tool is wrapped as an `MCPTool`, whose `input_schema` is a **lenient** Pydantic model built by `_json_schema_to_model` (best-effort field typing off the MCP tool's raw JSON Schema, `extra="allow"`, required fields kept required) — its `model_json_schema()` is overridden to return the server's original schema verbatim, so the provider sees the real schema rather than a lossy reconstruction.
- `MCPTool.execute` calls `provider.call_tool(remote_name, arguments)` and returns, in order of preference, `result.data`, then `result.structured_content`, then the concatenated text of any text content blocks.
- Connection failures anywhere in this lifecycle raise `MCPConnectionError`.

### The built-in `question` tool (`AskUserQuestionTool`)

`src/orchestration_agent/tools/ask_user_question.py` ships a `BaseTool` that lets the model ask the human one or more multiple-choice questions mid-run — to gather preferences, clarify ambiguous instructions, or offer a choice of direction. No transport/UI is implemented here: how the question actually reaches the human (terminal prompt, web socket, queue, ...) is entirely up to the embedder, supplied as a required `handler` callback. It is **not** auto-registered on every `Agent` — pass it in like any other tool: `Agent(tools=[AskUserQuestionTool(handler=...), ...])`.

```python
# src/orchestration_agent/tools/ask_user_question.py
class QuestionOption(BaseModel):
    label: str
    description: str

class Question(BaseModel):
    question: str
    header: str
    options: List[QuestionOption] = Field(min_length=1)
    multiple: bool

class AskUserQuestionInput(BaseModel):
    questions: List[Question] = Field(min_length=1)

class AskUserQuestionMetadata(BaseModel):
    answers: List[List[str]]

class AskUserQuestionOutput(BaseModel):
    title: str
    output: str
    metadata: AskUserQuestionMetadata

QuestionHandler = Callable[[List[Question]], Awaitable[List[List[str]]]]

class AskUserQuestionTool(BaseTool):
    name = "question"
    input_schema = AskUserQuestionInput
    output_schema = AskUserQuestionOutput

    def __init__(self, handler: QuestionHandler) -> None:
        self.handler = handler
```

- `execute` calls `self.handler(input.questions)` and expects back one answer-list per question, in the same order — `multiple=False` questions still get a list (of one answer).
- If the handler returns a different number of answer-lists than questions were asked, `execute` raises `ToolError("handler returned a mismatched number of answers")`, which `Agent.invoke_tool` catches and formats via `error_schema` exactly like any other tool execution failure — the run is not aborted.
- On success, `output` is a human-readable rendering (`"{header}: {question}\n  -> {answers}"` per question) and `metadata.answers` carries the raw `List[List[str]]`, so a caller can consume either the summary text or the structured answers.
- `skills/laptop-search/SKILL.md` is the worked multi-turn example in this repo: it drives a narrowing search via repeated `search_laptops` calls, using `question` to ask for missing filters or let the user pick among remaining candidates (`tests/test_laptop_search_conversation.py`).

---

## Agent Interface & Lifecycle

### `BaseAgent`

Not an ABC — a plain class holding construction and skill loading only.

```python
# src/orchestration_agent/agent.py
class BaseAgent:
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
        ...
        self.skills_registry = SkillRegistry()
        self._load_skills()

    def _load_skills(self) -> None:
        if not self.skills_path:
            return
        for metadata in SkillLoader().load_skills_from_path(self.skills_path):
            self.skills_registry.register(metadata)
        self.system_prompt = inject_skills_into_prompt(self.system_prompt, self.skills_registry)
```

Notes on this exact parameter list:

- **`output_schema`**, not `response_schema`. It is the parameter that constrains the agent's *final answer*; "`response_schema`" is only the conceptual name used in this document's prose, matching the design rationale — the actual keyword argument is `output_schema`.
- **No `streaming` parameter.** `OpenAIProvider.complete()` does support `stream=True` and has a working `_stream_completion` generator, but `Agent` never sets it — every call `Agent` makes passes `stream` at its default (`False`). Streaming is a provider capability a caller can use directly against `OpenAIProvider`; the `Agent` loop does not expose it.
- **`timeout_seconds`** is an overall wall-clock budget for the whole run, checked before every provider call (`Agent._remaining_timeout`); it has no counterpart in `output_schema`/`budget` and is enforced independently of both.
- `Budget` defaults to `max_invocations=25`, `max_turns=15` (`models/agent.py`).

### `Agent`

```python
class Agent(BaseAgent):
    def __init__(
        self,
        provider: BaseProvider,
        system_prompt: str,
        conversation_history: Optional[ConversationHistory] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        on_message_added: Optional[Callable[..., Any]] = None,
        on_history_cleared: Optional[Callable[[], Any]] = None,
        **kwargs: Any,   # forwarded to BaseAgent.__init__
    ) -> None:
        super().__init__(provider, system_prompt, **kwargs)
        self.conversation_history = conversation_history or InMemoryHistory()
        self.history_formatter = history_formatter
        self.on_message_added = on_message_added
        self.on_history_cleared = on_history_cleared
```

`history_formatter` is stored but **`Agent.run()` never reads it** — nothing in the run loop calls `conversation_history.serialize_for_prompt()`. History is prepended to `provider_messages` as raw message dicts (`get_all()`'s return value, unformatted), never rendered through a formatter. `history_formatter` only matters if the caller separately calls `history.serialize_for_prompt(formatter=...)` or `history.summarize(history_formatter=...)` themselves.

`on_message_added(role, content, skills_invoked)` and `on_history_cleared()` may be sync or async — `Agent` checks `asyncio.iscoroutine(result)` and awaits it if so.

### Run-scoped state versus conversation history

A run's in-flight state is an explicit, private dataclass, `_RunState`, discarded when `run()` returns:

```python
@dataclass
class _RunState:
    run_start: float
    provider_messages: List[Dict[str, Any]]
    trace: List[SkillOutput] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
```

`ConversationHistory` is separate and longer-lived: a run reads it once via `get_all(session_id=self.session_id)` at the start of `run()`, and writes to it via `_add_to_history()` as it goes. The two are never conflated.

---

## Skill Dispatch & Output Enforcement

`invoke_skill` is the one tool published for every skill, taking exactly `skill_name: string`. It is provider-injected on *every* call automatically (`_invoke_skill_tool_schema()` in `openai_provider.py`) — it is not part of the caller's `tools=[...]` list and is offered even when the agent has zero skills loaded.

### Resolution (`Agent._dispatch_skill`)

1. **`skill_name` missing from the call args** → a `SkillOutput` with `type="invoke_skill"` (not the empty/unknown name — literally the tool's own name, since there is no skill name to attribute it to), `status="error"`, `error.code="SKILL_NOT_FOUND"`, message `"invoke_skill requires skill_name"`. The tool reply sent back to the model is `{"error": "invoke_skill requires skill_name"}`.
2. **`skill_name` given but not in the registry** → `Agent.invoke_skill` raises `SkillNotFoundError`; `_dispatch_skill` catches it and produces a `SkillOutput` with `type=<the given skill_name>`, `status="error"`, `error.code="SKILL_NOT_FOUND"`. The tool reply is `{"error": "Skill '<name>' not found"}`.
3. **Resolves** → `SkillOutput(type=skill_name, status="success", data=<markdown content>, input={"skill_name": skill_name})`. The tool reply is the raw markdown, verbatim (not JSON-wrapped — `_tool_message(call_id, tool_content, raw=True)`).

`Agent.invoke_skill(skill_name) -> str` itself is the simple half of this: it looks the skill up and either returns its `content` or raises `SkillNotFoundError`. All `SkillOutput` assembly, timing, and error formatting happens one level up, in `_dispatch_skill`.

### Binding a skill's output (`Agent._dispatch_round`, `_bind_skill_output`)

While processing a round's tool calls, every successful `invoke_skill` dispatch whose resolved `SkillMetadata.output is not None` overwrites `pending_contract` with `(skill.name, skill.output)` — **last dispatch wins** if several output-declaring skills are invoked in the same round. After the round's replies are all appended, `Agent.run()` checks `pending_contract` and, if set, calls `_bind_skill_output`:

```python
async def _bind_skill_output(self, run_state, skill_name, schema, step_label) -> SkillOutput:
    # First attempt, tools disabled, schema bound:
    payload, violations = await self._bound_completion(run_state, run_state.provider_messages, schema, skill_name, step_label)
    if violations is None:
        return SkillOutput(type=skill_name, status="success", data=payload, duration_ms=...)

    # Exactly one repair, feeding the violations back as a user-role message:
    repair_messages = run_state.provider_messages + [_violation_feedback_message(violations)]
    payload2, violations2 = await self._bound_completion(run_state, repair_messages, schema, skill_name, f"{step_label}:repair")
    if violations2 is None:
        return SkillOutput(type=skill_name, status="success", data=payload2, duration_ms=...)

    return SkillOutput(
        type=skill_name, status="error", data=None,
        error=ErrorDetail(code="OUTPUT_VALIDATION_ERROR", message="...", details={"violations": [...]}),
        duration_ms=...,   # covers BOTH attempts
    )
```

A `ModelRefusalError` or `asyncio.TimeoutError` raised by either attempt short-circuits immediately to an error `SkillOutput` with `error.code` `"MODEL_REFUSAL"` or `"TIMEOUT"` respectively — no repair is attempted for either of those.

### `_bound_completion`: native path first, fallback only on schema rejection

```python
async def _bound_completion(self, run_state, messages, schema, schema_name, step_label):
    try:
        completion = await self._call_provider(run_state, messages, tools=None,
                                                 structured_output=schema, structured_output_name=schema_name,
                                                 step_label=step_label)
    except SchemaCompilationError:
        completion = await self._fallback_completion(run_state, messages, schema, step_label)
    return _validate_payload(completion.content, schema)
```

`_call_provider` always passes `tools=None` here — the bound completion never offers `invoke_skill` or anything else, regardless of what tools the agent has registered.

`OpenAIProvider` raises `SchemaCompilationError` only when `client.responses.create(...)` raises `openai.BadRequestError` **and** `structured_output` was set on that call — a `BadRequestError` on a call with no structured output is re-raised unchanged (and, uncaught anywhere else, surfaces at the top of `Agent.run()` as `status="error"`, `error.code="AGENT_ERROR"`).

### `_fallback_completion`

```python
async def _fallback_completion(self, run_state, messages, schema, step_label):
    schema_json = schema if isinstance(schema, (dict, bool)) else schema.model_json_schema()
    instruction = {"role": "user", "content": f"Respond with only a single JSON object conforming exactly to this JSON Schema (no prose, no markdown fences):\n{json.dumps(schema_json)}"}
    completion = await self._call_provider(run_state, messages + [instruction], tools=None,
                                             structured_output=None, step_label=f"{step_label}:fallback")
    completion.content = _strip_fenced_code_block(completion.content)  # tolerates ```json ... ``` fences
    return completion
```

The substitution is invisible to the model in the trace: this produces the same single bound-completion `SkillOutput` as the native path would have, not a separate error entry (`test_skill_output_schema_compilation_falls_back_invisibly` asserts exactly two `SkillOutput` entries for the round: the dispatch and one bound completion).

### Validation at the boundary (`_validate_payload`)

```python
def _validate_payload(content, schema) -> Tuple[Optional[Any], Optional[str]]:
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
```

A skill's `output` (a raw `dict`/`bool`) is validated with `jsonschema.validate`; the caller's `output_schema` (a Pydantic class) is validated with `model_validate`, returning the parsed model instance rather than a plain dict. Exactly one of `(payload, violations)` is non-`None`.

If the invoked skill declares no `output`, none of this runs — dispatch is the entire interaction for that skill, and the next model turn is an ordinary step.

---

## `SkillOutput` and `AgentResponse`

Both are plain dataclasses (not Pydantic models) — `SkillOutput` self-validates its own invariants in `__post_init__`.

```python
# src/orchestration_agent/models/skill_output.py
@dataclass
class ErrorDetail:
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class SkillOutput:
    type: str
    status: str            # "success" | "error" | "partial"
    data: Any = None
    input: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        if not self.type:
            raise ValueError("SkillOutput.type must be a non-empty string")
        if self.status not in ("success", "error", "partial"):
            raise ValueError(f"Invalid SkillOutput.status: {self.status!r}")
        if self.status == "error":
            if self.error is None:
                raise ValueError("SkillOutput with status='error' requires `error`")
            if self.data is not None:
                raise ValueError("SkillOutput with status='error' must have data=None")
        if self.status == "partial" and self.error is None:
            raise ValueError("SkillOutput with status='partial' requires `error`")
```

Constructing an invalid `SkillOutput` (e.g. `status="error"` with `data` set) raises immediately — this is enforced in code, not merely documented.

```python
# src/orchestration_agent/models/agent.py
@dataclass
class StepRecord:
    step_number: int
    lm_call: Dict[str, Any]                  # {"model", "message_count", "stop_reason", "label"}
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)   # every call this turn, skill or external tool
    token_usage: TokenUsage = field(default_factory=TokenUsage)

@dataclass
class Budget:
    max_invocations: int = 25
    max_turns: int = 15

@dataclass
class AgentResponse:
    messages: List[SkillOutput]
    response: Any
    status: str             # "success" | "error" | "partial"
    error: Optional[ErrorDetail] = None
    duration_ms: Optional[float] = None
    steps: List[StepRecord] = field(default_factory=list)
    total_tokens: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
```

`response` is `Any`: a plain `str` when no `output_schema` was supplied, or an instance of the caller's Pydantic model when it was. `messages` holds only skill `SkillOutput`s — external tool calls are on `steps` (via `StepRecord.tool_calls`), never on `messages`.

### `response` is the answer; `messages` is the evidence

A consumer reads `response`. `messages` exists for debugging, audit, replay, and cost attribution — there is no orchestrator-side projection from a skill's `data` into `response`; the model writes `response` directly, having seen every `SkillOutput` that preceded it. A caller that reconstructs its answer by digging through `messages` instead of trusting `response` has been given the wrong `output_schema` for its needs — fix the schema rather than mining the trace.

`messages` is ordered by dispatch time and is append-only: nothing in `Agent` ever reorders, deduplicates, or removes an entry once appended to `run_state.trace`, including failed dispatches and failed bound completions.

### Status semantics

- **`success`** — the payload is authoritative: unvalidated markdown on a dispatch entry, or a schema-validated payload on a bound-completion entry.
- **`error`** — no usable payload; `data` is enforced to be `None` by `SkillOutput.__post_init__`. Nothing in this entry should be treated as a result.
- **`partial`** — reserved for a genuinely actionable but incomplete payload; nothing in the current codebase actually constructs a `partial` `SkillOutput` (the `TIMEOUT` code, for instance, is always emitted as `status="error"`, not `"partial"` — see [Skill Dispatch & Output Enforcement](#skill-dispatch--output-enforcement)). `AgentResponse.status` *does* use `"partial"`, for the reserved-final-turn path in `_final_turn_and_finish`.

### Run-level status

| `status` | Condition (as `Agent.run()` actually produces it) |
| --- | --- |
| `success` | `_finalize` validated a `response` (or no `output_schema` was set at all). Individual skills may still have failed — the model routed around them. |
| `partial` | `_final_turn_and_finish` produced a valid answer on the one reserved, tools-disabled turn after the turn/invocation budget ran out or the overall `timeout_seconds` fired. |
| `error` | `_finalize`'s one repair still didn't validate (`RESPONSE_VALIDATION_ERROR`); the reserved final turn itself failed or didn't validate (`BUDGET_EXHAUSTED`); the rate limiter refused a reservation or a top-up (`RATE_LIMIT_EXCEEDED`); or anything else propagated to `Agent.run()`'s outer `except Exception` (`AGENT_ERROR`). `messages` is still returned in full in every case. |

---

## Orchestration Flow (`Agent.run()`)

```python
async def run(self, user_input: str) -> AgentResponse:
    history_messages = await self.conversation_history.get_all(session_id=self.session_id)
    provider_messages = [*history_messages, {"role": "user", "content": user_input}]
    await self._add_to_history("user", user_input)

    run_state = _RunState(run_start=time.monotonic(), provider_messages=provider_messages)
    turn_count = 0
    invocation_count = 0
    effective_max_turns = max(self.budget.max_turns - 1, 0)   # reserve one turn for the final answer

    try:
        while turn_count < effective_max_turns and invocation_count < self.budget.max_invocations:
            completion = await self._step_with_retry(run_state, turn_count)
            turn_count += 1

            if not completion.tool_calls or completion.stop_reason in _TERMINAL_STOP_REASONS:
                return await self._finalize(run_state, completion)

            invoked, pending_contract = await self._dispatch_round(run_state, completion, turn_count)
            invocation_count += invoked

            if pending_contract:
                bound_output = await self._bind_skill_output(run_state, *pending_contract, f"turn:{turn_count}:bound")
                run_state.trace.append(bound_output)

        return await self._final_turn_and_finish(run_state)   # budget exhausted

    except asyncio.TimeoutError:
        return await self._final_turn_and_finish(run_state)
    except RateLimitExceededError as e:
        return self._response(run_state, response=None, status="error",
                               error=ErrorDetail(code="RATE_LIMIT_EXCEEDED", message=str(e)))
    except Exception as e:
        return self._response(run_state, response=None, status="error",
                               error=ErrorDetail(code="AGENT_ERROR", message=str(e)))
```

`_TERMINAL_STOP_REASONS = {"stop", "length", "content_filter"}`.

### A load-bearing detail: `output_schema` is bound on *every* ordinary turn, not just the last one

`_step_with_retry` calls `_call_provider` with **both** `tools=list(self.tools.values())` (skills' `invoke_skill` is always additionally offered by the provider itself) **and** `structured_output=self.output_schema` set simultaneously, on every ordinary turn of the loop — not only on a dedicated final turn:

```python
return await self._call_provider(
    run_state, run_state.provider_messages,
    tools=list(self.tools.values()) if self.tools else None,
    structured_output=self.output_schema,
    step_label=f"turn:{turn_count}",
)
```

This relies on the Responses API accepting a `text.format` json-schema constraint alongside `tools`: if the model calls a tool that turn, the schema simply doesn't apply to that turn's (empty) text output; if it instead answers with text, that text is already constrained. There is no separate "the loop finished, now bind the schema" step for the ordinary termination path — `_finalize` (below) validates what the model already produced under that constraint, and only repairs if it somehow still didn't parse/validate.

### Ordinary termination — `_finalize`

Reached when a turn produces no tool calls (or a terminal `stop_reason`):

```python
async def _finalize(self, run_state, completion) -> AgentResponse:
    await self._add_to_history("assistant", completion.content)
    if not self.output_schema:
        return self._response(run_state, response=completion.content, status="success")

    payload, violations = _validate_payload(completion.content, self.output_schema)
    if violations is None:
        return self._response(run_state, response=payload, status="success")

    # One repair, reusing the same tools-disabled/schema-bound machinery as a skill's bound completion:
    repair_messages = run_state.provider_messages + [
        {"role": "assistant", "content": completion.content},
        _violation_feedback_message(violations),
    ]
    payload2, violations2 = await self._bound_completion(run_state, repair_messages, self.output_schema, "response", "response:repair")
    if violations2 is None:
        return self._response(run_state, response=payload2, status="success")

    return self._response(run_state, response=None, status="error",
                           error=ErrorDetail(code="RESPONSE_VALIDATION_ERROR", message="...", details={"violations": [...]}))
```

A `ModelRefusalError`/`SchemaCompilationError`/`asyncio.TimeoutError` raised during that one repair attempt is also mapped to `RESPONSE_VALIDATION_ERROR` (not to `MODEL_REFUSAL`/`TIMEOUT` — those codes are reserved for a *skill's* bound completion, not the final response repair).

### Budget/timeout termination — `_final_turn_and_finish`

Reached when the `while` loop's turn or invocation budget runs out, or an `asyncio.TimeoutError` bubbles up from anywhere in the loop:

```python
async def _final_turn_and_finish(self, run_state) -> AgentResponse:
    try:
        completion = await self._call_provider(run_state, run_state.provider_messages,
                                                 tools=None, structured_output=self.output_schema, step_label="final")
    except Exception as e:
        return self._response(run_state, response=None, status="error",
                               error=ErrorDetail(code="BUDGET_EXHAUSTED", message=f"... {e}"))

    if self.output_schema:
        payload, violations = _validate_payload(completion.content, self.output_schema)
        if violations is not None:
            return self._response(run_state, response=None, status="error",
                                   error=ErrorDetail(code="BUDGET_EXHAUSTED", message="...", details={"violations": [...]}))
    else:
        payload = completion.content

    await self._add_to_history("assistant", completion.content)
    return self._response(run_state, response=payload, status="partial")
```

This turn is issued with `tools=None` — the model cannot call anything, skill or tool, on this last attempt — and **no repair** is attempted if the schema-constrained answer still fails to parse; that failure is immediately `BUDGET_EXHAUSTED`, not `RESPONSE_VALIDATION_ERROR`. Both `Budget.max_turns` exhaustion and `Budget.max_invocations` exhaustion funnel through this exact same method — invocation exhaustion is not a distinct code path.

### Tool-call round (`_dispatch_round`)

- The provider's own assistant turn is replayed **verbatim** via `completion.raw_message` (falling back to a synthetic `{"role": "assistant", "content": ...}` only if `raw_message` is absent) — appended once per round, not once per call. This matters concretely for `OpenAIProvider`: the Responses API's own output items must be replayed back as `input` items for a follow-up `function_call_output` to be considered valid against that same call's `call_id`s.
- Each reply is addressed to `tool_call.id or tool_call.name` and appended as `{"role": "tool", "tool_call_id": ..., "content": ...}`.
- `invoke_skill` calls become dispatch `SkillOutput`s on `run_state.trace`, with a `raw=True` tool reply (the markdown passed through unescaped) on success, or a JSON error object otherwise.
- Calls matching a registered tool name go through `invoke_tool`; their result (or a caught `ToolNotFoundError`/other exception) becomes a JSON tool reply, never a `SkillOutput`.
- Any other function name gets `{"error": "Unknown tool '<name>'"}` as its reply and a `logger.warning` — the run is not aborted.
- After the round, `_add_to_history("assistant", completion.content, skills_invoked=..., tools_invoked=...)` records both kinds of invocation for `ConversationHistory` — `skills_invoked` entries are `{"name": ...}`; `tools_invoked` entries are `{"name", "parameters", "result"}`. `tools_invoked` is logged via `self.logger.debug` but has no dedicated column in `InMemoryHistory`/`MongoDBConversationHistory` beyond whatever the caller's own message content or `on_message_added` callback does with it.

### Multiple calls in one round are dispatched sequentially, not concurrently

`_dispatch_round` processes `completion.tool_calls` with a plain `for` loop, `await`ing each `_dispatch_skill`/`invoke_tool` call in turn before moving to the next — there is no `asyncio.gather` anywhere in this path. Several `invoke_skill` calls in one round are therefore dispatched, and appear in `messages`, in that same fixed left-to-right order, one fully completing before the next starts. This is a correction worth being explicit about: a design that has no side effects to order-guard against does not by itself imply concurrent execution — this implementation simply doesn't do it.

### Chaining

One skill's result feeding another's use is an ordinary model decision, not a mechanism: the model reads `data` from an earlier `SkillOutput` — a dispatch's markdown or a bound completion's validated payload — and acts on it in a later turn, because that `SkillOutput` is sitting right there in the conversation as a prior tool result. There is no reference syntax, no template expansion, and no orchestrator-side variable binding; `Agent` never inspects one `SkillOutput` while producing another.

---

## Provider Abstraction Layer

### `BaseProvider`

```python
# src/orchestration_agent/provider/base.py
class BaseProvider(ABC):
    def __init__(self, model: str, api_key: str, **kwargs): ...

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
        **kwargs,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """Raises RateLimitError, TemporaryProviderError, ModelRefusalError,
        or SchemaCompilationError. The provider itself never validates
        `structured_output` against the result — that's the orchestrator's job."""
```

`structured_output` may be a Pydantic class **or** a raw JSON Schema `dict` — the latter is how a skill's `output` reaches the provider; `structured_output_name` supplies the schema's name in that case (OpenAI requires a `name` on every `json_schema` format).

### `OpenAIProvider` — built on the Responses API

`OpenAIProvider.complete()` calls `self.client.responses.create(...)`, **not** `chat.completions.create`. Key mechanics, all real:

- **Message conversion.** `_messages_to_responses_input` turns the agent's heterogeneous message list into Responses `input` items: a `{"type": "response_output", "output": [...]}` wrapper (this provider's own `raw_message` marker) is expanded back into its original output items; `{"role": "tool", "tool_call_id", "content"}` becomes `{"type": "function_call_output", "call_id", "output"}`; anything else with a `role` passes through as `{"role", "content"}`.
- **System prompt** is sent as `instructions`, not prepended as a `system`-role message.
- **`max_output_tokens`**, not `max_tokens`, is the Responses API's parameter name; `Agent._call_provider` always passes the literal `2048` — there is no agent-level knob to change it today.
- **Reasoning models** (`gpt-5*`, `o1`/`o3`/`o4`-prefixed) reject a non-default `temperature` — detected via `_REASONING_MODEL_RE` and the `temperature` param is omitted entirely for them; a `reasoning_effort` kwarg, if supplied, is sent as the nested `reasoning.effort` param instead of a top-level field.
- **"luna" models** (matched case-insensitively by `_LUNA_MODEL_RE`) reject a request that combines `reasoning.effort` with function tools; since `invoke_skill` is always sent as a tool, `reasoning_effort` is silently dropped for any model whose name contains "luna".
- **Structured output** becomes `completion_params["text"] = {"format": {"type": "json_schema", "name": ..., "schema": _make_strict_json_schema(...), "strict": True}}`.
- **Tool calls**: every `function_call` item in `response.output` becomes a `ToolCall(name, arguments, id=call_id)` — not filtered to `invoke_skill` only, so external tool calls are captured too.
- **`raw_message`** on the returned `CompletionResponse` is `{"type": "response_output", "output": [<every output item, dumped>]}` — see message conversion above.
- **Refusal detection**: only checked when `structured_output` was requested; scans `response.output` for a `message`-type item containing a `refusal`-type content part, and raises `ModelRefusalError` if found — distinct from `BadRequestError` (schema rejection, raised *before* generation), which maps to `SchemaCompilationError` **only** when `structured_output` was set on that call.
- **`stop_reason` mapping**: `"completed"/"cancelled"/"failed"` → `"stop"`; an `incomplete` status with reason `max_output_tokens` → `"length"`, `content_filter` → `"content_filter"`. The Responses API reports `"completed"` even when the model's turn ended on a function call (no separate "requires action" status), so if `tool_calls` is non-empty and the mapped reason is `"stop"`, it's overridden to `"tool_calls"` specifically so `Agent.run()`'s `_TERMINAL_STOP_REASONS` check doesn't mistake a pending tool call for a final answer.
- **Token usage** reads `response.usage.input_tokens` / `.output_tokens` / `.input_tokens_details.cached_tokens` (mapped to `cache_read_tokens`); `cache_creation_tokens` is always `None` — not applicable to this API.
- **Error mapping**: `openai.RateLimitError` → `RateLimitError`; `InternalServerError`/`APIConnectionError`/`APITimeoutError` → `TemporaryProviderError`. Any other exception (including a `BadRequestError` with no `structured_output` in play) propagates unchanged.
- The client itself is constructed with `max_retries=0` — the SDK's own retry is disabled because `Agent._step_with_retry` owns all retry/backoff.
- **Tool schema conversion** (`_convert_tool_to_openai_schema`) is a direct, one-level Responses function-tool: `{"type": "function", "name": tool.name, "description": tool.description, "parameters": {"type": "object", "properties": ..., "required": ...}}` — taken straight from `tool.input_schema.model_json_schema()`, with no strict-mode rewrite applied to tool parameter schemas (the strict rewrite is only applied to `structured_output`, never to tool call schemas).

No `AnthropicProvider` or `OllamaProvider` exists in this codebase, and `pyproject.toml` defines no such optional extras — only `openai` (`openai>=1.50`) ships as a provider dependency.

---

## Token Tracking & Rate Limiting

### `RateLimiterContext`

```python
# src/orchestration_agent/models/rate_limiter.py
class RateLimiterContext(ABC):
    @abstractmethod
    def check_and_consume(self, tokens: int) -> bool: ...
    @abstractmethod
    def get_current_bucket(self) -> Dict[str, Any]: ...
    @abstractmethod
    def reset(self) -> None: ...

    def release(self, tokens: int) -> None:
        """No-op default; override to support exact reconciliation."""
```

There is no `reserve()`/`reconcile()` pair — the same `check_and_consume` method both makes the *initial* reservation and, later, tops it up if actual usage exceeded the estimate; `release` gives back the excess if actual usage came in under the estimate. `check_and_consume` is called **synchronously** (no `await`) from `Agent._call_provider` — a `RateLimiterContext` implementation must not itself be async.

### Reserve-then-reconcile, as actually implemented (`Agent._call_provider`)

```python
reserved = None
if self.rate_limiter_context:
    estimate = _estimate_tokens(messages) + 2048          # +2048: the fixed max_output_tokens
    if not self.rate_limiter_context.check_and_consume(tokens=estimate):
        raise RateLimitExceededError(estimate)
    reserved = estimate

try:
    completion = await (asyncio.wait_for(coro, timeout=remaining) if remaining is not None else coro)
except Exception:
    if reserved is not None:
        self.rate_limiter_context.release(reserved)       # full release on any failure
    raise

# ... token accounting into run_state.token_usage happens unconditionally here ...

if reserved is not None:
    actual = completion.token_usage.total()
    if actual < reserved:
        self.rate_limiter_context.release(reserved - actual)
    elif actual > reserved:
        if not self.rate_limiter_context.check_and_consume(tokens=actual - reserved):
            raise RateLimitExceededError(actual)          # after the call already happened
```

`_estimate_tokens` is a crude, non-tokenizer heuristic: `max(len(json.dumps(messages, default=str)) // 4, 1)`. If the provider call itself raises, any reservation is fully released before re-raising. If actual usage exceeds the reservation and the top-up itself is refused, `Agent.run()` still surfaces this as a normal `RateLimitExceededError` → `status="error"`, `error.code="RATE_LIMIT_EXCEEDED"` — even though the tokens were already spent and the call already succeeded; there's no way to undo a completion that already happened.

### `TokenUsage` accounting rule

```python
run_state.token_usage.input_tokens += usage.input_tokens
run_state.token_usage.output_tokens += usage.output_tokens
```

Each step's own `usage` (never the running `run_state.token_usage` total) is what's passed to `check_and_consume`/`release`. `AgentResponse.total_tokens` and `AgentResponse.token_usage` are the separate, purely additive running totals reported at the end.

### `RedisRateLimiter`

Ships in `models/rate_limiter.py` (not merely an example — a concrete, usable implementation), and deliberately uses the **synchronous** `redis` client, not `redis.asyncio`, since `check_and_consume` is called without `await`:

```python
class RedisRateLimiter(RateLimiterContext):
    def __init__(self, redis_client, tokens_per_minute=10_000, tokens_per_hour=1_000_000, key_prefix="rate_limit:agent:"): ...

    def check_and_consume(self, tokens: int) -> bool:
        # reads current minute/hour counters; if either would exceed its
        # limit, returns False without consuming anything; otherwise
        # atomically INCRBYs both keys via a pipeline and (re-)sets their
        # EXPIRE (60s / 3600s).

    def release(self, tokens: int) -> None:
        # best-effort, NON-atomic: reads both counters, subtracts, clamps
        # at 0, and SETs them back with keepttl=True. Not atomic against
        # concurrent writers — an accepted trade-off, matching the fixed-
        # window (not sliding-window) design of check_and_consume itself.

    def get_current_bucket(self) -> Dict[str, Any]: ...
    def reset(self) -> None: ...   # deletes both keys
```

This is a fixed-window, not sliding-window, limiter — imprecise exactly at window boundaries, an accepted trade-off for this implementation.

---

## Retry & Error Handling

### Error taxonomy (`models/errors.py`, `ERROR_CODES`)

```python
ERROR_CODES = frozenset({
    "SKILL_NOT_FOUND",
    "OUTPUT_VALIDATION_ERROR",
    "MODEL_REFUSAL",
    "TIMEOUT",
    "RESPONSE_VALIDATION_ERROR",
    "BUDGET_EXHAUSTED",
    "RATE_LIMIT_EXCEEDED",
    "AGENT_ERROR",
})
```

| `code` | Where it's produced | Skill ran | Who recovers |
| --- | --- | --- | --- |
| `SKILL_NOT_FOUND` | `_dispatch_skill` — missing or unresolved `skill_name` | No | Model, on its next turn |
| `OUTPUT_VALIDATION_ERROR` | `_bind_skill_output` — both attempts failed to validate | Yes (dispatch succeeded) | Orchestrator (one repair), then model |
| `MODEL_REFUSAL` | `_bind_skill_output` — `ModelRefusalError` from either attempt | Yes | Model |
| `TIMEOUT` | `_bind_skill_output` — `asyncio.TimeoutError` from either attempt | Yes | Model |
| `RESPONSE_VALIDATION_ERROR` | `_finalize` — final answer invalid after one repair (or the repair itself raised) | — | Caller |
| `BUDGET_EXHAUSTED` | `_final_turn_and_finish` — the reserved turn errored, or its answer didn't validate | — | Caller |
| `RATE_LIMIT_EXCEEDED` | `Agent.run()`'s outer `except RateLimitExceededError` | — | Caller |
| `AGENT_ERROR` | `Agent.run()`'s outer, catch-all `except Exception` — includes exhausted transport retries | — | Caller |

`AGENT_ERROR` is a genuine catch-all: exhausting `max_retries` on a `TemporaryProviderError`/`RateLimitError` re-raises past `_step_with_retry`, is not caught anywhere more specific, and lands here — there is no dedicated "retry exhausted" code.

### Three retry mechanisms

1. **Transport-level (`Agent._step_with_retry`).** Catches only `RetryableError` (`RateLimitError`, `TemporaryProviderError`), retries with `self.backoff_strategy(attempt)` (default `default_exponential_backoff`: `2**attempt + jitter(0, 10%)`), up to `max_retries` extra attempts (so `max_retries + 1` total). Exhaustion re-raises the last error, invisibly to the model until it becomes `AGENT_ERROR` at the top of `run()`.
2. **Output-validation internal repair (`_bind_skill_output`, `_finalize`).** Exactly one repair, feeding the violation message back as a `user`-role turn — never a bare retry. Both attempts' time counts toward one `SkillOutput.duration_ms`.
3. **Model-level recovery.** Every `SkillOutput` with `status="error"` (or `"partial"`) is just another tool reply the model sees on its next turn; the model decides whether to re-invoke, substitute, or give up. The orchestrator never retries a skill dispatch on the model's behalf.

`StructuredOutputValidationError` exists in `models/errors.py` as a documented, deliberately-unused-as-a-`RetryableError` marker type — the actual repair path in `agent.py` works off the `(payload, violations)` tuple from `_validate_payload`, not by catching this exception; it's part of the exception hierarchy for API completeness/backward compatibility, not something the current dispatch code raises or catches.

### Failure is not fatal

A failed skill dispatch or bound completion does not end the run — its `SkillOutput` is appended and the loop continues. A run can finish `status="success"` with one or more `error` entries in `messages`.

---

## Conversation History

```python
# src/orchestration_agent/conversation/base.py
class ConversationHistory(ABC):
    async def add_message(self, role, content, session_id, skills_invoked=None) -> None: ...
    async def get_all(self, session_id=None) -> List[Dict[str, Any]]: ...
    async def serialize_for_prompt(self, session_id=None, formatter=None) -> str: ...
    async def clear(self, session_id=None) -> None: ...
    async def summarize(self, model_provider, system_prompt, session_id=None, history_formatter=None) -> str: ...


async def summarize_messages(messages, formatted_history, model_provider, system_prompt) -> str:
    """Shared by every implementation: if `messages` is empty, returns
    "No conversation history to summarize." without calling the provider.
    Otherwise wraps `formatted_history` in a fixed prompt template and
    calls model_provider.complete(messages=[{"role": "user", ...}],
    system_prompt=system_prompt, temperature=0.7, max_tokens=1024,
    structured_output=None, stream=False, tools=None), returning
    completion.content."""
```

Both shipped implementations (`InMemoryHistory`, `MongoDBConversationHistory`) implement `summarize()` identically, as three lines: `get_all()`, `serialize_for_prompt()`, then `summarize_messages(...)` — the shared helper is what actually talks to the provider, so neither backend duplicates that logic.

### `InMemoryHistory`

A process-local `List[Dict[str, Any]]`. Each message dict is `{"session_id", "role", "content", "timestamp"}` plus `"skills_invoked"` when given. `get_all`/`clear` filter by `session_id` when supplied, else operate on everything.

### `MongoDBConversationHistory`

Backed by any Motor-compatible client (`motor.motor_asyncio.AsyncIOMotorClient` — note the real class name; it is *not* `AsyncClient`, which doesn't exist in `motor`). Lazily creates a `session_id` index and, if `ttl_seconds` is set (default 7 days), a TTL index on `createdAt`, on first use (`_ensure_indexes`, idempotent, best-effort — an index-already-exists error is caught and printed, not raised).

### `default_formatter`

```python
# src/orchestration_agent/conversation/formatters.py
def default_formatter(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if msg.get("skills_invoked"):
            content = f"{content} [Used: {', '.join(s['name'] for s in msg['skills_invoked'])}]"
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)
```

Any role other than `"user"` — including `"tool"` — is labeled `"Assistant"`. There is no distinct `"Tool"` label.

**`Agent` does not call `serialize_for_prompt` or apply `history_formatter` anywhere in `run()`.** History is read via `get_all()` and spliced into `provider_messages` as raw dicts. `history_formatter`/`serialize_for_prompt`/`summarize` only come into play if the caller invokes them directly against `agent.conversation_history`.

---

## Logging & Observability

```python
# src/orchestration_agent/logging/base.py
class Logger(ABC):
    def debug(self, message: str, **kwargs) -> None: ...
    def info(self, message: str, **kwargs) -> None: ...
    def warning(self, message: str, **kwargs) -> None: ...
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None: ...
```

`NoOpLogger` (the default) and `StdoutLogger` (prints `[LEVEL] message`, appending `: {exception}` on `error` when one is given) ship. A caller wanting Datadog/Splunk/OTel integration implements `Logger` directly and passes it as `Agent(logger=...)`.

### What determinism means here

The model-driven design guarantees the *shape* of the result, not its *content*:

| Property | Guaranteed | Why |
| --- | --- | --- |
| `SkillOutput`/`AgentResponse` shape | Yes | Both are dataclasses; `SkillOutput.__post_init__` enforces its own invariants at construction time |
| A bound completion's `data` conformance | Yes | Validated by `_validate_payload` before it ever reaches `SkillOutput.data`; non-conforming output never ships as `status="success"` |
| `response` conformance | Yes, when `output_schema` is set | Validated in `_finalize`/`_final_turn_and_finish` before `AgentResponse` is returned; failure is a run-level error |
| Trace completeness | Yes | Every dispatch and every bound completion is appended to `run_state.trace`, including failed ones |
| Trace ordering | Yes | Dispatch order, and — per the previous section — sequential dispatch order within a round too |
| Which skills get invoked, and how many times | No | The model decides |
| `response` content | No | The model writes it |

A caller can write a typed client against `output_schema` and never defensively parse `response`. It cannot assume the same skills run, in the same order, on a repeat call with the same input — that depends on model sampling, not on anything this codebase fixes.

### The trace as a product surface

`AgentResponse.messages` ships in the response, not as a side log:

- **Audit** — `SkillOutput.input` (`{"skill_name": ...}`) makes every dispatch reconstructible: what was asked of a skill, and what came back.
- **Cost and latency attribution** — `SkillOutput.duration_ms` per entry, summed against `AgentResponse.duration_ms`, shows how much of a run's wall-clock was skill dispatch/validation versus ordinary model turns. `StepRecord.token_usage` gives the same breakdown for tokens.
- **Debugging a wrong `response`** — with the trace, a bad final answer is traceable to either a bad skill result (visible as an `error`/unexpected `data` entry in `messages`) or bad synthesis by the model over otherwise-good entries. Without the trace, both look identical from the outside.
- **Replay** — because every dispatch is deterministic given the same skill files (a lookup and, at most, one schema-bound completion — no side effects), the recorded `messages` can in principle be replayed against an unchanged `skills/` directory to test a skill-content change offline. Nothing in this codebase currently automates that; it's a property the shape enables, not a shipped tool.

### Metrics worth emitting

Nothing in this codebase emits metrics today — `Logger` is the only observability hook that exists, and neither `NoOpLogger` nor `StdoutLogger` aggregates anything. A caller wiring up real metrics from the trace and from `Logger` calls would want, at minimum:

| Metric | Dimensions | Watch for |
| --- | --- | --- |
| Skill invocations | skill name (`SkillOutput.type`), `status` | Error-rate shifts per skill |
| Output validation failures | skill name, native vs. fallback path | A high rate on a given skill argues for rewriting its `output` to the strict subset (run `validate-skill` against it) |
| Repair attempts | skill name | Repeated repairs on the same skill mean its schema or `description`s need work |
| Response validation failures | caller / `output_schema` | A rising rate means `output_schema` is too tight, too complex, or the system prompt isn't steering the model toward it |
| Budget exhaustion (`BUDGET_EXHAUSTED`, or `status="partial"`) | caller | Tasks are outgrowing `Budget.max_turns`/`max_invocations`, or `timeout_seconds` is too tight |
| Rate-limit refusals (`RATE_LIMIT_EXCEEDED`) | caller | The configured `RateLimiterContext` limits are too tight for actual load, or the crude `_estimate_tokens` heuristic is over-reserving |
| `StepRecord.duration`/token usage | skill vs. ordinary turn, percentile | Where a run's wall-clock and spend actually go |

### Retention

`SkillOutput.data` on a bound-completion entry may carry whatever a skill's `output` schema was written to hold, including caller-provided or regulated data that passed through the model. Retention policy for `AgentResponse`/`ConversationHistory` content is entirely deployment-specific — the only built-in retention mechanism in this codebase is `MongoDBConversationHistory`'s optional TTL index on `createdAt` (default 7 days), and that only covers conversation history, not a single run's `AgentResponse`.

---

## File Structure

```
orchestrator/
├── pyproject.toml
├── src/
│   └── orchestration_agent/
│       ├── __init__.py              # re-exports the public surface (see below)
│       ├── agent.py                 # BaseAgent, Agent — the entire orchestration loop
│       ├── provider/
│       │   ├── __init__.py          # exports BaseProvider; OpenAIProvider only if importable
│       │   ├── base.py              # BaseProvider
│       │   ├── openai_provider.py   # OpenAIProvider (Responses API)
│       │   └── errors.py            # thin re-export shim onto models.errors
│       ├── skills/
│       │   ├── __init__.py
│       │   ├── models.py            # SkillMetadata
│       │   ├── loader.py            # SkillLoader
│       │   └── registry.py          # SkillRegistry
│       ├── tools/
│       │   ├── __init__.py          # exports BaseError, BaseTool, AskUserQuestionTool + its models
│       │   ├── base.py              # BaseError, BaseTool
│       │   ├── ask_user_question.py # AskUserQuestionTool, Question(s), QuestionHandler
│       │   └── mcp.py               # MCPToolProvider, MCPTool, MCPConnectionError (opt-in, needs fastmcp)
│       ├── conversation/
│       │   ├── __init__.py
│       │   ├── base.py              # ConversationHistory, summarize_messages()
│       │   ├── in_memory.py         # InMemoryHistory
│       │   ├── mongodb.py           # MongoDBConversationHistory (imported only if motor is present)
│       │   └── formatters.py        # default_formatter
│       ├── logging/
│       │   ├── __init__.py
│       │   ├── base.py              # Logger
│       │   └── implementations.py   # NoOpLogger, StdoutLogger
│       ├── models/
│       │   ├── __init__.py
│       │   ├── agent.py             # AgentResponse, Budget, StepRecord
│       │   ├── skill_output.py      # SkillOutput, ErrorDetail
│       │   ├── provider.py          # TokenUsage, ToolCall, CompletionResponse
│       │   ├── rate_limiter.py      # RateLimiterContext, RedisRateLimiter
│       │   └── errors.py            # the entire exception hierarchy + ERROR_CODES + violations_to_list
│       ├── scripts/
│       │   ├── __init__.py
│       │   └── validate_skill.py    # the `validate-skill` CLI
│       └── utils/
│           ├── __init__.py
│           ├── prompt.py            # inject_skills_into_prompt, serialize_skills_to_prompt
│           └── retry.py             # default_exponential_backoff (only)
├── skills/                          # the actual skills this repo ships
│   ├── text-summarizer/SKILL.md     # declares `output`
│   ├── recipe-helper/SKILL.md       # no `output` — dispatch-only
│   └── laptop-search/SKILL.md       # discriminated-union `output`; drives search_laptops + question tool
├── examples/
│   └── basic_usage.py               # the only example that exists
└── tests/
    ├── conftest.py                  # loads .env; redis/mongo/openai-key availability fixtures
    ├── fakes.py                     # FakeProvider + text_response/json_response/tool_call_response
    ├── test_agent.py                # the orchestration loop, end to end, against FakeProvider
    ├── test_ask_user_question.py    # AskUserQuestionTool in isolation and through Agent
    ├── test_conversation_history.py
    ├── test_laptop_search_conversation.py / _live.py  # multi-turn: skill + search_laptops + question tool
    ├── test_mcp_tool_provider.py / test_mcp_tool_provider_live.py
    ├── test_openai_provider_live.py
    ├── test_rate_limiter.py
    ├── test_retry.py
    ├── test_skill_loader.py
    └── test_validate_skill.py
```

There is no `skills/dispatch.py`, `skills/executor.py`, `models/envelope.py`, or `tools/errors.py` — dispatch logic lives entirely as private methods on `Agent`, and every exception lives in `models/errors.py`.

### `src/orchestration_agent/__init__.py` re-exports

```python
from .agent import Agent, BaseAgent
from .conversation import ConversationHistory, InMemoryHistory
from .logging import Logger, NoOpLogger, StdoutLogger
from .models import (
    AgentError, AgentResponse, Budget, BudgetExhaustedError, CompletionResponse,
    ErrorDetail, ModelRefusalError, RateLimiterContext, RateLimitExceededError,
    RedisRateLimiter, ResponseValidationError, SchemaCompilationError,
    SkillNotFoundError, SkillOutput, StepRecord, StructuredOutputValidationError,
    TokenUsage, ToolCall,
)
from .provider.base import BaseProvider
from .skills import SkillLoader, SkillMetadata, SkillRegistry
from .tools import BaseError, BaseTool

# MongoDBConversationHistory is appended to __all__ only if `motor` imports successfully.
```

`OpenAIProvider` is **not** re-exported from the top-level package — import it explicitly from `orchestration_agent.provider.openai_provider` (or `orchestration_agent.provider`, which re-exports it conditionally). `MCPToolProvider`/`MCPTool` are likewise never re-exported anywhere outside `orchestration_agent.tools.mcp` itself.

---

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "orchestration-agent"
version = "0.1.0"
description = "Decoupled, opt-in orchestration agent with skill loading and provider abstraction"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }

dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "typing-extensions>=4.0",
    "jsonschema>=4.20",
]

[project.scripts]
validate-skill = "orchestration_agent.scripts.validate_skill:main"

[project.optional-dependencies]
openai = ["openai>=1.50"]
mongodb = ["motor>=3.0"]
redis = ["redis>=5.0"]
mcp = ["fastmcp>=2.0"]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "python-dotenv>=1.0",
]
all = ["orchestration-agent[openai,mongodb,redis,mcp]"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

There is no `anthropic` or `ollama` extra, and no `black`/`isort`/`mypy`/`ruff` configuration in this project today.

---

## Usage

The only example that ships, verbatim (`examples/basic_usage.py`):

```python
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv
from orchestration_agent import Agent, Budget, StdoutLogger
from orchestration_agent.provider.openai_provider import OpenAIProvider

REPO_ROOT = Path(__file__).resolve().parent.parent

SNIPPET = '''
def get_user(request):
    user_id = request.GET["id"]
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''

async def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
    provider = OpenAIProvider(model=model, api_key=os.environ["OPENAI_API_KEY"])

    agent = Agent(
        provider=provider,
        system_prompt=(
            "You are a helpful coding assistant with access to specialized skills.\n\n"
            "{{skills_catalog}}\n\n"
            "When a task matches a skill above, call invoke_skill with that skill's "
            "name, then follow the returned instructions to answer the user."
        ),
        skills_path=str(REPO_ROOT / "skills"),
        logger=StdoutLogger(),
        budget=Budget(max_turns=5),
    )

    response = await agent.run(f"Review this code for security issues:\n\n{SNIPPET}")

    print(response.response)
    for entry in response.messages:
        print(f"{entry.type}: {entry.status}")
    print(f"status={response.status} steps={len(response.steps)} total_tokens={response.total_tokens}")

asyncio.run(main())
```

Note this example asks for a *code review*, but the only skills that exist in `skills/` are `text-summarizer` and `recipe-helper` — neither matches, so in practice the model answers directly without invoking a skill unless the repository's skill set is extended.

### A schema-validated skill, and a required final schema (illustrative — no test/example exercises this exact combination, but every piece is real API)

```python
from pydantic import BaseModel
from orchestration_agent import Agent
from orchestration_agent.provider.openai_provider import OpenAIProvider

class SummaryDelivered(BaseModel):
    summary_delivered: bool
    note: str

agent = Agent(
    provider=OpenAIProvider(model="gpt-5.4-mini", api_key="..."),
    system_prompt="You are a summarization assistant.\n\n{{skills_catalog}}",
    skills_path="./skills",
    output_schema=SummaryDelivered,   # NOTE: the kwarg is output_schema, not response_schema
)

response = await agent.run("Summarize this article for me: <article text>")
# response.response is a SummaryDelivered instance when status == "success"
```

### External tools + MCP tools together

```python
from orchestration_agent import Agent
from orchestration_agent.tools.mcp import MCPToolProvider

async with MCPToolProvider("https://example.com/mcp") as mcp:
    agent = Agent(
        provider=provider,
        system_prompt="...\n\n{{skills_catalog}}",
        skills_path="./skills",
        tools=[*mcp.tools, MyHandWrittenTool()],
    )
    response = await agent.run("...")
```

### The `question` tool

```python
from orchestration_agent.tools import AskUserQuestionTool, Question

async def prompt_in_terminal(questions: list[Question]) -> list[list[str]]:
    answers = []
    for q in questions:
        print(f"{q.header}: {q.question}")
        for i, opt in enumerate(q.options):
            print(f"  {i + 1}. {opt.label} — {opt.description}")
        choice = input("> ")
        answers.append([q.options[int(choice) - 1].label])
    return answers

agent = Agent(
    provider=provider,
    system_prompt="...",
    tools=[AskUserQuestionTool(handler=prompt_in_terminal)],
)
response = await agent.run("Help me pick a laptop.")
```

`prompt_in_terminal` is the `QuestionHandler`: whatever it returns becomes `metadata.answers` on the tool's result. See `skills/laptop-search/SKILL.md` for a full multi-turn flow that combines this tool with a custom `search_laptops` tool and a discriminated-union `output` schema.

### Rate limiting

```python
import redis
from orchestration_agent import RedisRateLimiter

limiter = RedisRateLimiter(redis.Redis(host="localhost", port=6379), tokens_per_minute=10_000, tokens_per_hour=1_000_000)
agent = Agent(provider=provider, system_prompt="...", rate_limiter_context=limiter)
```

### A complete trace, worked through by hand

Given the `text-summarizer` skill shown earlier and a run that only ever invokes it, `AgentResponse.messages` looks like this (rendered as JSON for readability — the real objects are `SkillOutput` dataclass instances):

```json
[
  {
    "type": "text-summarizer",
    "status": "success",
    "input": { "skill_name": "text-summarizer" },
    "data": "## Text Summarization Instructions\n\nYou are an expert editor...",
    "error": null,
    "duration_ms": 1.2
  },
  {
    "type": "text-summarizer",
    "status": "success",
    "input": null,
    "data": {
      "summary": "The article covers three recent changes to the API and their migration paths.",
      "key_points": [
        "Endpoint X is deprecated in favor of Y",
        "Auth tokens now expire after 24 hours",
        "Batch requests are capped at 100 items"
      ]
    },
    "error": null,
    "duration_ms": 612.4
  }
]
```

Reading it: the first entry is the dispatch — the skill's raw markdown, unvalidated, `input` naming the call that produced it. The second entry is the bound completion — same `type`, `input: null` because it didn't originate from a distinct tool call, `data` already validated against the skill's `output` schema (`test_skill_with_output_binds_next_completion` asserts this exact shape). `AgentResponse.response` is neither of these — it's whatever the model wrote afterward, under `output_schema` if one was set, having seen both entries.

### Summarization

```python
from orchestration_agent import InMemoryHistory

history = InMemoryHistory()
agent = Agent(provider=provider, system_prompt="...", conversation_history=history)

await agent.run("What's the best practice for database indexing?")

summary = await history.summarize(
    model_provider=provider,
    system_prompt="Create a concise summary.",
    session_id=agent.session_id,
)
```

`system_prompt` is the caller's only real lever over `summarize()`'s output — `summarize_messages` wraps it around the formatted history and calls `model_provider.complete()` with no tools and no structured output, so whatever style the prompt asks for is exactly what comes back as plain text. Useful strategies to pass:

```
Create a bullet-point summary of this conversation.
Focus on key topics discussed and conclusions reached.
Keep each point to one sentence maximum.
```

```
Write a professional executive summary suitable for stakeholders.
Include: main topics and objectives, key findings and insights,
recommendations or next steps, estimated effort or impact.
```

```
Create meeting notes from this conversation.
Format: Participants (inferred), Agenda items discussed,
Decisions made, Action items (owner: person/date).
```

Common patterns built on top of the same primitive:

- **Archive, then clear.** Call `summarize()`, store the result alongside `session_id`, then `await history.clear(session_id=session_id)` to drop the full transcript once the summary is safely stored.
- **Context for a new session.** Summarize each of a user's prior `session_id`s and fold the resulting strings into a new agent's `system_prompt` before its first `run()` — `summarize()` never reads or writes `ConversationHistory` beyond the one `get_all()` call, so this composes freely across sessions.
- **Multiple sessions, one shared history object.** `MongoDBConversationHistory.summarize(session_id=...)` reads are independent per `session_id` and share no mutable state, so summarizing two different sessions concurrently (`asyncio.gather(...)`) against the same `MongoDBConversationHistory` instance is safe. `InMemoryHistory` has no concurrency concerns either way — it's a single process-local list.
- **Empty history is free.** `summarize()` returns the literal string `"No conversation history to summarize."` without ever calling `model_provider.complete()` when `get_all()` comes back empty (`test_in_memory_summarize_empty_history` asserts `provider.calls == []`) — safe to call speculatively without worrying about wasted tokens on a session that never started.

---

## Conformance Checklist

1. `SkillLoader` MUST read `output` verbatim and MUST NOT validate it in any way.
2. Only `orchestration_agent.scripts.validate_skill` (the `validate-skill` CLI) MUST parse `output` as a JSON Schema document, check it against the Draft 2020-12 meta-schema, reject non-local `$ref`s, and classify it strict/permissive; this MUST NOT run inside `Agent.__init__` or `SkillLoader`.
3. The skill catalogue serialized via `{{skills_catalog}}`/`{{ skills_catalog }}` MUST publish only `name` and `description`; `output` MUST NOT be transmitted before a bound completion.
4. `invoke_skill` MUST accept exactly `skill_name`, provider-injected on every call regardless of the agent's registered `tools`.
5. A missing `skill_name` MUST produce `SkillOutput(type="invoke_skill", status="error", error.code="SKILL_NOT_FOUND")`; an unresolved one MUST produce `SkillOutput(type=<given name>, ...)` with the same code.
6. External tools MUST be exposed to the model as their own named functions — never through a generic `invoke_tool(tool_name, tool_parameters)` wrapper — and MUST be dispatched by matching `tool_call.name` against the registered tool dict.
7. External tool calls MUST be recorded on `AgentResponse.steps`, never on `AgentResponse.messages`, and MUST NOT set or clear a skill's pending output contract.
8. A tool's return value MUST be validated against its `output_schema` before being handed back to the provider as the tool result; a validation failure MUST be formatted via `error_schema`, the same as an input-validation or execution failure.
9. The provider's own assistant turn (its `raw_message`) MUST be replayed verbatim once per round; each reply MUST be addressed to that call's own ID; every `tool_calls` entry MUST get a reply.
10. Where a skill declares `output`, only the **last** output-declaring dispatch in a round MUST bind the next completion; that completion MUST be issued with `tools=None`.
11. The bound completion's result MUST be validated (`_validate_payload`) before entering `SkillOutput.data`; on failure, exactly one repair (feeding violations back) MUST be attempted before giving up with `OUTPUT_VALIDATION_ERROR`.
12. A provider schema-compilation rejection (`SchemaCompilationError`) MUST trigger the prompt-injection fallback invisibly (no extra `SkillOutput` entry); a content refusal (`ModelRefusalError`) MUST map to `MODEL_REFUSAL` and MUST NOT be treated as a schema rejection.
13. `RateLimiterContext.check_and_consume` MUST be called synchronously with the step's own token estimate/delta, never the cumulative running total; `release` MUST return unused reservation.
14. `Agent.run()`'s ordinary loop MUST reserve at least one turn (`max(budget.max_turns - 1, 0)`) for a final, tools-disabled answer, given via `_final_turn_and_finish` on budget or timeout exhaustion, with no repair attempted there.
15. `output_schema`, where set, MUST be bound as structured output on every ordinary turn (alongside whatever tools are offered), not only on a dedicated final turn.
16. `ConversationHistory.summarize()` MUST retrieve via `get_all`, render via `serialize_for_prompt`/`history_formatter`, and generate via the given `model_provider`, returning `"No conversation history to summarize."` without a provider call when history is empty.
17. A run's in-flight state (`_RunState`: trace, steps, token usage) MUST be scoped to that `run()` call and MUST NOT be conflated with `ConversationHistory`.
18. Exhausting transport-level retries (`RetryableError`) MUST surface, uncaught by anything more specific, as `status="error"`, `error.code="AGENT_ERROR"`.
19. Multiple `invoke_skill`/tool calls in one round MUST be dispatched sequentially, in the order the provider returned them, and MUST appear in `messages`/`steps` in that same order — this implementation does not dispatch a round concurrently.
20. `messages` MUST be append-only, ordered by dispatch, and MUST NOT be reordered, deduplicated, or removed — including failed dispatches and failed bound completions.
