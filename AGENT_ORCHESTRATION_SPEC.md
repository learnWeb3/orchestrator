# Orchestration Agent Architecture Specification

**Version**: 1.0  
**Status**: Implementation-Ready  
**Last Updated**: 2026-09-13

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Design Principles](#architecture--design-principles)
3. [Skill System & APM Compliance](#skill-system--apm-compliance)
4. [System Prompt Injection](#system-prompt-injection)
5. [Agent Interface & Lifecycle](#agent-interface--lifecycle)
6. [Provider Abstraction Layer](#provider-abstraction-layer)
7. [Token Tracking & Rate Limiting Integration](#token-tracking--rate-limiting-integration)
8. [Retry & Error Handling Strategy](#retry--error-handling-strategy)
9. [Skill Discovery & Loading Algorithm](#skill-discovery--loading-algorithm)
10. [Conversation History Interface](#conversation-history-interface)
11. [Agent Loop Pseudocode](#agent-loop-pseudocode)
12. [Structured Output Handling](#structured-output-handling)
13. [Logging & Observability (Pluggable)](#logging--observability-pluggable)
14. [File Structure & Organization](#file-structure--organization)
15. [pyproject.toml Configuration](#pyprojecttoml-configuration)
16. [Usage Patterns & Examples](#usage-patterns--examples)
17. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Goal
Build a **decoupled, opt-in orchestration agent** that:
- Loads skills from a local skill directory (SKILL.md files with APM metadata)
- Injects skill catalogs into system prompts for LLM awareness
- Orchestrates skill invocation based on LLM decisions
- Tracks token consumption (input + output + cache) for external rate limiting
- Supports structured output validation via Pydantic models
- Handles retries at the LLM call level (up to 3 retries = 4 total attempts)
- Enables provider abstraction (OpenAI-first, extensible)
- Maintains pluggable logging without data store assumptions

### Key Principles
1. **Decoupling**: Every component can be used independently
2. **Opt-in**: Skills, structured output, streaming, tool use are all optional
3. **Provider Abstraction**: Interface-driven, not hardcoded to OpenAI
4. **Token Transparency**: Every token consumption is tracked and exposed
5. **Simple Skill Chaining**: Skills can call other skills; depth managed by `max_steps`
6. **Rate Limiter Integration**: Token tracking feeds external bucket-based rate limiter
7. **One-shot Model**: No multi-turn conversation for v1 (can extend later)

---

## Architecture & Design Principles

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Agent.run(prompt)    │
         └────────────┬───────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │ Load skills from skills_path        │
    │ Inject {{skills_catalog}} into      │
    │ system_prompt                       │
    └─────────────┬───────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────────┐
    │ Agent.Step(0):                          │
    │  - Build messages list                  │
    │  - Call provider.complete()             │
    │  - Track token_usage                    │
    │  - Check against rate limiter           │
    └─────────┬───────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────┐
    │ LLM Response                 │
    │ (text or skill invocation)   │
    └─────────┬────────────────────┘
              │
          ┌───┴───┐
          │       │
    ┌─────▼──┐ ┌──▼──────────┐
    │ Text?  │ │ Skill Call? │
    └────┬───┘ └──┬───────────┘
         │        │
    ┌────▼──────┐ ▼
    │ Return    │ Invoke skill(s)
    │ response  │ ├─ Load SKILL.md
    └───────────┘ ├─ Execute skill code
                  ├─ Track skill token_usage
                  └─ Skill can call other skills
                        │
                        ▼
                  ┌──────────────────┐
                  │ Skill Result     │
                  │ + token_usage    │
                  └─────┬────────────┘
                        │
                        ▼
                  ┌──────────────────────────┐
                  │ Step count ++            │
                  │ Retry count reset        │
                  │ Accumulate tokens        │
                  └─────┬────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        │                                │
    ┌───▼────────┐            ┌─────────▼──┐
    │ max_steps? │            │ Loop again │
    └───┬────────┘            └────────────┘
        │
   ┌────▼─────┐
   │ Yes      │
   └────┬─────┘
        │
        ▼
   ┌─────────────────────────┐
   │ Return AgentResponse    │
   │ (output, steps, tokens) │
   └─────────────────────────┘
```

### Component Hierarchy

```
BaseProvider (abstract)
├── OpenAIProvider
├── AnthropicProvider (future)
└── OllamaProvider (future)

BaseAgent (abstract)
├── Agent (concrete implementation)

BaseModel (Pydantic)
├── SkillMetadata (APM-compliant)
├── TokenUsage
├── ToolCall
├── StepRecord
├── AgentResponse
├── RateLimiterContext

SkillLoader (concrete)
├── load_skills_from_path()
├── parse_yaml_frontmatter()
└── validate_apm_schema()

SkillRegistry (concrete)
├── register()
├── get_by_name()
├── get_all()
└── serialize_to_prompt()

Logger (pluggable interface)
├── DefaultLogger (no-op)
└── (User can inject custom)
```

---

## External Tools System

### BaseTool Abstract Interface

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, Any, Optional


class BaseError(BaseModel):
    """Default error response schema."""
    error: str
    details: Optional[str] = None


class BaseTool(ABC):
    """
    Abstract base class for executable external tools.
    
    Tools are code-based, executable units of work that:
    - Accept structured input (Pydantic BaseModel)
    - Execute deterministically with parameters
    - Return structured output
    - Can define custom error schemas
    
    Designed for fastMCP compatibility and provider-agnostic execution.
    """
    
    name: str
    """Unique tool identifier (e.g., 'fetch_user', 'send_email')"""
    
    description: str
    """Clear description of what the tool does"""
    
    input_schema: Type[BaseModel]
    """Pydantic model defining required and optional parameters"""
    
    error_schema: Type[BaseModel] = BaseError
    """Pydantic model for error responses (default: BaseError)"""
    
    @abstractmethod
    async def execute(self, input: BaseModel) -> Any:
        """
        Execute the tool with validated parameters.
        
        Args:
            input: Validated instance of input_schema
        
        Returns:
            Tool result (any JSON-serializable type)
        
        Raises:
            Exception: Tool raises on execution failure
                      Agent catches and formats using error_schema
        """
        pass


class ToolError(Exception):
    """Base exception for tool execution errors."""
    pass
```

### Tool Implementation Examples

```python
from pydantic import BaseModel, Field

# Example 1: Fetch User Tool

class FetchUserInput(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    include_metadata: bool = Field(default=False, description="Include extra metadata")

class FetchUserError(BaseModel):
    error: str
    user_id: Optional[str] = None
    details: Optional[str] = None

class FetchUserTool(BaseTool):
    name = "fetch_user"
    description = "Retrieve user information from database by user_id"
    input_schema = FetchUserInput
    error_schema = FetchUserError
    
    async def execute(self, input: FetchUserInput) -> dict:
        """Execute tool - fetch user from database."""
        try:
            user = await database.get_user(input.user_id)
            if not user:
                raise ValueError(f"User {input.user_id} not found")
            
            result = {"id": user.id, "email": user.email, "name": user.name}
            
            if input.include_metadata:
                result["metadata"] = user.metadata
            
            return result
        except Exception as e:
            # Raise - agent will catch and format using error_schema
            raise ToolError(str(e)) from e


# Example 2: Send Email Tool

class SendEmailInput(BaseModel):
    recipient: str = Field(..., description="Email recipient address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    html: bool = Field(default=False, description="Is body HTML formatted")

class SendEmailResult(BaseModel):
    success: bool
    message_id: Optional[str] = None
    sent_at: Optional[str] = None

class SendEmailTool(BaseTool):
    name = "send_email"
    description = "Send an email via SMTP"
    input_schema = SendEmailInput
    error_schema = BaseError  # Use default error schema
    
    async def execute(self, input: SendEmailInput) -> SendEmailResult:
        """Execute tool - send email."""
        try:
            message_id = await email_service.send(
                to=input.recipient,
                subject=input.subject,
                body=input.body,
                html=input.html
            )
            
            return SendEmailResult(
                success=True,
                message_id=message_id,
                sent_at=datetime.utcnow().isoformat()
            )
        except Exception as e:
            raise ToolError(str(e)) from e
```

---

## Skill System

### SKILL.md Format

Each skill is a single file: `SKILL.md` with **minimal YAML frontmatter** + **detailed markdown instructions**.

The skill instructions in markdown become the **tool result content** passed to the LLM when invoked.

#### Example: `skills/code-review/SKILL.md`

```markdown
---
name: code-review
description: Review code for security issues, best practices, and potential bugs
compatibility: "python>=3.10, openai, anthropic"
---

## Code Review Instructions

You are an expert code reviewer specializing in security and best practices.

When asked to review code:

1. **Security Analysis**
   - Check for SQL injection vulnerabilities
   - Identify authentication/authorization flaws
   - Look for XSS vulnerabilities
   - Review data handling and encryption
   - Check for secrets/credentials in code

2. **Code Quality**
   - Review naming conventions
   - Identify code duplication
   - Check error handling
   - Assess performance implications
   - Review test coverage

3. **Best Practices**
   - Check compliance with language conventions
   - Identify anti-patterns
   - Review dependency usage
   - Check logging and monitoring

4. **Report Format**
   - List each issue with severity (Critical/High/Medium/Low)
   - Provide specific line references
   - Suggest concrete improvements
   - Include code examples where helpful

Focus on actionable feedback that improves security and maintainability.
```

#### Example: `skills/sql-analysis/SKILL.md`

```markdown
---
name: sql-analysis
description: Analyze SQL queries for performance, correctness, and security issues
compatibility: "python>=3.10, openai, anthropic, ollama"
---

## SQL Analysis Instructions

You are a database expert specializing in SQL optimization and security.

When analyzing SQL queries:

1. **Security Review**
   - Check for SQL injection vulnerabilities
   - Verify parameterized queries are used
   - Review permission requirements

2. **Performance Analysis**
   - Identify missing indexes
   - Check for N+1 queries
   - Review join strategies
   - Assess query complexity

3. **Correctness**
   - Verify logic accuracy
   - Check for edge cases
   - Review transaction handling
   - Validate data types

4. **Optimization Recommendations**
   - Suggest indexes
   - Recommend query refactoring
   - Identify redundant subqueries
   - Propose caching strategies

Provide detailed explanations with specific SQL examples.
```

### SKILL.md Structure Rules

1. **YAML Frontmatter** (required):
   - `name` (string): Unique skill identifier
   - `description` (string): One-line description shown to LLM in skill catalog
   - `compatibility` (string): Comma-separated requirements (informational only, no validation)

2. **Markdown Content** (required):
   - Detailed instructions for LLM on how to use this skill
   - Becomes the **full tool result content** when skill is invoked
   - Can include:
     - Step-by-step guidance
     - Examples and edge cases
     - Output format requirements
     - Context understanding instructions
     - Constraints and limitations

3. **Critical Points**:
   - **No input schemas**: LLM determines what to extract from conversation
   - **No output schemas**: LLM determines response format
   - **Full markdown becomes tool result**: Instructions must be complete enough for LLM to understand task

### SkillMetadata Pydantic Model

```python
class SkillMetadata(BaseModel):
    """Minimal skill metadata from SKILL.md frontmatter."""
    name: str  # Unique identifier
    description: str  # One-line description for catalog
    compatibility: str  # Comma-separated: "python>=3.10, openai, anthropic"
    content: str  # Full markdown content (instructions)
    
    class Config:
        extra = "forbid"  # No extra fields allowed
```

---

## System Prompt Injection

### Injection Mechanism

The agent automatically injects a `{{skills_catalog}}` placeholder in the system prompt with a **simple list of available skills**.

#### Template Substitution Algorithm

```python
def inject_skills_into_prompt(
    system_prompt: str,
    skills_registry: SkillRegistry
) -> str:
    """
    Replace {{skills_catalog}} with skill listing.
    """
    if "{{skills_catalog}}" not in system_prompt:
        return system_prompt
    
    catalog = serialize_skills_to_prompt(skills_registry)
    return system_prompt.replace("{{skills_catalog}}", catalog)


def serialize_skills_to_prompt(registry: SkillRegistry) -> str:
    """
    Convert SkillRegistry into simple catalog for system prompt.
    
    Returns:
        str: Simple skill listing with names and descriptions
    """
    if not registry.skills:
        return "[No skills available]"
    
    lines = ["## Available Skills\n"]
    
    for skill in registry.get_all():
        lines.append(f"- **{skill.name}**: {skill.description}")
    
    return "\n".join(lines)
```

#### Example Injected System Prompt

```
You are a helpful AI assistant with access to specialized tools.

## Available Skills

- **code-review**: Review code for security issues, best practices, and potential bugs
- **sql-analysis**: Analyze SQL queries for performance, correctness, and security issues
- **deploy**: Deploy applications to production environment
- **test-runner**: Execute and analyze test suites
- **documentation**: Generate and update technical documentation

## How to Use Skills

When you need to use a skill to help the user:

1. Call the invoke_skill function with the skill name
2. I will provide you with the skill's detailed instructions
3. Read and follow those instructions to complete the task
4. Apply the skill's guidance to the user's request

Example:
User: "Review this code for security issues"
You: [call invoke_skill with name="code-review"]
Tool: [provides detailed code review instructions]
You: [follow instructions and provide analysis]
```

### System Prompt Design Guidelines

1. **Keep it simple**: List skill names and descriptions only
2. **Explain the invoke_skill mechanism**: Tell LLM how to request skills
3. **Let instructions guide**: When skill is invoked, LLM receives detailed instructions
4. **Avoid parameter documentation**: No need—skills have no input parameters

### System Prompt Best Practices

```markdown
You are an AI assistant that can use specialized skills and external tools to help users.

You have access to skills (instruction-based) and tools (executable functions).

## Available Skills

{{ skills_catalog }}

## Using Skills vs Tools

### Skills (Instruction-Based, Business Logic)

Skills provide detailed guidance for complex tasks. They are business and process oriented.

When you need skill guidance, call invoke_skill with:
- skill_name: the name of the skill from the available skills list

Example:
- invoke_skill(skill_name="code-review") → I'll provide detailed code review instructions
- invoke_skill(skill_name="sql-optimization") → I'll provide SQL optimization guidance

Skills help you understand WHAT to do and WHY.

### Tools (Executable, Units of Work)

Tools execute specific operations with structured parameters. They are data-oriented and return immediate results.

When you need to execute an operation, call invoke_tool with:
- tool_name: the name of the tool
- tool_parameters: the required parameters as JSON object

Example:
- invoke_tool(tool_name="fetch_user", tool_parameters={"user_id": "123"}) → Returns user data
- invoke_tool(tool_name="send_email", tool_parameters={"recipient": "...", "subject": "...", "body": "..."}) → Sends email

Tools help you EXECUTE actions and get results.

## Decision Guide

Use **skills** when:
- You need detailed analysis or review (code, SQL, etc.)
- You need step-by-step guidance
- The task requires expertise and judgment

Use **tools** when:
- You need to fetch or query data
- You need to execute a specific operation
- You need immediate structured results

Both can be used in a single response to accomplish complex tasks.
```

---

## Agent Interface & Lifecycle

### BaseAgent Abstract Class

```python
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Type, Dict, Any, Callable
from pydantic import BaseModel
import asyncio


class BaseAgent(ABC):
    """
    Abstract base class for all agent implementations.
    
    Defines the contract that all agents must fulfill.
    """
    
    def __init__(
        self,
        provider: "BaseProvider",
        system_prompt: str,
        skills_path: Optional[str] = None,
        tools: Optional[List["BaseTool"]] = None,
        max_steps: int = 10,
        max_retries: int = 3,
        output_schema: Optional[Type[BaseModel]] = None,
        streaming: bool = False,
        backoff_strategy: Optional[Callable[[int], float]] = None,
        logger: Optional["Logger"] = None,
        rate_limiter_context: Optional["RateLimiterContext"] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize agent.
        
        Args:
            provider: LLM provider instance (OpenAI, Anthropic, etc.)
            system_prompt: Base system prompt (can contain {{skills_catalog}})
            skills_path: Path to skills directory (optional)
            tools: List of BaseTool instances (optional, external executable tools)
            max_steps: Maximum agent steps before halting (default: 10)
            max_retries: Max retries per LLM call (default: 3)
            output_schema: Pydantic model for structured output (optional)
            streaming: Enable streaming responses (default: False)
            backoff_strategy: Custom backoff function fn(attempt: int) -> float (default: exponential)
            logger: Custom logger instance (default: no-op)
            rate_limiter_context: Context for external rate limiter (optional)
            session_id: UUID string for conversation session (optional, auto-generated if not provided)
        """
        from uuid import uuid4
        
        self.provider = provider
        self.system_prompt = system_prompt
        self.skills_path = skills_path
        self.tools = {tool.name: tool for tool in (tools or [])}  # Internal dict for quick lookup
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.output_schema = output_schema
        self.streaming = streaming
        self.backoff_strategy = backoff_strategy or self._default_exponential_backoff
        self.logger = logger or NoOpLogger()
        self.rate_limiter_context = rate_limiter_context
        self.session_id = session_id or str(uuid4())  # Auto-generate UUID if not provided
        
        self.skills_registry: "SkillRegistry" = SkillRegistry()
        self._load_skills()
    
    @abstractmethod
    async def run(self, user_input: str) -> "AgentResponse":
        """
        Execute agent with user input (one-shot).
        
        Args:
            user_input: User query/prompt
        
        Returns:
            AgentResponse with output, steps, token tracking
        
        Raises:
            AgentError: On fatal failure
        """
        pass
    
    @abstractmethod
    async def _step(self, messages: List[Dict[str, str]]) -> "StepResult":
        """
        Execute a single agent step.
        
        Args:
            messages: Message history
        
        Returns:
            StepResult with LLM output and metadata
        """
        pass
    
    @abstractmethod
    async def invoke_skill(self, skill_name: str) -> str:
        """
        Invoke a skill by name.
        
        Args:
            skill_name: Name of skill to invoke
        
        Returns:
            Full skill markdown content (instructions)
        
        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        pass
    
    def _load_skills(self):
        """Load all SKILL.md files from skills_path if provided."""
        if not self.skills_path:
            return
        
        loader = SkillLoader()
        skills = loader.load_skills_from_path(self.skills_path)
        
        for skill_metadata in skills:
            self.skills_registry.register(skill_metadata)
        
        # Inject skills into system prompt
        self.system_prompt = inject_skills_into_prompt(
            self.system_prompt,
            self.skills_registry
        )
    
    @staticmethod
    def _default_exponential_backoff(attempt: int) -> float:
        """
        Default exponential backoff: 2^attempt seconds with jitter.
        
        Attempt 0: ~1 second
        Attempt 1: ~2 seconds
        Attempt 2: ~4 seconds
        """
        import random
        base_delay = 2 ** attempt
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
```

### Concrete Agent Implementation

```python
class Agent(BaseAgent):
    """Production-ready agent implementation."""
    
    async def run(self, user_input: str) -> "AgentResponse":
        """
        Execute one-shot agent run.
        
        Flow:
        1. Initialize message history with system prompt
        2. Loop up to max_steps:
           a. Call provider.complete() with retry logic
           b. Parse LLM response
           c. If skill invocation: execute skill, append to history
           d. If text response: return immediately
           e. Check step/token limits
        3. Return structured response
        
        Returns:
            AgentResponse with output, steps, cumulative tokens
        """
        self.logger.info(f"Agent.run() starting with input: {user_input[:100]}...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        cumulative_token_usage = TokenUsage()
        steps = []
        step_count = 0
        
        try:
            while step_count < self.max_steps:
                self.logger.debug(f"Step {step_count + 1}/{self.max_steps}")
                
                # Call LLM with retry
                step_result = await self._step_with_retry(messages, step_count)
                step_count += 1
                
                # Track tokens
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
                
                # Record step
                steps.append(step_result.step_record)
                
                # Check rate limiter
                if self.rate_limiter_context:
                    allowed = self.rate_limiter_context.check_and_consume(
                        tokens=cumulative_token_usage.total()
                    )
                    if not allowed:
                        self.logger.warning("Rate limit exceeded")
                        raise RateLimitExceededError(cumulative_token_usage.total())
                
                # Parse response
                if step_result.stop_reason == "end_turn" or not step_result.tool_calls:
                    # Final text response
                    output = step_result.content
                    
                    # Validate against schema if provided
                    if self.output_schema:
                        try:
                            validated = self.output_schema.model_validate_json(output)
                            output = validated
                        except Exception as e:
                            self.logger.error(f"Output validation failed: {e}")
                            raise StructuredOutputValidationError(str(e))
                    
                    return AgentResponse(
                        output=output,
                        steps=steps,
                        total_tokens=cumulative_token_usage.total(),
                        token_usage=cumulative_token_usage,
                        success=True,
                    )
                
                # Process tool calls: both invoke_skill and invoke_tool can appear
                # Append assistant's single response for all tool calls
                messages.append({
                    "role": "assistant",
                    "content": step_result.content,
                })
                
                skills_invoked = []
                tools_invoked = []
                
                for tool_call in step_result.tool_calls:
                    if tool_call.name == "invoke_skill":
                        # Instruction-based skill invocation
                        skill_name = tool_call.arguments.get("skill_name")
                        if not skill_name:
                            self.logger.error("invoke_skill called without skill_name")
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": "invoke_skill requires skill_name parameter"}),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "invoke_skill",
                            })
                            continue
                        
                        self.logger.debug(f"Invoking skill: {skill_name}")
                        
                        try:
                            # Get skill markdown content (instructions)
                            skill_content = await self.invoke_skill(skill_name)
                            
                            # Append skill content as tool result
                            # LLM will read these instructions and apply them
                            messages.append({
                                "role": "tool",
                                "content": skill_content,
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "invoke_skill",
                            })
                            
                            skills_invoked.append({"name": skill_name})
                            
                        except SkillNotFoundError as e:
                            self.logger.error(f"Skill not found: {skill_name}")
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": f"Skill '{skill_name}' not found"}),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "invoke_skill",
                            })
                        except Exception as e:
                            self.logger.error(f"Skill invocation failed: {e}")
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": f"Error invoking skill '{skill_name}': {str(e)}"}),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "invoke_skill",
                            })
                    
                    elif tool_call.name == "invoke_tool":
                        # Executable tool invocation
                        tool_name = tool_call.arguments.get("tool_name")
                        parameters = tool_call.arguments.get("tool_parameters", {})
                        
                        if not tool_name:
                            self.logger.error("invoke_tool called without tool_name")
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": "invoke_tool requires tool_name parameter"}),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "invoke_tool",
                            })
                            continue
                        
                        self.logger.debug(f"Invoking tool: {tool_name} with params: {parameters}")
                        
                        try:
                            # Execute tool with parameters
                            result = await self.invoke_tool(tool_name, parameters)
                            
                            # Append tool result (or error formatted by tool)
                            messages.append({
                                "role": "tool",
                                "content": json.dumps(result),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else tool_name,
                            })
                            
                            tools_invoked.append({
                                "name": tool_name,
                                "parameters": parameters,
                                "result": result,
                            })
                            
                        except ToolNotFoundError as e:
                            self.logger.error(f"Tool not found: {tool_name}")
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": f"Tool '{tool_name}' not found"}),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else tool_name,
                            })
                        except Exception as e:
                            self.logger.error(f"Tool invocation failed: {e}")
                            messages.append({
                                "role": "tool",
                                "content": json.dumps({"error": f"Error invoking tool '{tool_name}': {str(e)}"}),
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else tool_name,
                            })
                    
                    else:
                        self.logger.warning(f"Unknown tool: {tool_call.name}")
                
                # Track in conversation history if any skills or tools were invoked
                if skills_invoked or tools_invoked:
                    await self._add_to_history(
                        "assistant",
                        step_result.content,
                        skills_invoked=skills_invoked if skills_invoked else None,
                        tools_invoked=tools_invoked if tools_invoked else None,
                    )
            
            # Max steps reached
            return AgentResponse(
                output="Max steps reached without final response",
                steps=steps,
                total_tokens=cumulative_token_usage.total(),
                token_usage=cumulative_token_usage,
                success=False,
                error="max_steps_exceeded",
            )
        
        except Exception as e:
            self.logger.error(f"Agent.run() failed: {e}")
            return AgentResponse(
                output=None,
                steps=steps,
                total_tokens=cumulative_token_usage.total(),
                token_usage=cumulative_token_usage,
                success=False,
                error=str(e),
            )
    
    async def _step_with_retry(
        self,
        messages: List[Dict[str, str]],
        step_count: int,
    ) -> "StepResult":
        """
        Execute single step with retry logic.
        
        Retries up to max_retries times on:
        - Rate limit errors (429)
        - Temporary errors (500, 503)
        - Structured output validation failures
        
        Uses exponential backoff with jitter.
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                self.logger.debug(f"Step attempt {attempt + 1}/{self.max_retries + 1}")
                
                step_result = await self._step(messages)
                return step_result
            
            except (RateLimitError, TemporaryProviderError, StructuredOutputValidationError) as e:
                last_error = e
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_strategy(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
                    raise
        
        raise last_error
    
    async def _step(self, messages: List[Dict[str, str]]) -> "StepResult":
        """Execute single LLM call."""
        completion = await self.provider.complete(
            messages=messages,
            system_prompt=self.system_prompt,
            model=self.provider.model,
            temperature=0.7,
            max_tokens=2048,
            structured_output=self.output_schema,
            stream=self.streaming,
            tools=list(self.tools.values()) if self.tools else None,  # Pass external tools to provider
        )
        
        # Extract content, tool calls, and tokens
        content = completion.content
        tool_calls = completion.tool_calls
        token_usage = completion.token_usage
        stop_reason = completion.stop_reason
        
        # Build step record
        step_record = StepRecord(
            step_number=len([]),  # Would be passed from caller
            lm_call={
                "model": self.provider.model,
                "messages": messages,
                "stop_reason": stop_reason,
            },
            tool_calls=[
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_calls
            ],
            token_usage=token_usage,
        )
        
        return StepResult(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            stop_reason=stop_reason,
            step_record=step_record,
        )
    
    async def invoke_skill(self, skill_name: str) -> str:
        """
        Invoke skill by name.
        
        Returns the full skill markdown content (instructions) to be used as
        tool result in the LLM conversation. The LLM then reads the instructions
        and applies them to the conversation context.
        
        Args:
            skill_name: Name of the skill to invoke
        
        Returns:
            Full markdown content of the skill (instructions)
        
        Raises:
            SkillNotFoundError: If skill doesn't exist
        """
        skill_metadata = self.skills_registry.get_by_name(skill_name)
        if not skill_metadata:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found")
        
        self.logger.info(f"Invoking skill: {skill_name}")
        
        # Return full skill content (markdown instructions)
        # This becomes the tool result that LLM receives
        return skill_metadata.content
    
    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Union[Dict[str, Any], Any]:
        """
        Invoke executable tool by name with parameters.
        
        Validates parameters against tool's input_schema, executes tool,
        and handles errors using tool's error_schema.
        
        Args:
            tool_name: Name of the tool to invoke
            parameters: Parameters matching tool's input_schema
        
        Returns:
            Tool result (JSON-serializable)
            On error: formatted error using tool's error_schema
        
        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        tool = self.tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")
        
        self.logger.info(f"Invoking tool: {tool_name} with parameters: {parameters}")
        
        try:
            # Validate parameters against input_schema
            validated_input = tool.input_schema(**parameters)
        except Exception as e:
            # Parameter validation error - return as error tool result
            self.logger.error(f"Tool {tool_name} parameter validation failed: {e}")
            error_instance = tool.error_schema(
                error="Invalid parameters",
                details=str(e)
            )
            return error_instance.model_dump()
        
        try:
            # Execute tool with validated parameters
            result = await tool.execute(validated_input)
            return result
        except Exception as e:
            # Tool execution error - format using error_schema
            self.logger.error(f"Tool {tool_name} execution failed: {e}")
            error_instance = tool.error_schema(
                error=str(e),
                details=f"Execution failed in {tool_name}"
            )
            return error_instance.model_dump()
```

---

## Provider Abstraction Layer

### BaseProvider Abstract Class

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type, AsyncIterator
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token consumption tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    
    def total(self) -> int:
        """Total tokens consumed (excluding cache)."""
        return self.input_tokens + self.output_tokens
    
    def total_with_cache(self) -> int:
        """Total tokens including cache operations."""
        total = self.input_tokens + self.output_tokens
        if self.cache_creation_tokens:
            total += self.cache_creation_tokens
        if self.cache_read_tokens:
            total += self.cache_read_tokens
        return total


@dataclass
class ToolCall:
    """Represents a tool/skill invocation from LLM."""
    name: str
    arguments: Dict[str, Any]  # Function arguments as dict


@dataclass
class CompletionResponse:
    """Response from provider.complete()."""
    content: str  # LLM-generated text
    tool_calls: List[ToolCall] = None  # If tool use enabled
    token_usage: TokenUsage = None
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", "length", etc.
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.token_usage is None:
            self.token_usage = TokenUsage()


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Defines the contract for interacting with any LLM backend.
    """
    
    def __init__(self, model: str, api_key: str, **kwargs):
        """
        Initialize provider.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-sonnet")
            api_key: Authentication token
            **kwargs: Provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        structured_output: Optional[Type[BaseModel]] = None,
        stream: bool = False,
        tools: Optional[List["BaseTool"]] = None,
        **kwargs,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """
        Generate completion from LLM.
        
        Args:
            messages: Message history [{"role": "user"|"assistant", "content": str}, ...]
            system_prompt: System prompt
            model: Model to use (may override self.model)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens
            structured_output: Pydantic model for output validation
            stream: Enable streaming response
            tools: List of BaseTool instances (provider converts to target format)
            **kwargs: Provider-specific options
        
        Returns:
            CompletionResponse (or AsyncIterator if streaming)
        
        Raises:
            RateLimitError: If rate limited (429)
            TemporaryProviderError: If temporary error (500, 503)
            StructuredOutputValidationError: If output doesn't match schema
        """
        pass
```

### OpenAI Provider Implementation

```python
import json
import openai
from openai import AsyncOpenAI, APIError, RateLimitError as OpenAIRateLimitError


class OpenAIProvider(BaseProvider):
    """OpenAI API provider (GPT-4, GPT-4o, etc.)."""
    
    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        structured_output: Optional[Type[BaseModel]] = None,
        stream: bool = False,
        tools: Optional[List["BaseTool"]] = None,
        **kwargs,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponse]]:
        """
        Call OpenAI API with invoke_skill and tool support.
        
        Handles:
        - Message formatting with system prompt
        - invoke_skill tool registration (for skills)
        - External tool schema conversion (provider gateway role)
        - Token tracking
        - Structured output validation
        """
        
        # Prepend system prompt to messages
        full_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        
        # Prepare completion parameters
        completion_params = {
            "model": model or self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        
        # Build tools list: invoke_skill + external tools
        tool_list = []
        
        # Register invoke_skill tool (for instruction-based skills)
        invoke_skill_tool = {
            "type": "function",
            "function": {
                "name": "invoke_skill",
                "description": "Invoke a skill from the available skill catalog",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill to invoke"
                        }
                    },
                    "required": ["skill_name"]
                }
            }
        }
        tool_list.append(invoke_skill_tool)
        
        # Convert external tools to OpenAI function schema
        if tools:
            for tool in tools:
                tool_function = self._convert_tool_to_openai_schema(tool)
                tool_list.append(tool_function)
        
        completion_params["tools"] = tool_list if tool_list else None
        
        # Add structured output if provided
        if structured_output:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": structured_output.__name__,
                    "schema": structured_output.model_json_schema(),
                    "strict": True,
                },
            }
        
        # Add any provider-specific options
        completion_params.update(kwargs)
        
        try:
            if stream:
                return self._stream_completion(completion_params, structured_output)
            else:
                return await self._non_stream_completion(completion_params, structured_output)
        
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit: {e}") from e
        except APIError as e:
            if e.status_code in [500, 503]:
                raise TemporaryProviderError(f"OpenAI temporary error: {e}") from e
            raise
    
    async def _non_stream_completion(
        self,
        params: Dict[str, Any],
        structured_output: Optional[Type[BaseModel]],
    ) -> CompletionResponse:
        """Non-streaming completion."""
        response = await self.client.chat.completions.create(**params)
        
        # Extract content
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Parse tool calls (invoke_skill)
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == "invoke_skill":
                    tool_calls.append(
                        ToolCall(
                            name="invoke_skill",
                            arguments=json.loads(tc.function.arguments),
                        )
                    )
        
        # Track tokens
        token_usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", None),
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", None),
        )
        
        # Validate structured output if required
        if structured_output and content:
            try:
                structured_output.model_validate_json(content)
            except Exception as e:
                raise StructuredOutputValidationError(str(e)) from e
        
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            stop_reason=choice.finish_reason,
        )
    
    def _convert_tool_to_openai_schema(self, tool: "BaseTool") -> Dict[str, Any]:
        """
        Convert BaseTool to OpenAI function schema.
        
        Provider acts as gateway: converts domain-agnostic tool definitions
        to provider-specific format (OpenAI functions in this case).
        
        Args:
            tool: BaseTool instance with Pydantic input_schema
        
        Returns:
            OpenAI function schema
        """
        # Get JSON schema from Pydantic model
        input_schema = tool.input_schema.model_json_schema()
        
        # Convert to OpenAI function parameters format
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": input_schema.get("properties", {}),
                    "required": input_schema.get("required", []),
                }
            }
        }
    
    async def _stream_completion(
        self,
        params: Dict[str, Any],
        structured_output: Optional[Type[BaseModel]],
    ) -> AsyncIterator[CompletionResponse]:
        """Streaming completion."""
        stream = await self.client.chat.completions.create(**params)
        
        accumulated_content = ""
        total_input_tokens = 0
        total_output_tokens = 0
        
        async for event in stream:
            choice = event.choices[0]
            delta = choice.delta
            
            # Accumulate content
            if delta.content:
                accumulated_content += delta.content
                yield CompletionResponse(
                    content=accumulated_content,
                    token_usage=TokenUsage(
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    ),
                    stop_reason="streaming",
                )
            
            # Extract final token counts (if available)
            if hasattr(event, "usage") and event.usage:
                total_input_tokens = event.usage.prompt_tokens
                total_output_tokens = event.usage.completion_tokens
```

### Provider Interface Contract

```python
class RateLimitError(Exception):
    """Raised when provider rate limits are hit."""
    pass


class TemporaryProviderError(Exception):
    """Raised on temporary provider errors (500, 503)."""
    pass


class StructuredOutputValidationError(Exception):
    """Raised when structured output doesn't match schema."""
    pass
```

---

## Token Tracking & Rate Limiting Integration

### TokenUsage Model (Already Defined Above)

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    
    def total(self) -> int:
        """Total input + output tokens (primary for rate limiting)."""
        return self.input_tokens + self.output_tokens
    
    def total_with_cache(self) -> int:
        """Total including cache operations."""
        total = self.total()
        if self.cache_creation_tokens:
            total += self.cache_creation_tokens
        if self.cache_read_tokens:
            total += self.cache_read_tokens
        return total
```

### RateLimiterContext Interface

```python
from abc import ABC, abstractmethod


class RateLimiterContext(ABC):
    """
    Interface for external rate limiter integration.
    
    Agent checks this before/after LLM calls to respect rate limits.
    Implementation uses bucket-based or similar algorithm in Redis/DB.
    """
    
    @abstractmethod
    def check_and_consume(self, tokens: int) -> bool:
        """
        Check if tokens can be consumed and consume them if allowed.
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if consumed, False if rate limited
        
        Raises:
            RateLimiterError: On backend failure
        """
        pass
    
    @abstractmethod
    def get_current_bucket(self) -> Dict[str, Any]:
        """Get current rate limiter state."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset rate limiter state."""
        pass


class RedisRateLimiter(RateLimiterContext):
    """
    Example: Redis-backed bucket-based rate limiter.
    
    Tracks tokens_per_minute, tokens_per_hour, etc.
    """
    
    def __init__(
        self,
        redis_client,
        tokens_per_minute: int = 10000,
        tokens_per_hour: int = 1000000,
        key_prefix: str = "rate_limit:agent:",
    ):
        self.redis = redis_client
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_hour = tokens_per_hour
        self.key_prefix = key_prefix
    
    def check_and_consume(self, tokens: int) -> bool:
        """Check rate limit buckets and consume tokens."""
        # Pseudo-implementation (actual Redis logic omitted)
        minute_key = f"{self.key_prefix}minute"
        hour_key = f"{self.key_prefix}hour"
        
        minute_tokens = self.redis.get(minute_key) or 0
        hour_tokens = self.redis.get(hour_key) or 0
        
        if minute_tokens + tokens > self.tokens_per_minute:
            return False
        if hour_tokens + tokens > self.tokens_per_hour:
            return False
        
        self.redis.incr(minute_key, tokens)
        self.redis.incr(hour_key, tokens)
        self.redis.expire(minute_key, 60)
        self.redis.expire(hour_key, 3600)
        
        return True
    
    def get_current_bucket(self) -> Dict[str, Any]:
        """Get current consumption state."""
        minute_key = f"{self.key_prefix}minute"
        hour_key = f"{self.key_prefix}hour"
        return {
            "minute_tokens": self.redis.get(minute_key) or 0,
            "minute_limit": self.tokens_per_minute,
            "hour_tokens": self.redis.get(hour_key) or 0,
            "hour_limit": self.tokens_per_hour,
        }
    
    def reset(self):
        """Reset buckets."""
        minute_key = f"{self.key_prefix}minute"
        hour_key = f"{self.key_prefix}hour"
        self.redis.delete(minute_key)
        self.redis.delete(hour_key)
```

### Agent Integration

```python
# In Agent.__init__
self.rate_limiter_context = rate_limiter_context

# In Agent.run() loop
if self.rate_limiter_context:
    allowed = self.rate_limiter_context.check_and_consume(
        tokens=cumulative_token_usage.total()
    )
    if not allowed:
        raise RateLimitExceededError(
            f"Rate limit exceeded. Current: {self.rate_limiter_context.get_current_bucket()}"
        )
```

---

## Retry & Error Handling Strategy

### Retry Logic

```python
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3  # 0-indexed, so 4 total attempts
    backoff_multiplier: float = 2.0
    backoff_base_seconds: float = 1.0
    jitter_ratio: float = 0.1  # 10% jitter


def exponential_backoff(attempt: int, config: RetryConfig) -> float:
    """Calculate wait time for exponential backoff with jitter."""
    import random
    wait = config.backoff_base_seconds * (config.backoff_multiplier ** attempt)
    jitter = random.uniform(0, wait * config.jitter_ratio)
    return wait + jitter
```

### Retry Trigger Conditions

```python
class RetryableError(Exception):
    """Base class for errors that trigger retry."""
    pass


class RateLimitError(RetryableError):
    """HTTP 429 - Too Many Requests."""
    pass


class TemporaryProviderError(RetryableError):
    """HTTP 500, 503, etc."""
    pass


class StructuredOutputValidationError(RetryableError):
    """Structured output doesn't match schema."""
    pass


# Non-retryable errors
class SkillNotFoundError(Exception):
    """Skill doesn't exist."""
    pass


class InvalidParametersError(Exception):
    """Skill parameters don't match schema."""
    pass


class ToolNotFoundError(Exception):
    """Tool doesn't exist."""
    pass


class ToolError(Exception):
    """Tool execution error."""
    pass
```

### Retry Flow (in Agent._step_with_retry)

```python
async def _step_with_retry(
    self,
    messages: List[Dict[str, str]],
) -> StepResult:
    """
    Retry logic for a single LLM step.
    
    Retry up to max_retries times (so max_retries + 1 total attempts)
    on retryable errors with exponential backoff.
    """
    for attempt in range(self.max_retries + 1):
        try:
            self.logger.debug(f"Step attempt {attempt + 1}/{self.max_retries + 1}")
            
            result = await self._step(messages)
            return result
        
        except RetryableError as e:
            if attempt < self.max_retries:
                wait_time = self.backoff_strategy(attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed with "
                    f"retryable error: {e}. Waiting {wait_time:.2f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                self.logger.error(
                    f"All {self.max_retries + 1} attempts exhausted. Final error: {e}"
                )
                raise
        
        except Exception as e:
            # Non-retryable error
            self.logger.error(f"Non-retryable error: {e}")
            raise
```

---

## Skill Discovery & Loading Algorithm

### SkillLoader Implementation

```python
import yaml
from pathlib import Path


class SkillLoader:
    """Load and parse SKILL.md files from disk."""
    
    def load_skills_from_path(self, skills_path: str) -> List[SkillMetadata]:
        """
        Discover and load all SKILL.md files from directory.
        
        Scans directory recursively for **/*.md files, parses YAML frontmatter
        and markdown content.
        
        Args:
            skills_path: Path to skills directory
        
        Returns:
            List of SkillMetadata objects
        
        Raises:
            ValueError: On invalid YAML or missing required fields
        """
        skills = []
        skills_dir = Path(skills_path)
        
        if not skills_dir.exists():
            raise ValueError(f"Skills directory not found: {skills_path}")
        
        # Find all SKILL.md files
        for md_file in skills_dir.rglob("SKILL.md"):
            try:
                metadata = self.parse_skill_file(md_file)
                skills.append(metadata)
                print(f"✓ Loaded skill: {metadata.name}")
            except Exception as e:
                print(f"✗ Failed to load {md_file}: {e}")
                # Continue loading other skills
        
        return skills
    
    def parse_skill_file(self, file_path: Path) -> SkillMetadata:
        """
        Parse SKILL.md file and extract YAML frontmatter + markdown content.
        
        Format:
        ---
        name: skill_name
        description: One-line description
        compatibility: "python>=3.10, openai, anthropic"
        ---
        
        ## Detailed Instructions
        
        Full markdown instructions that will be used as tool result content.
        
        Args:
            file_path: Path to SKILL.md file
        
        Returns:
            SkillMetadata object
        
        Raises:
            ValueError: If YAML is invalid or required fields missing
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract YAML frontmatter
        if not content.startswith("---"):
            raise ValueError(f"SKILL.md must start with --- (found in {file_path})")
        
        # Find closing ---
        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"SKILL.md must have closing --- delimiter (found in {file_path})")
        
        yaml_str = parts[1]
        markdown_content = parts[2].strip()
        
        # Parse YAML
        try:
            yaml_data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
        
        # Validate required fields
        required_fields = ["name", "description", "compatibility"]
        for field in required_fields:
            if field not in yaml_data:
                raise ValueError(f"Missing required field '{field}' in {file_path}")
        
        # Validate field types
        if not isinstance(yaml_data["name"], str):
            raise ValueError(f"Field 'name' must be string in {file_path}")
        if not isinstance(yaml_data["description"], str):
            raise ValueError(f"Field 'description' must be string in {file_path}")
        if not isinstance(yaml_data["compatibility"], str):
            raise ValueError(f"Field 'compatibility' must be comma-separated string in {file_path}")
        
        # Create SkillMetadata with full markdown content
        try:
            metadata = SkillMetadata(
                name=yaml_data["name"],
                description=yaml_data["description"],
                compatibility=yaml_data["compatibility"],
                content=markdown_content,
            )
        except Exception as e:
            raise ValueError(f"Invalid skill metadata in {file_path}: {e}")
        
        return metadata
```

### SkillRegistry Implementation

```python
class SkillRegistry:
    """Registry for loaded skills."""
    
    def __init__(self):
        self.skills: Dict[str, SkillMetadata] = {}
    
    def register(self, metadata: SkillMetadata):
        """Register a skill."""
        if metadata.name in self.skills:
            raise ValueError(f"Skill '{metadata.name}' already registered")
        self.skills[metadata.name] = metadata
    
    def get_by_name(self, name: str) -> Optional[SkillMetadata]:
        """Get skill by name."""
        return self.skills.get(name)
    
    def get_all(self) -> List[SkillMetadata]:
        """Get all registered skills."""
        return list(self.skills.values())
    
    def serialize_to_prompt(self) -> str:
        """Serialize skills to LLM-readable format (for system prompt injection)."""
        return serialize_skills_to_prompt(self)
```

---

## Conversation History Interface

### Overview

The **Conversation History** system provides a **linear, append-only message store** for multi-turn conversations. It makes **no assumptions about persistence**—storage is handled via callbacks at the Agent level.

Key principles:
- **Simple structure**: Messages are minimal tuples `{role, content}`
- **No persistence assumptions**: Agent-level callbacks handle storage
- **Skill tracking**: Skills invoked during assistant messages are embedded as metadata
- **Manual message building**: Agent prepends history to message list once per `run()`
- **Pluggable formatting**: Optional formatter function for serialization
- **No truncation**: Caller decides how much history to include

### ConversationHistory Abstract Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime


class ConversationMessage(BaseModel):
    """Message in conversation history with session tracking."""
    session_id: str  # UUID string to group messages by session
    role: str  # "user" or "assistant"
    content: str
    skills_invoked: Optional[List[Dict[str, Any]]] = None
    timestamp: Optional[datetime] = None


class ConversationHistory(ABC):
    """
    Abstract base class for conversation history storage.
    
    No assumptions about persistence backend—subclasses implement storage logic.
    This is a dumb container; persistence callbacks happen at Agent level.
    """
    
    @abstractmethod
    async def add_message(
        self,
        role: str,  # "user" or "assistant"
        content: str,
        session_id: str,  # UUID string to identify this conversation session
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: Message author ("user" or "assistant")
            content: Message text
            session_id: Session identifier (UUID string) to group related messages
            skills_invoked: Optional list of skill invocations executed
                [
                    {
                        "name": "code-review"
                    }
                ]
        
        Raises:
            ConversationHistoryError: On append failure
        """
        pass
    
    @abstractmethod
    async def get_all(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all messages in order.
        
        Args:
            session_id: Optional session ID to filter messages by session.
                       If None, retrieve messages from all sessions (or current session).
        
        Returns:
            List of message dicts:
            [
                {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "role": "user",
                    "content": "Hello"
                },
                {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "role": "assistant",
                    "content": "Hi there!",
                    "skills_invoked": [...]
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    async def serialize_for_prompt(
        self,
        session_id: Optional[str] = None,
        formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """
        Serialize conversation history to plain text for LLM context.
        
        Args:
            session_id: Optional session ID to filter messages.
                       If None, serialize all messages or current session.
            formatter: Optional custom formatting function.
                      Signature: fn(messages: List[Dict]) -> str
                      If None, uses default conversational format.
        
        Returns:
            Plain text representation of conversation history
        """
        pass
    
    @abstractmethod
    async def clear(self, session_id: Optional[str] = None) -> None:
        """
        Clear conversation history.
        
        Args:
            session_id: Optional session ID to clear only that session.
                       If None, clear all history.
        
        Raises:
            ConversationHistoryError: On clear failure
        """
        pass
    
    @abstractmethod
    async def summarize(
        self,
        model_provider: "BaseProvider",
        system_prompt: str,
        session_id: Optional[str] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """
        Summarize conversation history using an LLM provider.
        
        Retrieves all messages from history, formats them, and sends to LLM
        for summarization. Useful for:
        - Creating conversation summaries for long chats
        - Generating meeting notes
        - Condensing history before archival
        - Creating executive summaries
        
        Args:
            model_provider: BaseProvider instance (e.g., OpenAIProvider)
                           Used to generate the summary via LLM
            system_prompt: System prompt guiding the summarization
                          Should instruct LLM on summarization style/length
                          E.g., "Create a concise bullet-point summary of this conversation"
            session_id: Optional session ID to summarize only that session.
                       If None, summarizes all messages or current session context.
            history_formatter: Optional custom formatter for the messages.
                              Signature: fn(messages: List[Dict]) -> str
                              If None, uses default conversational formatter.
                              Controls how history is presented to LLM before summarization.
        
        Returns:
            str: LLM-generated summary of the conversation
        
        Raises:
            ConversationHistoryError: On retrieval or summarization failure
        """
        pass


class ConversationHistoryError(Exception):
    """Base exception for conversation history errors."""
    pass
```

### InMemoryHistory Reference Implementation

```python
class InMemoryHistory(ConversationHistory):
    """
    Simple in-memory conversation history (reference implementation).
    
    Stores messages in a list with session tracking. No persistence to disk/DB.
    Good for testing and single-run agents.
    
    For multi-turn persistence, extend this class or implement custom
    subclass with callbacks to external storage.
    """
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
    
    async def add_message(
        self,
        role: str,
        content: str,
        session_id: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add message to in-memory list with session tracking."""
        from datetime import datetime
        
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
        }
        
        if skills_invoked:
            message["skills_invoked"] = skills_invoked
        
        self.messages.append(message)
    
    async def get_all(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all messages, optionally filtered by session_id."""
        if session_id:
            return [m for m in self.messages if m["session_id"] == session_id]
        return self.messages.copy()
    
    async def serialize_for_prompt(
        self,
        session_id: Optional[str] = None,
        formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """Serialize using provided formatter or default."""
        messages = await self.get_all(session_id=session_id)
        
        if formatter:
            return formatter(messages)
        else:
            return self._default_formatter(messages)
    
    async def clear(self, session_id: Optional[str] = None) -> None:
        """Clear all messages, optionally filtered by session_id."""
        if session_id:
            self.messages = [m for m in self.messages if m["session_id"] != session_id]
        else:
            self.messages.clear()
    
    async def summarize(
        self,
        model_provider: "BaseProvider",
        system_prompt: str,
        session_id: Optional[str] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """
        Summarize conversation history using LLM provider.
        
        Flow:
        1. Get messages filtered by session_id
        2. Format using provided formatter or default
        3. Create summarization prompt
        4. Call provider.complete() with formatted history
        5. Return LLM-generated summary
        """
        # Get messages for this session
        messages = await self.get_all(session_id=session_id)
        
        if not messages:
            return "No conversation history to summarize."
        
        # Format the history
        formatted_history = await self.serialize_for_prompt(
            session_id=session_id,
            formatter=history_formatter
        )
        
        # Create summarization prompt
        summarization_prompt = f"""Please summarize the following conversation:

---
{formatted_history}
---

Provide a clear, concise summary of the key points and outcomes."""
        
        # Call provider to generate summary
        completion = await model_provider.complete(
            messages=[
                {"role": "user", "content": summarization_prompt}
            ],
            system_prompt=system_prompt,
            model=model_provider.model,
            temperature=0.7,
            max_tokens=1024,
            structured_output=None,
            stream=False,
            tools=None,
        )
        
        # Return the summary content
        return completion.content
    
    @staticmethod
    def _default_formatter(messages: List[Dict[str, Any]]) -> str:
        """
        Default conversational formatter.
        
        Produces clean, readable format:
        
        User: Hello, what's the weather?
        Assistant: I'll check the forecast for you. [Used: weather_forecast]
        User: It's 20°C and sunny today.
        Assistant: There you go!
        """
        if not messages:
            return ""
        
        lines = []
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            
            # Append skill info if present
            # skills_invoked: [{"name": "code-review"}, {"name": "sql-analysis"}]
            if msg.get("skills_invoked"):
                skill_names = ", ".join(s["name"] for s in msg["skills_invoked"])
                content += f" [Used: {skill_names}]"
            
            lines.append(f"{role_label}: {content}")
        
        return "\n".join(lines)
```

### Custom Formatter Function

Users can provide custom formatters for domain-specific serialization:

```python
def custom_formatter(messages: List[Dict[str, Any]]) -> str:
    """
    Example: Formatted with turn numbers and timestamps.
    """
    lines = []
    for i, msg in enumerate(messages, 1):
        role = "👤 USER" if msg["role"] == "user" else "🤖 ASSISTANT"
        lines.append(f"[Turn {i}] {role}")
        lines.append(f"  {msg['content']}")
        
        if msg.get("skills_invoked"):
            for skill in msg["skills_invoked"]:
                lines.append(f"  → Used skill: {skill['name']}")
                if skill.get("result"):
                    lines.append(f"     Result: {skill['result']}")
        
        lines.append("")
    
    return "\n".join(lines)


# Usage
agent = Agent(
    ...,
    conversation_history=InMemoryHistory(),
    history_formatter=custom_formatter,  # Override default
)
```

### Agent-Level Persistence Callbacks

The Agent doesn't assume how history is persisted. Instead, users attach callbacks:

```python
class Agent(BaseAgent):
    
    def __init__(
        self,
        provider: "BaseProvider",
        system_prompt: str,
        conversation_history: Optional[ConversationHistory] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        on_message_added: Optional[Callable[[str, str, Optional[List]], None]] = None,
        on_history_cleared: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """
        Initialize agent with optional conversation history and callbacks.
        
        Args:
            conversation_history: ConversationHistory instance (optional)
            history_formatter: Custom formatter function for serialization (optional)
            on_message_added: Callback fn(role: str, content: str, skills_invoked: List) -> None
                              Called after each message is added to history
            on_history_cleared: Callback fn() -> None
                                Called when history is cleared
        
        Callbacks are user-provided and handle all persistence logic:
        - Write to Redis
        - Write to database
        - Write to file
        - Send to external service
        - etc.
        """
        super().__init__(provider, system_prompt, **kwargs)
        
        self.conversation_history = conversation_history or InMemoryHistory()
        self.history_formatter = history_formatter
        self.on_message_added = on_message_added
        self.on_history_cleared = on_history_cleared
    
    async def _add_to_history(
        self,
        role: str,
        content: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
        tools_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add message to history and trigger callback.
        
        Flow:
        1. Add to ConversationHistory with session_id
        2. Track both skills and tools invoked
        3. Call on_message_added callback
        """
        # Combine skills and tools for history tracking
        invocations = {}
        if skills_invoked:
            invocations["skills_invoked"] = skills_invoked
        if tools_invoked:
            invocations["tools_invoked"] = tools_invoked
        
        await self.conversation_history.add_message(
            role=role,
            content=content,
            session_id=self.session_id,
            skills_invoked=skills_invoked,
        )
        
        # Store additional tools info if needed (can extend ConversationHistory if needed)
        # For now, tools are implicitly in the message content/history
        
        if self.on_message_added:
            await self.on_message_added(role, content, skills_invoked)
    
    async def _clear_history(self) -> None:
        """
        Clear history for this session and trigger callback.
        """
        await self.conversation_history.clear(session_id=self.session_id)
        
        if self.on_history_cleared:
            await self.on_history_cleared()
```

### Agent.run() Integration

```python
class Agent(BaseAgent):
    
    async def run(self, user_input: str) -> "AgentResponse":
        """
        Execute agent with conversation history support.
        
        Flow:
        1. Retrieve all messages from history
        2. Build initial message list: [system] + [history] + [user_input]
        3. Enter step loop
        4. After each LLM call, add response to history
        5. After skill invocations, record in history
        6. Return final response
        """
        self.logger.info(f"Agent.run() starting with input: {user_input[:100]}...")
        
        # Build initial message list with history prepended
        # Retrieve only messages from this session
        history_messages = await self.conversation_history.get_all(session_id=self.session_id)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history_messages,  # Prepend session history once
            {"role": "user", "content": user_input},
        ]
        
        # Add user input to history
        await self._add_to_history("user", user_input)
        
        cumulative_token_usage = TokenUsage()
        steps = []
        step_count = 0
        
        try:
            while step_count < self.max_steps:
                self.logger.debug(f"Step {step_count + 1}/{self.max_steps}")
                
                # Call LLM with retry
                step_result = await self._step_with_retry(messages, step_count)
                step_count += 1
                
                # Track tokens
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
                
                # Record step
                steps.append(step_result.step_record)
                
                # Check rate limiter
                if self.rate_limiter_context:
                    allowed = self.rate_limiter_context.check_and_consume(
                        tokens=cumulative_token_usage.total()
                    )
                    if not allowed:
                        self.logger.warning("Rate limit exceeded")
                        raise RateLimitExceededError(cumulative_token_usage.total())
                
                # Parse response
                if step_result.stop_reason == "end_turn" or not step_result.tool_calls:
                    # Final text response
                    output = step_result.content
                    
                    # Add assistant response to history
                    await self._add_to_history("assistant", output)
                    
                    # Validate against schema if provided
                    if self.output_schema:
                        try:
                            validated = self.output_schema.model_validate_json(output)
                            output = validated
                        except Exception as e:
                            self.logger.error(f"Output validation failed: {e}")
                            raise StructuredOutputValidationError(str(e))
                    
                    return AgentResponse(
                        output=output,
                        steps=steps,
                        total_tokens=cumulative_token_usage.total(),
                        token_usage=cumulative_token_usage,
                        success=True,
                    )
                
                # Process tool calls (skill invocations)
                skills_invoked = []
                for tool_call in step_result.tool_calls:
                    self.logger.debug(f"Invoking skill: {tool_call.name}")
                    
                    try:
                        skill_result = await self.invoke_skill(
                            skill_name=tool_call.name,
                            parameters=tool_call.arguments,
                            context={"user_input": user_input},
                        )
                        
                        skills_invoked.append({
                            "name": tool_call.name,
                            "parameters": tool_call.arguments,
                            "result": skill_result,
                        })
                        
                        # Append assistant response to message list
                        messages.append({
                            "role": "assistant",
                            "content": step_result.content,
                        })
                        
                        # Append skill result as user message
                        messages.append({
                            "role": "user",
                            "content": f"Skill {tool_call.name} result: {skill_result}",
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Skill invocation failed: {e}")
                        messages.append({
                            "role": "user",
                            "content": f"Skill {tool_call.name} error: {str(e)}",
                        })
                
                # Add assistant response to history (with skills)
                await self._add_to_history(
                    "assistant",
                    step_result.content,
                    skills_invoked=skills_invoked if skills_invoked else None,
                )
            
            # Max steps reached
            return AgentResponse(
                output="Max steps reached without final response",
                steps=steps,
                total_tokens=cumulative_token_usage.total(),
                token_usage=cumulative_token_usage,
                success=False,
                error="max_steps_exceeded",
            )
        
        except Exception as e:
            self.logger.error(f"Agent.run() failed: {e}")
            return AgentResponse(
                output=None,
                steps=steps,
                total_tokens=cumulative_token_usage.total(),
                token_usage=cumulative_token_usage,
                success=False,
                error=str(e),
            )
```

### Persistence Example: Redis Backend

Users can extend `ConversationHistory` for custom persistence:

```python
import json
import redis.asyncio as redis


class MongoDBConversationHistory(ConversationHistory):
    """
    MongoDB-backed conversation history.
    
    Persists messages to MongoDB with session tracking and optional TTL indexes.
    Supports querying by session_id for multi-session conversations.
    """
    
    def __init__(
        self,
        mongo_client,  # motor.motor_asyncio.AsyncClient or pymongo.MongoClient
        database_name: str = "agent_db",
        collection_name: str = "conversations",
        ttl_seconds: Optional[int] = 604800,  # 7 days default
    ):
        """
        Initialize MongoDB-backed conversation history.
        
        Args:
            mongo_client: Motor AsyncClient or PyMongo client
            database_name: MongoDB database name
            collection_name: Collection name for storing messages
            ttl_seconds: TTL for documents (optional, uses MongoDB TTL index)
        """
        self.db = mongo_client[database_name]
        self.collection = self.db[collection_name]
        self.ttl_seconds = ttl_seconds
        
        # Ensure indexes exist
        # Index on session_id for fast queries
        # Index on createdAt for TTL if ttl_seconds is set
        self._ensure_indexes()
    
    async def _ensure_indexes(self) -> None:
        """Create necessary indexes on the collection."""
        try:
            # Index for session_id queries
            await self.collection.create_index("session_id")
            
            # TTL index on createdAt if ttl_seconds is set
            if self.ttl_seconds:
                await self.collection.create_index(
                    "createdAt",
                    expireAfterSeconds=self.ttl_seconds
                )
        except Exception as e:
            # Index may already exist; log and continue
            print(f"Warning: Could not create indexes: {e}")
    
    async def add_message(
        self,
        role: str,
        content: str,
        session_id: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Store message in MongoDB with session tracking."""
        from datetime import datetime
        
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "createdAt": datetime.utcnow(),
        }
        
        if skills_invoked:
            message["skills_invoked"] = skills_invoked
        
        # Insert into MongoDB
        await self.collection.insert_one(message)
    
    async def get_all(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve messages from MongoDB, optionally filtered by session_id."""
        query = {}
        if session_id:
            query["session_id"] = session_id
        
        # Get messages sorted by creation time
        cursor = self.collection.find(query).sort("createdAt", 1)
        messages = []
        
        async for doc in cursor:
            # Remove MongoDB _id field from response
            doc.pop("_id", None)
            messages.append(doc)
        
        return messages
    
    async def serialize_for_prompt(
        self,
        session_id: Optional[str] = None,
        formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """Serialize with optional formatter."""
        messages = await self.get_all(session_id=session_id)
        
        if formatter:
            return formatter(messages)
        else:
            return InMemoryHistory._default_formatter(messages)
    
    async def clear(self, session_id: Optional[str] = None) -> None:
        """Clear messages from MongoDB, optionally filtered by session_id."""
        query = {}
        if session_id:
            query["session_id"] = session_id
        
        await self.collection.delete_many(query)
    
    async def summarize(
        self,
        model_provider: "BaseProvider",
        system_prompt: str,
        session_id: Optional[str] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """
        Summarize conversation history from MongoDB using LLM provider.
        
        Flow:
        1. Query MongoDB for messages (filtered by session_id if provided)
        2. Format using provided formatter or default
        3. Create summarization prompt
        4. Call provider.complete() with formatted history
        5. Return LLM-generated summary
        """
        # Get messages for this session
        messages = await self.get_all(session_id=session_id)
        
        if not messages:
            return "No conversation history to summarize."
        
        # Format the history
        formatted_history = await self.serialize_for_prompt(
            session_id=session_id,
            formatter=history_formatter
        )
        
        # Create summarization prompt
        summarization_prompt = f"""Please summarize the following conversation:

---
{formatted_history}
---

Provide a clear, concise summary of the key points and outcomes."""
        
        # Call provider to generate summary
        completion = await model_provider.complete(
            messages=[
                {"role": "user", "content": summarization_prompt}
            ],
            system_prompt=system_prompt,
            model=model_provider.model,
            temperature=0.7,
            max_tokens=1024,
            structured_output=None,
            stream=False,
            tools=None,
        )
        
        # Return the summary content
        return completion.content
```

---

## Conversation History Summarization API

### Overview

The ConversationHistory abstraction provides a `summarize()` method that:
- Retrieves all messages from conversation history
- Formats them for LLM consumption
- Uses a dedicated provider (model) to generate a summary
- Returns a concise summary of the conversation

This enables:
- **Context compression** for long conversations
- **History archival** with summaries
- **Meeting notes** generation
- **Quality assurance** reviews
- **Executive summaries** of conversations
- **Context preparation** for new conversations

### API Signature

```python
async def summarize(
    self,
    model_provider: "BaseProvider",
    system_prompt: str,
    session_id: Optional[str] = None,
    history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> str:
    """
    Summarize conversation history using an LLM provider.
    
    Args:
        model_provider: BaseProvider instance for generating the summary
                       (e.g., OpenAIProvider, AnthropicProvider)
        
        system_prompt: System prompt guiding the summarization strategy.
                      Examples:
                      - "Create a concise bullet-point summary"
                      - "Write an executive summary focusing on decisions"
                      - "Generate meeting notes from this discussion"
                      - "Identify action items and owners"
        
        session_id: Optional session ID to summarize only that session.
                   If None, summarizes all available messages.
                   Useful in multi-session scenarios (e.g., MongoDB).
        
        history_formatter: Optional custom formatter function.
                          Signature: fn(messages: List[Dict]) -> str
                          Controls how messages are presented to the LLM.
                          If None, uses default conversational formatter.
    
    Returns:
        str: LLM-generated summary of the conversation
    
    Raises:
        ConversationHistoryError: If history retrieval or summarization fails
    """
```

### Implementation Details

**Flow:**
1. **Retrieve messages:** Call `get_all(session_id)` to fetch messages
2. **Format history:** Call `serialize_for_prompt(session_id, formatter)` to format
3. **Create prompt:** Build summarization prompt with formatted history
4. **Call provider:** Use provider.complete() to generate summary
5. **Return result:** Extract and return summary content

**Key Design:**
- Reuses existing provider infrastructure (no new provider methods needed)
- Respects history formatters for customizable presentation
- Session-scoped summarization for multi-session storage (MongoDB)
- Error schemas not involved (summary is plain text, not a tool result)
- Provider acts as summarization engine (agnostic to provider type)

### Summarization Strategies (System Prompt Examples)

**Concise Summary:**
```
Create a bullet-point summary of this conversation.
Focus on key topics discussed and conclusions reached.
Keep each point to one sentence maximum.
```

**Executive Summary:**
```
Write a professional executive summary suitable for stakeholders.
Include:
- Main topics and objectives
- Key findings and insights
- Recommendations or next steps
- Estimated effort or impact
```

**Meeting Notes:**
```
Create meeting notes from this conversation.
Format:
- Participants (inferred from context)
- Agenda items discussed
- Decisions made
- Action items (owner: person/date)
```

**Technical Analysis:**
```
Summarize this technical discussion focusing on:
- Problems identified
- Solutions proposed
- Trade-offs discussed
- Final recommendations
- Technical decisions made
```

**Customer Support:**
```
Write a concise summary for the customer support ticket.
Include:
- Issue/question reported
- Solution provided
- Resolution outcome
- Follow-up actions (if any)
```

### Thread-Safe Considerations

**MongoDB (Async):**
```python
# Safe for concurrent summarization requests
summary1 = await history.summarize(..., session_id="user-a-session")
summary2 = await history.summarize(..., session_id="user-b-session")
# Can run in parallel; session_id isolates
```

**InMemory (Single-threaded):**
```python
# Works within single event loop
summary = await history.summarize(...)
```

### Performance Considerations

- **Large histories:** Summaries of 100+ turn conversations may be token-intensive
  - Consider limiting history before summarization: `messages[-50:]`
  - Or use custom formatter to excerpt key exchanges
  
- **Provider costs:** Each summarization call invokes the LLM
  - Summarization tokens count toward rate limits and costs
  - Consider caching summaries for frequently-accessed sessions

- **Timeout handling:** Summarization may take several seconds
  - Call within async context with appropriate timeouts
  - Consider async patterns for batch summarization

### Storage and Retrieval Patterns

**Summarization + Storage:**
```python
# Generate summary and store separately
summary = await history.summarize(
    model_provider=provider,
    system_prompt="Create a concise executive summary.",
    session_id=session_id,
)

# Store in archive with metadata
await archive_collection.insert_one({
    "session_id": session_id,
    "summary": summary,
    "full_history_location": "mongodb://...",
    "summarized_at": datetime.utcnow(),
    "message_count": len(await history.get_all(session_id)),
})
```

**Lazy Summarization Pattern:**
```python
async def get_summary(session_id: str, force_regenerate: bool = False):
    """Get summary, generating if needed."""
    cached = await cache.get(f"summary:{session_id}")
    if cached and not force_regenerate:
        return cached
    
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="...",
        session_id=session_id,
    )
    
    # Cache for future requests
    await cache.set(f"summary:{session_id}", summary, ttl=86400)
    return summary
```

### Usage Examples

#### Basic: In-Memory History

```python
from orchestration_agent import Agent, InMemoryHistory

agent = Agent(
    provider=OpenAIProvider(model="gpt-4o", api_key="..."),
    system_prompt="You are a helpful assistant.",
    conversation_history=InMemoryHistory(),
    max_steps=10,
)

# First turn
response1 = await agent.run("Hello, what's the weather?")
print(response1.output)

# Second turn - history is in memory
response2 = await agent.run("How about tomorrow?")
print(response2.output)
```

#### With MongoDB Persistence

```python
from motor.motor_asyncio import AsyncClient
from orchestration_agent import Agent, MongoDBConversationHistory
from orchestration_agent.provider import OpenAIProvider


async def main():
    # Connect to MongoDB
    mongo_client = AsyncClient("mongodb://localhost:27017")
    
    async def on_message_added(role: str, content: str, skills_invoked):
        """Callback when message is added."""
        print(f"✓ Added {role} message")
        if skills_invoked:
            for skill in skills_invoked:
                print(f"  └─ Used skill: {skill['name']}")
    
    async def on_history_cleared():
        """Callback when history is cleared."""
        print("✓ History cleared")
    
    # Use MongoDB-backed history with session tracking
    agent = Agent(
        provider=OpenAIProvider(model="gpt-4o", api_key="..."),
        system_prompt="You are a helpful assistant.",
        conversation_history=MongoDBConversationHistory(
            mongo_client=mongo_client,
            database_name="agent_db",
            collection_name="conversations",
            ttl_seconds=604800,  # 7 days
        ),
        on_message_added=on_message_added,
        on_history_cleared=on_history_cleared,
        # Session ID is auto-generated, or you can provide your own
        # session_id="user-123-session-abc"
    )
    
    # Conversation is automatically persisted to MongoDB with session tracking
    response = await agent.run("Hello!")
    print(f"Response: {response.output}")


asyncio.run(main())
```

**MongoDB Benefits:**
- Session-based message organization via `session_id`
- TTL indexes for automatic message cleanup
- Easy querying by session for multi-user scenarios
- Rich indexing and querying capabilities
- Scalable for high-volume conversations

#### With Custom Formatter

```python
def structured_formatter(messages: List[Dict[str, Any]]) -> str:
    """Custom formatter with timestamps and turn numbers."""
    lines = ["=== Conversation ==="]
    
    for i, msg in enumerate(messages, 1):
        role_upper = msg["role"].upper()
        lines.append(f"\n[Turn {i}] {role_upper}:")
        lines.append(f"  {msg['content']}")
        
        if msg.get("skills_invoked"):
            lines.append("  Skills used:")
            for skill in msg["skills_invoked"]:
                lines.append(f"    - {skill['name']}")
    
    return "\n".join(lines)


agent = Agent(
    provider=OpenAIProvider(model="gpt-4o", api_key="..."),
    system_prompt="You are a helpful assistant.",
    conversation_history=InMemoryHistory(),
    history_formatter=structured_formatter,
)

response = await agent.run("Hello!")
```

#### Multiple Persistence Backends

```python
async def persist_to_database(role: str, content: str, skills_invoked: List) -> None:
    """Callback to store in PostgreSQL."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversation_messages (role, content, skills_invoked) VALUES ($1, $2, $3)",
            role,
            content,
            json.dumps(skills_invoked or []),
        )


agent = Agent(
    provider=OpenAIProvider(model="gpt-4o", api_key="..."),
    system_prompt="You are a helpful assistant.",
    conversation_history=InMemoryHistory(),
    on_message_added=persist_to_database,  # Also persist to DB
)

response = await agent.run("Hello!")
```

---

## Agent Loop Pseudocode

```
FUNCTION Agent.run(user_input: string) -> AgentResponse
    
    // Retrieve conversation history (if available)
    history_messages = AWAIT conversation_history.get_all()
    
    // Build message list: system + history + user input
    messages = [
        { role: "system", content: system_prompt },
        ...history_messages,  // Prepend entire history once
        { role: "user", content: user_input }
    ]
    
    // Add user input to history
    AWAIT _add_to_history("user", user_input)
    
    cumulative_tokens = TokenUsage()
    steps = []
    step_count = 0
    
    WHILE step_count < max_steps DO
        
        // Single step with retry
        FOR attempt = 0 TO max_retries DO
            TRY
                completion = RETRY_CALL(provider.complete(
                    messages,
                    system_prompt,
                    model,
                    temperature=0.7,
                    max_tokens=2048,
                    structured_output,
                    stream=False
                ))
                
                BREAK from retry loop
            
            CATCH RetryableError as e
                IF attempt < max_retries THEN
                    wait_time = backoff_strategy(attempt)
                    SLEEP(wait_time)
                ELSE
                    RAISE e
                END IF
            END CATCH
        END FOR
        
        // Track tokens
        cumulative_tokens.input_tokens += completion.token_usage.input_tokens
        cumulative_tokens.output_tokens += completion.token_usage.output_tokens
        // ... cache tokens if present
        
        // Check rate limiter
        IF rate_limiter_context THEN
            IF NOT rate_limiter_context.check_and_consume(cumulative_tokens.total()) THEN
                RAISE RateLimitExceededError
            END IF
        END IF
        
        // Log step
        steps.APPEND(StepRecord(...))
        
        // Parse response
        IF completion.stop_reason == "end_turn" OR NO tool_calls THEN
            
            // Validate structured output if required
            IF output_schema THEN
                TRY
                    completion.content = output_schema.validate_json(completion.content)
                CATCH
                    IF within max_retries THEN
                        // Retry will happen on next loop iteration
                        CONTINUE to next attempt
                    END IF
                END CATCH
            END IF
            
            // Add final assistant response to history
            AWAIT _add_to_history("assistant", completion.content)
            
            RETURN AgentResponse(
                output: completion.content,
                steps: steps,
                total_tokens: cumulative_tokens.total(),
                success: True
            )
        
        ELSE IF completion.tool_calls THEN
            
            // Tool call must be invoke_skill
            FOR EACH tool_call IN completion.tool_calls DO
                
                IF tool_call.name != "invoke_skill" THEN
                    LOG WARNING: "Unknown tool: " + tool_call.name
                    CONTINUE
                END IF
                
                skill_name = tool_call.arguments["skill_name"]
                
                IF NOT skill_name THEN
                    messages.APPEND({
                        role: "user",
                        content: "Error: invoke_skill requires skill_name parameter"
                    })
                    CONTINUE
                END IF
                
                // Append assistant's tool call to message list
                messages.APPEND({
                    role: "assistant",
                    content: completion.content
                })
                
                // Get skill markdown content (instructions)
                TRY
                    skill_content = AWAIT Agent.invoke_skill(skill_name)
                    
                    // Append skill content as tool result
                    // LLM will read these instructions and apply them
                    messages.APPEND({
                        role: "tool",
                        content: skill_content,
                        tool_call_id: "invoke_skill"
                    })
                    
                    // Track in history
                    AWAIT _add_to_history(
                        "assistant",
                        completion.content,
                        skills_invoked: [{ name: skill_name }]
                    )
                
                CATCH SkillNotFoundError as e
                    messages.APPEND({
                        role: "user",
                        content: "Error: Skill '" + skill_name + "' not found"
                    })
                
                CATCH Exception as e
                    messages.APPEND({
                        role: "user",
                        content: "Error invoking skill '" + skill_name + "': " + str(e)
                    })
                
                END CATCH
            
            END FOR
        
        END IF
        
        step_count += 1
    
    END WHILE
    
    RETURN AgentResponse(
        output: "Max steps reached",
        steps: steps,
        total_tokens: cumulative_tokens.total(),
        success: False,
        error: "max_steps_exceeded"
    )

END FUNCTION
```

---

## Structured Output Handling

### Output Validation Flow

```python
class Agent(BaseAgent):
    
    async def run(self, user_input: str) -> AgentResponse:
        # ... agent loop ...
        
        # In the final response handling:
        if self.output_schema:
            try:
                # Parse JSON if it's a string
                if isinstance(completion.content, str):
                    validated_output = self.output_schema.model_validate_json(
                        completion.content
                    )
                else:
                    validated_output = self.output_schema.model_validate(
                        completion.content
                    )
                
                final_output = validated_output
            
            except ValidationError as e:
                # Re-raise as retryable error
                # This will trigger retry logic
                raise StructuredOutputValidationError(
                    f"Output validation failed: {e.json()}"
                )
        else:
            final_output = completion.content
        
        return AgentResponse(output=final_output, ...)
```

### Example Usage

```python
from pydantic import BaseModel


class UserDataResponse(BaseModel):
    """Expected output schema."""
    user_id: str
    email: str
    status: str  # active, inactive, etc.


agent = Agent(
    provider=OpenAIProvider(model="gpt-4o", api_key="..."),
    system_prompt="You are a user data assistant. {{skills_catalog}}",
    skills_path="./skills",
    output_schema=UserDataResponse,
)

response = await agent.run("Fetch info for user 12345")
# response.output is of type UserDataResponse, validated
```

---

## Logging & Observability (Pluggable)

### Logger Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, Any


class Logger(ABC):
    """Pluggable logging interface."""
    
    @abstractmethod
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs):
        """Log info message."""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        pass
    
    @abstractmethod
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message."""
        pass


class NoOpLogger(Logger):
    """Default no-op logger (logs nothing)."""
    
    def debug(self, message: str, **kwargs):
        pass
    
    def info(self, message: str, **kwargs):
        pass
    
    def warning(self, message: str, **kwargs):
        pass
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        pass


class StdoutLogger(Logger):
    """Simple stdout logger."""
    
    def debug(self, message: str, **kwargs):
        print(f"[DEBUG] {message}")
    
    def info(self, message: str, **kwargs):
        print(f"[INFO] {message}")
    
    def warning(self, message: str, **kwargs):
        print(f"[WARNING] {message}")
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            print(f"[ERROR] {message}: {exception}")
        else:
            print(f"[ERROR] {message}")
```

### Custom Logger Example

```python
class CustomLogger(Logger):
    """Custom logger integrating with external observability (e.g., Datadog, Splunk)."""
    
    def __init__(self, service_name: str, datadog_client=None):
        self.service_name = service_name
        self.datadog = datadog_client
    
    def info(self, message: str, **kwargs):
        # Send to Datadog
        if self.datadog:
            self.datadog.gauge(
                f"{self.service_name}.info",
                1,
                tags=[f"message:{message}"] + [f"{k}:{v}" for k, v in kwargs.items()]
            )
    
    # ... other methods ...


# Usage
agent = Agent(
    provider=...,
    logger=CustomLogger(service_name="agent", datadog_client=datadog_client),
    ...
)
```

### What's Logged (Recommendations)

- ✓ Agent lifecycle: start, end, steps completed
- ✓ LLM calls: model, tokens, stop reason
- ✓ Skill invocations: name, parameters, result
- ✓ Errors and retries: error type, attempt number, backoff time
- ✗ Rate limiter state (rate limiter is separate component)
- ✗ User data or sensitive content (PII handling is caller's responsibility)

---

## File Structure & Organization

```
project_root/
├── pyproject.toml                 # Python project configuration
├── src/
│   └── orchestration_agent/
│       ├── __init__.py
│       ├── agent.py               # Concrete Agent implementation
│       ├── base_agent.py          # BaseAgent abstract class
│       ├── provider/
│       │   ├── __init__.py
│       │   ├── base.py            # BaseProvider abstract class
│       │   ├── openai_provider.py # OpenAI implementation
│       │   ├── errors.py          # Provider errors
│       │   └── models.py          # CompletionResponse, TokenUsage, etc.
│       ├── skills/
│       │   ├── __init__.py
│       │   ├── loader.py          # SkillLoader
│       │   ├── registry.py        # SkillRegistry
│       │   ├── models.py          # SkillMetadata
│       │   └── executor.py        # Skill execution logic
│       ├── conversation/
│       │   ├── __init__.py
│       │   ├── base.py            # ConversationHistory ABC (with summarize method)
│       │   ├── in_memory.py       # InMemoryHistory reference impl
│       │   ├── mongodb.py         # MongoDBConversationHistory example
│       │   ├── formatters.py      # Default & custom formatters
│       │   └── errors.py          # Conversation-specific errors
│       │   # Note: summarize() method integrated into all implementations
│       ├── logging/
│       │   ├── __init__.py
│       │   ├── base.py            # Logger interface
│       │   └── implementations.py # NoOpLogger, StdoutLogger, etc.
│       ├── models/
│       │   ├── __init__.py
│       │   ├── agent.py           # AgentResponse, StepRecord
│       │   ├── rate_limiter.py    # RateLimiterContext, TokenUsage
│       │   └── errors.py          # Custom exceptions
│       └── utils/
│           ├── __init__.py
│           ├── prompt.py          # inject_skills_into_prompt()
│           └── retry.py           # Retry logic
├── skills/                         # Example skills directory
│   ├── fetch_user_data/
│   │   └── SKILL.md
│   ├── send_email/
│   │   └── SKILL.md
│   └── process_payment/
│       └── SKILL.md
├── tests/
│   ├── test_agent.py
│   ├── test_skill_loader.py
│   ├── test_provider.py
│   └── test_retry.py
└── examples/
    ├── basic_usage.py
    ├── with_structured_output.py
    └── custom_rate_limiter.py
```

---

## pyproject.toml Configuration

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

# Core dependencies (minimal)
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "typing-extensions>=4.0",
]

# Optional provider dependencies
[project.optional-dependencies]
openai = [
    "openai>=1.0",
]
anthropic = [
    "anthropic>=0.7",
]
ollama = [
    "ollama>=0.0.8",
]

# Optional persistence backends
mongodb = [
    "motor>=3.0",  # Async MongoDB driver
]
redis = [
    "redis>=5.0",  # For custom rate limiting implementations
]

# Development dependencies
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "black>=23.0",
    "isort>=5.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

# All optional dependencies
all = [
    "orchestration-agent[openai,anthropic,ollama,mongodb,redis]",
]

[project.urls]
Homepage = "https://github.com/example/orchestration-agent"
Documentation = "https://orchestration-agent.readthedocs.io"
Repository = "https://github.com/example/orchestration-agent.git"

[tool.setuptools]
packages = ["orchestration_agent"]

[tool.setuptools.package-dir]
"" = "src"

# Black formatting
[tool.black]
line-length = 100
target-version = ["py310"]

# isort configuration
[tool.isort]
profile = "black"
line_length = 100

# pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=src/orchestration_agent --cov-report=term-missing"

# mypy type checking
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

# Ruff linting
[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "W", "I"]
```

---

## Usage Patterns & Examples

### Basic Usage with Skills

```python
import asyncio
from orchestration_agent import Agent
from orchestration_agent.provider import OpenAIProvider


async def main():
    # Initialize provider
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="sk-...",
    )
    
    # Initialize agent with skills
    agent = Agent(
        provider=provider,
        system_prompt="""
You are a helpful code review assistant.

You have access to the following specialized tools:

{{ skills_catalog }}

When you need to perform a specialized task, call the invoke_skill function
with the name of the relevant skill. You will then receive detailed instructions
for that skill. Read them carefully and apply them to the user's request.

Use skills whenever the user asks for code review, optimization analysis,
security review, or other specialized expertise.
        """,
        skills_path="./skills",  # Load skills from SKILL.md files
        max_steps=10,
        max_retries=3,
    )
    
    # Run
    response = await agent.run("Review this Python code for security issues")
    
    print(f"Response: {response.output}")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Steps completed: {len(response.steps)}")


asyncio.run(main())
```

**How it works:**
1. Agent loads skills from `skills/` directory (all SKILL.md files)
2. System prompt tells LLM about available skills
3. User asks a question that requires a skill
4. LLM calls `invoke_skill` with the skill name
5. Agent returns the full skill markdown (instructions)
6. LLM reads the instructions and applies them to answer the user

### With External Tools

```python
import asyncio
from pydantic import BaseModel, Field
from orchestration_agent import Agent, BaseTool
from orchestration_agent.provider import OpenAIProvider


class FetchUserInput(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")


class FetchUserTool(BaseTool):
    name = "fetch_user"
    description = "Retrieve user information from database"
    input_schema = FetchUserInput
    
    async def execute(self, input: FetchUserInput) -> dict:
        # Simulate database lookup
        return {
            "user_id": input.user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "status": "active"
        }


class SendEmailInput(BaseModel):
    recipient: str = Field(..., description="Email recipient")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")


class SendEmailTool(BaseTool):
    name = "send_email"
    description = "Send an email"
    input_schema = SendEmailInput
    
    async def execute(self, input: SendEmailInput) -> dict:
        # Simulate email sending
        return {
            "success": True,
            "message_id": "msg_123456",
            "sent_at": "2024-01-15T10:30:00Z"
        }


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    
    agent = Agent(
        provider=provider,
        system_prompt="""
You are a helpful assistant with access to both skills and tools.

{{ skills_catalog }}

You have external tools available:
- fetch_user: Retrieve user information
- send_email: Send an email

Use skills for analysis and guidance. Use tools to execute operations.
        """,
        skills_path="./skills",
        tools=[FetchUserTool(), SendEmailTool()],  # Register external tools
        max_steps=10,
        max_retries=3,
    )
    
    # LLM can call both skills and tools
    response = await agent.run("Fetch user 123 and send them an email about their account")
    
    print(f"Response: {response.output}")
    print(f"Tokens used: {response.total_tokens}")


asyncio.run(main())
```

**How it works:**
1. Define BaseTool subclasses with input_schema (Pydantic models)
2. Register tools at Agent init: `tools=[FetchUserTool(), SendEmailTool()]`
3. Provider converts tools to OpenAI function schema automatically
4. LLM sees both invoke_skill and tools available
5. LLM can call invoke_skill for guidance or invoke_tool to execute
6. Both types of invocations can appear in same step
7. Tools execute immediately and return results
8. Results formatted using tool's error_schema on errors

### With Structured Output

```python
from pydantic import BaseModel


class UserInfo(BaseModel):
    user_id: str
    email: str
    full_name: str
    status: str


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="...")
    
    agent = Agent(
        provider=provider,
        system_prompt="You are a user data assistant. {{skills_catalog}}",
        skills_path="./skills",
        output_schema=UserInfo,  # Validate output
    )
    
    response = await agent.run("Get info for user 12345")
    
    # response.output is guaranteed to be UserInfo instance
    assert isinstance(response.output, UserInfo)
    print(f"User: {response.output.full_name}")


asyncio.run(main())
```

### With Rate Limiting

```python
from orchestration_agent.models import RedisRateLimiter
import redis


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="...")
    
    # Initialize Redis rate limiter
    redis_client = redis.Redis(host="localhost", port=6379)
    rate_limiter = RedisRateLimiter(
        redis_client=redis_client,
        tokens_per_minute=10000,
        tokens_per_hour=1000000,
    )
    
    agent = Agent(
        provider=provider,
        system_prompt="...",
        skills_path="./skills",
        max_retries=3,
        rate_limiter_context=rate_limiter,
    )
    
    response = await agent.run("...")
    print(f"Rate limit status: {rate_limiter.get_current_bucket()}")


asyncio.run(main())
```

### With Custom Logging

```python
from orchestration_agent.logging import Logger


class CustomLogger(Logger):
    def info(self, message: str, **kwargs):
        # Send to external service
        print(f"[INFO] {message} | {kwargs}")


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="...")
    
    agent = Agent(
        provider=provider,
        system_prompt="...",
        skills_path="./skills",
        logger=CustomLogger(),
    )
    
    response = await agent.run("...")


asyncio.run(main())
```

### Streaming Response

```python
async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="...")
    
    agent = Agent(
        provider=provider,
        system_prompt="...",
        streaming=True,  # Enable streaming
    )
    
    # Note: For v1, streaming support is provider-dependent
    # Agent loop may emit partial StepResult objects


asyncio.run(main())
```

### Multi-Turn with Conversation History

```python
from orchestration_agent import Agent, InMemoryHistory
from orchestration_agent.provider import OpenAIProvider


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    
    # Shared history across multiple turns
    history = InMemoryHistory()
    
    agent = Agent(
        provider=provider,
        system_prompt="""
You are a helpful weather assistant. Provide accurate weather information
based on the user's requests.
        """,
        skills_path="./skills",
        conversation_history=history,
        max_steps=10,
        max_retries=3,
    )
    
    # Turn 1: User asks about weather
    print("=== Turn 1 ===")
    response1 = await agent.run("What's the weather in London?")
    print(f"Assistant: {response1.output}")
    print(f"Tokens used: {response1.total_tokens}")
    
    # Turn 2: Follow-up question (history is preserved)
    print("\n=== Turn 2 ===")
    response2 = await agent.run("How about Paris?")
    print(f"Assistant: {response2.output}")
    
    # Turn 3: Referencing previous context
    print("\n=== Turn 3 ===")
    response3 = await agent.run("Compare the two.")
    print(f"Assistant: {response3.output}")
    
    # View full conversation
    print("\n=== Full Conversation ===")
    serialized = await history.serialize_for_prompt()
    print(serialized)


asyncio.run(main())
```

### With MongoDB Multi-Session Support

```python
from motor.motor_asyncio import AsyncClient
from orchestration_agent import Agent, MongoDBConversationHistory
from orchestration_agent.provider import OpenAIProvider


async def main():
    # Connect to MongoDB
    mongo_client = AsyncClient("mongodb://localhost:27017")
    
    async def on_message_added(role: str, content: str, skills_invoked):
        """Callback: Log when messages are added."""
        print(f"[MESSAGE] {role}: {content[:50]}...")
        if skills_invoked:
            for skill in skills_invoked:
                print(f"  └─ Used skill: {skill['name']}")
    
    async def on_history_cleared():
        """Callback: Log when history is cleared."""
        print("[HISTORY] Cleared")
    
    # Shared MongoDB connection for multiple sessions
    mongo_history = MongoDBConversationHistory(
        mongo_client=mongo_client,
        database_name="agent_db",
        collection_name="conversations",
        ttl_seconds=604800,  # 7 days
    )
    
    # Session 1: User interactions
    print("=== Session 1: Code Review ===")
    session1_id = "user-alice-session-001"
    agent1 = Agent(
        provider=OpenAIProvider(model="gpt-4o", api_key="sk-..."),
        system_prompt="You are a code review expert.",
        conversation_history=mongo_history,
        session_id=session1_id,  # Explicit session ID
        on_message_added=on_message_added,
        on_history_cleared=on_history_cleared,
    )
    
    response1 = await agent1.run("Review this Python code")
    print(f"Response: {response1.output}\n")
    
    # Session 2: Different user
    print("=== Session 2: Documentation ===")
    session2_id = "user-bob-session-002"
    agent2 = Agent(
        provider=OpenAIProvider(model="gpt-4o", api_key="sk-..."),
        system_prompt="You are a documentation writer.",
        conversation_history=mongo_history,
        session_id=session2_id,  # Different session ID
        on_message_added=on_message_added,
        on_history_cleared=on_history_cleared,
    )
    
    response2 = await agent2.run("Write API documentation")
    print(f"Response: {response2.output}\n")
    
    # Query history by session
    print("=== Session 1 History ===")
    history1 = await mongo_history.get_all(session_id=session1_id)
    print(f"Messages in session 1: {len(history1)}")
    
    print("\n=== Session 2 History ===")
    history2 = await mongo_history.get_all(session_id=session2_id)
    print(f"Messages in session 2: {len(history2)}")


asyncio.run(main())
```

**Multi-Session Features:**
- Each agent instance gets its own `session_id` (auto-generated UUID or explicit)
- MongoDB stores and retrieves messages filtered by session_id
- Multiple users/sessions can share same MongoDB collection
- TTL indexes automatically clean up old conversations
- Query and audit individual session conversations

### Custom Formatter for History

```python
from typing import List, Dict, Any
from orchestration_agent import Agent, InMemoryHistory
from orchestration_agent.provider import OpenAIProvider


def detailed_formatter(messages: List[Dict[str, Any]]) -> str:
    """
    Custom formatter with rich formatting.
    Useful for debugging or logging.
    """
    lines = ["╔════════════════════════════════════╗"]
    lines.append("║        CONVERSATION HISTORY        ║")
    lines.append("╚════════════════════════════════════╝\n")
    
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        
        if role == "USER":
            lines.append(f"👤 User #{i}:")
        else:
            lines.append(f"🤖 Assistant #{i}:")
        
        # Format content with word wrap
        content = msg["content"]
        lines.append(f"   {content}\n")
        
        # Show skills if used
        if msg.get("skills_invoked"):
            lines.append("   📌 Skills invoked:")
            for skill in msg["skills_invoked"]:
                lines.append(f"      • {skill['name']}")
                if skill.get("result"):
                    lines.append(f"        → Result: {str(skill['result'])[:60]}...")
            lines.append("")
    
    return "\n".join(lines)


async def main():
    agent = Agent(
        provider=OpenAIProvider(model="gpt-4o", api_key="sk-..."),
        system_prompt="You are a helpful assistant.",
        conversation_history=InMemoryHistory(),
        history_formatter=detailed_formatter,  # Custom formatter
    )
    
    await agent.run("What can you help me with?")
    
    # Later, serialize history with custom format
    history = agent.conversation_history
    formatted = await history.serialize_for_prompt()
    print(formatted)


asyncio.run(main())
```

### History Summarization

Summarize long conversations using an LLM provider:

```python
import asyncio
from orchestration_agent import Agent, InMemoryHistory
from orchestration_agent.provider import OpenAIProvider


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    
    # Create agent with history
    history = InMemoryHistory()
    agent = Agent(
        provider=provider,
        system_prompt="You are a helpful technical assistant.",
        conversation_history=history,
    )
    
    # Have a multi-turn conversation
    await agent.run("What's the best practice for database indexing?")
    await agent.run("How does query optimization work?")
    await agent.run("Can you explain execution plans?")
    await agent.run("What about performance monitoring?")
    
    # Now summarize the conversation
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="You are a technical documentation writer.",
        session_id=agent.session_id,  # Summarize this session only
        history_formatter=None,  # Use default formatter
    )
    
    print("=== CONVERSATION SUMMARY ===")
    print(summary)
    print("\n=== FULL HISTORY ===")
    full_history = await history.serialize_for_prompt(session_id=agent.session_id)
    print(full_history)


asyncio.run(main())
```

**Output Example:**
```
=== CONVERSATION SUMMARY ===
The conversation covered database optimization best practices:

1. **Database Indexing**: Discussed best practices for creating effective indexes 
   to speed up query performance, including composite indexes and index selection criteria.

2. **Query Optimization**: Explored how query optimization works, including how the 
   database planner chooses execution strategies based on statistics and costs.

3. **Execution Plans**: Explained how to read and interpret execution plans to 
   understand query performance and identify bottlenecks.

4. **Performance Monitoring**: Covered tools and techniques for monitoring database 
   performance including slow query logs, profiling, and metrics collection.

Key Takeaway: Effective database performance requires understanding indexing 
strategies, query planning, and continuous monitoring.
```

### Advanced Summarization with Custom Formatter

```python
def executive_summary_formatter(messages: List[Dict[str, Any]]) -> str:
    """Format for executive summary - concise, business-focused."""
    lines = ["EXECUTIVE SUMMARY - CONVERSATION TRANSCRIPT\n"]
    
    for msg in messages:
        role = "📊 ANALYST" if msg["role"] == "assistant" else "❓ STAKEHOLDER"
        lines.append(f"\n{role}:")
        lines.append(f"{msg['content']}\n")
        
        # Show which tools were used for business context
        if msg.get("tools_invoked"):
            lines.append("Operations performed:")
            for tool in msg["tools_invoked"]:
                lines.append(f"  • {tool['name']}: {tool.get('parameters', {})}")
    
    return "\n".join(lines)


async def main():
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    history = InMemoryHistory()
    
    # ... agent interactions ...
    
    # Create business-focused summary
    business_summary = await history.summarize(
        model_provider=provider,
        system_prompt="""You are a business analyst. 
Summarize this conversation focusing on:
- Business outcomes
- Data retrieved
- Actions taken
- Decisions made""",
        history_formatter=executive_summary_formatter,
    )
    
    print(business_summary)
```

### MongoDB Session Summarization

```python
from motor.motor_asyncio import AsyncClient
from orchestration_agent import MongoDBConversationHistory

async def main():
    mongo_client = AsyncClient("mongodb://localhost:27017")
    provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
    
    # Shared MongoDB history
    shared_history = MongoDBConversationHistory(
        mongo_client=mongo_client,
        database_name="agent_db",
        collection_name="conversations",
    )
    
    # Summarize a specific user's session
    session_id = "user-alice-session-001"
    
    summary = await shared_history.summarize(
        model_provider=provider,
        system_prompt="Create a concise summary of this support conversation for the customer.",
        session_id=session_id,  # Only this session
    )
    
    # Store summary in MongoDB for later retrieval
    await mongo_client["agent_db"]["conversation_summaries"].insert_one({
        "session_id": session_id,
        "summary": summary,
        "created_at": datetime.utcnow(),
    })
    
    print(f"Session {session_id} summarized and stored.")


asyncio.run(main())
```

### Use Cases for Summarization

**1. Long-Running Conversations:**
- Condense 50+ turn conversations into key points
- Create meeting notes from extended discussions

**2. History Archival:**
```python
# Archive old session with summary
summary = await history.summarize(
    model_provider=provider,
    system_prompt="Create a brief executive summary.",
    session_id=old_session_id,
)

# Store summary + delete full history
await db.store_archived_session(old_session_id, summary)
await history.clear(session_id=old_session_id)
```

**3. Context Preparation:**
```python
# Before starting new conversation, summarize previous sessions
summaries = []
for session_id in user_sessions:
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="Summarize key outcomes from this conversation.",
        session_id=session_id,
    )
    summaries.append(summary)

# Use summaries as context for new conversation
combined_context = "\n".join(summaries)
await new_agent.run(
    user_input,
    system_prompt=f"Previous interactions:\n{combined_context}"
)
```

**4. Quality Assurance:**
```python
# Generate summary for QA review
qa_summary = await history.summarize(
    model_provider=provider,
    system_prompt="Identify any potential issues, errors, or improvements needed.",
    session_id=session_id,
)

# Flag for human review if summary contains warnings
if "error" in qa_summary.lower() or "issue" in qa_summary.lower():
    await flagged_queue.add(session_id, qa_summary)
```

---

## Implementation Checklist

- [ ] **Core Classes**
  - [ ] BaseAgent abstract class
  - [ ] Agent concrete implementation
  - [ ] BaseProvider abstract class
  - [ ] OpenAIProvider implementation

- [ ] **Skills System**
  - [ ] SkillMetadata Pydantic model (APM-compliant)
  - [ ] SkillLoader (YAML parsing)
  - [ ] SkillRegistry
  - [ ] serialize_skills_to_prompt()
  - [ ] inject_skills_into_prompt()

- [ ] **External Tools System**
  - [ ] BaseTool abstract class with error_schema
  - [ ] BaseError default error schema
  - [ ] Tool parameter validation (Pydantic input_schema)
  - [ ] Tool execution with error handling
  - [ ] Error result formatting using tool's error_schema
  - [ ] Tool registration at Agent init
  - [ ] Internal tool dict management (name → tool lookup)
  - [ ] invoke_tool method in Agent
  - [ ] Provider tool schema conversion (BaseProvider.convert_tool_to_openai_schema)
  - [ ] Dual invoke_skill + invoke_tool in same LLM response
  - [ ] Tool result as "tool" role message
  - [ ] Tools tracking in conversation history
  - [ ] ToolNotFoundError, ToolError exceptions
  - [ ] System prompt guidance for tool usage

- [ ] **Models & Data Structures**
  - [ ] TokenUsage
  - [ ] CompletionResponse
  - [ ] ToolCall
  - [ ] AgentResponse
  - [ ] StepRecord
  - [ ] RateLimiterContext interface
  - [ ] Custom exceptions

- [ ] **Retry & Error Handling**
  - [ ] Retry logic with exponential backoff
  - [ ] Retryable error types
  - [ ] _step_with_retry() method

- [ ] **Provider Abstraction**
  - [ ] BaseProvider.complete() interface with tools parameter
  - [ ] OpenAI token tracking
  - [ ] Structured output validation
  - [ ] Error handling (RateLimit, TemporaryError, ValidationError)
  - [ ] Provider acts as gateway for tool schema conversion
  - [ ] OpenAIProvider._convert_tool_to_openai_schema() method
  - [ ] Tool schema validation (Pydantic → OpenAI function format)

- [ ] **Conversation History**
  - [ ] ConversationMessage Pydantic model with session_id
  - [ ] ConversationHistory abstract class with session_id support
  - [ ] InMemoryHistory reference implementation
  - [ ] Default conversational formatter
  - [ ] Custom formatter support
  - [ ] Agent-level persistence callbacks (on_message_added, on_history_cleared)
  - [ ] MongoDBConversationHistory example with TTL indexes
  - [ ] Session-based message filtering
  - [ ] Integration in Agent.run()
  - [ ] Skill metadata tracking in history
  - [ ] Tool invocation tracking in history
  - [ ] Auto-generation of session_id using UUID
  - [ ] **History Summarization**
    - [ ] ConversationHistory.summarize() abstract method
    - [ ] InMemoryHistory.summarize() implementation
    - [ ] MongoDBConversationHistory.summarize() implementation
    - [ ] LLM provider integration for summarization
    - [ ] Custom formatter support in summarization
    - [ ] Session-scoped summarization (optional session_id param)
    - [ ] Error handling for summarization failures
    - [ ] Examples: long conversations, archival, context prep, QA review

- [ ] **Logging & Observability**
  - [ ] Logger abstract interface
  - [ ] NoOpLogger implementation
  - [ ] StdoutLogger implementation
  - [ ] Integration points in agent loop

- [ ] **Testing**
  - [ ] Unit tests for SkillLoader
  - [ ] Unit tests for Agent loop
  - [ ] Mock provider tests
  - [ ] Integration tests with real provider
  - [ ] Error handling tests

- [ ] **Documentation**
  - [ ] README with quick start
  - [ ] API documentation
  - [ ] Architecture guide
  - [ ] Example SKILL.md files

- [ ] **Configuration**
  - [ ] pyproject.toml with optional dependencies
  - [ ] Example .env configuration

---

## Appendix: Example SKILL.md Files

### Example 1: Code Review Skill

```markdown
---
name: code-review
description: Review code for security issues, best practices, and potential bugs
compatibility: "python>=3.10, openai, anthropic"
---

## Code Review Instructions

You are an expert code reviewer specializing in security, performance, and best practices.

### Your Mission

When a user asks you to review code, analyze it thoroughly following this process:

### 1. Security Analysis

- **Injection Attacks**: Look for SQL injection, command injection, XXE, etc.
- **Authentication & Authorization**: Check for proper access controls
- **Secrets Management**: Identify exposed API keys, tokens, or credentials
- **Data Protection**: Review encryption and secure data handling
- **Input Validation**: Ensure all inputs are validated and sanitized

### 2. Code Quality

- **Naming Conventions**: Check variable, function, and class names for clarity
- **Code Duplication**: Identify repeated code blocks that should be refactored
- **Error Handling**: Verify errors are handled gracefully with appropriate logging
- **Performance**: Flag obvious performance issues or inefficiencies
- **Test Coverage**: Comment on test coverage and missing test cases

### 3. Best Practices

- **Language Conventions**: Follow standard patterns for the language
- **Design Patterns**: Identify anti-patterns and suggest improvements
- **Documentation**: Check for missing or unclear documentation
- **Dependencies**: Review third-party library choices for appropriateness

### 4. Report Format

For each issue found:
- **Severity**: Critical | High | Medium | Low
- **Location**: Specific file, function, and line number
- **Description**: What's wrong and why it's a problem
- **Recommendation**: How to fix it with code example if helpful

### Example Report Structure

```
## Code Review Report

### Critical Issues (Must Fix)
- **Lack of Input Validation** [auth.py:45]
  User input is used directly in SQL query...

### High Priority
- **Hardcoded Database Credentials** [config.py:12]
  Remove credentials and use environment variables...

### Medium Priority
- **Missing Error Handling** [handler.py:78]
  Add try-catch to prevent crashes...

### Suggestions
- Consider using prepared statements throughout
```

Be thorough, specific, and constructive. Provide actionable feedback.
```

### Example 2: SQL Optimization Skill

```markdown
---
name: sql-optimization
description: Analyze SQL queries for performance improvements and optimization opportunities
compatibility: "python>=3.10, openai, anthropic, ollama"
---

## SQL Optimization Instructions

You are a database performance expert specializing in query optimization.

### Your Mission

When asked to optimize a SQL query, provide detailed analysis covering:

### 1. Performance Analysis

- **Index Usage**: Identify missing indexes that could improve performance
- **Join Strategy**: Review join order and suggest optimizations
- **Query Complexity**: Assess computational complexity and suggest simplifications
- **N+1 Problems**: Identify queries that load too much data
- **Execution Plan**: Explain potential execution plan inefficiencies

### 2. Correctness Review

- **Logic Verification**: Ensure the query returns correct results
- **Edge Cases**: Identify potential edge case failures
- **Data Type Matching**: Check for type coercion issues
- **NULL Handling**: Verify NULL values are handled correctly
- **Aggregation Logic**: Ensure GROUP BY and HAVING are correct

### 3. Optimization Recommendations

Provide specific, actionable improvements:

**For each optimization:**
- Explain why it improves performance
- Provide the optimized query
- Estimate performance improvement if possible
- Note any trade-offs

### Example Recommendations

- Add composite index on (column1, column2)
- Refactor subquery to JOIN for better performance
- Add materialized view for frequently aggregated data
- Use UNION instead of OR for better optimization
- Partition table by date for faster queries

### 4. Production Considerations

- **Backward Compatibility**: Ensure changes don't break existing code
- **Migration Strategy**: How to safely implement changes
- **Monitoring**: What metrics to watch after optimization
- **Rollback Plan**: How to revert if issues occur

Focus on concrete, measurable improvements backed by database principles.
```

---

## End of Specification

This document is **implementation-ready**. Each section provides sufficient detail
to guide development. All interfaces, models, and algorithms are specified.

**Next Steps:**
1. Create base project structure with pyproject.toml
2. Implement BaseAgent and Agent classes
3. Implement BaseProvider and OpenAIProvider
4. Implement skill loading and registry
5. Write comprehensive tests
6. Document with examples and API reference

