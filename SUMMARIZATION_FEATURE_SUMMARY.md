# Conversation History Summarization Feature

## Overview

The ConversationHistory abstraction now includes a `summarize()` method that enables LLM-powered summarization of conversation history. This provides a unified way to generate summaries, meeting notes, executive summaries, and more across all history backends (InMemory, MongoDB, etc.).

---

## API Addition

### New Abstract Method in ConversationHistory

```python
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
    
    Args:
        model_provider: BaseProvider instance for generating summary
        system_prompt: System prompt guiding summarization strategy
        session_id: Optional session ID to summarize only that session
        history_formatter: Optional custom formatter for message presentation
    
    Returns:
        str: LLM-generated summary of the conversation
    """
```

---

## Key Features

### 1. **Provider-Agnostic**
- Works with any BaseProvider implementation (OpenAI, Anthropic, etc.)
- Provider automatically handles schema conversion
- Reuses existing provider infrastructure

### 2. **Flexible Formatting**
- Respects custom history formatters
- Falls back to default conversational formatter
- Formatter controls how messages are presented to LLM

### 3. **Session-Scoped**
- Optional `session_id` parameter for multi-session scenarios
- Perfect for MongoDB with multiple concurrent sessions
- Each session summarized independently

### 4. **Customizable Strategies**
- System prompt guides summarization approach
- Examples:
  - `"Create a concise bullet-point summary"`
  - `"Write an executive summary for stakeholders"`
  - `"Generate meeting notes with action items"`
  - `"Identify technical decisions and rationale"`

---

## Implementations

### InMemoryHistory.summarize()
```python
async def summarize(
    self,
    model_provider: "BaseProvider",
    system_prompt: str,
    session_id: Optional[str] = None,
    history_formatter: Optional[Callable] = None,
) -> str:
    """
    Flow:
    1. Get messages filtered by session_id
    2. Format using provided formatter or default
    3. Create summarization prompt
    4. Call provider.complete()
    5. Return summary content
    """
```

### MongoDBConversationHistory.summarize()
```python
async def summarize(
    self,
    model_provider: "BaseProvider",
    system_prompt: str,
    session_id: Optional[str] = None,
    history_formatter: Optional[Callable] = None,
) -> str:
    """
    Identical flow to InMemory, but:
    - Queries from MongoDB
    - Session-scoped retrieval (session_id in query)
    - TTL-managed document cleanup compatible
    """
```

---

## Usage Patterns

### Basic Summarization
```python
summary = await history.summarize(
    model_provider=provider,
    system_prompt="Create a concise summary of this conversation.",
    session_id=agent.session_id,
)
```

### Custom Strategy
```python
summary = await history.summarize(
    model_provider=provider,
    system_prompt="""Write an executive summary including:
    - Main topics discussed
    - Key decisions made
    - Action items with owners""",
)
```

### With Custom Formatter
```python
def executive_formatter(messages):
    # Format for business focus
    return formatted_messages

summary = await history.summarize(
    model_provider=provider,
    system_prompt="Create a business-focused summary.",
    history_formatter=executive_formatter,
)
```

### Multi-Session (MongoDB)
```python
# Summarize specific user session
session_id = "user-alice-session-001"
summary = await shared_history.summarize(
    model_provider=provider,
    system_prompt="Create customer-facing summary.",
    session_id=session_id,  # Only this session
)
```

---

## Use Cases

### 1. **Long Conversation Compression**
- Summarize 50+ turn conversations into key points
- Reduces context size for subsequent interactions
- Preserves critical information

### 2. **Meeting Notes Generation**
```python
summary = await history.summarize(
    model_provider=provider,
    system_prompt="""Generate meeting notes with:
    - Attendees (inferred)
    - Agenda items
    - Decisions made
    - Action items: task (owner, date)"""
)
```

### 3. **History Archival**
```python
# Create summary before archiving
summary = await history.summarize(
    model_provider=provider,
    system_prompt="Create a brief executive summary.",
    session_id=old_session_id,
)

# Archive with summary
await archive_db.insert({
    "session_id": old_session_id,
    "summary": summary,
    "archived_at": datetime.utcnow(),
})

# Clear full history
await history.clear(session_id=old_session_id)
```

### 4. **Context Preparation**
```python
# Summarize previous sessions for context
previous_summaries = []
for session_id in user.previous_session_ids:
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="Summarize key outcomes.",
        session_id=session_id,
    )
    previous_summaries.append(summary)

# Use in new conversation
new_agent.system_prompt += f"\n\nPrevious interactions:\n" + "\n".join(previous_summaries)
```

### 5. **Quality Assurance Review**
```python
qa_summary = await history.summarize(
    model_provider=provider,
    system_prompt="""Review this conversation for:
    - Errors or issues encountered
    - Potential improvements
    - Compliance concerns
    - Quality issues"""
)

# Flag for human review if issues found
if "error" in qa_summary.lower():
    await review_queue.add(session_id, qa_summary)
```

### 6. **Batch Summarization**
```python
# Summarize all sessions for a user
summaries = {}
for session_id in user.session_ids:
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="Create a brief summary.",
        session_id=session_id,
    )
    summaries[session_id] = summary

# Store in bulk
await archive_db.insert_many([
    {
        "session_id": sid,
        "summary": summary,
        "summarized_at": datetime.utcnow(),
    }
    for sid, summary in summaries.items()
])
```

---

## Implementation Details

### Method Flow

```
summarize()
├─ Get messages: get_all(session_id)
├─ Format: serialize_for_prompt(session_id, formatter)
├─ Create prompt: "Summarize:\n---\n{formatted_history}\n---"
├─ Call provider: provider.complete(
│    messages=[{role: "user", content: prompt}],
│    system_prompt=system_prompt,
│    model=provider.model,
│    temperature=0.7,
│    max_tokens=1024,
│    tools=None,
│  )
└─ Return: completion.content
```

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Provider param** | Required | Flexibility to use different models for summarization |
| **System prompt param** | Required | Guides summarization strategy |
| **Session_id param** | Optional | Supports multi-session storage (MongoDB) |
| **Formatter param** | Optional | Customizable message presentation |
| **Return type** | String | Simple, LLM-generated summary text |
| **Error handling** | Exception propagation | Caller handles retry logic |
| **Caching** | Not included | Caller implements if needed |
| **Timeout handling** | Not included | Caller uses asyncio.timeout() if needed |

---

## Performance Considerations

### Token Usage
- Summarization invokes LLM, consuming tokens
- Large histories (100+ turns) may be expensive
- Consider limiting history size before summarization

### Provider Costs
- Each summarization call costs tokens/money
- Implement caching for frequently-accessed summaries
- Batch summarization when possible

### Latency
- Summarization may take several seconds
- Use within async context with proper timeout handling
- Not suitable for real-time UI updates (pre-compute summaries)

### Storage Optimization
- Store summaries alongside session IDs for quick retrieval
- Archive old sessions with their summaries (reduce storage)
- Clear full history after summarization if not needed

---

## Example: Complete Workflow

```python
async def archive_user_session(user_id: str, session_id: str):
    """Archive a user session with summary."""
    
    provider = OpenAIProvider(model="gpt-4o", api_key="...")
    history = MongoDBConversationHistory(mongo_client)
    
    # 1. Generate summary
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="""Create a concise summary of this conversation.
        Include: main topics, decisions made, next steps.""",
        session_id=session_id,
    )
    
    # 2. Store in archive
    await mongo_client["agent_db"]["archived_sessions"].insert_one({
        "user_id": user_id,
        "session_id": session_id,
        "summary": summary,
        "message_count": len(await history.get_all(session_id)),
        "archived_at": datetime.utcnow(),
        "ttl": 2592000,  # 30 days
    })
    
    # 3. Clear full history to save storage
    await history.clear(session_id=session_id)
    
    print(f"✓ Archived session {session_id}")
    print(f"  Summary: {summary[:100]}...")
```

---

## Checklist for Implementation

- [ ] Add `summarize()` abstract method to ConversationHistory
- [ ] Implement `summarize()` in InMemoryHistory
- [ ] Implement `summarize()` in MongoDBConversationHistory
- [ ] Add error handling for summarization failures
- [ ] Test with different LLM providers
- [ ] Test with different system prompts
- [ ] Test with custom formatters
- [ ] Test session-scoped summarization (MongoDB)
- [ ] Performance test on large histories
- [ ] Document usage patterns and examples
- [ ] Add to implementation checklist

---

## Summary

The conversation history summarization feature provides:

✅ **Unified API** - Single `summarize()` method across all backends  
✅ **Provider-Agnostic** - Works with any BaseProvider  
✅ **Customizable** - System prompts and formatters guide output  
✅ **Session-Scoped** - Perfect for multi-user/multi-session scenarios  
✅ **Extensible** - Easy to add new history backends  
✅ **Production-Ready** - Suitable for long-running agents and archival  

This enables powerful workflows like:
- Compressing long conversations for context reuse
- Creating executive summaries for stakeholders
- Archiving old sessions with summaries
- Preparing context for new conversations
- Quality assurance and compliance reviews
