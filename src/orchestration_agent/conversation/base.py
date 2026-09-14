"""Conversation history abstraction: a linear, append-only message store.

No persistence assumptions — this is a dumb container. Agent-level callbacks
(`on_message_added`, `on_history_cleared`) handle any additional persistence.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ..provider.base import BaseProvider


class ConversationHistory(ABC):
    """Abstract base class for conversation history storage."""

    @abstractmethod
    async def add_message(
        self,
        role: str,
        content: str,
        session_id: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add a message to conversation history."""

    @abstractmethod
    async def get_all(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all messages in order, optionally filtered by session_id."""

    @abstractmethod
    async def serialize_for_prompt(
        self,
        session_id: Optional[str] = None,
        formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """Serialize conversation history to plain text for LLM context."""

    @abstractmethod
    async def clear(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history, optionally scoped to a session."""

    @abstractmethod
    async def summarize(
        self,
        model_provider: "BaseProvider",
        system_prompt: str,
        session_id: Optional[str] = None,
        history_formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        """Summarize conversation history using an LLM provider."""


async def summarize_messages(
    messages: List[Dict[str, Any]],
    formatted_history: str,
    model_provider: "BaseProvider",
    system_prompt: str,
) -> str:
    """Shared summarization flow used by every ConversationHistory implementation.

    Builds the summarization prompt from already-formatted history and calls
    `model_provider.complete()`. Extracted so InMemoryHistory and
    MongoDBConversationHistory don't duplicate the same four lines.
    """
    if not messages:
        return "No conversation history to summarize."

    summarization_prompt = (
        "Please summarize the following conversation:\n\n"
        f"---\n{formatted_history}\n---\n\n"
        "Provide a clear, concise summary of the key points and outcomes."
    )

    completion = await model_provider.complete(
        messages=[{"role": "user", "content": summarization_prompt}],
        system_prompt=system_prompt,
        model=model_provider.model,
        temperature=0.7,
        max_tokens=1024,
        structured_output=None,
        stream=False,
        tools=None,
    )

    return completion.content
