"""Simple in-memory conversation history (reference implementation)."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .base import ConversationHistory, summarize_messages
from .formatters import default_formatter

if TYPE_CHECKING:
    from ..provider.base import BaseProvider


class InMemoryHistory(ConversationHistory):
    """Stores messages in a process-local list. No persistence to disk/DB.

    Good for testing and single-run agents; for multi-turn persistence across
    process restarts, use a custom `ConversationHistory` subclass (e.g.
    `MongoDBConversationHistory`).
    """

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    async def add_message(
        self,
        role: str,
        content: str,
        session_id: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        message: Dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
        }
        if skills_invoked:
            message["skills_invoked"] = skills_invoked

        self.messages.append(message)

    async def get_all(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if session_id:
            return [m for m in self.messages if m["session_id"] == session_id]
        return self.messages.copy()

    async def serialize_for_prompt(
        self,
        session_id: Optional[str] = None,
        formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        messages = await self.get_all(session_id=session_id)
        return (formatter or default_formatter)(messages)

    async def clear(self, session_id: Optional[str] = None) -> None:
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
        messages = await self.get_all(session_id=session_id)
        formatted = await self.serialize_for_prompt(
            session_id=session_id, formatter=history_formatter
        )
        return await summarize_messages(messages, formatted, model_provider, system_prompt)
