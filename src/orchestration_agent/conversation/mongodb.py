"""MongoDB-backed conversation history (motor async driver).

Note: the spec's pseudocode names the client class `motor.motor_asyncio.AsyncClient`,
which doesn't exist in `motor` — the real class is `AsyncIOMotorClient`. This
implementation accepts any object with the same Motor client interface.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .base import ConversationHistory, summarize_messages
from .formatters import default_formatter

if TYPE_CHECKING:
    from ..provider.base import BaseProvider


class MongoDBConversationHistory(ConversationHistory):
    """Persists messages to MongoDB with session tracking and an optional TTL index."""

    def __init__(
        self,
        mongo_client: Any,
        database_name: str = "agent_db",
        collection_name: str = "conversations",
        ttl_seconds: Optional[int] = 604_800,  # 7 days
    ) -> None:
        self.db = mongo_client[database_name]
        self.collection = self.db[collection_name]
        self.ttl_seconds = ttl_seconds
        self._indexes_ready = False

    async def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        try:
            await self.collection.create_index("session_id")
            if self.ttl_seconds:
                await self.collection.create_index(
                    "createdAt", expireAfterSeconds=self.ttl_seconds
                )
        except Exception as e:  # noqa: BLE001 - index may already exist
            print(f"Warning: could not create indexes: {e}")
        self._indexes_ready = True

    async def add_message(
        self,
        role: str,
        content: str,
        session_id: str,
        skills_invoked: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        await self._ensure_indexes()

        message: Dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "createdAt": datetime.now(timezone.utc),
        }
        if skills_invoked:
            message["skills_invoked"] = skills_invoked

        await self.collection.insert_one(message)

    async def get_all(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        await self._ensure_indexes()

        query: Dict[str, Any] = {"session_id": session_id} if session_id else {}
        cursor = self.collection.find(query).sort("createdAt", 1)

        messages = []
        async for doc in cursor:
            doc.pop("_id", None)
            messages.append(doc)
        return messages

    async def serialize_for_prompt(
        self,
        session_id: Optional[str] = None,
        formatter: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ) -> str:
        messages = await self.get_all(session_id=session_id)
        return (formatter or default_formatter)(messages)

    async def clear(self, session_id: Optional[str] = None) -> None:
        query: Dict[str, Any] = {"session_id": session_id} if session_id else {}
        await self.collection.delete_many(query)

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
