import pytest

from orchestration_agent.conversation.in_memory import InMemoryHistory

from .fakes import FakeProvider, text_response


async def test_in_memory_add_and_get_all():
    history = InMemoryHistory()
    await history.add_message("user", "hello", session_id="s1")
    await history.add_message("assistant", "hi there", session_id="s1")
    await history.add_message("user", "other session", session_id="s2")

    all_s1 = await history.get_all(session_id="s1")
    assert [m["content"] for m in all_s1] == ["hello", "hi there"]

    everything = await history.get_all()
    assert len(everything) == 3


async def test_in_memory_clear_scoped_and_full():
    history = InMemoryHistory()
    await history.add_message("user", "a", session_id="s1")
    await history.add_message("user", "b", session_id="s2")

    await history.clear(session_id="s1")
    assert await history.get_all(session_id="s1") == []
    assert len(await history.get_all(session_id="s2")) == 1

    await history.clear()
    assert await history.get_all() == []


async def test_in_memory_serialize_for_prompt_default_formatter():
    history = InMemoryHistory()
    await history.add_message("user", "hello", session_id="s1")
    await history.add_message(
        "assistant", "used a skill", session_id="s1", skills_invoked=[{"name": "code-review"}]
    )

    text = await history.serialize_for_prompt(session_id="s1")
    assert "User: hello" in text
    assert "Assistant: used a skill [Used: code-review]" in text


async def test_in_memory_serialize_for_prompt_custom_formatter():
    history = InMemoryHistory()
    await history.add_message("user", "hello", session_id="s1")

    text = await history.serialize_for_prompt(
        session_id="s1", formatter=lambda msgs: f"COUNT={len(msgs)}"
    )
    assert text == "COUNT=1"


async def test_in_memory_summarize_uses_provider():
    history = InMemoryHistory()
    await history.add_message("user", "What's 2+2?", session_id="s1")
    await history.add_message("assistant", "4", session_id="s1")

    provider = FakeProvider([text_response("A short math exchange.")])
    summary = await history.summarize(
        model_provider=provider,
        system_prompt="Summarize briefly.",
        session_id="s1",
    )

    assert summary == "A short math exchange."
    sent_messages = provider.calls[0]["messages"]
    assert "2+2" in sent_messages[0]["content"]


async def test_in_memory_summarize_empty_history():
    history = InMemoryHistory()
    provider = FakeProvider([text_response("unused")])

    summary = await history.summarize(
        model_provider=provider, system_prompt="Summarize.", session_id="empty"
    )

    assert summary == "No conversation history to summarize."
    assert provider.calls == []


# --- MongoDB integration (requires docker-compose mongo, skipped otherwise) ---


@pytest.fixture
async def mongo_history(mongo_available):
    if not mongo_available:
        pytest.skip("mongodb not reachable on localhost:27017 (run `docker compose up -d`)")

    motor = pytest.importorskip("motor.motor_asyncio")
    from orchestration_agent.conversation.mongodb import MongoDBConversationHistory

    client = motor.AsyncIOMotorClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
    db_name = "orchestration_agent_test"
    history = MongoDBConversationHistory(
        mongo_client=client, database_name=db_name, collection_name="conversations"
    )
    try:
        yield history
    finally:
        await client.drop_database(db_name)
        client.close()


async def test_mongodb_add_get_clear_roundtrip(mongo_history):
    await mongo_history.add_message("user", "hello mongo", session_id="m1")
    await mongo_history.add_message(
        "assistant", "hi", session_id="m1", skills_invoked=[{"name": "code-review"}]
    )

    messages = await mongo_history.get_all(session_id="m1")
    assert [m["content"] for m in messages] == ["hello mongo", "hi"]
    assert messages[1]["skills_invoked"] == [{"name": "code-review"}]

    await mongo_history.clear(session_id="m1")
    assert await mongo_history.get_all(session_id="m1") == []


async def test_mongodb_summarize_uses_provider(mongo_history):
    await mongo_history.add_message("user", "What's the capital of France?", session_id="m2")
    await mongo_history.add_message("assistant", "Paris.", session_id="m2")

    provider = FakeProvider([text_response("A geography Q&A.")])
    summary = await mongo_history.summarize(
        model_provider=provider, system_prompt="Summarize.", session_id="m2"
    )

    assert summary == "A geography Q&A."
