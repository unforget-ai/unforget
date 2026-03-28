"""Integration tests for MemoryStore — requires PostgreSQL + pgvector."""

import uuid

import pytest

from unforget import (
    HistoryAction,
    MemoryItem,
    MemoryQuotaExceeded,
    MemoryResult,
    MemoryType,
    WriteItem,
)

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


@requires_db
class TestWrite:
    async def test_write_returns_memory_item(self, store):
        item = await store.write("User prefers Python", org_id=ORG, agent_id=AGENT)
        assert isinstance(item, MemoryItem)
        assert item.content == "User prefers Python"
        assert item.memory_type == MemoryType.INSIGHT
        assert item.org_id == ORG
        assert item.agent_id == AGENT
        assert item.importance == 0.5
        assert item.access_count == 0
        assert item.shared is False
        assert item.immutable is False

    async def test_write_custom_type(self, store):
        item = await store.write(
            "Deployed to Fly on March 15",
            org_id=ORG,
            agent_id=AGENT,
            memory_type=MemoryType.EVENT,
            tags=["deploy", "fly"],
            importance=0.8,
        )
        assert item.memory_type == MemoryType.EVENT
        assert item.tags == ["deploy", "fly"]
        assert item.importance == 0.8

    async def test_write_string_type(self, store):
        item = await store.write("test", org_id=ORG, agent_id=AGENT, memory_type="raw")
        assert item.memory_type == MemoryType.RAW

    async def test_write_duplicate_raises(self, store):
        await store.write("same content", org_id=ORG, agent_id=AGENT)
        with pytest.raises(Exception):  # UniqueViolationError
            await store.write("same content", org_id=ORG, agent_id=AGENT)

    async def test_write_creates_history(self, store):
        item = await store.write("fact with history", org_id=ORG, agent_id=AGENT)
        hist = await store.history(item.id)
        assert len(hist) == 1
        assert hist[0].action == HistoryAction.CREATED
        assert hist[0].new_content == "fact with history"
        assert hist[0].old_content is None

    async def test_write_shared(self, store):
        item = await store.write("shared fact", org_id=ORG, agent_id=AGENT, shared=True)
        assert item.shared is True

    async def test_write_immutable(self, store):
        item = await store.write("locked fact", org_id=ORG, agent_id=AGENT, immutable=True)
        assert item.immutable is True


@requires_db
class TestWriteBatch:
    async def test_batch_write(self, store):
        items = [
            WriteItem(content="fact one"),
            WriteItem(content="fact two", memory_type=MemoryType.EVENT),
            WriteItem(content="fact three", tags=["test"]),
        ]
        results = await store.write_batch(items, org_id=ORG, agent_id=AGENT)
        assert len(results) == 3
        assert results[0].content == "fact one"
        assert results[1].memory_type == MemoryType.EVENT
        assert results[2].tags == ["test"]

    async def test_batch_write_skips_duplicates(self, store):
        await store.write("existing", org_id=ORG, agent_id=AGENT)
        items = [
            WriteItem(content="existing"),  # duplicate — skipped
            WriteItem(content="new one"),
        ]
        results = await store.write_batch(items, org_id=ORG, agent_id=AGENT)
        assert len(results) == 1
        assert results[0].content == "new one"

    async def test_batch_write_empty(self, store):
        results = await store.write_batch([], org_id=ORG, agent_id=AGENT)
        assert results == []


@requires_db
class TestRecall:
    async def test_recall_finds_similar(self, store):
        await store.write("User deploys to Fly.io", org_id=ORG, agent_id=AGENT)
        await store.write("User loves cats", org_id=ORG, agent_id=AGENT)

        results = await store.recall("deployment target", org_id=ORG, agent_id=AGENT)
        assert len(results) > 0
        assert isinstance(results[0], MemoryResult)
        # The deployment memory should rank higher than the cats memory
        assert "Fly" in results[0].content or "deploy" in results[0].content.lower()

    async def test_recall_respects_org_scope(self, store):
        await store.write("org1 fact", org_id="org1", agent_id=AGENT)
        await store.write("org2 fact", org_id="org2", agent_id=AGENT)

        results = await store.recall("fact", org_id="org1", agent_id=AGENT, include_shared=False)
        contents = [r.content for r in results]
        assert "org1 fact" in contents
        assert "org2 fact" not in contents

    async def test_recall_includes_shared(self, store):
        await store.write("private fact", org_id=ORG, agent_id="agent-a")
        await store.write("shared fact", org_id=ORG, agent_id="agent-b", shared=True)

        results = await store.recall(
            "fact", org_id=ORG, agent_id="agent-a", include_shared=True
        )
        contents = [r.content for r in results]
        assert "shared fact" in contents

    async def test_recall_updates_access(self, store):
        item = await store.write("accessed memory", org_id=ORG, agent_id=AGENT)
        assert item.access_count == 0

        await store.recall("accessed", org_id=ORG, agent_id=AGENT)

        updated = await store.get(item.id)
        assert updated is not None
        assert updated.access_count == 1

    async def test_recall_empty(self, store):
        results = await store.recall("anything", org_id=ORG, agent_id=AGENT)
        assert results == []

    async def test_recall_with_type_filter(self, store):
        await store.write("insight fact", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("raw chunk", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW)

        results = await store.recall(
            "fact", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT
        )
        types = {r.memory_type for r in results}
        assert MemoryType.RAW not in types

    async def test_recall_threshold(self, store):
        await store.write("User prefers Python", org_id=ORG, agent_id=AGENT)

        # Very high threshold — likely no results
        results = await store.recall(
            "completely unrelated query about pizza",
            org_id=ORG,
            agent_id=AGENT,
            threshold=0.99,
        )
        assert len(results) == 0


@requires_db
class TestAutoRecall:
    async def test_auto_recall_format(self, store):
        await store.write("User prefers Fly.io", org_id=ORG, agent_id=AGENT)
        await store.write("Team uses Go", org_id=ORG, agent_id=AGENT)

        context = await store.auto_recall("deployment", org_id=ORG, agent_id=AGENT)
        assert context.startswith("[Memory Context]")
        assert "- " in context

    async def test_auto_recall_empty(self, store):
        context = await store.auto_recall("anything", org_id=ORG, agent_id=AGENT)
        assert context == ""


@requires_db
class TestGetAndList:
    async def test_get_existing(self, store):
        item = await store.write("get me", org_id=ORG, agent_id=AGENT)
        fetched = await store.get(item.id)
        assert fetched is not None
        assert fetched.content == "get me"

    async def test_get_nonexistent(self, store):
        result = await store.get(uuid.uuid4())
        assert result is None

    async def test_list_basic(self, store):
        await store.write("one", org_id=ORG, agent_id=AGENT)
        await store.write("two", org_id=ORG, agent_id=AGENT)
        await store.write("three", org_id=ORG, agent_id=AGENT)

        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(items) == 3

    async def test_list_type_filter(self, store):
        await store.write("insight", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("event", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.EVENT)

        items = await store.list(org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        assert len(items) == 1
        assert items[0].content == "insight"

    async def test_list_tag_filter(self, store):
        await store.write("tagged", org_id=ORG, agent_id=AGENT, tags=["infra"])
        await store.write("untagged", org_id=ORG, agent_id=AGENT)

        items = await store.list(org_id=ORG, agent_id=AGENT, tags=["infra"])
        assert len(items) == 1
        assert items[0].content == "tagged"

    async def test_list_pagination(self, store):
        for i in range(5):
            await store.write(f"memory {i}", org_id=ORG, agent_id=AGENT)

        page1 = await store.list(org_id=ORG, agent_id=AGENT, page=1, page_size=2)
        page2 = await store.list(org_id=ORG, agent_id=AGENT, page=2, page_size=2)
        page3 = await store.list(org_id=ORG, agent_id=AGENT, page=3, page_size=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1


@requires_db
class TestUpdate:
    async def test_update_content(self, store):
        item = await store.write("old content", org_id=ORG, agent_id=AGENT)
        updated = await store.update(item.id, content="new content")
        assert updated is not None
        assert updated.content == "new content"

    async def test_update_creates_history(self, store):
        item = await store.write("original", org_id=ORG, agent_id=AGENT)
        await store.update(item.id, content="modified")

        hist = await store.history(item.id)
        assert len(hist) == 2
        assert hist[0].action == HistoryAction.CREATED
        assert hist[1].action == HistoryAction.UPDATED
        assert hist[1].old_content == "original"
        assert hist[1].new_content == "modified"

    async def test_update_tags(self, store):
        item = await store.write("taggable", org_id=ORG, agent_id=AGENT)
        updated = await store.update(item.id, tags=["new-tag"])
        assert updated is not None
        assert updated.tags == ["new-tag"]

    async def test_update_importance(self, store):
        item = await store.write("important", org_id=ORG, agent_id=AGENT, importance=0.3)
        updated = await store.update(item.id, importance=0.9)
        assert updated is not None
        assert updated.importance == 0.9

    async def test_update_immutable_raises(self, store):
        item = await store.write("locked", org_id=ORG, agent_id=AGENT, immutable=True)
        with pytest.raises(ValueError, match="immutable"):
            await store.update(item.id, content="try to change")

    async def test_update_nonexistent(self, store):
        result = await store.update(uuid.uuid4(), content="nope")
        assert result is None


@requires_db
class TestForget:
    async def test_forget_existing(self, store):
        item = await store.write("forget me", org_id=ORG, agent_id=AGENT)
        result = await store.forget(item.id)
        assert result is True
        assert await store.get(item.id) is None

    async def test_forget_nonexistent(self, store):
        result = await store.forget(uuid.uuid4())
        assert result is False

    async def test_forget_all(self, store):
        await store.write("a", org_id=ORG, agent_id=AGENT)
        await store.write("b", org_id=ORG, agent_id=AGENT)
        await store.write("c", org_id="other-org", agent_id=AGENT)

        count = await store.forget_all(org_id=ORG, agent_id=AGENT)
        assert count == 2

        # Other org unaffected
        items = await store.list(org_id="other-org", agent_id=AGENT)
        assert len(items) == 1


@requires_db
class TestStats:
    async def test_stats(self, store):
        await store.write("insight1", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("insight2", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("event1", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.EVENT)

        s = await store.stats(org_id=ORG, agent_id=AGENT)
        assert s.total == 3
        assert s.by_type["insight"] == 2
        assert s.by_type["event"] == 1
        assert s.avg_importance == pytest.approx(0.5)
        assert s.oldest is not None
        assert s.newest is not None

    async def test_stats_empty(self, store):
        s = await store.stats(org_id=ORG, agent_id=AGENT)
        assert s.total == 0
        assert s.by_type == {}


@requires_db
class TestQuota:
    async def test_quota_exceeded(self, store):
        # Create a store with very low quota
        low_quota_store = type(store)(
            database_url=store._database_url,
            max_memories_per_agent=2,
        )
        low_quota_store._pool = store._pool
        low_quota_store._embedder = store._embedder

        await low_quota_store.write("one", org_id=ORG, agent_id="quota-agent")
        await low_quota_store.write("two", org_id=ORG, agent_id="quota-agent")

        with pytest.raises(MemoryQuotaExceeded):
            await low_quota_store.write("three", org_id=ORG, agent_id="quota-agent")
