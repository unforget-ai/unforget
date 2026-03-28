"""Integration tests for production features — requires PostgreSQL + pgvector."""

from datetime import UTC, datetime, timedelta

import pytest

from unforget import MemoryType, RateLimitExceeded

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


@requires_db
class TestRateLimitIntegration:
    async def test_write_rate_limited(self, store):
        """Rate limiter should block excessive writes."""
        # Default is 100/min — we need a store with lower limit
        from unforget import MemoryStore

        limited = MemoryStore(
            database_url=store._database_url,
            max_writes_per_minute=3,
        )
        limited._pool = store._pool
        limited._embedder = store._embedder
        limited._reranker = store._reranker

        await limited.write("one", org_id=ORG, agent_id="rate-agent")
        await limited.write("two", org_id=ORG, agent_id="rate-agent")
        await limited.write("three", org_id=ORG, agent_id="rate-agent")

        with pytest.raises(RateLimitExceeded):
            await limited.write("four", org_id=ORG, agent_id="rate-agent")


@requires_db
class TestBulkDelete:
    async def test_bulk_delete_by_type(self, store):
        await store.write("insight", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("raw one", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW)
        await store.write("raw two", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW)

        count = await store.bulk_delete(
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )
        assert count == 2

        remaining = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(remaining) == 1
        assert remaining[0].memory_type == MemoryType.INSIGHT

    async def test_bulk_delete_by_tags(self, store):
        await store.write("tagged", org_id=ORG, agent_id=AGENT, tags=["temp"])
        await store.write("untagged", org_id=ORG, agent_id=AGENT)

        count = await store.bulk_delete(org_id=ORG, agent_id=AGENT, tags=["temp"])
        assert count == 1

    async def test_bulk_delete_by_age(self, store):
        old = await store.write("old", org_id=ORG, agent_id=AGENT)
        await store.pool.execute(
            "UPDATE memory SET created_at = $2 WHERE id = $1",
            old.id,
            datetime.now(UTC) - timedelta(days=60),
        )
        await store.write("new", org_id=ORG, agent_id=AGENT)

        count = await store.bulk_delete(
            org_id=ORG, agent_id=AGENT,
            older_than=datetime.now(UTC) - timedelta(days=30),
        )
        assert count == 1

    async def test_bulk_delete_by_importance(self, store):
        await store.write("important", org_id=ORG, agent_id=AGENT, importance=0.9)
        await store.write("meh", org_id=ORG, agent_id=AGENT, importance=0.1)

        count = await store.bulk_delete(
            org_id=ORG, agent_id=AGENT, importance_below=0.3,
        )
        assert count == 1

    async def test_bulk_delete_requires_filter(self, store):
        with pytest.raises(ValueError, match="At least one filter"):
            await store.bulk_delete(org_id=ORG, agent_id=AGENT)

    async def test_bulk_delete_combined_filters(self, store):
        await store.write("old raw", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW)
        old = (await store.list(org_id=ORG, agent_id=AGENT))[0]
        await store.pool.execute(
            "UPDATE memory SET created_at = $2 WHERE id = $1",
            old.id,
            datetime.now(UTC) - timedelta(days=60),
        )
        await store.write("new raw", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW)
        await store.write("old insight", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)

        count = await store.bulk_delete(
            org_id=ORG, agent_id=AGENT,
            memory_type=MemoryType.RAW,
            older_than=datetime.now(UTC) - timedelta(days=30),
        )
        assert count == 1  # only old raw, not new raw or old insight


@requires_db
class TestAdvancedList:
    async def test_list_created_after(self, store):
        old = await store.write("old", org_id=ORG, agent_id=AGENT)
        await store.pool.execute(
            "UPDATE memory SET created_at = $2 WHERE id = $1",
            old.id,
            datetime.now(UTC) - timedelta(days=60),
        )
        await store.write("new", org_id=ORG, agent_id=AGENT)

        items = await store.list(
            org_id=ORG, agent_id=AGENT,
            created_after=datetime.now(UTC) - timedelta(days=1),
        )
        assert len(items) == 1
        assert items[0].content == "new"

    async def test_list_created_before(self, store):
        old = await store.write("old", org_id=ORG, agent_id=AGENT)
        await store.pool.execute(
            "UPDATE memory SET created_at = $2 WHERE id = $1",
            old.id,
            datetime.now(UTC) - timedelta(days=60),
        )
        await store.write("new", org_id=ORG, agent_id=AGENT)

        items = await store.list(
            org_id=ORG, agent_id=AGENT,
            created_before=datetime.now(UTC) - timedelta(days=30),
        )
        assert len(items) == 1
        assert items[0].content == "old"

    async def test_list_importance_range(self, store):
        await store.write("low", org_id=ORG, agent_id=AGENT, importance=0.1)
        await store.write("mid", org_id=ORG, agent_id=AGENT, importance=0.5)
        await store.write("high", org_id=ORG, agent_id=AGENT, importance=0.9)

        items = await store.list(
            org_id=ORG, agent_id=AGENT,
            importance_gte=0.4, importance_lte=0.6,
        )
        assert len(items) == 1
        assert items[0].content == "mid"

    async def test_list_fulltext_search(self, store):
        await store.write("Python is a great language for backends", org_id=ORG, agent_id=AGENT)
        await store.write("JavaScript runs in the browser", org_id=ORG, agent_id=AGENT)

        items = await store.list(org_id=ORG, agent_id=AGENT, search="Python backend")
        assert len(items) >= 1
        assert "Python" in items[0].content

    async def test_list_sort_by_importance(self, store):
        await store.write("low", org_id=ORG, agent_id=AGENT, importance=0.1)
        await store.write("high", org_id=ORG, agent_id=AGENT, importance=0.9)
        await store.write("mid", org_id=ORG, agent_id=AGENT, importance=0.5)

        items = await store.list(
            org_id=ORG, agent_id=AGENT,
            sort_by="importance", sort_order="desc",
        )
        importances = [i.importance for i in items]
        assert importances == sorted(importances, reverse=True)

    async def test_list_sort_ascending(self, store):
        await store.write("first", org_id=ORG, agent_id=AGENT)
        await store.write("second", org_id=ORG, agent_id=AGENT)

        items = await store.list(
            org_id=ORG, agent_id=AGENT,
            sort_by="created_at", sort_order="asc",
        )
        assert items[0].content == "first"

    async def test_list_include_shared(self, store):
        await store.write("my fact", org_id=ORG, agent_id=AGENT)
        await store.write("shared fact", org_id=ORG, agent_id="other", shared=True)

        items = await store.list(
            org_id=ORG, agent_id=AGENT, include_shared=True,
        )
        contents = [i.content for i in items]
        assert "shared fact" in contents

    async def test_list_invalid_sort_falls_back(self, store):
        """Invalid sort_by should fall back to created_at without error."""
        await store.write("fact", org_id=ORG, agent_id=AGENT)
        items = await store.list(
            org_id=ORG, agent_id=AGENT, sort_by="invalid_column",
        )
        assert len(items) == 1
