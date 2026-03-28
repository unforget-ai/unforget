"""Integration tests for consolidation — requires PostgreSQL + pgvector."""

from datetime import UTC, datetime, timedelta

import pytest

from unforget import ConsolidationReport, MemoryType

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


@requires_db
class TestDeduplication:
    async def test_near_duplicates_merged(self, store):
        """Two very similar memories should be merged during consolidation."""
        await store.write("User deploys to Fly.io", org_id=ORG, agent_id=AGENT)
        await store.write("User deploys applications to Fly.io", org_id=ORG, agent_id=AGENT)

        report = await store.consolidate(
            org_id=ORG, agent_id=AGENT, similarity_threshold=0.85,
        )

        assert report.duplicates_merged >= 1

        # Only one should remain active
        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(items) == 1

    async def test_different_memories_not_merged(self, store):
        """Unrelated memories should not be merged."""
        await store.write("User deploys to Fly.io", org_id=ORG, agent_id=AGENT)
        await store.write("The weather is sunny today", org_id=ORG, agent_id=AGENT)

        report = await store.consolidate(org_id=ORG, agent_id=AGENT)

        assert report.duplicates_merged == 0
        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(items) == 2

    async def test_immutable_not_merged(self, store):
        """Immutable memories should not be touched by dedup."""
        await store.write(
            "Important locked fact", org_id=ORG, agent_id=AGENT, immutable=True
        )
        await store.write(
            "Important locked fact about things", org_id=ORG, agent_id=AGENT,
        )

        await store.consolidate(
            org_id=ORG, agent_id=AGENT, similarity_threshold=0.80,
        )

        # The immutable one should survive
        items = await store.list(org_id=ORG, agent_id=AGENT)
        immutable_items = [i for i in items if i.immutable]
        assert len(immutable_items) == 1

    async def test_merge_with_llm(self, store):
        """When LLM is provided, it merges content."""
        await store.write("User prefers Python", org_id=ORG, agent_id=AGENT)
        await store.write("User likes Python programming", org_id=ORG, agent_id=AGENT)

        async def mock_llm(prompt: str) -> str:
            return "User prefers Python for programming"

        report = await store.consolidate(
            org_id=ORG, agent_id=AGENT,
            similarity_threshold=0.85,
            llm=mock_llm,
        )

        assert report.duplicates_merged >= 1
        items = await store.list(org_id=ORG, agent_id=AGENT)
        assert len(items) == 1
        # LLM's merged content should be used
        assert "Python" in items[0].content

    async def test_merge_creates_history(self, store):
        """Merged (superseded) memories should have history entries."""
        m1 = await store.write("deploy to fly", org_id=ORG, agent_id=AGENT)
        m2 = await store.write("deploy to fly.io", org_id=ORG, agent_id=AGENT)

        await store.consolidate(
            org_id=ORG, agent_id=AGENT, similarity_threshold=0.85,
        )

        # One should be superseded — check history
        hist1 = await store.history(m1.id)
        hist2 = await store.history(m2.id)

        all_actions = [h.action.value for h in hist1] + [h.action.value for h in hist2]
        assert "superseded" in all_actions


@requires_db
class TestImportanceDecay:
    async def test_decay_old_memories(self, store):
        """Memories unaccessed for 30+ days should have importance decayed."""
        m = await store.write(
            "old fact", org_id=ORG, agent_id=AGENT, importance=0.8,
        )

        # Manually set accessed_at to 31 days ago
        await store.pool.execute(
            "UPDATE memory SET accessed_at = $2 WHERE id = $1",
            m.id,
            datetime.now(UTC) - timedelta(days=31),
        )

        report = await store.consolidate(org_id=ORG, agent_id=AGENT)
        assert report.memories_decayed >= 1

        updated = await store.get(m.id)
        assert updated is not None
        assert updated.importance < 0.8  # decayed

    async def test_recent_memories_not_decayed(self, store):
        """Recently accessed memories should keep their importance."""
        m = await store.write(
            "fresh fact", org_id=ORG, agent_id=AGENT, importance=0.8,
        )

        await store.consolidate(org_id=ORG, agent_id=AGENT)

        updated = await store.get(m.id)
        assert updated is not None
        assert updated.importance == pytest.approx(0.8)

    async def test_low_importance_soft_deleted(self, store):
        """Memories that decay below min_importance get soft-deleted."""
        m = await store.write(
            "forgettable", org_id=ORG, agent_id=AGENT, importance=0.05,
        )

        # Set accessed_at to 31 days ago so decay applies
        await store.pool.execute(
            "UPDATE memory SET accessed_at = $2 WHERE id = $1",
            m.id,
            datetime.now(UTC) - timedelta(days=31),
        )

        await store.consolidate(org_id=ORG, agent_id=AGENT)

        updated = await store.get(m.id)
        assert updated is not None
        assert updated.valid_to is not None  # soft-deleted


@requires_db
class TestRawExpiry:
    async def test_old_raw_expired(self, store):
        """Raw memories older than TTL should be soft-deleted."""
        m = await store.write(
            "raw chunk from conversation",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )

        # Set created_at to 31 days ago
        await store.pool.execute(
            "UPDATE memory SET created_at = $2 WHERE id = $1",
            m.id,
            datetime.now(UTC) - timedelta(days=31),
        )

        report = await store.consolidate(org_id=ORG, agent_id=AGENT)
        assert report.memories_expired >= 1

        updated = await store.get(m.id)
        assert updated is not None
        assert updated.valid_to is not None

    async def test_fresh_raw_not_expired(self, store):
        """Recent raw memories should not be expired."""
        m = await store.write(
            "fresh raw chunk",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )

        await store.consolidate(org_id=ORG, agent_id=AGENT)

        updated = await store.get(m.id)
        assert updated is not None
        assert updated.valid_to is None

    async def test_insights_not_expired_by_raw_ttl(self, store):
        """Insight memories should not be affected by raw TTL."""
        m = await store.write(
            "old insight", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT,
        )

        await store.pool.execute(
            "UPDATE memory SET created_at = $2 WHERE id = $1",
            m.id,
            datetime.now(UTC) - timedelta(days=60),
        )

        await store.consolidate(org_id=ORG, agent_id=AGENT)

        updated = await store.get(m.id)
        assert updated is not None
        # Should still be active (not expired by raw TTL, importance still ok)


@requires_db
class TestPromotion:
    async def test_promote_raw_to_insight_with_llm(self, store):
        """LLM should distill raw chunks into insights."""
        await store.write(
            "User said they prefer deploying on Fly.io for production workloads",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )
        await store.write(
            "The team uses PostgreSQL as their primary database for all services",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )
        await store.write(
            "CI/CD pipeline runs on GitHub Actions with Docker containers",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )

        async def mock_llm(prompt: str) -> str:
            return "User primarily uses Python for their projects"

        report = await store.consolidate(
            org_id=ORG, agent_id=AGENT, llm=mock_llm,
        )

        assert report.memories_promoted >= 1

        # Should have new insight(s)
        insights = await store.list(
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT,
        )
        assert len(insights) >= 1

    async def test_no_promotion_without_llm(self, store):
        """Without LLM, no promotion happens."""
        await store.write(
            "raw chunk", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )

        report = await store.consolidate(org_id=ORG, agent_id=AGENT)
        assert report.memories_promoted == 0


@requires_db
class TestConsolidationReport:
    async def test_report_structure(self, store):
        """Report should have all expected fields."""
        report = await store.consolidate(org_id=ORG, agent_id=AGENT)

        assert isinstance(report, ConsolidationReport)
        assert isinstance(report.duplicates_merged, int)
        assert isinstance(report.memories_decayed, int)
        assert isinstance(report.memories_expired, int)
        assert isinstance(report.memories_promoted, int)
        assert isinstance(report.errors, list)

    async def test_consolidated_at_updated(self, store):
        """All active memories should have consolidated_at set after consolidation."""
        m = await store.write("test fact", org_id=ORG, agent_id=AGENT)
        assert m.consolidated_at is None

        await store.consolidate(org_id=ORG, agent_id=AGENT)

        updated = await store.get(m.id)
        assert updated is not None
        assert updated.consolidated_at is not None
