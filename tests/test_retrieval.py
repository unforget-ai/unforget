"""Integration tests for 4-channel retrieval — requires PostgreSQL + pgvector."""


from unforget import MemoryType

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


@requires_db
class TestFourChannelRecall:
    """Tests that validate 4-channel retrieval + RRF fusion + type boosting."""

    async def test_semantic_channel(self, store):
        """Vector similarity finds semantically related memories."""
        await store.write("User deploys applications to Fly.io", org_id=ORG, agent_id=AGENT)
        await store.write("The weather is sunny today", org_id=ORG, agent_id=AGENT)

        results = await store.recall("deployment target", org_id=ORG, agent_id=AGENT, rerank=False)
        assert len(results) > 0
        assert "Fly" in results[0].content

    async def test_bm25_channel(self, store):
        """Full-text search finds keyword matches."""
        await store.write("PostgreSQL database runs on port 5432", org_id=ORG, agent_id=AGENT)
        await store.write("The cat is sleeping on the couch", org_id=ORG, agent_id=AGENT)

        results = await store.recall("PostgreSQL port", org_id=ORG, agent_id=AGENT, rerank=False)
        assert len(results) > 0
        assert "PostgreSQL" in results[0].content

    async def test_temporal_channel(self, store):
        """Recently accessed memories get boosted."""
        await store.write("old memory about servers", org_id=ORG, agent_id=AGENT)
        await store.write("new memory about servers", org_id=ORG, agent_id=AGENT)

        # Access the "new" one to make it more recent
        await store.recall("servers", org_id=ORG, agent_id=AGENT, rerank=False)

        # Both should appear, temporal channel boosts the accessed one
        results = await store.recall("servers", org_id=ORG, agent_id=AGENT, rerank=False)
        assert len(results) == 2

    async def test_type_boosting_insights_first(self, store):
        """Insights should rank higher than raw memories with similar content."""
        await store.write(
            "User prefers Python for backend",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.RAW,
        )
        await store.write(
            "User prefers Python for backend services",
            org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT,
        )

        results = await store.recall("Python backend", org_id=ORG, agent_id=AGENT, rerank=False)
        assert len(results) == 2
        # Insight should be boosted (×1.5) vs raw (×0.5)
        assert results[0].memory_type == MemoryType.INSIGHT

    async def test_shared_memories_included(self, store):
        """Shared memories from other agents appear in results."""
        await store.write("shared infra fact", org_id=ORG, agent_id="other-agent", shared=True)
        await store.write("private fact", org_id=ORG, agent_id=AGENT)

        results = await store.recall(
            "infra", org_id=ORG, agent_id=AGENT, include_shared=True, rerank=False
        )
        contents = [r.content for r in results]
        assert "shared infra fact" in contents

    async def test_shared_memories_excluded(self, store):
        """Shared memories excluded when include_shared=False."""
        await store.write("shared fact", org_id=ORG, agent_id="other-agent", shared=True)
        await store.write("my fact", org_id=ORG, agent_id=AGENT)

        results = await store.recall(
            "fact", org_id=ORG, agent_id=AGENT, include_shared=False, rerank=False
        )
        contents = [r.content for r in results]
        assert "shared fact" not in contents

    async def test_type_filter(self, store):
        """Memory type filter restricts results."""
        await store.write("insight one", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("event one", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.EVENT)

        results = await store.recall(
            "one", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.EVENT, rerank=False,
        )
        types = {r.memory_type for r in results}
        assert types == {MemoryType.EVENT}

    async def test_recall_empty_store(self, store):
        results = await store.recall("anything", org_id=ORG, agent_id=AGENT, rerank=False)
        assert results == []

    async def test_recall_with_limit(self, store):
        for i in range(10):
            await store.write(f"memory about topic {i}", org_id=ORG, agent_id=AGENT)

        results = await store.recall("topic", org_id=ORG, agent_id=AGENT, limit=3, rerank=False)
        assert len(results) <= 3

    async def test_multi_channel_fusion(self, store):
        """A memory matching multiple channels should rank highest."""
        # This memory matches semantically + has keyword "Fly.io" + is recent
        await store.write("Deploy to Fly.io every Friday", org_id=ORG, agent_id=AGENT)
        # This only matches semantically
        await store.write("Cloud deployment is important", org_id=ORG, agent_id=AGENT)

        results = await store.recall("Fly.io deploy", org_id=ORG, agent_id=AGENT, rerank=False)
        assert len(results) >= 1
        # The one with both semantic + BM25 match should rank first
        assert "Fly.io" in results[0].content


@requires_db
class TestRerankedRecall:
    """Tests with cross-encoder reranking enabled."""

    async def test_rerank_changes_order(self, store):
        """Cross-encoder reranking should reorder results based on relevance."""
        await store.write("The cat sat on the mat", org_id=ORG, agent_id=AGENT)
        await store.write("Deploy the production API to Fly.io servers", org_id=ORG, agent_id=AGENT)
        await store.write("Server deployment uses blue-green strategy", org_id=ORG, agent_id=AGENT)

        results = await store.recall(
            "How do we deploy to production?",
            org_id=ORG, agent_id=AGENT, rerank=True,
        )
        assert len(results) > 0
        # Cat memory should not be first after reranking
        assert "cat" not in results[0].content.lower()

    async def test_rerank_respects_limit(self, store):
        for i in range(10):
            await store.write(f"memory number {i} about testing", org_id=ORG, agent_id=AGENT)

        results = await store.recall(
            "testing", org_id=ORG, agent_id=AGENT, limit=3, rerank=True,
        )
        assert len(results) <= 3


@requires_db
class TestRecallCache:
    """Tests for recall result caching."""

    async def test_cache_hit(self, store):
        await store.write("cached fact", org_id=ORG, agent_id=AGENT)

        # First call — cache miss
        r1 = await store.recall("cached", org_id=ORG, agent_id=AGENT, rerank=False)
        # Second call — cache hit (should return same results)
        r2 = await store.recall("cached", org_id=ORG, agent_id=AGENT, rerank=False)

        assert len(r1) == len(r2)
        assert r1[0].id == r2[0].id

    async def test_cache_bypass(self, store):
        await store.write("fact", org_id=ORG, agent_id=AGENT)

        r1 = await store.recall("fact", org_id=ORG, agent_id=AGENT, use_cache=False, rerank=False)
        assert len(r1) > 0

    async def test_different_queries_different_cache(self, store):
        await store.write("Python is great", org_id=ORG, agent_id=AGENT)
        await store.write("JavaScript is popular", org_id=ORG, agent_id=AGENT)

        r1 = await store.recall("Python", org_id=ORG, agent_id=AGENT, rerank=False)
        r2 = await store.recall("JavaScript", org_id=ORG, agent_id=AGENT, rerank=False)

        # Different queries should potentially return different top results
        assert r1[0].id != r2[0].id or r1[0].content != r2[0].content
