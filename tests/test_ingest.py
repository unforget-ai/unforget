"""Integration tests for conversation ingest — requires PostgreSQL + pgvector."""

import pytest

from unforget import MemoryType

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"

SAMPLE_CONVERSATION = [
    {"role": "user", "content": "I'm a backend engineer at Acme Corp. We use Python and deploy on Fly.io."},
    {"role": "assistant", "content": "Got it! I'll keep that in mind for our work together."},
    {"role": "user", "content": "Can you check the auth middleware? We switched from JWT to session tokens last week."},
    {"role": "assistant", "content": "Looking at the auth middleware now. I see the JWT code is still here."},
    {"role": "user", "content": "Yeah we need to remove the old JWT code. The new session handling is in auth/sessions.py."},
]


@requires_db
class TestBackgroundMode:
    async def test_ingest_creates_raw_chunks(self, store):
        results = await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="background",
        )
        assert len(results) >= 1
        for r in results:
            assert r.memory_type == MemoryType.RAW
            assert r.importance == pytest.approx(0.3)

    async def test_ingest_extracts_entities(self, store):
        results = await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="background",
        )
        all_entities = []
        for r in results:
            all_entities.extend(r.entities)
        # Should find some tech entities
        entity_set = set(all_entities)
        assert len(entity_set) > 0

    async def test_ingest_links_thread(self, store):
        results = await store.ingest(
            SAMPLE_CONVERSATION,
            org_id=ORG, agent_id=AGENT, mode="background",
            source_thread_id="thread-123",
        )
        for r in results:
            assert r.source_thread_id == "thread-123"

    async def test_ingest_empty(self, store):
        results = await store.ingest([], org_id=ORG, agent_id=AGENT, mode="background")
        assert results == []

    async def test_ingest_dedup(self, store):
        """Ingesting the same conversation twice should not create duplicates."""
        await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="background",
        )
        r2 = await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="background",
        )
        # Second ingest should skip duplicates
        assert len(r2) == 0

    async def test_ingest_recallable(self, store):
        """Ingested memories should be found by recall."""
        await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="background",
        )
        results = await store.recall(
            "auth middleware JWT", org_id=ORG, agent_id=AGENT, rerank=False,
        )
        assert len(results) > 0


@requires_db
class TestImmediateMode:
    async def test_ingest_immediate_creates_insights(self, store):
        async def mock_llm(prompt: str) -> str:
            return (
                "- User is a backend engineer at Acme Corp\n"
                "- Team uses Python and deploys on Fly.io\n"
                "- Recently migrated from JWT to session tokens\n"
                "- New session handling code is in auth/sessions.py"
            )

        results = await store.ingest(
            SAMPLE_CONVERSATION,
            org_id=ORG, agent_id=AGENT, mode="immediate", llm=mock_llm,
        )

        assert len(results) == 4
        for r in results:
            assert r.memory_type == MemoryType.INSIGHT
            assert r.importance == pytest.approx(0.6)

    async def test_ingest_immediate_requires_llm(self, store):
        with pytest.raises(ValueError, match="requires an llm"):
            await store.ingest(
                SAMPLE_CONVERSATION,
                org_id=ORG, agent_id=AGENT, mode="immediate",
            )

    async def test_ingest_immediate_filters_short(self, store):
        """Facts shorter than 10 chars should be filtered out."""
        async def mock_llm(prompt: str) -> str:
            return "- Short\n- This is a proper fact about the user's preferences"

        results = await store.ingest(
            SAMPLE_CONVERSATION,
            org_id=ORG, agent_id=AGENT, mode="immediate", llm=mock_llm,
        )
        assert len(results) == 1


@requires_db
class TestLightweightMode:
    async def test_ingest_lightweight_creates_events(self, store):
        results = await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="lightweight",
        )
        assert len(results) >= 1
        for r in results:
            assert r.memory_type == MemoryType.EVENT
            assert r.importance == pytest.approx(0.5)

    async def test_ingest_lightweight_no_llm(self, store):
        """Lightweight mode should work without LLM."""
        results = await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id=AGENT, mode="lightweight",
        )
        assert len(results) >= 1


@requires_db
class TestIngestModes:
    async def test_invalid_mode_raises(self, store):
        with pytest.raises(ValueError, match="Unknown ingest mode"):
            await store.ingest(
                SAMPLE_CONVERSATION,
                org_id=ORG, agent_id=AGENT, mode="invalid",
            )

    async def test_all_modes_recallable(self, store):
        """Memories from all ingest modes should be recallable."""
        async def mock_llm(prompt: str) -> str:
            return "- User works at Acme Corp as backend engineer"

        # Ingest with each mode using different agents to avoid dedup
        await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id="bg-agent", mode="background",
        )
        await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id="im-agent", mode="immediate", llm=mock_llm,
        )
        await store.ingest(
            SAMPLE_CONVERSATION, org_id=ORG, agent_id="lw-agent", mode="lightweight",
        )

        for agent in ["bg-agent", "im-agent", "lw-agent"]:
            results = await store.recall(
                "Acme backend", org_id=ORG, agent_id=agent, rerank=False,
            )
            assert len(results) > 0, f"No results for {agent}"
