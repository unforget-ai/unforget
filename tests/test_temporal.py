"""Integration tests for temporal operations — requires PostgreSQL + pgvector."""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from unforget import HistoryAction, MemoryType

from .conftest import requires_db

ORG = "test-org"
AGENT = "test-agent"


@requires_db
class TestSupersede:
    async def test_supersede_basic(self, store):
        """Superseding creates a new memory and soft-deletes the old one."""
        old = await store.write("Deploys to AWS", org_id=ORG, agent_id=AGENT)

        old_updated, new = await store.supersede(
            old.id, "Deploys to Fly.io", org_id=ORG, agent_id=AGENT,
        )

        # Old memory is soft-deleted
        assert old_updated.valid_to is not None
        assert old_updated.superseded_by == new.id

        # New memory is active
        assert new.content == "Deploys to Fly.io"
        assert new.valid_to is None
        assert new.superseded_by is None

    async def test_supersede_inherits_properties(self, store):
        """New memory inherits type, tags, importance from old by default."""
        old = await store.write(
            "Uses Python 3.11",
            org_id=ORG, agent_id=AGENT,
            memory_type=MemoryType.INSIGHT,
            tags=["stack", "python"],
            importance=0.9,
            shared=True,
        )

        _, new = await store.supersede(
            old.id, "Uses Python 3.12", org_id=ORG, agent_id=AGENT,
        )

        assert new.memory_type == MemoryType.INSIGHT
        assert new.tags == ["stack", "python"]
        assert new.importance == 0.9
        assert new.shared is True

    async def test_supersede_overrides(self, store):
        """Can override type, tags, importance on supersede."""
        old = await store.write(
            "raw note", org_id=ORG, agent_id=AGENT,
            memory_type=MemoryType.RAW, tags=["old"], importance=0.3,
        )

        _, new = await store.supersede(
            old.id, "distilled insight",
            org_id=ORG, agent_id=AGENT,
            memory_type=MemoryType.INSIGHT,
            tags=["new"],
            importance=0.8,
        )

        assert new.memory_type == MemoryType.INSIGHT
        assert new.tags == ["new"]
        assert new.importance == 0.8

    async def test_supersede_creates_history(self, store):
        """Both old and new memories get history entries."""
        old = await store.write("v1 fact", org_id=ORG, agent_id=AGENT)
        old_updated, new = await store.supersede(
            old.id, "v2 fact", org_id=ORG, agent_id=AGENT,
        )

        old_hist = await store.history(old.id)
        new_hist = await store.history(new.id)

        # Old: created + superseded
        assert len(old_hist) == 2
        assert old_hist[0].action == HistoryAction.CREATED
        assert old_hist[1].action == HistoryAction.SUPERSEDED

        # New: created
        assert len(new_hist) == 1
        assert new_hist[0].action == HistoryAction.CREATED

    async def test_supersede_old_invisible_in_recall(self, store):
        """Superseded memories don't appear in recall results."""
        old = await store.write("old deployment fact", org_id=ORG, agent_id=AGENT)
        await store.supersede(
            old.id, "new deployment fact", org_id=ORG, agent_id=AGENT,
        )

        results = await store.recall("deployment", org_id=ORG, agent_id=AGENT, rerank=False)
        ids = [r.id for r in results]
        assert old.id not in ids

    async def test_supersede_immutable_raises(self, store):
        item = await store.write("locked", org_id=ORG, agent_id=AGENT, immutable=True)
        with pytest.raises(ValueError, match="immutable"):
            await store.supersede(item.id, "try", org_id=ORG, agent_id=AGENT)

    async def test_supersede_already_superseded_raises(self, store):
        old = await store.write("v1", org_id=ORG, agent_id=AGENT)
        await store.supersede(old.id, "v2", org_id=ORG, agent_id=AGENT)
        with pytest.raises(ValueError, match="already superseded"):
            await store.supersede(old.id, "v3", org_id=ORG, agent_id=AGENT)

    async def test_supersede_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            await store.supersede(uuid.uuid4(), "nope", org_id=ORG, agent_id=AGENT)


@requires_db
class TestTimeline:
    async def test_timeline_current(self, store):
        """Timeline at current time returns all active memories."""
        await store.write("fact one", org_id=ORG, agent_id=AGENT)
        await store.write("fact two", org_id=ORG, agent_id=AGENT)

        now = datetime.now(UTC)
        items = await store.timeline(org_id=ORG, agent_id=AGENT, at=now)
        assert len(items) == 2

    async def test_timeline_excludes_superseded(self, store):
        """Superseded memories don't appear at current time."""
        old = await store.write("old fact", org_id=ORG, agent_id=AGENT)
        await store.supersede(old.id, "new fact", org_id=ORG, agent_id=AGENT)

        now = datetime.now(UTC)
        items = await store.timeline(org_id=ORG, agent_id=AGENT, at=now)
        contents = [i.content for i in items]
        assert "old fact" not in contents
        assert "new fact" in contents

    async def test_timeline_past_shows_old(self, store):
        """Timeline at a past time shows the old memory, not the new one."""
        old = await store.write("deploys to AWS", org_id=ORG, agent_id=AGENT)

        past = datetime.now(UTC)
        await asyncio.sleep(0.05)  # ensure time difference

        await store.supersede(old.id, "deploys to Fly.io", org_id=ORG, agent_id=AGENT)

        items = await store.timeline(org_id=ORG, agent_id=AGENT, at=past)
        contents = [i.content for i in items]
        assert "deploys to AWS" in contents
        assert "deploys to Fly.io" not in contents

    async def test_timeline_type_filter(self, store):
        await store.write("insight", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.INSIGHT)
        await store.write("event", org_id=ORG, agent_id=AGENT, memory_type=MemoryType.EVENT)

        now = datetime.now(UTC)
        items = await store.timeline(
            org_id=ORG, agent_id=AGENT, at=now, memory_type=MemoryType.EVENT,
        )
        assert len(items) == 1
        assert items[0].content == "event"

    async def test_timeline_future_shows_nothing(self, store):
        """Memories created in the future shouldn't appear at past time."""
        very_past = datetime(2020, 1, 1, tzinfo=UTC)

        await store.write("recent fact", org_id=ORG, agent_id=AGENT)

        items = await store.timeline(org_id=ORG, agent_id=AGENT, at=very_past)
        assert len(items) == 0

    async def test_timeline_limit(self, store):
        for i in range(10):
            await store.write(f"fact {i}", org_id=ORG, agent_id=AGENT)

        now = datetime.now(UTC)
        items = await store.timeline(org_id=ORG, agent_id=AGENT, at=now, limit=3)
        assert len(items) == 3


@requires_db
class TestSupersessionChain:
    async def test_chain_single(self, store):
        """A non-superseded memory returns a chain of 1."""
        item = await store.write("standalone", org_id=ORG, agent_id=AGENT)
        chain = await store.supersession_chain(item.id)
        assert len(chain) == 1
        assert chain[0].id == item.id

    async def test_chain_two(self, store):
        """v1 → v2 supersession chain."""
        v1 = await store.write("v1", org_id=ORG, agent_id=AGENT)
        _, v2 = await store.supersede(v1.id, "v2", org_id=ORG, agent_id=AGENT)

        # Query from v1
        chain = await store.supersession_chain(v1.id)
        assert len(chain) == 2
        assert chain[0].content == "v1"
        assert chain[1].content == "v2"

        # Query from v2 — same chain
        chain = await store.supersession_chain(v2.id)
        assert len(chain) == 2
        assert chain[0].content == "v1"
        assert chain[1].content == "v2"

    async def test_chain_three(self, store):
        """v1 → v2 → v3 supersession chain."""
        v1 = await store.write("v1", org_id=ORG, agent_id=AGENT)
        _, v2 = await store.supersede(v1.id, "v2", org_id=ORG, agent_id=AGENT)
        _, v3 = await store.supersede(v2.id, "v3", org_id=ORG, agent_id=AGENT)

        chain = await store.supersession_chain(v1.id)
        assert len(chain) == 3
        assert [c.content for c in chain] == ["v1", "v2", "v3"]

        # Query from middle
        chain = await store.supersession_chain(v2.id)
        assert len(chain) == 3

    async def test_chain_nonexistent(self, store):
        chain = await store.supersession_chain(uuid.uuid4())
        assert chain == []
