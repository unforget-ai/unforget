"""Tests for co-occurrence association linking and retrieval pull."""

import pytest

from unforget.associations import build_associations, calc_strength, calc_entity_strength, get_associations

from .conftest import requires_db

ORG = "test-assoc"
AGENT = "agent-1"


class TestCalcStrength:
    """Unit tests for strength calculation — no DB needed."""

    def test_same_time_no_entities(self):
        s = calc_strength(0.0, 0)
        assert 0.55 < s < 0.65  # 0.6 * 1.0 + 0.4 * 0.0 = 0.6

    def test_same_time_shared_entities(self):
        s = calc_strength(0.0, 3)
        assert s == pytest.approx(0.96)  # 0.6 * 1.0 + 0.4 * 0.9 = 0.96

    def test_far_apart_no_entities(self):
        s = calc_strength(600.0, 0)
        assert s == 0.0  # time_score=0, entity_score=0

    def test_far_apart_with_entities(self):
        s = calc_strength(600.0, 2)
        assert 0.2 < s < 0.3  # 0.6 * 0.0 + 0.4 * 0.6 = 0.24

    def test_mid_time_one_entity(self):
        s = calc_strength(300.0, 1)
        # time_score = 0.5, entity_score = 0.3
        # 0.6 * 0.5 + 0.4 * 0.3 = 0.42
        assert 0.4 < s < 0.45

    def test_negative_time_uses_abs(self):
        assert calc_strength(-100, 0) == calc_strength(100, 0)


@requires_db
class TestBuildAssociations:
    """Integration tests for link building."""

    @pytest.mark.asyncio
    async def test_no_memories_returns_zero(self, store):
        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count == 0

    @pytest.mark.asyncio
    async def test_single_memory_returns_zero(self, store):
        await store.write("fact one", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count == 0

    @pytest.mark.asyncio
    async def test_same_thread_creates_links(self, store):
        m1 = await store.write("Alice lives in Brooklyn", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        m2 = await store.write("Alice recommended Pizza Place", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        m3 = await store.write("The weather was nice", org_id=ORG, agent_id=AGENT, source_thread_id="t1")

        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count == 3  # m1↔m2, m1↔m3, m2↔m3

    @pytest.mark.asyncio
    async def test_different_threads_not_linked(self, store):
        await store.write("fact A", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        await store.write("fact B", org_id=ORG, agent_id=AGENT, source_thread_id="t2")

        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count == 0

    @pytest.mark.asyncio
    async def test_no_thread_id_not_linked(self, store):
        await store.write("fact A", org_id=ORG, agent_id=AGENT)
        await store.write("fact B", org_id=ORG, agent_id=AGENT)

        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count == 0

    @pytest.mark.asyncio
    async def test_idempotent(self, store):
        await store.write("fact A", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        await store.write("fact B", org_id=ORG, agent_id=AGENT, source_thread_id="t1")

        count1 = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        count2 = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count1 == 1
        assert count2 == 1  # upsert, same count (no new rows but no error)


@requires_db
class TestGetAssociations:
    """Integration tests for association lookup."""

    @pytest.mark.asyncio
    async def test_lookup_returns_links(self, store):
        m1 = await store.write("Alice lives in Brooklyn", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        m2 = await store.write("Alice recommended Pizza Place", org_id=ORG, agent_id=AGENT, source_thread_id="t1")

        await build_associations(store.pool, org_id=ORG, agent_id=AGENT)

        assocs = await get_associations(store.pool, [m1.id])
        assert m1.id in assocs
        assert len(assocs[m1.id]) == 1
        assert assocs[m1.id][0].memory_id == m2.id
        assert assocs[m1.id][0].strength > 0

    @pytest.mark.asyncio
    async def test_lookup_bidirectional(self, store):
        m1 = await store.write("fact A", org_id=ORG, agent_id=AGENT, source_thread_id="t1")
        m2 = await store.write("fact B", org_id=ORG, agent_id=AGENT, source_thread_id="t1")

        await build_associations(store.pool, org_id=ORG, agent_id=AGENT)

        # Looking up from either side finds the other
        assocs_a = await get_associations(store.pool, [m1.id])
        assocs_b = await get_associations(store.pool, [m2.id])
        assert m2.id == assocs_a[m1.id][0].memory_id
        assert m1.id == assocs_b[m2.id][0].memory_id

    @pytest.mark.asyncio
    async def test_empty_ids_returns_empty(self, store):
        assocs = await get_associations(store.pool, [])
        assert assocs == {}


class TestCalcEntityStrength:

    def test_min_shared(self):
        s = calc_entity_strength(2)
        assert s == pytest.approx(0.6)  # base strength

    def test_extra_entities(self):
        s = calc_entity_strength(5)
        assert s == pytest.approx(0.9)  # 0.6 + 3 * 0.1

    def test_capped_at_one(self):
        s = calc_entity_strength(10)
        assert s == 1.0


@requires_db
class TestEntityLinking:

    @pytest.mark.asyncio
    async def test_cross_thread_entity_links(self, store):
        """Memories in different threads with shared entities get linked."""
        m1 = await store.write(
            "Caroline lives in Austin Texas",
            org_id=ORG, agent_id=AGENT, source_thread_id="t1",
        )
        m2 = await store.write(
            "Caroline started a new art project in Austin",
            org_id=ORG, agent_id=AGENT, source_thread_id="t2",
        )

        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count > 0

        assocs = await get_associations(store.pool, [m1.id])
        linked_ids = [a.memory_id for a in assocs.get(m1.id, [])]
        assert m2.id in linked_ids

        # Check link type is "entity"
        entity_links = [a for a in assocs[m1.id] if a.link_type == "entity"]
        assert len(entity_links) > 0

    @pytest.mark.asyncio
    async def test_no_link_without_enough_shared(self, store):
        """Memories with only 1 shared entity (below threshold) are not linked."""
        await store.write(
            "The weather was sunny today",
            org_id=ORG, agent_id=AGENT, source_thread_id="t1",
        )
        await store.write(
            "The weather turned cold overnight",
            org_id=ORG, agent_id=AGENT, source_thread_id="t2",
        )

        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        # Only co-occurrence links (0, since different threads) — no entity links
        # because they'd share at most 1 non-generic entity
        assert count == 0

    @pytest.mark.asyncio
    async def test_generic_entities_filtered(self, store):
        """Generic entities like 'user' don't count toward linking."""
        await store.write(
            "The user mentioned something about food",
            org_id=ORG, agent_id=AGENT, source_thread_id="t1",
        )
        await store.write(
            "The user asked about the time today",
            org_id=ORG, agent_id=AGENT, source_thread_id="t2",
        )

        count = await build_associations(store.pool, org_id=ORG, agent_id=AGENT)
        assert count == 0  # "user" is generic, filtered out


@requires_db
class TestAssociationPull:
    """Integration test: association pull boosts linked memories in recall."""

    @pytest.mark.asyncio
    async def test_pull_boosts_linked_memory(self, store):
        # Create two memories in the same thread
        m1 = await store.write(
            "Sarah lives in Brooklyn and loves art",
            org_id=ORG, agent_id=AGENT, source_thread_id="conv-1",
        )
        m2 = await store.write(
            "Sarah recommended the amazing pizza place on 5th avenue",
            org_id=ORG, agent_id=AGENT, source_thread_id="conv-1",
        )
        # Unrelated memory
        await store.write(
            "The weather forecast shows rain tomorrow afternoon",
            org_id=ORG, agent_id=AGENT, source_thread_id="conv-2",
        )

        # Build associations
        await build_associations(store.pool, org_id=ORG, agent_id=AGENT)

        # Recall with a query that should match m2 directly
        # and pull m1 via association
        results = await store.recall(
            "What pizza place was recommended?",
            org_id=ORG, agent_id=AGENT, limit=10,
        )

        result_ids = [r.id for r in results]
        # m2 should be high (direct match on "pizza" + "recommended")
        assert m2.id in result_ids

        # m1 should also be present (pulled by association with m2)
        # Without association pull, "Sarah lives in Brooklyn" wouldn't match
        # "What pizza place was recommended?" at all
        assert m1.id in result_ids
