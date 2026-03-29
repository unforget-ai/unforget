"""Association linking for memory recall boost.

Two passes:
  Pass 1 — Co-occurrence: memories in the same conversation thread are linked.
  Pass 2 — Entity: memories sharing 2+ entities across different threads are linked.

Links are built in the background consolidation scheduler — never on the
hot path. The association pull is applied post-RRF-fusion in the retrieval
path, adding ~2-5ms.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import asyncpg

logger = logging.getLogger("unforget.associations")

# Pass 1: co-occurrence
MAX_TIME_GAP_SECONDS = 600.0
MAX_ASSOCIATIONS_PER_MEMORY = 20

# Pass 2: entity linking
MIN_SHARED_ENTITIES = 2       # need at least 2 shared entities to link
ENTITY_LINK_STRENGTH = 0.6    # base strength for entity links
ENTITY_BONUS_PER_EXTRA = 0.1  # bonus per additional shared entity (beyond 2)
MAX_ENTITY_LINKS_PER_MEMORY = 10
# Skip generic entities that appear in almost every memory
GENERIC_ENTITIES = frozenset({
    "user", "assistant", "one", "time", "day", "thing", "way", "lot",
    "something", "anything", "nothing", "everything",
})


@dataclass
class AssociationLink:
    """A link between two memories with a strength score."""

    memory_id: uuid.UUID
    strength: float
    link_type: str = "co_occurrence"


def calc_strength(time_gap_seconds: float, shared_entity_count: int) -> float:
    """Calculate co-occurrence strength from time proximity and entity overlap."""
    time_score = max(0.0, 1.0 - (abs(time_gap_seconds) / MAX_TIME_GAP_SECONDS))
    entity_score = min(1.0, shared_entity_count * 0.3)
    return min(1.0, time_score * 0.6 + entity_score * 0.4)


def calc_entity_strength(shared_count: int) -> float:
    """Calculate entity link strength from shared entity count."""
    base = ENTITY_LINK_STRENGTH
    bonus = (shared_count - MIN_SHARED_ENTITIES) * ENTITY_BONUS_PER_EXTRA
    return min(1.0, base + bonus)


async def build_associations(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    agent_id: str,
    min_strength: float = 0.1,
) -> int:
    """Build association links — co-occurrence (Pass 1) + entity (Pass 2).

    Returns the total number of links created/updated.
    """
    total = 0
    total += await _build_co_occurrence(pool, org_id=org_id, agent_id=agent_id, min_strength=min_strength)
    total += await _build_entity_links(pool, org_id=org_id, agent_id=agent_id)
    return total


# ---------------------------------------------------------------------------
# Pass 1: Co-occurrence linking (same thread)
# ---------------------------------------------------------------------------

async def _build_co_occurrence(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    agent_id: str,
    min_strength: float = 0.1,
) -> int:
    """Link memories within the same source_thread_id."""
    rows = await pool.fetch(
        """
        SELECT id, source_thread_id, entities, created_at
        FROM memory
        WHERE org_id = $1 AND agent_id = $2
          AND valid_to IS NULL
          AND source_thread_id IS NOT NULL
        ORDER BY source_thread_id, created_at
        """,
        org_id, agent_id,
    )

    if len(rows) < 2:
        return 0

    threads: dict[str, list[asyncpg.Record]] = {}
    for row in rows:
        tid = row["source_thread_id"]
        if tid not in threads:
            threads[tid] = []
        threads[tid].append(row)

    links: list[tuple[uuid.UUID, uuid.UUID, float, str]] = []

    for thread_id, members in threads.items():
        if len(members) < 2:
            continue

        for i, mem_a in enumerate(members):
            links_for_a = 0
            for j in range(i + 1, len(members)):
                if links_for_a >= MAX_ASSOCIATIONS_PER_MEMORY:
                    break

                mem_b = members[j]
                time_gap = abs(
                    (mem_b["created_at"] - mem_a["created_at"]).total_seconds()
                )
                entities_a = set(mem_a["entities"] or [])
                entities_b = set(mem_b["entities"] or [])
                shared = len(entities_a & entities_b)

                strength = calc_strength(time_gap, shared)
                if strength < min_strength:
                    continue

                a_id, b_id = mem_a["id"], mem_b["id"]
                if a_id > b_id:
                    a_id, b_id = b_id, a_id
                links.append((a_id, b_id, strength, "co_occurrence"))
                links_for_a += 1

    count = await _upsert_links(pool, links)
    if count:
        logger.info(
            "Pass 1 (co-occurrence): %d links for %s/%s across %d threads",
            count, org_id, agent_id, len(threads),
        )
    return count


# ---------------------------------------------------------------------------
# Pass 2: Entity linking (across threads)
# ---------------------------------------------------------------------------

async def _build_entity_links(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    agent_id: str,
) -> int:
    """Link memories across threads when they share 2+ non-generic entities.

    Uses a SQL query to find cross-thread pairs efficiently, avoiding
    O(n^2) Python loops. Limits to strongest links per memory.
    """
    # Find cross-thread pairs with shared entities via SQL
    rows = await pool.fetch(
        """
        WITH pairs AS (
            SELECT a.id AS id_a, b.id AS id_b,
                   ARRAY(
                       SELECT e FROM unnest(a.entities) e
                       WHERE e = ANY(b.entities)
                         AND e != ALL($3::text[])
                   ) AS shared_entities
            FROM memory a
            JOIN memory b ON a.org_id = b.org_id AND a.agent_id = b.agent_id
                         AND a.id < b.id
                         AND a.valid_to IS NULL AND b.valid_to IS NULL
            WHERE a.org_id = $1 AND a.agent_id = $2
              AND COALESCE(a.source_thread_id, '') != COALESCE(b.source_thread_id, '')
              AND a.entities && b.entities
        )
        SELECT id_a, id_b, shared_entities
        FROM pairs
        WHERE array_length(shared_entities, 1) >= $4
        """,
        org_id, agent_id, list(GENERIC_ENTITIES), MIN_SHARED_ENTITIES,
    )

    if not rows:
        return 0

    # Score and limit links per memory
    link_counts: dict[uuid.UUID, int] = {}
    links: list[tuple[uuid.UUID, uuid.UUID, float, str]] = []

    # Sort by shared entity count descending — strongest links first
    scored_rows = sorted(rows, key=lambda r: len(r["shared_entities"]), reverse=True)

    for row in scored_rows:
        a_id, b_id = row["id_a"], row["id_b"]
        shared_count = len(row["shared_entities"])

        # Enforce per-memory limit
        count_a = link_counts.get(a_id, 0)
        count_b = link_counts.get(b_id, 0)
        if count_a >= MAX_ENTITY_LINKS_PER_MEMORY or count_b >= MAX_ENTITY_LINKS_PER_MEMORY:
            continue

        strength = calc_entity_strength(shared_count)

        # Canonical ordering already guaranteed by a.id < b.id in SQL
        links.append((a_id, b_id, strength, "entity"))
        link_counts[a_id] = count_a + 1
        link_counts[b_id] = count_b + 1

    count = await _upsert_links(pool, links)
    if count:
        logger.info(
            "Pass 2 (entity): %d links for %s/%s from %d candidate pairs",
            count, org_id, agent_id, len(rows),
        )
    return count


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _upsert_links(
    pool: asyncpg.Pool,
    links: list[tuple[uuid.UUID, uuid.UUID, float, str]],
) -> int:
    """Batch upsert association links. Returns count of links processed."""
    if not links:
        return 0

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO memory_associations (memory_a, memory_b, strength, link_type)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (memory_a, memory_b)
            DO UPDATE SET strength = GREATEST(memory_associations.strength, EXCLUDED.strength),
                          link_type = CASE
                              WHEN EXCLUDED.strength > memory_associations.strength
                              THEN EXCLUDED.link_type
                              ELSE memory_associations.link_type
                          END
            """,
            links,
        )

    return len(links)


async def get_associations(
    conn: asyncpg.Pool | asyncpg.Connection,
    memory_ids: list[uuid.UUID],
) -> dict[uuid.UUID, list[AssociationLink]]:
    """Look up all associations for a set of memory IDs.

    Returns a dict mapping each input memory_id to its linked memories.
    Used at recall time for association pull.
    """
    if not memory_ids:
        return {}

    rows = await conn.fetch(
        """
        SELECT memory_a, memory_b, strength, link_type
        FROM memory_associations
        WHERE memory_a = ANY($1) OR memory_b = ANY($1)
        """,
        memory_ids,
    )

    result: dict[uuid.UUID, list[AssociationLink]] = {}
    id_set = set(memory_ids)

    for row in rows:
        a, b = row["memory_a"], row["memory_b"]
        strength = row["strength"]
        link_type = row["link_type"]

        if a in id_set:
            if a not in result:
                result[a] = []
            result[a].append(AssociationLink(memory_id=b, strength=strength, link_type=link_type))
        if b in id_set:
            if b not in result:
                result[b] = []
            result[b].append(AssociationLink(memory_id=a, strength=strength, link_type=link_type))

    return result
