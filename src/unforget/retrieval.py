"""4-channel retrieval with RRF fusion and type boosting.

Channels:
  1. Semantic — pgvector cosine similarity
  2. BM25 — PostgreSQL tsvector full-text search
  3. Entity — text[] array overlap
  4. Temporal — recency-weighted by accessed_at

All 4 channels execute in a single SQL round trip via CTEs.
Fusion: Reciprocal Rank Fusion (RRF) with configurable k and channel weights.
Boosting: insight ×1.5, event ×1.0, raw ×0.5 (configurable).
"""

from __future__ import annotations

import logging
import time as _time
import uuid
from dataclasses import dataclass, field
from typing import Any

import asyncpg

from unforget.types import MemoryResult, MemoryType

logger = logging.getLogger("unforget.retrieval")


@dataclass
class RetrievalConfig:
    """Configuration for 4-channel retrieval."""

    rrf_k: int = 60
    channel_weights: dict[str, float] = field(default_factory=lambda: {
        "semantic": 1.0,
        "bm25": 1.0,
        "entity": 0.7,
        "temporal": 0.3,
    })
    type_boosts: dict[str, float] = field(default_factory=lambda: {
        "insight": 1.5,
        "event": 1.0,
        "raw": 0.5,
    })
    per_channel_limit: int = 15
    ef_search: int = 40


async def four_channel_recall(
    conn: asyncpg.Connection,
    *,
    embedding: list[float],
    query_text: str,
    query_entities: list[str],
    org_id: str,
    agent_id: str,
    include_shared: bool = True,
    memory_type: str | None = None,
    limit: int = 10,
    config: RetrievalConfig,
) -> list[MemoryResult]:
    """Execute 4-channel retrieval with RRF fusion in a single SQL round trip.

    All 4 channels run as CTEs in one query, eliminating 3 network round trips.
    PostgreSQL can share the base table scan across CTEs and parallelize internally.

    Returns up to `limit` results, ranked by fused score with type boosting.
    """
    t0 = _time.perf_counter()

    # Set HNSW search param (validated as int to prevent injection)
    await conn.execute(f"SET hnsw.ef_search = {int(config.ef_search)}")

    # Build shared scope filter and params
    params: list[Any] = []

    if include_shared:
        scope_where = (
            f"org_id = ${_next(params, org_id)} "
            f"AND (agent_id = ${_next(params, agent_id)} OR shared = true)"
        )
    else:
        scope_where = (
            f"org_id = ${_next(params, org_id)} "
            f"AND agent_id = ${_next(params, agent_id)}"
        )

    if memory_type:
        scope_where += f" AND memory_type = ${_next(params, memory_type)}"

    scope_where += " AND valid_to IS NULL"

    # Channel-specific params
    vec_str = "[" + ",".join(map(str, embedding)) + "]"
    vec_idx = _next(params, vec_str)
    query_idx = _next(params, query_text)
    lim_idx = _next(params, config.per_channel_limit)

    _cols = (
        "id, content, memory_type, tags, entities, importance, "
        "access_count, created_at, accessed_at"
    )

    has_entities = bool(query_entities)
    if has_entities:
        ent_idx = _next(params, query_entities)

    ctes = []
    selects = []

    ctes.append(f"""semantic AS (
        SELECT {_cols},
               ROW_NUMBER() OVER (ORDER BY embedding <=> ${vec_idx}::vector) AS rank,
               'semantic'::text AS channel
        FROM memory
        WHERE {scope_where}
        ORDER BY embedding <=> ${vec_idx}::vector
        LIMIT ${lim_idx}
    )""")
    selects.append("SELECT * FROM semantic")

    ctes.append(f"""bm25 AS (
        SELECT {_cols},
               ROW_NUMBER() OVER (
                   ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', ${query_idx})) DESC
               ) AS rank,
               'bm25'::text AS channel
        FROM memory
        WHERE {scope_where}
          AND search_vector @@ plainto_tsquery('english', ${query_idx})
        ORDER BY rank
        LIMIT ${lim_idx}
    )""")
    selects.append("SELECT * FROM bm25")

    if has_entities:
        ctes.append(f"""entity AS (
            SELECT {_cols},
                   ROW_NUMBER() OVER (
                       ORDER BY array_length(
                           ARRAY(SELECT unnest(entities) INTERSECT SELECT unnest(${ent_idx}::text[])),
                           1
                       ) DESC NULLS LAST
                   ) AS rank,
                   'entity'::text AS channel
            FROM memory
            WHERE {scope_where}
              AND entities && ${ent_idx}::text[]
            ORDER BY rank
            LIMIT ${lim_idx}
        )""")
        selects.append("SELECT * FROM entity")

    ctes.append(f"""temporal AS (
        SELECT {_cols},
               ROW_NUMBER() OVER (ORDER BY accessed_at DESC) AS rank,
               'temporal'::text AS channel
        FROM memory
        WHERE {scope_where}
        ORDER BY accessed_at DESC
        LIMIT ${lim_idx}
    )""")
    selects.append("SELECT * FROM temporal")

    sql = "WITH " + ",\n    ".join(ctes) + "\n    " + "\n    UNION ALL ".join(selects)

    rows = await conn.fetch(sql, *params)

    t1 = _time.perf_counter()
    logger.debug(
        "[4-channel] single CTE query: %.1fms (%d rows)",
        (t1 - t0) * 1000,
        len(rows),
    )

    # --- Group rows by id + channel, then RRF fuse ---
    channel_ranks: dict[uuid.UUID, dict[str, int]] = {}
    id_to_row: dict[uuid.UUID, asyncpg.Record] = {}

    for row in rows:
        rid = row["id"]
        channel = row["channel"]
        rank = row["rank"]

        channel_ranks.setdefault(rid, {})[channel] = rank
        if rid not in id_to_row:
            id_to_row[rid] = row

    # --- RRF Fusion ---
    k = config.rrf_k
    weights = config.channel_weights
    boosts = config.type_boosts

    scored: list[tuple[uuid.UUID, float]] = []
    for rid, ranks in channel_ranks.items():
        rrf_score = 0.0
        for channel, rank in ranks.items():
            w = weights.get(channel, 1.0)
            rrf_score += w * (1.0 / (k + rank))

        # Type boosting
        row = id_to_row[rid]
        mt = row["memory_type"]
        boost = boosts.get(mt, 1.0)
        rrf_score *= boost

        scored.append((rid, rrf_score))

    # Sort by fused score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build results
    results = []
    for rid, score in scored[:limit]:
        row = id_to_row[rid]
        results.append(MemoryResult(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            tags=row["tags"] or [],
            entities=row["entities"] or [],
            importance=row["importance"],
            score=score,
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
        ))

    return results


def _next(params: list, value: Any) -> int:
    """Append value to params list, return its $N index."""
    params.append(value)
    return len(params)
