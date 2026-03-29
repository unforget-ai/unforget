"""Background consolidation worker — dedup, contradictions, promotion, decay.

Runs async, never on the hot path. Optional LLM for merging duplicates.
Without LLM: keeps the longer/newer memory, soft-deletes the other.
With LLM: merges duplicates into a single clean memory.

Inspired by Letta's sleep-time compute pattern.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import asyncpg
import numpy as np

from unforget.associations import build_associations
from unforget.embedder import BaseEmbedder
from unforget.entities import extract_entities

logger = logging.getLogger("unforget.consolidation")


@dataclass
class ConsolidationReport:
    """Summary of what consolidation did."""

    duplicates_merged: int = 0
    memories_decayed: int = 0
    memories_expired: int = 0
    memories_promoted: int = 0
    associations_linked: int = 0
    errors: list[str] = field(default_factory=list)


# Type for optional LLM callable: (prompt: str) -> str
LLMCallable = Callable[[str], Awaitable[str]]


async def consolidate(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    *,
    org_id: str,
    agent_id: str,
    similarity_threshold: float = 0.92,
    decay_7d_factor: float = 0.95,
    decay_30d_factor: float = 0.80,
    min_importance: float = 0.1,
    raw_ttl_days: int = 30,
    llm: LLMCallable | None = None,
) -> ConsolidationReport:
    """Run consolidation for an agent's memories.

    Steps:
      1. Find and merge near-duplicates (cosine > threshold)
      2. Decay importance of unaccessed memories
      3. Expire old raw chunks past TTL
      4. Promote raw → event → insight (with optional LLM)
      5. Mark consolidated_at on all touched memories

    Args:
        pool: Database connection pool.
        embedder: For re-embedding merged content.
        org_id: Organization scope.
        agent_id: Agent scope.
        similarity_threshold: Cosine similarity above which two memories are duplicates.
        decay_7d_factor: Multiply importance by this for memories unaccessed 7+ days.
        decay_30d_factor: Multiply importance by this for memories unaccessed 30+ days.
        min_importance: Soft-delete memories below this importance.
        raw_ttl_days: Hard TTL for raw-type memories.
        llm: Optional async callable for merging. Signature: async (prompt) -> str.

    Returns:
        ConsolidationReport with counts of actions taken.
    """
    report = ConsolidationReport()

    # Step 1: Deduplicate
    await _deduplicate(
        pool, embedder, org_id, agent_id, similarity_threshold, llm, report
    )

    coros = [
        _decay_importance(pool, org_id, agent_id, decay_7d_factor, decay_30d_factor, min_importance, report),
        _expire_raw(pool, org_id, agent_id, raw_ttl_days, report),
    ]
    if llm:
        coros.append(_promote_with_llm(pool, embedder, org_id, agent_id, llm, report))
    await asyncio.gather(*coros)

    # Step 5: Build co-occurrence association links
    try:
        report.associations_linked = await build_associations(
            pool, org_id=org_id, agent_id=agent_id,
        )
    except Exception as e:
        report.errors.append(f"Association linking failed: {e}")
        logger.warning("Association linking failed for %s/%s: %s", org_id, agent_id, e)

    # Step 6: Mark consolidated_at
    await pool.execute(
        """
        UPDATE memory SET consolidated_at = now()
        WHERE org_id = $1 AND agent_id = $2 AND valid_to IS NULL
        """,
        org_id,
        agent_id,
    )

    logger.info(
        "Consolidation complete for %s/%s: %d merged, %d decayed, %d expired, %d promoted, %d linked",
        org_id, agent_id,
        report.duplicates_merged, report.memories_decayed,
        report.memories_expired, report.memories_promoted,
        report.associations_linked,
    )
    return report


async def _deduplicate(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    org_id: str,
    agent_id: str,
    threshold: float,
    llm: LLMCallable | None,
    report: ConsolidationReport,
) -> None:
    """Find near-duplicate memories and merge them.

    Fetches all memories in one query, computes pairwise cosine similarity
    in numpy (O(n²) but with BLAS-optimized matmul), then merges pairs.
    Eliminates N individual database round trips.
    """
    rows = await pool.fetch(
        """
        SELECT id, content, memory_type, importance, embedding, created_at
        FROM memory
        WHERE org_id = $1 AND agent_id = $2
          AND valid_to IS NULL AND immutable = false
        ORDER BY created_at ASC
        """,
        org_id,
        agent_id,
    )

    if len(rows) < 2:
        return

    n = len(rows)

    def _parse_embedding(val):
        if isinstance(val, str):
            return np.fromstring(val.strip("[]"), dtype=np.float32, sep=",")
        return np.asarray(val, dtype=np.float32)

    first_emb = _parse_embedding(rows[0]["embedding"])
    dim = len(first_emb)
    mat = np.empty((n, dim), dtype=np.float32)
    mat[0] = first_emb
    for i in range(1, n):
        mat[i] = _parse_embedding(rows[i]["embedding"])

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    mat_normed = mat / norms

    sim = mat_normed @ mat_normed.T

    pair_is, pair_js = np.where(np.triu(sim, k=1) >= threshold)

    if len(pair_is) == 0:
        return

    merged_ids: set[uuid.UUID] = set()
    supersede_ids = []
    supersede_by = []
    history_ids = []
    history_contents = []
    update_newer = []

    for idx in range(len(pair_is)):
        i, j = int(pair_is[idx]), int(pair_js[idx])
        if rows[i]["id"] in merged_ids or rows[j]["id"] in merged_ids:
            continue

        ri, rj = rows[i], rows[j]
        if ri["created_at"] <= rj["created_at"]:
            older_row, newer_row = ri, rj
        else:
            older_row, newer_row = rj, ri

        if llm:
            prompt = (
                "Merge these two similar memories into one concise, complete memory. "
                "Keep all unique information. Return only the merged text, nothing else.\n\n"
                f"Memory 1: {older_row['content']}\n"
                f"Memory 2: {newer_row['content']}"
            )
            try:
                merged_content = (await llm(prompt)).strip()
            except Exception as e:
                report.errors.append(f"Merge failed for {older_row['id']}/{newer_row['id']}: {e}")
                continue
        else:
            if len(newer_row["content"]) >= len(older_row["content"]):
                merged_content = newer_row["content"]
            else:
                merged_content = older_row["content"]

        if merged_content != newer_row["content"]:
            embedding = embedder.embed(merged_content)
            vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
            update_newer.append((merged_content, vec_str, extract_entities(merged_content), newer_row["id"]))

        supersede_ids.append(older_row["id"])
        supersede_by.append(newer_row["id"])
        history_ids.append(older_row["id"])
        history_contents.append(older_row["content"])
        merged_ids.add(older_row["id"])
        report.duplicates_merged += 1

    if not supersede_ids:
        return

    async with pool.acquire() as conn:
        async with conn.transaction():
            for content, vec_str, entities, nid in update_newer:
                # Use savepoint so a unique violation doesn't abort the whole transaction
                try:
                    async with conn.transaction():
                        await conn.execute(
                            "UPDATE memory SET content = $1, embedding = $2::vector, entities = $3 WHERE id = $4",
                            content, vec_str, entities, nid,
                        )
                except asyncpg.UniqueViolationError:
                    logger.debug("Dedup merge skipped (content collision): %s", content[:60])
            await conn.executemany(
                "UPDATE memory SET valid_to = now(), superseded_by = $2 WHERE id = $1",
                list(zip(supersede_ids, supersede_by)),
            )
            await conn.executemany(
                "INSERT INTO memory_history (memory_id, action, old_content, new_content, changed_by) VALUES ($1, 'superseded', $2, NULL, 'consolidation')",
                list(zip(history_ids, history_contents)),
            )


async def _decay_importance(
    pool: asyncpg.Pool,
    org_id: str,
    agent_id: str,
    decay_7d: float,
    decay_30d: float,
    min_importance: float,
    report: ConsolidationReport,
) -> None:
    """Decay importance of unaccessed memories. Soft-delete if below minimum."""
    now = datetime.now(UTC)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    row = await pool.fetchrow(
        """
        WITH decay_30d AS (
            UPDATE memory
            SET importance = importance * $4
            WHERE org_id = $1 AND agent_id = $2
              AND valid_to IS NULL AND immutable = false
              AND accessed_at < $3
              AND importance > $5
            RETURNING id
        ), decay_7d AS (
            UPDATE memory
            SET importance = importance * $6
            WHERE org_id = $1 AND agent_id = $2
              AND valid_to IS NULL AND immutable = false
              AND accessed_at < $7 AND accessed_at >= $3
              AND importance > $5
              AND id NOT IN (SELECT id FROM decay_30d)
            RETURNING id
        ), expire AS (
            UPDATE memory
            SET valid_to = now()
            WHERE org_id = $1 AND agent_id = $2
              AND valid_to IS NULL AND immutable = false
              AND importance < $5
            RETURNING id
        )
        SELECT
            (SELECT count(*) FROM decay_30d) + (SELECT count(*) FROM decay_7d) AS decayed,
            (SELECT count(*) FROM expire) AS expired
        """,
        org_id, agent_id, cutoff_30d, decay_30d, min_importance, decay_7d, cutoff_7d,
    )
    report.memories_decayed = row["decayed"]
    report.memories_expired += row["expired"]


async def _expire_raw(
    pool: asyncpg.Pool,
    org_id: str,
    agent_id: str,
    ttl_days: int,
    report: ConsolidationReport,
) -> None:
    """Soft-delete raw memories older than TTL."""
    cutoff = datetime.now(UTC) - timedelta(days=ttl_days)

    result = await pool.execute(
        """
        UPDATE memory
        SET valid_to = now()
        WHERE org_id = $1 AND agent_id = $2
          AND memory_type = 'raw'
          AND valid_to IS NULL
          AND created_at < $3
        """,
        org_id, agent_id, cutoff,
    )
    count = int(result.split()[-1])
    report.memories_expired += count


async def _promote_with_llm(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    org_id: str,
    agent_id: str,
    llm: LLMCallable,
    report: ConsolidationReport,
) -> None:
    """Promote raw memories to event/insight using LLM distillation.

    Groups raw memories by topic similarity, then asks LLM to distill
    each group into a concise insight or event.
    """
    # Get unconsolidated raw memories
    rows = await pool.fetch(
        """
        SELECT id, content, created_at
        FROM memory
        WHERE org_id = $1 AND agent_id = $2
          AND memory_type = 'raw'
          AND valid_to IS NULL
          AND consolidated_at IS NULL
        ORDER BY created_at ASC
        LIMIT 50
        """,
        org_id, agent_id,
    )

    if len(rows) < 2:
        return

    # Batch distill — send all raw chunks to LLM for insight extraction
    raw_texts = "\n".join(f"- {r['content']}" for r in rows)
    prompt = (
        "Extract the key facts and insights from these conversation fragments. "
        "Return one insight per line, starting each line with '- '. "
        "Be concise. Only include non-obvious, factual information worth remembering "
        "(e.g. user preferences, decisions, personal details). "
        "If there is nothing meaningful to extract, respond with exactly: NONE\n\n"
        f"{raw_texts}"
    )

    try:
        response = await llm(prompt)
    except Exception as e:
        report.errors.append(f"Promotion LLM call failed: {e}")
        return

    # Check if LLM found nothing worth promoting
    stripped = response.strip()
    if stripped.upper() in ("NONE", "NONE.", "N/A", "- NONE"):
        logger.info("Promotion: LLM found no insights to extract from %d raw memories", len(rows))
        return

    insights = [line.strip().lstrip("- ") for line in stripped.split("\n") if line.strip()]

    # Store each extracted insight
    for insight_text in insights:
        # Skip short lines, meta-commentary, and refusal-like responses
        if len(insight_text) < 10:
            continue
        lower = insight_text.lower()
        if any(phrase in lower for phrase in [
            "cannot extract", "no meaningful", "insufficient",
            "nothing to extract", "no key facts", "no insights",
            "not enough", "too brief", "no substantive",
            "based on the conversation", "there is no",
        ]):
            continue
        embedding = embedder.embed(insight_text)
        vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

        try:
            await pool.execute(
                """
                INSERT INTO memory (
                    org_id, agent_id, content, memory_type, tags, embedding,
                    entities, importance
                ) VALUES ($1, $2, $3, 'insight', ARRAY[]::TEXT[], $4, $5, 0.6)
                ON CONFLICT (org_id, agent_id, content) DO NOTHING
                """,
                org_id, agent_id, insight_text, vec_str,
                extract_entities(insight_text),
            )
            report.memories_promoted += 1
        except Exception as e:
            report.errors.append(f"Promotion insert failed: {e}")
