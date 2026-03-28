"""Temporal validity operations — supersede, timeline, expiry.

Implements the Graphiti/Zep pattern: facts have valid_from/valid_to.
Contradictions soft-delete (valid_to = now) with a superseded_by link.
Full audit trail via memory_history.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import asyncpg

from unforget.entities import extract_entities
from unforget.types import MemoryItem, MemoryType


async def supersede(
    pool: asyncpg.Pool,
    *,
    old_id: uuid.UUID,
    new_content: str,
    new_embedding: list[float],
    org_id: str,
    agent_id: str,
    memory_type: MemoryType | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
) -> tuple[MemoryItem, MemoryItem] | None:
    """Supersede an existing memory with a new one.

    1. Sets old memory's valid_to = now(), superseded_by = new.id
    2. Creates a new memory inheriting org/agent/type/tags from old (or overrides)
    3. Records both changes in history

    Returns (old_item, new_item) or None if old_id not found.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Fetch old memory
            old_row = await conn.fetchrow(
                "SELECT * FROM memory WHERE id = $1 FOR UPDATE",
                old_id,
            )
            if not old_row:
                return None

            if old_row["immutable"]:
                raise ValueError(
                    f"Memory {old_id} is immutable and cannot be superseded."
                )

            if old_row["valid_to"] is not None:
                raise ValueError(
                    f"Memory {old_id} is already superseded (valid_to={old_row['valid_to']})."
                )

            # Resolve values — inherit from old if not overridden
            mt = memory_type or MemoryType(old_row["memory_type"])
            mt_val = mt.value if isinstance(mt, MemoryType) else mt
            new_tags = tags if tags is not None else (old_row["tags"] or [])
            new_importance = importance if importance is not None else old_row["importance"]

            vec_str = "[" + ",".join(f"{v:.6f}" for v in new_embedding) + "]"

            # Insert new memory
            new_row = await conn.fetchrow(
                """
                INSERT INTO memory (
                    org_id, agent_id, content, memory_type, tags, embedding,
                    entities, importance, shared, immutable,
                    source_thread_id, source_message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, false,
                    $10, $11
                )
                RETURNING *
                """,
                org_id,
                agent_id,
                new_content,
                mt_val,
                new_tags,
                vec_str,
                extract_entities(new_content),
                new_importance,
                old_row["shared"],
                old_row["source_thread_id"],
                old_row["source_message"],
            )

            new_id = new_row["id"]

            # Mark old as superseded
            updated_old = await conn.fetchrow(
                """
                UPDATE memory
                SET valid_to = now(), superseded_by = $2
                WHERE id = $1
                RETURNING *
                """,
                old_id,
                new_id,
            )

            # History for old memory
            await conn.execute(
                """
                INSERT INTO memory_history (memory_id, action, old_content, new_content, changed_by)
                VALUES ($1, 'superseded', $2, NULL, 'agent')
                """,
                old_id,
                old_row["content"],
            )

            # History for new memory
            await conn.execute(
                """
                INSERT INTO memory_history (memory_id, action, old_content, new_content, changed_by)
                VALUES ($1, 'created', NULL, $2, 'agent')
                """,
                new_id,
                new_content,
            )

    return _row_to_item(updated_old), _row_to_item(new_row)


async def timeline(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    agent_id: str,
    at: datetime,
    memory_type: str | None = None,
    limit: int = 50,
) -> list[MemoryItem]:
    """Query "what was true at time T" for an agent.

    Returns memories that were valid at the given timestamp:
    - valid_from <= at AND (valid_to IS NULL OR valid_to > at)
    """
    conditions = [
        "org_id = $1",
        "agent_id = $2",
        "valid_from <= $3",
        "(valid_to IS NULL OR valid_to > $3)",
    ]
    params: list = [org_id, agent_id, at]
    idx = 4

    if memory_type:
        conditions.append(f"memory_type = ${idx}")
        params.append(memory_type)
        idx += 1

    params.append(limit)

    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT * FROM memory
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ${idx}
        """,
        *params,
    )
    return [_row_to_item(r) for r in rows]


async def get_supersession_chain(
    pool: asyncpg.Pool,
    memory_id: uuid.UUID,
) -> list[MemoryItem]:
    """Follow the supersession chain from a memory.

    Returns [oldest → ... → newest], including the given memory.
    Walks backwards (via content matching) and forwards (via superseded_by).
    """
    chain = []

    # Walk forward from the given ID

    # First, walk backward to find the root
    backward: list[asyncpg.Record] = []
    seen = {memory_id}
    row = await pool.fetchrow("SELECT * FROM memory WHERE id = $1", memory_id)
    if not row:
        return []

    backward.append(row)

    # Find predecessors (memories that were superseded by this one)
    prev_id = memory_id
    while True:
        prev_row = await pool.fetchrow(
            "SELECT * FROM memory WHERE superseded_by = $1",
            prev_id,
        )
        if not prev_row or prev_row["id"] in seen:
            break
        seen.add(prev_row["id"])
        backward.append(prev_row)
        prev_id = prev_row["id"]

    backward.reverse()
    chain.extend(backward)

    # Walk forward via superseded_by links
    current_row = row
    while current_row["superseded_by"] is not None:
        next_id = current_row["superseded_by"]
        if next_id in seen:
            break
        seen.add(next_id)
        next_row = await pool.fetchrow("SELECT * FROM memory WHERE id = $1", next_id)
        if not next_row:
            break
        chain.append(next_row)
        current_row = next_row

    return [_row_to_item(r) for r in chain]


def _row_to_item(row: asyncpg.Record) -> MemoryItem:
    """Convert a database row to a MemoryItem."""
    return MemoryItem(
        id=row["id"],
        org_id=row["org_id"],
        agent_id=row["agent_id"],
        content=row["content"],
        memory_type=MemoryType(row["memory_type"]),
        tags=row["tags"] or [],
        entities=row["entities"] or [],
        importance=row["importance"],
        access_count=row["access_count"],
        shared=row["shared"],
        immutable=row["immutable"],
        valid_from=row["valid_from"],
        valid_to=row.get("valid_to"),
        superseded_by=row.get("superseded_by"),
        source_thread_id=row.get("source_thread_id"),
        source_message=row.get("source_message"),
        expires_at=row.get("expires_at"),
        created_at=row["created_at"],
        accessed_at=row["accessed_at"],
        consolidated_at=row.get("consolidated_at"),
    )
