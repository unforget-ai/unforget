"""Conversation ingest — 3 modes for storing full conversations as memories.

Modes:
  background:  Store raw chunks instantly, extract clean facts async (0 LLM hot path)
  immediate:   LLM extracts facts before storing (~1-2s, 1 LLM call)
  lightweight:  NER + heuristics only, no LLM (~50ms)
"""

from __future__ import annotations

import logging
from typing import Literal

import asyncpg

from unforget.chunker import chunk_messages
from unforget.consolidation import LLMCallable
from unforget.embedder import BaseEmbedder
from unforget.entities import extract_entities
from unforget.types import MemoryItem, MemoryType

logger = logging.getLogger("unforget.ingest")

IngestMode = Literal["background", "immediate", "lightweight"]


async def ingest_conversation(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    *,
    messages: list[dict[str, str]],
    org_id: str,
    agent_id: str,
    mode: IngestMode = "background",
    llm: LLMCallable | None = None,
    source_thread_id: str | None = None,
    min_sentences: int = 2,
    max_sentences: int = 8,
    overlap: int = 1,
) -> list[MemoryItem]:
    """Ingest a conversation into memory.

    Args:
        pool: Database connection pool.
        embedder: Embedding model.
        messages: Conversation as [{"role": "user", "content": "..."}, ...].
        org_id: Organization scope.
        agent_id: Agent scope.
        mode: "background" (raw chunks, 0 LLM), "immediate" (LLM extract),
              "lightweight" (NER heuristics).
        llm: Required for mode="immediate". async (prompt) -> str.
        source_thread_id: Link memories to their source thread.

    Returns:
        List of created MemoryItems.
    """
    if mode == "immediate" and llm is None:
        raise ValueError("mode='immediate' requires an llm callable.")

    if not messages:
        return []

    if mode == "background":
        return await _ingest_background(
            pool, embedder, messages, org_id, agent_id, source_thread_id,
            min_sentences, max_sentences, overlap,
        )
    elif mode == "immediate":
        return await _ingest_immediate(
            pool, embedder, messages, org_id, agent_id, source_thread_id, llm,
        )
    elif mode == "lightweight":
        return await _ingest_lightweight(
            pool, embedder, messages, org_id, agent_id, source_thread_id,
            min_sentences, max_sentences, overlap,
        )
    else:
        raise ValueError(f"Unknown ingest mode: {mode}")


async def _ingest_background(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    messages: list[dict[str, str]],
    org_id: str,
    agent_id: str,
    source_thread_id: str | None,
    min_sentences: int,
    max_sentences: int,
    overlap: int,
) -> list[MemoryItem]:
    """Store raw chunks immediately. Consolidation will promote them later."""
    chunks = chunk_messages(
        messages,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        overlap=overlap,
    )

    if not chunks:
        return []

    # Batch embed all chunks
    embeddings = embedder.embed_batch(chunks)

    results: list[MemoryItem] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for chunk, embedding in zip(chunks, embeddings):
                entities = extract_entities(chunk)
                vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

                row = await conn.fetchrow(
                    """
                    INSERT INTO memory (
                        org_id, agent_id, content, memory_type, tags, embedding,
                        entities, importance, source_thread_id
                    ) VALUES ($1, $2, $3, 'raw', ARRAY[]::TEXT[], $4, $5, 0.3, $6)
                    ON CONFLICT (org_id, agent_id, content) DO NOTHING
                    RETURNING *
                    """,
                    org_id, agent_id, chunk, vec_str, entities, source_thread_id,
                )
                if row:
                    results.append(_row_to_item(row))

    logger.debug("Ingested %d raw chunks for %s/%s (background)", len(results), org_id, agent_id)
    return results


async def _ingest_immediate(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    messages: list[dict[str, str]],
    org_id: str,
    agent_id: str,
    source_thread_id: str | None,
    llm: LLMCallable,
) -> list[MemoryItem]:
    """LLM extracts facts from conversation, stores as insights."""
    # Build conversation text for LLM
    conv_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
    )

    prompt = (
        "Extract the key facts, preferences, and decisions from this conversation. "
        "Return one fact per line. Be concise. Only include information worth remembering "
        "across future conversations. Skip greetings and small talk.\n\n"
        f"{conv_text}"
    )

    response = await llm(prompt)
    facts = [
        line.strip().lstrip("- •0123456789.")
        for line in response.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    if not facts:
        return []

    # Batch embed all facts
    embeddings = embedder.embed_batch(facts)

    results: list[MemoryItem] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for fact, embedding in zip(facts, embeddings):
                entities = extract_entities(fact)
                vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

                row = await conn.fetchrow(
                    """
                    INSERT INTO memory (
                        org_id, agent_id, content, memory_type, tags, embedding,
                        entities, importance, source_thread_id
                    ) VALUES ($1, $2, $3, 'insight', ARRAY[]::TEXT[], $4, $5, 0.6, $6)
                    ON CONFLICT (org_id, agent_id, content) DO NOTHING
                    RETURNING *
                    """,
                    org_id, agent_id, fact, vec_str, entities, source_thread_id,
                )
                if row:
                    results.append(_row_to_item(row))

    logger.debug("Ingested %d insights for %s/%s (immediate)", len(results), org_id, agent_id)
    return results


async def _ingest_lightweight(
    pool: asyncpg.Pool,
    embedder: BaseEmbedder,
    messages: list[dict[str, str]],
    org_id: str,
    agent_id: str,
    source_thread_id: str | None,
    min_sentences: int,
    max_sentences: int,
    overlap: int,
) -> list[MemoryItem]:
    """Chunk + NER heuristics, no LLM. Stores as events."""
    chunks = chunk_messages(
        messages,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        overlap=overlap,
    )

    if not chunks:
        return []

    embeddings = embedder.embed_batch(chunks)

    results: list[MemoryItem] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for chunk, embedding in zip(chunks, embeddings):
                entities = extract_entities(chunk)
                vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

                row = await conn.fetchrow(
                    """
                    INSERT INTO memory (
                        org_id, agent_id, content, memory_type, tags, embedding,
                        entities, importance, source_thread_id
                    ) VALUES ($1, $2, $3, 'event', ARRAY[]::TEXT[], $4, $5, 0.5, $6)
                    ON CONFLICT (org_id, agent_id, content) DO NOTHING
                    RETURNING *
                    """,
                    org_id, agent_id, chunk, vec_str, entities, source_thread_id,
                )
                if row:
                    results.append(_row_to_item(row))

    logger.debug("Ingested %d events for %s/%s (lightweight)", len(results), org_id, agent_id)
    return results


def _row_to_item(row: asyncpg.Record) -> MemoryItem:
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
