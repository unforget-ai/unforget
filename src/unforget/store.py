"""MemoryStore — the main API for unforget."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:
    from unforget.scoped import ScopedMemory

from unforget.cache import TTLCache
from unforget.consolidation import ConsolidationReport, LLMCallable
from unforget.consolidation import consolidate as _consolidate
from unforget.embedder import BaseEmbedder, Embedder
from unforget.entities import extract_entities
from unforget.ingest import IngestMode, ingest_conversation
from unforget.quotas import RateLimiter
from unforget.reranker import Reranker
from unforget.retrieval import RetrievalConfig, four_channel_recall
from unforget.schema import ensure_schema
from unforget.temporal import (
    get_supersession_chain,
)
from unforget.temporal import (
    supersede as _supersede,
)
from unforget.temporal import (
    timeline as _timeline,
)
from unforget.types import (
    HistoryAction,
    MemoryHistoryEntry,
    MemoryItem,
    MemoryResult,
    MemoryStats,
    MemoryType,
    WriteItem,
)

logger = logging.getLogger("unforget.store")


class MemoryStore:
    """Persistent memory for AI agents.

    Zero LLM on write. 4-channel retrieval. PostgreSQL-only.

    Usage::

        store = MemoryStore("postgresql://user:pass@localhost/db")
        await store.initialize()

        await store.write("User prefers Fly.io", org_id="acme", agent_id="bot")
        results = await store.recall("deployment preferences", org_id="acme", agent_id="bot")

        await store.close()
    """

    def __init__(
        self,
        database_url: str,
        *,
        embedder: BaseEmbedder | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_enabled: bool = True,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        ef_search: int = 100,
        max_memories_per_agent: int = 10_000,
        rrf_k: int = 60,
        channel_weights: dict[str, float] | None = None,
        type_boosts: dict[str, float] | None = None,
        recall_cache_ttl: float = 60.0,
        recall_cache_size: int = 1000,
        max_writes_per_minute: int = 100,
    ):
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None
        self._embedder: BaseEmbedder = embedder or Embedder(embedding_model)
        self._reranker = Reranker(reranker_model) if reranker_enabled else None
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._ef_search = ef_search
        self._max_memories_per_agent = max_memories_per_agent
        self._rate_limiter = RateLimiter(max_per_minute=max_writes_per_minute)
        self._retrieval_config = RetrievalConfig(
            rrf_k=rrf_k,
            channel_weights=channel_weights or RetrievalConfig().channel_weights,
            type_boosts=type_boosts or RetrievalConfig().type_boosts,
            ef_search=ef_search,
        )
        self._recall_cache = TTLCache(maxsize=recall_cache_size, ttl=recall_cache_ttl)
        self._scheduler: Any | None = None  # ConsolidationScheduler, if attached
        self._write_count: int = 0
        self._quota_cache: dict[tuple[str, str], int] = {}
        self._quota_check_interval: int = 50

    def bind(self, org_id: str, agent_id: str) -> ScopedMemory:
        """Return a scoped client with org_id + agent_id pre-filled.

        Usage::

            memory = store.bind(org_id="acme", agent_id="bot")
            await memory.write("User prefers dark mode")
            results = await memory.recall("user preferences")
        """
        from unforget.scoped import ScopedMemory
        return ScopedMemory(self, org_id, agent_id)

    def attach_scheduler(self, scheduler: Any) -> None:
        """Attach a ConsolidationScheduler to this store."""
        self._scheduler = scheduler

    def detach_scheduler(self) -> None:
        """Detach the consolidation scheduler."""
        self._scheduler = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("MemoryStore not initialized. Call await store.initialize() first.")
        return self._pool

    @property
    def dims(self) -> int:
        return self._embedder.dims

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to the database, create tables, preload models."""
        self._embedder.preload()
        # Warm inference: first call is always slow due to PyTorch JIT/CPU ramp-up.
        # Do a throwaway inference so real queries are fast from the start.
        self._embedder.embed("warmup")
        if self._reranker:
            self._reranker.preload()
            # Warm inference with a dummy rerank call
            from unforget.types import MemoryResult
            _dummy = MemoryResult(
                id=uuid.uuid4(), content="warmup", memory_type=MemoryType.INSIGHT,
                tags=[], entities=[], importance=0.5, score=0.0,
                created_at=datetime.now(UTC), accessed_at=datetime.now(UTC),
            )
            self._reranker.rerank("warmup", [_dummy])
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )
        await ensure_schema(self._pool, dims=self._embedder.dims)
        logger.info("MemoryStore initialized (dims=%d)", self._embedder.dims)

    async def close(self) -> None:
        """Close the connection pool. Stops scheduler if running."""
        if self._scheduler is not None and self._scheduler.is_running:
            await self._scheduler.stop()
        if self._pool:
            await self._pool.close()
            self._pool = None

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    async def write(
        self,
        content: str,
        *,
        org_id: str,
        agent_id: str,
        memory_type: MemoryType | str = MemoryType.INSIGHT,
        tags: list[str] | None = None,
        importance: float = 0.5,
        shared: bool = False,
        immutable: bool = False,
        source_thread_id: str | None = None,
        source_message: str | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryItem:
        """Store a single memory. Instant — zero LLM calls.

        Raises asyncpg.UniqueViolationError if exact duplicate content exists.
        """
        if isinstance(memory_type, str):
            memory_type = MemoryType(memory_type)

        # Check rate limit + quota
        self._rate_limiter.check(org_id, agent_id)
        await self._check_quota(org_id, agent_id)

        # Embed
        embedding = self._embedder.embed(content)

        # Insert + audit trail in a single round-trip
        row = await self.pool.fetchrow(
            """
            WITH ins AS (
                INSERT INTO memory (
                    org_id, agent_id, content, memory_type, tags, embedding,
                    entities, importance, shared, immutable,
                    source_thread_id, source_message, expires_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12, $13
                )
                RETURNING *
            ), hist AS (
                INSERT INTO memory_history (memory_id, action, old_content, new_content, changed_by)
                SELECT id, 'created', NULL, content, 'agent' FROM ins
            )
            SELECT * FROM ins
            """,
            org_id,
            agent_id,
            content,
            memory_type.value,
            tags or [],
            _vec_str(embedding),
            extract_entities(content),
            importance,
            shared,
            immutable,
            source_thread_id,
            source_message,
            expires_at,
        )

        logger.debug("Wrote memory %s for %s/%s", row["id"], org_id, agent_id)
        self._write_count += 1
        if self._scheduler is not None:
            self._scheduler.notify_write()
        return _row_to_item(row)

    async def write_batch(
        self,
        items: list[WriteItem],
        *,
        org_id: str,
        agent_id: str,
    ) -> list[MemoryItem]:
        """Write multiple memories in a single batch. Uses batch embedding."""
        if not items:
            return []

        self._rate_limiter.check(org_id, agent_id)
        await self._check_quota(org_id, agent_id, count=len(items))

        # Batch embed
        texts = [item.content for item in items]
        embeddings = self._embedder.embed_batch(texts)

        results = []
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for item, embedding in zip(items, embeddings):
                    memory_type = (
                        item.memory_type
                        if isinstance(item.memory_type, MemoryType)
                        else MemoryType(item.memory_type)
                    )
                    row = await conn.fetchrow(
                        """
                        WITH ins AS (
                            INSERT INTO memory (
                                org_id, agent_id, content, memory_type, tags, embedding,
                                entities, importance, shared, immutable,
                                source_thread_id, source_message, expires_at
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6,
                                $7, $8, $9, $10,
                                $11, $12, $13
                            )
                            ON CONFLICT (org_id, agent_id, content) DO NOTHING
                            RETURNING *
                        ), hist AS (
                            INSERT INTO memory_history (memory_id, action, old_content, new_content, changed_by)
                            SELECT id, 'created', NULL, content, 'agent' FROM ins
                        )
                        SELECT * FROM ins
                        """,
                        org_id,
                        agent_id,
                        item.content,
                        memory_type.value,
                        item.tags,
                        _vec_str(embedding),
                        [],
                        item.importance,
                        item.shared,
                        item.immutable,
                        item.source_thread_id,
                        item.source_message,
                        item.expires_at,
                    )
                    if row:
                        results.append(_row_to_item(row))

        logger.debug("Batch wrote %d/%d memories for %s/%s", len(results), len(items), org_id, agent_id)
        if results and self._scheduler is not None:
            self._write_count += len(results)
            for _ in results:
                self._scheduler.notify_write()
        return results

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    async def ingest(
        self,
        messages: list[dict[str, str]],
        *,
        org_id: str,
        agent_id: str,
        mode: IngestMode = "background",
        llm: LLMCallable | None = None,
        source_thread_id: str | None = None,
    ) -> list[MemoryItem]:
        """Ingest a full conversation into memory.

        Three modes:
          - "background": Store raw chunks instantly (0 LLM). Consolidation promotes later.
          - "immediate": LLM extracts facts, stores as insights (1 LLM call).
          - "lightweight": NER + heuristics, stores as events (0 LLM).

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            mode: Processing mode.
            llm: Required for mode="immediate". async (prompt) -> str.
            source_thread_id: Link memories to source thread.
        """
        results = await ingest_conversation(
            self.pool,
            self._embedder,
            messages=messages,
            org_id=org_id,
            agent_id=agent_id,
            mode=mode,
            llm=llm,
            source_thread_id=source_thread_id,
        )

        # Invalidate cache
        if results:
            self._recall_cache.clear()

        return results

    # -------------------------------------------------------------------------
    # Recall
    # -------------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        *,
        org_id: str,
        agent_id: str,
        limit: int = 10,
        memory_type: MemoryType | str | None = None,
        include_shared: bool = True,
        threshold: float = 0.0,
        rerank: bool = True,
        use_cache: bool = True,
    ) -> list[MemoryResult]:
        """Search memories using 4-channel retrieval with RRF fusion.

        Channels: semantic (pgvector), BM25 (tsvector), entity overlap, temporal recency.
        Results are fused with Reciprocal Rank Fusion, boosted by memory type,
        and optionally reranked with a cross-encoder model.
        """
        import time as _time
        _t0 = _time.perf_counter()

        mt = memory_type.value if isinstance(memory_type, MemoryType) else memory_type

        # Check cache
        cache_key = None
        if use_cache:
            cache_key = TTLCache.make_key(
                query, org_id, agent_id,
                limit=limit, memory_type=mt, include_shared=include_shared,
                threshold=threshold, rerank=rerank,
            )
            cached = self._recall_cache.get(cache_key)
            if cached is not None:
                logger.info("[recall] CACHE HIT — 0ms")
                return cached

        _t1 = _time.perf_counter()
        embedding = self._embedder.embed(query)
        _t2 = _time.perf_counter()

        query_entities = extract_entities(query)
        _t3 = _time.perf_counter()

        # 4-channel retrieval
        # Reranker: fetch limit + tiny buffer. Each extra pair costs ~3-5ms in reranking.
        fetch_limit = min(limit + 2, 20) if (rerank and self._reranker) else limit

        async with self.pool.acquire() as conn:
            results = await four_channel_recall(
                conn,
                embedding=embedding,
                query_text=query,
                query_entities=query_entities,
                org_id=org_id,
                agent_id=agent_id,
                include_shared=include_shared,
                memory_type=mt,
                limit=fetch_limit,
                config=self._retrieval_config,
            )
        _t4 = _time.perf_counter()

        # Cross-encoder reranking
        if rerank and self._reranker and results:
            results = self._reranker.rerank(query, results, top_k=limit)
        else:
            results = results[:limit]
        _t5 = _time.perf_counter()

        # Filter by threshold
        if threshold > 0:
            results = [r for r in results if r.score >= threshold]

        # Update access timestamps
        ids_to_touch = [r.id for r in results]
        if ids_to_touch:
            await self._touch_memories(ids_to_touch)
        _t6 = _time.perf_counter()

        # Cache results
        if cache_key is not None:
            self._recall_cache.set(cache_key, results)

        _total = (_t6 - _t0) * 1000
        logger.debug(
            "[recall] TOTAL=%.1fms | embed=%.1fms | entities=%.1fms | "
            "4-channel=%.1fms | rerank=%.1fms | touch=%.1fms | results=%d",
            _total,
            (_t2 - _t1) * 1000,
            (_t3 - _t2) * 1000,
            (_t4 - _t3) * 1000,
            (_t5 - _t4) * 1000,
            (_t6 - _t5) * 1000,
            len(results),
        )

        return results

    async def auto_recall(
        self,
        query: str,
        *,
        org_id: str,
        agent_id: str,
        max_tokens: int = 2000,
        limit: int = 10,
    ) -> str:
        """Recall memories and format them for system prompt injection.

        Returns a string like:
            [Memory Context]
            - User prefers Fly.io for deployments
            - Last deploy used rolling strategy
        """
        results = await self.recall(
            query, org_id=org_id, agent_id=agent_id, limit=limit
        )
        if not results:
            return ""

        lines = []
        token_estimate = 0
        for r in results:
            line = f"- {r.content}"
            # Rough token estimate: ~4 chars per token
            token_estimate += len(line) // 4
            if token_estimate > max_tokens:
                break
            lines.append(line)

        if not lines:
            return ""

        return "[Memory Context]\n" + "\n".join(lines)

    # -------------------------------------------------------------------------
    # Get / List
    # -------------------------------------------------------------------------

    async def get(self, memory_id: uuid.UUID) -> MemoryItem | None:
        """Get a single memory by ID."""
        row = await self.pool.fetchrow("SELECT * FROM memory WHERE id = $1", memory_id)
        return _row_to_item(row) if row else None

    async def list(
        self,
        *,
        org_id: str,
        agent_id: str,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        include_expired: bool = False,
        include_shared: bool = False,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        importance_gte: float | None = None,
        importance_lte: float | None = None,
        search: str | None = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 100,
    ) -> list[MemoryItem]:
        """List memories with advanced filtering, search, and pagination.

        Filters:
            memory_type: Filter by type (insight/event/raw).
            tags: Filter by tag overlap (any match).
            include_expired: Include superseded/expired memories.
            include_shared: Include shared memories from other agents.
            created_after/before: Date range filter.
            importance_gte/lte: Importance range filter.
            search: Full-text search within content (uses tsvector).
            sort_by: Column to sort by (created_at, importance, accessed_at).
            sort_order: "asc" or "desc".
        """
        if include_shared:
            conditions = ["org_id = $1", "(agent_id = $2 OR shared = true)"]
        else:
            conditions = ["org_id = $1", "agent_id = $2"]
        params: list = [org_id, agent_id]
        idx = 3

        if not include_expired:
            conditions.append("valid_to IS NULL")

        if memory_type:
            mt = memory_type if isinstance(memory_type, str) else memory_type.value
            conditions.append(f"memory_type = ${idx}")
            params.append(mt)
            idx += 1

        if tags:
            conditions.append(f"tags && ${idx}::text[]")
            params.append(tags)
            idx += 1

        if created_after:
            conditions.append(f"created_at >= ${idx}")
            params.append(created_after)
            idx += 1

        if created_before:
            conditions.append(f"created_at <= ${idx}")
            params.append(created_before)
            idx += 1

        if importance_gte is not None:
            conditions.append(f"importance >= ${idx}")
            params.append(importance_gte)
            idx += 1

        if importance_lte is not None:
            conditions.append(f"importance <= ${idx}")
            params.append(importance_lte)
            idx += 1

        if search:
            conditions.append(f"search_vector @@ plainto_tsquery('english', ${idx})")
            params.append(search)
            idx += 1

        # Validate sort
        allowed_sorts = {"created_at", "importance", "accessed_at", "access_count"}
        if sort_by not in allowed_sorts:
            sort_by = "created_at"
        direction = "ASC" if sort_order.lower() == "asc" else "DESC"

        offset = (page - 1) * page_size
        params.extend([page_size, offset])

        where = " AND ".join(conditions)
        rows = await self.pool.fetch(
            f"""
            SELECT * FROM memory
            WHERE {where}
            ORDER BY {sort_by} {direction}
            LIMIT ${idx} OFFSET ${idx + 1}
            """,
            *params,
        )
        return [_row_to_item(r) for r in rows]

    # -------------------------------------------------------------------------
    # Update / Delete
    # -------------------------------------------------------------------------

    async def update(
        self,
        memory_id: uuid.UUID,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> MemoryItem | None:
        """Update a memory's content, tags, or importance."""
        existing = await self.pool.fetchrow("SELECT * FROM memory WHERE id = $1", memory_id)
        if not existing:
            return None
        if existing["immutable"]:
            raise ValueError(f"Memory {memory_id} is immutable and cannot be updated.")

        sets = []
        params: list = []
        idx = 1

        old_content = existing["content"]

        if content is not None and content != old_content:
            embedding = self._embedder.embed(content)
            sets.append(f"content = ${idx}")
            params.append(content)
            idx += 1
            sets.append(f"embedding = ${idx}::vector")
            params.append(_vec_str(embedding))
            idx += 1

        if tags is not None:
            sets.append(f"tags = ${idx}")
            params.append(tags)
            idx += 1

        if importance is not None:
            sets.append(f"importance = ${idx}")
            params.append(importance)
            idx += 1

        if not sets:
            return _row_to_item(existing)

        params.append(memory_id)
        set_clause = ", ".join(sets)
        row = await self.pool.fetchrow(
            f"UPDATE memory SET {set_clause} WHERE id = ${idx} RETURNING *",
            *params,
        )

        if content is not None and content != old_content:
            await self._record_history(
                memory_id, HistoryAction.UPDATED, old_content, content, "user"
            )

        return _row_to_item(row) if row else None

    async def forget(self, memory_id: uuid.UUID) -> bool:
        """Delete a single memory (hard delete)."""
        existing = await self.pool.fetchrow(
            "SELECT content FROM memory WHERE id = $1", memory_id
        )
        if not existing:
            return False

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._record_history(
                    memory_id, HistoryAction.DELETED, existing["content"], None, "user", conn=conn
                )
                # Clear superseded_by references pointing to this memory
                await conn.execute(
                    "UPDATE memory SET superseded_by = NULL WHERE superseded_by = $1", memory_id
                )
                await conn.execute("DELETE FROM memory WHERE id = $1", memory_id)
        return True

    async def forget_all(self, *, org_id: str, agent_id: str) -> int:
        """Delete all memories for an agent. Returns count deleted."""
        result = await self.pool.execute(
            "DELETE FROM memory WHERE org_id = $1 AND agent_id = $2",
            org_id,
            agent_id,
        )
        count = int(result.split()[-1])
        logger.info("Deleted %d memories for %s/%s", count, org_id, agent_id)
        return count

    async def bulk_delete(
        self,
        *,
        org_id: str,
        agent_id: str,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        older_than: datetime | None = None,
        importance_below: float | None = None,
    ) -> int:
        """Delete memories matching filters. Returns count deleted.

        At least one filter (beyond org/agent) is required to prevent
        accidental full wipes — use forget_all() for that.
        """
        if not any([memory_type, tags, older_than, importance_below]):
            raise ValueError(
                "At least one filter required for bulk_delete. "
                "Use forget_all() to delete everything."
            )

        conditions = ["org_id = $1", "agent_id = $2"]
        params: list = [org_id, agent_id]
        idx = 3

        if memory_type:
            mt = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
            conditions.append(f"memory_type = ${idx}")
            params.append(mt)
            idx += 1

        if tags:
            conditions.append(f"tags && ${idx}::text[]")
            params.append(tags)
            idx += 1

        if older_than:
            conditions.append(f"created_at < ${idx}")
            params.append(older_than)
            idx += 1

        if importance_below is not None:
            conditions.append(f"importance < ${idx}")
            params.append(importance_below)
            idx += 1

        where = " AND ".join(conditions)
        result = await self.pool.execute(f"DELETE FROM memory WHERE {where}", *params)
        count = int(result.split()[-1])
        logger.info("Bulk deleted %d memories for %s/%s", count, org_id, agent_id)
        return count

    # -------------------------------------------------------------------------
    # Temporal
    # -------------------------------------------------------------------------

    async def supersede(
        self,
        old_id: uuid.UUID,
        new_content: str,
        *,
        org_id: str,
        agent_id: str,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> tuple[MemoryItem, MemoryItem]:
        """Supersede an existing memory with a new one.

        Soft-deletes the old memory (sets valid_to) and creates a replacement
        linked via superseded_by. Full audit trail recorded.

        Returns (old_item, new_item).
        Raises ValueError if old memory is immutable or already superseded.
        """
        mt = MemoryType(memory_type) if isinstance(memory_type, str) else memory_type
        embedding = self._embedder.embed(new_content)

        result = await _supersede(
            self.pool,
            old_id=old_id,
            new_content=new_content,
            new_embedding=embedding,
            org_id=org_id,
            agent_id=agent_id,
            memory_type=mt,
            tags=tags,
            importance=importance,
        )
        if result is None:
            raise ValueError(f"Memory {old_id} not found.")

        # Invalidate recall cache — data changed
        self._recall_cache.clear()

        return result

    async def timeline(
        self,
        *,
        org_id: str,
        agent_id: str,
        at: datetime,
        memory_type: MemoryType | str | None = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Query "what was true at time T" for an agent.

        Returns memories that were valid at the given timestamp:
        valid_from <= at AND (valid_to IS NULL OR valid_to > at)
        """
        mt = None
        if memory_type:
            mt = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        return await _timeline(
            self.pool,
            org_id=org_id,
            agent_id=agent_id,
            at=at,
            memory_type=mt,
            limit=limit,
        )

    async def supersession_chain(self, memory_id: uuid.UUID) -> list[MemoryItem]:
        """Follow the supersession chain for a memory.

        Returns [oldest → ... → newest] — the full evolution of a fact.
        """
        return await get_supersession_chain(self.pool, memory_id)

    # -------------------------------------------------------------------------
    # Consolidation
    # -------------------------------------------------------------------------

    async def consolidate(
        self,
        *,
        org_id: str,
        agent_id: str,
        llm: LLMCallable | None = None,
        similarity_threshold: float | None = None,
    ) -> ConsolidationReport:
        """Run background consolidation for an agent's memories.

        Deduplicates near-duplicates, decays importance of stale memories,
        expires old raw chunks, and optionally promotes raw→insight via LLM.

        Args:
            org_id: Organization scope.
            agent_id: Agent scope.
            llm: Optional async callable for merging/promoting. async (prompt) -> str.
            similarity_threshold: Override default threshold for dedup.
        """
        report = await _consolidate(
            self.pool,
            self._embedder,
            org_id=org_id,
            agent_id=agent_id,
            similarity_threshold=similarity_threshold or 0.92,
            llm=llm,
        )

        # Invalidate cache — data changed
        self._recall_cache.clear()

        return report

    # -------------------------------------------------------------------------
    # History / Stats
    # -------------------------------------------------------------------------

    async def history(self, memory_id: uuid.UUID) -> list[MemoryHistoryEntry]:
        """Get the full audit trail for a memory."""
        rows = await self.pool.fetch(
            """
            SELECT * FROM memory_history
            WHERE memory_id = $1
            ORDER BY changed_at ASC
            """,
            memory_id,
        )
        return [
            MemoryHistoryEntry(
                id=r["id"],
                memory_id=r["memory_id"],
                action=HistoryAction(r["action"]),
                old_content=r["old_content"],
                new_content=r["new_content"],
                changed_at=r["changed_at"],
                changed_by=r["changed_by"],
            )
            for r in rows
        ]

    async def stats(self, *, org_id: str, agent_id: str) -> MemoryStats:
        """Aggregate stats for an agent's memory."""
        row = await self.pool.fetchrow(
            """
            SELECT
                COUNT(*) AS total,
                AVG(importance) AS avg_importance,
                MIN(created_at) AS oldest,
                MAX(created_at) AS newest
            FROM memory
            WHERE org_id = $1 AND agent_id = $2 AND valid_to IS NULL
            """,
            org_id,
            agent_id,
        )

        type_rows = await self.pool.fetch(
            """
            SELECT memory_type, COUNT(*) AS cnt
            FROM memory
            WHERE org_id = $1 AND agent_id = $2 AND valid_to IS NULL
            GROUP BY memory_type
            """,
            org_id,
            agent_id,
        )

        return MemoryStats(
            total=row["total"],
            by_type={r["memory_type"]: r["cnt"] for r in type_rows},
            avg_importance=float(row["avg_importance"] or 0),
            oldest=row["oldest"],
            newest=row["newest"],
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _check_quota(
        self, org_id: str, agent_id: str, *, count: int = 1
    ) -> None:
        """Raise if agent would exceed max_memories_per_agent."""
        key = (org_id, agent_id)
        cached = self._quota_cache.get(key)
        if cached is not None and cached + count <= self._max_memories_per_agent - self._quota_check_interval:
            self._quota_cache[key] = cached + count
            return

        current = await self.pool.fetchval(
            "SELECT COUNT(*) FROM memory WHERE org_id = $1 AND agent_id = $2 AND valid_to IS NULL",
            org_id,
            agent_id,
        )
        self._quota_cache[key] = current
        if current + count > self._max_memories_per_agent:
            raise MemoryQuotaExceeded(
                f"Agent {agent_id} has {current} memories "
                f"(limit: {self._max_memories_per_agent}). "
                f"Delete old memories or increase the quota."
            )

    async def _record_history(
        self,
        memory_id: uuid.UUID,
        action: HistoryAction,
        old_content: str | None,
        new_content: str | None,
        changed_by: str,
        *,
        conn: asyncpg.Connection | None = None,
    ) -> None:
        """Insert an audit trail record."""
        executor = conn or self.pool
        await executor.execute(
            """
            INSERT INTO memory_history (memory_id, action, old_content, new_content, changed_by)
            VALUES ($1, $2, $3, $4, $5)
            """,
            memory_id,
            action.value,
            old_content,
            new_content,
            changed_by,
        )

    async def _touch_memories(self, ids: list[uuid.UUID]) -> None:
        """Update accessed_at and increment access_count."""
        await self.pool.execute(
            """
            UPDATE memory
            SET accessed_at = now(), access_count = access_count + 1
            WHERE id = ANY($1::uuid[])
            """,
            ids,
        )


class MemoryQuotaExceeded(Exception):
    """Raised when an agent exceeds their memory quota."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec_str(vec: list[float]) -> str:
    """Format a vector as a pgvector-compatible string '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


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



