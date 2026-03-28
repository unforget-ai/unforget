"""Generic FastAPI router factory for unforget.

Usage:
    from fastapi import FastAPI
    from unforget import MemoryStore
    from unforget.api import create_memory_router

    app = FastAPI()
    store = MemoryStore("postgresql://...")
    app.include_router(create_memory_router(store), prefix="/v1/memory")

Provides 17 endpoints covering write, recall, ingest, list, get, update,
delete, bulk-delete, supersede, timeline, chain, history, stats, consolidate,
and auto-recall.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from unforget.consolidation import LLMCallable
from unforget.quotas import RateLimitExceeded
from unforget.store import MemoryQuotaExceeded, MemoryStore
from unforget.types import (
    MemoryHistoryEntry,
    MemoryItem,
    MemoryResult,
    MemoryStats,
    MemoryType,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class WriteRequest(BaseModel):
    content: str
    org_id: str
    agent_id: str
    memory_type: str = "insight"
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    shared: bool = False
    immutable: bool = False
    source_thread_id: str | None = None
    source_message: str | None = None
    expires_at: datetime | None = None


class WriteBatchRequest(BaseModel):
    items: list[WriteRequest]
    org_id: str
    agent_id: str


class RecallRequest(BaseModel):
    query: str
    org_id: str
    agent_id: str
    limit: int = 10
    memory_type: str | None = None
    include_shared: bool = True
    threshold: float = 0.0
    rerank: bool = True


class AutoRecallRequest(BaseModel):
    query: str
    org_id: str
    agent_id: str
    max_tokens: int = 2000
    limit: int = 10


class IngestRequest(BaseModel):
    messages: list[dict[str, str]]
    org_id: str
    agent_id: str
    mode: str = "background"
    source_thread_id: str | None = None


class SupersedeRequest(BaseModel):
    new_content: str
    org_id: str
    agent_id: str
    memory_type: str | None = None
    tags: list[str] | None = None
    importance: float | None = None


class UpdateRequest(BaseModel):
    content: str | None = None
    tags: list[str] | None = None
    importance: float | None = None


class BulkDeleteRequest(BaseModel):
    org_id: str
    agent_id: str
    memory_type: str | None = None
    tags: list[str] | None = None
    older_than: datetime | None = None
    importance_below: float | None = None


class ConsolidateRequest(BaseModel):
    org_id: str
    agent_id: str
    similarity_threshold: float | None = None


class TimelineRequest(BaseModel):
    org_id: str
    agent_id: str
    at: datetime
    memory_type: str | None = None
    limit: int = 50


class AutoRecallResponse(BaseModel):
    context: str
    memory_count: int


class CountResponse(BaseModel):
    count: int


class ConsolidateResponse(BaseModel):
    duplicates_merged: int
    memories_decayed: int
    memories_expired: int
    memories_promoted: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_memory_router(
    store: MemoryStore,
    *,
    llm: LLMCallable | None = None,
) -> APIRouter:
    """Create a FastAPI router with all unforget endpoints.

    Args:
        store: Initialized MemoryStore instance.
        llm: Optional LLM callable for ingest mode="immediate" and consolidation.

    Returns:
        APIRouter to mount on any FastAPI app.
    """
    router = APIRouter(tags=["memory"])

    # -------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------

    @router.post("/write", response_model=MemoryItem)
    async def write_memory(req: WriteRequest):
        """Store a single memory. Instant — zero LLM calls."""
        try:
            return await store.write(
                req.content,
                org_id=req.org_id,
                agent_id=req.agent_id,
                memory_type=req.memory_type,
                tags=req.tags,
                importance=req.importance,
                shared=req.shared,
                immutable=req.immutable,
                source_thread_id=req.source_thread_id,
                source_message=req.source_message,
                expires_at=req.expires_at,
            )
        except MemoryQuotaExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))
        except RateLimitExceeded as e:
            raise HTTPException(status_code=429, detail=str(e))
        except Exception as e:
            if "UniqueViolation" in type(e).__name__:
                raise HTTPException(status_code=409, detail="Duplicate memory content.")
            raise

    @router.post("/write/batch", response_model=list[MemoryItem])
    async def write_batch(req: WriteBatchRequest):
        """Write multiple memories in a single batch."""
        from unforget.types import WriteItem

        items = [
            WriteItem(
                content=r.content,
                memory_type=MemoryType(r.memory_type),
                tags=r.tags,
                importance=r.importance,
                shared=r.shared,
                immutable=r.immutable,
                source_thread_id=r.source_thread_id,
                source_message=r.source_message,
                expires_at=r.expires_at,
            )
            for r in req.items
        ]
        try:
            return await store.write_batch(items, org_id=req.org_id, agent_id=req.agent_id)
        except (MemoryQuotaExceeded, RateLimitExceeded) as e:
            raise HTTPException(status_code=429, detail=str(e))

    # -------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------

    @router.post("/ingest", response_model=list[MemoryItem])
    async def ingest(req: IngestRequest):
        """Ingest a full conversation. Modes: background, immediate, lightweight."""
        ingest_llm = llm if req.mode == "immediate" else None
        if req.mode == "immediate" and ingest_llm is None:
            raise HTTPException(
                status_code=400,
                detail="mode='immediate' requires an LLM to be configured on the server.",
            )
        try:
            return await store.ingest(
                req.messages,
                org_id=req.org_id,
                agent_id=req.agent_id,
                mode=req.mode,
                llm=ingest_llm,
                source_thread_id=req.source_thread_id,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # -------------------------------------------------------------------
    # Recall
    # -------------------------------------------------------------------

    @router.post("/recall", response_model=list[MemoryResult])
    async def recall(req: RecallRequest):
        """4-channel retrieval with RRF fusion and cross-encoder reranking."""
        return await store.recall(
            req.query,
            org_id=req.org_id,
            agent_id=req.agent_id,
            limit=req.limit,
            memory_type=req.memory_type,
            include_shared=req.include_shared,
            threshold=req.threshold,
            rerank=req.rerank,
        )

    @router.post("/auto-recall", response_model=AutoRecallResponse)
    async def auto_recall(req: AutoRecallRequest):
        """Recall and format for system prompt injection."""
        context = await store.auto_recall(
            req.query,
            org_id=req.org_id,
            agent_id=req.agent_id,
            max_tokens=req.max_tokens,
            limit=req.limit,
        )
        count = context.count("\n- ") if context else 0
        return AutoRecallResponse(context=context, memory_count=count)

    # -------------------------------------------------------------------
    # Get / List (fixed paths MUST come before /{memory_id} wildcard)
    # -------------------------------------------------------------------

    @router.get("/stats/", response_model=MemoryStats)
    async def memory_stats(
        org_id: str = Query(...),
        agent_id: str = Query(...),
    ):
        """Aggregate memory stats for an agent."""
        return await store.stats(org_id=org_id, agent_id=agent_id)

    @router.get("/", response_model=list[MemoryItem])
    async def list_memories(
        org_id: str = Query(...),
        agent_id: str = Query(...),
        memory_type: str | None = Query(None),
        tags: str | None = Query(None, description="Comma-separated tags"),
        include_expired: bool = Query(False),
        include_shared: bool = Query(False),
        created_after: datetime | None = Query(None),
        created_before: datetime | None = Query(None),
        importance_gte: float | None = Query(None),
        importance_lte: float | None = Query(None),
        search: str | None = Query(None),
        sort_by: str = Query("created_at"),
        sort_order: str = Query("desc"),
        page: int = Query(1, ge=1),
        page_size: int = Query(100, ge=1, le=1000),
    ):
        """List memories with advanced filtering and pagination."""
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        return await store.list(
            org_id=org_id,
            agent_id=agent_id,
            memory_type=memory_type,
            tags=tag_list,
            include_expired=include_expired,
            include_shared=include_shared,
            created_after=created_after,
            created_before=created_before,
            importance_gte=importance_gte,
            importance_lte=importance_lte,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size,
        )

    # -------------------------------------------------------------------
    # Get by ID (after fixed paths to avoid capturing /stats/ etc.)
    # -------------------------------------------------------------------

    @router.get("/{memory_id}", response_model=MemoryItem | None)
    async def get_memory(memory_id: uuid.UUID):
        """Get a single memory by ID."""
        item = await store.get(memory_id)
        if not item:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return item

    # -------------------------------------------------------------------
    # Update / Delete
    # -------------------------------------------------------------------

    @router.put("/{memory_id}", response_model=MemoryItem | None)
    async def update_memory(memory_id: uuid.UUID, req: UpdateRequest):
        """Update a memory's content, tags, or importance."""
        try:
            item = await store.update(
                memory_id,
                content=req.content,
                tags=req.tags,
                importance=req.importance,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not item:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return item

    @router.delete("/{memory_id}", response_model=CountResponse)
    async def delete_memory(memory_id: uuid.UUID):
        """Delete a single memory."""
        success = await store.forget(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return CountResponse(count=1)

    @router.post("/bulk-delete", response_model=CountResponse)
    async def bulk_delete(req: BulkDeleteRequest):
        """Delete memories matching filters."""
        try:
            count = await store.bulk_delete(
                org_id=req.org_id,
                agent_id=req.agent_id,
                memory_type=req.memory_type,
                tags=req.tags,
                older_than=req.older_than,
                importance_below=req.importance_below,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return CountResponse(count=count)

    # -------------------------------------------------------------------
    # Temporal
    # -------------------------------------------------------------------

    @router.post("/{memory_id}/supersede", response_model=list[MemoryItem])
    async def supersede_memory(memory_id: uuid.UUID, req: SupersedeRequest):
        """Supersede a memory with a new version. Returns [old, new]."""
        try:
            old, new = await store.supersede(
                memory_id,
                req.new_content,
                org_id=req.org_id,
                agent_id=req.agent_id,
                memory_type=req.memory_type,
                tags=req.tags,
                importance=req.importance,
            )
            return [old, new]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/timeline", response_model=list[MemoryItem])
    async def timeline(req: TimelineRequest):
        """Query what was true at a specific time."""
        return await store.timeline(
            org_id=req.org_id,
            agent_id=req.agent_id,
            at=req.at,
            memory_type=req.memory_type,
            limit=req.limit,
        )

    @router.get("/{memory_id}/chain", response_model=list[MemoryItem])
    async def supersession_chain(memory_id: uuid.UUID):
        """Get the full supersession chain for a memory (oldest → newest)."""
        chain = await store.supersession_chain(memory_id)
        if not chain:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return chain

    # -------------------------------------------------------------------
    # History / Stats / Consolidation
    # -------------------------------------------------------------------

    @router.get("/{memory_id}/history", response_model=list[MemoryHistoryEntry])
    async def memory_history(memory_id: uuid.UUID):
        """Get the full audit trail for a memory."""
        return await store.history(memory_id)

    @router.get("/consolidation/status")
    async def consolidation_status():
        """Get the status of the background consolidation scheduler."""
        scheduler = store._scheduler
        if scheduler is None:
            return {
                "is_running": False,
                "message": "No scheduler attached",
            }
        last_report = scheduler.last_report
        return {
            "is_running": scheduler.is_running,
            "total_cycles": scheduler.total_cycles,
            "last_run": scheduler.last_run,
            "last_report": {
                "agents_processed": last_report.agents_processed,
                "agents_failed": last_report.agents_failed,
                "total_duplicates_merged": last_report.total_duplicates_merged,
                "total_memories_decayed": last_report.total_memories_decayed,
                "total_memories_expired": last_report.total_memories_expired,
                "total_memories_promoted": last_report.total_memories_promoted,
                "errors": last_report.errors,
                "duration_seconds": last_report.duration_seconds,
            }
            if last_report
            else None,
        }

    @router.post("/consolidate", response_model=ConsolidateResponse)
    async def consolidate(req: ConsolidateRequest):
        """Trigger background consolidation (dedup, decay, expire, promote)."""
        report = await store.consolidate(
            org_id=req.org_id,
            agent_id=req.agent_id,
            llm=llm,
            similarity_threshold=req.similarity_threshold,
        )
        return ConsolidateResponse(
            duplicates_merged=report.duplicates_merged,
            memories_decayed=report.memories_decayed,
            memories_expired=report.memories_expired,
            memories_promoted=report.memories_promoted,
            errors=report.errors,
        )

    return router
