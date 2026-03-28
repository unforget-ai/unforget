"""ScopedMemory — bind org_id + agent_id once, then forget about them.

Usage::

    store = MemoryStore("postgresql://...")
    await store.initialize()

    memory = store.bind(org_id="acme", agent_id="support-bot")

    await memory.write("User prefers dark mode")
    results = await memory.recall("user preferences")
    context = await memory.auto_recall("help the user")
    await memory.consolidate()
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unforget.consolidation import ConsolidationReport, LLMCallable
    from unforget.store import MemoryStore
    from unforget.types import (
        MemoryHistoryEntry,
        MemoryItem,
        MemoryResult,
        MemoryStats,
        MemoryType,
        WriteItem,
    )


class ScopedMemory:
    """A MemoryStore bound to a specific org_id + agent_id.

    All methods that normally require org_id and agent_id have them
    pre-filled. Methods that don't need scoping (get, update, forget,
    history, supersession_chain) are passed through directly.
    """

    def __init__(self, store: MemoryStore, org_id: str, agent_id: str):
        self._store = store
        self._org_id = org_id
        self._agent_id = agent_id

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def org_id(self) -> str:
        return self._org_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -- Write --

    async def write(
        self,
        content: str,
        *,
        memory_type: MemoryType | str = "insight",
        tags: list[str] | None = None,
        importance: float = 0.5,
        shared: bool = False,
        immutable: bool = False,
        source_thread_id: str | None = None,
        source_message: str | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryItem:
        return await self._store.write(
            content,
            org_id=self._org_id,
            agent_id=self._agent_id,
            memory_type=memory_type,
            tags=tags,
            importance=importance,
            shared=shared,
            immutable=immutable,
            source_thread_id=source_thread_id,
            source_message=source_message,
            expires_at=expires_at,
        )

    async def write_batch(self, items: list[WriteItem]) -> list[MemoryItem]:
        return await self._store.write_batch(items, org_id=self._org_id, agent_id=self._agent_id)

    # -- Recall --

    async def recall(
        self,
        query: str,
        *,
        limit: int = 10,
        memory_type: MemoryType | str | None = None,
        include_shared: bool = True,
        threshold: float = 0.0,
        rerank: bool = True,
        use_cache: bool = True,
    ) -> list[MemoryResult]:
        return await self._store.recall(
            query,
            org_id=self._org_id,
            agent_id=self._agent_id,
            limit=limit,
            memory_type=memory_type,
            include_shared=include_shared,
            threshold=threshold,
            rerank=rerank,
            use_cache=use_cache,
        )

    async def auto_recall(
        self,
        query: str,
        *,
        max_tokens: int = 2000,
        limit: int = 10,
    ) -> str:
        return await self._store.auto_recall(
            query,
            org_id=self._org_id,
            agent_id=self._agent_id,
            max_tokens=max_tokens,
            limit=limit,
        )

    # -- Ingest --

    async def ingest(
        self,
        messages: list[dict[str, str]],
        *,
        mode: str = "background",
        llm: LLMCallable | None = None,
        source_thread_id: str | None = None,
    ) -> list[MemoryItem]:
        return await self._store.ingest(
            messages,
            org_id=self._org_id,
            agent_id=self._agent_id,
            mode=mode,
            llm=llm,
            source_thread_id=source_thread_id,
        )

    # -- List / Stats --

    async def list(self, **kwargs: Any) -> list[MemoryItem]:
        return await self._store.list(org_id=self._org_id, agent_id=self._agent_id, **kwargs)

    async def stats(self) -> MemoryStats:
        return await self._store.stats(org_id=self._org_id, agent_id=self._agent_id)

    # -- Temporal --

    async def supersede(
        self,
        old_id: Any,
        new_content: str,
        *,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> tuple[MemoryItem, MemoryItem]:
        return await self._store.supersede(
            old_id,
            new_content,
            org_id=self._org_id,
            agent_id=self._agent_id,
            memory_type=memory_type,
            tags=tags,
            importance=importance,
        )

    async def timeline(
        self,
        *,
        at: datetime,
        memory_type: MemoryType | str | None = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        return await self._store.timeline(
            org_id=self._org_id,
            agent_id=self._agent_id,
            at=at,
            memory_type=memory_type,
            limit=limit,
        )

    # -- Delete --

    async def bulk_delete(self, **kwargs: Any) -> int:
        return await self._store.bulk_delete(org_id=self._org_id, agent_id=self._agent_id, **kwargs)

    async def forget_all(self) -> int:
        return await self._store.forget_all(org_id=self._org_id, agent_id=self._agent_id)

    # -- Consolidation --

    async def consolidate(
        self,
        *,
        llm: LLMCallable | None = None,
        similarity_threshold: float | None = None,
    ) -> ConsolidationReport:
        return await self._store.consolidate(
            org_id=self._org_id,
            agent_id=self._agent_id,
            llm=llm,
            similarity_threshold=similarity_threshold,
        )

    # -- Pass-through (no scoping needed) --

    async def get(self, memory_id: Any) -> MemoryItem | None:
        return await self._store.get(memory_id)

    async def update(self, memory_id: Any, **kwargs: Any) -> MemoryItem | None:
        return await self._store.update(memory_id, **kwargs)

    async def forget(self, memory_id: Any) -> bool:
        return await self._store.forget(memory_id)

    async def history(self, memory_id: Any) -> list[MemoryHistoryEntry]:
        return await self._store.history(memory_id)

    async def supersession_chain(self, memory_id: Any) -> list[MemoryItem]:
        return await self._store.supersession_chain(memory_id)
