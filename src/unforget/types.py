"""Core types for unforget."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    """Hierarchical memory types. Insight > Event > Raw in retrieval boosting."""

    INSIGHT = "insight"  # Distilled knowledge, preferences, rules (long-lived)
    EVENT = "event"  # Specific interactions with timestamps (medium-lived)
    RAW = "raw"  # Ingested conversation chunks (short-lived, auto-expires)


class WriteItem(BaseModel):
    """Input for writing a single memory."""

    content: str
    memory_type: MemoryType = MemoryType.INSIGHT
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    shared: bool = False
    immutable: bool = False
    source_thread_id: str | None = None
    source_message: str | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """A stored memory record."""

    id: uuid.UUID
    org_id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    tags: list[str]
    entities: list[str]
    importance: float
    access_count: int
    shared: bool
    immutable: bool
    valid_from: datetime
    valid_to: datetime | None
    superseded_by: uuid.UUID | None
    source_thread_id: str | None
    source_message: str | None
    expires_at: datetime | None
    created_at: datetime
    accessed_at: datetime
    consolidated_at: datetime | None


class MemoryResult(BaseModel):
    """A memory returned from recall(), with relevance score."""

    id: uuid.UUID
    content: str
    memory_type: MemoryType
    tags: list[str]
    entities: list[str]
    importance: float
    score: float
    created_at: datetime
    accessed_at: datetime


class MemoryStats(BaseModel):
    """Aggregate stats for an agent's memory."""

    total: int
    by_type: dict[str, int]
    avg_importance: float
    oldest: datetime | None
    newest: datetime | None


class HistoryAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    PROMOTED = "promoted"


class MemoryHistoryEntry(BaseModel):
    """A single entry in the audit trail."""

    id: uuid.UUID
    memory_id: uuid.UUID
    action: HistoryAction
    old_content: str | None
    new_content: str | None
    changed_at: datetime
    changed_by: str
