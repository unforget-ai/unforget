"""Unit tests for types — no database needed."""

import uuid
from datetime import UTC, datetime

from unforget.types import (
    HistoryAction,
    MemoryHistoryEntry,
    MemoryResult,
    MemoryStats,
    MemoryType,
    WriteItem,
)


class TestMemoryType:
    def test_values(self):
        assert MemoryType.INSIGHT == "insight"
        assert MemoryType.EVENT == "event"
        assert MemoryType.RAW == "raw"

    def test_from_string(self):
        assert MemoryType("insight") == MemoryType.INSIGHT
        assert MemoryType("event") == MemoryType.EVENT
        assert MemoryType("raw") == MemoryType.RAW


class TestWriteItem:
    def test_defaults(self):
        item = WriteItem(content="test fact")
        assert item.content == "test fact"
        assert item.memory_type == MemoryType.INSIGHT
        assert item.tags == []
        assert item.importance == 0.5
        assert item.shared is False
        assert item.immutable is False
        assert item.expires_at is None

    def test_custom_values(self):
        item = WriteItem(
            content="deploy to fly",
            memory_type=MemoryType.EVENT,
            tags=["infra"],
            importance=0.9,
            shared=True,
        )
        assert item.memory_type == MemoryType.EVENT
        assert item.tags == ["infra"]
        assert item.importance == 0.9
        assert item.shared is True

    def test_importance_bounds(self):
        import pytest

        with pytest.raises(Exception):
            WriteItem(content="x", importance=1.5)
        with pytest.raises(Exception):
            WriteItem(content="x", importance=-0.1)


class TestMemoryResult:
    def test_construction(self):
        now = datetime.now(UTC)
        r = MemoryResult(
            id=uuid.uuid4(),
            content="test",
            memory_type=MemoryType.INSIGHT,
            tags=["a"],
            entities=["Fly.io"],
            importance=0.8,
            score=0.95,
            created_at=now,
            accessed_at=now,
        )
        assert r.score == 0.95
        assert r.memory_type == MemoryType.INSIGHT


class TestMemoryStats:
    def test_construction(self):
        s = MemoryStats(
            total=100,
            by_type={"insight": 50, "event": 30, "raw": 20},
            avg_importance=0.6,
            oldest=datetime(2026, 1, 1, tzinfo=UTC),
            newest=datetime(2026, 3, 23, tzinfo=UTC),
        )
        assert s.total == 100
        assert s.by_type["insight"] == 50


class TestHistoryEntry:
    def test_construction(self):
        e = MemoryHistoryEntry(
            id=uuid.uuid4(),
            memory_id=uuid.uuid4(),
            action=HistoryAction.CREATED,
            old_content=None,
            new_content="new fact",
            changed_at=datetime.now(UTC),
            changed_by="agent",
        )
        assert e.action == HistoryAction.CREATED
        assert e.old_content is None
