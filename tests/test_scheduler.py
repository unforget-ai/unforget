"""Tests for ConsolidationScheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from unforget.consolidation import ConsolidationReport
from unforget.scheduler import ConsolidationScheduler


@pytest.fixture
def mock_store():
    """Create a mock MemoryStore with a mock pool."""
    store = MagicMock()
    store.pool = MagicMock()
    store.pool.fetch = AsyncMock(return_value=[])
    store.consolidate = AsyncMock(return_value=ConsolidationReport())
    store._scheduler = None
    return store


@pytest.mark.asyncio
async def test_start_stop(mock_store):
    """Scheduler starts, reports running, then stops cleanly."""
    scheduler = ConsolidationScheduler(mock_store, interval_seconds=9999)
    assert not scheduler.is_running

    await scheduler.start()
    assert scheduler.is_running

    await scheduler.stop()
    assert not scheduler.is_running


@pytest.mark.asyncio
async def test_start_idempotent(mock_store):
    """Starting twice doesn't create duplicate tasks."""
    scheduler = ConsolidationScheduler(mock_store, interval_seconds=9999)
    await scheduler.start()
    task1 = scheduler._task
    await scheduler.start()  # should warn, not create new task
    assert scheduler._task is task1
    await scheduler.stop()


@pytest.mark.asyncio
async def test_interval_triggers_consolidation(mock_store):
    """After interval elapses, consolidation runs."""
    mock_store.pool.fetch = AsyncMock(
        return_value=[{"org_id": "org1", "agent_id": "agent1"}]
    )
    mock_store.consolidate = AsyncMock(return_value=ConsolidationReport())

    scheduler = ConsolidationScheduler(mock_store, interval_seconds=0.05)
    await scheduler.start()

    # Wait for at least one cycle
    await asyncio.sleep(0.15)

    await scheduler.stop()

    assert scheduler.total_cycles >= 1
    assert mock_store.consolidate.called


@pytest.mark.asyncio
async def test_write_threshold_triggers(mock_store):
    """After N notify_write calls, consolidation fires early."""
    mock_store.pool.fetch = AsyncMock(
        return_value=[{"org_id": "org1", "agent_id": "agent1"}]
    )
    mock_store.consolidate = AsyncMock(return_value=ConsolidationReport())

    scheduler = ConsolidationScheduler(
        mock_store, interval_seconds=9999, write_threshold=5
    )
    await scheduler.start()

    # Fire 5 writes to hit threshold
    for _ in range(5):
        scheduler.notify_write()

    # Give the loop time to wake and run
    await asyncio.sleep(0.1)

    await scheduler.stop()

    assert scheduler.total_cycles >= 1
    assert mock_store.consolidate.called


@pytest.mark.asyncio
async def test_error_in_agent_doesnt_crash_loop(mock_store):
    """One agent failing doesn't crash the scheduler."""
    mock_store.pool.fetch = AsyncMock(
        return_value=[
            {"org_id": "org1", "agent_id": "agent1"},
            {"org_id": "org2", "agent_id": "agent2"},
        ]
    )

    call_count = 0

    async def consolidate_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("org_id") == "org1":
            raise RuntimeError("DB error for agent1")
        return ConsolidationReport(duplicates_merged=1)

    mock_store.consolidate = AsyncMock(side_effect=consolidate_side_effect)

    scheduler = ConsolidationScheduler(mock_store, interval_seconds=0.05)
    await scheduler.start()

    await asyncio.sleep(0.15)
    await scheduler.stop()

    assert scheduler.total_cycles >= 1
    report = scheduler.last_report
    assert report is not None
    assert report.agents_failed >= 1
    assert report.agents_processed >= 1  # agent2 should have succeeded


@pytest.mark.asyncio
async def test_discover_agents(mock_store):
    """Discovers distinct org/agent pairs from DB."""
    mock_store.pool.fetch = AsyncMock(
        return_value=[
            {"org_id": "org1", "agent_id": "a1"},
            {"org_id": "org1", "agent_id": "a2"},
            {"org_id": "org2", "agent_id": "a1"},
        ]
    )

    scheduler = ConsolidationScheduler(mock_store)
    agents = await scheduler._discover_agents()
    assert agents == [("org1", "a1"), ("org1", "a2"), ("org2", "a1")]


@pytest.mark.asyncio
async def test_stop_when_not_running(mock_store):
    """Stopping a scheduler that never started is a no-op."""
    scheduler = ConsolidationScheduler(mock_store)
    await scheduler.stop()  # Should not raise
    assert not scheduler.is_running


@pytest.mark.asyncio
async def test_backoff_on_full_failure(mock_store):
    """Consecutive failures increase the sleep time."""
    scheduler = ConsolidationScheduler(mock_store, interval_seconds=10)

    # No failures — normal interval
    scheduler._consecutive_failures = 0
    assert scheduler._get_sleep_time() == 10

    # 1 failure — 20s
    scheduler._consecutive_failures = 1
    assert scheduler._get_sleep_time() == 20

    # 2 failures — 40s
    scheduler._consecutive_failures = 2
    assert scheduler._get_sleep_time() == 40

    # Many failures — caps at 3600
    scheduler._consecutive_failures = 20
    assert scheduler._get_sleep_time() == 3600


@pytest.mark.asyncio
async def test_last_report_populated(mock_store):
    """After a cycle, last_report contains aggregated data."""
    mock_store.pool.fetch = AsyncMock(
        return_value=[{"org_id": "org1", "agent_id": "a1"}]
    )
    mock_store.consolidate = AsyncMock(
        return_value=ConsolidationReport(
            duplicates_merged=3, memories_decayed=2, memories_expired=1
        )
    )

    scheduler = ConsolidationScheduler(mock_store, interval_seconds=0.05)
    await scheduler.start()
    await asyncio.sleep(0.15)
    await scheduler.stop()

    report = scheduler.last_report
    assert report is not None
    assert report.total_duplicates_merged == 3
    assert report.total_memories_decayed == 2
    assert report.total_memories_expired == 1
    assert report.agents_processed == 1
    assert report.duration_seconds > 0


@pytest.mark.asyncio
async def test_write_count_resets_after_cycle(mock_store):
    """Write counter resets to 0 after a consolidation cycle."""
    mock_store.pool.fetch = AsyncMock(return_value=[])
    mock_store.consolidate = AsyncMock(return_value=ConsolidationReport())

    scheduler = ConsolidationScheduler(
        mock_store, interval_seconds=9999, write_threshold=3
    )
    await scheduler.start()

    # Hit threshold
    for _ in range(3):
        scheduler.notify_write()

    await asyncio.sleep(0.1)
    assert scheduler._write_count == 0  # Reset after cycle

    await scheduler.stop()
