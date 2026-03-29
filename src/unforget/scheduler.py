"""Async consolidation scheduler — runs consolidation periodically in the background.

Two trigger modes (both can be active simultaneously):
1. Interval — run every N seconds (e.g., every 60 minutes)
2. Write-threshold — run after N writes since last consolidation (e.g., every 50 writes)

Auto-discovers active agents by querying distinct (org_id, agent_id) pairs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from unforget.consolidation import ConsolidationReport, LLMCallable

logger = logging.getLogger("unforget.scheduler")

# Avoid circular import — MemoryStore is only used for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unforget.store import MemoryStore


@dataclass
class CycleReport:
    """Summary of a single consolidation cycle across all agents."""

    agents_processed: int = 0
    agents_failed: int = 0
    total_duplicates_merged: int = 0
    total_memories_decayed: int = 0
    total_memories_expired: int = 0
    total_memories_promoted: int = 0
    total_associations_linked: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class ConsolidationScheduler:
    """Background scheduler that periodically consolidates all active agents.

    Usage::

        scheduler = ConsolidationScheduler(store, interval_seconds=3600)
        await scheduler.start()
        # ... scheduler runs in background ...
        await scheduler.stop()
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        interval_seconds: float = 3600,
        write_threshold: int | None = 50,
        llm: LLMCallable | None = None,
        similarity_threshold: float = 0.92,
    ):
        self._store = store
        self._interval_seconds = interval_seconds
        self._write_threshold = write_threshold
        self._llm = llm
        self._similarity_threshold = similarity_threshold

        self._task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()
        self._write_count: int = 0
        self._consecutive_failures: int = 0

        # Status tracking
        self._total_cycles: int = 0
        self._last_run: float | None = None
        self._last_report: CycleReport | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def total_cycles(self) -> int:
        return self._total_cycles

    @property
    def last_run(self) -> float | None:
        return self._last_run

    @property
    def last_report(self) -> CycleReport | None:
        return self._last_report

    async def start(self) -> None:
        """Start the background consolidation loop."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        self._wake_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="unforget-consolidation-scheduler")
        logger.info(
            "Consolidation scheduler started (interval=%ds, write_threshold=%s)",
            self._interval_seconds,
            self._write_threshold,
        )

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("Consolidation scheduler stopped")

    def notify_write(self) -> None:
        """Called by MemoryStore.write() to bump the write counter."""
        self._write_count += 1
        if self._write_threshold is not None and self._write_count >= self._write_threshold:
            self._wake_event.set()

    async def _run_loop(self) -> None:
        """Main loop: sleep until interval OR write threshold hit, then consolidate."""
        while True:
            sleep_time = self._get_sleep_time()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=sleep_time)
            except TimeoutError:
                pass  # Interval elapsed — time to consolidate

            self._wake_event.clear()

            try:
                report = await self._consolidate_all()
                self._total_cycles += 1
                self._last_run = time.time()
                self._last_report = report
                self._write_count = 0
                self._consecutive_failures = 0

                logger.info(
                    "Consolidation cycle #%d complete: %d agents (%d failed), "
                    "merged=%d decayed=%d expired=%d promoted=%d (%.1fs)",
                    self._total_cycles,
                    report.agents_processed,
                    report.agents_failed,
                    report.total_duplicates_merged,
                    report.total_memories_decayed,
                    report.total_memories_expired,
                    report.total_memories_promoted,
                    report.duration_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self._consecutive_failures += 1
                logger.exception(
                    "Consolidation cycle failed (consecutive_failures=%d)",
                    self._consecutive_failures,
                )

    def _get_sleep_time(self) -> float:
        """Calculate sleep time with exponential backoff on consecutive failures."""
        if self._consecutive_failures == 0:
            return self._interval_seconds
        backoff = self._interval_seconds * (2 ** self._consecutive_failures)
        return min(backoff, 3600)  # Cap at 1 hour

    async def _consolidate_all(self) -> CycleReport:
        """Discover active agents, consolidate each, log results."""
        start = time.monotonic()
        agents = await self._discover_agents()

        report = CycleReport()

        for org_id, agent_id in agents:
            try:
                agent_report: ConsolidationReport = await self._store.consolidate(
                    org_id=org_id,
                    agent_id=agent_id,
                    llm=self._llm,
                    similarity_threshold=self._similarity_threshold,
                )
                report.agents_processed += 1
                report.total_duplicates_merged += agent_report.duplicates_merged
                report.total_memories_decayed += agent_report.memories_decayed
                report.total_memories_expired += agent_report.memories_expired
                report.total_memories_promoted += agent_report.memories_promoted
                report.total_associations_linked += agent_report.associations_linked
                if agent_report.errors:
                    report.errors.extend(
                        f"{org_id}/{agent_id}: {e}" for e in agent_report.errors
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                report.agents_failed += 1
                report.errors.append(f"{org_id}/{agent_id}: {e}")
                logger.warning(
                    "Consolidation failed for %s/%s: %s", org_id, agent_id, e
                )

        report.duration_seconds = time.monotonic() - start
        return report

    async def _discover_agents(self) -> list[tuple[str, str]]:
        """Query DB for distinct (org_id, agent_id) pairs with active memories."""
        rows = await self._store.pool.fetch(
            "SELECT DISTINCT org_id, agent_id FROM memory WHERE valid_to IS NULL"
        )
        return [(r["org_id"], r["agent_id"]) for r in rows]
