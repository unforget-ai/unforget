"""Rate limiting and quota enforcement.

Two quota types:
  - max_memories_per_agent: hard cap on total active memories
  - max_writes_per_minute: sliding window rate limit per org+agent
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """Sliding window rate limiter. Per org_id+agent_id scope.

    Thread-safe. No external dependencies (no Redis).
    """

    def __init__(self, max_per_minute: int = 100):
        self._max = max_per_minute
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, org_id: str, agent_id: str) -> None:
        """Raise RateLimitExceeded if limit is hit."""
        if self._max <= 0:
            return

        key = f"{org_id}:{agent_id}"
        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            # Prune old entries
            timestamps = self._windows[key]
            self._windows[key] = [t for t in timestamps if t > cutoff]

            if len(self._windows[key]) >= self._max:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {agent_id}: "
                    f"{self._max} writes/minute. Try again shortly."
                )

            self._windows[key].append(now)

    def reset(self, org_id: str | None = None, agent_id: str | None = None) -> None:
        """Reset rate limit counters."""
        with self._lock:
            if org_id and agent_id:
                key = f"{org_id}:{agent_id}"
                self._windows.pop(key, None)
            else:
                self._windows.clear()


class RateLimitExceeded(Exception):
    """Raised when write rate limit is hit."""
