"""LRU + TTL cache for recall results and embeddings."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors keyed by content hash."""

    def __init__(self, maxsize: int = 4096):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = Lock()

    def get(self, text: str) -> list[float] | None:
        key = hashlib.sha256(text.encode()).hexdigest()
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, text: str, embedding: list[float]) -> None:
        key = hashlib.sha256(text.encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = embedding
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def get_batch(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        results: list[list[float] | None] = []
        missed = []
        for i, t in enumerate(texts):
            cached = self.get(t)
            results.append(cached)
            if cached is None:
                missed.append(i)
        return results, missed

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL expiration.

    Used by recall() to avoid repeated DB queries for the same agent+query
    within a short window (e.g., auto-recall at thread start).
    """

    def __init__(self, maxsize: int = 1000, ttl: float = 60.0):
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        """Get a cached value. Returns None if missing or expired."""
        with self._lock:
            if key not in self._cache:
                return None
            ts, value = self._cache[key]
            if time.monotonic() - ts > self._ttl:
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with current timestamp."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.monotonic(), value)
            # Evict oldest if over capacity
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def make_key(query: str, org_id: str, agent_id: str, **kwargs) -> str:
        """Generate a deterministic cache key from recall parameters."""
        parts = f"{query}|{org_id}|{agent_id}"
        for k in sorted(kwargs):
            v = kwargs[k]
            if v is not None:
                parts += f"|{k}={v}"
        return hashlib.md5(parts.encode()).hexdigest()
