"""Unit tests for TTL cache — no database needed."""

import time

from unforget.cache import TTLCache


class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(maxsize=10, ttl=60)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing(self):
        cache = TTLCache()
        assert cache.get("nope") is None

    def test_expiry(self):
        cache = TTLCache(maxsize=10, ttl=0.1)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"
        time.sleep(0.15)
        assert cache.get("k1") is None

    def test_lru_eviction(self):
        cache = TTLCache(maxsize=2, ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_access_refreshes(self):
        cache = TTLCache(maxsize=2, ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # refresh "a" — now "b" is oldest
        cache.set("c", 3)  # evicts "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_invalidate(self):
        cache = TTLCache()
        cache.set("k1", "v1")
        cache.invalidate("k1")
        assert cache.get("k1") is None

    def test_clear(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0

    def test_size(self):
        cache = TTLCache()
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1

    def test_make_key_deterministic(self):
        k1 = TTLCache.make_key("query", "org", "agent", limit=10, rerank=True)
        k2 = TTLCache.make_key("query", "org", "agent", limit=10, rerank=True)
        assert k1 == k2

    def test_make_key_different_params(self):
        k1 = TTLCache.make_key("query", "org", "agent", limit=10)
        k2 = TTLCache.make_key("query", "org", "agent", limit=20)
        assert k1 != k2

    def test_make_key_ignores_none(self):
        k1 = TTLCache.make_key("q", "o", "a", memory_type=None)
        k2 = TTLCache.make_key("q", "o", "a")
        assert k1 == k2
