"""Unit tests for rate limiter — no database needed."""


import pytest

from unforget.quotas import RateLimiter, RateLimitExceeded


class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = RateLimiter(max_per_minute=10)
        # Should not raise
        for _ in range(10):
            limiter.check("org", "agent")

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_per_minute=3)
        limiter.check("org", "agent")
        limiter.check("org", "agent")
        limiter.check("org", "agent")
        with pytest.raises(RateLimitExceeded):
            limiter.check("org", "agent")

    def test_per_agent_isolation(self):
        limiter = RateLimiter(max_per_minute=2)
        limiter.check("org", "agent-a")
        limiter.check("org", "agent-a")
        # agent-b has its own counter
        limiter.check("org", "agent-b")
        limiter.check("org", "agent-b")
        # agent-a is blocked, agent-b is blocked
        with pytest.raises(RateLimitExceeded):
            limiter.check("org", "agent-a")
        with pytest.raises(RateLimitExceeded):
            limiter.check("org", "agent-b")

    def test_window_expires(self):
        limiter = RateLimiter(max_per_minute=2)
        limiter.check("org", "agent")
        limiter.check("org", "agent")
        # Manually expire the window by clearing
        limiter.reset("org", "agent")
        # Should work again
        limiter.check("org", "agent")

    def test_reset_all(self):
        limiter = RateLimiter(max_per_minute=1)
        limiter.check("org", "a")
        limiter.check("org", "b")
        limiter.reset()
        # Both should work again
        limiter.check("org", "a")
        limiter.check("org", "b")

    def test_zero_limit_allows_all(self):
        limiter = RateLimiter(max_per_minute=0)
        # Should not raise — 0 means disabled
        for _ in range(100):
            limiter.check("org", "agent")

    def test_error_message(self):
        limiter = RateLimiter(max_per_minute=1)
        limiter.check("org", "bot")
        with pytest.raises(RateLimitExceeded, match="bot"):
            limiter.check("org", "bot")
