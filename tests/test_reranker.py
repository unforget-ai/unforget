"""Unit tests for cross-encoder reranker — needs sentence-transformers, no database."""

import uuid
from datetime import UTC, datetime

from unforget.reranker import Reranker
from unforget.types import MemoryResult, MemoryType


def _make_result(content: str, score: float = 0.5) -> MemoryResult:
    now = datetime.now(UTC)
    return MemoryResult(
        id=uuid.uuid4(),
        content=content,
        memory_type=MemoryType.INSIGHT,
        tags=[],
        entities=[],
        importance=0.5,
        score=score,
        created_at=now,
        accessed_at=now,
    )


class TestReranker:
    def test_rerank_reorders(self):
        reranker = Reranker()
        results = [
            _make_result("The cat sat on the mat"),
            _make_result("Deploy the API to Fly.io production server"),
            _make_result("User prefers deploying to Fly.io"),
        ]

        reranked = reranker.rerank("deployment target", results)

        # The deployment-related results should rank higher than the cat
        assert "cat" not in reranked[0].content.lower()

    def test_rerank_top_k(self):
        reranker = Reranker()
        results = [_make_result(f"memory {i}") for i in range(10)]

        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    def test_rerank_empty(self):
        reranker = Reranker()
        assert reranker.rerank("query", []) == []

    def test_rerank_updates_scores(self):
        reranker = Reranker()
        results = [_make_result("some content", score=0.99)]

        reranked = reranker.rerank("query", results)
        # Score should be updated by cross-encoder (not the original 0.99)
        assert reranked[0].score != 0.99

    def test_rerank_preserves_ids(self):
        reranker = Reranker()
        original_ids = set()
        results = []
        for i in range(5):
            r = _make_result(f"content {i}")
            original_ids.add(r.id)
            results.append(r)

        reranked = reranker.rerank("query", results)
        reranked_ids = {r.id for r in reranked}
        assert reranked_ids == original_ids

    def test_preload(self):
        reranker = Reranker()
        reranker.preload()
        # Should not raise
        assert reranker.model is not None
