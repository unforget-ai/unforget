"""Unit tests for embedder — needs sentence-transformers but no database."""

from unforget.embedder import Embedder


class TestEmbedder:
    def test_embed_single(self):
        e = Embedder("all-MiniLM-L6-v2")
        vec = e.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_batch(self):
        e = Embedder("all-MiniLM-L6-v2")
        vecs = e.embed_batch(["hello", "world", "test"])
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_embed_batch_empty(self):
        e = Embedder("all-MiniLM-L6-v2")
        assert e.embed_batch([]) == []

    def test_dims(self):
        e = Embedder("all-MiniLM-L6-v2")
        assert e.dims == 384

    def test_preload(self):
        e = Embedder("all-MiniLM-L6-v2")
        e.preload()
        # Should not raise, model is loaded
        assert e.dims == 384

    def test_similar_texts_closer(self):
        """Semantically similar texts should have higher cosine similarity."""
        e = Embedder("all-MiniLM-L6-v2")
        v1 = e.embed("deploy to production server")
        v2 = e.embed("push code to prod")
        v3 = e.embed("the cat sat on the mat")

        # Cosine similarity (vectors are normalized)
        sim_12 = sum(a * b for a, b in zip(v1, v2))
        sim_13 = sum(a * b for a, b in zip(v1, v3))

        assert sim_12 > sim_13, "Related texts should have higher similarity"

    def test_singleton_reuse(self):
        """Two Embedders with same model name should share the underlying model."""
        e1 = Embedder("all-MiniLM-L6-v2")
        e1.preload()
        e2 = Embedder("all-MiniLM-L6-v2")
        # e2 should reuse the loaded model (singleton)
        assert e2.model is e1.model
