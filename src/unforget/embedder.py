"""Embedding interface + implementations (local ONNX/PyTorch + remote OpenAI).

The BaseEmbedder ABC defines the contract. Implementations:
- Embedder: local inference via ONNX Runtime (default) or PyTorch fallback
- OpenAIEmbedder: remote inference via OpenAI API

Pass any BaseEmbedder to MemoryStore(embedder=...) to swap models.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger("unforget.embedder")


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Interface for embedding models.

    Implement embed() and embed_batch() to plug in any embedding provider.
    The dims property must return the embedding dimensionality.

    Example::

        class MyEmbedder(BaseEmbedder):
            @property
            def dims(self) -> int:
                return 768

            def embed(self, text: str) -> list[float]:
                return my_model.encode(text)

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return my_model.encode_batch(texts)

        store = MemoryStore("postgresql://...", embedder=MyEmbedder())
    """

    @property
    @abstractmethod
    def dims(self) -> int:
        """Return the embedding dimensionality (e.g., 384, 768, 1536)."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns a list of floats."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns a list of embedding vectors."""
        ...

    def preload(self) -> None:
        """Optional: force-load the model at startup."""
        pass


# ---------------------------------------------------------------------------
# Local Embedder (ONNX + PyTorch fallback) — the default
# ---------------------------------------------------------------------------

_ort_session = None
_tokenizer = None
_model_dir: str | None = None

_DEFAULT_ONNX_DIR = str(Path(__file__).resolve().parent.parent.parent / "models" / "embedder-onnx")


def _load_onnx(model_dir: str):
    """Load ONNX session + tokenizer (once)."""
    global _ort_session, _tokenizer, _model_dir
    if _ort_session is not None and _model_dir == model_dir:
        return _ort_session, _tokenizer

    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    logger.info("Loading ONNX embedder from %s", model_dir)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count() or 4

    session = ort.InferenceSession(onnx_path, sess_options)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _ort_session = session
    _tokenizer = tokenizer
    _model_dir = model_dir
    logger.info("ONNX embedder loaded.")
    return _ort_session, _tokenizer


_st_model = None
_st_model_name: str | None = None


def _load_st_model(model_name: str):
    global _st_model, _st_model_name
    if _st_model is not None and _st_model_name == model_name:
        return _st_model
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformers model: %s (ONNX not found, using fallback)", model_name)
    _st_model = SentenceTransformer(model_name)
    _st_model_name = model_name
    return _st_model


class Embedder(BaseEmbedder):
    """Local embedding — ONNX Runtime when available, PyTorch fallback.

    This is the default embedder. Uses all-MiniLM-L6-v2 (384 dims) with an
    LRU embedding cache for repeated content.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        onnx_dir: str | None = None,
    ):
        self._model_name = model_name
        self._onnx_dir = onnx_dir or _DEFAULT_ONNX_DIR
        self._session = None
        self._tokenizer = None
        self._use_onnx: bool | None = None
        self._st_model = None
        from unforget.cache import EmbeddingCache
        self._cache = EmbeddingCache()

    @property
    def model(self):
        """Backward compat: returns the ONNX session or ST model."""
        self._init_backend()
        if self._use_onnx:
            return self._session
        return self._st_model

    def _init_backend(self):
        if self._use_onnx is not None:
            return

        onnx_path = os.path.join(self._onnx_dir, "model.onnx") if self._onnx_dir else ""
        if onnx_path and os.path.exists(onnx_path):
            try:
                self._session, self._tokenizer = _load_onnx(self._onnx_dir)
                self._use_onnx = True
                return
            except Exception as e:
                logger.warning("ONNX load failed, falling back to PyTorch: %s", e)

        self._st_model = _load_st_model(self._model_name)
        self._tokenizer = self._st_model.tokenizer
        self._use_onnx = False

    @property
    def dims(self) -> int:
        self._init_backend()
        if self._use_onnx:
            return 384
        return self._st_model.get_sentence_embedding_dimension()

    def _encode_onnx(self, texts: list[str]) -> NDArray[np.float32]:
        encoded = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            if "token_type_ids" in encoded:
                inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            else:
                inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        outputs = self._session.run(None, inputs)
        token_embeddings = outputs[0]

        mask = encoded["attention_mask"].astype(np.float32)
        mask_expanded = np.expand_dims(mask, -1)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        return (pooled / norms).astype(np.float32)

    def _encode_st(self, texts: list[str]) -> NDArray[np.float32]:
        return self._st_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
        self._init_backend()
        if self._use_onnx:
            return self._encode_onnx(texts)
        return self._encode_st(texts)

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self._encode([text])
        result = vec[0].tolist()
        self._cache.put(text, result)
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cached_results, missed_indices = self._cache.get_batch(texts)
        if not missed_indices:
            return cached_results  # type: ignore[return-value]
        missed_texts = [texts[i] for i in missed_indices]
        vecs = self._encode(missed_texts)
        computed = vecs.tolist()
        for idx, embedding in zip(missed_indices, computed):
            cached_results[idx] = embedding
            self._cache.put(texts[idx], embedding)
        return cached_results  # type: ignore[return-value]

    def preload(self) -> None:
        self._init_backend()


# ---------------------------------------------------------------------------
# OpenAI Embedder (remote)
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """Remote embedding via OpenAI API.

    Uses text-embedding-3-small (1536 dims) by default. Requires openai package.

    Note: every write() and recall() will make an API call (~50-200ms + cost).
    Use the local Embedder for zero-latency writes.

    Example::

        from unforget.embedder import OpenAIEmbedder
        store = MemoryStore("postgresql://...", embedder=OpenAIEmbedder())

        # Or with a specific model:
        store = MemoryStore("postgresql://...", embedder=OpenAIEmbedder(
            model="text-embedding-3-large",  # 3072 dims
            api_key="sk-...",
        ))
    """

    # Dimension lookup for known models
    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions or self._DIMS.get(model, 1536)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def dims(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self._model)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]

    def preload(self) -> None:
        self._get_client()
