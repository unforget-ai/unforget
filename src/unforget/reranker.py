"""Cross-encoder reranker — ONNX Runtime for fast inference.

The single most important accuracy component: +30.7pp in ablation studies.
Uses ONNX when available (2-3x faster), falls back to sentence-transformers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unforget.types import MemoryResult

logger = logging.getLogger("unforget.reranker")

_ort_session = None
_tokenizer = None
_model_dir: str | None = None

# Disable ONNX for reranker by default — PyTorch is faster on Apple Silicon
# for small batch cross-encoder inference. Set env UNFORGET_RERANKER_ONNX=1 to enable.
_DEFAULT_ONNX_DIR = str(Path(__file__).resolve().parent.parent.parent / "models" / "reranker-onnx") if os.environ.get("UNFORGET_RERANKER_ONNX") == "1" else ""

# Fallback
_st_model = None
_st_model_name: str | None = None


def _load_onnx(model_dir: str):
    global _ort_session, _tokenizer, _model_dir
    if _ort_session is not None and _model_dir == model_dir:
        return _ort_session, _tokenizer

    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX reranker not found at {onnx_path}")

    logger.info("Loading ONNX reranker from %s", model_dir)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count() or 4

    _ort_session = ort.InferenceSession(onnx_path, sess_options)
    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model_dir = model_dir
    logger.info("ONNX reranker loaded.")
    return _ort_session, _tokenizer


def _load_st_model(model_name: str):
    global _st_model, _st_model_name
    if _st_model is not None and _st_model_name == model_name:
        return _st_model
    from sentence_transformers import CrossEncoder
    logger.info("Loading CrossEncoder: %s (ONNX not found, using fallback)", model_name)
    _st_model = CrossEncoder(model_name)
    _st_model_name = model_name
    return _st_model


class Reranker:
    """Cross-encoder reranker — ONNX Runtime when available, PyTorch fallback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        onnx_dir: str | None = None,
    ):
        self._model_name = model_name
        self._onnx_dir = onnx_dir or _DEFAULT_ONNX_DIR
        self._session = None
        self._tokenizer = None
        self._use_onnx: bool | None = None
        self._st_model = None

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

        onnx_path = os.path.join(self._onnx_dir, "model.onnx")
        if os.path.exists(onnx_path):
            try:
                self._session, self._tokenizer = _load_onnx(self._onnx_dir)
                self._use_onnx = True
                return
            except Exception as e:
                logger.warning("ONNX reranker load failed, falling back: %s", e)

        self._st_model = _load_st_model(self._model_name)
        self._use_onnx = False

    def preload(self) -> None:
        """Force-load the model at startup."""
        self._init_backend()

    def rerank(
        self,
        query: str,
        results: list[MemoryResult],
        top_k: int | None = None,
    ) -> list[MemoryResult]:
        """Rerank results using cross-encoder scores."""
        if not results:
            return []

        import time as _time
        _t0 = _time.perf_counter()

        self._init_backend()

        if self._use_onnx:
            scores = self._predict_onnx(query, results)
        else:
            scores = self._predict_st(query, results)

        _dur = (_time.perf_counter() - _t0) * 1000
        logger.debug("[rerank] %d pairs in %.1fms (%.1fms/pair)", len(results), _dur, _dur / len(results) if results else 0)

        scored = []
        for result, score in zip(results, scores):
            updated = result.model_copy(update={"score": float(score)})
            scored.append(updated)

        scored.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def _predict_onnx(self, query: str, results: list[MemoryResult]) -> list[float]:
        """Score pairs using ONNX Runtime."""
        pairs = [(query, r.content) for r in results]
        encoded = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
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
        logits = outputs[0]  # (batch, 1) or (batch,)
        if logits.ndim == 2:
            logits = logits[:, 0]
        return logits.tolist()

    def _predict_st(self, query: str, results: list[MemoryResult]) -> list[float]:
        """Score pairs using sentence-transformers (fallback)."""
        pairs = [(query, r.content) for r in results]
        scores = self._st_model.predict(
            pairs,
            batch_size=len(pairs),
            show_progress_bar=False,
        )
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
