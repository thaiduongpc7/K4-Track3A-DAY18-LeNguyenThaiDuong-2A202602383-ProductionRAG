from __future__ import annotations

"""Module 3: Cross-Encoder reranking and latency benchmark."""

import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K

_MODEL_CACHE: dict[str, object] = {}
_UNAVAILABLE_MODELS: set[str] = set()


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load the configured sentence-transformers CrossEncoder once."""
        if self._model is not None:
            return self._model
        if self.model_name in _UNAVAILABLE_MODELS:
            return None
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                local_files_only=True,
            )
            _MODEL_CACHE[self.model_name] = self._model
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            _UNAVAILABLE_MODELS.add(self.model_name)
            self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Rerank candidate documents and return the highest-scoring results."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        if model is None:
            scores = [
                _fallback_relevance_score(query, document.get("text", ""))
                for document in documents
            ]
        else:
            pairs = [(query, document.get("text", "")) for document in documents]
            try:
                scores = _normalise_scores(model.predict(pairs))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                scores = [
                    _fallback_relevance_score(query, document.get("text", ""))
                    for document in documents
                ]

        scored = sorted(
            (
                (float(score), index, document)
                for index, (score, document) in enumerate(zip(scores, documents))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata") or {}),
                rank=rank,
            )
            for rank, (score, _, document) in enumerate(scored[:top_k])
        ]


def _normalise_scores(scores) -> list[float]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        scores = [scores]
    return [float(score) for score in scores]


def _fallback_relevance_score(query: str, document: str) -> float:
    """Deterministic lexical fallback for offline or unavailable model runs."""
    query_tokens = Counter(_tokens(query))
    document_tokens = Counter(_tokens(document))
    if not query_tokens or not document_tokens:
        return 0.0

    overlap = sum(
        min(query_frequency, document_tokens[token])
        for token, query_frequency in query_tokens.items()
    )
    weighted_overlap = sum(
        min(query_frequency, document_tokens[token])
        * (2.0 if token.isdigit() else 1.0)
        for token, query_frequency in query_tokens.items()
    )
    query_terms = set(query_tokens)
    document_terms = set(document_tokens)
    phrase_bonus = 1.0 if query_terms and query_terms <= document_terms else 0.0
    return weighted_overlap / max(len(query_tokens), 1) + overlap * 0.01 + phrase_bonus


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class FlashrankReranker:
    """Lightweight optional ONNX-based alternative."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker

            self._model = Ranker()
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []

        from flashrank import RerankRequest

        passages = [
            {"id": index, "text": document.get("text", "")}
            for index, document in enumerate(documents)
        ]
        results = self._load_model().rerank(
            RerankRequest(query=query, passages=passages)
        )
        return [
            RerankResult(
                text=documents[int(result["id"])].get("text", ""),
                original_score=float(documents[int(result["id"])].get("score", 0.0)),
                rerank_score=float(result["score"]),
                metadata=dict(documents[int(result["id"])].get("metadata") or {}),
                rank=rank,
            )
            for rank, result in enumerate(results[:top_k])
        ]


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark reranking latency over n_runs."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
