from __future__ import annotations

"""Module 2: Vietnamese BM25 + Dense Search + Reciprocal Rank Fusion."""

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text and expand underthesea compound-word markers."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        segmented = text
    return segmented.replace("_", " ")


def _tokenize(text: str) -> list[str]:
    return segment_vietnamese(text).split()


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25 index from chunk dictionaries."""
        self.documents = list(chunks)
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in self.documents]
        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _FallbackBM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Return relevant BM25 results in descending score order."""
        if self.bm25 is None or not self.documents or top_k <= 0:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if float(score) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        results = []
        for index, score in ranked_indices:
            document = self.documents[index]
            results.append(
                SearchResult(
                    text=document.get("text", ""),
                    score=score,
                    metadata=dict(document.get("metadata") or {}),
                    method="bm25",
                )
            )
        return results


class _FallbackBM25Okapi:
    """Small BM25Okapi-compatible fallback for environments without rank_bm25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.document_count = len(corpus)
        self.document_lengths = [len(document) for document in corpus]
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )
        document_frequency = Counter(
            token for document in corpus for token in set(document)
        )
        self.inverse_document_frequency = {
            token: math.log(
                1
                + (self.document_count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        query_counts = Counter(query)
        scores = []
        for document, document_length in zip(self.corpus, self.document_lengths):
            term_counts = Counter(document)
            score = 0.0
            length_ratio = document_length / max(self.average_document_length, 1e-9)
            for token, query_frequency in query_counts.items():
                if token not in term_counts:
                    continue
                term_frequency = term_counts[token]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio
                )
                score += (
                    self.inverse_document_frequency.get(token, 0.0)
                    * term_frequency
                    * (self.k1 + 1)
                    / max(denominator, 1e-9)
                    * query_frequency
                )
            scores.append(score)
        return scores


class DenseSearch:
    def __init__(self):
        self.client = None
        self._encoder = None
        self._local_points: dict[str, list[tuple[list[float], dict]]] = {}
        self._qdrant_error: str | None = None

        try:
            from qdrant_client import QdrantClient

            try:
                self.client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    timeout=2,
                )
                self.client.get_collections()
            except Exception:  # noqa: BLE001 - use local Qdrant mode if server is unavailable
                self.client = QdrantClient(":memory:")
        except ImportError:
            # Tests and local development can use the in-process fallback.
            self.client = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    EMBEDDING_MODEL,
                    local_files_only=True,
                )
            except (ImportError, OSError, RuntimeError, ValueError):
                self._encoder = _HashingDenseEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(
        self, chunks: list[dict], collection: str = COLLECTION_NAME
    ) -> None:
        """Encode and index chunks in Qdrant."""
        texts = [chunk.get("text", "") for chunk in chunks]
        if not texts:
            self._local_points[collection] = []
            return

        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        vector_lists = [_to_vector(vector) for vector in vectors]
        payloads = [
            {**(chunk.get("metadata") or {}), "text": chunk.get("text", "")}
            for chunk in chunks
        ]

        if self.client is not None:
            try:
                from qdrant_client.models import Distance, PointStruct, VectorParams

                self.client.recreate_collection(
                    collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                points = [
                    PointStruct(id=index, vector=vector, payload=payload)
                    for index, (vector, payload) in enumerate(
                        zip(vector_lists, payloads)
                    )
                ]
                self.client.upsert(collection, points)
                return
            except Exception as exc:  # noqa: BLE001 - preserve a working local fallback
                self._qdrant_error = str(exc)

        self._local_points[collection] = list(zip(vector_lists, payloads))

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search Qdrant with query_points, or the local fallback if needed."""
        if top_k <= 0:
            return []

        query_vector = _to_vector(self._get_encoder().encode(query))
        if self.client is not None:
            try:
                response = self.client.query_points(
                    collection,
                    query=query_vector,
                    limit=top_k,
                )
                points = getattr(response, "points", response)
                return [
                    SearchResult(
                        text=point.payload.get("text", ""),
                        score=float(point.score),
                        metadata=dict(point.payload or {}),
                        method="dense",
                    )
                    for point in points
                    if point.payload
                ]
            except Exception as exc:  # noqa: BLE001 - local fallback handles unavailable Qdrant
                self._qdrant_error = str(exc)

        points = self._local_points.get(collection, [])
        ranked_points = sorted(
            (
                (cosine_similarity(query_vector, vector), payload)
                for vector, payload in points
            ),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]
        return [
            SearchResult(
                text=payload.get("text", ""),
                score=float(score),
                metadata=dict(payload),
                method="dense",
            )
            for score, payload in ranked_points
        ]


def _to_vector(vector) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if vector and isinstance(vector[0], list):
        vector = vector[0]
    return [float(value) for value in vector]


class _HashingDenseEncoder:
    """Offline fallback that keeps the pipeline runnable without HF/Qdrant."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def encode(self, texts, show_progress_bar: bool = False):
        if isinstance(texts, str):
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE):
            vector[hash(token) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


def cosine_similarity(first: list[float], second: list[float]) -> float:
    numerator = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    return numerator / (first_norm * second_norm + 1e-9)


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked result lists using RRF: sum(1 / (k + rank + 1))."""
    if top_k <= 0:
        return []

    fused: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in fused:
                fused[result.text] = {
                    "score": 0.0,
                    "result": result,
                    "first_seen": len(fused),
                }
            fused[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(
        fused.values(),
        key=lambda item: (-item["score"], item["first_seen"]),
    )[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25, Dense Search, and Reciprocal Rank Fusion."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(
        self, query: str, top_k: int = HYBRID_TOP_K
    ) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion(
            [bm25_results, dense_results],
            top_k=top_k,
        )


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép năm"
    print(f"Original: {query}")
    print(f"Segmented: {segment_vietnamese(query)}")
