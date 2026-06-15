"""
Улучшение 5 — Hybrid Retriever: TF-IDF + BM25, объединены через RRF.

Reciprocal Rank Fusion (RRF) — простой и надёжный способ слить два ранжирования:
    rrf_score(doc) = 1/(k + rank_tfidf) + 1/(k + rank_bm25)

Почему это лучше одного retriever:
- TF-IDF хорош на точных совпадениях слов
- BM25 нормирует длину документа и не переоценивает частые слова
- RRF сглаживает ошибки каждого в отдельности
"""

import pickle
from pathlib import Path

import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity

from app.bm25 import BM25
from app.chunker import load_documents
from app.config import (
    INDEX_CHUNKS_JSONL,
    MATRIX_NPZ,
    TOP_K,
    VECTORIZER_PKL,
)


class HybridRetriever:
    """
    Объединяет TF-IDF cosine и BM25 через Reciprocal Rank Fusion.

    Параметры:
        rrf_k: константа RRF (стандарт 60), снижает влияние топ-1 позиций.
        w_tfidf, w_bm25: веса каждого ранжирования в финальном score.
    """

    def __init__(
        self,
        vectorizer_path: Path = VECTORIZER_PKL,
        matrix_path: Path = MATRIX_NPZ,
        chunks_path: Path = INDEX_CHUNKS_JSONL,
        rrf_k: int = 60,
        w_tfidf: float = 0.5,
        w_bm25: float = 0.5,
    ) -> None:
        # TF-IDF
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Индекс не найден: {vectorizer_path}. "
                "Запустите: uv run python scripts/build_index.py"
            )
        with vectorizer_path.open("rb") as f:
            self.vectorizer = pickle.load(f)
        self.matrix = scipy.sparse.load_npz(matrix_path)

        # Чанки
        self.chunks = load_documents(chunks_path)

        # BM25 — строится из тех же текстов, что и TF-IDF матрица
        texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25(texts)

        self.rrf_k = rrf_k
        self.w_tfidf = w_tfidf
        self.w_bm25 = w_bm25

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """Hybrid поиск: RRF(TF-IDF, BM25) → top-k чанков."""
        if not query.strip():
            return []

        n = len(self.chunks)
        k_ret = min(k, n)

        # ── TF-IDF ранжирование ───────────────────────────────────────────────
        qvec = self.vectorizer.transform([query.strip()])
        tfidf_scores = cosine_similarity(qvec, self.matrix).flatten()
        tfidf_rank = {
            idx: rank
            for rank, idx in enumerate(tfidf_scores.argsort()[::-1], start=1)
        }

        # ── BM25 ранжирование ─────────────────────────────────────────────────
        bm25_scores_raw = self.bm25.scores(query)
        bm25_rank = {
            idx: rank
            for rank, idx in enumerate(
                sorted(range(n), key=lambda i: bm25_scores_raw[i], reverse=True),
                start=1,
            )
        }

        # ── RRF fusion ────────────────────────────────────────────────────────
        rrf: dict[int, float] = {}
        for idx in range(n):
            r_tfidf = tfidf_rank.get(idx, n)
            r_bm25 = bm25_rank.get(idx, n)
            rrf[idx] = (
                self.w_tfidf / (self.rrf_k + r_tfidf)
                + self.w_bm25 / (self.rrf_k + r_bm25)
            )

        top_indices = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k_ret]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "name": chunk["name"],
                    "score": round(rrf[idx], 6),
                    "score_tfidf": round(float(tfidf_scores[idx]), 4),
                    "score_bm25": round(bm25_scores_raw[idx], 4),
                }
            )
        return results
