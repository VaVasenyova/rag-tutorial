"""Тесты Hybrid Retriever (улучшение 5)."""

import json
import pickle
from pathlib import Path

import pytest
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from app.hybrid_retriever import HybridRetriever


@pytest.fixture
def mini_hybrid(tmp_path: Path) -> HybridRetriever:
    """Мини-индекс для изолированного тестирования HybridRetriever."""
    chunks = [
        {
            "chunk_id": "2_0",
            "doc_id": "2",
            "name": "Ипотека — закрытие ипотечной сделки (Citibank)",
            "text": "Продукт: ипотека. Проблема: закрытие ипотечной сделки. Citibank ставка.",
        },
        {
            "chunk_id": "1_0",
            "doc_id": "1",
            "name": "Студенческий кредит",
            "text": "Продукт: студенческий кредит. Трудности с погашением займа.",
        },
        {
            "chunk_id": "3_0",
            "doc_id": "3",
            "name": "Взыскание долга",
            "text": "Коллекторское агентство взыскивает несуществующий долг.",
        },
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    vp = tmp_path / "vectorizer.pkl"
    mp = tmp_path / "matrix.npz"
    with vp.open("wb") as f:
        pickle.dump(vectorizer, f)
    scipy.sparse.save_npz(mp, matrix)

    return HybridRetriever(
        vectorizer_path=vp,
        matrix_path=mp,
        chunks_path=chunks_path,
    )


def test_hybrid_returns_results(mini_hybrid):
    results = mini_hybrid.search("ипотека Citibank", k=2)
    assert len(results) >= 1


def test_hybrid_results_have_required_fields(mini_hybrid):
    results = mini_hybrid.search("кредит", k=2)
    for r in results:
        assert "doc_id" in r
        assert "text" in r
        assert "score" in r
        assert "score_tfidf" in r
        assert "score_bm25" in r


def test_hybrid_sorted_by_rrf_score(mini_hybrid):
    results = mini_hybrid.search("ипотека закрытие", k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_mortgage_ranks_first(mini_hybrid):
    results = mini_hybrid.search("ипотека Citibank ставка", k=3)
    assert results[0]["doc_id"] == "2"


def test_hybrid_empty_query_returns_empty(mini_hybrid):
    assert mini_hybrid.search("") == []
    assert mini_hybrid.search("   ") == []
