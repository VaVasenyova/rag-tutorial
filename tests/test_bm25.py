"""Тесты BM25 (улучшение 5)."""

from app.bm25 import BM25


CORPUS = [
    "ипотека закрытие сделки Citibank ставка",
    "студенческий кредит трудности погашение займа",
    "взыскание долга коллектор медицинский счёт",
]


def test_bm25_returns_scores_for_all_docs():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("ипотека")
    assert len(scores) == len(CORPUS)


def test_bm25_relevant_doc_scores_highest():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("ипотека Citibank ставка")
    assert scores[0] == max(scores)


def test_bm25_unrelated_query_scores_zero():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("как приготовить борщ")
    assert all(s == 0.0 for s in scores)


def test_bm25_empty_query_all_zero():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("")
    assert all(s == 0.0 for s in scores)


def test_bm25_scores_nonnegative():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("кредит долг счёт")
    assert all(s >= 0 for s in scores)
