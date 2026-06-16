"""Тесты BM25 (улучшение 5)."""

from app.bm25 import BM25


CORPUS = [
    "Tyrannosaurus carnivore large theropod Cretaceous USA predator",
    "Triceratops herbivore ceratopsid three horns Late Cretaceous",
    "Brachiosaurus herbivore sauropod long neck Jurassic Tanzania",
]


def test_bm25_returns_scores_for_all_docs():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("carnivore")
    assert len(scores) == len(CORPUS)


def test_bm25_relevant_doc_scores_highest():
    bm25 = BM25(CORPUS)
    scores = bm25.scores("carnivore theropod predator")
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
    scores = bm25.scores("herbivore sauropod neck")
    assert all(s >= 0 for s in scores)
