"""Demo-ответ: top-k чанки -> текст + источники (без внешней LLM)."""

from app.config import TOP_K
from app.prompts import MIN_SCORE, REFUSAL_EMPTY_QUESTION, REFUSAL_NO_CONTEXT
from app.retriever import Retriever


def build_answer(hits: list[dict], min_score: float = MIN_SCORE) -> str:
    """Формирует ответ только из чанков с score >= min_score."""
    relevant = [h for h in hits if h["score"] >= min_score]
    if not relevant:
        return REFUSAL_NO_CONTEXT

    parts = ["На основании найденных фрагментов:"]
    for i, hit in enumerate(relevant, 1):
        parts.append(f"\n[{i}] {hit['name']}")
        parts.append(f"doc_id={hit['doc_id']}, score={hit['score']:.2f}")
        parts.append(hit["text"])
    return "\n".join(parts)


def format_sources(hits: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": hit["doc_id"],
            "name": hit.get("name", ""),
            "text": hit["text"],
            "score": hit["score"],
            # Улучшение 5: пробрасываем sub-scores если есть (hybrid)
            **({
                "score_tfidf": hit["score_tfidf"],
                "score_bm25": hit["score_bm25"],
            } if "score_tfidf" in hit else {}),
        }
        for hit in hits
    ]


def ask(
    question: str,
    k: int = TOP_K,
    retriever: Retriever | None = None,
    min_score: float = MIN_SCORE,
) -> dict:
    """Вопрос -> ответ и список источников."""
    if not question.strip():
        return {"answer": REFUSAL_EMPTY_QUESTION, "sources": []}

    r = retriever or Retriever()
    hits = r.search(question.strip(), k=k)
    return {
        "answer": build_answer(hits, min_score=min_score),
        "sources": format_sources(hits),
    }
