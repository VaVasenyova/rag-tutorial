"""Streamlit UI: вопрос -> фрагменты -> ответ -> источники.

Улучшение 3: фильтр по порогу score в UI, история запросов, подсветка слов.
Улучшение 5: переключатель между TF-IDF и Hybrid (TF-IDF + BM25 через RRF).
"""

import re

import streamlit as st

from app.config import INDEX_CHUNKS_JSONL, MATRIX_NPZ, TOP_K, VECTORIZER_PKL
from app.generator import ask
from app.hybrid_retriever import HybridRetriever
from app.prompts import MIN_SCORE
from app.retriever import Retriever

DEMO_QUESTIONS = [
    "Ипотека - закрытие ипотечной сделки",
    "Какие переменные в датасете про безработицу?",
    "За какой период данные об инфляции?",
    "Как приготовить борщ?",
    "Capital One просроченный счёт",
    "Wells Fargo закрытие счёта",
]


def index_exists() -> bool:
    return all(p.exists() for p in (VECTORIZER_PKL, MATRIX_NPZ, INDEX_CHUNKS_JSONL))


@st.cache_resource
def load_retriever(mode: str) -> Retriever | HybridRetriever:
    """Загружаем retriever один раз и кешируем."""
    if mode == "hybrid":
        return HybridRetriever()
    return Retriever()


# ── Улучшение 3: подсветка совпавших слов ────────────────────────────────────
def highlight(text: str, query: str) -> str:
    """Оборачивает совпавшие слова запроса в <mark>."""
    if not query.strip():
        return text
    terms = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", query.lower())
    pattern = r"(" + "|".join(re.escape(t) for t in terms if t) + r")"
    highlighted = re.sub(
        pattern,
        r"<mark style='background:#fff3a3;padding:0 2px;border-radius:2px'>\1</mark>",
        text,
        flags=re.IGNORECASE,
    )
    return highlighted


def render_chunk(i: int, src: dict, query: str = "", expanded: bool = True) -> None:
    label = f"[{i}] doc_id={src['doc_id']} · score={src['score']:.4f}"
    # Улучшение 5: показываем sub-scores для hybrid
    if "score_tfidf" in src:
        label += f" (tfidf={src['score_tfidf']:.3f}, bm25={src['score_bm25']:.3f})"
    with st.expander(label, expanded=expanded):
        st.markdown(f"**{src['name']}**")
        # Улучшение 3: подсветка
        st.markdown(highlight(src["text"], query), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="RAG Tutorial", layout="wide")
    st.title("RAG Tutorial")
    st.caption("Учебный RAG: TF-IDF / Hybrid (TF-IDF + BM25) + demo-ответ с источниками")

    if not index_exists():
        st.error(
            "Индекс не собран. Сначала выполните:\n\n"
            "`uv run python scripts/build_index.py`"
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Настройки")

    # Улучшение 5: выбор режима retriever
    retriever_mode = st.sidebar.radio(
        "Retriever",
        options=["tfidf", "hybrid"],
        format_func=lambda x: "TF-IDF (cosine)" if x == "tfidf" else "Hybrid (TF-IDF + BM25, RRF)",
        index=0,
        help=(
            "TF-IDF — поиск по совпадению слов.\n"
            "Hybrid — объединяет TF-IDF и BM25 через Reciprocal Rank Fusion, "
            "устойчивее к вариациям длины документов."
        ),
    )

    # Улучшение 3: ползунок порога score
    score_threshold = st.sidebar.slider(
        "Минимальный score (порог отказа)",
        min_value=0.0,
        max_value=1.0,
        value=MIN_SCORE,
        step=0.01,
        help="Фрагменты ниже этого порога считаются нерелевантными.",
    )

    st.sidebar.divider()
    st.sidebar.header("Demo-вопросы")
    for q in DEMO_QUESTIONS:
        if st.sidebar.button(q, use_container_width=True):
            st.session_state["question"] = q

    # ── Инициализация истории ─────────────────────────────────────────────────
    # Улучшение 3: история запросов
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # ── Основная форма ────────────────────────────────────────────────────────
    question = st.text_input("Ваш вопрос", key="question")

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("Спросить", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Введите вопрос.")
            st.stop()

        retriever = load_retriever(retriever_mode)

        with st.spinner("Поиск..."):
            result = ask(
                question.strip(),
                k=TOP_K,
                retriever=retriever,
                min_score=score_threshold,
            )

        # Улучшение 3: сохраняем в историю
        st.session_state["history"].insert(
            0,
            {
                "q": question.strip(),
                "answer": result["answer"],
                "sources": result["sources"],
                "mode": retriever_mode,
                "threshold": score_threshold,
            },
        )

        # Показываем фрагменты с подсветкой
        st.subheader("Найденные фрагменты (top-k)")
        if not result["sources"]:
            st.info("Фрагменты не найдены.")
        for i, src in enumerate(result["sources"], 1):
            render_chunk(i, src, query=question, expanded=src["score"] >= score_threshold)

        st.subheader("Ответ")
        st.text(result["answer"])

        st.subheader("Источники")
        for i, src in enumerate(result["sources"], 1):
            render_chunk(i, src, query=question, expanded=False)

    # ── Улучшение 3: история запросов ─────────────────────────────────────────
    if st.session_state["history"]:
        st.divider()
        with st.expander(f"📜 История запросов ({len(st.session_state['history'])})", expanded=False):
            for entry in st.session_state["history"]:
                mode_label = "Hybrid" if entry["mode"] == "hybrid" else "TF-IDF"
                st.markdown(
                    f"**❓ {entry['q']}** "
                    f"<span style='color:grey;font-size:0.8em'>[{mode_label}, порог={entry['threshold']:.2f}]</span>",
                    unsafe_allow_html=True,
                )
                st.caption(entry["answer"][:200] + ("…" if len(entry["answer"]) > 200 else ""))
                st.divider()

        if st.button("Очистить историю"):
            st.session_state["history"] = []
            st.rerun()


if __name__ == "__main__":
    main()
