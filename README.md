# 🦕 DinoRAG — RAG на базе данных динозавров

Учебный RAG на Dinosaur Genera Dataset: TF-IDF / Hybrid (BM25+TF-IDF) + demo-ответ с источниками.  
Pipeline: данные → чанки → индекс → поиск → ответ.

## Требования

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Быстрый старт

```bash
# 1. Зависимости
uv sync

# 2. Сборка индекса (ingest + chunk + TF-IDF)
uv run python scripts/build_index.py

# 3. Запуск UI
uv run streamlit run app/main.py
```

Открыть: http://localhost:8501

## Demo-вопросы

| Вопрос | Ожидание |
|--------|----------|
| `What did T. rex eat?` | Tyrannosaurus, score > 0.15 |
| `Dinosaurs that lived in Argentina` | Carnotaurus, Giganotosaurus |
| `Armored dinosaur with tail club` | Ankylosaurus, score > 0.25 |
| `How to cook pasta carbonara?` | ❌ отказ (нет данных) |

## Тесты

```bash
uv run pytest tests/ -v
# 20 passed
```

## Структура

```
rag-tutorial/
├── app/
│   ├── config.py           # пути и параметры
│   ├── chunker.py          # нарезка текста
│   ├── retriever.py        # TF-IDF + cosine
│   ├── bm25.py             # BM25 (улучшение 5)
│   ├── hybrid_retriever.py # TF-IDF + BM25 через RRF (улучшение 5)
│   ├── generator.py        # сборка ответа
│   ├── prompts.py          # порог отказа
│   └── main.py             # Streamlit UI (улучшение 3)
├── scripts/
│   ├── ingest.py
│   └── build_index.py
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_bm25.py        # новые тесты (улучшение 5)
│   └── test_hybrid.py      # новые тесты (улучшение 5)
├── data/
│   └── raw/datasets.json   # 97 динозавров
├── doc/
├── homework/
│   ├── IMPROVEMENTS.md
│   └── SUBMISSION.md
└── pyproject.toml
```

## Реализованные улучшения

### Улучшение 3 — Интерфейс Streamlit
- Ползунок порога score в sidebar
- История запросов (сохраняется в session)
- Подсветка слов запроса в найденных фрагментах

### Улучшение 5 — Hybrid Search
- `app/bm25.py` — Okapi BM25 без внешних библиотек
- `app/hybrid_retriever.py` — RRF объединяет TF-IDF и BM25
- Переключатель TF-IDF / Hybrid в sidebar UI

## Данные

**Dinosaur Genera Dataset**, 97 записей.  
Kaggle: [canozensoy/dinosaur-genera-dataset](https://www.kaggle.com/datasets/canozensoy/dinosaur-genera-dataset)  
Подробнее: [doc/DATA.md](doc/DATA.md)
