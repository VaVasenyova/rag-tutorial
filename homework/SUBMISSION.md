# Submission

## Ссылка на репозиторий с заданием

- Repo URL: https://github.com/VaVasenyova/rag-tutorial

## Автор

- Васенёва Валерия, группа БАСБ252

## Датасет

**Dinosaur Genera Dataset** — 97 родов динозавров.
Источник: Kaggle (`canozensoy/dinosaur-genera-dataset`).
Поля: name, diet, period, lived_in, type, length, taxonomy, named_by.

## Demo-вопросы с ответами

### 1. What did T. rex eat?
Найдено: Tyrannosaurus (doc_id=0, score > 0.15)
Ответ: описание плотоядного теропода позднего мела.

### 2. Dinosaurs that lived in Argentina
Найдено: Carnotaurus, Giganotosaurus, Eoraptor (score > 0.15)
Ответ: список динозавров с описанием.

### 3. Armored dinosaur with tail club
Найдено: Ankylosaurus (doc_id=6, score > 0.25)
Ответ: описание бронированного динозавра с хвостовой булавой.

## Negative-вопрос

**Запрос:** "How to cook pasta carbonara?"
**Ответ:** "В базе не найдено релевантных фрагментов. Ответить по данным невозможно."
**Score:** < 0.15 → корректный отказ ✅

## Тесты

`uv run pytest tests/ -v` — 20 тестов, все passed:
- `test_chunking.py` — 4 теста
- `test_retrieval.py` — 6 тестов
- `test_bm25.py` — 5 тестов
- `test_hybrid.py` — 5 тестов

## Реализованные улучшения

**Улучшение 3 — UI:** ползунок порога score, история запросов, подсветка слов.
**Улучшение 5 — Hybrid Search:** BM25 + TF-IDF через Reciprocal Rank Fusion.
