"""
Улучшение 5 — BM25 (Okapi BM25), реализован без внешних библиотек.

Используется в hybrid retriever вместе с TF-IDF.
"""

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text.lower())


class BM25:
    """
    Okapi BM25 поверх списка текстов.

    Параметры:
        k1: насыщение TF (стандарт 1.5)
        b:  нормировка длины документа (стандарт 0.75)
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(t) for t in texts]
        N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / N if N else 1.0

        # document frequency
        df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        # IDF с Robertson–Sparck Jones сглаживанием
        self.idf: dict[str, float] = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """Возвращает BM25-score для каждого документа."""
        q_terms = tokenize(query)
        result = []
        for doc in self.corpus:
            dl = len(doc)
            tf_map = Counter(doc)
            score = 0.0
            for term in q_terms:
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[term] * num / den
            result.append(score)
        return result
