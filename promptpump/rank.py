"""Relevance ranking: pure-Python BM25 over file paths, symbols, and content."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping, Sequence

_STOPWORDS = frozenset(
    {"the", "a", "an", "in", "on", "of", "to", "and", "or", "for", "with", "is", "it",
     "this", "that", "each", "every", "all", "by", "from", "as", "be", "at", "my", "our",
     # generic request verbs carry no signal about *where* to change code
     "add", "make", "create", "update", "fix", "implement", "change", "use", "new", "logic",
     "return", "get", "set"}
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _stem(token: str) -> str:
    """Crude plural stripping, applied identically to queries and documents."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, breaking camelCase and snake_case identifiers."""
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        tokens.extend(part.lower() for part in _CAMEL_RE.findall(word))
    return [_stem(t) for t in tokens if len(t) > 1 and t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.freqs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        n = len(docs)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        df: Counter[str] = Counter()
        for freq in self.freqs:
            df.update(freq.keys())
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: Sequence[str], index: int) -> float:
        freq, length = self.freqs[index], self.lengths[index]
        total = 0.0
        for term in set(query):
            tf = freq.get(term, 0)
            if not tf:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1.0))
            total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
        return total

    def scores(self, query: Sequence[str]) -> list[float]:
        return [self.score(query, i) for i in range(len(self.freqs))]


def rank_files(
    docs: Mapping[str, str],
    query: str,
    symbols: Mapping[str, str] | None = None,
    path_boost: int = 3,
    symbol_boost: int = 3,
) -> list[tuple[str, float]]:
    """Rank ``{path: text}`` by relevance to ``query`` (highest first; ties by path).

    Path and symbol names (classes, functions) are repeated so a file that *defines* or is
    *named after* the topic outranks one that merely mentions it in a long body of text.
    """
    paths = sorted(docs)
    if not paths:
        return []
    symbols = symbols or {}
    tokenized = [
        tokenize(p) * path_boost + tokenize(symbols.get(p, "")) * symbol_boost + tokenize(docs[p])
        for p in paths
    ]
    scores = BM25(tokenized).scores(tokenize(query))
    return sorted(zip(paths, scores), key=lambda item: (-item[1], item[0]))
