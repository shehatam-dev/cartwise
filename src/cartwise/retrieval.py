"""BM25 retrieval over the store's policy documents.

Pure Python on purpose: the corpus is small and this keeps the project runnable offline.
`Retriever` is a protocol, so an embedding-backed store can be swapped in without touching the agent.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol

from .text import tokenize


@dataclass(frozen=True)
class Chunk:
    source: str      # e.g. "returns.md#Opened electronics"
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[Hit]: ...


def chunk_markdown(name: str, markdown: str) -> list[Chunk]:
    """Split a policy file into one chunk per `##` section."""
    chunks, heading, lines = [], None, []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if heading and lines:
                chunks.append(Chunk(f"{name}#{heading}", " ".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif heading and line.strip():
            lines.append(line.strip())
    if heading and lines:
        chunks.append(Chunk(f"{name}#{heading}", " ".join(lines).strip()))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, min_score: float = 0.5):
        self.chunks = chunks
        self.k1, self.b, self.min_score = k1, b, min_score
        # Headings carry a lot of signal ("Opened electronics"), so index them with the body.
        self._docs = [Counter(tokenize(f"{c.source.split('#')[-1]} {c.text}")) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / max(len(self._lengths), 1)
        df = Counter(term for doc in self._docs for term in doc)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_policies(cls, directory: str | Path | None = None) -> BM25Retriever:
        root = Path(directory) if directory else resources.files("cartwise").joinpath("data/policies")
        chunks = []
        for path in sorted(root.iterdir(), key=lambda p: p.name):
            if path.name.endswith(".md"):
                chunks.extend(chunk_markdown(path.name, path.read_text()))
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = tokenize(query)
        hits = []
        for chunk, doc, length in zip(self.chunks, self._docs, self._lengths, strict=True):
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg)
                    score += self._idf[t] * tf * (self.k1 + 1) / norm
            if score >= self.min_score:
                hits.append(Hit(chunk, round(score, 3)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
