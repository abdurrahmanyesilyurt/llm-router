"""Embedding backends.

* ``FastEmbedEmbedder`` runs a small multilingual model locally on CPU (ONNX), so cache
  lookups and kNN routing cost nothing per call and prompts never leave the machine.
  The default (paraphrase-multilingual-MiniLM-L12-v2, 384 dims) truncates input at 128
  tokens, which is fine for short chat prompts; longer prompts only hit the cache on an
  exact match (see ``CacheGuard``).
* ``HashingEmbedder`` is a deterministic bag-of-n-grams embedder for tests and CI: no
  model download, same vectors on every machine.

Every vector is L2-normalised, so a dot product is the cosine similarity.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .text import tr_lower, words


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def embed_one(embedder: Embedder, text: str) -> np.ndarray:
    return embedder.embed([text])[0]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32) -> None:
        from fastembed import TextEmbedding  # heavy import, only when a real model is used

        self.name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)
        self._batch_size = batch_size
        self.dim = int(self.embed(["boyut"]).shape[1])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = list(self._model.embed(list(texts), batch_size=self._batch_size))
        return _unit_rows(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Signed feature hashing of words and character trigrams."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = words(text)
        grams = [g for w in tokens for g in _trigrams(w)]
        for feature in tokens + grams:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _unit_rows(np.asarray([self._vector(t) for t in texts], dtype=np.float32))


def _trigrams(word: str) -> list[str]:
    padded = f"#{tr_lower(word)}#"
    return [padded[i : i + 3] for i in range(len(padded) - 2)]


def create_embedder(model_name: str, cache_dir: str | None = None) -> Embedder:
    if model_name.startswith("hashing"):
        dim = int(model_name.split("-", 1)[1]) if "-" in model_name else 256
        return HashingEmbedder(dim)
    return FastEmbedEmbedder(model_name, cache_dir=cache_dir)
