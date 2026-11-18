"""k-nearest-neighbour routing over labelled example prompts.

The examples (the train split of ``eval/prompts.jsonl``) are embedded once with the same
local model the cache uses, so routing a request costs one matrix-vector product on top
of the embedding the cache already computed. The vote is similarity-weighted; when it is
too close to call (``|P(hard) - 0.5| < margin``) the request goes to the large tier,
because an unnecessary large call costs money while a wrong small answer costs quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .text import normalise


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    label: str  # "easy" | "hard"


@dataclass
class Vote:
    p_hard: float
    neighbours: list[tuple[str, str, float]]  # (id, label, similarity)

    @property
    def hard_votes(self) -> int:
        return sum(1 for _, label, _ in self.neighbours if label == "hard")


class KnnRouter:
    def __init__(
        self, embedder: Embedder, examples: Sequence[Example], k: int = 11, margin: float = 0.0
    ) -> None:
        if not examples:
            raise ValueError("kNN routing needs at least one labelled example")
        self.embedder = embedder
        self.examples = list(examples)
        self.k = min(k, len(self.examples))
        self.margin = margin
        self.matrix = embedder.embed([normalise(e.prompt) for e in self.examples])
        self.hard = np.array([e.label == "hard" for e in self.examples], dtype=np.float32)

    def vote(self, vector: np.ndarray) -> Vote:
        sims = self.matrix @ np.asarray(vector, dtype=np.float32)
        top = np.argsort(-sims, kind="stable")[: self.k]
        weights = np.clip(sims[top], 1e-6, None)
        p_hard = float((weights * self.hard[top]).sum() / weights.sum())
        neighbours = [(self.examples[i].id, self.examples[i].label, float(sims[i])) for i in top]
        return Vote(p_hard, neighbours)

    def route(self, vector: np.ndarray) -> tuple[str, float, str]:
        v = self.vote(vector)
        nearest_id, nearest_label, nearest_sim = v.neighbours[0]
        summary = (
            f"{v.hard_votes}/{len(v.neighbours)} neighbours hard, P(hard)={v.p_hard:.2f}; "
            f"nearest {nearest_id} ({nearest_label}, {nearest_sim:.2f})"
        )
        if abs(v.p_hard - 0.5) < self.margin:
            return "large", v.p_hard, f"{summary}; unsure (margin {self.margin:.2f}) -> large"
        tier = "large" if v.p_hard >= 0.5 else "small"
        return tier, v.p_hard, summary
