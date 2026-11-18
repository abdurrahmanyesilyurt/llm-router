"""Loading the labelled eval files and the deterministic train/test split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .knn import Example

TEST_FRACTION = 0.3


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rank(item_id: str) -> str:
    return hashlib.blake2b(item_id.encode("utf-8"), digest_size=8).hexdigest()


def split(items: Iterable[dict[str, Any]], test_fraction: float = TEST_FRACTION) -> tuple[list, list]:
    """Stratified by (label, lang), ordered by a hash of the id: the same file always gives
    the same split, and adding a prompt moves at most a few others."""
    items = list(items)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[(item["label"], item.get("lang", ""))].append(item)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda it: _rank(it["id"]))
        n_test = round(len(ordered) * test_fraction)
        test.extend(ordered[:n_test])
        train.extend(ordered[n_test:])
    order = {it["id"]: i for i, it in enumerate(items)}
    train.sort(key=lambda it: order[it["id"]])
    test.sort(key=lambda it: order[it["id"]])
    return train, test


def load_examples(path: Path) -> list[Example]:
    """The train split of the labelled prompts, as kNN examples."""
    train, _ = split(load_jsonl(path))
    return [Example(it["id"], it["prompt"], it["label"]) for it in train]
