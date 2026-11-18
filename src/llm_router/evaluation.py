"""Offline evaluation (no Claude calls): cache thresholds and router label accuracy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .data import split
from .embeddings import Embedder
from .guard import CacheGuard
from .heuristic import HeuristicRouter
from .knn import Example, KnnRouter
from .pricing import cost_usd, estimate_tokens
from .text import normalise

CACHE_THRESHOLDS = (0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)

# Cost model for routing (no API call): output tokens per request by difficulty, including
# any adaptive thinking, assumed equal on both tiers; input tokens are estimated from the
# prompt text (~3.5 chars/token) plus a small fixed overhead.
OUTPUT_TOKENS = {"easy": 300, "hard": 900}
INPUT_OVERHEAD_TOKENS = 20


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# -- cache ------------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def threshold_rows(
    similarities: Sequence[float],
    should_hit: Sequence[bool],
    thresholds: Sequence[float] = CACHE_THRESHOLDS,
    vetoed: Sequence[bool] | None = None,
) -> list[dict[str, Any]]:
    sims = np.asarray(similarities)
    truth = np.asarray(should_hit, dtype=bool)
    allowed = ~np.asarray(vetoed, dtype=bool) if vetoed is not None else np.ones_like(truth)
    positives, negatives = int(truth.sum()), int((~truth).sum())
    rows = []
    for t in thresholds:
        pred = (sims >= t) & allowed
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = positives - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, positives)
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
        rows.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_hit_rate": _ratio(fp, negatives),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def recommend(rows: list[dict[str, Any]], max_false_hit_rate: float = 0.0) -> dict[str, Any]:
    """Highest recall whose false-hit rate stays within budget; ties go to the higher threshold.
    If no threshold meets the budget, the one with the fewest false hits."""
    safe = [r for r in rows if (r["false_hit_rate"] or 0.0) <= max_false_hit_rate]
    if safe:
        best = max(safe, key=lambda r: (r["recall"] or 0.0, r["threshold"]))
        return {**best, "met_budget": True}
    best = min(rows, key=lambda r: (r["false_hit_rate"] or 0.0, -(r["recall"] or 0.0)))
    return {**best, "met_budget": False}


def cache_eval(
    embedder: Embedder,
    pairs: list[dict[str, Any]],
    guard: CacheGuard | None = None,
    thresholds: Sequence[float] = CACHE_THRESHOLDS,
    max_false_hit_rate: float = 0.0,
) -> dict[str, Any]:
    a = embedder.embed([normalise(p["a"]) for p in pairs])
    b = embedder.embed([normalise(p["b"]) for p in pairs])
    sims = [float(x) for x in (a * b).sum(axis=1)]
    truth = [bool(p["should_hit"]) for p in pairs]
    guard = guard or CacheGuard()
    vetoes = [guard.check(p["b"], p["a"]) for p in pairs]  # b is the new query, a the cached prompt

    plain = threshold_rows(sims, truth, thresholds)
    guarded = threshold_rows(sims, truth, thresholds, vetoed=[v is not None for v in vetoes])
    details = [
        {
            "id": p["id"],
            "similarity": round(s, 4),
            "should_hit": p["should_hit"],
            "veto": v,
            "note": p["note"],
            "a": p["a"][:90],
            "b": p["b"][:90],
        }
        for p, s, v in zip(pairs, sims, vetoes, strict=True)
    ]
    negatives = sorted((d for d in details if not d["should_hit"]), key=lambda d: -d["similarity"])
    positives = sorted((d for d in details if d["should_hit"]), key=lambda d: d["similarity"])
    return {
        "embedding_model": embedder.name,
        "pairs": len(pairs),
        "should_hit": sum(truth),
        "should_not_hit": len(truth) - sum(truth),
        "max_false_hit_rate": max_false_hit_rate,
        "embedding_only": {"rows": plain, "recommended": recommend(plain, max_false_hit_rate)},
        "with_guard": {"rows": guarded, "recommended": recommend(guarded, max_false_hit_rate)},
        "guard_false_vetoes": [d for d in details if d["should_hit"] and d["veto"]],
        "hardest_negatives": negatives[:12],
        "weakest_positives": positives[:8],
        "details": details,
    }


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def _threshold_table(rows: list[dict[str, Any]], recommended: float) -> list[str]:
    lines = [
        "| threshold | precision | recall | F1 | false-hit rate | TP / FP / FN |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        mark = " **←**" if r["threshold"] == recommended else ""
        lines.append(
            f"| {r['threshold']:.2f}{mark} | {_fmt(r['precision'])} | {_fmt(r['recall'])} "
            f"| {_fmt(r['f1'])} | {_fmt(r['false_hit_rate'], pct=True)} | {r['tp']} / {r['fp']} / {r['fn']} |"
        )
    return lines


def cache_markdown(report: dict[str, Any]) -> str:
    plain, guarded = report["embedding_only"], report["with_guard"]
    lines = [
        f"## Cache threshold sweep ({report['embedding_model']})",
        "",
        f"{report['pairs']} labelled pairs: {report['should_hit']} paraphrases that should hit, "
        f"{report['should_not_hit']} near-misses that must not. A false hit serves the wrong answer, "
        f"so the recommendation is the highest recall with a false-hit rate of at most "
        f"{report['max_false_hit_rate'] * 100:.0f}%.",
        "",
        "### Embedding similarity only",
        "",
        *_threshold_table(plain["rows"], plain["recommended"]["threshold"]),
        "",
        "### Embedding similarity + lexical guard",
        "",
        *_threshold_table(guarded["rows"], guarded["recommended"]["threshold"]),
        "",
    ]
    for name, block in (("embedding only", plain), ("with guard", guarded)):
        rec = block["recommended"]
        status = (
            "" if rec["met_budget"] else " (no threshold met the false-hit budget; fewest false hits shown)"
        )
        lines.append(
            f"- Recommended ({name}): **{rec['threshold']:.2f}**, recall {_fmt(rec['recall'])}, "
            f"precision {_fmt(rec['precision'])}, "
            f"false-hit rate {_fmt(rec['false_hit_rate'], pct=True)}{status}"
        )
    lines += [
        "",
        "### Most similar near-misses (embedding similarity)",
        "",
        "| id | similarity | note | guard |",
        "|---|---:|---|---|",
    ]
    for d in report["hardest_negatives"]:
        lines.append(f"| {d['id']} | {d['similarity']:.3f} | {d['note']} | {d['veto'] or '**passes**'} |")
    if report["guard_false_vetoes"]:
        lines += ["", "### Paraphrases the guard wrongly vetoes", "", "| id | similarity | note | veto |"]
        lines.append("|---|---:|---|---|")
        for d in report["guard_false_vetoes"]:
            lines.append(f"| {d['id']} | {d['similarity']:.3f} | {d['note']} | {d['veto']} |")
    return "\n".join(lines) + "\n"


# -- router -----------------------------------------------------------------------------


def estimated_cost(prompt: str, label: str, model: str) -> float:
    return cost_usd(model, estimate_tokens(prompt) + INPUT_OVERHEAD_TOKENS, OUTPUT_TOKENS[label])


def classification_metrics(labels: Sequence[str], tiers: Sequence[str]) -> dict[str, Any]:
    pairs = list(zip(labels, tiers, strict=True))
    tp = sum(1 for lab, t in pairs if lab == "hard" and t == "large")
    fp = sum(1 for lab, t in pairs if lab == "easy" and t == "large")
    fn = sum(1 for lab, t in pairs if lab == "hard" and t == "small")
    tn = sum(1 for lab, t in pairs if lab == "easy" and t == "small")
    n = len(pairs)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "n": n,
        "accuracy": _ratio(tp + tn, n),
        "precision_hard": precision,
        "recall_hard": recall,
        "small_share": _ratio(tn + fn, n),
        "hard_sent_to_small": fn,
        "easy_sent_to_large": fp,
    }


KNN_GRID_K = (3, 5, 7, 9, 11)
KNN_GRID_MARGIN = (0.0, 0.05, 0.1, 0.15, 0.2)


def knn_leave_one_out(
    knn: KnnRouter, ks: Sequence[int] = KNN_GRID_K, margins: Sequence[float] = KNN_GRID_MARGIN
) -> list[dict[str, Any]]:
    """Leave-one-out over the kNN examples (the train split) for each (k, margin).

    Errors are weighted 2:1 (hard sent to small : easy sent to large) because a wrong
    small answer costs quality while an unnecessary large call only costs money."""
    sims = knn.matrix @ knn.matrix.T
    np.fill_diagonal(sims, -np.inf)
    labels = [e.label for e in knn.examples]
    rows = []
    for k in ks:
        top = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        weights = np.clip(np.take_along_axis(sims, top, axis=1), 1e-6, None)
        p_hard = (weights * knn.hard[top]).sum(axis=1) / weights.sum(axis=1)
        for margin in margins:
            tiers = ["large" if abs(p - 0.5) < margin or p >= 0.5 else "small" for p in p_hard]
            m = classification_metrics(labels, tiers)
            m.update(
                k=k, margin=margin, weighted_errors=2 * m["hard_sent_to_small"] + m["easy_sent_to_large"]
            )
            rows.append(m)
    return rows


def best_knn_params(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda r: (r["weighted_errors"], r["hard_sent_to_small"], r["k"], r["margin"]))


def router_eval(
    embedder: Embedder,
    items: list[dict[str, Any]],
    small_model: str,
    large_model: str,
    heuristic_threshold: float = 1.5,
    knn_k: int = 11,
    knn_margin: float = 0.0,
) -> dict[str, Any]:
    train, test = split(items)
    heuristic = HeuristicRouter(heuristic_threshold)
    knn = KnnRouter(
        embedder, [Example(it["id"], it["prompt"], it["label"]) for it in train], knn_k, knn_margin
    )
    vectors = embedder.embed([normalise(it["prompt"]) for it in test])
    models = {"small": small_model, "large": large_model}

    decisions: dict[str, list[tuple[str, str]]] = {"heuristic": [], "knn": []}
    for it, vec in zip(test, vectors, strict=True):
        tier, _, reason = heuristic.route(it["prompt"])
        decisions["heuristic"].append((tier, reason))
        tier, _, reason = knn.route(vec)
        decisions["knn"].append((tier, reason))
    decisions["always-small"] = [("small", "") for _ in test]
    decisions["always-large"] = [("large", "") for _ in test]
    decisions["oracle (labels)"] = [("large" if it["label"] == "hard" else "small", "") for it in test]

    labels = [it["label"] for it in test]
    large_cost = sum(estimated_cost(it["prompt"], it["label"], large_model) for it in test)
    results = {}
    for name, decided in decisions.items():
        tiers = [t for t, _ in decided]
        m = classification_metrics(labels, tiers)
        for lang in ("tr", "en"):
            idx = [i for i, it in enumerate(test) if it["lang"] == lang]
            m[f"accuracy_{lang}"] = classification_metrics([labels[i] for i in idx], [tiers[i] for i in idx])[
                "accuracy"
            ]
        cost = sum(
            estimated_cost(it["prompt"], it["label"], models[t]) for it, t in zip(test, tiers, strict=True)
        )
        m["cost_per_1k_usd"] = round(cost / len(test) * 1000, 4)
        m["saving_vs_large"] = round(1 - cost / large_cost, 4)
        m["misrouted"] = [
            {"id": it["id"], "label": it["label"], "tier": t, "reason": r}
            for it, (t, r) in zip(test, decided, strict=True)
            if (t == "large") != (it["label"] == "hard")
        ]
        results[name] = m

    train_tiers = [heuristic.route(it["prompt"])[0] for it in train]
    loo = knn_leave_one_out(knn)
    best = best_knn_params(loo)
    return {
        "embedding_model": embedder.name,
        "models": models,
        "train": len(train),
        "test": len(test),
        "test_mix": {"easy": labels.count("easy"), "hard": labels.count("hard")},
        "params": {"heuristic_threshold": heuristic_threshold, "knn_k": knn_k, "knn_margin": knn_margin},
        "cost_assumption": {
            "output_tokens": OUTPUT_TOKENS,
            "input_tokens": "len(prompt)/3.5 + 20",
            "same_tokens_on_both_tiers": True,
        },
        "heuristic_train_accuracy": classification_metrics([it["label"] for it in train], train_tiers)[
            "accuracy"
        ],
        "knn_leave_one_out": loo,
        "knn_loo_best": {
            k: best[k] for k in ("k", "margin", "accuracy", "hard_sent_to_small", "weighted_errors")
        },
        "strategies": results,
    }


def router_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"## Router label accuracy (test split: {report['test']} prompts, "
        f"{report['test_mix']['easy']} easy / {report['test_mix']['hard']} hard; "
        f"kNN trained on {report['train']})",
        "",
        "| strategy | accuracy | TR | EN | precision (hard) | recall (hard) | routed to small | hard → small "
        "| est. cost / 1k req | cost vs always-large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in report["strategies"].items():
        lines.append(
            f"| {name} | {_fmt(m['accuracy'])} | {_fmt(m['accuracy_tr'])} | {_fmt(m['accuracy_en'])} | "
            f"{_fmt(m['precision_hard'])} | {_fmt(m['recall_hard'])} | {_fmt(m['small_share'], pct=True)} | "
            f"{m['hard_sent_to_small']} | ${m['cost_per_1k_usd']:.2f} | "
            f"{(1 - m['saving_vs_large']) * 100:.0f}% |"
        )
    a = report["cost_assumption"]
    lines += [
        "",
        f"Cost model: {report['models']['small']} vs {report['models']['large']} list prices; input tokens "
        f"≈ {a['input_tokens']}, output tokens {a['output_tokens']['easy']} (easy) / "
        f"{a['output_tokens']['hard']} (hard) including thinking, assumed equal on both tiers. "
        f"Heuristic accuracy on the train split (used while writing the rules): "
        f"{_fmt(report['heuristic_train_accuracy'])}.",
        "",
        f"kNN parameters used: k={report['params']['knn_k']}, margin={report['params']['knn_margin']:.2f}. "
        f"Leave-one-out on the train split (errors weighted 2:1, hard→small : easy→large) picks "
        f"k={report['knn_loo_best']['k']}, margin={report['knn_loo_best']['margin']:.2f} "
        f"(LOO accuracy {_fmt(report['knn_loo_best']['accuracy'])}):",
        "",
        "| k | margin | LOO accuracy | routed to small | hard → small | easy → large |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["knn_leave_one_out"]:
        lines.append(
            f"| {r['k']} | {r['margin']:.2f} | {_fmt(r['accuracy'])} | {_fmt(r['small_share'], pct=True)} | "
            f"{r['hard_sent_to_small']} | {r['easy_sent_to_large']} |"
        )
    return "\n".join(lines) + "\n"
