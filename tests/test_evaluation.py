import json
from pathlib import Path

import pytest

from llm_router import cli
from llm_router.data import load_examples, load_jsonl, split
from llm_router.embeddings import HashingEmbedder
from llm_router.evaluation import (
    cache_eval,
    classification_metrics,
    recommend,
    router_eval,
    router_markdown,
    threshold_rows,
)
from llm_router.live_eval import plan

EVAL = Path(__file__).resolve().parents[1] / "eval"


def test_threshold_rows_count_false_hits():
    sims = [0.99, 0.95, 0.90, 0.97, 0.85]
    truth = [True, True, True, False, False]
    rows = {r["threshold"]: r for r in threshold_rows(sims, truth, thresholds=(0.90, 0.96, 0.98))}
    assert (rows[0.90]["tp"], rows[0.90]["fp"], rows[0.90]["fn"]) == (3, 1, 0)
    assert rows[0.90]["false_hit_rate"] == 0.5 and rows[0.90]["precision"] == 0.75
    assert (rows[0.96]["tp"], rows[0.96]["fp"]) == (1, 1)
    assert (rows[0.98]["tp"], rows[0.98]["fp"]) == (1, 0)
    vetoed = threshold_rows(sims, truth, thresholds=(0.90,), vetoed=[False, False, False, True, False])[0]
    assert vetoed["fp"] == 0 and vetoed["precision"] == 1.0


def test_recommend_prefers_recall_within_the_false_hit_budget():
    rows = threshold_rows(
        [0.99, 0.95, 0.90, 0.97, 0.85], [True, True, True, False, False], (0.90, 0.96, 0.98)
    )
    best = recommend(rows, max_false_hit_rate=0.0)
    assert best["threshold"] == 0.98 and best["met_budget"]
    assert recommend(rows, max_false_hit_rate=0.5)["threshold"] == 0.90
    none_safe = threshold_rows([0.99, 0.99], [True, False], (0.9, 0.95))
    assert recommend(none_safe)["met_budget"] is False


def test_split_is_deterministic_and_stratified():
    items = load_jsonl(EVAL / "prompts.jsonl")
    train, test = split(items)
    _, test2 = split(list(reversed(items)))
    assert {it["id"] for it in test} == {it["id"] for it in test2}
    assert not {it["id"] for it in train} & {it["id"] for it in test}
    for label in ("easy", "hard"):
        for lang in ("tr", "en"):
            group = [it for it in items if it["label"] == label and it["lang"] == lang]
            in_test = [it for it in test if it["label"] == label and it["lang"] == lang]
            assert len(in_test) == round(len(group) * 0.3)
    assert len(load_examples(EVAL / "prompts.jsonl")) == len(train)


def test_eval_files_are_well_formed():
    prompts = load_jsonl(EVAL / "prompts.jsonl")
    assert len(prompts) >= 120 and len({p["id"] for p in prompts}) == len(prompts)
    assert {p["label"] for p in prompts} == {"easy", "hard"}
    assert all(p["rationale"] and p["prompt"] for p in prompts)
    for name in ("cache_pairs.jsonl", "cache_pairs_holdout.jsonl"):
        pairs = load_jsonl(EVAL / name)
        assert all({"a", "b", "should_hit", "note"} <= set(p) for p in pairs)
    assert len(load_jsonl(EVAL / "cache_pairs.jsonl")) >= 80


def test_classification_metrics():
    m = classification_metrics(["hard", "hard", "easy", "easy"], ["large", "small", "small", "large"])
    assert m["accuracy"] == 0.5 and m["recall_hard"] == 0.5 and m["precision_hard"] == 0.5
    assert m["hard_sent_to_small"] == 1 and m["easy_sent_to_large"] == 1 and m["small_share"] == 0.5


def test_router_eval_runs_offline_with_a_hashing_embedder():
    items = load_jsonl(EVAL / "prompts.jsonl")
    report = router_eval(HashingEmbedder(128), items, "claude-haiku-5-5", "claude-sonnet-5-5")
    s = report["strategies"]
    assert s["always-large"]["saving_vs_large"] == 0.0 and s["always-small"]["small_share"] == 1.0
    assert s["oracle (labels)"]["accuracy"] == 1.0
    assert (
        s["always-small"]["cost_per_1k_usd"]
        < s["heuristic"]["cost_per_1k_usd"]
        < s["always-large"]["cost_per_1k_usd"]
    )
    assert report["knn_loo_best"]["k"] in (3, 5, 7, 9, 11)
    assert "| heuristic |" in router_markdown(report)


def test_cache_eval_runs_offline():
    pairs = load_jsonl(EVAL / "cache_pairs.jsonl")
    report = cache_eval(HashingEmbedder(128), pairs)
    assert report["pairs"] == len(pairs)
    assert len(report["embedding_only"]["rows"]) == 10
    # the guard can only remove hits, never add them
    for plain, guarded in zip(report["embedding_only"]["rows"], report["with_guard"]["rows"], strict=True):
        assert guarded["tp"] + guarded["fp"] <= plain["tp"] + plain["fp"]


def test_live_plan_counts_calls_and_bounds_cost():
    items = [{"id": "a", "prompt": "x" * 350}, {"id": "b", "prompt": "y" * 35}]
    p = plan(items, {"small": "claude-haiku-5-5", "large": "claude-sonnet-5-5"}, max_tokens=1000)
    assert p["generation_calls"] == 6 and p["judge_calls"] == 4 and p["total_calls"] == 10
    assert p["max_cost_usd"] > 0
    large_part = next(v for k, v in p["max_cost_by_part_usd"].items() if k.startswith("large ("))
    assert large_part == pytest.approx(round((100 * 2 + 1000 * 10 + 10 * 2 + 1000 * 10) / 1e6, 2))


def test_eval_live_dry_run_makes_no_calls(monkeypatch, capsys, tmp_path):
    import anthropic

    def forbidden(*a, **kw):
        raise AssertionError("eval-live --dry-run must not create an API client")

    monkeypatch.setattr(anthropic, "Anthropic", forbidden)
    assert (
        cli.main(["eval-live", "--dry-run", "--prompts", str(EVAL / "prompts.jsonl"), "--out", str(tmp_path)])
        == 0
    )
    out = capsys.readouterr().out
    assert "generation calls" in out and "upper bound on cost" in out
    assert not list(tmp_path.iterdir())
    # without --yes it also only prints the plan
    assert cli.main(["eval-live", "--prompts", str(EVAL / "prompts.jsonl"), "--out", str(tmp_path)]) == 0
    assert "Nothing was called" in capsys.readouterr().out


def test_live_scoring_from_stored_outputs(tmp_path):
    from llm_router.heuristic import HeuristicRouter
    from llm_router.live_eval import score

    def row(item_id, arm, model, text, inp, out):
        return {
            "id": item_id,
            "arm": arm,
            "text": text,
            "model": model,
            "requested_model": model,
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    confident = json.dumps({"answer": "ok", "confidence": 0.9, "needs_stronger_model": False})
    unsure = json.dumps({"answer": "?", "confidence": 0.2, "needs_stronger_model": False})
    outputs = tmp_path / "out.jsonl"
    judgments = tmp_path / "judge.jsonl"
    rows = [
        row("p1", "small", "claude-haiku-5-5", confident, 100, 100),
        row("p1", "large", "claude-sonnet-5-5", "ref", 100, 100),
        row("p1", "large_low", "claude-sonnet-5-5", "low", 100, 50),
        row("p2", "small", "claude-haiku-5-5", unsure, 100, 100),
        row("p2", "large", "claude-sonnet-5-5", "ref", 100, 100),
        row("p2", "large_low", "claude-sonnet-5-5", "low", 100, 50),
    ]
    outputs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    verdicts = [
        {"id": "p1", "arm": "small", "acceptable": True},
        {"id": "p1", "arm": "large_low", "acceptable": True},
        {"id": "p2", "arm": "small", "acceptable": False},
        {"id": "p2", "arm": "large_low", "acceptable": True},
    ]
    judgments.write_text("\n".join(json.dumps(v) for v in verdicts), encoding="utf-8")
    items = [
        {"id": "p1", "prompt": "Translate to French: hello"},
        {"id": "p2", "prompt": "Prove the theorem."},
    ]
    report = score(items, outputs, judgments, HeuristicRouter(), None, cascade_threshold=0.7)
    s = report["strategies"]
    assert s["always-small"]["quality"] == 0.5 and s["always-large"]["quality"] == 1.0
    assert s["cascade"]["quality"] == 1.0 and s["cascade"]["small_share"] == 0.5
    assert s["large, effort low"]["quality"] == 1.0
    assert s["cascade"]["cost_usd"] > s["always-small"]["cost_usd"]
