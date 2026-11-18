"""Command line: offline evaluations (free), the live evaluation plan, routing explain, serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Settings

EVAL = Path("eval")
RESULTS = EVAL / "results"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="llm-router")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "eval-cache", help="precision/recall/false hits per similarity threshold (free, local)"
    )
    p.add_argument("--pairs", type=Path, default=EVAL / "cache_pairs.jsonl")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--max-false-hit-rate", type=float, default=0.0)

    p = sub.add_parser(
        "eval-router", help="heuristic and kNN routing vs labels on the test split (free, local)"
    )
    p.add_argument("--prompts", type=Path, default=EVAL / "prompts.jsonl")
    p.add_argument("--out", type=Path, default=RESULTS)

    p = sub.add_parser("eval-live", help="quality vs cost with real Claude calls and an Opus judge")
    p.add_argument("--prompts", type=Path, default=EVAL / "prompts.jsonl")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dry-run", action="store_true", help="print the planned calls and max cost, call nothing"
    )
    p.add_argument("--score-only", action="store_true", help="re-score stored outputs without new calls")
    p.add_argument("--yes", action="store_true", help="confirm that real API calls (and spend) are intended")

    p = sub.add_parser("route", help="explain how each strategy would route a prompt (no Claude call)")
    p.add_argument("prompt")

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("llm_router.api:app", host=args.host, port=args.port)
        return 0
    if args.command == "eval-live":
        return _eval_live(args, settings)

    from .data import load_jsonl
    from .embeddings import create_embedder
    from .evaluation import cache_eval, cache_markdown, router_eval, router_markdown, write_json

    embedder = create_embedder(settings.embedding_model, cache_dir=settings.embedding_cache_dir)
    if args.command == "eval-cache":
        from .guard import CacheGuard

        report = cache_eval(
            embedder,
            load_jsonl(args.pairs),
            guard=CacheGuard(settings.cache_exact_over_chars),
            max_false_hit_rate=args.max_false_hit_rate,
        )
        write_json(args.out / "cache.json", report)
        md = cache_markdown(report)
        (args.out / "cache.md").write_text(md, encoding="utf-8")
        print(md)
    elif args.command == "eval-router":
        report = router_eval(
            embedder,
            load_jsonl(args.prompts),
            settings.small_model,
            settings.large_model,
            settings.heuristic_threshold,
            settings.knn_k,
            settings.knn_margin,
        )
        write_json(args.out / "router.json", report)
        md = router_markdown(report)
        (args.out / "router.md").write_text(md, encoding="utf-8")
        print(md)
    elif args.command == "route":
        from .data import load_examples
        from .embeddings import embed_one
        from .heuristic import HeuristicRouter
        from .knn import KnnRouter
        from .text import normalise

        heuristic = HeuristicRouter(settings.heuristic_threshold)
        print("heuristic:", *heuristic.route(args.prompt))
        if settings.knn_examples.exists():
            knn = KnnRouter(
                embedder, load_examples(settings.knn_examples), settings.knn_k, settings.knn_margin
            )
            print("knn:      ", *knn.route(embed_one(embedder, normalise(args.prompt))))
    return 0


def _eval_live(args: argparse.Namespace, settings: Settings) -> int:
    from .data import load_jsonl, split
    from .live_eval import generate, judge_all, plan, plan_text, score, score_markdown

    _, test = split(load_jsonl(args.prompts))
    test = test[: args.limit]
    models = {"small": settings.small_model, "large": settings.large_model}
    outputs, judgments = args.out / "live_outputs.jsonl", args.out / "live_judgments.jsonl"

    if args.dry_run or not (args.yes or args.score_only):
        print(plan_text(plan(test, models, args.max_tokens)))
        if not args.dry_run:
            print("\nNothing was called. Re-run with --yes to make these calls (needs ANTHROPIC_API_KEY).")
        return 0

    from .data import load_examples
    from .embeddings import create_embedder
    from .evaluation import write_json
    from .heuristic import HeuristicRouter
    from .knn import KnnRouter

    if not args.score_only:
        import anthropic

        from .claude import ClaudeClient

        claude = ClaudeClient(anthropic.Anthropic())
        print(plan_text(plan(test, models, args.max_tokens)))
        generate(claude, test, models, outputs, args.max_tokens)
        judge_all(claude, test, outputs, judgments)

    embedder = create_embedder(settings.embedding_model, cache_dir=settings.embedding_cache_dir)
    knn = KnnRouter(embedder, load_examples(args.prompts), settings.knn_k, settings.knn_margin)
    report = score(
        test,
        outputs,
        judgments,
        HeuristicRouter(settings.heuristic_threshold),
        knn,
        settings.cascade_threshold,
    )
    write_json(args.out / "live.json", report)
    md = score_markdown(report)
    (args.out / "live.md").write_text(md, encoding="utf-8")
    print(json.dumps(report["strategies"], indent=2) if not report["prompts_scored"] else md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
