"""Request flow: PII check -> semantic cache -> router -> Claude -> cache write -> metrics."""

from __future__ import annotations

import logging
import time
from typing import Any

from .cache import RedisSemanticCache, SemanticCache, namespace
from .cascade import Cascade
from .claude import ClaudeClient, Completion
from .config import STRATEGIES, Settings
from .embeddings import Embedder, create_embedder, embed_one
from .guard import CacheGuard
from .heuristic import HeuristicRouter
from .knn import KnnRouter
from .metrics import Metrics
from .pii import find_pii
from .text import normalise

log = logging.getLogger(__name__)


class RoutingError(ValueError):
    """The request asks for a strategy this gateway cannot run."""


def _call_usage(tier: str, c: Completion) -> dict[str, Any]:
    return {
        "tier": tier,
        "model": c.model,
        "input_tokens": c.usage.input_tokens,
        "output_tokens": c.usage.output_tokens,
        "cache_read_input_tokens": c.usage.cache_read_input_tokens,
        "cache_creation_input_tokens": c.usage.cache_creation_input_tokens,
        "stop_reason": c.stop_reason,
        "cost_usd": round(c.cost_usd, 8),
        "latency_ms": c.latency_ms,
    }


class Gateway:
    def __init__(
        self,
        *,
        settings: Settings,
        claude: ClaudeClient,
        embedder: Embedder,
        cache: SemanticCache | None,
        knn: KnnRouter | None,
        metrics: Metrics | None = None,
    ) -> None:
        self.settings = settings
        self.claude = claude
        self.embedder = embedder
        self.cache = cache
        self.knn = knn
        self.metrics = metrics or Metrics()
        self.heuristic = HeuristicRouter(settings.heuristic_threshold)
        self.cascade = Cascade(
            claude,
            settings.small_model,
            settings.large_model,
            settings.cascade_threshold,
            settings.small_effort,
            settings.large_effort,
        )
        self.guard = CacheGuard(settings.cache_exact_over_chars) if settings.cache_guard else None
        self.models = {"small": settings.small_model, "large": settings.large_model}
        self.efforts = {"small": settings.small_effort, "large": settings.large_effort}

    # -- decisions ---------------------------------------------------------------------

    def cache_skip_reason(
        self, use_cache: bool, system: str | None, messages: list[dict[str, str]]
    ) -> str | None:
        if not use_cache:
            return "disabled_by_request"
        if not self.settings.cache_enabled:
            return "disabled_by_config"
        if self.cache is None:
            return "cache_unavailable"
        kinds = sorted(
            {k for text in [system or "", *(m["content"] for m in messages)] for k in find_pii(text)}
        )
        if kinds:
            return "pii:" + ",".join(kinds)
        return None

    def decide(self, strategy: str, prompt: str, vector: Any) -> tuple[str, float | None, str]:
        if strategy == "always-small":
            return "small", None, "baseline: always the small tier"
        if strategy == "always-large":
            return "large", None, "baseline: always the large tier"
        if strategy == "heuristic":
            return self.heuristic.route(prompt)
        if strategy == "knn":
            if self.knn is None:
                raise RoutingError("kNN routing is unavailable: no labelled examples were loaded")
            return self.knn.route(vector)
        raise RoutingError(f"unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}")

    # -- the request -------------------------------------------------------------------

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int | None = None,
        strategy: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        strategy = strategy or self.settings.default_strategy
        max_tokens = max_tokens or self.settings.default_max_tokens
        if strategy not in STRATEGIES:
            raise RoutingError(f"unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}")
        if strategy == "knn" and self.knn is None:
            raise RoutingError("kNN routing is unavailable: no labelled examples were loaded")
        if not messages or messages[-1]["role"] != "user":
            raise RoutingError("the last message must come from the user")

        prompt = messages[-1]["content"]
        ns = namespace(system, messages[:-1])
        skip = self.cache_skip_reason(use_cache, system, messages)
        cache_info: dict[str, Any] = {"hit": False, "similarity": None, "stored": False, "skipped": skip}
        vector = None
        if skip is None or strategy == "knn":
            vector = embed_one(self.embedder, normalise(prompt))

        if skip is None:
            hit = self._lookup(ns, prompt, vector, cache_info)
            if hit is not None:
                return self._hit_response(hit, strategy, cache_info, started)
        else:
            self.metrics.cache_skipped.labels(skip.split(":")[0]).inc()

        calls: list[tuple[str, Completion]]
        if strategy == "cascade":
            result = self.cascade.run(system, messages, max_tokens)
            calls = result.calls
            verdict = result.verdict
            route = {
                "strategy": strategy,
                "reason": verdict.reason if verdict else "",
                "score": verdict.confidence if verdict else None,
                "escalated": result.escalated,
            }
            if result.escalated and verdict:
                self.metrics.escalations.labels(verdict.kind).inc()
            final = result.final
        else:
            tier, score, reason = self.decide(strategy, prompt, vector)
            final = self.claude.complete(
                model=self.models[tier],
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                effort=self.efforts[tier],
            )
            calls = [(tier, final)]
            route = {"strategy": strategy, "reason": reason, "score": score}
        tier = calls[-1][0]
        cost = sum(c.cost_usd for _, c in calls)

        if skip is None and final.finished:
            try:
                self.cache.store(  # type: ignore[union-attr]
                    ns,
                    vector,
                    prompt,
                    {
                        "text": final.text,
                        "model": final.model,
                        "tier": tier,
                        "strategy": strategy,
                        "cost_usd": cost,
                    },
                )
                cache_info["stored"] = True
            except Exception as exc:  # the cache is an optimisation, never a reason to fail
                log.warning("cache write failed: %s", exc)
                self.metrics.errors.labels("cache_write").inc()

        latency_ms = int((time.perf_counter() - started) * 1000)
        cache_label = "miss" if skip is None else "skipped"
        self._observe(strategy, tier, cache_label, calls, latency_ms)
        return {
            "text": final.text,
            "model": final.model,
            "tier": tier,
            "route": route,
            "cache": cache_info,
            "usage": {
                "input_tokens": sum(c.usage.input_tokens for _, c in calls),
                "output_tokens": sum(c.usage.output_tokens for _, c in calls),
                "calls": [_call_usage(t, c) for t, c in calls],
            },
            "cost_usd": round(cost, 8),
            "latency_ms": latency_ms,
            "stop_reason": final.stop_reason,
            "refused": final.refused,
        }

    def _lookup(self, ns: str, prompt: str, vector: Any, cache_info: dict[str, Any]) -> Any:
        assert self.cache is not None
        try:
            nearest = self.cache.nearest(ns, vector)
        except Exception as exc:
            log.warning("cache lookup failed: %s", exc)
            self.metrics.errors.labels("cache_read").inc()
            cache_info["error"] = "lookup_failed"
            return None
        if nearest is not None:
            cache_info["similarity"] = round(nearest.similarity, 4)
        if nearest is None or nearest.similarity < self.settings.cache_threshold:
            self.metrics.cache_lookup("miss")
            self._count("record_miss")
            return None
        veto = self.guard.check(prompt, nearest.prompt) if self.guard else None
        if veto is not None:
            cache_info["vetoed"] = veto
            self.metrics.cache_lookup("vetoed")
            self.metrics.cache_vetoed.labels(veto.split(":")[0]).inc()
            self._count("record_miss")
            return None
        try:
            cache_info["hits"] = self.cache.record_hit(nearest.key)
        except Exception as exc:
            log.warning("cache hit counter failed: %s", exc)
        self.metrics.cache_lookup("hit")
        return nearest

    def _count(self, method: str) -> None:
        fn = getattr(self.cache, method, None)
        if fn is not None:
            try:
                fn()
            except Exception as exc:
                log.warning("cache counter failed: %s", exc)

    def _hit_response(
        self, hit: Any, strategy: str, cache_info: dict[str, Any], started: float
    ) -> dict[str, Any]:
        cache_info.update(hit=True, saved_cost_usd=round(hit.cost_usd, 8), cached_by=hit.strategy)
        threshold = self.settings.cache_threshold
        self.metrics.saved.inc(hit.cost_usd)
        latency_ms = int((time.perf_counter() - started) * 1000)
        self._observe(strategy, hit.tier, "hit", [], latency_ms)
        return {
            "text": hit.text,
            "model": hit.model,
            "tier": hit.tier,
            "route": {
                "strategy": strategy,
                "reason": f"semantic cache hit (similarity {hit.similarity:.3f} >= {threshold})",
                "score": None,
            },
            "cache": cache_info,
            "usage": {"input_tokens": 0, "output_tokens": 0, "calls": []},
            "cost_usd": 0.0,
            "latency_ms": latency_ms,
            "stop_reason": "end_turn",
            "refused": False,
        }

    def _observe(
        self, strategy: str, tier: str, cache: str, calls: list[tuple[str, Completion]], latency_ms: int
    ) -> None:
        m = self.metrics
        m.requests.labels(strategy, tier, cache).inc()
        m.latency.labels(cache).observe(latency_ms / 1000)
        for _, c in calls:
            m.tokens.labels(c.model, "input").inc(c.usage.input_tokens)
            m.tokens.labels(c.model, "output").inc(c.usage.output_tokens)
            m.cost.labels(c.model).inc(c.cost_usd)

    def health(self) -> dict[str, Any]:
        cache: dict[str, Any] = {
            "enabled": self.settings.cache_enabled,
            "threshold": self.settings.cache_threshold,
        }
        if self.cache is not None:
            try:
                cache.update(self.cache.stats())
            except Exception as exc:
                cache["error"] = str(exc)
        return {
            "status": "ok",
            "models": self.models,
            "default_strategy": self.settings.default_strategy,
            "strategies": [s for s in STRATEGIES if s != "knn" or self.knn is not None],
            "embedding_model": self.embedder.name,
            "knn_examples": len(self.knn.examples) if self.knn else 0,
            "cache": cache,
        }


def build_gateway(settings: Settings | None = None) -> Gateway:
    settings = settings or Settings()
    embedder = create_embedder(settings.embedding_model, cache_dir=settings.embedding_cache_dir)
    if settings.claude_backend == "stub":
        from .stub import StubAnthropic

        client: Any = StubAnthropic()
        log.warning("Claude backend is the offline stub: answers are placeholders")
    else:
        import anthropic

        client = anthropic.Anthropic()
    cache: SemanticCache | None = None
    if settings.cache_enabled:
        try:
            cache = RedisSemanticCache.from_url(
                settings.redis_url, embedder.dim, embedder.name, settings.cache_ttl_seconds
            )
        except Exception as exc:
            log.warning("semantic cache disabled, Redis unavailable at %s: %s", settings.redis_url, exc)
    knn = None
    if settings.knn_examples.exists():
        from .data import load_examples

        knn = KnnRouter(embedder, load_examples(settings.knn_examples), settings.knn_k, settings.knn_margin)
    else:
        log.warning("kNN routing disabled: %s not found", settings.knn_examples)
    return Gateway(settings=settings, claude=ClaudeClient(client), embedder=embedder, cache=cache, knn=knn)
