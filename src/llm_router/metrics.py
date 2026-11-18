"""Prometheus metrics. Each gateway owns its registry, so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)


class Metrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "llm_router_requests", "Chat requests served", ["strategy", "tier", "cache"], registry=r
        )
        self.tokens = Counter("llm_router_tokens", "Claude tokens", ["model", "kind"], registry=r)
        self.cost = Counter("llm_router_cost_usd", "Claude spend in USD", ["model"], registry=r)
        self.saved = Counter("llm_router_cache_saved_usd", "Spend avoided by cache hits (USD)", registry=r)
        self.latency = Histogram(
            "llm_router_request_latency_seconds",
            "End-to-end /v1/chat latency",
            ["cache"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.cache_lookups = Counter("llm_router_cache_lookups", "Cache lookups", ["result"], registry=r)
        self.cache_skipped = Counter(
            "llm_router_cache_skipped", "Requests not cached", ["reason"], registry=r
        )
        self.cache_vetoed = Counter(
            "llm_router_cache_guard_vetoes",
            "Hits above threshold vetoed by the guard",
            ["reason"],
            registry=r,
        )
        self.hit_ratio = Gauge("llm_router_cache_hit_ratio", "Cache hits / lookups since start", registry=r)
        self.escalations = Counter("llm_router_escalations", "Cascade escalations", ["reason"], registry=r)
        self.errors = Counter("llm_router_errors", "Failed requests", ["kind"], registry=r)
        self._hits = 0
        self._lookups = 0

    def cache_lookup(self, result: str) -> None:
        """result: hit | miss | vetoed"""
        self.cache_lookups.labels(result).inc()
        self._lookups += 1
        self._hits += result == "hit"
        self.hit_ratio.set(self._hits / self._lookups)

    def render(self) -> bytes:
        return generate_latest(self.registry)
