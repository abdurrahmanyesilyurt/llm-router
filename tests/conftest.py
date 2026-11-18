from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

from llm_router.cache import CachedAnswer, prompt_id
from llm_router.claude import ClaudeClient
from llm_router.config import Settings
from llm_router.embeddings import HashingEmbedder
from llm_router.gateway import Gateway
from llm_router.knn import Example, KnnRouter

REDIS_IMAGE = "redis:8"


# -- Redis (integration tests) ------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """REDIS_URL if set (CI service container), otherwise a throwaway Testcontainers Redis 8."""
    if url := os.environ.get("REDIS_URL"):
        yield url
        return
    try:
        from testcontainers.community.redis import RedisContainer

        container = RedisContainer(REDIS_IMAGE)
        container.start()
    except Exception as exc:  # Docker not available
        pytest.skip(f"Redis 8 is not available: {exc}")
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"
    finally:
        container.stop()


# -- fakes ------------------------------------------------------------------------------


def response(
    text: str = "answer",
    model: str = "claude-haiku-5-5",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
    thinking: bool = True,
) -> Any:
    blocks = [NS(type="thinking", thinking="")] if thinking else []
    blocks.append(NS(type="text", text=text))
    usage = NS(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return NS(content=blocks, model=model, stop_reason=stop_reason, usage=usage)


def structured(answer: str, confidence: float, needs_stronger_model: bool = False, **kw: Any) -> Any:
    payload = {"answer": answer, "confidence": confidence, "needs_stronger_model": needs_stronger_model}
    return response(json.dumps(payload), **kw)


class _Endpoint:
    def __init__(self, owner: FakeAnthropic, beta: bool) -> None:
        self.owner, self.beta = owner, beta

    def create(self, **request: Any) -> Any:
        self.owner.requests.append({**request, "_endpoint": "beta" if self.beta else "messages"})
        return self.owner.responder(request)


class FakeAnthropic:
    """Records requests; `responder(request)` returns an SDK-shaped response or raises."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responder = responder or (
            lambda req: response(f"answer from {req['model']}", model=req["model"])
        )
        self.messages = _Endpoint(self, beta=False)
        self.beta = NS(messages=_Endpoint(self, beta=True))


class FakeCache:
    """In-memory stand-in for RedisSemanticCache (same nearest/store/record_hit surface)."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.misses = 0

    def nearest(self, ns: str, vector: np.ndarray) -> CachedAnswer | None:
        best = None
        for key, e in self.entries.items():
            if e["ns"] != ns:
                continue
            sim = float(np.dot(e["vector"], vector))
            if best is None or sim > best[0]:
                best = (sim, key, e)
        if best is None:
            return None
        sim, key, e = best
        return CachedAnswer(
            key,
            sim,
            e["prompt"],
            e["text"],
            e["model"],
            e["tier"],
            e["strategy"],
            e["cost_usd"],
            e["hits"],
            0.0,
        )

    def store(self, ns: str, vector: np.ndarray, prompt: str, answer: dict[str, Any]) -> str:
        key = f"{ns}:{prompt_id(prompt)}"
        self.entries[key] = {"ns": ns, "vector": np.asarray(vector), "prompt": prompt, "hits": 0, **answer}
        return key

    def record_hit(self, key: str) -> int:
        self.entries[key]["hits"] += 1
        return self.entries[key]["hits"]

    def record_miss(self) -> None:
        self.misses += 1

    def stats(self) -> dict[str, Any]:
        return {"backend": "fake", "entries": len(self.entries)}


EXAMPLES = [
    Example("e1", "Şu cümleyi İngilizceye çevir: Merhaba dünya", "easy"),
    Example("e2", "Translate to French: good morning", "easy"),
    Example("e3", "Bu metni özetle: toplantı yarın", "easy"),
    Example("e4", "Prove that there are infinitely many primes", "hard"),
    Example("e5", "Design a distributed rate limiter and compare token bucket with sliding window", "hard"),
    Example("e6", "Bu Go kodunda goroutine sızıntısı neden oluyor, adım adım analiz et", "hard"),
]


def make_gateway(
    client: FakeAnthropic | None = None,
    cache: Any = "fake",
    with_knn: bool = True,
    embedder: HashingEmbedder | None = None,
    **overrides: Any,
) -> Gateway:
    settings = Settings(_env_file=None, **overrides)
    embedder = embedder or HashingEmbedder(256)
    knn = KnnRouter(embedder, EXAMPLES, k=3, margin=0.0) if with_knn else None
    return Gateway(
        settings=settings,
        claude=ClaudeClient(client or FakeAnthropic()),
        embedder=embedder,
        cache=FakeCache() if cache == "fake" else cache,
        knn=knn,
    )


@pytest.fixture
def fake_client() -> FakeAnthropic:
    return FakeAnthropic()
