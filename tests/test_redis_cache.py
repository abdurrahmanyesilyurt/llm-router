"""Integration tests against a real Redis 8 (Query Engine vector search)."""

import numpy as np
import pytest
import redis
from conftest import FakeAnthropic, make_gateway

from llm_router.cache import RedisSemanticCache, namespace
from llm_router.embeddings import HashingEmbedder
from llm_router.text import normalise

pytestmark = pytest.mark.redis

EMB = HashingEmbedder(64)


def vec(text):
    return EMB.embed([normalise(text)])[0]


@pytest.fixture
def cache(redis_url):
    c = RedisSemanticCache(
        redis.Redis.from_url(redis_url, decode_responses=True), EMB.dim, f"test-{EMB.name}", ttl_seconds=120
    )
    c.drop()
    c.ensure_index()
    yield c
    c.drop()


ANSWER = {
    "text": "Ankara",
    "model": "claude-haiku-5-5",
    "tier": "small",
    "strategy": "heuristic",
    "cost_usd": 0.0001,
}


def test_query_engine_is_available(redis_url):
    r = redis.Redis.from_url(redis_url)
    assert isinstance(r.execute_command("FT._LIST"), list)


def test_store_and_nearest_within_namespace(cache):
    ns = namespace(None, [])
    key = cache.store(ns, vec("Türkiye'nin başkenti neresi?"), "Türkiye'nin başkenti neresi?", ANSWER)
    hit = cache.nearest(ns, vec("türkiye'nin başkenti neresi"))
    assert hit is not None and hit.key == key
    assert hit.similarity == pytest.approx(1.0, abs=1e-4)
    assert (hit.text, hit.model, hit.tier, hit.cost_usd) == ("Ankara", "claude-haiku-5-5", "small", 0.0001)
    other = cache.nearest(ns, vec("Fransa'nın başkenti neresi?"))
    assert other is not None and other.similarity < hit.similarity


def test_namespaces_never_cross(cache):
    ns_a = namespace("Sen bir coğrafya öğretmenisin.", [])
    ns_b = namespace("Sen bir şair olarak cevap ver.", [])
    cache.store(ns_a, vec("Başkent neresi?"), "Başkent neresi?", ANSWER)
    assert cache.nearest(ns_b, vec("Başkent neresi?")) is None
    assert cache.nearest(ns_a, vec("Başkent neresi?")) is not None


def test_ttl_hits_and_stats(cache):
    ns = namespace(None, [])
    key = cache.store(ns, vec("REST API nedir?"), "REST API nedir?", ANSWER)
    assert 0 < cache.ttl(key) <= 120
    assert cache.record_hit(key) == 1
    assert cache.record_hit(key) == 2
    cache.record_miss()
    stats = cache.stats()
    assert (
        stats["entries"] == 1 and stats["hits"] == 2 and stats["misses"] == 1 and stats["hit_ratio"] == 0.6667
    )
    assert cache.nearest(ns, vec("REST API nedir?")).hits == 2


def test_same_prompt_overwrites_instead_of_duplicating(cache):
    ns = namespace(None, [])
    cache.store(ns, vec("REST API nedir?"), "REST API nedir?", ANSWER)
    cache.store(ns, vec("rest api nedir"), "rest api nedir", {**ANSWER, "text": "newer"})
    assert cache.stats()["entries"] == 1
    assert cache.nearest(ns, vec("REST API nedir?")).text == "newer"


def test_ensure_index_is_idempotent(cache):
    cache.ensure_index()
    cache.ensure_index()
    assert cache.r.ft(cache.index).info()["num_docs"] in (0, "0")


def test_gateway_end_to_end_on_redis(cache):
    fake = FakeAnthropic()
    gw = make_gateway(fake, cache=cache, embedder=EMB)
    first = gw.chat(messages=[{"role": "user", "content": "Python'da liste nasıl ters çevrilir?"}])
    second = gw.chat(messages=[{"role": "user", "content": "python'da liste nasıl ters çevrilir"}])
    vetoed = gw.chat(messages=[{"role": "user", "content": "Java'da liste nasıl ters çevrilir?"}])
    assert first["cache"]["stored"] and not first["cache"]["hit"]
    assert second["cache"]["hit"] and second["cost_usd"] == 0.0 and second["cache"]["hits"] == 1
    assert not vetoed["cache"]["hit"]
    assert len(fake.requests) == 2
    assert np.isclose(second["cache"]["similarity"], 1.0, atol=1e-4)
