"""Semantic cache on Redis 8 (built-in Query Engine, HNSW vector index).

Entries are hashes ``<prefix><namespace>:<sha of normalised prompt>`` holding the answer,
the prompt it answered, a hit counter and the embedding. A lookup is a KNN 1 query
restricted to one namespace by a TAG filter, so a question asked under a different system
prompt or after different earlier turns can never hit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .text import normalise

FIELDS = ("prompt", "text", "model", "tier", "strategy", "cost_usd", "hits", "created_at")


def namespace(system: str | None, history: Sequence[Mapping[str, str]]) -> str:
    """Hash of everything that conditions the answer besides the last user message."""
    payload = json.dumps(
        {"system": (system or "").strip(), "history": [[m["role"], m["content"]] for m in history]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(normalise(prompt).encode("utf-8")).hexdigest()[:16]


@dataclass
class CachedAnswer:
    key: str
    similarity: float
    prompt: str
    text: str
    model: str
    tier: str
    strategy: str
    cost_usd: float
    hits: int
    created_at: float


class SemanticCache(Protocol):
    def nearest(self, ns: str, vector: np.ndarray) -> CachedAnswer | None: ...

    def store(self, ns: str, vector: np.ndarray, prompt: str, answer: Mapping[str, Any]) -> str: ...

    def record_hit(self, key: str) -> int: ...

    def stats(self) -> dict[str, Any]: ...


class RedisSemanticCache:
    def __init__(
        self,
        client: Any,
        dim: int,
        embedder_name: str,
        ttl_seconds: int = 86400,
        prefix: str = "llmr",
    ) -> None:
        import redis  # noqa: F401  (fail early if the client library is missing)

        self.r = client
        self.dim = dim
        self.ttl_seconds = ttl_seconds
        slug = hashlib.sha1(f"{embedder_name}:{dim}".encode()).hexdigest()[:8]
        self.index = f"{prefix}:idx:{slug}"
        self.key_prefix = f"{prefix}:cache:{slug}:"
        self.stats_key = f"{prefix}:stats:{slug}"

    @classmethod
    def from_url(cls, url: str, dim: int, embedder_name: str, ttl_seconds: int = 86400) -> RedisSemanticCache:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)
        cache = cls(client, dim, embedder_name, ttl_seconds)
        cache.ensure_index()
        return cache

    def ensure_index(self) -> None:
        from redis.commands.search.field import NumericField, TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        from redis.exceptions import ResponseError

        try:
            self.r.ft(self.index).info()
            return
        except ResponseError:
            pass
        schema = (
            TagField("ns"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )
        definition = IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH)
        try:
            self.r.ft(self.index).create_index(schema, definition=definition)
        except ResponseError as exc:  # another worker created it first
            if "already exists" not in str(exc).lower():
                raise

    def nearest(self, ns: str, vector: np.ndarray) -> CachedAnswer | None:
        from redis.commands.search.query import Query

        query = (
            Query(f"(@ns:{{{ns}}})=>[KNN 1 @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("distance", *FIELDS)
            .paging(0, 1)
            .dialect(2)
        )
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        result = self.r.ft(self.index).search(query, query_params={"vec": blob})
        if not result.docs:
            return None
        doc = result.docs[0]
        return CachedAnswer(
            key=doc.id,
            similarity=1.0 - float(doc.distance),
            prompt=doc.prompt,
            text=doc.text,
            model=doc.model,
            tier=doc.tier,
            strategy=doc.strategy,
            cost_usd=float(doc.cost_usd),
            hits=int(doc.hits),
            created_at=float(doc.created_at),
        )

    def store(self, ns: str, vector: np.ndarray, prompt: str, answer: Mapping[str, Any]) -> str:
        key = f"{self.key_prefix}{ns}:{prompt_id(prompt)}"
        mapping = {
            "ns": ns,
            "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
            "prompt": prompt,
            "text": answer["text"],
            "model": answer["model"],
            "tier": answer["tier"],
            "strategy": answer["strategy"],
            "cost_usd": repr(float(answer["cost_usd"])),
            "hits": 0,
            "created_at": time.time(),
        }
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, self.ttl_seconds)
        pipe.hincrby(self.stats_key, "stores", 1)
        pipe.execute()
        return key

    def record_hit(self, key: str) -> int:
        pipe = self.r.pipeline(transaction=False)
        pipe.hincrby(key, "hits", 1)
        pipe.hincrby(self.stats_key, "hits", 1)
        hits, _ = pipe.execute()
        return int(hits)

    def record_miss(self) -> None:
        self.r.hincrby(self.stats_key, "misses", 1)

    def ttl(self, key: str) -> int:
        return int(self.r.ttl(key))

    def stats(self) -> dict[str, Any]:
        info = self.r.ft(self.index).info()
        counters = {k: int(v) for k, v in (self.r.hgetall(self.stats_key) or {}).items()}
        hits, misses = counters.get("hits", 0), counters.get("misses", 0)
        return {
            "backend": "redis",
            "index": self.index,
            "entries": int(info.get("num_docs", 0)),
            "hits": hits,
            "misses": misses,
            "stores": counters.get("stores", 0),
            "hit_ratio": round(hits / (hits + misses), 4) if hits + misses else None,
        }

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def drop(self) -> None:
        """Delete the index and every entry (used by the tests)."""
        from redis.exceptions import ResponseError

        with contextlib.suppress(ResponseError):
            self.r.ft(self.index).dropindex(delete_documents=True)
        self.r.delete(self.stats_key)
