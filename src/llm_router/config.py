from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Strategy = Literal["heuristic", "knn", "cascade", "always-small", "always-large"]
STRATEGIES: tuple[str, ...] = ("heuristic", "knn", "cascade", "always-small", "always-large")


class Settings(BaseSettings):
    """Configuration from LLM_ROUTER_* environment variables (or a .env file).

    The Anthropic SDK reads ANTHROPIC_API_KEY itself; it is never stored here.
    """

    model_config = SettingsConfigDict(env_prefix="LLM_ROUTER_", env_file=".env", extra="ignore")

    # Claude
    small_model: str = "claude-haiku-5-5"
    large_model: str = "claude-sonnet-5-5"
    small_effort: str | None = None  # low | medium | high | xhigh | max; None = model default
    large_effort: str | None = None
    default_max_tokens: int = 4096  # thinking counts towards max_tokens; truncated answers are not cached
    claude_backend: Literal["anthropic", "stub"] = "anthropic"  # stub = offline canned answers

    # Routing
    default_strategy: Strategy = "heuristic"
    heuristic_threshold: float = 1.5  # feature score at or above this goes to the large tier (tuned on train)
    knn_k: int = 11  # k and margin chosen by leave-one-out on the train split (llm-router eval-router)
    knn_margin: float = 0.0  # |P(hard) - 0.5| below this is "unsure" and goes to the large tier
    knn_examples: Path = Path("eval/prompts.jsonl")  # the train split is used
    cascade_threshold: float = 0.7  # small-model confidence below this escalates

    # Semantic cache
    redis_url: str = "redis://localhost:56379/0"
    cache_enabled: bool = True
    cache_threshold: float = (
        0.88  # cosine similarity; recommended by `llm-router eval-cache` (with the guard)
    )
    cache_ttl_seconds: int = 24 * 3600
    cache_guard: bool = True  # lexical checks (numbers, negation, polarity, entities) on every hit
    cache_exact_over_chars: int = 400  # longer prompts exceed the embedding window: exact match only

    # Embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_cache_dir: str = ".cache/fastembed"
