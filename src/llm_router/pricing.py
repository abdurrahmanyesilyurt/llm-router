"""The one price table. USD per million tokens, Claude API first-party rates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    # Some models switch to a second rate card when the prompt is long.
    long_prompt_tokens: int | None = None
    long_input: float | None = None
    long_output: float | None = None


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, long_prompt_tokens=100_000, long_input=0.50, long_output=2.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
}

# Prompt-caching multipliers on the input rate (5-minute writes, reads).
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    price = PRICES.get(model)
    if price is None:
        raise KeyError(f"No price for model {model!r}; add it to llm_router.pricing.PRICES")
    rate_in, rate_out = price.input, price.output
    prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    if price.long_prompt_tokens and prompt_tokens > price.long_prompt_tokens:
        rate_in, rate_out = price.long_input or rate_in, price.long_output or rate_out
    total = (
        input_tokens * rate_in
        + cache_creation_input_tokens * rate_in * CACHE_WRITE_MULTIPLIER
        + cache_read_input_tokens * rate_in * CACHE_READ_MULTIPLIER
        + output_tokens * rate_out
    )
    return total / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token count for planning only (no API call): ~3.5 characters per token.

    Turkish agglutination makes tokens shorter than in English, so this sits between the
    usual 4 chars/token for English and ~3 for Turkish.
    """
    return max(1, round(len(text) / 3.5))
