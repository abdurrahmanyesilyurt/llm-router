"""The only place that talks to the Claude API.

``ClaudeClient`` wraps an injected ``anthropic.Anthropic`` (or any object with the same
``messages.create`` / ``beta.messages.create`` surface, such as the offline stub or a
test fake) and returns a ``Completion`` with text, usage and cost.

Rules this module enforces:

* no ``temperature`` / ``top_p`` / ``top_k``: current models reject them; depth is
  controlled with ``output_config.effort``;
* ``stop_reason == "refusal"`` is checked before reading content;
* only ``text`` blocks are read (thinking blocks come back with empty text);
* models that support the server-side refusal fallback get it (beta endpoint,
  ``fallbacks="default"``); the others, such as ``claude-haiku-5-5``, are called plainly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .pricing import PRICES, cost_usd

FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeNotConfigured(RuntimeError):
    """The SDK could not resolve credentials (no ANTHROPIC_API_KEY or profile)."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_sdk(cls, usage: Any) -> Usage:
        if usage is None:
            return cls()
        return cls(**{k: int(getattr(usage, k, 0) or 0) for k in cls.__dataclass_fields__})


@dataclass
class Completion:
    text: str
    model: str  # the model that produced the answer (a server-side fallback may differ)
    requested_model: str
    stop_reason: str | None
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    refused: bool = False

    @property
    def priced_model(self) -> str:
        return self.model if self.model in PRICES else self.requested_model

    @property
    def cost_usd(self) -> float:
        u = self.usage
        return cost_usd(
            self.priced_model,
            u.input_tokens,
            u.output_tokens,
            u.cache_creation_input_tokens,
            u.cache_read_input_tokens,
        )

    @property
    def finished(self) -> bool:
        """A complete, usable answer: safe to cache."""
        return self.stop_reason == "end_turn" and not self.refused and bool(self.text.strip())

    def json(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.text)
        except (json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost_usd"] = round(self.cost_usd, 8)
        return data


class ClaudeClient:
    def __init__(self, client: Any) -> None:
        self.client = client

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            request["system"] = system
        output_config: dict[str, Any] = {}
        if effort:
            output_config["effort"] = effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            request["output_config"] = output_config
        if model in FALLBACK_MODELS:
            # On a safety decline the API re-runs the request on a fallback model inside
            # the same call instead of returning an empty answer.
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def complete(self, **kwargs: Any) -> Completion:
        request = self.build_request(**kwargs)
        create = self.client.beta.messages.create if "betas" in request else self.client.messages.create
        started = time.perf_counter()
        try:
            response = create(**request)
        except TypeError as exc:  # raised by the SDK before sending when no credentials resolve
            raise ClaudeNotConfigured(str(exc)) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        return parse_response(request["model"], response, latency_ms)


def parse_response(requested_model: str, response: Any, latency_ms: int = 0) -> Completion:
    stop_reason = getattr(response, "stop_reason", None)
    completion = Completion(
        text="",
        model=getattr(response, "model", None) or requested_model,
        requested_model=requested_model,
        stop_reason=stop_reason,
        usage=Usage.from_sdk(getattr(response, "usage", None)),
        latency_ms=latency_ms,
    )
    if stop_reason == "refusal":
        completion.refused = True
        return completion
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    completion.text = "".join(parts).strip()
    return completion
