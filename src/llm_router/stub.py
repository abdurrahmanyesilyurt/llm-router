"""An offline stand-in for ``anthropic.Anthropic`` (LLM_ROUTER_CLAUDE_BACKEND=stub).

It returns canned answers with token counts estimated from the text, so the gateway,
cache, metrics and dashboards can be exercised end to end without an API key. Nothing
it returns is a model output; the text says so.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any

from .pricing import estimate_tokens


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    return " ".join(str(m.get("content", "")) for m in messages)


class _Messages:
    def __init__(self, log: list[dict[str, Any]]) -> None:
        self._log = log

    def create(self, **request: Any) -> Any:
        self._log.append(request)
        prompt = _prompt_text(request["messages"])
        fmt = (request.get("output_config") or {}).get("format")
        answer = f"[stub {request['model']}] This is an offline placeholder answer, not a model output."
        if fmt:
            confident = len(prompt) < 400
            text = json.dumps(
                {
                    "answer": answer,
                    "confidence": 0.9 if confident else 0.4,
                    "needs_stronger_model": not confident,
                }
            )
        else:
            text = answer
        usage = NS(
            input_tokens=estimate_tokens(prompt + (request.get("system") or "")),
            output_tokens=min(request["max_tokens"], estimate_tokens(text) + 40),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return NS(
            model=request["model"],
            stop_reason="end_turn",
            content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
            usage=usage,
        )


class StubAnthropic:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.messages = _Messages(self.requests)
        self.beta = NS(messages=_Messages(self.requests))
