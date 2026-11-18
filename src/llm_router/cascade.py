"""Cascade routing: ask the small model first, escalate when it is not sure.

The small model answers with structured output (``output_config.format`` JSON schema):
the answer, a self-reported confidence and an explicit ``needs_stronger_model`` flag.
The request escalates to the large model when the flag is set, the confidence is below
the threshold, or the small call did not produce a usable answer (refusal, truncation,
invalid JSON). An escalated request pays for both calls; both usages are returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .claude import ClaudeClient, Completion

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "description": "The complete answer to the user's last message."},
        "confidence": {
            "type": "number",
            "description": "From 0 to 1: how likely the answer is correct and complete.",
        },
        "needs_stronger_model": {
            "type": "boolean",
            "description": "True if the task needs deeper reasoning or expertise than you can reliably give.",
        },
    },
    "required": ["answer", "confidence", "needs_stronger_model"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You are the first, fast tier of a two-tier assistant. Answer the user's last \
message and assess your own answer.

- Put your complete answer, written for the user, in "answer" (same language as the user).
- Set "confidence" between 0 and 1 for how likely the answer is correct and complete.
- Set "needs_stronger_model" to true when the task needs careful multi-step reasoning, a proof, \
non-trivial debugging or design trade-offs, legal or regulatory analysis, or exact calculations you \
cannot double-check. A stronger model will then answer instead of you.
- Be calibrated: escalating costs a little; a confident wrong answer reaches the user."""

JSON_HEADROOM_TOKENS = 256  # room for the JSON wrapper around the answer


@dataclass
class Verdict:
    escalate: bool
    reason: str
    kind: str  # accepted | refused | truncated | invalid_output | empty | flag | low_confidence
    confidence: float | None = None
    answer: str | None = None


@dataclass
class CascadeResult:
    final: Completion
    calls: list[tuple[str, Completion]] = field(default_factory=list)  # (tier, completion)
    verdict: Verdict | None = None

    @property
    def escalated(self) -> bool:
        return bool(self.verdict and self.verdict.escalate)


def judge(small: Completion, threshold: float) -> Verdict:
    """Decide from the small model's structured reply whether to escalate."""
    if small.refused:
        return Verdict(True, "small model refused", "refused")
    if small.stop_reason == "max_tokens":
        return Verdict(True, "small answer truncated (max_tokens)", "truncated")
    data = small.json()
    if data is None or not isinstance(data.get("answer"), str):
        return Verdict(True, "small model returned no valid structured answer", "invalid_output")
    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    answer = data["answer"].strip()
    if not answer:
        return Verdict(True, "small model returned an empty answer", "empty", confidence)
    if data.get("needs_stronger_model") is True:
        reason = f"small model asked for a stronger model (confidence {confidence:.2f})"
        return Verdict(True, reason, "flag", confidence)
    if confidence < threshold:
        return Verdict(True, f"confidence {confidence:.2f} < {threshold:.2f}", "low_confidence", confidence)
    reason = f"small model confident ({confidence:.2f} >= {threshold:.2f})"
    return Verdict(False, reason, "accepted", confidence, answer)


def cascade_system(system: str | None) -> str:
    return f"{system.strip()}\n\n{INSTRUCTIONS}" if system and system.strip() else INSTRUCTIONS


class Cascade:
    def __init__(
        self,
        claude: ClaudeClient,
        small_model: str,
        large_model: str,
        threshold: float = 0.7,
        small_effort: str | None = None,
        large_effort: str | None = None,
    ) -> None:
        self.claude = claude
        self.small_model = small_model
        self.large_model = large_model
        self.threshold = threshold
        self.small_effort = small_effort
        self.large_effort = large_effort

    def run(self, system: str | None, messages: list[dict[str, Any]], max_tokens: int) -> CascadeResult:
        small = self.claude.complete(
            model=self.small_model,
            messages=messages,
            system=cascade_system(system),
            max_tokens=max_tokens + JSON_HEADROOM_TOKENS,
            effort=self.small_effort,
            json_schema=SCHEMA,
        )
        verdict = judge(small, self.threshold)
        if not verdict.escalate:
            final = replace(small, text=verdict.answer or "")
            return CascadeResult(final=final, calls=[("small", small)], verdict=verdict)
        large = self.claude.complete(
            model=self.large_model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            effort=self.large_effort,
        )
        return CascadeResult(final=large, calls=[("small", small), ("large", large)], verdict=verdict)
