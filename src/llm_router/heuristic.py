"""Transparent rule-based difficulty score.

Every feature has a fixed weight; the score is their sum and the prompt goes to the large
tier when the score reaches the threshold. The features that fired are returned, so every
routing decision can be explained ("code(+1.0) reasoning:neden,karşılaştır(+2.0) = 3.0").

Term syntax: ``word`` matches a whole token, ``word*`` a token prefix (Turkish suffixes:
"hata*" matches "hatası"), and a term with a space or punctuation matches as a substring.
The weights were set by looking only at the train split of ``eval/prompts.jsonl``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import tr_lower, words

STRONG_REASONING = [
    "kanıtla*", "ispatla*", "ispat*", "prove", "proof", "derive", "türet*", "adım adım", "step by step",
    "step-by-step", "trade-off*", "tradeoff*", "ödünleşim*", "root cause*", "kök neden*", "gerekçelendir*",
    "justify",
]  # fmt: skip
REASONING = [
    "neden", "niçin", "niye", "why", "karşılaştır*", "compare", "comparison", "analiz et*", "analyze",
    "analyse", "değerlendir*", "evaluate", "tasarla*", "design", "optimize*", "optimise*", "strateji*",
    "strategy", "hangisi daha", "which is better", "should i", "should we", "should our", "pros and cons",
    "artıları ve eksileri", "öner", "önerir misin", "recommend", "hipotez*", "hypothes*", "cause*",
    "walk me through", "teşhis*", "diagnose", "yol harita*", "roadmap", "risk*", "implications",
]  # fmt: skip
TECHNICAL = [
    "race condition", "deadlock", "concurrency", "eşzamanlı*", "mimari*", "architecture", "ölçekle*",
    "scalab*", "latency", "throughput", "consistency", "tutarlılı*", "transaction*", "indeks*", "index*",
    "complexity", "karmaşıklı*", "microservice*", "mikroservis*", "distributed", "dağıtık*", "idempoten*",
    "sharding", "replication", "kubernetes", "postgres*", "kafka", "clickhouse", "timescale*", "redis",
    "elasticsearch", "opensearch", "grpc", "graphql", "jvm", "ef core", "thread pool", "goroutine*",
    "pprof", "heap", "pg_wal", "wal", "saga", "two-phase", "iki aşamalı", "rate limit*", "event sourcing",
    "outbox", "webhook*", "strangler", "monolit*", "monolith*", "token*", "kimlik doğrulama",
    "authentication", "saldırı*", "attack*", "güvenlik", "security", "multi-tenant", "çok kiracılı",
    "kiracı*", "tenant*",
    "vector database", "availability", "failure scenario*", "hata senaryo*", "out-of-order",
]  # fmt: skip
DEBUG = [
    "hata*", "error", "exception", "traceback", "bug", "bugs", "çalışmıyor", "doesn't work", "does not work",
    "crash*", "çöküyor", "stack trace", "segfault", "null pointer", "fails", "failing", "yanlış sonuç",
    "wrong result", "timeout", "zaman aşımı", "leak*", "sızıntı*", "oom*", "asılı kal*", "hangs", "flaky",
    "intermittent", "silently", "sapıyor", "sapma*", "inflated", "sorun ne", "what's wrong", "duplicate",
    "beklenmedik", "unexpected", "eksiye düşüyor", "bazen", "sometimes", "bellek kullanımı", "memory usage",
]  # fmt: skip
MATH = [
    "integral*", "türev*", "derivative", "olasılı*", "probability", "denklem*", "equation*", "kombinasyon*",
    "permütasyon*", "combinator*", "kaç farklı", "how many ways", "kaç yolla", "faiz*", "interest rate",
    "geometri*", "üçgen*", "triangle", "expected value", "expected number", "beklenen değer", "matris*",
    "matrix", "asal", "prime", "modulo", "logaritma*", "logarithm", "istatistik*", "statistic*",
    "anlamlı mı", "significant", "p-value", "growth rate", "cagr", "compound", "bileşik", "ekstremum*",
    "maksimum değer*", "formül*", "formula", "taksit*", "olasılığı*", "sensitivity", "specificity",
]  # fmt: skip
LEGAL = [
    "kvkk", "gdpr", "kanun*", "yönetmelik*", "mevzuat*", "madde*", "tbk", "tck", "sözleşme*", "contract*",
    "liability", "regulation*", "compliance", "uyum*", "dava*", "lawsuit", "tazminat*", "hukuk*", "legal",
    "yasal", "fesih", "feshet*", "termination", "ihbar", "kıdem", "vergi*", "tax", "licen*", "lisans*",
    "telif*", "copyright", "ai act", "yapay zeka yasası", "açık rıza", "rıza*", "consent", "aydınlatma",
    "obligation*", "yükümlülü*", "enforceab*", "non-compete", "data processing agreement",
    "transfer mechanism*", "aktarım mekanizma*", "labour law", "labor law", "social security",
    "permanent establishment",
    "terms-of-service", "terms of service", "data-mining", "data mining",
]  # fmt: skip
EASY_TASK = [
    "çevir*", "translate", "tercüme*", "özet*", "summarize", "summarise", "summary", "tl;dr",
    "yeniden yaz", "rewrite", "rephrase", "paraphrase", "kısalt*", "shorten", "düzelt", "imla*", "grammar",
    "dilbilgisi*", "extract", "ayıkla", "listele", "list every", "list the", "json olarak", "as json",
    "başlık öner*", "suggest a title", "daha resmi", "more professional", "more formal", "kibar*", "politer",
    "sınıflandır*", "classify", "sentiment", "docstring", "rename", "commit message", "suggest five",
    "çıkar:", "json formatında", "into a table", "ingilizceye", "türkçeye", "almancaya",
]  # fmt: skip
FACTUAL_OPENERS = [
    "nedir", "kimdir", "ne zaman", "nerede", "nerededir", "kaç yılında", "hangi yıl", "what is", "who is",
    "who wrote", "who was", "when did", "when was", "where is", "what does", "başkenti", "capital of",
    "ne demek", "what's", "kaç", "how do i", "nasıl",
]  # fmt: skip

_CODE_FENCE = re.compile(r"```|^\s{4,}\S", re.MULTILINE)
_CODE_HINT = re.compile(
    r"\bdef \w+\(|\bclass \w+|\bfunction \w*\(|=>|;\s*$|\bSELECT\b.+\bFROM\b|\bimport \w+|\bpublic \w+|"
    r"\bconst \w+ =|\bvar \w+ =|\breturn\b.*;|#include|\bfor \(|\bwhile \(",
    re.MULTILINE | re.IGNORECASE,
)
_MATH_EXPR = re.compile(r"\d\s*[+*/^×÷=]\s*\d|\d\s+-\s+\d|\\frac|\\int|\\sum|[a-z]\^\d|√|∑|∫")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•]|[a-z]\))\s+\S", re.MULTILINE)
_MULTI_CONNECTORS = [
    "ayrıca",
    "bunun yanında",
    "additionally",
    "also explain",
    "and then",
    "sonra da",
    "hem de",
]
_TR_ALTERNATIVES = re.compile(r"\bm[ıiuü]\b[^?.]*\b(?:m[ıiuü]|yoksa)\b")
_EN_ALTERNATIVES = re.compile(r"\b(?:should (?:we|i|our)|which)\b[^?.]*\bor\b|\bvs\.?\b|\bversus\b")


def _matches(text: str, tokens: list[str], terms: list[str]) -> list[str]:
    token_set = set(tokens)
    hits = []
    for term in terms:
        stem = term.removesuffix("*")
        if not stem.isalpha():  # several words or punctuation: substring of the text
            hit = stem in text
        elif term.endswith("*"):
            hit = any(t.startswith(stem) for t in tokens)
        else:
            hit = stem in token_set
        if hit:
            hits.append(stem)
    return hits


@dataclass
class HeuristicScore:
    score: float
    features: dict[str, float] = field(default_factory=dict)
    details: dict[str, list[str]] = field(default_factory=dict)

    def explain(self) -> str:
        if not self.features:
            return "no difficulty features"
        parts = []
        for name, weight in self.features.items():
            detail = self.details.get(name)
            label = f"{name}:{','.join(detail[:3])}" if detail else name
            parts.append(f"{label}({weight:+.1f})")
        return " ".join(parts)


class HeuristicRouter:
    def __init__(self, threshold: float = 1.5) -> None:
        self.threshold = threshold

    def score(self, prompt: str) -> HeuristicScore:
        text = tr_lower(prompt)
        tokens = words(prompt)
        n_words = len(tokens)
        result = HeuristicScore(0.0)

        def add(name: str, weight: float, detail: list[str] | None = None) -> None:
            result.features[name] = weight
            if detail:
                result.details[name] = detail
            result.score += weight

        if n_words >= 300:
            add("length", 1.0, [f"{n_words}w"])
        elif n_words >= 120:
            add("length", 0.5, [f"{n_words}w"])

        has_code = bool(_CODE_FENCE.search(prompt) or len(_CODE_HINT.findall(prompt)) >= 2)
        if has_code:
            add("code", 1.0)
        debug = _matches(text, tokens, DEBUG)
        if debug:
            add("debugging", 1.0, debug)

        math_terms = _matches(text, tokens, MATH)
        if _MATH_EXPR.search(prompt) or math_terms:
            add("math", 1.5, math_terms or ["expression"])
        n_numbers = len(_NUMBER.findall(prompt))
        if n_numbers >= 3 and not has_code:
            add("numbers", 0.5, [f"{n_numbers} numbers"])

        parts = prompt.count("?") + len(_LIST_ITEM.findall(prompt))
        connectors = _matches(text, tokens, _MULTI_CONNECTORS)
        if parts >= 3 or (parts >= 2 and connectors):
            add("multi_part", 1.0, [f"{parts} parts"])
        elif parts >= 2 or connectors:
            add("multi_part", 0.5, [f"{parts} parts", *connectors])
        if _TR_ALTERNATIVES.search(text) or _EN_ALTERNATIVES.search(text):
            add("alternatives", 1.0)

        strong = _matches(text, tokens, STRONG_REASONING)
        reasoning = _matches(text, tokens, REASONING)
        if strong or reasoning:
            add("reasoning", min(2.5, 2.0 * bool(strong) + 1.0 * len(reasoning)), strong + reasoning)
        if technical := _matches(text, tokens, TECHNICAL):
            add("technical", min(1.5, 0.75 * len(technical)), technical)
        legal = _matches(text, tokens, LEGAL)
        if legal:
            add("legal", min(2.0, 1.5 + 0.5 * (len(legal) - 1)), legal)

        if easy := _matches(text, tokens, EASY_TASK):
            hard_signal = has_code or debug or legal or "math" in result.features
            add("simple_task", -0.5 if hard_signal else -1.5, easy)
        if n_words <= 15 and (factual := _matches(text, tokens, FACTUAL_OPENERS)):
            add("short_factual", -0.5, factual)

        result.score = round(result.score, 2)
        return result

    def route(self, prompt: str) -> tuple[str, float, str]:
        s = self.score(prompt)
        tier = "large" if s.score >= self.threshold else "small"
        op = ">=" if tier == "large" else "<"
        return tier, s.score, f"{s.explain()} = {s.score:.1f} {op} {self.threshold:.1f}"
