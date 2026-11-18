"""Turkish-aware text helpers shared by the cache, the router and the evaluation."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\s\"'“”‘’«»(]+|[\s\"'“”‘’«».,;:!?…)]+$")


_TR_LETTERS = frozenset("çğıöşüÇĞİÖŞÜ")


def has_turkish_letters(text: str) -> bool:
    return not _TR_LETTERS.isdisjoint(text)


def tr_lower(text: str, turkish: bool | None = None) -> str:
    """Lowercase with Turkish dotted/dotless i rules when the text is Turkish.

    Python's str.lower() turns "İ" into "i̇" (i + combining dot) and "I" into "i", which
    breaks Turkish words such as "IŞIK" or "İSTANBUL". The Turkish mapping (I -> ı) is
    only applied when the text contains Turkish letters, so English "AI" or "Is" stay
    "ai" and "is" (and ASCII-only "ISTANBUL" becomes "istanbul", which is what people mean).
    Pass ``turkish`` to lowercase a single token the same way as the text it came from.
    """
    if turkish if turkish is not None else has_turkish_letters(text):
        text = text.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFC", text.lower())


def normalise(text: str) -> str:
    """Canonical form of a prompt for cache keys: NFC, Turkish lowercase, single spaces,
    no surrounding quotes or trailing punctuation. Inner punctuation is kept: "3.5" and
    "3,5" or "C++" must stay distinguishable."""
    text = unicodedata.normalize("NFC", text)
    text = _SPACE.sub(" ", tr_lower(text)).strip()
    return _EDGE_PUNCT.sub("", text)


def words(text: str) -> list[str]:
    """Lowercased word tokens (letters, digits, underscore)."""
    return _WORD.findall(tr_lower(text))
