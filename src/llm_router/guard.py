"""Lexical checks that veto a semantic cache hit.

Sentence embeddings are good at "same topic" and bad at "same question": "İstanbul'dan
Ankara'ya" and "Ankara'dan İstanbul'a", "en yüksek" and "en düşük", "100 dolar" and
"200 dolar" land almost on top of each other. Serving the cached answer to such a
near-miss is a *wrong answer*, not a slow one, so every hit above the similarity
threshold must also pass these cheap, explainable checks:

* identical normalised text always passes;
* prompts longer than the embedding window must match exactly;
* the numbers must be the same (multiset);
* the units and currencies must be the same;
* the count of negations must be the same (EN words, TR negative/privative suffixes);
* no polar pair may be flipped (highest/lowest, artış/azalış, ithalat/ihracat, ...);
* the named entities must be the same, and none may switch between source and target
  ("from X", "X'ten" vs "to X", "X'e").

The rules are deliberately simple and transparent; ``check`` returns the reason a hit
was vetoed so it can be reported and counted.
"""

from __future__ import annotations

import re
from collections import Counter

from .text import has_turkish_letters, normalise, tr_lower, words

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

_UNITS = {
    "usd": "usd", "dolar": "usd", "dollar": "usd", "dollars": "usd",
    "tl": "try", "lira": "try",
    "eur": "eur", "euro": "eur", "avro": "eur",
    "gbp": "gbp", "sterlin": "gbp",
    "km": "km", "kilometre": "km", "kilometer": "km", "kilometers": "km",
    "mil": "mile", "mile": "mile", "miles": "mile",
    "metre": "m", "meter": "m", "meters": "m",
    "cm": "cm", "santimetre": "cm", "inch": "inch", "inches": "inch", "inç": "inch",
    "kg": "kg", "kilogram": "kg", "kilo": "kg", "lb": "lb", "lbs": "lb", "pound": "lb",
    "gram": "g", "ton": "t",
    "celsius": "c", "santigrat": "c", "fahrenheit": "f",
    "gb": "gb", "mb": "mb", "tb": "tb", "kb": "kb",
    "litre": "l", "liter": "l", "galon": "gal", "gallon": "gal",
}  # fmt: skip
_UNIT_SYMBOLS = {"$": "usd", "₺": "try", "€": "eur", "£": "gbp", "°c": "c", "°f": "f"}
_GLUED_ONLY = {"m": "m", "g": "g", "l": "l", "c": "c", "f": "f"}  # only as "5m", "100g", "30c"

# Polar pairs: one side in the query and the other in the cached prompt vetoes the hit.
_POLAR = [
    ("en yüksek", "en düşük"), ("highest", "lowest"), ("yüksek", "düşük"), ("high", "low"),
    ("maksimum", "minimum"), ("maximum", "minimum"), ("max", "min"),
    ("en büyük", "en küçük"), ("largest", "smallest"), ("biggest", "smallest"),
    ("büyük", "küçük"), ("en çok", "en az"), ("en fazla", "en az"), ("most", "least"),
    ("artış", "azalış"), ("artış", "düşüş"), ("artır", "azalt"), ("increase", "decrease"),
    ("rise", "fall"), ("önce", "sonra"), ("before", "after"), ("ilk", "son"), ("first", "last"),
    ("ucuz", "pahalı"), ("cheapest", "most expensive"), ("cheap", "expensive"),
    ("giriş", "çıkış"), ("import", "export"), ("ithalat", "ihracat"),
    ("kuzey", "güney"), ("doğu", "batı"), ("north", "south"), ("east", "west"),
    ("avantaj", "dezavantaj"), ("advantage", "disadvantage"), ("pros", "cons"),
    ("kâr", "zarar"), ("kar", "zarar"), ("profit", "loss"), ("gain", "loss"),
    ("pozitif", "negatif"), ("positive", "negative"), ("doğru", "yanlış"), ("true", "false"),
    ("alış", "satış"), ("buy", "sell"), ("hızlı", "yavaş"), ("fast", "slow"),
    ("uzun", "kısa"), ("longest", "shortest"), ("sıcak", "soğuk"), ("hot", "cold"),
    ("erken", "geç"), ("early", "late"), ("ağır", "hafif"), ("heavy", "light"),
    ("upload", "download"), ("encrypt", "decrypt"), ("eski", "yeni"), ("oldest", "newest"),
]  # fmt: skip

_NEGATION_WORDS = re.compile(
    r"\b(?:not|never|no|none|without|nothing|nobody|değil\w*|yok\w*|hiç|hiçbir\w*|asla|hariç|dışında)\b"
    r"|n['’]t\b"
)
# Turkish negative and privative suffixes; only applied to Turkish text ("emphasize" is not "-siz").
_NEGATION_SUFFIXES = re.compile(
    r"\b\w{2,}m[ae]z\w*"  # gelmez, kullanılmaz
    r"|\b\w{2,}m[ae]y[ae]n\w*"  # içermeyen, olmayan
    r"|\b\w{2,}m[ae]d[ae]n\b"  # yapmadan
    r"|\b\w{2,}m[ıiuü]yor\w*"  # çalışmıyor
    r"|\b\w{2,}m[ae]d[ıiuü]\w*"  # gelmedi
    r"|\b\w{2,}m[ae]m[ae]k\w*"  # yapmamak
    r"|\b\w{2,}m[ae]m[ıi]ş\w*"  # gelmemiş
    r"|\b\w{2,}m[ae]ks[ıi]z[ıi]n\b"  # olmaksızın
    r"|\b\w{2,}s[ıiuü]z\w*"  # şekersiz, kablosuz
)
_TR_HINT = re.compile(
    r"[çğıöşü]|\b(?:ve|bir|bu|şu|için|ile|nedir|nasıl|neden|niçin|mi|mı|mu|mü|ne|kaç|hangi|daha|gibi|"
    r"olan|olarak|da|de|göre|arasındaki|midir|mıdır)\b"
)

_TOKEN = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü][\wÇĞİÖŞÜçğıöşü+#-]*(?:['’][\wçğıöşü]+)?")
_SOURCE_SUFFIX = ("dan", "den", "tan", "ten")
_TARGET_SUFFIX = ("a", "e", "ya", "ye", "na", "ne")
# Language names take case suffixes without an apostrophe: Türkçeden, İngilizceye, Almancaya.
_LANGUAGE = re.compile(r"^(\w+?[cç][ae])(dan|den|tan|ten|ya|ye)?$")
_SENTENCE_END = re.compile(r"[.!?:]\s*$")
_EN_CONTRACTIONS = {"s", "re", "ve", "ll", "d", "m", "t"}  # What's, I'm: not a Turkish case suffix
_ARTICLE = r"(?:(?:my|the|a|an|your|our)\s+)?"
_FROM_TO = re.compile(rf"\bfrom\s+{_ARTICLE}(\w+)[\w\s]*?\s+(?:to|into)\s+{_ARTICLE}(\w+)")
_X_TO_Y = re.compile(rf"\b(\w+)\s+(?:to|into)\s+{_ARTICLE}(\w+)")
_STOP = {"how", "want", "need", "going", "have", "has", "had", "able", "used", "way", "ways", "is", "are"}


def numbers(text: str) -> Counter[str]:
    return Counter(m.group().replace(",", ".") for m in _NUMBER.finditer(text))


def units(text: str) -> set[str]:
    found = {_UNITS[w] for w in words(text) if w in _UNITS}
    lowered = tr_lower(text)
    found |= {unit for sym, unit in _UNIT_SYMBOLS.items() if sym in lowered}
    for m in re.finditer(r"\d\s?([a-zçğıöşü°]+)\b", lowered):  # "5km", "10 kg", "100g"
        unit = m.group(1)
        if unit in _UNITS or unit in _GLUED_ONLY:
            found.add(_UNITS.get(unit) or _GLUED_ONLY[unit])
    return found


def is_turkish(text: str) -> bool:
    return _TR_HINT.search(tr_lower(text)) is not None


def negations(text: str) -> int:
    lowered = tr_lower(text)
    count = len(_NEGATION_WORDS.findall(lowered))
    if is_turkish(text):
        count += len(_NEGATION_SUFFIXES.findall(lowered))
    return count


def _has_term(text_norm: str, tokens: list[str], term: str) -> bool:
    if " " in term:
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text_norm) is not None
    if len(term) <= 3:
        return term in tokens
    return any(t.startswith(term) for t in tokens)


def flipped_polarity(a: str, b: str) -> str | None:
    na, nb = normalise(a), normalise(b)
    ta, tb = words(a), words(b)
    for left, right in _POLAR:
        al, ar = _has_term(na, ta, left), _has_term(na, ta, right)
        bl, br = _has_term(nb, tb, left), _has_term(nb, tb, right)
        if (al and not ar and br and not bl) or (ar and not al and bl and not br):
            return f"{left}/{right}"
    return None


def _split(token: str) -> tuple[str, str]:
    for apostrophe in ("'", "’"):
        if apostrophe in token:
            stem, suffix = token.split(apostrophe, 1)
            return stem, suffix
    return token, ""


def entities(text: str) -> dict[str, str]:
    """Named entities -> role ("source", "target" or "").

    Capitalised tokens count as entities. At the start of a sentence only when they carry
    an apostrophe suffix (Ankara'dan), are all caps (KVKK) or are a language name with a
    case suffix (Türkçeden), so "What" and "Nasıl" don't.
    """
    found: dict[str, str] = {}
    turkish = has_turkish_letters(text)
    for m in _TOKEN.finditer(text):
        token = m.group()
        if not token[0].isupper():
            continue
        stem, suffix = _split(token)
        role = ""
        language = None if suffix else _LANGUAGE.match(tr_lower(stem, turkish))
        if language and language.group(2):
            stem = language.group(1)
            role = "source" if language.group(2) in _SOURCE_SUFFIX else "target"
        before = text[: m.start()]
        sentence_start = not before.strip() or _SENTENCE_END.search(before) is not None
        if len(stem) < 2:  # the pronoun "I", initials
            continue
        is_caps = stem.isupper()
        turkish_suffix = bool(suffix) and tr_lower(suffix, turkish) not in _EN_CONTRACTIONS
        if sentence_start and not (turkish_suffix or is_caps or role):
            continue
        prev = tr_lower(before.rstrip().rsplit(" ", 1)[-1], turkish) if before.strip() else ""
        if prev == "from":
            role = "source"
        elif prev in ("to", "into"):
            role = "target"
        if suffix:
            low_suffix = tr_lower(suffix, turkish)
            if low_suffix in _SOURCE_SUFFIX:
                role = "source"
            elif low_suffix in _TARGET_SUFFIX:
                role = "target"
        key = tr_lower(stem, turkish)
        if key and (key not in found or role):
            found[key] = role or found.get(key, "")
    return found


def ordered_pairs(text: str) -> set[tuple[str, str]]:
    """English "from X to Y" / "X to Y" pairs, to catch swapped directions between common nouns
    ("from savings to checking", "miles to kilometres")."""
    lowered = tr_lower(text)
    pairs = {(a, b) for a, b in _FROM_TO.findall(lowered)}
    pairs |= {(a, b) for a, b in _X_TO_Y.findall(lowered) if a not in _STOP and not a.isdigit()}
    return {(a, b) for a, b in pairs if a != b}


def _mentions(tokens: list[str], entity: str) -> bool:
    return any(t.startswith(entity) for t in tokens)


class CacheGuard:
    def __init__(self, exact_over_chars: int = 400) -> None:
        self.exact_over_chars = exact_over_chars

    def check(self, query: str, cached: str) -> str | None:
        """None if the cached answer may be served for ``query``, else the veto reason."""
        nq, nc = normalise(query), normalise(cached)
        if nq == nc:
            return None
        if max(len(nq), len(nc)) > self.exact_over_chars:
            return "long_prompt_not_exact"
        if numbers(query) != numbers(cached):
            return "numbers_differ"
        if units(query) != units(cached):
            return "units_differ"
        if negations(query) != negations(cached):
            return "negation_differs"
        if pair := flipped_polarity(query, cached):
            return f"polarity_flipped:{pair}"
        eq, ec = entities(query), entities(cached)
        tq, tc = words(query), words(cached)
        for name in eq:
            if not _mentions(tc, name):
                return f"entity_differs:{name}"
        for name in ec:
            if not _mentions(tq, name):
                return f"entity_differs:{name}"
        swapped = {(b, a) for a, b in ordered_pairs(query)} & ordered_pairs(cached)
        if swapped:
            a, b = sorted(swapped)[0]
            return f"direction_differs:{b}->{a}"
        for name, role in eq.items():
            other = ec.get(name, "")
            if role and other and role != other:
                return f"direction_differs:{name}"
        return None
