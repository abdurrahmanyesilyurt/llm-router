"""Personal-data guard for the semantic cache.

A cached answer is served to *other* callers whose prompt is merely similar, so a prompt
that carries personal data must never be written to (or answered from) the cache: the
answer may echo the e-mail, phone number or ID back to someone else. A false positive
here only costs a cache miss, so the patterns lean towards flagging; checksums (TC
kimlik, IBAN mod-97, Luhn) are used where the format alone would hit ordinary numbers.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

_SEP = r"[\s.-]?"
_PHONE = re.compile(
    rf"""(?<![\w+])(?:
        (?:\+90|0090|0)?{_SEP}\(?5\d{{2}}\)?{_SEP}\d{{3}}{_SEP}\d{{2}}{_SEP}\d{{2}}        # TR mobile
      | (?:\+90|0090|0){_SEP}\(?[2-4]\d{{2}}\)?{_SEP}\d{{3}}{_SEP}\d{{2}}{_SEP}\d{{2}}     # TR landline
      | \+\d{{1,3}}{_SEP}\(?\d{{1,4}}\)?(?:{_SEP}\d{{2,4}}){{2,4}}                           # international
    )(?!\w)""",
    re.VERBOSE,
)

_ELEVEN_DIGITS = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
_IBAN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}(?: ?[A-Za-z0-9]{4}){2,7}(?: ?[A-Za-z0-9]{1,3})?(?![A-Za-z0-9])"
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def is_tc_kimlik(number: str) -> bool:
    """Turkish national ID: 11 digits, no leading zero, two check digits."""
    if len(number) != 11 or not number.isdigit() or number[0] == "0":
        return False
    d = [int(c) for c in number]
    odd = d[0] + d[2] + d[4] + d[6] + d[8]
    even = d[1] + d[3] + d[5] + d[7]
    return (odd * 7 - even) % 10 == d[9] and sum(d[:10]) % 10 == d[10]


def is_iban(candidate: str) -> bool:
    """TR IBANs by shape (TR + 24 digits); any other country by the ISO 13616 mod-97 check."""
    iban = candidate.replace(" ", "").upper()
    if iban.startswith("TR") and len(iban) == 26 and iban[2:].isdigit():
        return True
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def luhn(number: str) -> bool:
    digits = [int(c) for c in number][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def find_pii(text: str) -> list[str]:
    """Kinds of personal data found in the text, e.g. ["email", "tc_kimlik"]."""
    kinds: set[str] = set()
    if _EMAIL.search(text):
        kinds.add("email")
    for m in _PHONE.finditer(text):
        if sum(c.isdigit() for c in m.group()) >= 10:
            kinds.add("phone")
            break
    if any(is_tc_kimlik(m.group()) for m in _ELEVEN_DIGITS.finditer(text)):
        kinds.add("tc_kimlik")
    if any(is_iban(m.group()) for m in _IBAN.finditer(text)):
        kinds.add("iban")
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn(digits):
            kinds.add("card")
            break
    return sorted(kinds)
