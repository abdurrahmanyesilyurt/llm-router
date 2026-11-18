import pytest

from llm_router.pii import find_pii, is_iban, is_tc_kimlik, luhn
from llm_router.text import normalise, tr_lower


def test_turkish_lowercase_only_for_turkish_text():
    assert tr_lower("IŞIK İSTANBUL") == "ışık istanbul"
    assert tr_lower("EU AI Act") == "eu ai act"  # English stays English
    assert tr_lower("Is it") == "is it"
    assert tr_lower("ISPARTA", turkish=True) == "ısparta"


def test_normalise_is_canonical_but_keeps_inner_punctuation():
    assert normalise("  Merhaba,   REST API nedir?? ") == "merhaba, rest api nedir"
    assert normalise("“What is Python?”") == "what is python"
    assert normalise("3.5") != normalise("3,5")
    assert normalise("C++ nedir?") == "c++ nedir"


@pytest.mark.parametrize(
    "number, valid",
    [
        ("10000000146", True),  # well-known valid test number
        ("10000000147", False),  # last check digit wrong
        ("10000000156", False),  # tenth digit wrong
        ("01234567890", False),  # leading zero
        ("1000000014", False),  # 10 digits
    ],
)
def test_tc_kimlik_checksum(number, valid):
    assert is_tc_kimlik(number) is valid


def test_tc_kimlik_in_text_needs_a_valid_checksum():
    assert find_pii("TC kimlik numaram 10000000146, başvuru yapmak istiyorum") == ["tc_kimlik"]
    assert find_pii("Sipariş numarası 10000000147 olan paket nerede?") == []


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Bana ayse.demir@example.com adresinden ulaşın", "email"),
        ("Numaram 0532 123 45 67, arar mısınız?", "phone"),
        ("+90 (532) 123-45-67 numarasını kaydet", "phone"),
        ("Call me at +44 20 7946 0958 tomorrow", "phone"),
        ("IBAN: TR33 0006 1005 1978 6457 8413 26", "iban"),
        ("Pay to DE89 3704 0044 0532 0130 00 please", "iban"),
        ("Kart: 4111 1111 1111 1111", "card"),
    ],
)
def test_pii_kinds(text, kind):
    assert kind in find_pii(text)


@pytest.mark.parametrize(
    "text",
    [
        "2024 yılında enflasyon yüzde 44 oldu, 2023'te 64'tü.",
        "x^3 - 6x^2 + 9x + 1 fonksiyonunun türevini al",
        "Python 3.13'te yeni gelen özellikler neler?",
        "Fatura No: 2025-118 | Tarih: 14.02.2025 | Genel toplam: 3.000,00 TL",
        "What is 15% of 80?",
        "DE00 1234 5678 9012 3456 78 is not a valid IBAN",
    ],
)
def test_ordinary_numbers_are_not_pii(text):
    assert find_pii(text) == []


def test_iban_and_luhn_helpers():
    assert is_iban("GB82 WEST 1234 5698 7654 32")
    assert not is_iban("GB82 WEST 1234 5698 7654 33")
    assert luhn("4111111111111111")
    assert not luhn("4111111111111112")
