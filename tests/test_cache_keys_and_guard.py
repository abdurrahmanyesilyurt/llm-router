import pytest

from llm_router.cache import namespace, prompt_id
from llm_router.guard import CacheGuard, entities, negations, numbers, units

HISTORY = [
    {"role": "user", "content": "Merhaba"},
    {"role": "assistant", "content": "Merhaba! Nasıl yardımcı olabilirim?"},
]


def test_namespace_is_stable_and_covers_system_and_history():
    base = namespace("Sen bir asistansın.", HISTORY)
    assert base == namespace("Sen bir asistansın.", list(HISTORY))
    assert base == namespace("  Sen bir asistansın.\n", HISTORY)  # surrounding whitespace only
    assert base != namespace("Sen bir hukuk asistanısın.", HISTORY)
    assert base != namespace("Sen bir asistansın.", HISTORY[:1])
    assert base != namespace(None, HISTORY)
    assert namespace(None, []) == namespace("", [])
    assert len(base) == 24


def test_namespace_distinguishes_roles():
    a = [{"role": "user", "content": "x"}]
    b = [{"role": "assistant", "content": "x"}]
    assert namespace(None, a) != namespace(None, b)


def test_prompt_id_uses_normalised_text():
    assert prompt_id("REST API nedir?") == prompt_id("  rest api nedir ")
    assert prompt_id("REST API nedir?") != prompt_id("GraphQL nedir?")


guard = CacheGuard(exact_over_chars=400)


@pytest.mark.parametrize(
    "query, cached, reason",
    [
        (
            "İstanbul'dan Ankara'ya ilk tren kaçta?",
            "Ankara'dan İstanbul'a ilk tren kaçta?",
            "direction_differs",
        ),
        ("Türkçeden İngilizceye çevir: merhaba", "İngilizceden Türkçeye çevir: merhaba", "direction_differs"),
        ("Flights from Paris to Rome", "Flights from Rome to Paris", "direction_differs"),
        ("Convert 10 miles to kilometres", "Convert 10 kilometres to miles", "direction_differs"),
        ("100 dolar kaç TL?", "200 dolar kaç TL?", "numbers_differ"),
        ("5 kg kaç pound eder?", "5 kg kaç ons eder?", "units_differ"),
        ("Şekerli kahve sağlıklı mı?", "Şekersiz kahve sağlıklı mı?", "negation_differs"),
        ("Which foods should I eat?", "Which foods should I not eat?", "negation_differs"),
        ("Türkiye'nin en yüksek dağı?", "Türkiye'nin en düşük noktası?", "polarity_flipped"),
        ("What are the advantages of X?", "What are the disadvantages of X?", "polarity_flipped"),
        ("Ankara'nın nüfusu kaç?", "İzmir'in nüfusu kaç?", "entity_differs"),
        ("Who is the CEO of Microsoft?", "Who is the CEO of Apple?", "entity_differs"),
    ],
)
def test_guard_vetoes_near_misses(query, cached, reason):
    veto = guard.check(query, cached)
    assert veto is not None and veto.startswith(reason), veto


@pytest.mark.parametrize(
    "query, cached",
    [
        ("Python'da bir listeyi nasıl ters çeviririm?", "Python'da liste tersine nasıl çevrilir?"),
        ("What's the difference between TCP and UDP?", "Explain the difference between TCP and UDP."),
        ("How can I improve my sleep?", "Tips for sleeping better?"),
        ("Ekmek hamurum neden kabarmadı?", "Hamurum neden mayalanmadı?"),  # negation on both sides
        ("WHAT IS KUBERNETES", "what is kubernetes?"),
        (
            "Please emphasize the key points",
            "Emphasize the key points please",
        ),  # English "-size" is not "-siz"
    ],
)
def test_guard_lets_paraphrases_through(query, cached):
    assert guard.check(query, cached) is None


def test_long_prompts_must_match_exactly():
    long_a = "Özetle: " + "kelime " * 80 + "sonunda artış oldu."
    long_b = "Özetle: " + "kelime " * 80 + "sonunda azalış oldu."
    assert guard.check(long_a, long_b) == "long_prompt_not_exact"
    assert guard.check(long_a, "  " + long_a.replace("kelime", "KELİME") + "  ") is None  # same text


def test_guard_building_blocks():
    assert numbers("1.000 TL ve 3,5 kg") == numbers("3.5 kg ve 1.000 TL")
    assert units("10km yol, 5 kg yük, 20 $") == {"km", "kg", "usd"}
    assert negations("Bu kod çalışmıyor ve hiç log yok") == 3
    assert entities("İstanbul'dan Ankara'ya") == {"istanbul": "source", "ankara": "target"}
    assert "what" not in entities("What's the capital of France?")
    assert "i" not in entities("How can I fix it?")
