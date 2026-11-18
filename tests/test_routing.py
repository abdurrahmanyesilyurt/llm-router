import numpy as np
import pytest

from llm_router.embeddings import HashingEmbedder, embed_one
from llm_router.heuristic import HeuristicRouter
from llm_router.knn import Example, KnnRouter
from llm_router.text import normalise

heuristic = HeuristicRouter(threshold=1.5)


def test_code_debugging_and_reasoning_features_fire():
    prompt = (
        "Bu Node.js kodunda stok bazen eksiye düşüyor. Neden ve nasıl düzeltirim?\n"
        "```js\nconst p = await db.find(id);\n```"
    )
    s = heuristic.score(prompt)
    assert {"code", "debugging", "reasoning"} <= set(s.features)
    assert s.score >= 1.5
    tier, _, reason = heuristic.route(prompt)
    assert tier == "large" and "code(+1.0)" in reason and ">= 1.5" in reason


@pytest.mark.parametrize(
    "prompt, feature",
    [
        ("0,999... sayısının 1'e eşit olduğunu kanıtla.", "reasoning"),
        ("Prove that the square root of 2 is irrational.", "reasoning"),
        ("Bir torbada 5 kırmızı, 3 mavi top var; çekilen 2 topun aynı renk olma olasılığı nedir?", "math"),
        ("f(x) = x^3 - 6x^2 + 9x + 1 fonksiyonunun ekstremum noktalarını bul", "math"),
        ("KVKK'nın 9. maddesine göre yurt dışına aktarım şartları nelerdir?", "legal"),
        ("Under the EU AI Act, what obligations fall on the deployer?", "legal"),
        ("Should we use saga or two-phase commit between orders and payments?", "alternatives"),
        ("Sipariş ve ödeme arasında saga mı, iki aşamalı commit mi kullanmalıyız?", "alternatives"),
    ],
)
def test_difficulty_features(prompt, feature):
    s = heuristic.score(prompt)
    assert feature in s.features, s.explain()
    assert heuristic.route(prompt)[0] == "large"


@pytest.mark.parametrize(
    "prompt",
    [
        "Şu cümleyi İngilizceye çevir: 'Siparişiniz kargoya verildi.'",
        "Translate to French: 'Please keep your seatbelt fastened.'",
        "Türkiye Büyük Millet Meclisi hangi yıl açıldı?",
        "Neden gökyüzü mavi görünür? Kısaca anlat.",
        "Fix the grammar: 'Me and him goes to the office every days.'",
        "Bugün toplantı var mı?",  # "bug" must not match inside "bugün"
    ],
)
def test_easy_prompts_stay_small(prompt):
    tier, _, reason = heuristic.route(prompt)
    assert tier == "small", reason


def test_simple_task_discount_is_smaller_when_hard_signals_exist():
    plain = heuristic.score("Bu metni özetle: toplantı yarın saat onda.")
    with_code = heuristic.score("Bu fonksiyondaki hatayı düzelt:\n```python\ndef f(x): return x/0\n```")
    assert plain.features["simple_task"] == -1.5
    assert with_code.features["simple_task"] == -0.5


def test_explain_lists_fired_features():
    s = heuristic.score("Compare REST and GraphQL")
    assert s.explain().startswith("reasoning:compare(+1.0)")
    assert heuristic.score("hello").explain() == "no difficulty features"


EXAMPLES = [
    Example("easy-1", "translate this sentence to french", "easy"),
    Example("easy-2", "translate this word to german", "easy"),
    Example("easy-3", "summarise this short paragraph", "easy"),
    Example("hard-1", "prove the theorem about prime numbers", "hard"),
    Example("hard-2", "prove the lemma about prime gaps", "hard"),
    Example("hard-3", "design a distributed consensus protocol", "hard"),
]


def _vec(emb, text):
    return embed_one(emb, normalise(text))


def test_knn_vote_is_similarity_weighted():
    emb = HashingEmbedder(256)
    knn = KnnRouter(emb, EXAMPLES, k=3, margin=0.0)
    vote = knn.vote(_vec(emb, "prove the theorem about prime numbers"))
    assert vote.neighbours[0][0] == "hard-1" and vote.neighbours[0][2] == pytest.approx(1.0, abs=1e-5)
    assert vote.p_hard > 0.5
    tier, p, reason = knn.route(_vec(emb, "translate this sentence to french please"))
    assert tier == "small" and p < 0.5 and "nearest easy-1" in reason


def test_knn_unsure_vote_goes_large():
    emb = HashingEmbedder(64)
    examples = [Example("a", "same prompt", "easy"), Example("b", "same prompt", "hard")]
    knn = KnnRouter(emb, examples, k=2, margin=0.1)
    tier, p_hard, reason = knn.route(np.asarray(emb.embed(["same prompt"])[0]))
    assert p_hard == pytest.approx(0.5) and tier == "large" and "unsure" in reason


def test_knn_requires_examples_and_caps_k():
    emb = HashingEmbedder(32)
    with pytest.raises(ValueError):
        KnnRouter(emb, [])
    assert KnnRouter(emb, EXAMPLES[:2], k=10).k == 2
