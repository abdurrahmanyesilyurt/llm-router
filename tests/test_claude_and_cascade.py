import pytest
from conftest import FakeAnthropic, response, structured

from llm_router.cascade import SCHEMA, Cascade, judge
from llm_router.claude import (
    FALLBACK_BETA,
    ClaudeClient,
    ClaudeNotConfigured,
    Completion,
    Usage,
    parse_response,
)
from llm_router.pricing import PRICES, cost_usd

MSGS = [{"role": "user", "content": "Merhaba"}]


def test_large_model_goes_through_beta_with_server_side_fallback():
    fake = FakeAnthropic()
    ClaudeClient(fake).complete(model="claude-sonnet-5-5", messages=MSGS, max_tokens=256, effort="medium")
    req = fake.requests[0]
    assert req["_endpoint"] == "beta"
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert req["output_config"] == {"effort": "medium"}


def test_haiku_is_called_plainly():
    fake = FakeAnthropic()
    ClaudeClient(fake).complete(model="claude-haiku-5-5", messages=MSGS, system="Kısa yaz.", max_tokens=256)
    req = fake.requests[0]
    assert req["_endpoint"] == "messages"
    assert "betas" not in req and "fallbacks" not in req and "output_config" not in req
    assert req["system"] == "Kısa yaz."


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"])
def test_sampling_parameters_are_never_sent(model):
    fake = FakeAnthropic()
    ClaudeClient(fake).complete(model=model, messages=MSGS, max_tokens=64, effort="low", json_schema=SCHEMA)
    req = fake.requests[0]
    assert not {"temperature", "top_p", "top_k"} & set(req)
    assert req["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}


def test_only_text_blocks_are_read_and_refusals_are_flagged():
    c = parse_response("claude-haiku-5-5", response("Merhaba!", thinking=True))
    assert c.text == "Merhaba!" and c.finished and not c.refused
    refused = parse_response("claude-sonnet-5-5", response("partial", stop_reason="refusal"))
    assert refused.refused and refused.text == "" and not refused.finished
    truncated = parse_response("claude-haiku-5-5", response("yarım", stop_reason="max_tokens"))
    assert not truncated.finished


def test_missing_credentials_become_a_typed_error():
    def no_auth(request):
        raise TypeError("Could not resolve authentication method.")

    with pytest.raises(ClaudeNotConfigured):
        ClaudeClient(FakeAnthropic(no_auth)).complete(model="claude-haiku-5-5", messages=MSGS)


def test_cost_math():
    assert PRICES["claude-haiku-5-5"].input == 0.10 and PRICES["claude-sonnet-5-5"].output == 10.00
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert cost_usd("claude-opus-5-5", 1000, 500) == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    assert cost_usd("claude-haiku-5-5", 1000, 1000) == pytest.approx((1000 * 0.10 + 1000 * 0.50) / 1e6)
    # long prompts switch Haiku to its second rate card
    assert cost_usd("claude-haiku-5-5", 200_000, 0) == pytest.approx(200_000 * 0.50 / 1e6)
    # cache writes 1.25x, reads 0.1x of the input rate
    assert cost_usd("claude-sonnet-5-5", 0, 0, 1000, 1000) == pytest.approx(
        (1000 * 2 * 1.25 + 1000 * 2 * 0.1) / 1e6
    )
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", 1, 1)


def test_cost_uses_the_model_that_answered():
    c = Completion("x", "claude-sonnet-5-5", "claude-sonnet-5-5", "end_turn", Usage(1000, 1000))
    assert c.cost_usd == pytest.approx(0.012)
    fallback = Completion("x", "claude-sonnet-5", "claude-sonnet-5-5", "end_turn", Usage(1000, 1000))
    assert fallback.priced_model == "claude-sonnet-5-5"  # unknown fallback id: priced as requested


def _small(**kw):
    return parse_response("claude-haiku-5-5", structured(**kw))


@pytest.mark.parametrize(
    "small, escalate, kind",
    [
        (_small(answer="Ankara", confidence=0.95), False, "accepted"),
        (_small(answer="Ankara", confidence=0.5), True, "low_confidence"),
        (_small(answer="Belki", confidence=0.9, needs_stronger_model=True), True, "flag"),
        (_small(answer="", confidence=0.9), True, "empty"),
        (_small(answer="x", confidence=1.7), False, "accepted"),  # clamped to 1.0
        (parse_response("claude-haiku-5-5", response("not json")), True, "invalid_output"),
        (parse_response("claude-haiku-5-5", response("{}", stop_reason="refusal")), True, "refused"),
        (
            parse_response("claude-haiku-5-5", response('{"answer": "ya', stop_reason="max_tokens")),
            True,
            "truncated",
        ),
    ],
)
def test_cascade_judge(small, escalate, kind):
    verdict = judge(small, threshold=0.7)
    assert verdict.escalate is escalate and verdict.kind == kind


def test_cascade_accepts_confident_small_answer_with_one_call():
    fake = FakeAnthropic(lambda req: structured("Ankara", 0.92, model=req["model"]))
    result = Cascade(ClaudeClient(fake), "claude-haiku-5-5", "claude-sonnet-5-5", 0.7).run(None, MSGS, 512)
    assert not result.escalated and len(result.calls) == 1
    assert result.final.text == "Ankara"
    req = fake.requests[0]
    assert req["model"] == "claude-haiku-5-5" and req["output_config"]["format"]["type"] == "json_schema"
    assert req["max_tokens"] > 512  # headroom for the JSON wrapper
    assert "first, fast tier" in req["system"]


def test_cascade_escalates_and_pays_for_both_calls():
    def responder(req):
        if req["model"] == "claude-haiku-5-5":
            return structured("Emin değilim", 0.3, model=req["model"], input_tokens=200, output_tokens=100)
        return response("Ayrıntılı cevap", model=req["model"], input_tokens=150, output_tokens=400)

    fake = FakeAnthropic(responder)
    result = Cascade(ClaudeClient(fake), "claude-haiku-5-5", "claude-sonnet-5-5", 0.7).run(
        "Sistem", MSGS, 512
    )
    assert result.escalated and [t for t, _ in result.calls] == ["small", "large"]
    assert result.final.text == "Ayrıntılı cevap"
    large_req = fake.requests[1]
    assert (
        large_req["system"] == "Sistem"
        and "output_config" not in large_req
        and large_req["max_tokens"] == 512
    )
    total = sum(c.cost_usd for _, c in result.calls)
    assert total == pytest.approx(
        cost_usd("claude-haiku-5-5", 200, 100) + cost_usd("claude-sonnet-5-5", 150, 400)
    )
