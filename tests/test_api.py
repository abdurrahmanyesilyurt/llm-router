import anthropic
import httpx2
import pytest
from conftest import FakeAnthropic, FakeCache, make_gateway, response, structured
from fastapi.testclient import TestClient

from llm_router.api import create_app
from llm_router.pricing import cost_usd


def client_for(gateway):
    return TestClient(create_app(lambda: gateway))


def chat(client, content, **body):
    return client.post("/v1/chat", json={"messages": [{"role": "user", "content": content}], **body})


def test_chat_routes_calls_and_reports_cost():
    gw = make_gateway()
    with client_for(gw) as c:
        r = chat(c, "Şu cümleyi İngilizceye çevir: 'Merhaba dünya'", strategy="heuristic")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tier"] == "small" and body["model"] == "claude-haiku-5-5"
    assert body["route"]["strategy"] == "heuristic" and "simple_task" in body["route"]["reason"]
    assert body["cache"] == {"hit": False, "similarity": None, "stored": True, "skipped": None}
    assert body["usage"]["input_tokens"] == 100 and len(body["usage"]["calls"]) == 1
    assert body["cost_usd"] == pytest.approx(cost_usd("claude-haiku-5-5", 100, 50))
    assert body["text"] == "answer from claude-haiku-5-5"


def test_second_identical_question_is_a_free_cache_hit():
    fake = FakeAnthropic()
    gw = make_gateway(fake)
    with client_for(gw) as c:
        first = chat(c, "REST API nedir?", strategy="always-large").json()
        second = chat(c, "  rest api nedir ", strategy="always-small").json()
    assert len(fake.requests) == 1
    assert second["cache"]["hit"] is True and second["cache"]["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert second["cache"]["saved_cost_usd"] == pytest.approx(first["cost_usd"])
    assert second["cost_usd"] == 0.0 and second["tier"] == "large" and second["usage"]["calls"] == []
    assert second["route"]["reason"].startswith("semantic cache hit")


def test_same_question_under_another_system_prompt_misses():
    fake = FakeAnthropic()
    gw = make_gateway(fake)
    with client_for(gw) as c:
        chat(c, "Bunu özetle: toplantı yarın", system="Sen bir avukatsın.")
        r = chat(c, "Bunu özetle: toplantı yarın", system="Sen bir öğretmensin.").json()
    assert r["cache"]["hit"] is False and r["cache"]["similarity"] is None
    assert len(fake.requests) == 2


def test_guard_vetoes_a_near_miss_hit():
    fake = FakeAnthropic()
    gw = make_gateway(fake, cache_threshold=0.5)
    with client_for(gw) as c:
        chat(c, "İstanbul'dan Ankara'ya ilk tren kaçta?")
        r = chat(c, "Ankara'dan İstanbul'a ilk tren kaçta?").json()
    assert r["cache"]["hit"] is False and r["cache"]["vetoed"].startswith("direction_differs")
    assert len(fake.requests) == 2
    assert (
        'llm_router_cache_guard_vetoes_total{reason="direction_differs"} 1.0' in gw.metrics.render().decode()
    )


@pytest.mark.parametrize(
    "content, body, reason",
    [
        ("Bana ali@example.com adresinden yaz", {}, "pii:email"),
        ("TC kimlik numaram 10000000146", {}, "pii:tc_kimlik"),
        ("REST API nedir?", {"cache": False}, "disabled_by_request"),
    ],
)
def test_cache_is_skipped_with_a_reason(content, body, reason):
    gw = make_gateway()
    with client_for(gw) as c:
        r = chat(c, content, **body).json()
    assert r["cache"]["skipped"] == reason and r["cache"]["stored"] is False
    assert gw.cache.entries == {}


def test_pii_anywhere_in_the_conversation_skips_the_cache():
    gw = make_gateway()
    with client_for(gw) as c:
        r = c.post(
            "/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "IBAN'ım TR33 0006 1005 1978 6457 8413 26"},
                    {"role": "assistant", "content": "Tamam."},
                    {"role": "user", "content": "Bunu kaydet"},
                ]
            },
        ).json()
    assert r["cache"]["skipped"] == "pii:iban"


def test_cascade_escalation_returns_both_usages_and_summed_cost():
    def responder(req):
        if req["model"] == "claude-haiku-5-5":
            return structured("?", 0.2, model=req["model"], input_tokens=300, output_tokens=80)
        return response("Uzun ve dikkatli cevap", model=req["model"], input_tokens=120, output_tokens=600)

    gw = make_gateway(FakeAnthropic(responder))
    with client_for(gw) as c:
        r = chat(c, "Bu iki mimariyi karşılaştır", strategy="cascade").json()
    assert r["tier"] == "large" and r["route"]["escalated"] is True and r["route"]["score"] == 0.2
    assert [call["tier"] for call in r["usage"]["calls"]] == ["small", "large"]
    assert r["usage"]["input_tokens"] == 420 and r["usage"]["output_tokens"] == 680
    expected = cost_usd("claude-haiku-5-5", 300, 80) + cost_usd("claude-sonnet-5-5", 120, 600)
    assert r["cost_usd"] == pytest.approx(expected)
    assert 'llm_router_escalations_total{reason="low_confidence"} 1.0' in gw.metrics.render().decode()


def test_cascade_accepted_small_answer_is_cached_as_small():
    gw = make_gateway(FakeAnthropic(lambda req: structured("Ankara", 0.95, model=req["model"])))
    with client_for(gw) as c:
        r = chat(c, "Türkiye'nin başkenti neresi?", strategy="cascade").json()
    assert r["tier"] == "small" and r["text"] == "Ankara" and r["cache"]["stored"] is True
    assert next(iter(gw.cache.entries.values()))["text"] == "Ankara"


def test_knn_strategy_and_baselines():
    gw = make_gateway()
    with client_for(gw) as c:
        knn = chat(c, "Prove that there are infinitely many primes", strategy="knn", cache=False).json()
        small = chat(c, "Prove it", strategy="always-small", cache=False).json()
        large = chat(c, "hi", strategy="always-large", cache=False).json()
    assert knn["tier"] == "large" and "neighbours hard" in knn["route"]["reason"]
    assert small["tier"] == "small" and large["tier"] == "large"


def test_knn_unavailable_is_a_400():
    with client_for(make_gateway(with_knn=False)) as c:
        r = chat(c, "x", strategy="knn")
    assert r.status_code == 400 and "kNN" in r.json()["detail"]


def test_refusals_are_returned_but_not_cached():
    gw = make_gateway(FakeAnthropic(lambda req: response("", model=req["model"], stop_reason="refusal")))
    with client_for(gw) as c:
        r = chat(c, "something", strategy="always-large").json()
    assert r["refused"] is True and r["text"] == "" and r["cache"]["stored"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "hi"}], "strategy": "random"},
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
        {"messages": [{"role": "system", "content": "hi"}]},
    ],
)
def test_validation(payload):
    with client_for(make_gateway()) as c:
        assert c.post("/v1/chat", json=payload).status_code == 422


def _status_error(cls, status):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=req), body=None)


@pytest.mark.parametrize(
    "exc, status",
    [
        (_status_error(anthropic.RateLimitError, 429), 429),
        (_status_error(anthropic.AuthenticationError, 401), 503),
        (_status_error(anthropic.InternalServerError, 500), 502),
        (_status_error(anthropic.BadRequestError, 400), 400),
        (TypeError("Could not resolve authentication method."), 503),
    ],
)
def test_sdk_errors_map_to_http_codes(exc, status):
    def boom(request):
        raise exc

    gw = make_gateway(FakeAnthropic(boom))
    with client_for(gw) as c:
        r = chat(c, "hello", strategy="always-small")
    assert r.status_code == status


def test_a_broken_cache_never_fails_the_request():
    class Broken(FakeCache):
        def nearest(self, ns, vector):
            raise ConnectionError("redis down")

        def store(self, *a, **kw):
            raise ConnectionError("redis down")

    gw = make_gateway(cache=Broken())
    with client_for(gw) as c:
        r = chat(c, "hello")
    assert r.status_code == 200 and r.json()["cache"]["error"] == "lookup_failed"


def test_metrics_and_health():
    gw = make_gateway()
    with client_for(gw) as c:
        chat(c, "REST API nedir?")
        chat(c, "REST API nedir?")
        chat(c, "x@y.com adresine yaz")
        metrics = c.get("/metrics")
        health = c.get("/health").json()
    text = metrics.text
    assert metrics.headers["content-type"].startswith("text/plain")
    assert 'llm_router_requests_total{cache="hit",strategy="heuristic",tier="small"} 1.0' in text
    assert 'llm_router_requests_total{cache="miss",strategy="heuristic",tier="small"} 1.0' in text
    assert 'llm_router_cache_skipped_total{reason="pii"} 1.0' in text
    assert "llm_router_cache_hit_ratio 0.5" in text
    assert 'llm_router_cost_usd_total{model="claude-haiku-5-5"}' in text
    assert "llm_router_request_latency_seconds_bucket" in text
    assert health["models"] == {"small": "claude-haiku-5-5", "large": "claude-sonnet-5-5"}
    assert "knn" in health["strategies"] and health["cache"]["entries"] == 1
