"""One live check per language. Needs GROQ_API_KEY (or OPENAI_*); skipped without it.

    GROQ_API_KEY=... python -m pytest -q
"""
import os

import pytest

from groundedness import check

MODEL = os.environ.get("GROUNDEDNESS_MODEL", "qwen/qwen3.8-27b")
KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")

CASES = {
    "az": ("Diş klinikası, Nizami küç. 12. B.e–Şənbə 9:00–19:00. Konsultasiya 30 AZN.",
           "Salam! Konsultasiya 25 AZN-dir, bazar günü də işləyirik. Ünvan Nizami küçəsi 12.", "25"),
    "ru": ("Стоматология, ул. Низами 12. Пн–Сб 9:00–19:00. Консультация 30 AZN.",
           "Здравствуйте! Консультация стоит 25 AZN, работаем и по воскресеньям. Адрес: ул. Низами 12.", "25"),
    "en": ("Dental clinic, 12 Nizami St. Mon–Sat 9:00–19:00. A consultation is 30 AZN.",
           "Hello! A consultation is 25 AZN and we are open on Sundays too. We are at 12 Nizami St.", "25"),
}


@pytest.mark.skipif(not KEY, reason="no API key")
@pytest.mark.parametrize("lang", list(CASES))
def test_catches_wrong_price(lang):
    facts, answer, bad = CASES[lang]
    r = check(answer, [facts], MODEL)
    assert not r.grounded
    assert any(bad in u for u in r.unsupported), r.unsupported
    assert bad not in r.fixed or "30" in r.fixed, r.fixed


@pytest.mark.skipif(not KEY, reason="no API key")
def test_grounded_answer_unchanged():
    facts, _, _ = CASES["en"]
    ok = "Hello! A consultation is 30 AZN. We are at 12 Nizami St, open Monday to Saturday 9 to 19."
    r = check(ok, [facts], MODEL)
    assert r.grounded and r.score == 1.0


def test_no_sources_is_a_noop():
    r = check("anything", [], "any-model")
    assert r.grounded and r.fixed == "anything"


def _fake_judge(monkeypatch, content):
    """Make urlopen return one chat completion whose message is `content`."""
    import io
    import json
    import urllib.request

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Reply(body))


def test_offline_judge_reply_is_parsed(monkeypatch):
    _fake_judge(monkeypatch, '{"unsupported": ["25 AZN"], "answer": "I do not have the price."}')
    r = check("It costs 25 AZN.", ["It costs 30 AZN."], "m", api_key="x")
    assert r.judged and r.unsupported == ["25 AZN"] and r.fixed == "I do not have the price."


@pytest.mark.parametrize("value", ["null", '"25 AZN"', "{}", "3"])
def test_offline_malformed_unsupported_is_unjudged(monkeypatch, value):
    """Never raises, and never splits a string into characters."""
    _fake_judge(monkeypatch, '{"unsupported": %s, "answer": "x"}' % value)
    r = check("It costs 25 AZN.", ["It costs 30 AZN."], "m", api_key="x")
    assert not r.judged and r.unsupported == [] and not r.grounded


def test_offline_unjudged_has_no_score(monkeypatch):
    """An unusable reply is unknown, so it must not score as fully grounded."""
    _fake_judge(monkeypatch, "")
    r = check("It costs 25 AZN.", ["It costs 30 AZN."], "m", api_key="x")
    assert not r.judged and r.score is None
