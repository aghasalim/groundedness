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
