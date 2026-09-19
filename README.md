# groundedness

**Did the model make this up?** A claim-level groundedness check for LLM answers: one call, any OpenAI-compatible model, any language.

```python
from groundedness import check

facts  = "Diş klinikası, Nizami küç. 12. B.e–Şənbə 9:00–19:00. Konsultasiya 30 AZN."
answer = "Salam! Konsultasiya 25 AZN-dir, bazar günü də işləyirik. Ünvan Nizami küçəsi 12."

r = check(answer, sources=[facts], model="qwen/qwen3.8-27b")
r.unsupported  # ['Konsultasiya 25 AZN-dir', 'bazar günü də işləyirik']
r.fixed        # 'Salam! Konsultasiya haqqı və bazar günü işləyib-işləmədiyimiz haqqında məlumatım yoxdur. Ünvan Nizami küçəsi 12-dir.'
r.score        # 0.47
```

That is Azerbaijani. It works the same in Russian, Turkish, Arabic or English, because the judge reads the answer in its own language. Every trained hallucination detector published so far — Vectara HHEM, LettuceDetect, Patronus Lynx — is English-only.

## Why

RAG apps and support bots answer from documents. The failure that hurts is not a looping decode or a made-up historical date; it is *"a consultation is 25 AZN"* when the document says 30. `groundedness` finds that sentence, names it, and gives you the answer without it.

- **Any model.** Groq, OpenAI, Ollama, vLLM, OpenRouter — anything that speaks `/chat/completions`. No model to download, no GPU.
- **Any language.** No training data, no language list. If the model can read it, the judge can check it.
- **Claim-level output.** Not a score you cannot act on: the exact unsupported claims, and a rewrite. Highlight them, hand off to a human, or log the rate.
- **Zero dependencies.** `urllib` and `json`. Python 3.9+.

## Install

```
pip install groundedness
```

Set `OPENAI_BASE_URL` and `OPENAI_API_KEY`, or just `GROQ_API_KEY` (Groq's endpoint is the default when it is set). Or pass `base_url=` / `api_key=` to `check()`.

## CLI

```
echo "the answer" | groundedness --model qwen/qwen3.8-27b --sources facts.txt policy.md
```

Prints JSON: `grounded`, `score`, `unsupported`, `fixed`.

## What it is not

It checks an answer against **the sources you give it**, not against the world. A true claim that is not in your documents is reported as unsupported — which is what you want from a bot that must only say what the owner wrote. It is a judge call, so it costs one short completion per answer (about 350 tokens on the example above) and it is as good as the model judging; the tests use `qwen/qwen3.8-27b` on Groq, which caught every planted error in az/ru/en.

## Tests

```
GROQ_API_KEY=... python -m pytest -q
```

Three languages, one planted wrong price and one invented opening day each; plus a grounded answer that must come back unchanged.

## Benchmark: which model judges best, in eleven languages

The same case in en, az, ru, tr, uk, kk, ar, fa, hi, id, vi: a clinic's facts, a grounded answer that must pass, and two answers with one planted error each (a wrong price, an invented opening day). Every chat model on Groq's free tier, 19 September 2026. Reproduce with `python benchmark/run.py`; full per-language grid in [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

| Model | Planted errors caught | Grounded answers wrongly flagged | Median latency |
|---|---:|---:|---:|
| `openai/gpt-oss-120b` | 22/22 | 0/11 | 0.58 s |
| `openai/gpt-oss-20b` | 22/22 | 0/11 | 0.48 s |
| `qwen/qwen3.8-27b` | 22/22 | 0/11 | 0.27 s |
| `allam-2-7b` | 15/22 | 11/11 | 0.25 s |

And Google's models over their OpenAI-compatible endpoint (`python benchmark/run.py --gemini`, results in [`benchmark/gemini.md`](benchmark/gemini.md)):

| Model | Planted errors caught | Grounded answers wrongly flagged | Median latency |
|---|---:|---:|---:|
| `gemini-3.8-flash` | 22/22 | 0/11 | 2.7 s |
| `gemini-3.5-flash` | 22/22 | 0/11 | 4.4 s |
| `gemini-2.5-flash` | 22/22 | 0/11 | 4.3 s |
| `gemini-2.5-pro` | 21/21 | 0/11 | 10.6 s |

Seven models are perfect across all eleven languages, including Kazakh, Persian and Vietnamese, with no false alarms; `qwen/qwen3.8-27b` on Groq is the fastest by ten times. `allam-2-7b` flags every grounded answer and misses a third of the errors: a 7B Arabic-centred model is not a judge. One `gemini-2.5-pro` call (Azerbaijani, wrong price) hit a transient request error and is left out rather than guessed.

A lesson from the first run, kept here so nobody repeats it: thinking models spend their token budget before the first visible character. With `max_tokens: 1200` Gemini 2.5 Pro returned empty replies and an early version of this package scored an empty reply as "grounded" — 0/22 caught, 0 false alarms, a perfect-looking failure. The package now returns `judged=False` for an empty or non-JSON reply, and the benchmark counts it as a miss. The set is small on purpose (33 answers per model, one domain) — it is a smoke test that a model can read the language and follow the instruction, not a measure of fine judgement. Adding a language is one JSON entry in `benchmark/cases.json`; adding a model is one endpoint that lists it.

## Roadmap

- More domains per language (a return policy, a timetable, a contract clause) and harder errors: a right number in the wrong place, a plausible synonym.
- Models beyond Groq: run the same file against OpenRouter, Ollama, vLLM.
- A DeepEval / RAGAS metric that wraps this.

## Origin

Extracted from the groundedness judge that runs on every channel of [SIBA](https://siba.az)'s assistant. MIT.
