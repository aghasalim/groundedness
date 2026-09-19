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

## Roadmap

- A benchmark: the same planted-error set in ten non-English languages, and a table of which models judge best. There is no multilingual hallucination leaderboard yet.
- A DeepEval / RAGAS metric that wraps this.

## Origin

Extracted from the groundedness judge that runs on every channel of [SIBA](https://siba.az)'s assistant. MIT.
