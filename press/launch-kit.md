# Launch kit — grounded.siba.az · groundedness

Bracketed values `[…]` are filled from the final results at 10:00. Post order: HN first (it needs your reply within the first hour), then X, LinkedIn, Reddit, HF, email, press.

---

## 1. Hacker News — Show HN

**Title** (80 chars max):
`Show HN: Grounded – multilingual hallucination detector that runs on a Raspberry Pi`

**URL:** https://grounded.siba.az

**First comment (post it immediately after submitting):**

Hi HN, author here. We build document-grounded assistants for businesses in Azerbaijan, and the failure that costs money is never a made-up historical date — it's "a consultation is 25 AZN" when the document says 30. The open detectors for this (Vectara HHEM-2.1, LettuceDetect, Lynx) are trained on English, so we measured whether they transfer.

They don't. On a planted-error benchmark (the same 7 errors + 1 grounded control, per language, across [31] languages / [434] cases) HHEM's consistency score falls below the threshold for any non-English text whatever the content: in Russian, Arabic, Persian and Indonesian it flags every answer, grounded or not. LettuceDetect gets 2/12 in Kazakh and Hindi.

What we're releasing:
- `pip install groundedness` — a zero-dependency claim-level check that asks any OpenAI-compatible model to list the unsupported claims in the answer's own language and rewrite without them. Open 20–120B models catch [132/132] with 0 false alarms; a 27B model does it in 0.36 s.
- Our own detector: an XLM-R token classifier trained on [~12k] synthetic labelled answers in 30 languages + RAGTruth. It runs on a Raspberry Pi 5 behind the demo (that's what the default model on the page is), [X/132] caught, [Y/22] false alarms, ~[Z] ms on CPU. MIT, weights on HF.
- The benchmark, every raw judge reply, and the training scripts: https://github.com/aghasalim/groundedness

Honest limits: the benchmark is formulaic (short answers, one numeric error each) — it tells you whether a detector *reads the language*, not whether it catches subtle paraphrase. 20 of the 31 language sets were machine-translated from English templates and slot-checked; native review welcome, one JSON entry per language. Happy to answer anything.

---

## 2. X / Twitter thread

1/ The most-downloaded open hallucination detector flags every non-English answer as a hallucination.

We tested Vectara HHEM and LettuceDetect on the same planted errors in [31] languages. Then we trained one that works — and put it on a Raspberry Pi.

grounded.siba.az 🧵

2/ The benchmark: a clinic and a shop, 7 planted errors each (wrong price, hours, street number, phone digit, return window, invented Sunday, added claim) + 1 grounded control, identical in every language. [434] cases. [image: heatmap]

3/ HHEM-2.1-Open: 79% "recall" — but in Russian, Arabic, Persian, Indonesian it flags all 22 grounded answers too. Its score measures "is this English", not "is this supported". LettuceDetect: 2/12 in Kazakh and Hindi.

4/ Open LLM judges (gpt-oss-120b, qwen3.8-27b on @GroqInc) catch [132/132] with 0 false alarms, in every script. One prompt, `pip install groundedness`, zero deps, any OpenAI-compatible endpoint.

5/ Our own model: XLM-R token classifier, [~12k] labelled answers in 30 languages. [X/132] caught, [Y/22] false alarms, [Z] ms — on a Raspberry Pi 5 in Baku. MIT. Weights + training code in the repo.

6/ Try it (no key, unlimited): grounded.siba.az
Code + data + paper: github.com/aghasalim/groundedness
Built at @siba_az. Native speakers of any language: the case file is one JSON entry, PRs welcome.

---

## 3. LinkedIn (profile + SIBA page) — attach heatmap.png

The most-downloaded open hallucination detector thinks every Azerbaijani, Russian or Arabic answer is a hallucination.

We found this while building document-grounded assistants at SIBA. The tools that catch "the bot said 25 AZN, the document says 30" — Vectara HHEM, LettuceDetect — are trained on English only. So we built a benchmark: the same 7 planted errors + 1 grounded control, in [31] languages, [434] cases.

Results:
• HHEM-2.1-Open flags every non-English answer, grounded or not. Its score measures the language, not the claim.
• LettuceDetect: 8/12 in Latin-script languages, 2/12 in Kazakh and Hindi.
• Open LLM judges catch [132/132] with zero false alarms — in every script.
• Our own multilingual detector, trained on [~12k] labelled answers in 30 languages, catches [X/132] with [Y/22] false alarms — running on a Raspberry Pi in Baku, ~[Z] ms per check.

Everything is open: `pip install groundedness` (MIT), the live demo at grounded.siba.az (no key, unlimited), the model weights, the benchmark, every raw judge reply, and the technical report.

Co-authored with Teymur Eyvazov. Native speakers: adding a language is one JSON entry — we'd love your review.

#AI #LLM #hallucination #RAG #NLP #Azerbaijan #opensource

---

## 4. Reddit

**r/MachineLearning** — title: `[R] English hallucination detectors don't transfer: a 31-language planted-error benchmark, plus a multilingual detector that runs on a Raspberry Pi`
Body: the HN first comment, minus the "Hi HN" line, plus the paper link.

**r/LocalLLaMA** — title: `Multilingual hallucination/groundedness detector, MIT, runs on a Raspberry Pi 5 — and why HHEM fails outside English`
Body: lead with the Pi + `pip install` + "any local OpenAI-compatible endpoint (Ollama, vLLM) works as the judge", then the finding.

**r/LanguageTechnology** — title: `Do English-trained hallucination detectors work in other languages? Benchmark in 31 languages (paper + code)`

---

## 5. Hugging Face model card — `siba/grounded-multilingual-base`

```
---
language: [en, az, ru, tr, uk, kk, ar, fa, hi, id, vi, de, fr, es, it, pt, pl, nl, zh, ja, ko, uz, ka, he, bn, th, sw, ro, cs, el, hu]
license: mit
base_model: FacebookAI/xlm-roberta-base
pipeline_tag: token-classification
tags: [hallucination-detection, groundedness, rag, multilingual, factual-consistency]
datasets: [wandb/RAGTruth-processed]
---
# grounded-multilingual-base

Token classifier: which spans of an assistant's answer are **not supported** by the sources it was given. Input `<s> answer </s></s> sources </s>`; label 1 = unsupported.

Trained on [N] synthetic labelled answers in 30 languages (business documents, 100 domains, errors planted and marked by open LLMs) plus [M] RAGTruth answers. Fine-tuned from xlm-roberta-base, 384 tokens, [E] epochs, Apple M4.

Benchmark ([31] languages, [434] cases, github.com/aghasalim/groundedness): [X/132] planted errors caught, [Y/22] false alarms, [Z] ms on a Raspberry Pi 5 (int8). HHEM-2.1-Open on the same cases: 104/132 but 12/22 false alarms.

Use: `pip install groundedness` then `check(answer, sources, model="siba")` (hosted), or load with transformers and the `detect.py` in the repo.

Limits: short-answer business domain; long documents are truncated to 384 tokens; not a fact checker — it compares against the sources you give it.
```

---

## 6. Email to the LettuceDetect / Vectara authors (send before posting)

Subject: Your detector on 31 languages — results + a multilingual alternative (open)

Hi [name],

I ran [HHEM-2.1-Open / lettucedect-base-modernbert-en-v1] on a planted-error benchmark in 31 languages ([434] cases: the same 7 errors + 1 grounded control per language). Per-language table and every raw output: github.com/aghasalim/groundedness/blob/main/benchmark/results_v3-baselines.md

Short version: [HHEM's score falls below 0.5 for any non-English text, so it flags grounded answers in Russian/Arabic/Persian/Indonesian 22/22 · LettuceDetect gets 8/12 in Latin-script languages and 2/12 in Kazakh and Hindi]. In English the behaviour matches your card.

I'm publishing this on [date] with a multilingual detector we trained (MIT). If any of it misrepresents your model I'd rather fix it before posting — and if you'd like the case file for your own eval it's one JSON.

Aghasalim Mustafazada, SIBA, Baku

---

## 7. Press — see press-release-az.md (Azerbaijani). English one-paragraph version for international outlets:

Baku-based SIBA has released an open-source multilingual hallucination detector and the first leaderboard measuring whether English-trained detectors work in other languages. On a 434-case benchmark in 31 languages, the most-downloaded open detector (Vectara HHEM) flags every non-English answer as a hallucination; SIBA's model, trained on 30 languages, catches [X] of 132 planted errors with [Y] false alarms and runs on a Raspberry Pi. Demo, code, weights and paper: grounded.siba.az · github.com/aghasalim/groundedness.
