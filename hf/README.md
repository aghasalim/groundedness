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

Trained on 2,844 synthetic labelled answers in 30 languages (business documents, 100 domains, errors planted and marked by open LLMs) plus 3,883 RAGTruth answers. Fine-tuned from xlm-roberta-base, 384 tokens, 3 epochs, Apple M4.

Benchmark (31 languages, 434 cases, github.com/aghasalim/groundedness): 352/372 planted errors caught, 32/62 false alarms, ~400 ms on a Raspberry Pi 5 (int8). HHEM-2.1-Open on the same cases: 265/372 but 22/62 false alarms.

Use: `pip install groundedness` then `check(answer, sources, model="siba")` (hosted), or load with transformers and the `detect.py` in the repo.

Limits: short-answer business domain; long documents are truncated to 384 tokens; not a fact checker — it compares against the sources you give it.
