Every open-source hallucination detector is English-only. So I tested the obvious alternative in eleven languages.

HHEM, LettuceDetect, Lynx — the models people use to catch a RAG bot saying "the consultation is 25 AZN" when the document says 30 — are all trained on English. If your users write Azerbaijani, Kazakh, Persian or Vietnamese, you have nothing.

The alternative is boring: ask a capable general model to be the judge, in the answer's own language. Does it actually work outside English? I planted errors — a wrong price, an invented opening day — in the same clinic scenario in 11 languages (en, az, ru, tr, uk, kk, ar, fa, hi, id, vi) and ran 9 models.

Results:
• 7 of 9 models caught 22/22 planted errors with 0 false alarms, in every language, Kazakh and Persian included
• qwen3.8-27b on Groq does it in 0.27 s median — 10× faster than Gemini Flash, 40× faster than Gemini 2.5 Pro, same recall
• a 7B Arabic-tuned model flagged every correct answer and missed a third of the errors. Below ~20B it isn't a judge

One trap worth sharing: my first Gemini run showed 2.5 Pro at 0/22 caught, 0 false alarms — a perfect-looking failure. Thinking models spend the token budget before the first visible character; the reply came back empty and my scorer counted "nothing unsupported" as grounded. Never let the absence of a reply score as a pass.

It's a package now: pip install groundedness. One call, any OpenAI-compatible model, zero dependencies, MIT. It returns the exact unsupported claims and a rewrite without them — something you can highlight in a UI or hand to a human. The same judge runs on every channel of SIBA's assistant, which is where it came from.

Code, the 11-language case file and both result tables: github.com/aghasalim/groundedness

The benchmark is small on purpose (33 answers per model, one domain). If you want to add a language, it's one JSON entry — PRs welcome.

#LLM #RAG #hallucination #NLP #Azerbaijan
