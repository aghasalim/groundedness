# -*- coding: utf-8 -*-
"""The technical report as a PDF, for SSRN / arXiv. python3 paper/make_paper.py"""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, KeepTogether

S = '/System/Library/Fonts/Supplemental/'
for n, f in (('TNR', 'Times New Roman.ttf'), ('TNR-B', 'Times New Roman Bold.ttf'), ('TNR-I', 'Times New Roman Italic.ttf')):
    pdfmetrics.registerFont(TTFont(n, S + f))
pdfmetrics.registerFontFamily('TNR', normal='TNR', bold='TNR-B', italic='TNR-I', boldItalic='TNR-B')
INK, GRAY = HexColor('#111111'), HexColor('#555555')
st = lambda n, **k: ParagraphStyle(n, fontName=k.pop('fontName', 'TNR'), textColor=k.pop('textColor', INK), **k)
TITLE = st('t', fontSize=17, leading=21, alignment=TA_CENTER, fontName='TNR-B', spaceAfter=6)
AUTH = st('a', fontSize=10.5, leading=14, alignment=TA_CENTER, spaceAfter=2)
AFF = st('af', fontSize=9.5, leading=13, alignment=TA_CENTER, textColor=GRAY, spaceAfter=14)
H = st('h', fontSize=11.5, leading=15, fontName='TNR-B', spaceBefore=10, spaceAfter=4)
B = st('b', fontSize=10.2, leading=14, alignment=TA_JUSTIFY, spaceAfter=5)
ABS = st('ab', fontSize=9.8, leading=13.2, alignment=TA_JUSTIFY, leftIndent=18, rightIndent=18, spaceAfter=8)
SM = st('s', fontSize=8.6, leading=11.5, textColor=GRAY, alignment=TA_JUSTIFY, spaceAfter=6)
CODE = st('c', fontSize=8.8, leading=11.5, fontName='Courier', leftIndent=12, spaceAfter=6)
P = lambda t, s=B: Paragraph(t, s)

here = os.path.dirname(os.path.abspath(__file__))
groq = json.load(open(os.path.join(here, '..', 'benchmark', 'results.json'), encoding='utf-8'))
gem = json.load(open(os.path.join(here, '..', 'benchmark', 'gemini.json'), encoding='utf-8'))
langs = groq['languages']

def rows(res, provider):
    out = []
    for m, r in sorted(res['models'].items(), key=lambda kv: (-(kv[1]['recall'] or 0), kv[1]['false_alarms'], kv[1]['median_latency_s'] or 99)):
        full = sum(1 for x in r['per_language'].values() if x['price'] is True and x['day'] is True)
        out.append([m, provider, f"{r['caught']}/{r['planted']}", f"{r['false_alarms']}/{r['grounded_cases']}", f"{full}/{len(langs)}", f"{r['median_latency_s']:.2f} s" if r['median_latency_s'] else '—'])
    return out

def grid(data, widths):
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, 0), 'TNR-B', 8.8), ('FONT', (0, 1), (-1, -1), 'TNR', 8.8), ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK), ('LINEBELOW', (0, -1), (-1, -1), 0.6, INK), ('LINEABOVE', (0, 0), (-1, 0), 0.6, INK), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('ALIGN', (2, 1), (-1, -1), 'CENTER')]))
    return t

E = [P('Multilingual Groundedness Verification with LLM Judges: A Planted-Error Benchmark in Eleven Languages', TITLE),
     P('Aghasalim Mustafazada', AUTH),
     P('SIBA — "Süni İntellekt Biznes Avtomatlaşdırma" MMC, Baku, Azerbaijan · salim@siba.az · September 2026', AFF),
     P('<b>Abstract.</b> Retrieval-augmented assistants fail their users most often not by looping or inventing history but by stating a wrong price, hour or address that the documents they were given do not support. Every published open-weights detector for this failure (Vectara HHEM, LettuceDetect, Patronus Lynx) is trained on English. We describe <i>groundedness</i>, a dependency-free open-source package that checks an answer against its sources by asking a general-purpose language model to list the unsupported claims and rewrite the answer without them, in the answer\'s own language, over any OpenAI-compatible endpoint. We introduce a small planted-error benchmark: one business\'s facts, one grounded answer and two answers with a single planted error (a wrong price, an invented opening day) in eleven languages (English, Azerbaijani, Russian, Turkish, Ukrainian, Kazakh, Arabic, Persian, Hindi, Indonesian, Vietnamese). Seven of nine models tested catch all 22 planted errors with zero false alarms in every language, including Kazakh, Persian and Vietnamese; a 27B open model does so with a median latency of 0.27 s. A 7B Arabic-centred model flags every grounded answer and misses a third of the errors. We also report a measurement pitfall: thinking models given a small token budget return empty replies, and a scorer that treats an empty reply as "grounded" produces a perfect-looking zero. Code, data and results are at github.com/aghasalim/groundedness under the MIT licence.', ABS)]

E += [P('1. Problem', H),
      P('A business assistant that answers from the owner\'s documents has one failure that matters commercially: it says something the documents do not say. The Vectara hallucination leaderboard measures a related quantity, unsupported content in summaries, for roughly 150 models, in English. The open-weights detectors built for retrieval-augmented generation — HHEM-2.1 (a cross-encoder), LettuceDetect (a token classifier on RAGTruth) and Lynx (a Llama-3 fine-tune) — are likewise English-only, and Lynx is licensed for non-commercial use. A developer building for Azerbaijani, Kazakh or Vietnamese users has no detector at all.'),
      P('The question this report asks is narrow: does the obvious alternative — asking a capable general model to act as the judge, in the language of the answer — actually work outside English, and which models can be trusted to do it?')]

E += [P('2. Method', H),
      P('<b>The judge.</b> A single chat completion. The system prompt carries the sources as numbered facts and instructs the model to list every factual claim in the answer that the facts do not support (a price, a date, a time, an address, a phone, a name, an availability, a rule, a number, a quote), to treat greetings and offers to help as non-claims, and to rewrite the answer in the same language and tone with the unsupported claims removed, replying in JSON with fields <i>unsupported</i> and <i>answer</i>. Temperature 0. The package parses the JSON, strips code fences, and returns the unsupported list, the rewrite and a score (one minus the share of the answer\'s characters the unsupported claims occupy). An empty or non-JSON reply is returned as <i>judged = false</i> and is never treated as grounded (Section 4).'),
      P('<b>The benchmark.</b> One scenario, a dental clinic: address, opening days and hours, consultation price, phone. For each of eleven languages a native-form set was written by hand: the facts; a grounded answer that restates them; an answer with the price changed from 30 to 25; and an answer that claims Sunday opening when the facts say Monday to Saturday. A planted error counts as caught when the judge reports the answer as not grounded <i>and</i> the planted text appears in the unsupported list; a grounded answer counts as a false alarm when it is reported as not grounded. Eleven languages × 3 answers = 33 calls per model.'),
      P('<b>Models.</b> Every chat model available on Groq\'s free tier on 19 September 2026 (openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b, allam-2-7b) and Google\'s current Gemini line over its OpenAI-compatible endpoint (gemini-2.5-flash, gemini-2.5-pro, gemini-3.5-flash, gemini-3.8-flash). Calls were paced and retried on rate limits; latency is the median wall-clock time per call from a laptop in Baku.')]

E += [P('3. Results', H)]
data = [['Model', 'Provider', 'Errors caught', 'False alarms', 'Languages perfect', 'Median latency']] + rows(groq, 'Groq') + rows(gem, 'Google')
E += [KeepTogether([grid(data, [46, 18, 24, 22, 28, 24]), Spacer(1, 4),
      P('Table 1. Planted errors caught out of 22 (11 languages × 2), grounded answers wrongly flagged out of 11, languages in which both planted errors were caught, and median latency per call. One gemini-2.5-pro call (Azerbaijani, wrong price) returned a transient request error and is excluded rather than guessed, hence 21.', SM)])]
E += [P('Seven models are perfect: every planted error caught, no grounded answer flagged, in all eleven languages. The scripts span Latin (with Azerbaijani, Turkish and Vietnamese diacritics), Cyrillic (Russian, Ukrainian, Kazakh), Arabic and Perso-Arabic, and Devanagari; none of them changed the outcome for these models. The fastest perfect judge, qwen/qwen3.8-27b on Groq, answers in a median 0.27 s, an order of magnitude faster than the Gemini Flash models and forty times faster than gemini-2.5-pro, at no loss of recall on this set.'),
      P('The one clear failure is allam-2-7b, a 7B model tuned for Arabic: it reports every one of the 11 grounded answers as unsupported and misses 7 of 22 planted errors. It is the only model here below roughly 20B parameters, and the pattern — over-flagging plus misses — is what an instruction-following limit looks like, not a language limit: it fails in English too.')]

E += [P('4. A measurement pitfall', H),
      P('The first Gemini run produced a table in which gemini-2.5-pro caught 0 of 22 errors with 0 false alarms and gemini-2.5-flash caught 7. Both numbers were artefacts. The judge was called with a 1,200-token output budget; these models reason before their first visible character and had spent the budget on reasoning, returning an empty <i>content</i> field. An early version of the package parsed an empty reply as "no unsupported claims" and scored it as grounded. A perfectly grounded-looking model that never says anything is exactly the false result a benchmark author is most likely to publish without noticing, because it contains no false alarms.'),
      P('The corrected package returns <i>judged = false</i> for an empty or non-JSON reply and the benchmark counts that as a miss; the output budget is 8,000 tokens. Re-run, gemini-2.5-pro caught 21 of 21. We recommend that any LLM-as-judge evaluation report the count of unjudged calls separately from misses, and never let the absence of a reply score as a pass.')]

E += [P('5. Limitations', H),
      P('The benchmark is deliberately small: one domain, two error types, 33 answers per model, planted by hand by one author who is a native speaker of Azerbaijani and reads the other languages to varying degrees. It establishes that a model can read each language and follow the instruction; it does not measure fine judgement — a right number in the wrong sentence, a plausible synonym, a claim entailed but not stated. The perfect scores should be read as "not a language problem", not as "solved". Eleven languages were chosen for script diversity and for the markets the author works in; adding a language is one JSON entry. Only two providers were run, for cost; the runner takes any OpenAI-compatible endpoint and a list of model names.'),
      P('The judge checks against the sources it is given, not against the world: a true claim absent from the sources is reported as unsupported. For a business assistant that must only say what its owner wrote, that is the desired behaviour; for open-domain question answering it is not the right tool.')]

E += [P('6. Availability', H),
      P('The package (<i>pip install groundedness</i>; Python 3.9+, no dependencies), the eleven-language case file, the runner and both result sets are at <b>github.com/aghasalim/groundedness</b> under the MIT licence. The judge described here also runs on every channel of SIBA\'s assistant (siba.az), where each channel reports the share of answers that passed; that deployment is the origin of the method and the source of future cases.'),
      P('<b>Reproduce.</b>', B), P('pip install groundedness pytest<br/>GROQ_API_KEY=... python benchmark/run.py<br/>GEMINI_API_KEY=... python benchmark/run.py --gemini', CODE)]

E += [P('References', H),
      P('Vectara. Hallucination Leaderboard. github.com/vectara/hallucination-leaderboard, 2024–2026.', SM),
      P('Kovács, Á. et al. LettuceDetect: A Hallucination Detection Framework for RAG Applications. arXiv:2502.17125, 2025.', SM),
      P('Ravi, S. et al. Lynx: An Open Source Hallucination Evaluation Model. arXiv:2407.08488, 2024.', SM),
      P('Manakul, P., Liusie, A., Gales, M. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023.', SM),
      P('Farquhar, S. et al. Detecting hallucinations in large language models using semantic entropy. Nature 630, 2024.', SM),
      P('Aghayev, F., Ahmadbayli, E. SIMURG: Zero-Leak Online Detection of LLM Decoding Corruption in Production Streams. HAL-X AI, 2026.', SM)]

def deco(c, d):
    c.saveState(); c.setFont('TNR', 8); c.setFillColor(GRAY)
    c.drawString(22 * mm, 12 * mm, 'Mustafazada — Multilingual Groundedness Verification with LLM Judges'); c.drawRightString(188 * mm, 12 * mm, str(c.getPageNumber())); c.restoreState()

out = os.path.join(here, 'groundedness-eleven-languages.pdf')
doc = BaseDocTemplate(out, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm, title='Multilingual Groundedness Verification with LLM Judges', author='Aghasalim Mustafazada')
doc.addPageTemplates([PageTemplate(id='n', frames=[Frame(22 * mm, 20 * mm, 166 * mm, A4[1] - 40 * mm, id='f', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=deco)])
doc.build(E)
print('OK ->', out, os.path.getsize(out), 'bytes')
