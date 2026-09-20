# -*- coding: utf-8 -*-
"""Technical report v3: a trained multilingual detector against the LLM judges
and the English-trained classifiers, 31 languages. All numbers are read from
benchmark/results_v3*.json and train/out/<run>/training.json.
    python3 paper/make_paper_v3.py [run_dir]"""
import json, os, statistics, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image

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

here = os.path.dirname(os.path.abspath(__file__)); B_ = os.path.join(here, '..', 'benchmark')
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'train', 'out', 'run2')
cfg = json.load(open(os.path.join(B_, 'cases_v3.json'), encoding='utf-8'))
langs = list(cfg['languages']); variants = [v for v in cfg['variants'] if v != 'ok']; NL = len(langs)
LANGN = {'en': 'English', 'az': 'Azerbaijani', 'ru': 'Russian', 'tr': 'Turkish', 'uk': 'Ukrainian', 'kk': 'Kazakh', 'ar': 'Arabic', 'fa': 'Persian', 'hi': 'Hindi', 'id': 'Indonesian', 'vi': 'Vietnamese', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'nl': 'Dutch', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'uz': 'Uzbek', 'ka': 'Georgian', 'he': 'Hebrew', 'bn': 'Bengali', 'th': 'Thai', 'sw': 'Swahili', 'ro': 'Romanian', 'cs': 'Czech', 'el': 'Greek', 'hu': 'Hungarian'}
models = {}
for f, prov in (('results_v3-siba.json', 'ours'), ('results_v3.json', 'Groq'), ('results_v3-baselines.json', 'local CPU'), ('results_v2-gemini.json', 'Google, 11 languages')):
    p = os.path.join(B_, f)
    if not os.path.exists(p): continue
    for m, r in json.load(open(p, encoding='utf-8'))['models'].items(): models[m] = (prov, r)
tr = json.load(open(os.path.join(RUN, 'training.json')))
try:
    synth = sum(1 for _ in open(os.path.join(here, '..', 'train', 'data', 'synth.jsonl'), encoding='utf-8'))
except FileNotFoundError:
    synth = 0

def stats(r):
    c = r['calls']
    planted = [x for k, x in c.items() if not k.endswith('/ok') and 'error' not in x]; oks = [x for k, x in c.items() if k.endswith('/ok') and 'error' not in x]
    return dict(caught=sum(1 for x in planted if x.get('caught')), planted=len(planted), fa=sum(1 for x in oks if x.get('false_alarm')), oks=len(oks), lat=statistics.median(r['latency']) if r['latency'] else None, langs=len({k.split('/')[0] for k in c}))
def by_lang(m, l):
    c = models[m][1]['calls']; xs = [x for k, x in c.items() if k.startswith(l + '/') and not k.endswith('/ok') and 'error' not in x]
    return sum(1 for x in xs if x.get('caught')), len(xs), sum(1 for k, x in c.items() if k.startswith(l + '/') and x.get('false_alarm'))
def grid(data, widths, fs=8.6):
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, 0), 'TNR-B', fs), ('FONT', (0, 1), (-1, -1), 'TNR', fs), ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK), ('LINEBELOW', (0, -1), (-1, -1), 0.6, INK), ('LINEABOVE', (0, 0), (-1, 0), 0.6, INK), ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5), ('ALIGN', (1, 1), (-1, -1), 'CENTER')]))
    return t
short = {'vectara/HHEM-2.1-Open': 'HHEM-2.1-Open', 'KRLabsOrg/lettucedect-base-modernbert-en-v1': 'LettuceDetect-base (en)', 'siba/grounded-multilingual-base': 'grounded-multilingual-base (ours)'}
name = lambda m: short.get(m, m)
S_ = {m: stats(models[m][1]) for m in models}
order = sorted(models, key=lambda m: (-(S_[m]['caught'] / max(1, S_[m]['planted'])), S_[m]['fa']))
OURS = 'siba/grounded-multilingual-base'; o = S_[OURS]; hh = S_['vectara/HHEM-2.1-Open']; ld = S_['KRLabsOrg/lettucedect-base-modernbert-en-v1']
judges = [m for m in order if models[m][0] == 'Groq']; bestj = judges[0]; bj = S_[bestj]
NC = NL * 14; NP = NL * 12; NO = NL * 2
ours_zero_fa = sum(1 for l in langs if by_lang(OURS, l)[2] == 0); ours_full = sum(1 for l in langs if by_lang(OURS, l)[0] == 12)
hh_all_fa = [l for l in langs if by_lang('vectara/HHEM-2.1-Open', l)[2] == 2]
worst = sorted(langs, key=lambda l: by_lang(OURS, l)[0])[:3]

E = [P(f'A Multilingual Groundedness Detector That Runs on a Raspberry Pi: Trained Token Classification Against LLM Judges and English-Only Detectors in {NL} Languages', TITLE),
     P('Aghasalim Mustafazada<super>1</super> · Teymur Eyvazov<super>2</super>', AUTH),
     P('<super>1</super> Co-CEO and Head of AI, <super>2</super> Co-CEO and Head of Operations — SIBA, "Süni İntellekt Biznes Avtomatlaşdırma" MMC, Baku, Azerbaijan · September 2026 · v3', AFF),
     P(f'<b>Abstract.</b> Retrieval-augmented assistants fail their users most often by stating a price, hour, address or phone number that their documents do not support. The open-weights detectors for this failure — Vectara HHEM-2.1-Open, LettuceDetect — are trained on English. We extend our planted-error benchmark to {NL} languages across nine scripts ({NC} answers, {NP} with one planted error, {NO} grounded controls; identical errors in every language) and add a third kind of detector: a multilingual token classifier of our own, fine-tuned from XLM-RoBERTa-base on {tr.get("train_answers", 0):,} answers — {synth * 4:,} synthetic business answers in 30 languages whose unsupported spans were planted and marked by open LLMs, plus RAGTruth — with the decision threshold chosen on held-out data. It catches {o["caught"]} of {o["planted"]} planted errors ({100 * o["caught"] / o["planted"]:.0f} %) with {o["fa"]} false alarms in {o["oks"]} grounded answers, in {o["lat"] * 1000:.0f} ms per answer on a laptop CPU, and it serves the public demo at grounded.siba.az from a Raspberry Pi 5. On the same cases HHEM-2.1-Open reaches {100 * hh["caught"] / hh["planted"]:.0f} % recall only because its score falls below the threshold for any non-English text: it flags both grounded answers in {len(hh_all_fa)} of {NL} languages. LettuceDetect catches {100 * ld["caught"] / ld["planted"]:.0f} %. The best open LLM judge ({name(bestj)}) catches {bj["caught"]}/{bj["planted"]} with {bj["fa"]} false alarms at {bj["lat"]:.2f} s per network call — the accuracy ceiling, at ~{bj["lat"] / max(o["lat"], 1e-3):.0f}× the latency and a per-token price. Weights (MIT), data, training and evaluation code and every raw output: github.com/aghasalim/groundedness.', ABS),
     P('<b>Author contributions.</b> A.M. designed the benchmark, the data generation and the model, ran the experiments and wrote the paper. T.E. contributed the operational setting, the business requirements and review.', ABS)]

E += [P('1. Problem', H),
      P('A business assistant answering from its owner\'s documents has one failure that costs money: it says something the documents do not say. The detectors built for this in retrieval-augmented generation are trained and evaluated on English — HHEM-2.1-Open, a cross-encoder for factual consistency; LettuceDetect, a ModernBERT token classifier on RAGTruth; Lynx, a Llama-3 fine-tune under a non-commercial licence. A general LLM used as a judge reads every language but costs a network call per answer, has a per-token price, and cannot run inside a small deployment. Earlier versions of this report (v1, v2) established the first two facts on eleven languages. This version asks the practical question: can a small trained model, cheap enough for a Raspberry Pi, do the job outside English?'),
      P('2. Benchmark', H),
      P(f'The v2 benchmark is kept unchanged and extended. For each language and two domains (a dental clinic, an electronics shop) a facts block and a grounded answer are templates over five slots — street number, price, opening hours, phone number, return window — plus two extra sentences. The runner plants exactly one error per variant by substituting a slot value (30 → 25 AZN; 9:00–19:00 → 9:00–17:00; No. 12 → 14; one phone digit; 14 → 30 days) or appending a sentence (an invented Sunday opening; an added claim). Every language therefore receives the same seven errors and the same grounded control; {NL} languages × 2 domains × 7 answers = {NC} cases, {NP} with a planted error and {NO} grounded. The first eleven language sets (English, Azerbaijani, Russian, Turkish, Ukrainian, Kazakh, Arabic, Persian, Hindi, Indonesian, Vietnamese) were written by hand; the further twenty ({", ".join(LANGN[l] for l in langs[11:])}) were translated from the English templates by openai/gpt-oss-120b, validated automatically for slot integrity, and are marked as machine-translated in the case file; native review is invited. A planted error counts as caught only when a detector\'s named span or claim overlaps the planted text; a score-only detector is credited when it flags the answer.'),
      P('3. The detector', H),
      P(f'<b>Data.</b> No multilingual span-labelled corpus of unsupported claims exists, so we generate one. A prompt asks an open LLM (openai/gpt-oss-120b, gpt-oss-20b, qwen/qwen3.8-27b on Groq; gemini-3.8/3.5/2.5-flash for the first {int(synth * 0.45):,} documents) for one business document in one of 100 domains (clinics, hotels, tariffs, warranties, timetables, job postings, recipes, changelogs…) in one of 30 languages, a customer question, and four answers: one fully supported, three each with one to three unsupported claims — a changed number, an invented feature, an inverted rule, a wrong day, an overstated promise, an added detail — wrapped in ⟦ ⟧ markers by the generating model itself. The markers yield exact character spans without a second labelling pass; documents whose markers are malformed, missing from the error answers or present in the clean answer are rejected ({synth:,} documents kept, {synth * 4:,} answers). Three quarters of the documents are in the eleven benchmark languages, the rest in the further nineteen; the benchmark templates were never shown to the generator. RAGTruth\'s human-labelled English responses ({tr.get("train_answers", 0) - synth * 4 + 260:,} answers, hallucinated ones kept in full and clean ones matched in number) are added so the model also sees real model output on long passages.'),
      P(f'<b>Model.</b> XLM-RoBERTa-base (278M parameters, 100 languages) with a two-way token head. The input is the answer followed by the sources, 384 tokens, the sources truncated first; only answer tokens carry labels (1 inside an unsupported span). Cross-entropy with the positive class weighted {tr.get("pos_weight", 3)}×, AdamW, learning rate {tr.get("lr", "3e-5")}, batch 8, {tr.get("epoch", 0) + 1} epochs on an Apple M4 GPU in under three hours. After training the decision threshold is chosen on a held-out set of documents to maximise answer-level F1 (threshold {tr.get("threshold", 0.5)}; {tr.get("calibration", "")}). At inference a span is a maximal run of answer tokens above threshold; the score is one minus the share of the answer\'s characters inside spans. Served with dynamic int8 quantisation the model answers in about {o["lat"] * 1000:.0f} ms on a laptop core and under a second on a Raspberry Pi 5.')]

E += [P('4. Results', H)]
data = [['Detector', 'Kind', 'Languages', 'Errors caught', 'False alarms', 'Median latency']]
for m in order:
    s = S_[m]; prov, _ = models[m]
    kind = {'ours': 'trained classifier (ours)', 'local CPU': 'trained classifier', 'Groq': 'LLM judge (Groq)'}.get(prov, 'LLM judge (Google)')
    data.append([name(m), kind, str(s['langs']), f"{s['caught']}/{s['planted']} ({100 * s['caught'] / max(1, s['planted']):.0f} %)", f"{s['fa']}/{s['oks']}", f"{s['lat']:.2f} s" if s['lat'] else '—'])
E += [KeepTogether([grid(data, [50, 36, 18, 28, 20, 22]), Spacer(1, 4), P(f'Table 1. Planted errors caught and grounded answers wrongly flagged, all cases each detector was run on. The Google judges were run on the first eleven languages only (credits exhausted); every other row covers all {NL}. Latency for the judges is per network call from Baku; for the classifiers, per inference on a laptop CPU.', SM)])]
data = [['Detector'] + variants]
for m in order:
    c = models[m][1]['calls']; row = [name(m)]
    for v in variants:
        xs = [x for k, x in c.items() if k.endswith('/' + v) and 'error' not in x]; row.append(f"{sum(1 for x in xs if x.get('caught'))}/{len(xs)}")
    data.append(row)
E += [KeepTogether([Spacer(1, 6), grid(data, [50, 17, 17, 18, 17, 15, 18, 17]), Spacer(1, 4), P('Table 2. Recall by error type, all languages. The first five are slot substitutions; "sunday" and "extra" are appended sentences.', SM)])]
heat = os.path.join(here, '..', 'benchmark', 'heatmap.png')
if os.path.exists(heat):
    from PIL import Image as PILImage
    w, h = PILImage.open(heat).size
    E += [Spacer(1, 8), Image(heat, width=166 * mm, height=166 * mm * h / w), P(f'Figure 1. Planted errors caught out of 12 per language. Green: all twelve. Amber: one of the two grounded answers wrongly flagged; red: both — the detector flags everything in that language. Grey: not run.', SM)]
hh_en = by_lang('vectara/HHEM-2.1-Open', 'en')
E += [P(f'<b>Our detector.</b> {o["caught"]}/{o["planted"]} planted errors caught with {o["fa"]}/{o["oks"]} false alarms; all twelve caught in {ours_full} languages and no false alarm in {ours_zero_fa} of {NL}. The weakest languages are {", ".join(f"{LANGN[l]} ({by_lang(OURS, l)[0]}/12)" for l in worst)}. By error type (Table 2) the slot substitutions inside a sentence are the harder cases for a token classifier than the appended sentences, the opposite of the judges, which occasionally under-quote an appended claim. This is the row that did not exist before: a detector under 300 MB, no API, that reads Kazakh, Persian and Vietnamese.'),
      P(f'<b>The judges.</b> {name(bestj)} catches {bj["caught"]}/{bj["planted"]} with {bj["fa"]} false alarms across all {NL} languages at {bj["lat"]:.2f} s per call; the other Groq judges are within a few cases. Accuracy on this benchmark is therefore essentially solved by a prompt — the cost is latency, a network dependency and a per-token bill, which is exactly what the trained model removes.'),
      P(f'<b>HHEM-2.1-Open.</b> {hh["caught"]}/{hh["planted"]} caught reads as {100 * hh["caught"] / hh["planted"]:.0f} % recall, but the false-alarm column explains it: {hh["fa"]}/{hh["oks"]} grounded answers flagged, both of them in {len(hh_all_fa)} languages ({", ".join(LANGN[l] for l in hh_all_fa[:12])}{"…" if len(hh_all_fa) > 12 else ""}). Its consistency score falls under 0.5 for non-English text regardless of content; in English it catches {hh_en[0]} of 12 with no false alarm. It detects the language, not the claim. <b>LettuceDetect</b> catches {ld["caught"]}/{ld["planted"]} with {ld["fa"]}/{ld["oks"]} false alarms, recall falling with script distance from its English training data.')]

E += [P('5. Limitations', H),
      P(f'The benchmark is formulaic by construction — short answers, numeric slots, one error each — and measures whether a detector reads the language and localises a claim, not whether it catches paraphrase, implication or omission; those are the next case set and will lower every row. Twenty of the {NL} language sets are machine translations, checked for slot integrity but not yet by native speakers. The training data is synthetic: the generating models\' own errors and marker placement are the labels, and their style is one register (a business answering a customer); the RAGTruth share is English only. Our detector marks spans but does not rewrite the answer, so a deployment that wants the corrected text still needs a generator for the flagged cases. Threshold calibration used our own held-out documents, not the benchmark. The English classifiers were run with their default thresholds; tuning HHEM per language would raise its English recall and would not repair its behaviour on other languages, whose scores collapse regardless of content.'),
      P('6. Availability', H),
      P('Weights and the serving code, the data generator, the training and calibration scripts, the case templates for all languages, both runners, the raw per-call outputs for every detector, the site and this report\'s generator are at <b>github.com/aghasalim/groundedness</b> under the MIT licence. The detector answers publicly and without a key at <b>grounded.siba.az</b> (POST /check) and through the package (<i>pip install groundedness</i>; <i>check(answer, sources, model="siba")</i>). The same judge runs on every answer of SIBA\'s customer assistant across its channels.'),
      P('<b>Reproduce.</b>'), P('python train/gen_data.py --calls 4000 &amp;&amp; python train/prepare.py<br/>python train/train.py --epochs 3 --bs 8 --pos-weight 5 --out train/out/run2<br/>python train/eval_bench.py train/out/run2 &amp;&amp; bash benchmark/run_v3.sh', CODE)]
E += [P('References', H),
      P('Vectara. HHEM-2.1-Open and the Hallucination Leaderboard. huggingface.co/vectara/hallucination_evaluation_model.', SM),
      P('Kovács, Á. et al. LettuceDetect: A Hallucination Detection Framework for RAG Applications. arXiv:2502.17125, 2025.', SM),
      P('Ravi, S. et al. Lynx: An Open Source Hallucination Evaluation Model. arXiv:2407.08488, 2024.', SM),
      P('Niu, C. et al. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. ACL 2024.', SM),
      P('Conneau, A. et al. Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). ACL 2020.', SM),
      P('Mustafazada, A. Do English Hallucination Detectors Work in Other Languages? Technical report v2, SIBA, 2026. github.com/aghasalim/groundedness.', SM),
      P('Aghayev, F., Ahmadbayli, E. SIMURG: Zero-Leak Online Detection of LLM Decoding Corruption in Production Streams. HAL-X AI, 2026.', SM)]

def deco(c, d):
    c.saveState(); c.setFont('TNR', 8); c.setFillColor(GRAY)
    c.drawString(22 * mm, 12 * mm, 'Mustafazada, Eyvazov — A Multilingual Groundedness Detector That Runs on a Raspberry Pi'); c.drawRightString(188 * mm, 12 * mm, str(c.getPageNumber())); c.restoreState()
out = os.path.join(here, 'groundedness-multilingual-detector-v3.pdf')
doc = BaseDocTemplate(out, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm, title='A Multilingual Groundedness Detector That Runs on a Raspberry Pi', author='Aghasalim Mustafazada, Teymur Eyvazov')
doc.addPageTemplates([PageTemplate(id='n', frames=[Frame(22 * mm, 20 * mm, 166 * mm, A4[1] - 40 * mm, id='f', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=deco)])
doc.build(E)
print('OK ->', out, os.path.getsize(out), 'bytes')
