# -*- coding: utf-8 -*-
"""Technical report v2: LLM judges vs the English-trained detectors, 154 cases, eleven languages.
python3 paper/make_paper_v2.py"""
import json, os, statistics
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
B_ = os.path.join(here, '..', 'benchmark')
cfg = json.load(open(os.path.join(B_, 'cases_v2.json'), encoding='utf-8'))
langs = list(cfg['languages']); variants = [v for v in cfg['variants'] if v != 'ok']
LANGN = {'en': 'English', 'az': 'Azerbaijani', 'ru': 'Russian', 'tr': 'Turkish', 'uk': 'Ukrainian', 'kk': 'Kazakh', 'ar': 'Arabic', 'fa': 'Persian', 'hi': 'Hindi', 'id': 'Indonesian', 'vi': 'Vietnamese'}
models = {}
for f, prov in (('results_v2.json', 'Groq'), ('results_v2-gemini.json', 'Google'), ('results_v2-baselines.json', 'local CPU')):
    d = json.load(open(os.path.join(B_, f), encoding='utf-8'))
    for m, r in d['models'].items():
        models[m] = (prov, r)

def stats(r):
    c = r['calls']
    planted = [x for k, x in c.items() if not k.endswith('/ok') and 'error' not in x]
    oks = [x for k, x in c.items() if k.endswith('/ok') and 'error' not in x]
    return dict(caught=sum(1 for x in planted if x.get('caught')), planted=len(planted), fa=sum(1 for x in oks if x.get('false_alarm')), oks=len(oks),
                unj=sum(1 for x in c.values() if x.get('unjudged')), err=sum(1 for x in c.values() if 'error' in x), lat=statistics.median(r['latency']) if r['latency'] else None)

def grid(data, widths, fs=8.6):
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, 0), 'TNR-B', fs), ('FONT', (0, 1), (-1, -1), 'TNR', fs), ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK), ('LINEBELOW', (0, -1), (-1, -1), 0.6, INK), ('LINEABOVE', (0, 0), (-1, 0), 0.6, INK), ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5), ('ALIGN', (1, 1), (-1, -1), 'CENTER')]))
    return t

short = {'vectara/HHEM-2.1-Open': 'HHEM-2.1-Open', 'KRLabsOrg/lettucedect-base-modernbert-en-v1': 'LettuceDetect-base (en)'}
name = lambda m: short.get(m, m)
order = sorted(models, key=lambda m: (-(stats(models[m][1])['caught'] / max(1, stats(models[m][1])['planted'])), stats(models[m][1])['fa']))
S_ = {m: stats(models[m][1]) for m in models}
judges = [m for m in order if models[m][0] != 'local CPU']
best = [m for m in judges if S_[m]['caught'] == S_[m]['planted'] and S_[m]['fa'] == 0]

E = [P('Do English Hallucination Detectors Work in Other Languages? LLM Judges Against HHEM and LettuceDetect on a Planted-Error Benchmark in Eleven Languages', TITLE),
     P('Aghasalim Mustafazada<super>1</super> · Teymur Eyvazov<super>2</super>', AUTH),
     P('<super>1</super> Co-CEO and Head of AI, <super>2</super> Co-CEO and Head of Operations — SIBA, "Süni İntellekt Biznes Avtomatlaşdırma" MMC, Baku, Azerbaijan · September 2026 · v2', AFF),
     P(f'<b>Abstract.</b> Retrieval-augmented assistants fail their users most often by stating a price, hour, address or phone number that the documents they were given do not support. The open-weights detectors built for this failure — Vectara HHEM-2.1-Open and LettuceDetect — are trained on English. We ask whether they transfer, and whether the plain alternative, a general language model asked to list the unsupported claims in the answer\'s own language, does. We build a planted-error benchmark: two businesses (a dental clinic, an electronics shop), eleven languages across five scripts (English, Azerbaijani, Russian, Turkish, Ukrainian, Kazakh, Arabic, Persian, Hindi, Indonesian, Vietnamese), and for each a grounded answer plus seven single-error variants — a wrong price, wrong hours, a wrong street number, one phone digit changed, a wrong return window, an invented Sunday opening and an added claim — planted by slot substitution so every language receives identical errors: 154 answers, 132 of them wrong. Two open LLM judges ({", ".join(name(m) for m in best)}) catch all 132 with no false alarms; a 27B open model catches 131 in a median 0.36 s. HHEM-2.1-Open reaches 79 % recall, but its consistency score falls below the threshold for any non-English text: in Russian, Arabic, Persian and Indonesian it flags every answer, grounded or not, so its recall measures the language, not the claim; in English it catches 5 of 12. LettuceDetect catches 49 % overall, 8 of 12 in the Latin-script languages and 2 of 12 in Kazakh and Hindi. We also document a measurement pitfall in which an empty reply from a thinking model scores as "grounded", and the package fix. Code, data, raw outputs and the runner are at github.com/aghasalim/groundedness (MIT; <i>pip install groundedness</i>).', ABS)]

E += [P('<b>Author contributions.</b> A.M. designed the benchmark, wrote the package and the cases, ran the experiments and wrote the paper. T.E. contributed the operational setting, the business requirements and review.', ABS)]

E += [P('1. Problem', H),
      P('A business assistant answering from its owner\'s documents has one failure that costs money: it says something the documents do not say. The tools built to catch it in retrieval-augmented generation are trained and evaluated on English — HHEM-2.1-Open, a cross-encoder trained on factual-consistency data; LettuceDetect, a ModernBERT token classifier trained on RAGTruth; Patronus Lynx, a Llama-3 fine-tune under a non-commercial licence. Their model cards list one language. A developer serving Azerbaijani, Kazakh or Vietnamese users must either assume they transfer or use nothing.'),
      P('This report measures whether they transfer, on the same cases as the alternative that needs no training data: asking a capable general model, in the answer\'s language, which claims the sources do not support.')]

E += [P('2. Method', H),
      P('<b>Cases.</b> For each language a native-form facts block and a grounded answer were written by hand for two domains, as templates over five slots: street number, price, opening hours, phone number and (shop only) the return window. Two extra sentences per language were written for the appended-claim variants. The benchmark then plants exactly one error per variant by substituting a slot value (30 → 25 AZN; 9:00–19:00 → 9:00–17:00; No. 12 → 14; phone …01 02 → …01 03; 14 → 30 days) or appending a sentence (an invented Sunday opening; an added claim such as free parking). Because the substitution is mechanical, every language receives the same seven errors and the same grounded control; the only thing that varies is the language. Eleven languages × two domains × seven answers = 154 cases, of which 132 contain a planted error and 22 are grounded. The author is a native speaker of Azerbaijani and Russian and reads the others to varying degrees; the cases are short and formulaic by design, so that reading fluency, not idiom, is what is tested. Native-speaker review of the nine other languages is invited and has not yet happened.'),
      P('<b>Scoring.</b> A planted error is <i>caught</i> when the detector reports the answer as not grounded <i>and</i> names the planted claim: for LLM judges and LettuceDetect, the changed slot value must appear verbatim in a reported claim or span, and an appended sentence must share at least half its words with one; HHEM returns only a score, so for HHEM a flagged answer counts as caught. A grounded answer reported as not grounded is a <i>false alarm</i>. An empty or unparsable reply is <i>unjudged</i> and counts as a miss.'),
      P('<b>The judge.</b> One chat completion at temperature 0 over any OpenAI-compatible endpoint. The system prompt carries the sources as numbered facts and asks for every factual claim in the answer that the facts do not support, and a rewrite without them, as JSON. Judges were run on Groq (openai/gpt-oss-120b and -20b, qwen/qwen3.8-27b, allam-2-7b) and Google (gemini-3.8-flash, gemini-3.5-flash, gemini-2.5-flash) on 19–20 September 2026; the baselines on a laptop CPU with default thresholds (HHEM: consistency below 0.5 is a flag; LettuceDetect: any predicted span is a flag).')]

E += [P('3. Results', H)]
data = [['Detector', 'Kind', 'Errors caught', 'False alarms', 'Unjudged / errors', 'Median latency']]
for m in order:
    s = S_[m]; prov, _ = models[m]
    kind = 'trained classifier' if prov == 'local CPU' else f'LLM judge ({prov})'
    data.append([name(m), kind, f"{s['caught']}/{s['planted']} ({100 * s['caught'] / max(1, s['planted']):.0f} %)", f"{s['fa']}/{s['oks']}", f"{s['unj']}/{s['err']}", f"{s['lat']:.2f} s" if s['lat'] else '—'])
E += [KeepTogether([grid(data, [46, 32, 28, 22, 24, 22]), Spacer(1, 4), P('Table 1. All 154 cases. Planted errors caught out of 132; grounded answers wrongly flagged out of 22. Latency for the judges is per network call from Baku; for the classifiers, per inference on a laptop CPU.', SM)])]

# by error type
data = [['Detector'] + variants]
for m in order:
    c = models[m][1]['calls']; row = [name(m)]
    for v in variants:
        xs = [x for k, x in c.items() if k.endswith('/' + v) and 'error' not in x]
        row.append(f"{sum(1 for x in xs if x.get('caught'))}/{len(xs)}")
    data.append(row)
E += [KeepTogether([Spacer(1, 6), grid(data, [46, 17, 17, 18, 17, 15, 18, 17]), Spacer(1, 4), P('Table 2. Recall by error type, all languages. Wrong price, hours, street number, phone digit and return window are slot substitutions; "sunday" and "extra" are appended sentences. Hours applies to the clinic only and days to the shop only, hence 11 each.', SM)])]

# by language: judges (best three) + baselines
show = best[:2] + [m for m in judges if m not in best][:2] + [m for m in order if models[m][0] == 'local CPU']
data = [['Detector'] + langs]
for m in show:
    c = models[m][1]['calls']; row = [name(m)]
    for l in langs:
        xs = [x for k, x in c.items() if k.startswith(l + '/') and not k.endswith('/ok') and 'error' not in x]
        fa = sum(1 for k, x in c.items() if k.startswith(l + '/') and x.get('false_alarm'))
        row.append(f"{sum(1 for x in xs if x.get('caught'))}/{len(xs)}" + (f" ·{fa}" if fa else ''))
    data.append(row)
E += [KeepTogether([Spacer(1, 6), grid(data, [40] + [11.4] * 11, fs=7.8), Spacer(1, 4), P('Table 3. Recall by language (caught / 12 planted) and, after the dot, false alarms out of 2 grounded answers. en English, az Azerbaijani, ru Russian, tr Turkish, uk Ukrainian, kk Kazakh, ar Arabic, fa Persian, hi Hindi, id Indonesian, vi Vietnamese.', SM)])]

hh = S_['vectara/HHEM-2.1-Open']; ld = S_['KRLabsOrg/lettucedect-base-modernbert-en-v1']
hc = models['vectara/HHEM-2.1-Open'][1]['calls']
def hh_lang(l):
    xs = [x for k, x in hc.items() if k.startswith(l + '/') and not k.endswith('/ok')]; fa = sum(1 for k, x in hc.items() if k.startswith(l + '/') and x.get('false_alarm'))
    return sum(1 for x in xs if x.get('caught')), fa
lc = models['KRLabsOrg/lettucedect-base-modernbert-en-v1'][1]['calls']
def ld_lang(l):
    xs = [x for k, x in lc.items() if k.startswith(l + '/') and not k.endswith('/ok')]; return sum(1 for x in xs if x.get('caught'))
E += [P(f'<b>The judges.</b> {" and ".join(name(m) for m in best)} catch all 132 planted errors with no false alarms in all eleven languages; qwen/qwen3.8-27b misses one and gemini-3.5-flash three, still with no false alarms. The scripts — Latin with heavy diacritics, Cyrillic, Arabic, Perso-Arabic, Devanagari — make no visible difference to these models. The 7B allam-2-7b is the exception among judges: it over-flags grounded answers and misses errors in every language including English, which is an instruction-following limit rather than a language limit. Among the judges, harder error types separate the weaker models: the single changed phone digit and the wrong hours account for most of the misses (Table 2).'),
      P(f'<b>HHEM-2.1-Open.</b> Its headline recall, {hh["caught"]}/{hh["planted"]}, is not what it looks like. In English it catches {hh_lang("en")[0]} of 12 with no false alarms — a genuine but weak judgement on these short, number-heavy cases. In Russian, Arabic, Persian and Indonesian it catches 12 of 12 <i>and</i> flags both grounded answers: its consistency score for any non-English premise–hypothesis pair sits around 0.4, below the default threshold, whatever the content. The model is not detecting an unsupported claim; it is detecting that the text is not English. A detector that flags everything in a language has perfect recall and no use in it. Its false-alarm count, {hh["fa"]}/{hh["oks"]}, is the honest number.'),
      P(f'<b>LettuceDetect.</b> Recall follows script distance from its training data: {ld_lang("en")} of 12 in English, Azerbaijani and Turkish, {ld_lang("ru")} in Russian, {ld_lang("kk")} in Kazakh and {ld_lang("hi")} in Hindi, with false alarms in eight of the eleven languages. It does name spans, so where it works it is as actionable as a judge, and at 0.06 s on a CPU it is an order of magnitude cheaper. For a Latin-script language with a tolerance for misses it remains a reasonable first filter; for Cyrillic, Arabic-script or Devanagari text it is not.')]

E += [P('4. A measurement pitfall', H),
      P('An earlier run of this benchmark reported gemini-2.5-pro at 0 of 22 errors caught with 0 false alarms — a perfect-looking failure. Thinking models spend the output budget on reasoning before the first visible character; with a 1,200-token budget the reply came back empty, and the first version of the package scored an empty reply as "no unsupported claims", i.e. grounded. The corrected package returns <i>judged = false</i> for an empty or non-JSON reply, the benchmark counts it as a miss, and the budget is 8,000 tokens with one retry at the server\'s cap when a small model rejects it. A second, quieter version of the same mistake appeared in this report\'s baseline run: HHEM\'s score was first recorded as a "span" and then required to overlap the planted text, which no number can, giving 0 of 132; the raw scores were kept, so rescoring did not require rerunning. Both are reported because an evaluation that only publishes the tables that looked right is not one.')]

E += [P('5. Limitations', H),
      P('The benchmark is formulaic by construction: short answers, numeric slots, one error each. It measures whether a detector reads the language and follows the instruction; the near-perfect judge scores mean "not a language problem" and must not be read as "solved". Harder failures — a right number in the wrong sentence, a claim that is implied but not stated, an omitted condition, a paraphrase that changes meaning — are the next set and will lower every row of Table 1. The nine non-native languages were written by one author and await native review. Default thresholds were used for the classifiers; tuning HHEM\'s threshold per language would raise its English recall and would not repair its behaviour on non-English text, whose scores collapse regardless of content. Only two providers of judges were run, for cost; the runner accepts any OpenAI-compatible endpoint.')]

E += [P('6. Availability', H),
      P('Package (<i>pip install groundedness</i>, Python 3.9+, no dependencies), the case templates, both runners, the raw per-call outputs for every model and detector, and this report\'s generator are at <b>github.com/aghasalim/groundedness</b> under the MIT licence. The judge described here runs on every channel of SIBA\'s assistant (siba.az), where it reports the share of answers that passed; that deployment is the origin of the method.'),
      P('<b>Reproduce.</b>'), P('pip install groundedness pytest<br/>GROQ_API_KEY=... python benchmark/run_v2.py<br/>GEMINI_API_KEY=... python benchmark/run_v2.py --gemini<br/>pip install torch transformers lettucedetect &amp;&amp; python benchmark/run_baselines.py', CODE)]

E += [P('References', H),
      P('Vectara. HHEM-2.1-Open and the Hallucination Leaderboard. huggingface.co/vectara/hallucination_evaluation_model; github.com/vectara/hallucination-leaderboard.', SM),
      P('Kovács, Á. et al. LettuceDetect: A Hallucination Detection Framework for RAG Applications. arXiv:2502.17125, 2025.', SM),
      P('Ravi, S. et al. Lynx: An Open Source Hallucination Evaluation Model. arXiv:2407.08488, 2024.', SM),
      P('Niu, C. et al. RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. ACL 2024.', SM),
      P('Manakul, P., Liusie, A., Gales, M. SelfCheckGPT. EMNLP 2023.', SM),
      P('Farquhar, S. et al. Detecting hallucinations in large language models using semantic entropy. Nature 630, 2024.', SM),
      P('Aghayev, F., Ahmadbayli, E. SIMURG: Zero-Leak Online Detection of LLM Decoding Corruption in Production Streams. HAL-X AI, 2026.', SM)]

def deco(c, d):
    c.saveState(); c.setFont('TNR', 8); c.setFillColor(GRAY)
    c.drawString(22 * mm, 12 * mm, 'Mustafazada — Do English Hallucination Detectors Work in Other Languages?'); c.drawRightString(188 * mm, 12 * mm, str(c.getPageNumber())); c.restoreState()

out = os.path.join(here, 'groundedness-eleven-languages-v2.pdf')
doc = BaseDocTemplate(out, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm, title='Do English Hallucination Detectors Work in Other Languages?', author='Aghasalim Mustafazada')
doc.addPageTemplates([PageTemplate(id='n', frames=[Frame(22 * mm, 20 * mm, 166 * mm, A4[1] - 40 * mm, id='f', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=deco)])
doc.build(E)
print('OK ->', out, os.path.getsize(out), 'bytes')
