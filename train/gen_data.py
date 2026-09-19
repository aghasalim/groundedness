# -*- coding: utf-8 -*-
"""Synthetic training data for the multilingual detector.

Every call asks one LLM for a source document, a question and four answers in
one language and one domain: answer A grounded, B-D each with 1-3 unsupported
claims wrapped in ⟦ ⟧. The markers give exact character spans, so no second
labelling pass is needed. Models rotate so each one's free-tier daily quota is
spread; a 429 parks that model for a while. Output is appended to
train/data/synth.jsonl and the run is resumable.

    python train/gen_data.py --calls 3000 --hours 6
"""
import argparse, json, os, random, re, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'synth.jsonl')
CORE = ['en', 'az', 'ru', 'tr', 'uk', 'kk', 'ar', 'fa', 'hi', 'id', 'vi']
EXTRA = ['de', 'fr', 'es', 'it', 'pt', 'pl', 'nl', 'zh', 'ja', 'ko', 'uz', 'ka', 'he', 'bn', 'th', 'sw', 'ro', 'cs', 'el', 'hu']
LN = {'en': 'English', 'az': 'Azerbaijani', 'ru': 'Russian', 'tr': 'Turkish', 'uk': 'Ukrainian', 'kk': 'Kazakh', 'ar': 'Arabic', 'fa': 'Persian', 'hi': 'Hindi', 'id': 'Indonesian', 'vi': 'Vietnamese', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'nl': 'Dutch', 'zh': 'Chinese (Simplified)', 'ja': 'Japanese', 'ko': 'Korean', 'uz': 'Uzbek', 'ka': 'Georgian', 'he': 'Hebrew', 'bn': 'Bengali', 'th': 'Thai', 'sw': 'Swahili', 'ro': 'Romanian', 'cs': 'Czech', 'el': 'Greek', 'hu': 'Hungarian'}
DOMAINS = ['dental clinic', 'family doctor practice', 'pharmacy', 'veterinary clinic', 'hotel', 'hostel', 'restaurant', 'cafe', 'bakery', 'gym', 'yoga studio', 'swimming pool', 'bank branch', 'insurance product', 'mobile operator tariff', 'internet provider', 'airline baggage rules', 'train timetable', 'intercity bus', 'taxi service', 'car rental', 'car repair shop', 'real-estate listing', 'apartment for rent', 'university admissions', 'private school', 'language course', 'kindergarten', 'law firm', 'notary office', 'accounting firm', 'government service desk', 'visa application centre', 'museum', 'theatre', 'cinema', 'concert', 'tour agency', 'electronics shop', 'clothing shop', 'grocery delivery', 'furniture store', 'bookshop', 'flower shop', 'beauty salon', 'barber', 'spa', 'laundry', 'appliance repair', 'plumber', 'SaaS pricing page', 'event venue', 'wedding venue', 'coworking space', 'public library', 'football club tickets', 'ski resort', 'parking garage', 'EV charging network', 'company HR policy', 'company FAQ', 'product manual', 'warranty terms', 'return policy', 'shipping policy', 'job posting', 'news article', 'encyclopedia biography', 'recipe', 'medication leaflet', 'weather forecast', 'sports match report', 'quarterly financial summary', 'product spec sheet', 'city council notice', 'school timetable', 'conference programme', 'charity appeal', 'software changelog', 'landlord house rules', 'gas station services', 'post office', 'courier tracking page', 'online course syllabus', 'driving school', 'photography studio', 'printing shop', 'tailor', 'optician', 'hearing aid centre', 'physiotherapy', 'nutritionist', 'psychologist practice', 'pet shop', 'garden centre', 'hardware store', 'bicycle shop', 'phone repair', 'music school', 'dance studio', 'kids party venue', 'summer camp']
STYLES = ['formal customer-support tone', 'friendly chatty tone', 'short bullet points', 'a single dense paragraph', 'very polite with a greeting and sign-off', 'terse, as in a messenger chat', 'confident salesy tone', 'apologetic tone']
SRC_FORMS = ['a website "About / Prices" section', 'an FAQ with 5-7 Q&A pairs', 'an internal notes file for support staff', 'a plain list of facts', 'an email from the manager', 'a table rendered as text lines', 'a leaflet', 'a knowledge-base article']
ERR = ['change one number (price, time, quantity, date, percentage) slightly', 'change one number a lot', 'invent a service or feature the source never mentions', 'state the opposite of a rule in the source', 'add a wrong day or wrong opening hours', 'invent a name, address detail or phone digit', 'overstate a promise (always, guaranteed, free, unlimited)', 'add a plausible detail that is simply not in the source', 'mix up two entities from the source', 'give a wrong reason or condition']

PROMPT = '''Language: {lang}. Domain: {domain}.
Write everything in {lang} only (except this JSON structure). Produce:
1. "source": {form}, 80-180 words, with at least 8 concrete facts: numbers, prices, times, days, names, addresses, phone numbers, conditions, rules. Realistic, specific, in {lang}.
2. "question": one customer question that several of those facts answer.
3. "answers": exactly four assistant replies to that question, each 2-5 sentences, {style}, in {lang}:
   - answer 1: fully supported by the source. Every factual claim in it must be in the source.
   - answers 2, 3, 4: mostly correct, but each contains 1-3 claims the source does NOT support. Plant them as: {e1}; {e2}; {e3}. Wrap each unsupported span in ⟦ and ⟧, the shortest contiguous phrase that carries the wrong or invented claim, exactly as written in the sentence. Everything outside the markers must be supported by the source or be a harmless pleasantry.
Never wrap greetings or offers to help. Never put markers in answer 1 or in the source. Reply with JSON only:
{{"source": "...", "question": "...", "answers": ["...", "...", "...", "..."]}}'''

def endpoints():
    env = {}
    for p in (os.path.expanduser('~/SIBA/.env.local'),):
        if os.path.exists(p):
            for line in open(p, encoding='utf-8'):
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1); env[k.strip()] = v.strip().strip('"\'')
    groq = os.environ.get('GROQ_API_KEY') or env.get('GROQ_API_KEY')
    gem = os.environ.get('GEMINI_API_KEY') or env.get('GEMINI_API')
    eps = []
    if groq:
        for m in ['qwen/qwen3.8-27b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b']:
            eps.append(dict(model=m, url='https://api.groq.com/openai/v1/chat/completions', key=groq, gap=16))
    if gem:
        for m in ['gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-2.5-flash']:
            eps.append(dict(model=m, url='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions', key=gem, gap=8))
    return eps

def call(ep, prompt):
    body = {'model': ep['model'], 'temperature': 1.0, 'max_tokens': 4000, 'messages': [{'role': 'user', 'content': prompt}]}
    if 'qwen3' in ep['model']: body['reasoning_format'] = 'hidden'
    if 'gpt-oss' in ep['model']: body['reasoning_effort'] = 'low'
    if ep['model'].startswith('gemini'): body['max_tokens'] = 8000
    req = urllib.request.Request(ep['url'], data=json.dumps(body).encode(), headers={'authorization': 'Bearer ' + ep['key'], 'content-type': 'application/json', 'user-agent': 'groundedness-train/0.1'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        ra = e.headers.get('retry-after')
        return None, e.code, float(ra) if ra and ra.replace('.', '').isdigit() else None
    except Exception:
        return None, 0, None
    return (data.get('choices') or [{}])[0].get('message', {}).get('content') or '', 200, None

MARK = re.compile(r'⟦(.+?)⟧', re.S)
def parse(raw):
    raw = raw.strip()
    m = re.search(r'\{.*\}', raw, re.S)
    if not m: return None
    try: j = json.loads(m.group(0))
    except Exception:
        try: j = json.loads(m.group(0).replace('\n', ' '))
        except Exception: return None
    src, q, ans = j.get('source'), j.get('question'), j.get('answers')
    if not (isinstance(src, str) and isinstance(ans, list) and len(ans) == 4 and all(isinstance(a, str) for a in ans)): return None
    if '⟦' in src or len(src) < 80: return None
    out = []
    for i, a in enumerate(ans):
        spans, text, pos = [], '', 0
        for mm in MARK.finditer(a):
            text += a[pos:mm.start()]
            s = len(text); text += mm.group(1); spans.append([s, len(text)]); pos = mm.end()
        text += a[pos:]
        if '⟦' in text or '⟧' in text or len(text) < 20: return None
        if i == 0 and spans: return None
        if i > 0 and not spans: return None
        out.append({'text': text, 'spans': spans})
    return {'source': src.strip(), 'question': (q or '').strip() if isinstance(q, str) else '', 'answers': out}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--calls', type=int, default=3000); ap.add_argument('--hours', type=float, default=8); ap.add_argument('--core-share', type=float, default=0.75)
    a = ap.parse_args()
    eps = endpoints()
    if not eps: sys.exit('no keys')
    lock = threading.Lock(); done = {'ok': 0, 'bad': 0, 'calls': 0}; t0 = time.time()
    if os.path.exists(OUT): done['ok'] = sum(1 for _ in open(OUT, encoding='utf-8'))
    fh = open(OUT, 'a', encoding='utf-8')
    def worker(ep):
        rnd = random.Random(hash(ep['model']) ^ int(t0))
        while done['calls'] < a.calls and time.time() - t0 < a.hours * 3600:
            lang = rnd.choice(CORE) if rnd.random() < a.core_share else rnd.choice(EXTRA)
            errs = rnd.sample(ERR, 3)
            p = PROMPT.format(lang=LN[lang], domain=rnd.choice(DOMAINS), form=rnd.choice(SRC_FORMS), style=rnd.choice(STYLES), e1=errs[0], e2=errs[1], e3=errs[2])
            with lock: done['calls'] += 1
            raw, code, ra = call(ep, p)
            if code == 429:
                time.sleep(min(ra, 120) if ra else 45); continue
            if code != 200:
                time.sleep(30); continue
            rec = parse(raw)
            with lock:
                if rec:
                    rec.update(lang=lang, model=ep['model']); fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush(); done['ok'] += 1
                else: done['bad'] += 1
                if done['calls'] % 25 == 0:
                    print(f"[{int(time.time()-t0)}s] calls {done['calls']} records {done['ok']} rejected {done['bad']}", flush=True)
            time.sleep(ep['gap'])
    ts = [threading.Thread(target=worker, args=(ep,), daemon=True) for ep in eps]
    for t in ts: t.start()
    for t in ts: t.join()
    print('finished', done, flush=True)

if __name__ == '__main__':
    main()
