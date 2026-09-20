# -*- coding: utf-8 -*-
"""cases_v3.json = cases_v2.json + twenty more languages, templates translated by
Gemini with the {slot} markers kept verbatim, validated (same slots, no extras,
non-empty appended sentences) and written next to the eleven hand-written ones.
    GEMINI_API_KEY=... python benchmark/make_v3.py
"""
import json, os, re, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
v2 = json.load(open(os.path.join(HERE, 'cases_v2.json'), encoding='utf-8'))
LN = {'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'pl': 'Polish', 'nl': 'Dutch', 'zh': 'Chinese (Simplified)', 'ja': 'Japanese', 'ko': 'Korean', 'uz': 'Uzbek (Latin)', 'ka': 'Georgian', 'he': 'Hebrew', 'bn': 'Bengali', 'th': 'Thai', 'sw': 'Swahili', 'ro': 'Romanian', 'cs': 'Czech', 'el': 'Greek', 'hu': 'Hungarian'}
KEY = os.environ.get('GROQ_API_KEY') or [l.split('=', 1)[1].strip() for l in open(os.path.expanduser('~/SIBA/.env.local')) if l.startswith('GROQ_API_KEY=')][0]
SLOT = re.compile(r'\{[a-z_]+\}')

def ask(lang):
    # one domain per call: Groq's per-minute token budget is small
    out = {}
    for dom in ('clinic', 'shop'):
        prompt = ('Translate every string value in this JSON into ' + LN[lang] + '. Keep the keys. Keep every {placeholder} exactly as it is (same spelling, same braces) and keep each placeholder in the string it came from. "AZN" stays "AZN" (or the local word for manat). Keep the leading space of "sunday" and "extra". Natural, the way a real business would write it. Reply with JSON only.\n\n' + json.dumps(v2['languages']['en'][dom], ensure_ascii=False))
        body = {'model': 'openai/gpt-oss-120b', 'reasoning_effort': 'low', 'temperature': 0.2, 'max_tokens': 2500, 'messages': [{'role': 'user', 'content': prompt}]}
        req = urllib.request.Request('https://api.groq.com/openai/v1/chat/completions', data=json.dumps(body).encode(), headers={'authorization': 'Bearer ' + KEY, 'content-type': 'application/json', 'user-agent': 'groundedness/0.1.3'})
        for attempt in range(6):
            try:
                raw = json.load(urllib.request.urlopen(req, timeout=120))['choices'][0]['message']['content']; break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5: raise
                ra = e.headers.get('retry-after'); time.sleep(float(ra) + 1 if ra and ra.replace('.', '').isdigit() else 20)
        out[dom] = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
        time.sleep(3)
    return out

def valid(t):
    en = v2['languages']['en']
    for dom in ('clinic', 'shop'):
        for k in ('facts', 'answer', 'sunday', 'extra'):
            s = t.get(dom, {}).get(k)
            if not isinstance(s, str) or not s.strip(): return f'{dom}.{k} missing'
            if set(SLOT.findall(s)) != set(SLOT.findall(en[dom][k])): return f'{dom}.{k} slots {SLOT.findall(s)}'
        for k in ('sunday', 'extra'):
            if not t[dom][k].startswith(' '): t[dom][k] = ' ' + t[dom][k]
    return None

P3 = os.path.join(HERE, 'cases_v3.json')
out = json.load(open(P3, encoding='utf-8')) if os.path.exists(P3) else json.loads(json.dumps(v2))
out['_note'] = 'v3. ' + v2['_note'] + ' Languages beyond the first eleven were translated from the English templates by openai/gpt-oss-120b and validated for slot integrity; native review invited.'
for lang in LN:
    if lang in out['languages']: continue
    for attempt in range(4):
        try:
            t = ask(lang); err = valid(t)
        except Exception as e:  # noqa: BLE001
            err = str(e)[:80]; t = None
        if not err:
            out['languages'][lang] = {'clinic': t['clinic'], 'shop': t['shop']}; print(lang, 'ok', t['clinic']['answer'][:70], flush=True)
            json.dump(out, open(P3, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); break
        print(lang, 'retry', err, flush=True); time.sleep(5)
    else:
        sys.exit('failed ' + lang)
json.dump(out, open(P3, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(out['languages']), 'languages')
