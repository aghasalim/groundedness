# -*- coding: utf-8 -*-
"""Paraphrase augmentation against the false alarms: for each synthetic
document, ask an LLM to rewrite the SOURCE in two different surface forms —
abbreviations expanded or introduced, sentence order changed, numbers and
times restated (9:00–19:00 / 9 am to 7 pm), lists turned into prose — with
every fact kept identical. The four answers and their spans are unchanged: the
grounded one stays supported under the paraphrased source, the errors stay
errors. Writes train/data/synth-para.jsonl; resumable; free tiers only.
    python train/augment.py --hours 2
"""
import argparse, json, os, random, re, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import endpoints, call, LN  # noqa: E402
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUT = os.path.join(D, 'synth-para.jsonl')
PROMPT = '''Rewrite the following {lang} text twice, as two alternative versions a different business would write. Keep EVERY fact exactly the same — every number, price, time, day, name, address, phone digit, rule and condition — but change the surface form as much as possible: expand or introduce abbreviations (e.g. "Mon–Sat" ↔ "Monday to Saturday"), restate times and ranges differently, reorder sentences, turn lists into prose or prose into lists, use synonyms, change the register (one formal, one casual). Do not add or drop any fact. Write only in {lang}. Reply with JSON only: {{"versions": ["...", "..."]}}

TEXT:
{src}'''

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--hours', type=float, default=2); a = ap.parse_args()
    docs = [json.loads(l) for l in open(os.path.join(D, 'synth.jsonl'), encoding='utf-8')]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'): done.add(json.loads(l)['orig'])
    todo = [(i, d) for i, d in enumerate(docs) if d['source'][:80] not in done]
    random.Random(1).shuffle(todo)
    eps = [e for e in endpoints() if not e['model'].startswith('gemini')]
    lock = threading.Lock(); fh = open(OUT, 'a', encoding='utf-8'); t0 = time.time(); n = {'ok': 0, 'bad': 0}
    def worker(ep):
        while todo and time.time() - t0 < a.hours * 3600:
            with lock:
                if not todo: return
                i, d = todo.pop()
            raw, code, ra = call(ep, PROMPT.format(lang=LN[d['lang']], src=d['source']))
            if code == 429: time.sleep(min(ra, 120) if ra else 45); todo.append((i, d)); continue
            if code != 200: time.sleep(20); todo.append((i, d)); continue
            m = re.search(r'\{.*\}', raw or '', re.S)
            try: vs = json.loads(m.group(0))['versions']
            except Exception: vs = None
            with lock:
                if isinstance(vs, list) and all(isinstance(v, str) and len(v) > 60 for v in vs[:2]) and len(vs) >= 2:
                    fh.write(json.dumps({'orig': d['source'][:80], 'lang': d['lang'], 'question': d.get('question', ''), 'sources': vs[:2], 'answers': d['answers'], 'model': ep['model']}, ensure_ascii=False) + '\n'); fh.flush(); n['ok'] += 1
                else: n['bad'] += 1
                if (n['ok'] + n['bad']) % 25 == 0: print(f"[{int(time.time()-t0)}s] ok {n['ok']} bad {n['bad']} left {len(todo)}", flush=True)
            time.sleep(ep['gap'])
    ts = [threading.Thread(target=worker, args=(e,), daemon=True) for e in eps]
    for t in ts: t.start()
    for t in ts: t.join()
    print('finished', n, flush=True)

if __name__ == '__main__':
    main()
