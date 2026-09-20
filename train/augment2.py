# -*- coding: utf-8 -*-
"""Paraphrase augmentation, both directions, for the paid tier.
Per synthetic document, one call returns: two paraphrased SOURCES (facts kept)
and, for each of the four answers, one paraphrased ANSWER in which the ⟦ ⟧
marked spans are kept marked (wording inside may change, the claim must stay
the same and still unsupported). Grounded answers stay grounded. Output rows
have the synth.jsonl shape plus 'para': true; prepare.py folds them in.
    python train/augment2.py --workers 6 --hours 1.5
"""
import argparse, json, os, random, re, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import endpoints, call, LN, parse, MARK  # noqa: E402
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUT = os.path.join(D, 'synth-para2.jsonl')
PROMPT = '''You get a {lang} business document and four assistant answers. Some answers contain claims wrapped in ⟦ ⟧ that the document does NOT support.
Produce:
1. "sources": two rewrites of the document with EVERY fact identical (numbers, prices, times, days, names, addresses, phone digits, rules) but a different surface form: abbreviations expanded or introduced ("Mon–Sat" ↔ "Monday to Saturday"), times restated, sentences reordered, lists ↔ prose, synonyms, one formal and one casual.
2. "answers": one rewrite of EACH of the four answers, same order, in different wording (synonyms, reordered clauses, expanded abbreviations, a different greeting). Every ⟦ ⟧ marked span must remain in the rewrite, still wrapped in ⟦ ⟧, still making the same unsupported claim (its wording may change). Do not add new unsupported claims; anything outside the markers must stay supported by the document. An answer without markers must get no markers.
Write only in {lang}. Reply with JSON only: {{"sources": ["...", "..."], "answers": ["...", "...", "...", "..."]}}

DOCUMENT:
{src}

ANSWERS:
{answers}'''

def marked(a):
    t, off = a['text'], 0
    for s, e in a['spans']:
        t = t[:s + off] + '⟦' + t[s + off:e + off] + '⟧' + t[e + off:]; off += 2
    return t

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--hours', type=float, default=1.5); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--gap', type=float, default=1.0)
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(os.path.join(D, 'synth.jsonl'), encoding='utf-8')]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding='utf-8'): done.add(json.loads(l)['orig'])
    todo = [d for d in docs if d['source'][:80] not in done]; random.Random(2).shuffle(todo)
    eps = [e for e in endpoints() if 'gpt-oss-120b' in e['model'] or 'qwen' in e['model']]
    for e in eps: e['gap'] = a.gap
    lock = threading.Lock(); fh = open(OUT, 'a', encoding='utf-8'); t0 = time.time(); n = {'ok': 0, 'bad': 0}
    def worker(k):
        ep = eps[k % len(eps)]
        while todo and time.time() - t0 < a.hours * 3600:
            with lock:
                if not todo: return
                d = todo.pop()
            answers = '\n'.join(f'{i + 1}. {marked(x)}' for i, x in enumerate(d['answers']))
            raw, code, ra = call(ep, PROMPT.format(lang=LN[d['lang']], src=d['source'], answers=answers))
            if code == 429: time.sleep(min(ra, 60) if ra else 15); todo.append(d); continue
            if code != 200: time.sleep(10); todo.append(d); continue
            m = re.search(r'\{.*\}', raw or '', re.S)
            try: j = json.loads(m.group(0))
            except Exception: j = None
            rec = None
            if j and isinstance(j.get('sources'), list) and isinstance(j.get('answers'), list) and len(j['answers']) == 4:
                pa = parse(json.dumps({'source': d['source'], 'question': d.get('question', ''), 'answers': j['answers']}, ensure_ascii=False))
                srcs = [s for s in j['sources'][:2] if isinstance(s, str) and len(s) > 60]
                if pa and srcs:
                    # the paraphrased answers must keep one marked span per original span, roughly
                    ok = all(bool(x['spans']) == bool(y['spans']) for x, y in zip(pa['answers'], d['answers']))
                    if ok: rec = {'orig': d['source'][:80], 'lang': d['lang'], 'question': d.get('question', ''), 'source': d['source'], 'sources': srcs, 'answers': d['answers'], 'answers_para': pa['answers'], 'model': ep['model']}
            with lock:
                if rec: fh.write(json.dumps(rec, ensure_ascii=False) + '\n'); fh.flush(); n['ok'] += 1
                else: n['bad'] += 1
                if (n['ok'] + n['bad']) % 25 == 0: print(f"[{int(time.time()-t0)}s] ok {n['ok']} bad {n['bad']} left {len(todo)}", flush=True)
            time.sleep(ep['gap'])
    ts = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(a.workers)]
    for t in ts: t.start()
    for t in ts: t.join()
    print('finished', n, flush=True)

if __name__ == '__main__':
    main()
