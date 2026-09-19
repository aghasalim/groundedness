# -*- coding: utf-8 -*-
"""Merge the synthetic multilingual data with RAGTruth (English, real model
outputs with human span labels) into train/dev files of one shape:
    {"source": str, "answer": str, "spans": [[start, end], ...], "lang": str, "src": "synth"|"ragtruth"}
Dev is split by source document so no document is seen in both.
    python train/prepare.py [--ragtruth-cap 6000]
"""
import argparse, json, os, random
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def synth():
    docs = []
    p = os.path.join(D, 'synth.jsonl')
    if not os.path.exists(p): return docs
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        src = r['source'] + (('\n\nQuestion: ' + r['question']) if r.get('question') else '')
        docs.append([{'source': src, 'answer': a['text'], 'spans': a['spans'], 'lang': r['lang'], 'src': 'synth'} for a in r['answers']])
    return docs

def ragtruth(cap, rnd):
    info = {}
    for line in open(os.path.join(D, 'ragtruth_source_info.jsonl'), encoding='utf-8'):
        s = json.loads(line); info[s['source_id']] = s
    by_doc = {}
    for line in open(os.path.join(D, 'ragtruth_response.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if r.get('quality') != 'good': continue
        s = info.get(r['source_id'])
        if not s: continue
        # the prompt holds the passages / article / record the model was given
        src = s['prompt']
        spans = sorted({(l['start'], l['end']) for l in r['labels']})
        by_doc.setdefault(r['source_id'], []).append({'source': src, 'answer': r['response'], 'spans': [list(x) for x in spans], 'lang': 'en', 'src': 'ragtruth'})
    docs = list(by_doc.values()); rnd.shuffle(docs)
    # keep every hallucinated answer, and about as many clean ones
    out, n = [], 0
    for d in docs:
        bad = [x for x in d if x['spans']]; good = [x for x in d if not x['spans']]
        keep = bad + good[:max(1, len(bad))]
        if keep: out.append(keep); n += len(keep)
        if n >= cap: break
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ragtruth-cap', type=int, default=6000); ap.add_argument('--dev', type=float, default=0.04)
    a = ap.parse_args(); rnd = random.Random(7)
    docs = synth() + ragtruth(a.ragtruth_cap, rnd)
    rnd.shuffle(docs)
    ndev = int(len(docs) * a.dev)
    for name, part in (('dev', docs[:ndev]), ('train', docs[ndev:])):
        rows = [x for d in part for x in d]
        with open(os.path.join(D, name + '.jsonl'), 'w', encoding='utf-8') as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
        langs = {}
        for x in rows: langs[x['lang']] = langs.get(x['lang'], 0) + 1
        print(name, len(rows), 'answers,', sum(1 for x in rows if x['spans']), 'with spans;', 'synth', sum(1 for x in rows if x['src'] == 'synth'), 'ragtruth', sum(1 for x in rows if x['src'] == 'ragtruth'), '|', ' '.join(f'{k}:{v}' for k, v in sorted(langs.items(), key=lambda kv: -kv[1])[:14]))

if __name__ == '__main__':
    main()
