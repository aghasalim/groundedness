# -*- coding: utf-8 -*-
"""The trained detector on the 154 benchmark-v2 cases, same rules as the
baselines (a planted error is caught only when a span overlaps it), written to
benchmark/results_v2-siba.json so the site's build picks it up.
    python train/eval_bench.py train/out/grounded-multilingual-base [name]
"""
import json, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, '..', 'benchmark')
sys.path.insert(0, B); sys.path.insert(0, HERE)
from run_v2 import build, _words  # noqa: E402
from detect import Detector  # noqa: E402

def overlap(span, bad, var):
    if var in ('sunday', 'extra'):
        b, s = _words(bad), _words(span)
        return bool(b) and len(b & s) / len(b) >= 0.5
    return bad.lower() in span.lower() or span.lower() in bad.lower()

path = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else 'siba/grounded-multilingual-base'
CASES = 'cases_v3.json' if os.path.exists(os.path.join(B, 'cases_v3.json')) else 'cases_v2.json'; VER = 'v3' if CASES.endswith('v3.json') else 'v2'
det = Detector(path, quantize='--int8' in sys.argv)
cfg = json.load(open(os.path.join(B, CASES), encoding='utf-8')); cases = build(cfg)
row = {'calls': {}, 'latency': []}
for lang, dom, var, facts, answer, bad in cases:
    t = time.time(); r = det(answer, facts); row['latency'].append(round(time.time() - t, 3))
    key = f'{lang}/{dom}/{var}'
    if var == 'ok': row['calls'][key] = {'false_alarm': not r['grounded'], 'unsupported': r['unsupported']}
    else: row['calls'][key] = {'caught': any(overlap(x, bad, var) for x in r['unsupported']), 'unsupported': r['unsupported']}
    print(f"{key:22s} {json.dumps({k: v for k, v in row['calls'][key].items() if k != 'unsupported'})} {r['unsupported']}", flush=True)
out_p = os.path.join(B, f'results_{VER}-siba.json')
out = json.load(open(out_p, encoding='utf-8')) if os.path.exists(out_p) else {'cases': len(cases), 'models': {}}
out['models'][name] = row
json.dump(out, open(out_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
c = row['calls'].values()
caught = sum(1 for x in c if x.get('caught')); fa = sum(1 for x in c if x.get('false_alarm'))
planted = sum(1 for k in row['calls'] if not k.endswith('/ok')); oks = len(row['calls']) - planted
print(f'{name}: caught {caught}/{planted} false alarms {fa}/{oks} median {statistics.median(row["latency"]):.2f}s')
