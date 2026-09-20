#!/usr/bin/env bash
# Wait for the three parallel judge runs, merge them into results_v3.json, run the baselines on v3.
set -u; cd "$(dirname "$0")/.."; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
while pgrep -f "run_v2.py --cases cases_v3.json" >/dev/null; do sleep 30; done
.venv/bin/python - <<'PY'
import json
d=json.load(open('benchmark/results_v3.json'))
for t in ('qwen3.8-27b','gpt-oss-120b','gpt-oss-20b'):
    for m,r in json.load(open(f'benchmark/results_v3-{t}.json'))['models'].items(): d['models'][m]=r
d['cases']=434; json.dump(d, open('benchmark/results_v3.json','w'), ensure_ascii=False, indent=1)
print({m: len(r['calls']) for m,r in d['models'].items()})
PY
.venv/bin/python benchmark/run_v2.py --cases cases_v3.json --out results_v3.json --models none > /dev/null 2>&1  # rewrites the .md summary only
[ -f benchmark/results_v3-baselines.json ] || cp benchmark/results_v2-baselines.json benchmark/results_v3-baselines.json
.venv/bin/python benchmark/run_baselines.py --cases cases_v3.json --out results_v3-baselines.json > benchmark/run_v3-baselines.log 2>&1
echo "v3 judges+baselines done $(date +%H:%M)"
