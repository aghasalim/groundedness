#!/usr/bin/env bash
# Waits for the Groq key to show paid-tier limits, then: augment2 (both
# directions, all documents), rebuild data, run 5 from run 2, eval on v3,
# push if better than run 2 (352 caught / 32 false alarms).
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN="" PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
K=$(grep '^GROQ_API_KEY=' ~/SIBA/.env.local | cut -d= -f2- | tr -d '"'"'"' \r')
log(){ echo "[$(date +%H:%M)] $*"; }
log "augment2 on the free tier"
$PY train/augment2.py --workers 3 --hours 72 --gap 1 > train/data/augment2.log 2>&1
log "augment2: $(wc -l < train/data/synth-para2.jsonl) documents"
$PY train/prepare.py --ragtruth-cap 4000 --dev 0.05
caffeinate -dims $PY train/train.py --base train/out/run2 --epochs 2 --bs 8 --lr 2e-5 --pos-weight 4 --out train/out/run5 > train/out-run5.log 2>&1
log "run5: $(grep 'done best' train/out-run5.log)"
$PY - <<'PY'
import json; p='train/out/run5/training.json'; d=json.load(open(p)); d.pop('thresholds_by_script', None); d['threshold']=0.5; json.dump(d, open(p,'w'), indent=1)
PY
$PY train/eval_bench.py train/out/run5 siba/grounded-multilingual-base-probe > train/eval-run5.log 2>&1; tail -1 train/eval-run5.log
C=$(tail -1 train/eval-run5.log | grep -o 'caught [0-9]*' | cut -d' ' -f2); F=$(tail -1 train/eval-run5.log | grep -o 'alarms [0-9]*' | cut -d' ' -f2)
if [ "${C:-0}" -ge 345 ] && [ "${F:-99}" -lt 32 ]; then
  $PY - <<'PY'
import json; p='benchmark/results_v3-siba.json'; d=json.load(open(p)); d['models']['siba/grounded-multilingual-base']=d['models'].pop('siba/grounded-multilingual-base-probe'); json.dump(d,open(p,'w'),ensure_ascii=False,indent=1)
PY
  /Users/salim/SIBA/ops/pi/grounded/push-model.sh train/out/run5 && log "run5 live on the Pi"
else
  $PY - <<'PY'
import json; p='benchmark/results_v3-siba.json'; d=json.load(open(p)); d['models'].pop('siba/grounded-multilingual-base-probe',None); json.dump(d,open(p,'w'),ensure_ascii=False,indent=1)
PY
  log "run5 not better (caught $C, fa $F); run2 stays"
fi
log "paid run done"
