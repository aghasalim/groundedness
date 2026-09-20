#!/usr/bin/env bash
# After the paraphrase augmenter: rebuild data, train run 4 from run 2, eval on v3, push if better.
set -u; cd "$(dirname "$0")/.."; PY=.venv/bin/python
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN="" PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
log(){ echo "[$(date +%H:%M)] $*"; }
while pgrep -f "train/augment.py" >/dev/null; do sleep 60; done
log "augment: $(wc -l < train/data/synth-para.jsonl) paraphrased documents"
$PY train/prepare.py --ragtruth-cap 4000 --dev 0.05
caffeinate -dims $PY train/train.py --base train/out/run2 --epochs 2 --bs 8 --lr 2e-5 --pos-weight 4 --out train/out/run4 > train/out-run4.log 2>&1
log "run4: $(grep 'done best' train/out-run4.log)"
$PY train/eval_bench.py train/out/run4 siba/grounded-multilingual-base-probe > train/eval-run4.log 2>&1; tail -1 train/eval-run4.log
C4=$(tail -1 train/eval-run4.log | grep -o 'caught [0-9]*' | cut -d' ' -f2); F4=$(tail -1 train/eval-run4.log | grep -o 'alarms [0-9]*' | cut -d' ' -f2)
if [ "${C4:-0}" -ge 340 ] && [ "${F4:-99}" -lt 32 ]; then
  $PY - <<'PY'
import json; p='benchmark/results_v3-siba.json'; d=json.load(open(p)); d['models']['siba/grounded-multilingual-base']=d['models'].pop('siba/grounded-multilingual-base-probe'); json.dump(d,open(p,'w'),ensure_ascii=False,indent=1)
PY
  /Users/salim/SIBA/ops/pi/grounded/push-model.sh train/out/run4 && log "run4 live on the Pi"
else
  $PY - <<'PY'
import json; p='benchmark/results_v3-siba.json'; d=json.load(open(p)); d['models'].pop('siba/grounded-multilingual-base-probe',None); json.dump(d,open(p,'w'),ensure_ascii=False,indent=1)
PY
  log "run4 not better (caught $C4, fa $F4); run2 stays"
fi
log "night3 done"
