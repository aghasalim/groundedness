#!/usr/bin/env bash
# Second half of the night: wait for run 2, then for the 31-language case file,
# eval on it, push to the Pi (if not worse than run 1 on the first 11 languages).
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
log(){ echo "[$(date +%H:%M)] $*"; }
while pgrep -f "train/train.py --base train/out/run1 " >/dev/null; do sleep 60; done
log "run2 finished: $(grep 'done best' train/out-run2.log)"
until grep -q "^31 languages" benchmark/make_v3.log; do sleep 60; done
$PY train/eval_bench.py train/out/run2 siba/grounded-multilingual-base > train/eval-run2.log 2>&1; tail -1 train/eval-run2.log
/Users/salim/SIBA/ops/pi/grounded/push-model.sh train/out/run2 && log "run2 live on the Pi"
log "night done"
