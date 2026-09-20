#!/usr/bin/env bash
# The unattended pipeline for the night of 2026-09-20: wait for run 1, eval and
# push it; at 06:50 stop the generator, rebuild the data, warm-start run 2 on
# everything, eval, push. Log: train/night.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
log(){ echo "[$(date +%H:%M)] $*"; }
# 1. run 1
while pgrep -f "train/train.py --epochs 3" >/dev/null; do sleep 60; done
log "run1 finished: $(grep 'done best' train/out-run1.log)"
$PY train/eval_bench.py train/out/run1 siba/grounded-multilingual-base > train/eval-run1.log 2>&1; tail -1 train/eval-run1.log
/Users/salim/SIBA/ops/pi/grounded/push-model.sh train/out/run1 && log "run1 live on the Pi"
# 2. wait for the data cut-off, then run 2
while [ "$(date +%H%M)" -lt 0650 ]; do sleep 60; done
pkill -f "train/gen_data.py"; sleep 2
log "data: $(wc -l < train/data/synth.jsonl) documents"
$PY train/prepare.py --ragtruth-cap 8000 --dev 0.03
$PY train/train.py --base train/out/run1 --epochs 2 --bs 16 --lr 2e-5 --out train/out/run2 > train/out-run2.log 2>&1
log "run2 finished: $(grep 'done best' train/out-run2.log)"
$PY train/eval_bench.py train/out/run2 siba/grounded-multilingual-base > train/eval-run2.log 2>&1; tail -1 train/eval-run2.log
# keep whichever benchmark better: run2 unless it is clearly worse
R1=$(tail -1 train/eval-run1.log | grep -o 'caught [0-9]*' | cut -d' ' -f2); R2=$(tail -1 train/eval-run2.log | grep -o 'caught [0-9]*' | cut -d' ' -f2)
if [ "${R2:-0}" -ge "${R1:-0}" ]; then /Users/salim/SIBA/ops/pi/grounded/push-model.sh train/out/run2 && log "run2 live on the Pi"; else log "run2 worse ($R2 < $R1), run1 stays"; fi
log "night done"
