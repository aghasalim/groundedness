#!/usr/bin/env bash
# Benchmark v3 = the v2 cases in 31 languages. Seeds each results file from the
# v2 results (the runners are resumable, so the 154 finished cases are kept and
# only the 280 new ones are called), then runs the Groq judges, the two
# English baselines and our detector. Gemini is not rerun (credits gone).
set -u; cd "$(dirname "$0")/.."
PY=.venv/bin/python; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN=""
export GROQ_API_KEY=$(grep '^GROQ_API_KEY=' ~/SIBA/.env.local | cut -d= -f2- | tr -d '"'"'"' \r')
[ -f benchmark/results_v3.json ] || cp benchmark/results_v2.json benchmark/results_v3.json
$PY benchmark/run_v2.py --cases cases_v3.json --out results_v3.json --models qwen/qwen3.8-27b,openai/gpt-oss-120b,openai/gpt-oss-20b --pace 4 > benchmark/run_v3.log 2>&1
[ -f benchmark/results_v3-baselines.json ] || cp benchmark/results_v2-baselines.json benchmark/results_v3-baselines.json
$PY benchmark/run_baselines.py --cases cases_v3.json --out results_v3-baselines.json > benchmark/run_v3-baselines.log 2>&1
echo "v3 judges+baselines done $(date +%H:%M)"
