#!/usr/bin/env bash
set -euo pipefail

cd /nas1/zyj/Prompt/reelicit_demo
unset all_proxy
unset ALL_PROXY

export HF_HOME=/nas1/zyj/hf-cache
export HF_DATASETS_CACHE=/nas1/zyj/hf-cache/datasets
export REELICIT_EVAL_PROGRESS="${REELICIT_EVAL_PROGRESS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"

ROOT="${ROOT:-runs/paper_lite_qwen14b8b_once}"
TASKS="${TASKS:-snarks}"
METHODS="${METHODS:-reelicit ape opro promptbreeder textgrad no_refinement no_bo static_features independent_extraction}"
SEEDS="${SEEDS:-0}"

ARGS=(
  --mode local-benchmark
  --tasks "$TASKS"
  --methods "$METHODS"
  --seeds "$SEEDS"
  --local-model /nas1/zyj/models/Qwen3-14B
  --local-target-model /nas1/zyj/models/Qwen3-8B
  --optimizer-device cuda:1
  --target-device cuda:0
  --include-format-in-context
  --output-root "$ROOT"
)

if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ -n "${Q:-}" ]]; then ARGS+=(--q "$Q"); fi
if [[ -n "${TT:-}" ]]; then ARGS+=(--T "$TT"); fi
if [[ -n "${K:-}" ]]; then ARGS+=(--K "$K"); fi
if [[ -n "${M:-}" ]]; then ARGS+=(--M "$M"); fi
if [[ "${SKIP_COMPLETED:-1}" != "0" ]]; then ARGS+=(--skip-completed); fi

/home/zyj/anaconda3/envs/reelicit/bin/python suite.py "${ARGS[@]}"
/home/zyj/anaconda3/envs/reelicit/bin/python scripts/make_figures.py --root "$ROOT" --out "reports/$(basename "$ROOT")/figures"
/home/zyj/anaconda3/envs/reelicit/bin/python scripts/summarize_results.py --root "$ROOT" --out "reports/$(basename "$ROOT")"
