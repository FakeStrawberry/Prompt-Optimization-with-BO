#!/usr/bin/env bash
set -euo pipefail

cd /nas1/zyj/Prompt/reelicit_demo
unset all_proxy
unset ALL_PROXY

export HF_HOME=/nas1/zyj/hf-cache
export REELICIT_EVAL_PROGRESS=1
export CUDA_VISIBLE_DEVICES=1,2

ROOT="runs/paper_lite_qwen14b8b"
TASKS="${TASKS:-snarks}"
METHODS="${METHODS:-reelicit ape opro promptbreeder textgrad no_refinement no_bo static_features independent_extraction}"
SEEDS="${SEEDS:-0}"

# Defaults are paper budget. Override LIMIT/q/T/K/M from shell for faster tests.
LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

for task in $TASKS; do
  for seed in $SEEDS; do
    for method in $METHODS; do
      echo "=== task=$task seed=$seed method=$method started $(date) ==="
      /home/zyj/anaconda3/envs/reelicit/bin/python experiment.py \
        --method "$method" \
        --task "$task" \
        --seed "$seed" \
        --mode local-benchmark \
        --local-model /nas1/zyj/models/Qwen3-14B \
        --local-target-model /nas1/zyj/models/Qwen3-8B \
        --optimizer-device cuda:1 \
        --target-device cuda:0 \
        --include-format-in-context \
        --output-root "$ROOT" \
        "${LIMIT_ARG[@]}" \
        ${Q:+--q "$Q"} \
        ${TT:+--T "$TT"} \
        ${K:+--K "$K"} \
        ${M:+--M "$M"}
      echo "=== task=$task seed=$seed method=$method finished $(date) ==="
    done
  done
done

/home/zyj/anaconda3/envs/reelicit/bin/python scripts/summarize_results.py --root "$ROOT" --out reports/paper_lite_qwen14b8b
/home/zyj/anaconda3/envs/reelicit/bin/python scripts/make_figures.py --root "$ROOT" --out reports/paper_lite_qwen14b8b/figures
