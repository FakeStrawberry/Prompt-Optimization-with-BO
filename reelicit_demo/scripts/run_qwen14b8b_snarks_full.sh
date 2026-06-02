#!/usr/bin/env bash
set -euo pipefail

cd /nas1/zyj/Prompt/reelicit_demo

# Avoid the local SOCKS proxy path that breaks httpx/Hugging Face calls.
unset all_proxy
unset ALL_PROXY

export HF_HOME=/nas1/zyj/hf-cache
export REELICIT_EVAL_PROGRESS=1
export CUDA_VISIBLE_DEVICES=1,2

OUT_DIR="runs/qwen14b_opt_qwen8b_target_snarks_full_seed0_fullbudget"
mkdir -p "$OUT_DIR"

echo "Started at $(date)"
echo "Output dir: $OUT_DIR"
echo "Optimizer: /nas1/zyj/models/Qwen3-14B on process cuda:1 (physical GPU 2)"
echo "Target:    /nas1/zyj/models/Qwen3-8B on process cuda:0 (physical GPU 1)"
echo "Task: snarks full split, seed=0, q=5, T=6, K=5, M=10, N=30"

/home/zyj/anaconda3/envs/reelicit/bin/python demo.py \
  --mode local-benchmark \
  --local-model /nas1/zyj/models/Qwen3-14B \
  --local-target-model /nas1/zyj/models/Qwen3-8B \
  --optimizer-device cuda:1 \
  --target-device cuda:0 \
  --task snarks \
  --seed 0 \
  --q 5 \
  --T 6 \
  --K 5 \
  --M 10 \
  --include-format-in-context \
  --output "$OUT_DIR"

echo "Finished at $(date)"
