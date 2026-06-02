#!/usr/bin/env bash
set -euo pipefail

cd /nas1/zyj/Prompt/reelicit_demo

export ROOT="${ROOT:-runs/paper_all_tasks_qwen14b8b_seed0_limit50}"
export TASKS="${TASKS:-gsm8k mmlu boolean_expressions causal_judgement disambiguation_qa formal_fallacies hyperbaton penguins_in_a_table snarks tracking_shuffled_objects}"
export METHODS="${METHODS:-reelicit ape opro promptbreeder textgrad no_refinement no_bo static_features independent_extraction}"
export SEEDS="${SEEDS:-0}"
export LIMIT="${LIMIT:-50}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export REELICIT_EVAL_PROGRESS="${REELICIT_EVAL_PROGRESS:-1}"
export SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
export REELICIT_GSM8K_TEST_ARROW="${REELICIT_GSM8K_TEST_ARROW:-/nas1/wrj/decoding-prob/hf_cache/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow}"

mkdir -p logs
LOG="${LOG:-logs/$(basename "$ROOT").log}"
exec > >(tee -a "$LOG") 2>&1

date
echo "ROOT=$ROOT"
echo "TASKS=$TASKS"
echo "METHODS=$METHODS"
echo "SEEDS=$SEEDS"
echo "LIMIT=$LIMIT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "REELICIT_DATA_ROOT=${REELICIT_DATA_ROOT:-}"
echo "REELICIT_GSM8K_TEST_ARROW=$REELICIT_GSM8K_TEST_ARROW"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

./scripts/run_qwen14b8b_suite_once.sh

date
echo "Report: reports/$(basename "$ROOT")/SUMMARY_REPORT.zh-CN.md"
echo "Figures: reports/$(basename "$ROOT")/figures"
