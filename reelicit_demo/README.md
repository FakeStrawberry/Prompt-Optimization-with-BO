# ReElicit Demo

This folder implements a reproduction-oriented demo for the paper in `../2605.19093v1.pdf`: **Embedding by Elicitation: Dynamic Representations for Bayesian Optimization of System Prompts**.

I mirrored the paper settings in `configs/paper.json`: `N=30`, `q=5`, `T=6`, `K=5`, `M=10`, `tau=0.1`, extraction batch size `b=10`, history cap `nmax=12`, optimizer temperature `0.7`, greedy target decoding, optimizer model `Llama 3.3 70B Instruct`, target model `Llama 3.1 8B Instruct`, and BoTorch `SingleTaskGP` + `qLogNoisyExpectedImprovement` with 20 restarts and 512 raw samples.

## What is implemented

- ReElicit main loop from Algorithms 1 and 3.
- Initial dataset generation from Algorithm 2.
- Dynamic feature elicitation, feature extraction, incumbent feature re-scoring, GP CV feature-set selection, BO target selection, initial target realization, and feature-gap refinement.
- Appendix D prompts for `DEFINEFEATURES`, `EXTRACTFEATURES`, initial generation, refinement, and seed-prompt generation.
- Task contexts and answer extraction templates from Tables 5 and 6.
- Aggregate-only evaluator interface: the optimizer receives only one scalar score per prompt.
- A mock mode for local smoke tests without Llama weights.

The mock mode is not a scientific reproduction of the reported scores. It exists so the code path can be checked on a normal machine. For paper-faithful runs, use `--mode benchmark --strict-paper` with Llama endpoints and the dependencies in `requirements.txt`.

## Quick Smoke Run

```bash
cd /nas1/zyj/Prompt/reelicit_demo
python3 demo.py --mock --task snarks --seed 0
```

This uses the same paper budget and hyperparameters, but replaces the optimizer and target LLMs with deterministic local stand-ins.

## Paper-Style Run

Create and activate a Conda environment:

```bash
cd /nas1/zyj/Prompt/reelicit_demo
conda create -n reelicit -c conda-forge python=3.12 -y
conda activate reelicit
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using Conda for the environment and `python -m pip` for the package install keeps the dependencies tied to the active `reelicit` environment.

Serve the two Llama chat models behind OpenAI-compatible endpoints, for example with vLLM. Then set:

```bash
export OPTIMIZER_BASE_URL=http://localhost:8000/v1
export OPTIMIZER_API_KEY=EMPTY
export OPTIMIZER_MODEL=meta-llama/Llama-3.3-70B-Instruct

export TARGET_BASE_URL=http://localhost:8001/v1
export TARGET_API_KEY=EMPTY
export TARGET_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

Run one task:

```bash
python demo.py --mode benchmark --strict-paper --task snarks --seed 0
```

By default the evaluator uses the paper task sizes: 500 examples for GSM8K/MMLU and 250 for each BBH task. Use `--limit` only for debugging, because it changes the paper evaluation setting.

## Local Qwen 14B/8B Runs

This machine has the Conda environment `reelicit` and two local Qwen models:

```text
/nas1/zyj/models/Qwen3-14B   # optimizer LLM
/nas1/zyj/models/Qwen3-8B    # target LLM
```

Single-task local run:

```bash
conda activate reelicit
cd /nas1/zyj/Prompt/reelicit_demo
CUDA_VISIBLE_DEVICES=1,2 python demo.py \
  --mode local-benchmark \
  --task snarks \
  --seed 0 \
  --local-model /nas1/zyj/models/Qwen3-14B \
  --local-target-model /nas1/zyj/models/Qwen3-8B \
  --optimizer-device cuda:1 \
  --target-device cuda:0 \
  --include-format-in-context
```

Multi-method suite:

```bash
conda activate reelicit
cd /nas1/zyj/Prompt/reelicit_demo
ROOT=runs/paper_available_tasks_qwen14b8b_seed0_limit50 \
TASKS="gsm8k snarks" \
HF_HUB_OFFLINE=1 \
./scripts/run_qwen14b8b_all_tasks_limit50.sh
```

The suite uses the paper budget `q=5, T=6, K=5, M=10` and runs ReElicit, APE, OPRO, PromptBreeder, TextGrad, and the four ablations: `no_refinement`, `no_bo`, `static_features`, and `independent_extraction`. The default `LIMIT=50` keeps the first local reproduction tractable; increase or unset it for larger evaluations.

Background run:

```bash
tmux new-session -d -s reelicit_available_exp_limit50 \
  'ROOT=runs/paper_available_tasks_qwen14b8b_seed0_limit50 TASKS="gsm8k snarks" HF_HUB_OFFLINE=1 ./scripts/run_qwen14b8b_all_tasks_limit50.sh'
```

Progress and outputs:

```bash
tmux capture-pane -t reelicit_available_exp_limit50 -p | tail -n 80
tail -f logs/paper_available_tasks_qwen14b8b_seed0_limit50.log
```

When the suite finishes it writes `SUMMARY_REPORT.zh-CN.md`, `summary.csv`, and convergence/comparison figures under `reports/<run-name>/`.

Current local data availability is limited to Snarks plus a local Arrow copy of GSM8K test. MMLU and the other BBH tasks are not cached on this machine, and Hugging Face network access is currently unavailable, so those tasks need data cache setup before they can be added to `TASKS`.

## Notes

- The PDF does not publish the exact fixed subsample indices for GSM8K/MMLU. This demo exposes the dataset seed explicitly and records it in the output path.
- Output artifacts are written under `runs/<task>_seed<seed>_<mode>/`: `history.jsonl` stores every evaluated prompt and scalar score, and `state.json` stores the latest best prompt and selected feature definitions.
- `--strict-paper` requires `torch`, `gpytorch`, and `botorch`; without it, the code falls back to a small NumPy GP/EI selector for smoke tests.
- On this machine, `all_proxy=socks5://127.0.0.1:1803` can make `hf download` fail with `Network is unreachable`. Use `scripts/download_qwen_models.sh` for the Qwen downloads; it unsets `all_proxy` and stores models under `/nas1/zyj/models/`.
