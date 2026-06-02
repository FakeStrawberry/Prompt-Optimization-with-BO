# ReElicit 复现 Demo

这个目录是对 `../2605.19093v1.pdf` 论文 **Embedding by Elicitation: Dynamic Representations for Bayesian Optimization of System Prompts** 的复现型 demo。

默认实验设置已经写在 `configs/paper.json` 中，和论文保持一致：

- 总评估预算：`N=30`
- 每轮候选数：`q=5`
- 总 batch 数：`T=6`，其中第 0 轮是初始数据集 `D0`
- 每轮特征 elicitation 次数：`K=5`
- 每个 BO target 的生成/细化预算：`M=10`
- refinement 提前停止阈值：`tau=0.1`
- 特征抽取 batch size：`b=10`
- optimizer 侧历史样本上限：`nmax=12`
- optimizer LLM：`Llama 3.3 70B Instruct`
- target LLM：`Llama 3.1 8B Instruct`
- optimizer temperature：`0.7`
- target decoding：zero-shot greedy，temperature `0.0`
- BO surrogate：BoTorch `SingleTaskGP`
- acquisition：`qLogNoisyExpectedImprovement`
- acquisition 优化：20 restarts，512 raw samples

## 已实现内容

- Algorithm 1 / Algorithm 3 的 ReElicit 主循环。
- Algorithm 2 的初始 prompt 数据集生成。
- 动态特征定义 `DEFINEFEATURES`。
- 无 score 泄露的特征抽取 `EXTRACTFEATURES`。
- incumbent feature set 的重新抽取和交叉验证打分。
- 基于 GP CV MSE 的 feature set 选择。
- BO 在 `[0, 1]^d` 特征空间中选择目标 feature vector。
- LLM 将 BO target realization 成 system prompt。
- 基于 feature gap 的 sequential refinement。
- 附录 D 中的主要 ReElicit prompt。
- Table 5 的任务上下文和 Table 6 的答案抽取规则。
- aggregate-only 评估接口：优化器只能看到每个 prompt 的一个 scalar score。
- 本地 mock 模式：不需要 Llama 权重即可检查完整算法链路。

## 目录结构

```text
reelicit_demo/
  demo.py                    # CLI 入口
  configs/paper.json         # 论文默认配置
  requirements.txt           # 严格复现所需核心依赖
  requirements-paper.txt     # 额外评估相关依赖
  reelicit/
    algorithm.py             # ReElicit 主循环
    bo.py                    # GP CV 和 BO target selection
    evaluator.py             # mock / benchmark evaluator
    llm.py                   # mock / OpenAI-compatible LLM client
    prompts.py               # 论文附录 prompt
    tasks.py                 # 任务上下文、模板、答案抽取
    types.py                 # 数据结构
    utils.py                 # JSON 解析、采样、保存等工具
```

## 快速 Smoke Test

当前机器没有完整的 Llama / BoTorch 环境时，可以先跑 mock 模式：

```bash
cd /nas1/zyj/Prompt/reelicit_demo
python3 demo.py --mock --task snarks --seed 0
```

mock 模式仍然使用论文里的 `N=30, q=5, T=6, K=5, M=10` 等预算设置，但 optimizer LLM 和 target LLM 都替换成本地 deterministic stand-in。它只能验证代码流程，不代表论文实验分数。

运行后会生成：

```text
runs/<task>_seed<seed>_mock/
  history.jsonl   # 每个被评估 prompt 的 scalar score
  state.json      # 当前最佳 prompt 和最后选择的 feature set
```

## 严格论文设置运行

建议使用 Conda 管理环境：

```bash
cd /nas1/zyj/Prompt/reelicit_demo
conda create -n reelicit -c conda-forge python=3.12 -y
conda activate reelicit
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

这里用 Conda 管理 Python 环境，用 `python -m pip` 安装 requirements，可以确保依赖装进当前激活的 `reelicit` 环境里。

然后把两个 Llama 模型分别部署成 OpenAI-compatible endpoint，例如用 vLLM：

```bash
export OPTIMIZER_BASE_URL=http://localhost:8000/v1
export OPTIMIZER_API_KEY=EMPTY
export OPTIMIZER_MODEL=meta-llama/Llama-3.3-70B-Instruct

export TARGET_BASE_URL=http://localhost:8001/v1
export TARGET_API_KEY=EMPTY
export TARGET_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

运行单个任务：

```bash
python demo.py --mode benchmark --strict-paper --task snarks --seed 0
```

`--strict-paper` 会要求使用 BoTorch 的 `SingleTaskGP` 和 `qLogNoisyExpectedImprovement`。如果不加该参数，代码会在缺少 BoTorch 时退回到一个轻量 NumPy GP/EI selector，只适合调试流程。

## 本机 Qwen 14B/8B 复现实验

当前机器已经按 Conda 环境 `reelicit` 配好依赖，并下载了两个本地模型：

```text
/nas1/zyj/models/Qwen3-14B   # optimizer LLM
/nas1/zyj/models/Qwen3-8B    # target LLM
```

真实模型的单任务运行示例：

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

多方法/多任务复现实验入口：

```bash
conda activate reelicit
cd /nas1/zyj/Prompt/reelicit_demo
ROOT=runs/paper_available_tasks_qwen14b8b_seed0_limit50 \
TASKS="gsm8k snarks" \
HF_HUB_OFFLINE=1 \
./scripts/run_qwen14b8b_all_tasks_limit50.sh
```

这个脚本会使用论文预算 `q=5, T=6, K=5, M=10`，方法包括 `reelicit`、`ape`、`opro`、`promptbreeder`、`textgrad` 以及四个消融：`no_refinement`、`no_bo`、`static_features`、`independent_extraction`。默认 `LIMIT=50` 是为了先在本机可控时间内产生完整报告和图；要扩到任务全集，可以设置更大的 `LIMIT` 或取消该变量。

后台运行示例：

```bash
tmux new-session -d -s reelicit_available_exp_limit50 \
  'ROOT=runs/paper_available_tasks_qwen14b8b_seed0_limit50 TASKS="gsm8k snarks" HF_HUB_OFFLINE=1 ./scripts/run_qwen14b8b_all_tasks_limit50.sh'
```

查看进度：

```bash
tmux capture-pane -t reelicit_available_exp_limit50 -p | tail -n 80
tail -f logs/paper_available_tasks_qwen14b8b_seed0_limit50.log
```

运行完成后自动生成：

```text
reports/paper_available_tasks_qwen14b8b_seed0_limit50/SUMMARY_REPORT.zh-CN.md
reports/paper_available_tasks_qwen14b8b_seed0_limit50/summary.csv
reports/paper_available_tasks_qwen14b8b_seed0_limit50/figures/
```

当前本机可直接读取的数据集是 `snarks` 和一个本地 Arrow 版 GSM8K test split。MMLU 以及其它 BBH 任务还没有本地缓存；在没有 Hugging Face 网络的情况下会失败，需要先补齐数据缓存再加入 `TASKS`。

## 任务

目前任务 key 包括：

- `gsm8k`
- `mmlu`
- `boolean_expressions`
- `causal_judgement`
- `disambiguation_qa`
- `formal_fallacies`
- `hyperbaton`
- `penguins_in_a_table`
- `snarks`
- `tracking_shuffled_objects`

论文中 GSM8K 和 MMLU 使用固定 500 题子集，其余 BBH 任务使用 250 题。PDF 没有公开 GSM8K/MMLU 固定子采样的具体索引，所以这里把 `seed` 显式暴露出来，保证 demo 运行本身可复现。

## 注意事项

- mock 模式不是论文成绩复现，只是完整链路验证。
- benchmark 模式下，优化器仍然只能看到每个 prompt 的最终 aggregate scalar score，不会看到逐题错误、模型回答 trace 或文本 critique。
- 完整论文设置的计算成本较高：每个 prompt evaluation 都会让 target model 跑完整 benchmark 子集。
- 输出文件默认保存在 `runs/` 下，可以用 `--output` 指定目录。
- 这台机器上 `all_proxy=socks5://127.0.0.1:1803` 可能导致 `hf download` 报 `Network is unreachable`。下载 Hugging Face 模型时建议用 `scripts/download_qwen_models.sh`，脚本会取消 `all_proxy` 并把模型保存到 `/nas1/zyj/models/`。
