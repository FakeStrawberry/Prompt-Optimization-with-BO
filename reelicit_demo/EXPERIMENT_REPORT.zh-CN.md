# ReElicit 单任务实验报告：Snarks

## 1. 实验目的

本实验用于验证论文 `Embedding by Elicitation: Dynamic Representations for Bayesian Optimization of System Prompts` 中 ReElicit 的核心流程能否在本地真实模型和真实 benchmark 上跑通，并观察在 Snarks 任务上的优化表现。

本次实验不是论文的完整 10 任务、30 seeds 复现；它是一个单任务全量 split 的真实模型实验。

## 2. 实验配置

### 模型

| 角色 | 模型 | 本地路径 | 设备 |
|---|---|---|---|
| Optimizer LLM | Qwen3-14B | `/nas1/zyj/models/Qwen3-14B` | physical GPU 2 |
| Target LLM | Qwen3-8B | `/nas1/zyj/models/Qwen3-8B` | physical GPU 1 |

进程内设置为：

```bash
CUDA_VISIBLE_DEVICES=1,2
optimizer device: cuda:1
target device: cuda:0
```

### 任务与数据

| 项目 | 设置 |
|---|---|
| 任务 | Snarks |
| 数据集 | `lukaemon/bbh`, `snarks` |
| split | cached full test split |
| 样本数 | 178 |
| seed | 0 |
| 评估方式 | target model zero-shot greedy decoding，按答案抽取规则统计 accuracy |

### ReElicit 参数

| 参数 | 值 |
|---|---:|
| `q` | 5 |
| `T` | 6 |
| `N=q*T` | 30 |
| `K` | 5 |
| `M` | 10 |
| `tau` | 0.1 |
| `b` | 10 |
| `nmax` | 12 |
| `strict_paper` | false |

本次启用了 `--include-format-in-context`，即在 optimizer 看到的任务描述中加入输出格式要求：

```text
The assistant should answer with just the letter choice, e.g. (A).
```

这样做是为了让 Qwen3-14B 生成的 system prompt 与 evaluator 的答案抽取规则一致，避免输出解释、数字或裸字母导致格式性失分。

## 3. 运行信息

启动脚本：

```bash
/nas1/zyj/Prompt/reelicit_demo/scripts/run_qwen14b8b_snarks_full.sh
```

输出目录：

```text
/nas1/zyj/Prompt/reelicit_demo/runs/qwen14b_opt_qwen8b_target_snarks_full_seed0_fullbudget/
```

运行时间：

| 项目 | 时间 |
|---|---|
| 开始 | 2026-05-29 00:04:55 AWST |
| 结束 | 2026-05-29 00:39:21 AWST |
| 总耗时 | 约 34 分钟 |

运行期间 Hugging Face 数据集访问出现一次网络不可达提示，但程序自动使用本地缓存的 Snarks 数据集继续执行，实验未中断。

## 4. 结果

### 最终最佳结果

| 指标 | 值 |
|---|---:|
| best score | 0.651685 |
| 正确数 | 116 / 178 |
| 首次达到最佳 | D0 candidate 2 |
| 后续再次达到最佳 | ReElicit iteration 2 candidate 5 |

最佳 prompt：

```text
Your task is to compare two nearly identical statements and determine which one contains sarcasm. Consider the use of irony, exaggerated language, and the speaker's intent. Your response should be a single letter: (A) or (B), indicating the sarcastic statement.
```

### 每轮候选分数

| 轮次 | 候选分数 | 本轮均值 | 本轮最高 | best-so-far |
|---|---|---:|---:|---:|
| D0 | 0.5955, 0.6124, 0.6517, 0.5843, 0.6067 | 0.6101 | 0.6517 | 0.6517 |
| Iter 1 | 0.6011, 0.6124, 0.6180, 0.6067, 0.6180 | 0.6112 | 0.6180 | 0.6517 |
| Iter 2 | 0.5618, 0.6236, 0.6180, 0.6011, 0.6517 | 0.6112 | 0.6517 | 0.6517 |
| Iter 3 | 0.6236, 0.5955, 0.5618, 0.6236, 0.5899 | 0.5989 | 0.6236 | 0.6517 |
| Iter 4 | 0.6236, 0.6180, 0.5899, 0.6124, 0.5674 | 0.6022 | 0.6236 | 0.6517 |
| Iter 5 | 0.5955, 0.6236, 0.6067, 0.6124, 0.5955 | 0.6067 | 0.6236 | 0.6517 |

### 最终选中特征

最终 `state.json` 中保存的特征空间包含两个特征：

| 特征 | 含义 |
|---|---|
| `sarcasm_cue_specificity` | prompt 是否明确列出讽刺识别线索，例如 irony、tone、figurative language。 |
| `contextual_guidance` | prompt 是否明确指导模型从语境、隐含含义、speaker intent 等角度区分讽刺与字面含义。 |

这两个特征与 Snarks 的任务性质一致：任务核心是判断两个近似 statement 中哪一个具有讽刺语用含义。

## 5. 分析

本次实验完整跑通了 ReElicit 的真实模型链路：初始 prompt 生成、动态特征定义、特征抽取、BO 选点、目标 prompt realization/refinement，以及 target model 的全量 Snarks 评估。

从分数看，本次 run 的最佳 prompt 在初始 D0 阶段已经出现，后续 ReElicit 在第 2 轮再次生成了同一最佳 prompt，但没有超过它。也就是说，本次单 seed 实验中 ReElicit 没有带来 best-so-far 提升。

这并不一定说明算法无效，主要有几个原因：

1. 这是单任务单 seed 结果，论文报告的是多任务、多 seed 的统计结果。
2. 本实验使用 Qwen3-14B/Qwen3-8B 替代论文中的 Llama 3.3 70B / Llama 3.1 8B，结果不可直接对照论文表格。
3. `strict_paper=false`，BO acquisition 使用当前 demo 的 fallback GP/EI，而不是严格的 BoTorch `qLogNoisyExpectedImprovement`。
4. Snarks 任务上初始 prompt 已经达到本次 run 的最高点，后续搜索空间的提升余量较小。
5. 多个 ReElicit 候选分数集中在 0.59-0.62 附近，说明 optimizer 生成的 prompt 语义差异不大，探索多样性有限。

## 6. 结论

本次实验表明：

- 本地 Qwen3-14B optimizer + Qwen3-8B target 的双模型配置可运行。
- 单任务全量 Snarks split、论文默认 `N=30` 预算可以在约 34 分钟内完成。
- 最佳准确率为 0.651685，即 116/178。
- 本次单 seed 中 ReElicit 未超过 D0 中的最佳 prompt，但成功复现了该最佳 prompt。
- 输出格式提示对 Qwen optimizer 很重要，否则容易生成与 evaluator 不匹配的 prompt。

## 7. 产物路径

```text
runs/qwen14b_opt_qwen8b_target_snarks_full_seed0_fullbudget/history.jsonl
runs/qwen14b_opt_qwen8b_target_snarks_full_seed0_fullbudget/state.json
runs/qwen14b_opt_qwen8b_target_snarks_full_seed0_fullbudget/stdout.log
```

