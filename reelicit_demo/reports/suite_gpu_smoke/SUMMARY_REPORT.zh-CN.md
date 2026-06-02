# ReElicit 复现实验汇总

## 运行配置

- 任务：`snarks`
- 方法：`reelicit ape no_bo`
- Seeds：`0`
- 样本上限：`5`
- 预算：q=2, T=2, K=1, M=1
- Optimizer：`/nas1/zyj/models/Qwen3-14B`；Target：`/nas1/zyj/models/Qwen3-8B`

## 方法汇总

| Task | Method | Runs | Mean Best | Max Best |
|---|---|---:|---:|---:|
| snarks | ape | 1 | 0.600000 | 0.600000 |
| snarks | no_bo | 1 | 0.600000 | 0.600000 |
| snarks | reelicit | 1 | 0.400000 | 0.400000 |

## 每个任务的最佳方法

| Task | Best Method | Best Score | Seed | Evaluations |
|---|---|---:|---:|---:|
| snarks | ape | 0.600000 | 0 | 4 |

## 最佳 Prompt 摘要

- `snarks` / `ape` / seed `0` / 0.600000：You are a tone and sarcasm detection expert. When presented with two nearly identical statements, your task is to analyze the nuances in tone, pragmatics, and figurative language to determine which one is sarcastic. Focus on subtle cues such as exaggerated praise, irony, or contradiction in context. Provide your answer by selecting only the letter (A or B) corresponding to the sarcastic statement.

## 图像文件

- `figures/aggregate_method_comparison.png`
- `figures/snarks_convergence.png`

原始逐 run 数据见 `summary.csv`，各方法的完整轨迹见对应 `history.jsonl`。
