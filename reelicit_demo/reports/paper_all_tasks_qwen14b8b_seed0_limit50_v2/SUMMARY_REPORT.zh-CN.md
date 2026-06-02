# ReElicit 复现实验汇总

## 运行配置

- 任务：`gsm8k mmlu boolean_expressions causal_judgement disambiguation_qa formal_fallacies hyperbaton penguins_in_a_table snarks tracking_shuffled_objects`
- 方法：`reelicit ape opro promptbreeder textgrad no_refinement no_bo static_features independent_extraction`
- Seeds：`0`
- 样本上限：`50`
- 预算：q=None, T=None, K=None, M=None
- Optimizer：`/nas1/zyj/models/Qwen3-14B`；Target：`/nas1/zyj/models/Qwen3-8B`

## 方法汇总

| Task | Method | Runs | Mean Best | Max Best |
|---|---|---:|---:|---:|
| boolean_expressions | ape | 1 | 0.940000 | 0.940000 |
| boolean_expressions | independent_extraction | 1 | 0.920000 | 0.920000 |
| boolean_expressions | no_bo | 1 | 0.920000 | 0.920000 |
| boolean_expressions | no_refinement | 1 | 0.900000 | 0.900000 |
| boolean_expressions | opro | 1 | 0.920000 | 0.920000 |
| boolean_expressions | promptbreeder | 1 | 0.960000 | 0.960000 |
| boolean_expressions | reelicit | 1 | 0.940000 | 0.940000 |
| boolean_expressions | static_features | 1 | 0.920000 | 0.920000 |
| boolean_expressions | textgrad | 1 | 0.920000 | 0.920000 |
| causal_judgement | ape | 1 | 0.740000 | 0.740000 |
| causal_judgement | independent_extraction | 1 | 0.680000 | 0.680000 |
| causal_judgement | no_bo | 1 | 0.700000 | 0.700000 |
| causal_judgement | no_refinement | 1 | 0.680000 | 0.680000 |
| causal_judgement | opro | 1 | 0.660000 | 0.660000 |
| causal_judgement | promptbreeder | 1 | 0.720000 | 0.720000 |
| causal_judgement | reelicit | 1 | 0.700000 | 0.700000 |
| causal_judgement | static_features | 1 | 0.720000 | 0.720000 |
| causal_judgement | textgrad | 1 | 0.680000 | 0.680000 |
| disambiguation_qa | ape | 1 | 0.600000 | 0.600000 |
| disambiguation_qa | independent_extraction | 1 | 0.580000 | 0.580000 |
| disambiguation_qa | no_bo | 1 | 0.640000 | 0.640000 |
| disambiguation_qa | no_refinement | 1 | 0.640000 | 0.640000 |
| disambiguation_qa | opro | 1 | 0.640000 | 0.640000 |
| disambiguation_qa | promptbreeder | 1 | 0.620000 | 0.620000 |
| disambiguation_qa | reelicit | 1 | 0.560000 | 0.560000 |
| disambiguation_qa | static_features | 1 | 0.600000 | 0.600000 |
| disambiguation_qa | textgrad | 1 | 0.660000 | 0.660000 |
| formal_fallacies | ape | 1 | 0.640000 | 0.640000 |
| formal_fallacies | independent_extraction | 1 | 0.640000 | 0.640000 |
| formal_fallacies | no_bo | 1 | 0.640000 | 0.640000 |
| formal_fallacies | no_refinement | 1 | 0.660000 | 0.660000 |
| formal_fallacies | opro | 1 | 0.600000 | 0.600000 |
| formal_fallacies | promptbreeder | 1 | 0.640000 | 0.640000 |
| formal_fallacies | reelicit | 1 | 0.660000 | 0.660000 |
| formal_fallacies | static_features | 1 | 0.640000 | 0.640000 |
| formal_fallacies | textgrad | 1 | 0.620000 | 0.620000 |
| gsm8k | ape | 1 | 0.920000 | 0.920000 |
| gsm8k | independent_extraction | 1 | 0.920000 | 0.920000 |
| gsm8k | no_bo | 1 | 0.900000 | 0.900000 |
| gsm8k | no_refinement | 1 | 0.920000 | 0.920000 |
| gsm8k | opro | 1 | 0.920000 | 0.920000 |
| gsm8k | promptbreeder | 1 | 0.900000 | 0.900000 |
| gsm8k | reelicit | 1 | 0.920000 | 0.920000 |
| gsm8k | static_features | 1 | 0.920000 | 0.920000 |
| gsm8k | textgrad | 1 | 0.940000 | 0.940000 |
| hyperbaton | ape | 1 | 0.920000 | 0.920000 |
| hyperbaton | independent_extraction | 1 | 0.940000 | 0.940000 |
| hyperbaton | no_bo | 1 | 0.960000 | 0.960000 |
| hyperbaton | no_refinement | 1 | 0.680000 | 0.680000 |
| hyperbaton | opro | 1 | 0.720000 | 0.720000 |
| hyperbaton | promptbreeder | 1 | 0.860000 | 0.860000 |
| hyperbaton | reelicit | 1 | 0.940000 | 0.940000 |
| hyperbaton | static_features | 1 | 0.720000 | 0.720000 |
| hyperbaton | textgrad | 1 | 0.700000 | 0.700000 |
| mmlu | ape | 1 | 0.880000 | 0.880000 |
| mmlu | independent_extraction | 1 | 0.900000 | 0.900000 |
| mmlu | no_bo | 1 | 0.880000 | 0.880000 |
| mmlu | no_refinement | 1 | 0.880000 | 0.880000 |
| mmlu | opro | 1 | 0.840000 | 0.840000 |
| mmlu | promptbreeder | 1 | 0.880000 | 0.880000 |
| mmlu | reelicit | 1 | 0.860000 | 0.860000 |
| mmlu | static_features | 1 | 0.900000 | 0.900000 |
| mmlu | textgrad | 1 | 0.840000 | 0.840000 |
| penguins_in_a_table | ape | 1 | 0.780000 | 0.780000 |
| penguins_in_a_table | independent_extraction | 1 | 0.940000 | 0.940000 |
| penguins_in_a_table | no_bo | 1 | 0.740000 | 0.740000 |
| penguins_in_a_table | no_refinement | 1 | 0.920000 | 0.920000 |
| penguins_in_a_table | opro | 1 | 0.720000 | 0.720000 |
| penguins_in_a_table | promptbreeder | 1 | 0.960000 | 0.960000 |
| penguins_in_a_table | reelicit | 1 | 0.880000 | 0.880000 |
| penguins_in_a_table | static_features | 1 | 0.760000 | 0.760000 |
| penguins_in_a_table | textgrad | 1 | 0.920000 | 0.920000 |
| snarks | ape | 1 | 0.640000 | 0.640000 |
| snarks | independent_extraction | 1 | 0.660000 | 0.660000 |
| snarks | no_bo | 1 | 0.620000 | 0.620000 |
| snarks | no_refinement | 1 | 0.600000 | 0.600000 |
| snarks | opro | 1 | 0.620000 | 0.620000 |
| snarks | promptbreeder | 1 | 0.600000 | 0.600000 |
| snarks | reelicit | 1 | 0.580000 | 0.580000 |
| snarks | static_features | 1 | 0.620000 | 0.620000 |
| snarks | textgrad | 1 | 0.620000 | 0.620000 |
| tracking_shuffled_objects | ape | 1 | 0.860000 | 0.860000 |
| tracking_shuffled_objects | independent_extraction | 1 | 0.880000 | 0.880000 |
| tracking_shuffled_objects | no_bo | 1 | 0.880000 | 0.880000 |
| tracking_shuffled_objects | no_refinement | 1 | 0.880000 | 0.880000 |
| tracking_shuffled_objects | opro | 1 | 0.840000 | 0.840000 |
| tracking_shuffled_objects | promptbreeder | 1 | 0.840000 | 0.840000 |
| tracking_shuffled_objects | reelicit | 1 | 0.920000 | 0.920000 |
| tracking_shuffled_objects | static_features | 1 | 0.820000 | 0.820000 |
| tracking_shuffled_objects | textgrad | 1 | 0.820000 | 0.820000 |

## 每个任务的最佳方法

| Task | Best Method | Best Score | Seed | Evaluations |
|---|---|---:|---:|---:|
| boolean_expressions | promptbreeder | 0.960000 | 0 | 30 |
| causal_judgement | ape | 0.740000 | 0 | 30 |
| disambiguation_qa | textgrad | 0.660000 | 0 | 30 |
| formal_fallacies | reelicit | 0.660000 | 0 | 30 |
| gsm8k | textgrad | 0.940000 | 0 | 30 |
| hyperbaton | no_bo | 0.960000 | 0 | 30 |
| mmlu | static_features | 0.900000 | 0 | 30 |
| penguins_in_a_table | promptbreeder | 0.960000 | 0 | 30 |
| snarks | independent_extraction | 0.660000 | 0 | 30 |
| tracking_shuffled_objects | reelicit | 0.920000 | 0 | 30 |

## 最佳 Prompt 摘要

- `boolean_expressions` / `promptbreeder` / seed `0` / 0.960000：Carefully parse and evaluate the boolean expression following standard logical rules. Do not make assumptions about operator precedence or operand values. Respond with only 'True' or 'False' based on the correct evaluation.
- `causal_judgement` / `ape` / seed `0` / 0.740000：You are tasked with evaluating causality. For each scenario, determine if the given factor is the cause of the outcome. Use structured reasoning: identify the factor, trace its influence through the events, and conclude with 'Yes' or 'No' based on whether the outcome would have occurred without it.
- `disambiguation_qa` / `textgrad` / seed `0` / 0.660000：You are a coreference resolution expert. Given a sentence with an ambiguous pronoun and multiple possible antecedents, analyze the sentence for contextual clues, grammatical agreement, and narrative flow. Apply linguistic rules and prioritize the most salient and linguistically supported referent. Respond with only the letter of your answer, such as (E).
- `formal_fallacies` / `reelicit` / seed `0` / 0.660000：Analyze the deductive argument provided, which includes premises and a conclusion. Use formal logic to determine if the conclusion necessarily follows from the premises. Your answer must be either 'valid' or 'invalid' with no additional text.
- `gsm8k` / `textgrad` / seed `0` / 0.940000：You are a math problem solver. When given a grade-school math word problem, you will break it down into clear, logical steps. First, identify the question being asked. Then, extract the relevant numbers and relationships. Apply arithmetic operations in a logical sequence, explicitly stating each step and its intermediate result. Ensure that your reasoning is accurate, complete, and directly answers the question posed. Finally, provide the final numerical answer within a box.
- `hyperbaton` / `no_bo` / seed `0` / 0.960000：You are evaluating adjective order in English. Given two sentences with the same words but different adjective orders, select the one that is grammatically correct and natural in English. Use standard English adjective order conventions (e.g., opinion, size, age) when making your decision. Do not explain your reasoning—only respond with (A) or (B).
- `mmlu` / `static_features` / seed `0` / 0.900000：Analyze the question and all answer choices carefully, apply your knowledge to identify the most accurate response, and select the correct option. Provide only the letter (A, B, C, or D) corresponding to your answer.
- `penguins_in_a_table` / `promptbreeder` / seed `0` / 0.960000：Your task is to reason about the penguin data in the table and answer the question by selecting the correct option. Follow these steps: first, parse the table and identify the relevant attributes (name, age, height, weight). Second, analyze the question and determine what it is asking about the data. Third, use the data to logically deduce the correct answer. Ensure your reasoning is based solely on the data provided. Provide your final answer as a single letter (e.g., E).
- `snarks` / `independent_extraction` / seed `0` / 0.660000：Analyze the two statements carefully. Identify subtle differences in tone and implied meaning that may indicate sarcasm. Choose the statement that is most likely to be sarcastic. Respond with (A) or (B).
- `tracking_shuffled_objects` / `reelicit` / seed `0` / 0.920000：Be concise and logical. Analyze the sequence of swaps and keep a running tally of object ownership. Once all exchanges are processed, match the final positions of the objects to the correct answer choice. Output only the letter corresponding to your answer, like (D).

## 图像文件

- `figures/aggregate_method_comparison.png`
- `figures/boolean_expressions_convergence.png`
- `figures/causal_judgement_convergence.png`
- `figures/disambiguation_qa_convergence.png`
- `figures/formal_fallacies_convergence.png`
- `figures/gsm8k_convergence.png`
- `figures/hyperbaton_convergence.png`
- `figures/mmlu_convergence.png`
- `figures/penguins_in_a_table_convergence.png`
- `figures/snarks_convergence.png`
- `figures/tracking_shuffled_objects_convergence.png`

原始逐 run 数据见 `summary.csv`，各方法的完整轨迹见对应 `history.jsonl`。
