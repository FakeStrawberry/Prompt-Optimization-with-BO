import json
from typing import Dict, List, Sequence

from .types import EvaluatedPrompt, Feature


def _feature_lines(features: Sequence[Feature]) -> str:
    return "\n".join(f"- {f.name}: {f.description}" for f in features)


def _tiered_examples(
    examples: Sequence[EvaluatedPrompt],
    include_embeddings: bool = False,
) -> str:
    if not examples:
        return ""
    sorted_examples = sorted(examples, key=lambda e: e.score, reverse=True)
    cutoff = max(1, len(sorted_examples) // 2)
    rows: List[str] = []
    for i, entry in enumerate(sorted_examples):
        tier = "TOP" if i < cutoff else "BOTTOM"
        rows.append(f"-- [{tier}] Prompt {i} (Score: {entry.score:.6f}) --")
        if include_embeddings and entry.embedding:
            rows.append("Feature values: " + json.dumps(entry.embedding, sort_keys=True))
        rows.append(entry.prompt.strip())
    return "\n".join(rows)


def define_features_prompt(
    task_context: str,
    history: Sequence[EvaluatedPrompt],
    incumbent_features: Sequence[Feature],
) -> str:
    incumbent = ""
    if incumbent_features:
        incumbent = (
            "The following features are currently in use for this task:\n"
            + "\n".join(f"- {f.name}: {f.description}" for f in incumbent_features)
            + "\nAnalyze the data to identify patterns or properties NOT captured by these features. "
            "You may keep features that are clearly the most predictive, but your proposed set MUST "
            "differ from the current set by at least one meaningful change: add a genuinely new "
            "feature, remove a feature, or substantively redefine one. Renaming or trivially "
            "rephrasing an existing feature does NOT count as a change.\n\n"
        )
    return f"""You are an expert at analyzing text objects and identifying patterns that predict performance.
These text objects are system prompts for an AI assistant performing the following task: {task_context}

Define numerical features that capture what makes a prompt effective FOR THIS SPECIFIC TASK. Focus on properties that have a CAUSAL relationship to the AI's ability to solve this type of problem - properties that, if changed in a prompt, would directly affect performance. Avoid features that merely correlate with score or describe surface-level text properties without a causal mechanism.

Before defining features, closely inspect the data:
1. What specific patterns or properties are present in the TOP-performing text objects but absent from the BOTTOM ones?
2. What do the BOTTOM-performing text objects have that the TOP ones don't?
3. List 2-3 concrete observations about these differences.

Then define features that capture these observed causal differences.

Requirements for each feature:
- name: A short identifier.
- description: Explain what the feature measures for this specific task, what 0 and 1 represent (anchor semantics), and why it causally affects performance.
- All feature values must be in [0, 1].
- Each feature MUST be INDEPENDENT of the others. If you can predict one feature's value from the others, they are redundant - keep only the more causally relevant one.
- Before finalizing, verify: for each pair of features, can you imagine a text that is high on one and low on the other? If not, they are not independent - drop one.

Choose the number of features based on the available data. With small datasets (10 or fewer examples), prefer fewer features (1-2) - a single well-chosen feature is better than several noisy ones. As the dataset grows and patterns become clearer, additional features can capture richer structure that was not visible with less data. Every feature must earn its place by capturing genuinely independent variation.

Respond with a JSON array of objects, each with 'name' and 'description' fields.

{incumbent}Here are {len(history)} text objects with their performance scores (higher is better), sorted by performance tier:
{_tiered_examples(history, include_embeddings=False)}

You have {len(history)} text objects available. Be mindful of this sample size - fewer high-quality, genuinely distinct features are far better than many overlapping ones.
First list your observations about what distinguishes TOP from BOTTOM performers, then define features.
Return ONLY the JSON array."""


def extract_features_prompt(
    task_context: str, features: Sequence[Feature], texts: Sequence[Dict[str, str]]
) -> str:
    text_block = "\n".join(
        f'-- Text Object ID: "{item["id"]}" --\n{item["text"]}' for item in texts
    )
    return f"""You are an expert at analyzing text and rating it on specific features. For each text object, assign a value in [0, 1] for each feature based on the feature description.
These text objects are system prompts for an AI assistant performing the following task: {task_context}

Rate each text object considering how the features relate to this specific task.
Be consistent: similar texts should get similar scores. Use the full range of [0, 1] - don't cluster all values near the middle.

Respond with a JSON object keyed by text object ID, where each value is an object mapping feature names to numeric values.
Example:
{{
  "0": {{"feature_a": 0.75, "feature_b": 0.30}},
  "1": {{"feature_a": 0.45, "feature_b": 0.80}}
}}

Features to rate:
{_feature_lines(features)}

Text objects to rate:
{text_block}

Rate each text object on each feature. Values must be numbers in [0, 1]. Return ONLY the JSON object."""


def initial_generate_prompt(
    task_context: str,
    examples: Sequence[EvaluatedPrompt],
    features: Sequence[Feature],
    target: Dict[str, float],
) -> str:
    return f"""You are an expert prompt engineer. Generate a system prompt for an AI assistant performing the following task: {task_context}
The prompt should match specific target feature values.

You will be given:
1. Feature definitions with their semantics.
2. Example prompts labeled [TOP] or [BOTTOM] by performance, with their feature values and scores.
3. A target feature vector to aim for.

Study what makes the TOP-scoring examples effective and what makes the BOTTOM-scoring examples less effective. Learn from the best examples - understand the patterns and approaches that lead to high performance.

The target feature vector indicates a promising direction to explore. Generate a NEW system prompt that combines the successful patterns from the TOP examples while matching the target feature values.

Output ONLY the generated prompt text, with no additional commentary or formatting.

Feature definitions:
{_feature_lines(features)}

Example prompts (sorted by performance, with tier labels):
{_tiered_examples(examples, include_embeddings=True)}

Target feature vector:
{json.dumps(target, sort_keys=True)}

Generate a system prompt that combines the best patterns from the TOP examples while matching the target features. Output ONLY the prompt text."""


def feature_guided_refine_prompt(
    task_context: str,
    examples: Sequence[EvaluatedPrompt],
    current_prompt: str,
    features: Sequence[Feature],
    target: Dict[str, float],
    current: Dict[str, float],
) -> str:
    gaps = []
    feature_map = {f.name: f.description for f in features}
    for name, target_value in target.items():
        cur = float(current.get(name, 0.5))
        gap = float(target_value) - cur
        gaps.append(
            {
                "feature_name": name,
                "definition": feature_map.get(name, ""),
                "target": float(target_value),
                "current": cur,
                "gap": abs(gap),
                "direction": "increase" if gap > 0 else "decrease",
            }
        )
    gaps.sort(key=lambda row: row["gap"], reverse=True)
    return f"""You are an expert prompt engineer. Modify the given system prompt to better match target feature values.
The system prompt is for an AI assistant performing the following task: {task_context}

Consider what text patterns in the reference examples correspond to the desired feature values, and what specific phrases in the current prompt are causing the gaps.

Rules:
- Focus on the LARGEST gaps first (they are listed in order of priority).
- MODIFY the existing text - do not rewrite from scratch.
- PRESERVE aspects that are already well-aligned with their targets.
- Output ONLY the modified prompt text, with no additional commentary or formatting.

Reference examples (sorted by performance):
{_tiered_examples(examples, include_embeddings=True)}

Use the TOP examples as reference for the style and patterns that correspond to the desired feature values.

Current system prompt:
{current_prompt}

Feature gap analysis (sorted by gap magnitude, largest first):
{json.dumps(gaps, indent=2, sort_keys=True)}

Modify the system prompt to reduce the largest feature gaps. Output ONLY the modified prompt text."""


def initial_dataset_prompt(task_context: str, q: int) -> str:
    return f"""You are an expert prompt engineer. Generate diverse system prompts for an AI assistant.
Task description:
{task_context}

Generate exactly {q} diverse system prompts that would help an AI assistant perform well on this task.
Each prompt should take a different approach (e.g., step-by-step reasoning, concise instructions, structured format, etc.).
Return a JSON array of {q} strings, where each string is a complete system prompt."""


def ape_prompt(task_context: str, q: int) -> str:
    return f"""You are an expert prompt engineer. Generate diverse system prompts for an AI assistant to help it perform well on a specific task.
Task description:
{task_context}

Generate exactly {q} diverse system prompts. Each should take a different approach (e.g., step-by-step reasoning, concise instructions, structured format, direct commands, role-playing, etc.).
Return a JSON array of {q} strings, where each string is a complete system prompt."""


def opro_prompt(task_context: str, history: Sequence[EvaluatedPrompt], q: int) -> str:
    rows = []
    for entry in sorted(history, key=lambda e: e.score):
        rows.append(f"-- Prompt (Score: {entry.score:.6f}) --\n{entry.prompt.strip()}")
    return f"""You are an expert prompt optimizer. Analyze previous system prompts and their performance scores, then generate improved prompts.
Task description:
{task_context}

Here are previous system prompts and their scores (higher is better), sorted from worst to best:
{chr(10).join(rows)}

Analyze what makes the higher-scoring prompts better. Then generate exactly {q} new system prompts that should score even higher.
Return a JSON array of {q} strings."""


def textgrad_prompt(
    task_context: str,
    history: Sequence[EvaluatedPrompt],
    best_prompt: str,
    best_score: float,
    q: int,
) -> str:
    rows = []
    for entry in sorted(history, key=lambda e: e.score):
        rows.append(f"-- Prompt (Score: {entry.score:.6f}) --\n{entry.prompt.strip()}")
    return f"""You are an expert prompt optimizer. Given a trajectory of system prompts and their performance scores, analyze what makes some perform better than others, then critique the current best prompt and generate improved variants.
Task description:
{task_context}

Trajectory of prior prompts and their scores (higher is better, sorted worst-to-best):
{chr(10).join(rows)}

Current best prompt (score: {best_score:.6f}):
{best_prompt}

Step 1: briefly analyze the trajectory - what patterns separate high-scoring prompts from low-scoring ones?
Step 2: critique the current best prompt: what could be improved to get a higher score?
Step 3: generate exactly {q} improved variants based on your analysis and critique. Each variant should address a different aspect of the critique.
Return a JSON array of {q} strings, where each string is a complete improved system prompt."""


def promptbreeder_mutation_prompt(
    task_context: str, parent_prompt: str, instruction: str
) -> str:
    return f"""You are an expert prompt engineer. Modify system prompts to improve their effectiveness.
Task description:
{task_context}

Instruction: {instruction}

Original system prompt:
{parent_prompt}

Output ONLY the modified system prompt, no commentary."""


def promptbreeder_recombination_prompt(
    task_context: str, parent1: str, parent2: str
) -> str:
    return f"""You are an expert prompt engineer. Combine the best aspects of two system prompts into a single improved prompt.
Task description:
{task_context}

Parent prompt 1:
{parent1}

Parent prompt 2:
{parent2}

Create a new system prompt that combines the best aspects of both parents. Output ONLY the new prompt, no commentary."""
