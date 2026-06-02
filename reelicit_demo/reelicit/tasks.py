import os
import random
import re
from typing import Any, Dict, List, Optional

from .types import TaskSpec


TASKS: Dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        name="GSM8K",
        size=500,
        context=(
            "The AI assistant receives a grade-school math word problem requiring multi-step "
            "arithmetic reasoning. It must produce a step-by-step solution ending with the final "
            "numerical answer. Performance is measured by exact-match accuracy on the final answer."
        ),
        format_instruction="The assistant should end with the final numerical answer.",
        output_format="numeric",
        input_template="Question:\n{question}\nAnswer:",
        answer_regex=r"[-+]?\d[\d,]*(?:\.\d+)?",
        dataset_name="gsm8k",
        dataset_config="main",
    ),
    "mmlu": TaskSpec(
        name="MMLU",
        size=500,
        context=(
            "The AI assistant receives a multiple-choice knowledge question drawn from diverse "
            "academic subjects including STEM, humanities, social sciences, and professional domains. "
            "It must select the correct answer from 4 options (A-D). Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just the letter choice, e.g. (A).",
        output_format="four_way",
        input_template="Question:\n{question}\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\nAnswer:",
        answer_regex=r"\b([A-D])\b",
        dataset_name="cais/mmlu",
        dataset_config="all",
    ),
    "boolean_expressions": TaskSpec(
        name="Boolean Expressions",
        size=250,
        context=(
            "The AI assistant receives a boolean expression composed of True/False values connected "
            "by logical operators (and, or, not) and must evaluate the expression to determine whether "
            "the result is True or False. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just True or False.",
        output_format="true_false",
        input_template="Q: {input}\nAnswer with just True or False.\nA:",
        answer_regex=r"(true|false)",
        dataset_name="lukaemon/bbh",
        dataset_config="boolean_expressions",
    ),
    "causal_judgement": TaskSpec(
        name="Causal Judgement",
        size=250,
        context=(
            "The AI assistant receives a scenario describing events and outcomes and must determine "
            "whether a specific factor was the cause of the outcome, answering Yes or No. This tests "
            "causal reasoning and counterfactual thinking. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just Yes or No.",
        output_format="yes_no",
        input_template="Q: {input}\nAnswer with just Yes or No.\nA:",
        answer_regex=r"(yes|no)",
        dataset_name="lukaemon/bbh",
        dataset_config="causal_judgement",
    ),
    "disambiguation_qa": TaskSpec(
        name="Disambiguation QA",
        size=250,
        context=(
            "The AI assistant receives a sentence with an ambiguous pronoun and must determine which "
            "entity the pronoun refers to, selecting from multiple choices. This tests coreference "
            "resolution and language understanding. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just the letter choice, e.g. (A).",
        output_format="letter",
        input_template="Q: {input}\nAnswer with just the letter choice, e.g. (A).\nA:",
        answer_regex=r"\(([A-F])\)",
        dataset_name="lukaemon/bbh",
        dataset_config="disambiguation_qa",
    ),
    "formal_fallacies": TaskSpec(
        name="Formal Fallacies",
        size=250,
        context=(
            "The AI assistant receives a deductive argument consisting of premises and a conclusion "
            "and must determine whether the argument is logically valid or invalid. This tests formal "
            "logical reasoning. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just valid or invalid.",
        output_format="valid_invalid",
        input_template="Q: {input}\nAnswer with just valid or invalid.\nA:",
        answer_regex=r"(valid|invalid)",
        dataset_name="lukaemon/bbh",
        dataset_config="formal_fallacies",
    ),
    "hyperbaton": TaskSpec(
        name="Hyperbaton",
        size=250,
        context=(
            "The AI assistant receives two sentences that differ only in adjective ordering and must "
            "determine which sentence has the correct (natural) English adjective order. This tests "
            "knowledge of English adjective ordering conventions. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just the letter choice, e.g. (A).",
        output_format="letter",
        input_template="Q: {input}\nAnswer with just the letter choice, e.g. (A).\nA:",
        answer_regex=r"\(([A-F])\)",
        dataset_name="lukaemon/bbh",
        dataset_config="hyperbaton",
    ),
    "penguins_in_a_table": TaskSpec(
        name="Penguins in a Table",
        size=250,
        context=(
            "The AI assistant receives a table of penguin attributes (name, age, height, weight) and "
            "a question about the data. It must parse the table, reason about the attributes, and "
            "select the correct answer from multiple choices. This tests structured data parsing and "
            "tabular reasoning. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just the letter choice, e.g. (A).",
        output_format="letter",
        input_template="Q: {input}\nAnswer with just the letter choice, e.g. (A).\nA:",
        answer_regex=r"\(([A-F])\)",
        dataset_name="lukaemon/bbh",
        dataset_config="penguins_in_a_table",
    ),
    "snarks": TaskSpec(
        name="Snarks",
        size=250,
        context=(
            "The AI assistant receives two nearly identical statements and must determine which one "
            "is sarcastic. This tests understanding of pragmatics, tone, and the distinction between "
            "literal and figurative meaning. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just the letter choice, e.g. (A).",
        output_format="letter",
        input_template="Q: {input}\nAnswer with just the letter choice, e.g. (A).\nA:",
        answer_regex=r"\(([A-F])\)",
        dataset_name="lukaemon/bbh",
        dataset_config="snarks",
    ),
    "tracking_shuffled_objects": TaskSpec(
        name="Tracking Shuffled Objects",
        size=250,
        context=(
            "The AI assistant receives a description of objects being swapped between people in a "
            "series of exchanges. It must track the final position of each object and select the "
            "correct answer from multiple choices. Performance is measured by accuracy."
        ),
        format_instruction="The assistant should answer with just the letter choice, e.g. (A).",
        output_format="letter",
        input_template="Q: {input}\nAnswer with just the letter choice, e.g. (A).\nA:",
        answer_regex=r"\(([A-F])\)",
        dataset_name="lukaemon/bbh",
        dataset_config="tracking_shuffled_objects",
    ),
}


def get_task(task_name: str) -> TaskSpec:
    key = task_name.lower()
    if key not in TASKS:
        raise KeyError("Unknown task '%s'. Available: %s" % (task_name, ", ".join(sorted(TASKS))))
    return TASKS[key]


def optimizer_context(task: TaskSpec, include_format: bool = False) -> str:
    if include_format and task.format_instruction:
        return task.context + " " + task.format_instruction
    return task.context


def extract_answer(task: TaskSpec, text: str) -> Optional[str]:
    flags = re.IGNORECASE if task.output_format != "four_way" else 0
    matches = list(re.finditer(task.answer_regex, text, flags=flags))
    if not matches:
        if task.output_format == "letter":
            bare = re.search(r"\b([A-F])\b", text)
            return bare.group(1).upper() if bare else None
        return None
    if task.output_format == "numeric":
        return normalize_number(matches[-1].group(0))
    value = matches[0].group(1 if matches[0].lastindex else 0)
    if task.output_format in ("true_false", "yes_no", "valid_invalid"):
        return value.strip().lower()
    return value.strip().upper()


def normalize_number(value: str) -> str:
    value = value.replace(",", "").strip()
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def load_examples(task: TaskSpec, seed: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load the benchmark examples used by the demo evaluator.

    This follows the paper's task sizes, but the exact fixed subsample seed was not
    published in the PDF. The demo therefore exposes the seed explicitly.
    """
    from datasets import load_dataset

    limit = limit or task.size
    portable_ds = load_portable_dataset(task)
    if task.name == "GSM8K":
        arrow_path = os.environ.get("REELICIT_GSM8K_TEST_ARROW")
        if portable_ds is not None:
            ds = portable_ds
        elif arrow_path and os.path.exists(arrow_path):
            from datasets import Dataset

            ds = Dataset.from_file(arrow_path)
        else:
            ds = load_dataset(task.dataset_name, task.dataset_config, split="test")
        rows = [
            {
                "input": task.input_template.format(question=row["question"]),
                "answer": extract_gsm8k_gold(row["answer"]),
            }
            for row in ds
        ]
    elif task.name == "MMLU":
        ds = portable_ds or load_dataset(task.dataset_name, task.dataset_config, split="test")
        rows = []
        for row in ds:
            choices = row.get("choices") or row.get("options")
            if isinstance(choices, str):
                choices = [choices]
            if not choices or len(choices) < 4:
                continue
            answer = row.get("answer")
            if isinstance(answer, int):
                gold = "ABCD"[answer]
            else:
                gold = str(answer).strip().upper()[:1]
            rows.append(
                {
                    "input": task.input_template.format(
                        question=row["question"],
                        A=choices[0],
                        B=choices[1],
                        C=choices[2],
                        D=choices[3],
                    ),
                    "answer": gold,
                }
            )
    else:
        if portable_ds is not None:
            ds = portable_ds
        elif task.dataset_config == "tracking_shuffled_objects":
            from datasets import concatenate_datasets

            ds = concatenate_datasets(
                [
                    load_dataset(
                        task.dataset_name,
                        "tracking_shuffled_objects_three_objects",
                        split="test",
                    ),
                    load_dataset(
                        task.dataset_name,
                        "tracking_shuffled_objects_five_objects",
                        split="test",
                    ),
                    load_dataset(
                        task.dataset_name,
                        "tracking_shuffled_objects_seven_objects",
                        split="test",
                    ),
                ]
            )
        else:
            ds = load_dataset(task.dataset_name, task.dataset_config, split="test")
        rows = []
        for row in ds:
            inp = row.get("input") or row.get("question") or row.get("inputs")
            target = row.get("target") or row.get("answer") or row.get("targets")
            if isinstance(target, list):
                target = target[0]
            if inp is None or target is None:
                continue
            rows.append(
                {
                    "input": task.input_template.format(input=inp),
                    "answer": normalize_gold(task, str(target)),
                }
            )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def load_portable_dataset(task: TaskSpec):
    data_root = os.environ.get("REELICIT_DATA_ROOT")
    if not data_root:
        return None
    path = os.path.join(data_root, task_key(task))
    if not os.path.exists(path):
        return None
    from datasets import load_from_disk

    loaded = load_from_disk(path)
    if hasattr(loaded, "keys"):
        if "test" in loaded:
            return loaded["test"]
        first_key = next(iter(loaded.keys()))
        return loaded[first_key]
    return loaded


def task_key(task: TaskSpec) -> str:
    for key, spec in TASKS.items():
        if spec.name == task.name:
            return key
    return task.name.lower().replace(" ", "_")


def extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return normalize_number(answer.strip())


def normalize_gold(task: TaskSpec, text: str) -> str:
    extracted = extract_answer(task, text)
    if extracted is not None:
        return extracted
    text = text.strip()
    if task.output_format in ("true_false", "yes_no", "valid_invalid"):
        return text.lower()
    if task.output_format == "letter":
        return text.replace("(", "").replace(")", "").strip().upper()[:1]
    return text
