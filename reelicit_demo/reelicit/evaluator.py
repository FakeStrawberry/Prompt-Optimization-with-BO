import os
import time
from typing import Optional

import numpy as np

from .llm import BaseLLMClient
from .tasks import extract_answer, load_examples
from .types import TaskSpec
from .utils import stable_hash_float


class BaseEvaluator:
    def evaluate(self, system_prompt: str) -> float:
        raise NotImplementedError


class MockEvaluator(BaseEvaluator):
    """Fast scalar-only objective for local smoke tests."""

    def __init__(self, task: TaskSpec, seed: int = 0) -> None:
        self.task = task
        self.seed = seed

    def evaluate(self, system_prompt: str) -> float:
        text = system_prompt.lower()
        terms = {
            "format": ["only", "just", "final", "format", "letter", "true", "false", "yes", "no", "valid"],
            "reason": ["reason", "step", "careful", "analyze", "track", "parse", "compute", "check"],
            "specific": [
                "sarcasm",
                "literal",
                "figurative",
                "logical",
                "operator",
                "causal",
                "pronoun",
                "table",
                "swap",
                "adjective",
                "arithmetic",
                "task-specific",
            ],
        }
        score = 0.42
        score += 0.018 * sum(1 for t in terms["format"] if t in text)
        score += 0.016 * sum(1 for t in terms["reason"] if t in text)
        score += 0.018 * sum(1 for t in terms["specific"] if t in text)
        if len(text.split()) > 150:
            score -= 0.04
        score += stable_hash_float(system_prompt + str(self.seed) + self.task.name, -0.015, 0.015)
        return float(np.clip(score, 0.0, 1.0))


class OpenAIChatBenchmarkEvaluator(BaseEvaluator):
    """Benchmark evaluator with the paper's aggregate-only interface.

    It runs a target chat model with the candidate system prompt and returns one
    scalar accuracy. Per-example outputs are deliberately not exposed to the
    optimizer.
    """

    def __init__(
        self,
        task: TaskSpec,
        target_llm: BaseLLMClient,
        seed: int = 0,
        limit: Optional[int] = None,
        target_temperature: float = 0.0,
    ) -> None:
        self.task = task
        self.target_llm = target_llm
        self.examples = load_examples(task, seed=seed, limit=limit or task.size)
        self.target_temperature = target_temperature

    def evaluate(self, system_prompt: str) -> float:
        correct = 0
        show_progress = os.environ.get("REELICIT_EVAL_PROGRESS", "").lower() in {
            "1",
            "true",
            "yes",
        }
        start = time.time()
        total = len(self.examples)
        for i, row in enumerate(self.examples, start=1):
            output = self.target_llm.chat(
                system_prompt=system_prompt,
                user_prompt=row["input"],
                temperature=self.target_temperature,
                max_tokens=512,
            )
            pred = extract_answer(self.task, output)
            if pred is not None and pred == row["answer"]:
                correct += 1
            if show_progress and (i == total or i % 25 == 0):
                elapsed = time.time() - start
                print(
                    "[eval] %d/%d correct=%d acc=%.4f elapsed=%.1fs"
                    % (i, total, correct, correct / float(i), elapsed),
                    flush=True,
                )
        return correct / float(total or 1)
