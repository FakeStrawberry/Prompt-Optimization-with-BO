import json
import os
import random
from typing import List, Sequence

from .evaluator import BaseEvaluator
from .llm import BaseLLMClient
from .prompts import (
    ape_prompt,
    initial_dataset_prompt,
    opro_prompt,
    promptbreeder_mutation_prompt,
    promptbreeder_recombination_prompt,
    textgrad_prompt,
)
from .tasks import optimizer_context
from .types import EvaluatedPrompt, RunConfig, RunResult, TaskSpec
from .utils import best_entry, parse_json_relaxed, save_history_jsonl, stratified_subsample


class BaselineRunner:
    def __init__(
        self,
        method: str,
        task: TaskSpec,
        optimizer_llm: BaseLLMClient,
        evaluator: BaseEvaluator,
        config: RunConfig,
        seed: int,
        output_dir: str,
    ) -> None:
        self.method = method
        self.task = task
        self.optimizer_llm = optimizer_llm
        self.evaluator = evaluator
        self.config = config
        self.seed = seed
        self.output_dir = output_dir
        self.rng = random.Random(seed)
        self.task_context = optimizer_context(task, config.include_format_in_context)

    def run(self) -> RunResult:
        os.makedirs(self.output_dir, exist_ok=True)
        history = self._initial_dataset()
        self._write_state(0, history)
        for t in range(1, self.config.T):
            print(
                "[%s] iteration %d/%d: history=%d best=%.4f"
                % (self.method, t, self.config.T - 1, len(history), best_entry(history).score)
            )
            prompts = self._generate_batch(history)
            new_entries: List[EvaluatedPrompt] = []
            for j, prompt in enumerate(prompts[: self.config.q]):
                score = self.evaluator.evaluate(prompt)
                entry = EvaluatedPrompt(
                    prompt=prompt,
                    score=score,
                    meta={"iteration": str(t), "candidate": str(j), "method": self.method},
                )
                new_entries.append(entry)
                print("  candidate %d/%d: score=%.4f" % (j + 1, self.config.q, score))
            history.extend(new_entries)
            self._write_state(t, history)
        best = best_entry(history)
        return RunResult(best.prompt, best.score, history)

    def _initial_dataset(self) -> List[EvaluatedPrompt]:
        raw = self.optimizer_llm.complete(
            initial_dataset_prompt(self.task_context, self.config.q),
            temperature=self.config.optimizer_temperature,
            max_tokens=2048,
        )
        prompts = _json_string_list(raw)
        while len(prompts) < self.config.q:
            prompts.append(
                "Solve the task carefully and answer in exactly the requested final format."
            )
        history = []
        for i, prompt in enumerate(prompts[: self.config.q]):
            score = self.evaluator.evaluate(prompt)
            history.append(
                EvaluatedPrompt(
                    prompt=prompt,
                    score=score,
                    meta={"iteration": "0", "candidate": str(i), "method": "D0"},
                )
            )
            print("[D0] prompt %d/%d: score=%.4f" % (i + 1, self.config.q, score))
        return history

    def _generate_batch(self, history: Sequence[EvaluatedPrompt]) -> List[str]:
        method = self.method.lower()
        if method == "ape":
            raw = self.optimizer_llm.complete(
                ape_prompt(self.task_context, self.config.q),
                temperature=self.config.optimizer_temperature,
                max_tokens=2048,
            )
            return _ensure_q(_json_string_list(raw), self.config.q)

        if method == "opro":
            sample = stratified_subsample(history, self.config.nmax, self.rng)
            raw = self.optimizer_llm.complete(
                opro_prompt(self.task_context, sample, self.config.q),
                temperature=self.config.optimizer_temperature,
                max_tokens=4096,
            )
            return _ensure_q(_json_string_list(raw), self.config.q)

        if method == "textgrad":
            sample = stratified_subsample(history, self.config.nmax, self.rng)
            best = best_entry(history)
            raw = self.optimizer_llm.complete(
                textgrad_prompt(self.task_context, sample, best.prompt, best.score, self.config.q),
                temperature=self.config.optimizer_temperature,
                max_tokens=4096,
            )
            return _ensure_q(_json_string_list(raw), self.config.q)

        if method == "promptbreeder":
            population = sorted(history, key=lambda e: e.score, reverse=True)[: self.config.pmax]
            prompts: List[str] = []
            instructions = [
                "Rewrite the following system prompt to be clearer and more precise. Keep the core instructions but improve clarity.",
                "Modify the following system prompt to make reasoning steps more explicit. Add instructions for step-by-step thinking.",
                "Make the following system prompt more concise. Remove redundancy while preserving all important constraints.",
            ]
            for j in range(self.config.q):
                if j < self.config.q - 1 or len(population) < 2:
                    parent = self.rng.choice(population).prompt
                    inst = self.rng.choice(instructions)
                    raw = self.optimizer_llm.complete(
                        promptbreeder_mutation_prompt(self.task_context, parent, inst),
                        temperature=self.config.optimizer_temperature,
                        max_tokens=2048,
                    )
                else:
                    p1, p2 = self.rng.sample(population, 2)
                    raw = self.optimizer_llm.complete(
                        promptbreeder_recombination_prompt(
                            self.task_context, p1.prompt, p2.prompt
                        ),
                        temperature=self.config.optimizer_temperature,
                        max_tokens=2048,
                    )
                prompts.append(raw.strip())
            return _ensure_q(prompts, self.config.q)

        raise ValueError("Unknown baseline method: %s" % self.method)

    def _write_state(self, iteration: int, history: Sequence[EvaluatedPrompt]) -> None:
        save_history_jsonl(os.path.join(self.output_dir, "history.jsonl"), history)
        best = best_entry(history)
        with open(os.path.join(self.output_dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "iteration": iteration,
                    "method": self.method,
                    "best_score": best.score,
                    "best_prompt": best.prompt,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )


def _json_string_list(raw: str) -> List[str]:
    try:
        parsed = parse_json_relaxed(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip() for x in parsed if str(x).strip()]


def _ensure_q(prompts: List[str], q: int) -> List[str]:
    while len(prompts) < q:
        prompts.append(
            "Solve the task carefully and answer in exactly the requested final format."
        )
    return prompts[:q]
