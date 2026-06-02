import json
import os
import random
from typing import Dict, List, Sequence

import numpy as np

from .bo import cross_validate_mse, optimize_targets
from .evaluator import BaseEvaluator
from .llm import BaseLLMClient
from .prompts import (
    define_features_prompt,
    extract_features_prompt,
    feature_guided_refine_prompt,
    initial_dataset_prompt,
    initial_generate_prompt,
)
from .types import EvaluatedPrompt, Feature, RunConfig, RunResult, TaskSpec
from .tasks import optimizer_context
from .utils import (
    best_entry,
    clamp01,
    embedding_matrix,
    l2_distance,
    parse_json_relaxed,
    save_history_jsonl,
    set_seed,
    stratified_subsample,
    unique_features,
)


class ReElicitRunner:
    def __init__(
        self,
        task: TaskSpec,
        optimizer_llm: BaseLLMClient,
        evaluator: BaseEvaluator,
        config: RunConfig,
        seed: int = 0,
        output_dir: str = "runs/reelicit",
    ) -> None:
        self.task = task
        self.optimizer_llm = optimizer_llm
        self.evaluator = evaluator
        self.config = config
        self.seed = seed
        self.output_dir = output_dir
        self.rng = random.Random(seed)
        self.task_context = optimizer_context(
            self.task, include_format=self.config.include_format_in_context
        )
        set_seed(seed)

    def run(self) -> RunResult:
        os.makedirs(self.output_dir, exist_ok=True)
        history = self._initial_dataset()
        incumbent_features: List[Feature] = []

        self._write_state(0, history, incumbent_features)
        frozen_features: List[Feature] = []
        for t in range(1, self.config.T):
            print(
                f"[ReElicit] iteration {t}/{self.config.T - 1}: "
                f"history={len(history)} best={best_entry(history).score:.4f}"
            )
            if self.config.static_features and frozen_features:
                features = frozen_features
                embeddings = self._extract_features([e.prompt for e in history], features)
                z = embedding_matrix(embeddings, features)
                print(f"  static feature set reused: d={len(features)}")
            else:
                features, z = self._select_feature_set(t, history, incumbent_features)
                if self.config.static_features and not frozen_features:
                    frozen_features = features
            self._attach_embeddings(history, features, z)
            y = np.asarray([entry.score for entry in history], dtype=float)
            if self.config.no_bo:
                rng = np.random.RandomState(self.seed + 1000 * t)
                targets = rng.rand(self.config.q, len(features))
            else:
                targets = optimize_targets(
                    z=z,
                    y=y,
                    q=self.config.q,
                    seed=self.seed + 1000 * t,
                    num_restarts=self.config.bo_num_restarts,
                    raw_samples=self.config.bo_raw_samples,
                    strict_paper=self.config.strict_paper,
                )
            new_entries = []
            for j, target_vec in enumerate(targets):
                target = {features[i].name: float(target_vec[i]) for i in range(len(features))}
                prompt = self._generate_with_refinement(t, j, history, features, target)
                score = self.evaluator.evaluate(prompt)
                new_entries.append(
                    EvaluatedPrompt(
                        prompt=prompt,
                        score=score,
                        meta={"iteration": str(t), "candidate": str(j), "method": "ReElicit"},
                    )
                )
                print(f"  candidate {j + 1}/{self.config.q}: score={score:.4f}")
            history.extend(new_entries)
            incumbent_features = features
            self._write_state(t, history, incumbent_features)

        best = best_entry(history)
        return RunResult(best_prompt=best.prompt, best_score=best.score, history=history)

    def _initial_dataset(self) -> List[EvaluatedPrompt]:
        prompt = initial_dataset_prompt(self.task_context, self.config.q)
        raw = self.optimizer_llm.complete(
            prompt, temperature=self.config.optimizer_temperature, max_tokens=2048
        )
        try:
            prompts = parse_json_relaxed(raw)
        except Exception:
            prompts = []
        if not isinstance(prompts, list):
            prompts = []
        prompts = [str(p).strip() for p in prompts if str(p).strip()]
        while len(prompts) < self.config.q:
            prompts.append(
                "Solve the task carefully. Reason through the relevant details and return only the final answer in the requested format."
            )
        prompts = prompts[: self.config.q]
        history = []
        for i, system_prompt in enumerate(prompts):
            score = self.evaluator.evaluate(system_prompt)
            history.append(
                EvaluatedPrompt(
                    prompt=system_prompt,
                    score=score,
                    meta={"iteration": "0", "candidate": str(i), "method": "D0"},
                )
            )
            print(f"[D0] prompt {i + 1}/{self.config.q}: score={score:.4f}")
        return history

    def _select_feature_set(
        self,
        t: int,
        history: Sequence[EvaluatedPrompt],
        incumbent_features: Sequence[Feature],
    ) -> tuple:
        candidates = []
        for k in range(self.config.K):
            features = self._define_features(history, incumbent_features if t > 1 else [])
            embeddings = self._extract_features([e.prompt for e in history], features)
            z = embedding_matrix(embeddings, features)
            mse = cross_validate_mse(
                z,
                np.asarray([e.score for e in history]),
                seed=self.seed + k,
                use_botorch=self.config.strict_paper,
            )
            candidates.append((mse, features, z, "new-%d" % k))
            print(f"  feature set {k + 1}/{self.config.K}: d={len(features)} cv_mse={mse:.6f}")

        if t > 1 and incumbent_features:
            embeddings = self._extract_features([e.prompt for e in history], incumbent_features)
            z = embedding_matrix(embeddings, incumbent_features)
            mse = cross_validate_mse(
                z,
                np.asarray([e.score for e in history]),
                seed=self.seed + 99,
                use_botorch=self.config.strict_paper,
            )
            candidates.append((mse, list(incumbent_features), z, "incumbent"))
            print(f"  incumbent feature set: d={len(incumbent_features)} cv_mse={mse:.6f}")

        mse, features, z, label = min(candidates, key=lambda row: row[0])
        print(f"  selected feature set '{label}': d={len(features)} cv_mse={mse:.6f}")
        return features, z

    def _define_features(
        self,
        history: Sequence[EvaluatedPrompt],
        incumbent_features: Sequence[Feature],
    ) -> List[Feature]:
        prompt = define_features_prompt(self.task_context, history, incumbent_features)
        raw = self.optimizer_llm.complete(
            prompt, temperature=self.config.optimizer_temperature, max_tokens=4096
        )
        return unique_features(parse_json_relaxed(raw))

    def _extract_features(
        self, prompts: Sequence[str], features: Sequence[Feature]
    ) -> List[Dict[str, float]]:
        embeddings: Dict[str, Dict[str, float]] = {}
        indexed = [{"id": str(i), "text": text} for i, text in enumerate(prompts)]
        batch_size = 1 if self.config.independent_extraction else self.config.b
        for chunk in [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]:
            prompt = extract_features_prompt(self.task_context, features, chunk)
            raw = self.optimizer_llm.complete(
                prompt, temperature=self.config.optimizer_temperature, max_tokens=4096
            )
            parsed = parse_json_relaxed(raw)
            if not isinstance(parsed, dict):
                parsed = {}
            for item in chunk:
                values = parsed.get(item["id"], {})
                if not isinstance(values, dict):
                    values = {}
                embeddings[item["id"]] = {
                    feature.name: clamp01(values.get(feature.name, 0.5)) for feature in features
                }
        return [embeddings[str(i)] for i in range(len(prompts))]

    def _attach_embeddings(
        self, history: Sequence[EvaluatedPrompt], features: Sequence[Feature], z: np.ndarray
    ) -> None:
        for entry, row in zip(history, z):
            entry.embedding = {features[i].name: float(row[i]) for i in range(len(features))}

    def _generate_with_refinement(
        self,
        t: int,
        j: int,
        history: Sequence[EvaluatedPrompt],
        features: Sequence[Feature],
        target: Dict[str, float],
    ) -> str:
        minit = max(1, self.config.M // 2)
        mrefine = 0 if self.config.no_refinement else self.config.M - minit
        best_prompt = ""
        best_embedding: Dict[str, float] = {}
        best_gap = float("inf")

        for p in range(minit):
            rng = random.Random(self.seed + 100000 * t + 1000 * j + p)
            examples = stratified_subsample(history, self.config.nmax, rng)
            raw_prompt = initial_generate_prompt(self.task_context, examples, features, target)
            candidate = self.optimizer_llm.complete(
                raw_prompt, temperature=self.config.optimizer_temperature, max_tokens=2048
            ).strip()
            emb = self._extract_features([candidate], features)[0]
            gap = self._gap(target, emb, features)
            if gap < best_gap:
                best_prompt = candidate
                best_embedding = emb
                best_gap = gap

        for step in range(mrefine):
            if best_gap <= self.config.tau:
                break
            rng = random.Random(self.seed + 200000 * t + 1000 * j + step)
            examples = stratified_subsample(history, self.config.nmax, rng)
            refine_prompt = feature_guided_refine_prompt(
                self.task_context, examples, best_prompt, features, target, best_embedding
            )
            candidate = self.optimizer_llm.complete(
                refine_prompt, temperature=self.config.optimizer_temperature, max_tokens=2048
            ).strip()
            emb = self._extract_features([candidate], features)[0]
            gap = self._gap(target, emb, features)
            if gap < best_gap:
                best_prompt = candidate
                best_embedding = emb
                best_gap = gap
        return best_prompt

    def _gap(
        self, target: Dict[str, float], embedding: Dict[str, float], features: Sequence[Feature]
    ) -> float:
        a = [float(target[f.name]) for f in features]
        b = [float(embedding.get(f.name, 0.5)) for f in features]
        return l2_distance(a, b)

    def _write_state(
        self, iteration: int, history: Sequence[EvaluatedPrompt], features: Sequence[Feature]
    ) -> None:
        save_history_jsonl(os.path.join(self.output_dir, "history.jsonl"), history)
        state = {
            "iteration": iteration,
            "best_score": best_entry(history).score,
            "best_prompt": best_entry(history).prompt,
            "features": [feature.__dict__ for feature in features],
        }
        with open(os.path.join(self.output_dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=2)
