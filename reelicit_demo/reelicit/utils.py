import hashlib
import json
import math
import os
import random
import re
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .types import EvaluatedPrompt, Feature


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_hash_float(text: str, low: float = 0.0, high: float = 1.0) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    return low + (high - low) * value


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except Exception:
        x = default
    if math.isnan(x) or math.isinf(x):
        x = default
    return max(0.0, min(1.0, x))


def l2_distance(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    return float(np.linalg.norm(av - bv))


def chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def parse_json_relaxed(text: str) -> Any:
    """Parse JSON returned by an LLM, tolerating fenced blocks and extra prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    starts = sorted(i for i, ch in enumerate(text) if ch in "[{")
    for start in starts:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("No complete JSON object or array found in LLM response")


def unique_features(raw_features: Any) -> List[Feature]:
    if not isinstance(raw_features, list):
        raise ValueError("Feature response must be a JSON array")
    out: List[Feature] = []
    seen = set()
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        desc = str(item.get("description", "")).strip()
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
        if not name or not desc or name in seen:
            continue
        seen.add(name)
        out.append(Feature(name=name, description=desc))
    if not out:
        raise ValueError("No valid features found")
    return out


def embedding_matrix(
    embeddings: Sequence[Dict[str, float]], features: Sequence[Feature]
) -> np.ndarray:
    return np.asarray(
        [[clamp01(emb.get(f.name, 0.5)) for f in features] for emb in embeddings],
        dtype=float,
    )


def stratified_subsample(
    history: Sequence[EvaluatedPrompt], nmax: int, rng: random.Random
) -> List[EvaluatedPrompt]:
    """Paper's shared utility: top 25%, bottom 25%, random middle fill."""
    items = list(history)
    if len(items) <= nmax:
        return sorted(items, key=lambda e: e.score)

    sorted_items = sorted(items, key=lambda e: e.score)
    n = len(sorted_items)
    group = max(1, int(round(0.25 * nmax)))
    bottom = sorted_items[: max(1, n // 4)]
    top = sorted_items[-max(1, n // 4) :]
    middle = sorted_items[max(1, n // 4) : n - max(1, n // 4)]

    selected: List[EvaluatedPrompt] = []
    selected.extend(rng.sample(bottom, min(group, len(bottom))))
    selected.extend(rng.sample(top, min(group, len(top))))
    remaining = nmax - len(selected)
    pool = [x for x in middle if x not in selected]
    if remaining > 0 and pool:
        selected.extend(rng.sample(pool, min(remaining, len(pool))))

    if len(selected) < nmax:
        pool = [x for x in sorted_items if x not in selected]
        selected.extend(rng.sample(pool, min(nmax - len(selected), len(pool))))

    return sorted(selected, key=lambda e: e.score)


def best_entry(history: Sequence[EvaluatedPrompt]) -> EvaluatedPrompt:
    return max(history, key=lambda e: e.score)


def save_history_jsonl(path: str, history: Sequence[EvaluatedPrompt]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i, entry in enumerate(history):
            row = {
                "index": i,
                "score": entry.score,
                "prompt": entry.prompt,
                "embedding": entry.embedding,
                "meta": entry.meta,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
