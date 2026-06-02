import argparse
import json
import os
from typing import List

from demo import build_config, load_json
from reelicit.algorithm import ReElicitRunner
from reelicit.baselines import BaselineRunner
from reelicit.evaluator import MockEvaluator, OpenAIChatBenchmarkEvaluator
from reelicit.llm import LocalTransformersLLMClient, MockLLMClient
from reelicit.tasks import get_task


REELICIT_VARIANTS = {
    "reelicit",
    "no_refinement",
    "no_bo",
    "static_features",
    "independent_extraction",
}


def parse_words(value: str) -> List[str]:
    return [x for x in value.replace(",", " ").split() if x]


def configure_variant(base_config, method: str):
    config = build_config(base_config, strict_paper=False)
    if method == "no_refinement":
        config.no_refinement = True
    elif method == "no_bo":
        config.no_bo = True
    elif method == "static_features":
        config.static_features = True
    elif method == "independent_extraction":
        config.independent_extraction = True
    return config


def _completed(out_dir: str, expected_rows: int) -> bool:
    state_path = os.path.join(out_dir, "state.json")
    history_path = os.path.join(out_dir, "history.jsonl")
    if not os.path.exists(state_path) or not os.path.exists(history_path):
        return False
    try:
        with open(history_path, "r", encoding="utf-8") as fh:
            rows = sum(1 for line in fh if line.strip())
    except OSError:
        return False
    return rows >= expected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-process multi-method suite")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--tasks", default="snarks")
    parser.add_argument(
        "--methods",
        default="reelicit ape opro promptbreeder textgrad no_refinement no_bo static_features independent_extraction",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--mode", choices=["mock", "local-benchmark"], default="local-benchmark")
    parser.add_argument("--local-model", default="/nas1/zyj/models/Qwen3-14B")
    parser.add_argument("--local-target-model", default="/nas1/zyj/models/Qwen3-8B")
    parser.add_argument("--optimizer-device", default="cuda:1")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--include-format-in-context", action="store_true")
    parser.add_argument("--output-root", default="runs/paper_repro")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    raw_config = load_json(args.config)
    tasks = parse_words(args.tasks)
    methods = parse_words(args.methods)
    seeds = [int(x) for x in parse_words(args.seeds)]

    if args.mode == "mock":
        optimizer = MockLLMClient()
        target = None
    else:
        optimizer = LocalTransformersLLMClient(args.local_model, device=args.optimizer_device)
        target = LocalTransformersLLMClient(args.local_target_model, device=args.target_device)

    os.makedirs(args.output_root, exist_ok=True)
    with open(os.path.join(args.output_root, "suite_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, ensure_ascii=False)

    for task_name in tasks:
        task = get_task(task_name)
        for seed in seeds:
            for method in methods:
                config = configure_variant(raw_config, method)
                for name in ("q", "T", "K", "M"):
                    value = getattr(args, name)
                    if value is not None:
                        setattr(config, name, value)
                config.include_format_in_context = args.include_format_in_context

                if args.mode == "mock":
                    evaluator = MockEvaluator(task, seed)
                else:
                    evaluator = OpenAIChatBenchmarkEvaluator(
                        task,
                        target,
                        seed=seed,
                        limit=args.limit,
                        target_temperature=config.target_temperature,
                    )
                out_dir = os.path.join(args.output_root, task_name, method, "seed_%d" % seed)
                os.makedirs(out_dir, exist_ok=True)
                if args.skip_completed and _completed(out_dir, config.q * config.T):
                    print(
                        "SKIP task=%s seed=%d method=%s: completed output exists"
                        % (task_name, seed, method),
                        flush=True,
                    )
                    continue
                with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "task": task_name,
                            "seed": seed,
                            "method": method,
                            "limit": args.limit,
                            "q": config.q,
                            "T": config.T,
                            "K": config.K,
                            "M": config.M,
                            "include_format_in_context": config.include_format_in_context,
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )
                print("=== task=%s seed=%d method=%s ===" % (task_name, seed, method), flush=True)
                if method in REELICIT_VARIANTS:
                    result = ReElicitRunner(task, optimizer, evaluator, config, seed, out_dir).run()
                else:
                    result = BaselineRunner(
                        method, task, optimizer, evaluator, config, seed, out_dir
                    ).run()
                print(
                    "RESULT task=%s seed=%d method=%s best=%.6f"
                    % (task_name, seed, method, result.best_score),
                    flush=True,
                )


if __name__ == "__main__":
    main()
