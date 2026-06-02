import argparse
import json
import os

from demo import build_config, load_json
from reelicit.algorithm import ReElicitRunner
from reelicit.baselines import BaselineRunner
from reelicit.evaluator import MockEvaluator, OpenAIChatBenchmarkEvaluator
from reelicit.llm import LocalTransformersLLMClient, MockLLMClient
from reelicit.tasks import get_task


METHODS = {
    "reelicit",
    "ape",
    "opro",
    "promptbreeder",
    "textgrad",
    "no_refinement",
    "no_bo",
    "static_features",
    "independent_extraction",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReElicit experiment variants")
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
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
    args = parser.parse_args()

    raw = load_json(args.config)
    config = build_config(raw, strict_paper=False)
    for name in ("q", "T", "K", "M"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    config.include_format_in_context = args.include_format_in_context

    method = args.method
    if method == "no_refinement":
        config.no_refinement = True
        runner_method = "ReElicit-no-refinement"
    elif method == "no_bo":
        config.no_bo = True
        runner_method = "ReElicit-no-bo"
    elif method == "static_features":
        config.static_features = True
        runner_method = "ReElicit-static-features"
    elif method == "independent_extraction":
        config.independent_extraction = True
        runner_method = "ReElicit-independent-extraction"
    else:
        runner_method = method

    task = get_task(args.task)
    out_dir = os.path.join(args.output_root, args.task, method, "seed_%d" % args.seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, ensure_ascii=False)

    if args.mode == "mock":
        optimizer = MockLLMClient()
        evaluator = MockEvaluator(task, args.seed)
    else:
        optimizer = LocalTransformersLLMClient(args.local_model, device=args.optimizer_device)
        target = LocalTransformersLLMClient(args.local_target_model, device=args.target_device)
        evaluator = OpenAIChatBenchmarkEvaluator(
            task, target, seed=args.seed, limit=args.limit, target_temperature=config.target_temperature
        )

    if method == "reelicit" or method in {
        "no_refinement",
        "no_bo",
        "static_features",
        "independent_extraction",
    }:
        result = ReElicitRunner(task, optimizer, evaluator, config, args.seed, out_dir).run()
    else:
        result = BaselineRunner(
            runner_method, task, optimizer, evaluator, config, args.seed, out_dir
        ).run()

    print("RESULT method=%s task=%s seed=%d best=%.6f" % (method, args.task, args.seed, result.best_score))


if __name__ == "__main__":
    main()
