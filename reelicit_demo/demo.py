import argparse
import json
import os
from typing import Any, Dict

from reelicit.algorithm import ReElicitRunner
from reelicit.evaluator import MockEvaluator, OpenAIChatBenchmarkEvaluator
from reelicit.llm import (
    LocalTransformersLLMClient,
    MockLLMClient,
    OpenAICompatibleLLMClient,
)
from reelicit.tasks import get_task
from reelicit.types import RunConfig


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_config(raw: Dict[str, Any], strict_paper: bool) -> RunConfig:
    h = raw.get("hyperparameters", {})
    models = raw.get("models", {})
    bo = raw.get("bo", {})
    return RunConfig(
        q=int(h.get("q", 5)),
        T=int(h.get("T", 6)),
        K=int(h.get("K", 5)),
        M=int(h.get("M", 10)),
        tau=float(h.get("tau", 0.1)),
        b=int(h.get("b", 10)),
        nmax=int(h.get("nmax", 12)),
        pmax=int(h.get("Pmax", 20)),
        optimizer_temperature=float(models.get("optimizer_temperature", 0.7)),
        target_temperature=float(models.get("target_temperature", 0.0)),
        bo_num_restarts=int(bo.get("num_restarts", 20)),
        bo_raw_samples=int(bo.get("raw_samples", 512)),
        strict_paper=strict_paper,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ReElicit paper reproduction demo")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "paper.json"),
        help="Path to JSON config with paper settings.",
    )
    parser.add_argument("--task", default=None, help="Task key, e.g. snarks or gsm8k.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["mock", "benchmark", "local-mock", "local-benchmark"],
        default="mock",
        help="mock runs locally; benchmark uses OpenAI/vLLM endpoints; local-mock uses a local optimizer and mock scores; local-benchmark uses a local model for optimizer and target.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Alias for --mode mock, kept for the quick-start command.",
    )
    parser.add_argument(
        "--strict-paper",
        action="store_true",
        help="Require BoTorch SingleTaskGP + qLogNEI rather than the smoke-test fallback.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Benchmark example cap. Omit to use the paper task size.",
    )
    parser.add_argument("--output", default=None, help="Output directory for history/state.")
    parser.add_argument("--local-model", default="/nas1/zyj/models/Qwen3-8B")
    parser.add_argument("--local-target-model", default=None)
    parser.add_argument("--optimizer-device", default="cuda")
    parser.add_argument("--target-device", default=None)
    parser.add_argument(
        "--include-format-in-context",
        action="store_true",
        help="Append the task's evaluation output-format instruction to optimizer-side task context.",
    )
    parser.add_argument("--q", type=int, default=None, help="Override batch size.")
    parser.add_argument("--T", type=int, default=None, help="Override total evaluated batches.")
    parser.add_argument("--K", type=int, default=None, help="Override elicitation rounds.")
    parser.add_argument("--M", type=int, default=None, help="Override realization budget.")
    args = parser.parse_args()

    raw = load_json(args.config)
    if args.mock:
        args.mode = "mock"
    task_key = args.task or raw.get("default_task", "snarks")
    task = get_task(task_key)
    config = build_config(raw, strict_paper=args.strict_paper)
    if args.q is not None:
        config.q = args.q
    if args.T is not None:
        config.T = args.T
    if args.K is not None:
        config.K = args.K
    if args.M is not None:
        config.M = args.M
    if args.include_format_in_context:
        config.include_format_in_context = True
    output_dir = args.output or os.path.join("runs", "%s_seed%d_%s" % (task_key, args.seed, args.mode))

    if args.mode == "mock":
        optimizer_llm = MockLLMClient()
        evaluator = MockEvaluator(task, seed=args.seed)
    elif args.mode == "local-mock":
        optimizer_llm = LocalTransformersLLMClient(
            args.local_model, device=args.optimizer_device
        )
        evaluator = MockEvaluator(task, seed=args.seed)
    elif args.mode == "local-benchmark":
        optimizer_llm = LocalTransformersLLMClient(
            args.local_model, device=args.optimizer_device
        )
        target_model = args.local_target_model or args.local_model
        if target_model == args.local_model:
            target_llm = optimizer_llm
        else:
            target_llm = LocalTransformersLLMClient(
                target_model, device=args.target_device or args.optimizer_device
            )
        evaluator = OpenAIChatBenchmarkEvaluator(
            task=task,
            target_llm=target_llm,
            seed=args.seed,
            limit=args.limit,
            target_temperature=config.target_temperature,
        )
    else:
        models = raw.get("models", {})
        optimizer_llm = OpenAICompatibleLLMClient(
            model=os.environ.get("OPTIMIZER_MODEL", models["optimizer_model"]),
            base_url=os.environ.get("OPTIMIZER_BASE_URL"),
            api_key=os.environ.get("OPTIMIZER_API_KEY", "EMPTY"),
        )
        target_llm = OpenAICompatibleLLMClient(
            model=os.environ.get("TARGET_MODEL", models["target_model"]),
            base_url=os.environ.get("TARGET_BASE_URL"),
            api_key=os.environ.get("TARGET_API_KEY", "EMPTY"),
        )
        evaluator = OpenAIChatBenchmarkEvaluator(
            task=task,
            target_llm=target_llm,
            seed=args.seed,
            limit=args.limit,
            target_temperature=config.target_temperature,
        )

    print("Task:", task.name)
    print(
        "Paper budget/settings: N=%d q=%d T=%d K=%d M=%d tau=%.3f b=%d nmax=%d"
        % (config.N, config.q, config.T, config.K, config.M, config.tau, config.b, config.nmax)
    )
    runner = ReElicitRunner(
        task=task,
        optimizer_llm=optimizer_llm,
        evaluator=evaluator,
        config=config,
        seed=args.seed,
        output_dir=output_dir,
    )
    result = runner.run()
    print("\nBest score: %.6f" % result.best_score)
    print("Best prompt:\n%s" % result.best_prompt)
    print("Artifacts written to:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
