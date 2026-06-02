import argparse
import csv
import json
import os
from collections import defaultdict


def load_runs(root: str):
    rows = []
    for dirpath, _, filenames in os.walk(root):
        if "history.jsonl" not in filenames:
            continue
        parts = os.path.relpath(dirpath, root).split(os.sep)
        if len(parts) < 3:
            continue
        task, method, seed_part = parts[:3]
        try:
            seed = int(seed_part.replace("seed_", ""))
        except ValueError:
            seed = -1
        history_path = os.path.join(dirpath, "history.jsonl")
        history = [json.loads(line) for line in open(history_path, encoding="utf-8")]
        if not history:
            continue
        best_row = max(history, key=lambda row: row["score"])
        best = best_row["score"]
        state_path = os.path.join(dirpath, "state.json")
        best_prompt = best_row.get("prompt", "")
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
            best_prompt = state.get("best_prompt", best_prompt)
        rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "n_evals": len(history),
                "best_score": best,
                "best_prompt": best_prompt,
                "run_dir": dirpath,
            }
        )
    return rows


def write_summary(rows, root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task",
                "method",
                "seed",
                "n_evals",
                "best_score",
                "best_prompt",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["method"])].append(row["best_score"])

    md = ["# ReElicit 复现实验汇总", ""]
    suite_config = os.path.join(root, "suite_config.json")
    if os.path.exists(suite_config):
        with open(suite_config, encoding="utf-8") as fh:
            config = json.load(fh)
        md.extend(
            [
                "## 运行配置",
                "",
                "- 任务：`%s`" % config.get("tasks", ""),
                "- 方法：`%s`" % config.get("methods", ""),
                "- Seeds：`%s`" % config.get("seeds", ""),
                "- 样本上限：`%s`" % config.get("limit", "None"),
                "- 预算：q=%s, T=%s, K=%s, M=%s"
                % (
                    config.get("q", "paper"),
                    config.get("T", "paper"),
                    config.get("K", "paper"),
                    config.get("M", "paper"),
                ),
                "- Optimizer：`%s`；Target：`%s`"
                % (config.get("local_model", ""), config.get("local_target_model", "")),
                "",
            ]
        )

    md.append("## 方法汇总")
    md.append("")
    md.append("| Task | Method | Runs | Mean Best | Max Best |")
    md.append("|---|---|---:|---:|---:|")
    for (task, method), scores in sorted(grouped.items()):
        md.append(
            "| %s | %s | %d | %.6f | %.6f |"
            % (task, method, len(scores), sum(scores) / len(scores), max(scores))
        )

    md.extend(["", "## 每个任务的最佳方法", ""])
    md.append("| Task | Best Method | Best Score | Seed | Evaluations |")
    md.append("|---|---|---:|---:|---:|")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    for task, task_rows in sorted(by_task.items()):
        winner = max(task_rows, key=lambda row: row["best_score"])
        md.append(
            "| %s | %s | %.6f | %s | %s |"
            % (
                task,
                winner["method"],
                winner["best_score"],
                winner["seed"],
                winner["n_evals"],
            )
        )

    md.extend(["", "## 最佳 Prompt 摘要", ""])
    for task, task_rows in sorted(by_task.items()):
        winner = max(task_rows, key=lambda row: row["best_score"])
        prompt = " ".join(str(winner["best_prompt"]).split())
        if len(prompt) > 700:
            prompt = prompt[:697] + "..."
        md.append(
            "- `%s` / `%s` / seed `%s` / %.6f：%s"
            % (task, winner["method"], winner["seed"], winner["best_score"], prompt)
        )

    fig_dir = os.path.join(out_dir, "figures")
    if os.path.isdir(fig_dir):
        figures = sorted(
            name for name in os.listdir(fig_dir) if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if figures:
            md.extend(["", "## 图像文件", ""])
            for name in figures:
                md.append("- `%s`" % os.path.join("figures", name))

    md.append("")
    md.append("原始逐 run 数据见 `summary.csv`，各方法的完整轨迹见对应 `history.jsonl`。")
    with open(os.path.join(out_dir, "SUMMARY_REPORT.zh-CN.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/paper_repro")
    parser.add_argument("--out", default="reports/paper_repro")
    args = parser.parse_args()
    rows = load_runs(args.root)
    write_summary(rows, args.root, args.out)
    print("Loaded %d runs" % len(rows))
    print("Wrote", os.path.join(args.out, "summary.csv"))


if __name__ == "__main__":
    main()
