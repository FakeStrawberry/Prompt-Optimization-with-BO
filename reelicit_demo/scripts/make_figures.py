import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "reelicit",
    "ape",
    "opro",
    "promptbreeder",
    "textgrad",
    "no_refinement",
    "no_bo",
    "static_features",
    "independent_extraction",
]

METHOD_COLORS = {
    "reelicit": "#0072B2",
    "ape": "#E69F00",
    "opro": "#009E73",
    "promptbreeder": "#D55E00",
    "textgrad": "#CC79A7",
    "no_refinement": "#56B4E9",
    "no_bo": "#F0E442",
    "static_features": "#999999",
    "independent_extraction": "#332288",
}


def method_sort_key(method: str):
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def load_histories(root: str):
    histories = {}
    for dirpath, _, filenames in os.walk(root):
        if "history.jsonl" not in filenames:
            continue
        parts = os.path.relpath(dirpath, root).split(os.sep)
        if len(parts) < 3:
            continue
        task, method, seed_part = parts[:3]
        seed = seed_part.replace("seed_", "")
        rows = [json.loads(line) for line in open(os.path.join(dirpath, "history.jsonl"), encoding="utf-8")]
        histories[(task, method, seed)] = rows
    return histories


def convergence(rows):
    best = None
    xs = []
    ys = []
    for i, row in enumerate(rows, start=1):
        score = row["score"]
        best = score if best is None else max(best, score)
        xs.append(i)
        ys.append(best)
    return xs, ys


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#444444")


def aggregate_series(histories):
    by_method = defaultdict(dict)
    for (task, method, seed), rows in histories.items():
        _, ys = convergence(rows)
        by_method[method][task] = ys

    out = {}
    for method, task_series in by_method.items():
        max_len = max(len(series) for series in task_series.values())
        ys = []
        for i in range(max_len):
            vals = [
                series[i] if i < len(series) else series[-1]
                for series in task_series.values()
                if series
            ]
            ys.append(sum(vals) / len(vals))
        out[method] = ys
    return out


def write_aggregate_csv(series_by_method, out_dir: str) -> None:
    report_dir = os.path.dirname(out_dir)
    path = os.path.join(report_dir, "aggregate_convergence_mean_across_tasks.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("method,evaluation,mean_best_so_far\n")
        for method in sorted(series_by_method, key=method_sort_key):
            for i, value in enumerate(series_by_method[method], start=1):
                fh.write("%s,%d,%.8f\n" % (method, i, value))


def write_method_mean_csv(best_by_method, out_dir: str) -> None:
    report_dir = os.path.dirname(out_dir)
    path = os.path.join(report_dir, "method_mean_across_tasks.csv")
    rows = sorted(
        (
            (method, sum(scores) / len(scores), len(scores))
            for method, scores in best_by_method.items()
        ),
        key=lambda row: row[1],
        reverse=True,
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("method,mean_best_score,num_tasks\n")
        for method, mean_score, n in rows:
            fh.write("%s,%.8f,%d\n" % (method, mean_score, n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/paper_repro")
    parser.add_argument("--out", default="reports/paper_repro/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    histories = load_histories(args.root)
    by_task = defaultdict(list)
    for key, rows in histories.items():
        task, method, seed = key
        by_task[task].append((method, seed, rows))

    for task, items in sorted(by_task.items()):
        plt.figure(figsize=(9, 5.5))
        for method, seed, rows in sorted(items, key=lambda item: method_sort_key(item[0])):
            x, y = convergence(rows)
            label = method if len({item[1] for item in items}) == 1 else "%s seed=%s" % (method, seed)
            linewidth = 2.8 if method == "reelicit" else 1.8
            zorder = 10 if method == "reelicit" else 2
            plt.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=linewidth,
                label=label,
                color=method_color(method),
                zorder=zorder,
            )
        plt.title("Best-so-far convergence: %s" % task)
        plt.xlabel("# prompt evaluations")
        plt.ylabel("Best score")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        path = os.path.join(args.out, "%s_convergence.png" % task)
        plt.savefig(path, dpi=220)
        plt.close()

    # Aggregate bar plot.
    best_by_method = defaultdict(list)
    for (task, method, seed), rows in histories.items():
        best_by_method[method].append(max(row["score"] for row in rows))
    if best_by_method:
        methods = sorted(best_by_method, key=lambda m: sum(best_by_method[m]) / len(best_by_method[m]), reverse=True)
        values = [sum(best_by_method[m]) / len(best_by_method[m]) for m in methods]
        colors = [method_color(m) for m in methods]

        write_method_mean_csv(best_by_method, args.out)

        plt.figure(figsize=(max(9, len(methods) * 1.15), 5.8))
        bars = plt.bar(methods, values, color=colors, edgecolor="#263238", linewidth=0.6)
        plt.ylabel("Mean best score across 10 tasks")
        plt.title("Method Comparison: Mean Best Score Across All Tasks")
        plt.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=35, ha="right")
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                "%.3f" % value,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "aggregate_method_comparison.png"), dpi=180)
        plt.savefig(os.path.join(args.out, "method_mean_across_tasks.png"), dpi=220)
        plt.close()

    series_by_method = aggregate_series(histories)
    if series_by_method:
        write_aggregate_csv(series_by_method, args.out)
        plt.figure(figsize=(10.5, 6.2))
        for method in sorted(series_by_method, key=method_sort_key):
            ys = series_by_method[method]
            xs = list(range(1, len(ys) + 1))
            linewidth = 3.0 if method == "reelicit" else 1.9
            zorder = 10 if method == "reelicit" else 2
            plt.plot(
                xs,
                ys,
                marker="o",
                markersize=3.8,
                linewidth=linewidth,
                label=method,
                color=method_color(method),
                zorder=zorder,
            )
        plt.xlabel("# prompt evaluations per method")
        plt.ylabel("Mean best-so-far score across 10 tasks")
        plt.title("Aggregate Convergence Across All Tasks")
        plt.grid(True, alpha=0.25)
        plt.xticks(range(1, 31, 2))
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "aggregate_convergence_mean_across_tasks.png"), dpi=220)
        plt.close()
    print("Wrote figures to", args.out)


if __name__ == "__main__":
    main()
