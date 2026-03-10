#!/usr/bin/env python3
"""
Comprehensive comparative analysis between TagFex baseline and ANT+Gap experiments.

Experiments:
- Baseline: exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal (ant_beta=0, no ANT loss)
- ANT+Gap: exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5 (ant_beta=1, gap_target=0.7, gap_beta=0.5)

Analyzes:
1. Loss components evolution (NLL, ANT loss, Gap loss)
2. ANT distance statistics (pos_mean, neg_mean, current_gap)
3. Final evaluation metrics (accuracy, forgetting)
4. Comparative visualizations
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def parse_ant_stats_baseline(log_file):
    """Parse ANT stats from baseline experiment (no gap_loss)."""
    data = []
    # Baseline pattern without gap_loss metrics
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) .* neg_mean: ([-\d.]+) .* gap_mean: ([-\d.]+) .* margin: ([\d.]+) .* violation_pct: ([\d.]+)% \| ant_loss: ([\d.]+) \| gap_loss: ([\d.]+) \| current_gap: ([-\d.]+) \| gap_target: ([\d.]+) \| total_ant_loss: ([\d.]+)"

    print(f"Parsing baseline ANT stats from {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append(
                    {
                        "task": int(match.group(1)),
                        "epoch": int(match.group(2)),
                        "batch": int(match.group(3)),
                        "pos_mean": float(match.group(4)),
                        "neg_mean": float(match.group(5)),
                        "gap_mean": float(match.group(6)),
                        "margin": float(match.group(7)),
                        "violation_pct": float(match.group(8)),
                        "ant_loss": float(match.group(9)),
                        "gap_loss": float(match.group(10)),
                        "current_gap": float(match.group(11)),
                        "gap_target": float(match.group(12)),
                        "total_ant_loss": float(match.group(13)),
                    }
                )

    print(f"  Found {len(data)} ANT stat entries")
    return data


def parse_ant_stats_gap(log_file):
    """Parse ANT stats from ANT+Gap experiment (with gap_loss)."""
    return parse_ant_stats_baseline(log_file)  # Same pattern now


def parse_loss_components_baseline(log_file):
    """Parse loss components from baseline experiment."""
    data = []
    # Baseline pattern - no gap_loss in loss components
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] Loss components: contrast_infoNCE_nll: ([\d.]+) \| contrast_infoNCE_ant_loss: ([\d.]+) \| contrast_infoNCE_nce_weighted: ([\d.]+) \| contrast_infoNCE_ant_weighted: ([\d.]+) \| contrast_infoNCE_total: ([\d.]+)"

    print(f"Parsing baseline loss components from {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append(
                    {
                        "task": int(match.group(1)),
                        "epoch": int(match.group(2)),
                        "batch": int(match.group(3)),
                        "nll": float(match.group(4)),
                        "ant_loss": float(match.group(5)),
                        "gap_loss": 0.0,  # No gap loss in baseline
                        "total_ant_loss": float(match.group(5)),  # Same as ant_loss
                        "nce_weighted": float(match.group(6)),
                        "ant_weighted": float(match.group(7)),
                        "total_loss": float(match.group(8)),
                    }
                )

    print(f"  Found {len(data)} loss component entries")
    return data


def parse_loss_components_gap(log_file):
    """Parse loss components from ANT+Gap experiment."""
    data = []
    # Gap pattern - includes gap_loss
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] Loss components: contrast_infoNCE_nll: ([\d.]+) \| contrast_infoNCE_ant_loss: ([\d.]+) \| contrast_infoNCE_gap_loss: ([\d.]+) \| contrast_infoNCE_total_ant_loss: ([\d.]+) \| contrast_infoNCE_nce_weighted: ([\d.]+) \| contrast_infoNCE_ant_weighted: ([\d.]+) \| contrast_infoNCE_total: ([\d.]+)"

    print(f"Parsing ANT+Gap loss components from {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append(
                    {
                        "task": int(match.group(1)),
                        "epoch": int(match.group(2)),
                        "batch": int(match.group(3)),
                        "nll": float(match.group(4)),
                        "ant_loss": float(match.group(5)),
                        "gap_loss": float(match.group(6)),
                        "total_ant_loss": float(match.group(7)),
                        "nce_weighted": float(match.group(8)),
                        "ant_weighted": float(match.group(9)),
                        "total_loss": float(match.group(10)),
                    }
                )

    print(f"  Found {len(data)} loss component entries")
    return data


def parse_evaluation_metrics(log_file):
    """Parse evaluation metrics from exp_gistlog.log."""
    metrics = []

    print(f"Parsing evaluation metrics from {log_file}...")
    with open(log_file, "r") as f:
        content = f.read()

    # Pattern for task completion logs
    task_pattern = r"R0T\[(\d+)/\d+\]E\[\d+/\d+\] train\n├> eval_acc1 ([\d.]+) eval_acc5 ([\d.]+) eval_nme1 ([\d.]+) eval_nme5 ([\d.]+).*?avg_acc1 ([\d.]+) avg_acc5 ([\d.]+) avg_nme1 ([\d.]+) avg_nme5 ([\d.]+)"

    for match in re.finditer(task_pattern, content, re.DOTALL):
        task = int(match.group(1))
        metrics.append(
            {
                "task": task,
                "eval_acc1": float(match.group(2)),
                "eval_acc5": float(match.group(3)),
                "eval_nme1": float(match.group(4)),
                "eval_nme5": float(match.group(5)),
                "avg_acc1": float(match.group(6)),
                "avg_acc5": float(match.group(7)),
                "avg_nme1": float(match.group(8)),
                "avg_nme5": float(match.group(9)),
            }
        )

    print(f"  Found {len(metrics)} task completion metrics")
    return metrics


def aggregate_by_epoch(data, task):
    """Aggregate data by epoch for a specific task."""
    epoch_data = defaultdict(list)

    for entry in data:
        if entry["task"] == task:
            epoch_data[entry["epoch"]].append(entry)

    # Average across batches for each epoch
    aggregated = []
    for epoch in sorted(epoch_data.keys()):
        entries = epoch_data[epoch]
        avg_entry = {"task": task, "epoch": epoch}

        # Average all numeric fields except task and epoch
        numeric_fields = [
            k for k in entries[0].keys() if k not in ["task", "epoch", "batch"]
        ]
        for field in numeric_fields:
            values = [e[field] for e in entries]
            avg_entry[field] = np.mean(values)

        aggregated.append(avg_entry)

    return aggregated


def plot_loss_evolution(baseline_loss, gap_loss, output_dir):
    """Plot loss component evolution comparison."""
    print("\nGenerating loss evolution plots...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        "Loss Evolution Comparison: Baseline vs ANT+Gap", fontsize=16, fontweight="bold"
    )

    tasks_to_plot = [1, 2, 5, 10]  # First, second, middle, last tasks

    for idx, task in enumerate(tasks_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Aggregate by epoch
        baseline_agg = aggregate_by_epoch(baseline_loss, task)
        gap_agg = aggregate_by_epoch(gap_loss, task)

        if not baseline_agg or not gap_agg:
            continue

        baseline_epochs = [e["epoch"] for e in baseline_agg]
        gap_epochs = [e["epoch"] for e in gap_agg]

        # Plot NLL
        ax.plot(
            baseline_epochs,
            [e["nll"] for e in baseline_agg],
            "b-",
            label="Baseline NLL",
            linewidth=2,
        )
        ax.plot(
            gap_epochs,
            [e["nll"] for e in gap_agg],
            "b--",
            label="ANT+Gap NLL",
            linewidth=2,
        )

        # Plot Total Loss
        ax.plot(
            baseline_epochs,
            [e["total_loss"] for e in baseline_agg],
            "r-",
            label="Baseline Total",
            linewidth=2,
        )
        ax.plot(
            gap_epochs,
            [e["total_loss"] for e in gap_agg],
            "r--",
            label="ANT+Gap Total",
            linewidth=2,
        )

        # Plot ANT contribution (only visible in ANT+Gap)
        ax.plot(
            gap_epochs,
            [e["ant_weighted"] for e in gap_agg],
            "g--",
            label="ANT+Gap ANT Weighted",
            linewidth=2,
            alpha=0.7,
        )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(f"Task {task}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "loss_evolution_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_gap_evolution(baseline_stats, gap_stats, output_dir):
    """Plot gap (pos_mean - neg_mean) evolution comparison."""
    print("\nGenerating gap evolution plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Gap Evolution: Baseline vs ANT+Gap (current_gap = pos_mean - neg_mean)",
        fontsize=16,
        fontweight="bold",
    )

    tasks_to_plot = [1, 2, 5, 10]

    for idx, task in enumerate(tasks_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        baseline_agg = aggregate_by_epoch(baseline_stats, task)
        gap_agg = aggregate_by_epoch(gap_stats, task)

        if not baseline_agg or not gap_agg:
            continue

        baseline_epochs = [e["epoch"] for e in baseline_agg]
        gap_epochs = [e["epoch"] for e in gap_agg]

        # Plot real gap (pos_mean - neg_mean) - current_gap can be 0 if gap_target=0
        baseline_gap_values = [e["pos_mean"] - e["neg_mean"] for e in baseline_agg]
        gap_gap_values = [e["pos_mean"] - e["neg_mean"] for e in gap_agg]

        ax.plot(
            baseline_epochs,
            baseline_gap_values,
            "b-",
            label="Baseline Gap",
            linewidth=2,
        )
        ax.plot(
            gap_epochs,
            gap_gap_values,
            "r--",
            label="ANT+Gap Gap",
            linewidth=2,
        )

        # Add gap_target line for ANT+Gap
        if gap_agg:
            gap_target = gap_agg[0]["gap_target"]
            ax.axhline(
                y=gap_target,
                color="g",
                linestyle=":",
                linewidth=2,
                label=f"Gap Target ({gap_target})",
            )

        # Add margin line
        if baseline_agg:
            margin = baseline_agg[0]["margin"]
            ax.axhline(
                y=margin,
                color="orange",
                linestyle=":",
                linewidth=2,
                label=f"Margin ({margin})",
            )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Gap (pos_mean - neg_mean)", fontsize=10)
        ax.set_title(f"Task {task}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "gap_evolution_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_pos_neg_means(baseline_stats, gap_stats, output_dir):
    """Plot positive and negative means evolution."""
    print("\nGenerating pos/neg means plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Positive & Negative Means Evolution: Baseline vs ANT+Gap",
        fontsize=16,
        fontweight="bold",
    )

    tasks_to_plot = [1, 2, 5, 10]

    for idx, task in enumerate(tasks_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        baseline_agg = aggregate_by_epoch(baseline_stats, task)
        gap_agg = aggregate_by_epoch(gap_stats, task)

        if not baseline_agg or not gap_agg:
            continue

        baseline_epochs = [e["epoch"] for e in baseline_agg]
        gap_epochs = [e["epoch"] for e in gap_agg]

        # Baseline pos/neg means
        ax.plot(
            baseline_epochs,
            [e["pos_mean"] for e in baseline_agg],
            "b-",
            label="Baseline Pos Mean",
            linewidth=2,
        )
        ax.plot(
            baseline_epochs,
            [e["neg_mean"] for e in baseline_agg],
            "b:",
            label="Baseline Neg Mean",
            linewidth=2,
        )

        # ANT+Gap pos/neg means
        ax.plot(
            gap_epochs,
            [e["pos_mean"] for e in gap_agg],
            "r-",
            label="ANT+Gap Pos Mean",
            linewidth=2,
        )
        ax.plot(
            gap_epochs,
            [e["neg_mean"] for e in gap_agg],
            "r:",
            label="ANT+Gap Neg Mean",
            linewidth=2,
        )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        ax.set_title(f"Task {task}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "pos_neg_means_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_accuracy_curves(baseline_metrics, gap_metrics, output_dir):
    """Plot accuracy curves comparison."""
    print("\nGenerating accuracy curves plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Evaluation Metrics Comparison: Baseline vs ANT+Gap",
        fontsize=16,
        fontweight="bold",
    )

    metrics_to_plot = [
        ("avg_acc1", "Average Accuracy (Top-1)"),
        ("eval_acc1", "Current Task Accuracy (Top-1)"),
        ("avg_nme1", "Average NME Accuracy (Top-1)"),
        ("eval_nme1", "Current Task NME Accuracy (Top-1)"),
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        baseline_tasks = [m["task"] for m in baseline_metrics]
        gap_tasks = [m["task"] for m in gap_metrics]

        baseline_values = [m[metric] for m in baseline_metrics]
        gap_values = [m[metric] for m in gap_metrics]

        ax.plot(
            baseline_tasks,
            baseline_values,
            "b-o",
            label="Baseline",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            gap_tasks, gap_values, "r--s", label="ANT+Gap", linewidth=2, markersize=8
        )

        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(baseline_tasks)

    plt.tight_layout()
    output_path = output_dir / "accuracy_curves_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def calculate_forgetting(metrics):
    """Calculate forgetting metric."""
    if len(metrics) < 2:
        return []

    forgetting = []
    for task_idx in range(1, len(metrics) + 1):
        # Find max accuracy for this task across all subsequent evaluations
        max_acc = 0
        final_acc = 0

        for eval_metrics in metrics[task_idx - 1 :]:
            if "eval_acc1_per_task" in eval_metrics:
                continue  # Skip if no per-task data
            # This is simplified - would need per_task arrays from logs

        forgetting.append(0)  # Placeholder

    return forgetting


def generate_summary_report(
    baseline_metrics, gap_metrics, baseline_stats, gap_stats, output_dir
):
    """Generate comprehensive summary report."""
    print("\nGenerating summary report...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COMPARATIVE ANALYSIS: BASELINE vs ANT+GAP")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Experiment configurations
    report_lines.append("## Experiment Configurations")
    report_lines.append("")
    report_lines.append(
        "**Baseline (exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal)**"
    )
    report_lines.append("  - ant_beta: 0.0 (ANT loss disabled)")
    report_lines.append("  - nce_alpha: 1.0")
    report_lines.append("  - ant_margin: 0.1")
    report_lines.append("  - Pure InfoNCE training")
    report_lines.append("")
    report_lines.append(
        "**ANT+Gap (exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5)**"
    )
    report_lines.append("  - ant_beta: 1.0 (ANT loss enabled)")
    report_lines.append("  - gap_target: 0.7")
    report_lines.append("  - gap_beta: 0.5")
    report_lines.append("  - nce_alpha: 1.0")
    report_lines.append("  - ant_margin: 0.1")
    report_lines.append("  - Loss: NLL + ANT_loss + 0.5 * Gap_loss")
    report_lines.append("")

    # Final accuracy comparison
    report_lines.append("## Final Evaluation Metrics (Task 10)")
    report_lines.append("")

    baseline_final = baseline_metrics[-1]
    gap_final = gap_metrics[-1]

    report_lines.append("| Metric           | Baseline | ANT+Gap | Difference |")
    report_lines.append("|------------------|----------|---------|------------|")

    metrics_compare = [
        ("avg_acc1", "Avg Acc@1"),
        ("avg_acc5", "Avg Acc@5"),
        ("avg_nme1", "Avg NME@1"),
        ("avg_nme5", "Avg NME@5"),
        ("eval_acc1", "Task 10 Acc@1"),
        ("eval_acc5", "Task 10 Acc@5"),
    ]

    for metric, label in metrics_compare:
        baseline_val = baseline_final.get(metric, 0)
        gap_val = gap_final.get(metric, 0)
        diff = gap_val - baseline_val
        sign = "+" if diff >= 0 else ""
        report_lines.append(
            f"| {label:<16} | {baseline_val:>8.2f} | {gap_val:>7.2f} | {sign}{diff:>+9.2f} |"
        )

    report_lines.append("")

    # Gap statistics
    report_lines.append("## Gap Statistics Summary")
    report_lines.append("")

    # Final task gap stats
    baseline_final_stats = [s for s in baseline_stats if s["task"] == 10]
    gap_final_stats = [s for s in gap_stats if s["task"] == 10]

    if baseline_final_stats and gap_final_stats:
        # Calculate real gap: pos_mean - neg_mean (current_gap can be 0 if gap_target=0)
        baseline_final_gap = np.mean(
            [s["pos_mean"] - s["neg_mean"] for s in baseline_final_stats]
        )
        gap_final_gap = np.mean(
            [s["pos_mean"] - s["neg_mean"] for s in gap_final_stats]
        )
        gap_target = gap_final_stats[0]["gap_target"]

        report_lines.append(
            f"**Final Task (Task 10) Average Gap (pos_mean - neg_mean):**"
        )
        report_lines.append(f"  - Baseline:      {baseline_final_gap:.4f}")
        report_lines.append(f"  - ANT+Gap:       {gap_final_gap:.4f}")
        report_lines.append(f"  - Gap Target:    {gap_target:.4f}")
        report_lines.append(
            f"  - Difference:    {gap_final_gap - baseline_final_gap:+.4f}"
        )
        if gap_target > 0:
            report_lines.append(
                f"  - Baseline reach: {baseline_final_gap / gap_target * 100:.1f}% of target"
            )
            report_lines.append(
                f"  - ANT+Gap reach:  {gap_final_gap / gap_target * 100:.1f}% of target"
            )
        report_lines.append("")

    # Loss statistics
    report_lines.append("## Loss Statistics (Task 10, Final Epoch)")
    report_lines.append("")

    baseline_final_epoch = [
        s for s in baseline_stats if s["task"] == 10 and s["epoch"] == 170
    ]
    gap_final_epoch = [s for s in gap_stats if s["task"] == 10 and s["epoch"] == 170]

    if baseline_final_epoch and gap_final_epoch:
        baseline_ant = np.mean([s["ant_loss"] for s in baseline_final_epoch])
        gap_ant = np.mean([s["ant_loss"] for s in gap_final_epoch])
        gap_gap_loss = np.mean([s["gap_loss"] for s in gap_final_epoch])
        gap_total_ant = np.mean([s["total_ant_loss"] for s in gap_final_epoch])

        report_lines.append(f"**ANT Loss Components:**")
        report_lines.append(f"  - Baseline ANT Loss:        {baseline_ant:.4f}")
        report_lines.append(f"  - ANT+Gap ANT Loss:         {gap_ant:.4f}")
        report_lines.append(f"  - ANT+Gap Gap Loss:         {gap_gap_loss:.4f}")
        report_lines.append(f"  - ANT+Gap Total ANT Loss:   {gap_total_ant:.4f}")
        report_lines.append("")

    # Task progression
    report_lines.append("## Accuracy Progression Across Tasks")
    report_lines.append("")
    report_lines.append(
        "| Task | Baseline Avg Acc@1 | ANT+Gap Avg Acc@1 | Difference |"
    )
    report_lines.append("|------|-------------------|------------------|------------|")

    for baseline_m, gap_m in zip(baseline_metrics, gap_metrics):
        task = baseline_m["task"]
        baseline_acc = baseline_m["avg_acc1"]
        gap_acc = gap_m["avg_acc1"]
        diff = gap_acc - baseline_acc
        sign = "+" if diff >= 0 else ""
        report_lines.append(
            f"| {task:>4} | {baseline_acc:>17.2f} | {gap_acc:>16.2f} | {sign}{diff:>+9.2f} |"
        )

    report_lines.append("")

    # Conclusions
    report_lines.append("## Key Observations")
    report_lines.append("")

    final_diff = gap_final["avg_acc1"] - baseline_final["avg_acc1"]
    if abs(final_diff) < 0.5:
        conclusion = "minimal difference"
    elif final_diff > 0:
        conclusion = f"slight improvement (+{final_diff:.2f}%)"
    else:
        conclusion = f"slight degradation ({final_diff:.2f}%)"

    report_lines.append(
        f"1. **Final Performance**: ANT+Gap shows {conclusion} compared to baseline"
    )
    report_lines.append("")

    if baseline_final_stats and gap_final_stats:
        if gap_final_gap > baseline_final_gap:
            report_lines.append(
                f"2. **Gap Maximization**: Successfully increased gap from {baseline_final_gap:.4f} to {gap_final_gap:.4f}"
            )
        else:
            report_lines.append(
                f"2. **Gap Maximization**: Gap slightly decreased from {baseline_final_gap:.4f} to {gap_final_gap:.4f}"
            )
        report_lines.append("")

    report_lines.append(
        "3. **Loss Dynamics**: Check generated plots for detailed loss evolution patterns"
    )
    report_lines.append("")
    report_lines.append(
        "4. **Training Stability**: Compare loss curves for convergence behavior"
    )
    report_lines.append("")

    report_lines.append("=" * 80)

    # Write report
    report_path = output_dir / "comparison_summary.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"  Saved: {report_path}")

    # Also print to console
    print("\n" + "\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Compare TagFex baseline and ANT+Gap experiments"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline experiment logs directory",
    )
    parser.add_argument(
        "--ant-gap",
        type=str,
        required=True,
        help="Path to ANT+Gap experiment logs directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/experiments_comparison",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    gap_dir = Path(args.ant_gap)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPARATIVE ANALYSIS: BASELINE vs ANT+GAP")
    print("=" * 80)
    print(f"Baseline:  {baseline_dir}")
    print(f"ANT+Gap:   {gap_dir}")
    print(f"Output:    {output_dir}")
    print("=" * 80)

    # Parse baseline experiment
    print("\n[1/2] Parsing BASELINE experiment...")
    baseline_debug_log = baseline_dir / "exp_debug0.log"
    baseline_gist_log = baseline_dir / "exp_gistlog.log"

    baseline_stats = parse_ant_stats_baseline(baseline_debug_log)
    baseline_loss = parse_loss_components_baseline(baseline_debug_log)
    baseline_metrics = parse_evaluation_metrics(baseline_gist_log)

    # Parse ANT+Gap experiment
    print("\n[2/2] Parsing ANT+GAP experiment...")
    gap_debug_log = gap_dir / "exp_debug0.log"
    gap_gist_log = gap_dir / "exp_gistlog.log"

    gap_stats = parse_ant_stats_gap(gap_debug_log)
    gap_loss = parse_loss_components_gap(gap_debug_log)
    gap_metrics = parse_evaluation_metrics(gap_gist_log)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_loss_evolution(baseline_loss, gap_loss, output_dir)
    plot_gap_evolution(baseline_stats, gap_stats, output_dir)
    plot_pos_neg_means(baseline_stats, gap_stats, output_dir)
    plot_accuracy_curves(baseline_metrics, gap_metrics, output_dir)

    # Generate summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    generate_summary_report(
        baseline_metrics, gap_metrics, baseline_stats, gap_stats, output_dir
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - comparison_summary.txt")
    print("  - loss_evolution_comparison.png")
    print("  - gap_evolution_comparison.png")
    print("  - pos_neg_means_comparison.png")
    print("  - accuracy_curves_comparison.png")
    print("")


if __name__ == "__main__":
    main()
