#!/usr/bin/env python3
"""
Comparative analysis between Baseline (no ANT) and ANT+Gap experiments.
Compares gap evolution, loss components, and final performance metrics.
Integrates exp_matrix_debug0.log (loss components) with exp_gistlog.log (evaluation metrics).
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_ant_stats(log_file):
    """Parse ANT distance stats from exp_matrix_debug0.log."""
    data = []
    # Pattern for logs with gap_loss metrics
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) .* neg_mean: ([-\d.]+) .* gap_mean: ([-\d.]+) .* margin: ([\d.]+) .* violation_pct: ([\d.]+)% \| ant_loss: ([\d.]+) \| gap_loss: ([\d.]+) \| current_gap: ([-\d.]+) \| gap_target: ([\d.]+) \| total_ant_loss: ([\d.]+)"

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

    return data


def parse_loss_components(log_file):
    """Parse loss components from exp_matrix_debug0.log."""
    data = []
    # Pattern for loss components
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] Loss components: contrast_infoNCE_nll: ([\d.]+) \| contrast_infoNCE_ant_loss: ([\d.]+) \| contrast_infoNCE_gap_loss: ([\d.]+) \| contrast_infoNCE_total_ant_loss: ([\d.]+) \| contrast_infoNCE_nce_weighted: ([\d.]+) \| contrast_infoNCE_ant_weighted: ([\d.]+) \| contrast_infoNCE_total: ([\d.]+)"

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

    return data


def parse_evaluation_metrics(log_file):
    """Parse evaluation metrics from exp_gistlog.log."""
    metrics = []

    with open(log_file, "r") as f:
        content = f.read()

    # Pattern for task completion logs
    task_pattern = r"R0T\[(\d+)/\d+\]E\[\d+/\d+\] train\n├> eval_acc1 ([\d.]+) eval_acc5 ([\d.]+) eval_nme1 ([\d.]+) eval_nme5 ([\d.]+).*?avg_acc1 ([\d.]+) avg_acc5 ([\d.]+) avg_nme1 ([\d.]+) avg_nme5 ([\d.]+)"

    for match in re.finditer(task_pattern, content):
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

    # Extract curves for final task
    curve_pattern = r"acc1_curve \[([\d., ]+)\].*?nme1_curve \[([\d., ]+)\]"
    final_match = re.search(curve_pattern, content, re.DOTALL)

    curves = {}
    if final_match:
        acc1_curve = [float(x) for x in final_match.group(1).split()]
        nme1_curve = [float(x) for x in final_match.group(2).split()]
        curves = {
            "acc1_curve": acc1_curve,
            "nme1_curve": nme1_curve,
        }

    return metrics, curves


def aggregate_by_epoch(data):
    """Aggregate metrics by task and epoch."""
    results = {}

    for d in data:
        key = (d["task"], d["epoch"])
        if key not in results:
            results[key] = []
        results[key].append(d)

    # Calculate means
    aggregated = []
    for (task, epoch), batch_data in sorted(results.items()):
        aggregated.append(
            {
                "task": task,
                "epoch": epoch,
                "current_gap": np.mean([b["current_gap"] for b in batch_data]),
                "gap_loss": np.mean([b["gap_loss"] for b in batch_data]),
                "ant_loss": np.mean([b["ant_loss"] for b in batch_data]),
                "total_ant_loss": np.mean([b["total_ant_loss"] for b in batch_data]),
                "violation_pct": np.mean([b["violation_pct"] for b in batch_data]),
                "pos_mean": np.mean([b["pos_mean"] for b in batch_data]),
                "neg_mean": np.mean([b["neg_mean"] for b in batch_data]),
            }
        )

    return aggregated


def aggregate_loss_by_epoch(data):
    """Aggregate loss components by task and epoch."""
    results = {}

    for d in data:
        key = (d["task"], d["epoch"])
        if key not in results:
            results[key] = []
        results[key].append(d)

    # Calculate means
    aggregated = []
    for (task, epoch), batch_data in sorted(results.items()):
        aggregated.append(
            {
                "task": task,
                "epoch": epoch,
                "nll": np.mean([b["nll"] for b in batch_data]),
                "ant_loss": np.mean([b["ant_loss"] for b in batch_data]),
                "gap_loss": np.mean([b["gap_loss"] for b in batch_data]),
                "total_ant_loss": np.mean([b["total_ant_loss"] for b in batch_data]),
                "total_loss": np.mean([b["total_loss"] for b in batch_data]),
            }
        )

    return aggregated


def compare_experiments(
    baseline_matrix_log,
    ant_gap_matrix_log,
    baseline_gist_log,
    ant_gap_gist_log,
    output_dir,
):
    """Compare baseline and ANT+Gap experiments."""

    print("=" * 80)
    print("COMPARATIVE ANALYSIS: Baseline vs ANT+Gap")
    print("=" * 80)

    # Parse matrix debug logs (loss components)
    print("\n📊 Parsing matrix debug logs (loss components)...")
    baseline_data = parse_ant_stats(baseline_matrix_log)
    ant_gap_data = parse_ant_stats(ant_gap_matrix_log)

    baseline_loss = parse_loss_components(baseline_matrix_log)
    ant_gap_loss = parse_loss_components(ant_gap_matrix_log)

    if not baseline_data or not ant_gap_data:
        print("❌ Error: Could not parse matrix debug log files.")
        return

    print(f"  Baseline: {len(baseline_data)} batches")
    print(f"  ANT+Gap:  {len(ant_gap_data)} batches")

    # Parse gist logs (evaluation metrics)
    print("\n🎯 Parsing gist logs (evaluation metrics)...")
    baseline_metrics, baseline_curves = parse_evaluation_metrics(baseline_gist_log)
    ant_gap_metrics, ant_gap_curves = parse_evaluation_metrics(ant_gap_gist_log)

    if baseline_metrics:
        print(f"  Baseline: {len(baseline_metrics)} tasks completed")
    else:
        print("  ⚠️  Baseline: No evaluation metrics found (experiment incomplete?)")

    if ant_gap_metrics:
        print(f"  ANT+Gap:  {len(ant_gap_metrics)} tasks completed")
    else:
        print("  ⚠️  ANT+Gap: No evaluation metrics found (experiment incomplete?)")

    # Aggregate by epoch
    baseline_epochs = aggregate_by_epoch(baseline_data)
    ant_gap_epochs = aggregate_by_epoch(ant_gap_data)

    baseline_loss_epochs = aggregate_loss_by_epoch(baseline_loss)
    ant_gap_loss_epochs = aggregate_loss_by_epoch(ant_gap_loss)

    # Create visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # PLOT 1: Gap and Loss Components Evolution
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Row 1, Col 1: Gap comparison
    ax = axes[0, 0]
    baseline_gaps = [e["current_gap"] for e in baseline_epochs]
    ant_gap_gaps = [e["current_gap"] for e in ant_gap_epochs]

    ax.plot(
        range(len(baseline_gaps)), baseline_gaps, "b-", label="Baseline", linewidth=2
    )
    ax.plot(range(len(ant_gap_gaps)), ant_gap_gaps, "r-", label="ANT+Gap", linewidth=2)
    ax.axhline(y=0.7, color="g", linestyle="--", label="Target (0.7)", alpha=0.5)
    ax.set_xlabel("Epoch (across all tasks)", fontsize=10)
    ax.set_ylabel("Gap (pos_mean - neg_mean)", fontsize=10)
    ax.set_title("Gap Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 2: ANT Loss comparison
    ax = axes[0, 1]
    baseline_ant = [e["ant_loss"] for e in baseline_loss_epochs]
    ant_gap_ant = [e["ant_loss"] for e in ant_gap_loss_epochs]

    ax.plot(range(len(baseline_ant)), baseline_ant, "b-", label="Baseline", linewidth=2)
    ax.plot(range(len(ant_gap_ant)), ant_gap_ant, "r-", label="ANT+Gap", linewidth=2)
    ax.set_xlabel("Epoch (across all tasks)", fontsize=10)
    ax.set_ylabel("ANT Loss (base)", fontsize=10)
    ax.set_title("ANT Loss Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 3: Gap Loss (only for ANT+Gap)
    ax = axes[0, 2]
    ant_gap_gap_loss = [e["gap_loss"] for e in ant_gap_loss_epochs]

    ax.plot(range(len(ant_gap_gap_loss)), ant_gap_gap_loss, "r-", linewidth=2)
    ax.set_xlabel("Epoch (across all tasks)", fontsize=10)
    ax.set_ylabel("Gap Loss", fontsize=10)
    ax.set_title("Gap Maximization Loss (ANT+Gap)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Row 2, Col 1: Total Loss comparison
    ax = axes[1, 0]
    baseline_total = [e["total_loss"] for e in baseline_loss_epochs]
    ant_gap_total = [e["total_loss"] for e in ant_gap_loss_epochs]

    ax.plot(
        range(len(baseline_total)), baseline_total, "b-", label="Baseline", linewidth=2
    )
    ax.plot(
        range(len(ant_gap_total)), ant_gap_total, "r-", label="ANT+Gap", linewidth=2
    )
    ax.set_xlabel("Epoch (across all tasks)", fontsize=10)
    ax.set_ylabel("Total Loss (NCE + ANT)", fontsize=10)
    ax.set_title("Total Loss Evolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2, Col 2: Loss Components Breakdown (ANT+Gap final epoch)
    ax = axes[1, 1]
    if ant_gap_loss_epochs:
        final_epoch = ant_gap_loss_epochs[-1]
        components = ["NLL\n(InfoNCE)", "ANT\nBase", "Gap\nLoss", "Total\nANT"]
        values = [
            final_epoch["nll"],
            final_epoch["ant_loss"],
            final_epoch["gap_loss"],
            final_epoch["total_ant_loss"],
        ]
        colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

        bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel("Loss Value", fontsize=10)
        ax.set_title(
            f"Loss Components (ANT+Gap, Final Epoch)", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Row 2, Col 3: Violation percentage
    ax = axes[1, 2]
    baseline_viol = [e["violation_pct"] for e in baseline_epochs]
    ant_gap_viol = [e["violation_pct"] for e in ant_gap_epochs]

    ax.plot(
        range(len(baseline_viol)), baseline_viol, "b-", label="Baseline", linewidth=2
    )
    ax.plot(range(len(ant_gap_viol)), ant_gap_viol, "r-", label="ANT+Gap", linewidth=2)
    ax.set_xlabel("Epoch (across all tasks)", fontsize=10)
    ax.set_ylabel("Hard Negative Violation %", fontsize=10)
    ax.set_title("Hard Negatives Violation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_losses.png", dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {output_dir / 'comparison_losses.png'}")

    # ========================================================================
    # PLOT 2: Performance Metrics (if available)
    # ========================================================================
    if baseline_metrics and ant_gap_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Row 1, Col 1: NME1 evolution (most important metric)
        ax = axes[0, 0]
        baseline_tasks = [m["task"] for m in baseline_metrics]
        baseline_nme1 = [m["eval_nme1"] for m in baseline_metrics]
        ant_gap_tasks = [m["task"] for m in ant_gap_metrics]
        ant_gap_nme1 = [m["eval_nme1"] for m in ant_gap_metrics]

        ax.plot(
            baseline_tasks,
            baseline_nme1,
            "b-o",
            label="Baseline",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            ant_gap_tasks,
            ant_gap_nme1,
            "r-s",
            label="ANT+Gap",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("eval_nme1 (Top-1 NME Accuracy)", fontsize=11)
        ax.set_title("🥇 Task Performance (NME-based)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 1, Col 2: Average NME1 evolution (forgetting metric)
        ax = axes[0, 1]
        baseline_avg_nme1 = [m["avg_nme1"] for m in baseline_metrics]
        ant_gap_avg_nme1 = [m["avg_nme1"] for m in ant_gap_metrics]

        ax.plot(
            baseline_tasks,
            baseline_avg_nme1,
            "b-o",
            label="Baseline",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            ant_gap_tasks,
            ant_gap_avg_nme1,
            "r-s",
            label="ANT+Gap",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("avg_nme1 (Cumulative Average)", fontsize=11)
        ax.set_title("🏆 Forgetting Metric (avg_nme1)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 2, Col 1: ACC1 evolution (linear classifier)
        ax = axes[1, 0]
        baseline_acc1 = [m["eval_acc1"] for m in baseline_metrics]
        ant_gap_acc1 = [m["eval_acc1"] for m in ant_gap_metrics]

        ax.plot(
            baseline_tasks,
            baseline_acc1,
            "b-o",
            label="Baseline",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            ant_gap_tasks,
            ant_gap_acc1,
            "r-s",
            label="ANT+Gap",
            linewidth=2,
            markersize=8,
        )
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("eval_acc1 (Linear Classifier)", fontsize=11)
        ax.set_title(
            "Task Performance (Linear Classifier)", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Row 2, Col 2: Forgetting comparison bar chart (final)
        ax = axes[1, 1]
        if baseline_metrics and ant_gap_metrics:
            metrics_to_compare = ["avg_nme1", "avg_acc1", "eval_nme1", "eval_acc1"]
            baseline_final = baseline_metrics[-1]
            ant_gap_final = ant_gap_metrics[-1]

            baseline_vals = [baseline_final[m] for m in metrics_to_compare]
            ant_gap_vals = [ant_gap_final[m] for m in metrics_to_compare]

            x = np.arange(len(metrics_to_compare))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2,
                baseline_vals,
                width,
                label="Baseline",
                color="#3498db",
                alpha=0.7,
                edgecolor="black",
            )
            bars2 = ax.bar(
                x + width / 2,
                ant_gap_vals,
                width,
                label="ANT+Gap",
                color="#e74c3c",
                alpha=0.7,
                edgecolor="black",
            )

            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title("Final Performance Comparison", fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    "avg_nme1\n(Forgetting)",
                    "avg_acc1\n(Forgetting)",
                    "eval_nme1\n(Current)",
                    "eval_acc1\n(Current)",
                ],
                fontsize=9,
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_performance.png", dpi=150, bbox_inches="tight"
        )
        print(f"✅ Saved: {output_dir / 'comparison_performance.png'}")

        # ====================================================================
        # PLOT 3: Gap vs Performance Correlation
        # ====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Compute average gap per task
        baseline_gap_per_task = []
        ant_gap_gap_per_task = []

        for task in baseline_tasks:
            task_gaps = [e["current_gap"] for e in baseline_epochs if e["task"] == task]
            if task_gaps:
                baseline_gap_per_task.append(np.mean(task_gaps))

        for task in ant_gap_tasks:
            task_gaps = [e["current_gap"] for e in ant_gap_epochs if e["task"] == task]
            if task_gaps:
                ant_gap_gap_per_task.append(np.mean(task_gaps))

        # Left: Gap vs NME1 scatter
        ax = axes[0]
        if len(baseline_gap_per_task) == len(baseline_tasks):
            ax.scatter(
                baseline_gap_per_task,
                baseline_nme1,
                c="blue",
                s=100,
                alpha=0.6,
                edgecolors="black",
                label="Baseline",
                marker="o",
            )
        if len(ant_gap_gap_per_task) == len(ant_gap_tasks):
            ax.scatter(
                ant_gap_gap_per_task,
                ant_gap_nme1,
                c="red",
                s=100,
                alpha=0.6,
                edgecolors="black",
                label="ANT+Gap",
                marker="s",
            )

        # Add trend lines (with safety checks)
        if (
            len(baseline_gap_per_task) == len(baseline_tasks)
            and len(baseline_tasks) > 1
            and np.std(baseline_gap_per_task) > 1e-6  # Check for variance
            and np.std(baseline_nme1) > 1e-6
        ):
            try:
                z = np.polyfit(baseline_gap_per_task, baseline_nme1, 1)
                p = np.poly1d(z)
                x_line = np.linspace(
                    min(baseline_gap_per_task), max(baseline_gap_per_task), 100
                )
                ax.plot(x_line, p(x_line), "b--", alpha=0.5, linewidth=2)
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip trend line if fitting fails

        if (
            len(ant_gap_gap_per_task) == len(ant_gap_tasks)
            and len(ant_gap_tasks) > 1
            and np.std(ant_gap_gap_per_task) > 1e-6
            and np.std(ant_gap_nme1) > 1e-6
        ):
            try:
                z = np.polyfit(ant_gap_gap_per_task, ant_gap_nme1, 1)
                p = np.poly1d(z)
                x_line = np.linspace(
                    min(ant_gap_gap_per_task), max(ant_gap_gap_per_task), 100
                )
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)
            except (np.linalg.LinAlgError, ValueError):
                pass  # Skip trend line if fitting fails

        ax.set_xlabel("Average Gap per Task", fontsize=11)
        ax.set_ylabel("eval_nme1 (Task Performance)", fontsize=11)
        ax.set_title(
            "📈 Correlation: Gap vs Performance", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Right: Gap loss contribution vs Forgetting improvement
        ax = axes[1]
        if ant_gap_metrics and baseline_metrics:
            forgetting_diff = []
            avg_gap_loss_per_task = []

            for task in ant_gap_tasks:
                if task <= len(baseline_metrics):
                    baseline_avg = baseline_metrics[task - 1]["avg_nme1"]
                    ant_gap_avg = ant_gap_metrics[task - 1]["avg_nme1"]
                    forgetting_diff.append(ant_gap_avg - baseline_avg)

                    task_gap_losses = [
                        e["gap_loss"] for e in ant_gap_loss_epochs if e["task"] == task
                    ]
                    if task_gap_losses:
                        avg_gap_loss_per_task.append(np.mean(task_gap_losses))

            if forgetting_diff and avg_gap_loss_per_task:
                ax.scatter(
                    avg_gap_loss_per_task,
                    forgetting_diff,
                    c="purple",
                    s=100,
                    alpha=0.6,
                    edgecolors="black",
                    marker="D",
                )

                # Add trend line (with safety checks)
                if (
                    len(avg_gap_loss_per_task) > 1
                    and np.std(avg_gap_loss_per_task) > 1e-6
                    and np.std(forgetting_diff) > 1e-6
                ):
                    try:
                        z = np.polyfit(avg_gap_loss_per_task, forgetting_diff, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(
                            min(avg_gap_loss_per_task), max(avg_gap_loss_per_task), 100
                        )
                        ax.plot(
                            x_line,
                            p(x_line),
                            "purple",
                            linestyle="--",
                            alpha=0.5,
                            linewidth=2,
                        )
                    except (np.linalg.LinAlgError, ValueError):
                        pass  # Skip trend line if fitting fails

                ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
                ax.set_xlabel("Average Gap Loss per Task", fontsize=11)
                ax.set_ylabel(
                    "Forgetting Improvement\n(ANT+Gap avg_nme1 - Baseline avg_nme1)",
                    fontsize=11,
                )
                ax.set_title(
                    "📊 Gap Loss Impact on Forgetting", fontsize=13, fontweight="bold"
                )
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "comparison_correlation.png", dpi=150, bbox_inches="tight"
        )
        print(f"✅ Saved: {output_dir / 'comparison_correlation.png'}")

    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON - LOSS COMPONENTS")
    print("=" * 80)

    baseline_gaps_all = [d["current_gap"] for d in baseline_data]
    ant_gap_gaps_all = [d["current_gap"] for d in ant_gap_data]

    print("\n📊 Gap Statistics:")
    print(f"{'Metric':<20} {'Baseline':<15} {'ANT+Gap':<15} {'Improvement':<15}")
    print("-" * 65)

    b_mean = np.mean(baseline_gaps_all)
    a_mean = np.mean(ant_gap_gaps_all)
    print(
        f"{'Mean':<20} {b_mean:<15.4f} {a_mean:<15.4f} {((a_mean - b_mean) / abs(b_mean) * 100 if b_mean != 0 else 0):>+7.2f}%"
    )

    b_std = np.std(baseline_gaps_all)
    a_std = np.std(ant_gap_gaps_all)
    print(f"{'Std Dev':<20} {b_std:<15.4f} {a_std:<15.4f}")

    b_max = np.max(baseline_gaps_all)
    a_max = np.max(ant_gap_gaps_all)
    print(
        f"{'Max':<20} {b_max:<15.4f} {a_max:<15.4f} {((a_max - b_max) / abs(b_max) * 100 if b_max != 0 else 0):>+7.2f}%"
    )

    b_final = baseline_gaps[-1] if baseline_gaps else 0
    a_final = ant_gap_gaps[-1] if ant_gap_gaps else 0
    print(
        f"{'Final (last epoch)':<20} {b_final:<15.4f} {a_final:<15.4f} {((a_final - b_final) / abs(b_final) * 100 if b_final != 0 else 0):>+7.2f}%"
    )

    # Pos/Neg means comparison
    print("\n📊 Positive/Negative Similarity Statistics:")
    print(f"{'Metric':<20} {'Baseline':<15} {'ANT+Gap':<15}")
    print("-" * 50)

    baseline_pos_mean = np.mean([e["pos_mean"] for e in baseline_epochs])
    ant_gap_pos_mean = np.mean([e["pos_mean"] for e in ant_gap_epochs])
    print(f"{'Avg pos_mean':<20} {baseline_pos_mean:<15.4f} {ant_gap_pos_mean:<15.4f}")

    baseline_neg_mean = np.mean([e["neg_mean"] for e in baseline_epochs])
    ant_gap_neg_mean = np.mean([e["neg_mean"] for e in ant_gap_epochs])
    print(f"{'Avg neg_mean':<20} {baseline_neg_mean:<15.4f} {ant_gap_neg_mean:<15.4f}")

    # Task-by-task comparison
    print("\n📈 Task-by-Task Gap Progression:")
    print(
        f"{'Task':<10} {'Baseline Start':<18} {'Baseline End':<18} {'ANT+Gap Start':<18} {'ANT+Gap End':<18}"
    )
    print("-" * 82)

    for task in sorted(set(e["task"] for e in baseline_epochs)):
        b_task = [e for e in baseline_epochs if e["task"] == task]
        a_task = [e for e in ant_gap_epochs if e["task"] == task]

        if b_task and a_task:
            print(
                f"Task {task:<5} {b_task[0]['current_gap']:<18.4f} {b_task[-1]['current_gap']:<18.4f} "
                f"{a_task[0]['current_gap']:<18.4f} {a_task[-1]['current_gap']:<18.4f}"
            )

    # Performance comparison (if available)
    if baseline_metrics and ant_gap_metrics:
        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS COMPARISON")
        print("=" * 80)

        # Final task comparison
        if baseline_metrics and ant_gap_metrics:
            baseline_final = baseline_metrics[-1]
            ant_gap_final = ant_gap_metrics[-1]

            print(f"\n🏁 Final Task ({baseline_final['task']}) Performance:")
            print(f"{'Metric':<20} {'Baseline':<15} {'ANT+Gap':<15} {'Difference':<15}")
            print("-" * 65)

            metrics_to_show = [
                ("avg_nme1", "🥇 avg_nme1"),
                ("eval_nme1", "eval_nme1"),
                ("avg_acc1", "avg_acc1"),
                ("eval_acc1", "eval_acc1"),
            ]

            for metric_key, metric_label in metrics_to_show:
                b_val = baseline_final[metric_key]
                a_val = ant_gap_final[metric_key]
                diff = a_val - b_val
                print(f"{metric_label:<20} {b_val:<15.2f} {a_val:<15.2f} {diff:>+7.2f}")

            # Forgetting analysis
            print("\n📉 Forgetting Analysis (First Task → Last Task):")
            baseline_first = baseline_metrics[0]
            ant_gap_first = ant_gap_metrics[0]

            b_nme1_drop = baseline_first["eval_nme1"] - baseline_final["eval_nme1"]
            a_nme1_drop = ant_gap_first["eval_nme1"] - ant_gap_final["eval_nme1"]

            print(
                f"  Baseline NME1 drop:  {baseline_first['eval_nme1']:.2f} → {baseline_final['eval_nme1']:.2f} = -{b_nme1_drop:.2f}"
            )
            print(
                f"  ANT+Gap NME1 drop:   {ant_gap_first['eval_nme1']:.2f} → {ant_gap_final['eval_nme1']:.2f} = -{a_nme1_drop:.2f}"
            )

            if a_nme1_drop < b_nme1_drop:
                reduction = (b_nme1_drop - a_nme1_drop) / b_nme1_drop * 100
                print(f"  ✅ ANT+Gap reduces forgetting by {reduction:.2f}%")
            else:
                increase = (a_nme1_drop - b_nme1_drop) / b_nme1_drop * 100
                print(f"  ⚠️  ANT+Gap increases forgetting by {increase:.2f}%")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if a_mean > b_mean:
        improvement = (a_mean - b_mean) / abs(b_mean) * 100 if b_mean != 0 else 0
        print(
            f"\n✅ Gap Improvement: ANT+Gap achieves {improvement:.2f}% higher average gap"
        )
        print(f"   Baseline: {b_mean:.4f} | ANT+Gap: {a_mean:.4f}")
    else:
        print(f"\n⚠️  Baseline has higher average gap than ANT+Gap")

    if a_final >= 0.7 and b_final < 0.7:
        print(
            f"\n✅ Target Achievement: ANT+Gap reaches gap target (0.7), Baseline does not"
        )
        print(f"   Baseline final: {b_final:.4f} | ANT+Gap final: {a_final:.4f}")
    elif a_final >= 0.7:
        print(f"\n✅ Target Achievement: ANT+Gap reaches gap target (0.7)")
        print(f"   Final gap: {a_final:.4f}")
    elif b_final < 0.7 and a_final < 0.7:
        print(f"\n⚠️  Neither experiment reaches gap target (0.7)")
        print(f"   Baseline: {b_final:.4f} | ANT+Gap: {a_final:.4f}")

    avg_gap_loss = np.mean([d["gap_loss"] for d in ant_gap_loss])
    avg_total_loss = np.mean([d["total_loss"] for d in ant_gap_loss])
    print(f"\n📊 Gap Loss Contribution in ANT+Gap:")
    print(f"   Average gap_loss: {avg_gap_loss:.4f}")
    print(
        f"   With gap_beta=0.5, contributes ~{0.5 * avg_gap_loss:.4f} to weighted loss"
    )
    print(
        f"   Percentage of total loss: {(0.5 * avg_gap_loss / avg_total_loss * 100):.2f}%"
    )

    if baseline_metrics and ant_gap_metrics:
        baseline_final_nme1 = baseline_metrics[-1]["avg_nme1"]
        ant_gap_final_nme1 = ant_gap_metrics[-1]["avg_nme1"]

        if ant_gap_final_nme1 > baseline_final_nme1:
            improvement = ant_gap_final_nme1 - baseline_final_nme1
            print(f"\n✅ Performance: ANT+Gap improves avg_nme1 by +{improvement:.2f}%")
            print(
                f"   Baseline: {baseline_final_nme1:.2f}% | ANT+Gap: {ant_gap_final_nme1:.2f}%"
            )
        else:
            degradation = baseline_final_nme1 - ant_gap_final_nme1
            print(f"\n⚠️  Performance: ANT+Gap degrades avg_nme1 by -{degradation:.2f}%")
            print(
                f"   Baseline: {baseline_final_nme1:.2f}% | ANT+Gap: {ant_gap_final_nme1:.2f}%"
            )

        # Correlation analysis
        if len(ant_gap_metrics) > 2:
            gaps_per_task = []
            nme1_per_task = []
            for task in range(1, len(ant_gap_metrics) + 1):
                task_gaps = [
                    e["current_gap"] for e in ant_gap_epochs if e["task"] == task
                ]
                if task_gaps:
                    gaps_per_task.append(np.mean(task_gaps))
                    nme1_per_task.append(ant_gap_metrics[task - 1]["eval_nme1"])

            if (
                len(gaps_per_task) > 2
                and np.std(gaps_per_task) > 1e-6
                and np.std(nme1_per_task) > 1e-6
            ):
                correlation = np.corrcoef(gaps_per_task, nme1_per_task)[0, 1]
                print(f"\n📈 Correlation (Gap vs NME1 in ANT+Gap): {correlation:.3f}")
                if correlation > 0.5:
                    print(
                        "   ✅ Strong positive correlation - Higher gap → Better performance"
                    )
                elif correlation > 0.2:
                    print("   ⚠️  Moderate positive correlation")
                elif correlation < -0.2:
                    print("   ❌ Negative correlation - Higher gap → Worse performance")
                else:
                    print("   ⚠️  Weak or no clear correlation")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Baseline and ANT+Gap experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two experiments with default paths
    python compare_experiments.py \\
      --baseline-matrix ../../logs/exp_baseline/exp_debug0.log \\
      --baseline-gist ../../logs/exp_baseline/exp_gistlog.log \\
      --ant-gap-matrix ../../logs/exp_ant_gap/exp_debug0.log \\
      --ant-gap-gist ../../logs/exp_ant_gap/exp_gistlog.log  # Using short directory paths (auto-append filenames)
  python compare_experiments.py \\
      --baseline-dir ../../logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal \\
      --ant-gap-dir ../../logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5
        """,
    )
    parser.add_argument(
        "--baseline-matrix", type=str, help="Path to baseline exp_debug0.log"
    )
    parser.add_argument(
        "--baseline-gist", type=str, help="Path to baseline exp_gistlog.log"
    )
    parser.add_argument(
        "--ant-gap-matrix", type=str, help="Path to ANT+Gap exp_debug0.log"
    )
    parser.add_argument(
        "--ant-gap-gist", type=str, help="Path to ANT+Gap exp_gistlog.log"
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        help="Baseline experiment directory (auto-find logs)",
    )
    parser.add_argument(
        "--ant-gap-dir", type=str, help="ANT+Gap experiment directory (auto-find logs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results",
        help="Output directory for plots and results",
    )

    args = parser.parse_args()

    # Handle directory shortcuts
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        args.baseline_matrix = str(baseline_dir / "exp_matrix_debug0.log")
        args.baseline_gist = str(baseline_dir / "exp_gistlog.log")

    if args.ant_gap_dir:
        ant_gap_dir = Path(args.ant_gap_dir)
        args.ant_gap_matrix = str(ant_gap_dir / "exp_matrix_debug0.log")
        args.ant_gap_gist = str(ant_gap_dir / "exp_gistlog.log")

    # Validate required arguments
    if not all(
        [
            args.baseline_matrix,
            args.baseline_gist,
            args.ant_gap_matrix,
            args.ant_gap_gist,
        ]
    ):
        parser.error(
            "Must provide either --baseline-dir/--ant-gap-dir OR all four log paths"
        )

    baseline_matrix = Path(args.baseline_matrix)
    baseline_gist = Path(args.baseline_gist)
    ant_gap_matrix = Path(args.ant_gap_matrix)
    ant_gap_gist = Path(args.ant_gap_gist)

    # Check files exist
    missing_files = []
    for file_path, name in [
        (baseline_matrix, "Baseline matrix log"),
        (baseline_gist, "Baseline gist log"),
        (ant_gap_matrix, "ANT+Gap matrix log"),
        (ant_gap_gist, "ANT+Gap gist log"),
    ]:
        if not file_path.exists():
            missing_files.append(f"  ❌ {name}: {file_path}")

    if missing_files:
        print("Error: Missing required log files:")
        print("\n".join(missing_files))
        return 1

    compare_experiments(
        baseline_matrix, ant_gap_matrix, baseline_gist, ant_gap_gist, args.output
    )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {args.output}/")
    print("\nGenerated files:")
    print("  📊 comparison_losses.png       - Gap & loss components evolution")
    print("  📈 comparison_performance.png  - Task performance & forgetting metrics")
    print("  🔗 comparison_correlation.png  - Gap vs performance correlation")

    return 0


if __name__ == "__main__":
    exit(main())
