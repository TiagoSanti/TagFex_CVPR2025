#!/usr/bin/env python3
"""
Plot NME1 curves from all experiments in logs directory.
Creates a comparative visualization showing the performance trajectory across tasks.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_experiment_name(log_dir_name):
    """Parse experiment directory name to extract configuration details."""

    # Extract configuration parameters
    config = {
        "name": log_dir_name,
        "ant_beta": None,
        "nce_alpha": None,
        "ant_margin": None,
        "ant_scope": None,
        "gap_target": None,
        "gap_beta": None,
    }

    # Parse parameters from directory name
    ant_beta_match = re.search(r"antB([\d.]+)", log_dir_name)
    if ant_beta_match:
        config["ant_beta"] = float(ant_beta_match.group(1))

    nce_alpha_match = re.search(r"nceA([\d.]+)", log_dir_name)
    if nce_alpha_match:
        config["nce_alpha"] = float(nce_alpha_match.group(1))

    ant_margin_match = re.search(r"antM([\d.]+)", log_dir_name)
    if ant_margin_match:
        config["ant_margin"] = float(ant_margin_match.group(1))

    # Check for Local or Global
    if "antLocal" in log_dir_name:
        config["ant_scope"] = "Local"
    elif "antGlobal" in log_dir_name:
        config["ant_scope"] = "Global"

    # Gap parameters
    gap_target_match = re.search(r"gapT([\d.]+)", log_dir_name)
    if gap_target_match:
        config["gap_target"] = float(gap_target_match.group(1))

    gap_beta_match = re.search(r"gapB([\d.]+)", log_dir_name)
    if gap_beta_match:
        config["gap_beta"] = float(gap_beta_match.group(1))

    return config


def create_experiment_label(config):
    """Create a descriptive label for the experiment."""

    name = config["name"]

    # Check for baseline (antB0)
    if config["ant_beta"] == 0:
        return "Baseline (InfoNCE puro, ant_β=0)"

    # Check for original TagFex
    if "done_exp_cifar100_10-10" == name and config["ant_beta"] is None:
        return "TagFex Original"

    # Build label from parameters
    parts = []

    if config["ant_beta"] is not None:
        parts.append(f"ant_β={config['ant_beta']}")

    if config["ant_scope"]:
        parts.append(config["ant_scope"])

    if config["ant_margin"] is not None and config["ant_margin"] != 0.1:
        parts.append(f"margin={config['ant_margin']}")

    if config["gap_target"] is not None:
        parts.append(f"gap_target={config['gap_target']}")

    if config["gap_beta"] is not None:
        parts.append(f"gap_β={config['gap_beta']}")

    if parts:
        return ", ".join(parts)

    # Fallback
    return name.replace("done_exp_", "").replace("idone_exp_", "").replace("exp_", "")


def parse_nme1_curve(log_file):
    """Parse nme1_curve from exp_gistlog.log."""

    # Read the last occurrence of nme1_curve (final results)
    nme1_curve = None

    with open(log_file, "r") as f:
        for line in f:
            # Look for nme1_curve with array format
            match = re.search(r"nme1_curve \[([\d.\s]+)\]", line)
            if match:
                values_str = match.group(1)
                nme1_curve = [float(x) for x in values_str.split()]

    return nme1_curve


def get_all_experiments(logs_dir):
    """Get all experiment directories and their nme1 curves."""

    experiments = []

    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        gist_log = exp_dir / "exp_gistlog.log"
        if not gist_log.exists():
            print(f"  Skipping {exp_dir.name} (no exp_gistlog.log)")
            continue

        # Parse experiment config
        config = parse_experiment_name(exp_dir.name)

        # Parse nme1_curve
        nme1_curve = parse_nme1_curve(gist_log)

        if nme1_curve is None:
            print(f"  Skipping {exp_dir.name} (no nme1_curve found)")
            continue

        # Create label
        label = create_experiment_label(config)

        # Skip TagFex Original (redundant with Baseline)
        if "TagFex Original" in label:
            print(f"  Skipping {exp_dir.name} (redundant with Baseline)")
            continue

        # Skip experiment with gap loss
        if "gap_target" in label or "gap_β" in label:
            print(f"  Skipping {exp_dir.name} (gap loss experiment)")
            continue

        experiments.append(
            {
                "name": exp_dir.name,
                "config": config,
                "label": label,
                "nme1_curve": nme1_curve,
            }
        )

        print(f"  ✓ {exp_dir.name}")
        print(f"    Label: {label}")
        print(f"    NME1 Curve: {nme1_curve}")

    # Sort experiments: Baseline first, then others
    baseline_exps = [e for e in experiments if "Baseline" in e["label"]]
    other_exps = [e for e in experiments if "Baseline" not in e["label"]]
    experiments = baseline_exps + other_exps

    return experiments


def plot_nme1_curves(experiments, output_path):
    """Plot all nme1 curves in a single figure."""

    print("\nGenerating plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    # Markers
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Find best performance at each task
    max_tasks = max(len(exp["nme1_curve"]) for exp in experiments)
    best_at_task = {}  # task -> (exp_idx, value)

    for task_idx in range(max_tasks):
        best_val = -1
        best_exp_idx = -1
        for exp_idx, exp in enumerate(experiments):
            if task_idx < len(exp["nme1_curve"]):
                val = exp["nme1_curve"][task_idx]
                if val > best_val:
                    best_val = val
                    best_exp_idx = exp_idx
        if best_exp_idx >= 0:
            best_at_task[task_idx + 1] = (best_exp_idx, best_val)

    for idx, exp in enumerate(experiments):
        nme1_curve = exp["nme1_curve"]
        label = exp["label"]

        # Baseline without number, others numbered starting from 1
        if "Baseline" in label:
            numbered_label = label
        else:
            # Number other experiments starting from 1
            exp_number = idx  # idx=0 is baseline, so idx=1 becomes [1], etc.
            numbered_label = f"[{exp_number}] {label}"

        # Store numbered label for annotations
        exp["numbered_label"] = numbered_label

        tasks = list(range(1, len(nme1_curve) + 1))

        # Plot line
        ax.plot(
            tasks,
            nme1_curve,
            marker=markers[idx % len(markers)],
            linewidth=2.5,
            markersize=8,
            color=colors[idx],
            label=numbered_label,
            alpha=0.8,
        )

    # Highlight best performance at each task with stars and labels
    for task, (exp_idx, value) in best_at_task.items():
        # Plot star marker
        ax.plot(
            task,
            value,
            marker="*",
            markersize=25,
            color="gold",
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=100,
        )  # High zorder to draw on top

        # Add text annotation showing which experiment is best
        exp_label = experiments[exp_idx]["numbered_label"]
        # Extract just the number for annotation, or "Baseline" if no number
        if exp_label.startswith("["):
            short_label = exp_label.split("]")[0] + "]"
        else:
            short_label = "Baseline"

        # Position text above the star
        ax.annotate(
            short_label,
            xy=(task, value),
            xytext=(0, 8),  # 8 points above
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="yellow",
                edgecolor="black",
                alpha=0.8,
                linewidth=0.5,
            ),
            zorder=101,
        )

    # Formatting
    ax.set_xlabel("Task", fontsize=14, fontweight="bold")
    ax.set_ylabel("NME Top-1 Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Comparação de Performance: NME1 Curve Across All Experiments",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # X-axis: integer tasks
    max_tasks = max(len(exp["nme1_curve"]) for exp in experiments)
    ax.set_xticks(range(1, max_tasks + 1))
    ax.set_xlim(0.5, max_tasks + 0.5)

    # Y-axis: percentage
    ax.set_ylim(55, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Legend
    # Add star explanation to legend
    from matplotlib.lines import Line2D

    legend_elements = ax.get_legend_handles_labels()
    star_marker = Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=1.5,
        markersize=18,
        label="★ Melhor performance na task",
    )

    # Place legend outside plot if too many experiments
    if len(experiments) > 6:
        handles, labels = legend_elements
        handles.append(star_marker)
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            framealpha=0.9,
            edgecolor="black",
        )
        bbox_inches = "tight"
    else:
        handles, labels = legend_elements
        handles.append(star_marker)
        ax.legend(
            handles=handles, loc="best", fontsize=10, framealpha=0.9, edgecolor="black"
        )
        bbox_inches = "tight"

    # Add final performance annotation
    for idx, exp in enumerate(experiments):
        final_acc = exp["nme1_curve"][-1]
        final_task = len(exp["nme1_curve"])

        # Annotate only if not too crowded
        if len(experiments) <= 5:
            ax.annotate(
                f"{final_acc:.1f}%",
                xy=(final_task, final_acc),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                color=colors[idx],
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches=bbox_inches)
    print(f"  Saved: {output_path}")
    plt.close()


def plot_nme1_curves_separate(experiments, output_path):
    """Plot curves in separate subplots for better readability when many experiments."""

    print("\nGenerating detailed comparison plot...")

    n_exp = len(experiments)
    n_cols = 2
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(
        "Comparação Detalhada: NME1 Curve por Experimento",
        fontsize=16,
        fontweight="bold",
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_exp))

    for idx, exp in enumerate(experiments):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        nme1_curve = exp["nme1_curve"]
        tasks = list(range(1, len(nme1_curve) + 1))

        # Plot
        ax.plot(
            tasks,
            nme1_curve,
            "o-",
            linewidth=3,
            markersize=10,
            color=colors[idx],
            alpha=0.8,
        )

        # Formatting
        ax.set_xlabel("Task", fontsize=11)
        ax.set_ylabel("NME1 Accuracy (%)", fontsize=11)
        ax.set_title(exp["label"], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add values
        for t, acc in zip(tasks, nme1_curve):
            ax.text(t, acc + 1.5, f"{acc:.1f}", ha="center", fontsize=8, alpha=0.7)

        # Limits
        ax.set_xticks(tasks)
        ax.set_ylim(55, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

        # Highlight final performance
        final_acc = nme1_curve[-1]
        ax.axhline(y=final_acc, color="red", linestyle="--", linewidth=1, alpha=0.3)
        ax.text(
            0.98,
            final_acc / 100,
            f"Final: {final_acc:.2f}%",
            transform=ax.transData,
            ha="right",
            va="bottom",
            fontsize=9,
            color="red",
            alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Hide empty subplots
    for idx in range(n_exp, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def generate_summary_table(experiments, output_path):
    """Generate a summary table with key statistics."""

    print("\nGenerating summary table...")

    lines = []
    lines.append("=" * 100)
    lines.append("RESUMO: NME1 PERFORMANCE COMPARISON")
    lines.append("=" * 100)
    lines.append("")

    # Header
    lines.append(
        f"{'Experiment':<50} | {'Tasks':<6} | {'Final':<8} | {'Avg':<8} | {'Std':<8} | {'Max Drop':<10}"
    )
    lines.append("-" * 100)

    # Data
    for exp in sorted(experiments, key=lambda x: x["nme1_curve"][-1], reverse=True):
        curve = exp["nme1_curve"]
        label = exp["label"]

        n_tasks = len(curve)
        final = curve[-1]
        avg = np.mean(curve)
        std = np.std(curve)
        max_drop = max(curve) - min(curve)

        lines.append(
            f"{label:<50} | {n_tasks:<6} | {final:>7.2f}% | {avg:>7.2f}% | {std:>7.2f}% | {max_drop:>9.2f}%"
        )

    lines.append("-" * 100)
    lines.append("")

    # Best/Worst
    best = max(experiments, key=lambda x: x["nme1_curve"][-1])
    worst = min(experiments, key=lambda x: x["nme1_curve"][-1])

    lines.append(
        f"🏆 BEST Final Performance: {best['label']} ({best['nme1_curve'][-1]:.2f}%)"
    )
    lines.append(
        f"⚠️  WORST Final Performance: {worst['label']} ({worst['nme1_curve'][-1]:.2f}%)"
    )
    lines.append("")

    # Average forgetting
    lines.append("FORGETTING ANALYSIS (Task 1 → Final):")
    for exp in experiments:
        curve = exp["nme1_curve"]
        forgetting = curve[0] - curve[-1]
        lines.append(f"  {exp['label']:<50}: {forgetting:>6.2f}% drop")

    lines.append("")
    lines.append("=" * 100)

    # Write
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {output_path}")

    # Also print to console
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Plot NME1 curves from all experiments"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory containing experiment logs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_nme1_comparison",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NME1 CURVE COMPARISON: ALL EXPERIMENTS")
    print("=" * 80)
    print(f"Logs directory: {logs_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Get all experiments
    print("\nScanning experiments...")
    experiments = get_all_experiments(logs_dir)

    if not experiments:
        print("\n❌ No valid experiments found!")
        return

    print(f"\n✓ Found {len(experiments)} experiments")

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Main comparison plot
    plot_nme1_curves(experiments, output_dir / "nme1_curves_comparison.png")

    # Detailed subplots
    plot_nme1_curves_separate(experiments, output_dir / "nme1_curves_detailed.png")

    # Summary table
    generate_summary_table(experiments, output_dir / "nme1_summary.txt")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - nme1_curves_comparison.png    (all curves in one plot)")
    print("  - nme1_curves_detailed.png      (individual subplots)")
    print("  - nme1_summary.txt              (statistics table)")
    print("")


if __name__ == "__main__":
    main()
