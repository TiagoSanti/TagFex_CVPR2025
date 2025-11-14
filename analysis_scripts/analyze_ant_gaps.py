#!/usr/bin/env python3
"""
Script to analyze ANT gap statistics from the enhanced debug logs.

This script parses the exp_matrix_debug0.log files to extract and analyze
the detailed distance statistics logged by the ANT loss function, including:
- Positive similarities (anchor-positive pairs)
- Negative similarities (anchor-negative pairs)
- Gaps (distance from max negative to margin threshold)
- Margin violation percentages
- ANT loss values

The analysis validates the saturation hypothesis by checking if:
1. Distances improve over time (pos↑, neg↓)
2. Gaps remain constant despite distance improvements
3. Violation rate stays high (~98%)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_ant_distance_stats(log_file):
    """
    Parse ANT distance statistics from the debug log file.

    Args:
        log_file: Path to the exp_matrix_debug0.log file

    Returns:
        DataFrame with columns: task, epoch, batch, pos_mean, pos_std, pos_min, pos_max,
        neg_mean, neg_std, neg_min, neg_max, gap_mean, gap_std, gap_min, gap_max,
        margin, violation_pct, ant_loss
    """
    data = []

    # Regex pattern to match the ANT distance stats log lines
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: (.+)"

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                task = int(match.group(1))
                epoch = int(match.group(2))
                batch = int(match.group(3))
                stats_str = match.group(4)

                # Parse the stats string
                stats = {}
                for item in stats_str.split(" | "):
                    key, value = item.split(": ")
                    if key == "violation_pct":
                        stats[key] = float(value.rstrip("%"))
                    else:
                        stats[key] = float(value)

                # Combine all information
                entry = {"task": task, "epoch": epoch, "batch": batch, **stats}
                data.append(entry)

    if not data:
        print(f"Warning: No ANT distance stats found in {log_file}")
        return None

    return pd.DataFrame(data)


def analyze_gap_evolution(df, exp_name):
    """
    Analyze how gaps evolve over training epochs.

    Args:
        df: DataFrame with parsed statistics
        exp_name: Name of the experiment for titles
    """
    # Group by epoch and compute average statistics
    epoch_stats = (
        df.groupby("epoch")
        .agg(
            {
                "pos_mean": "mean",
                "neg_mean": "mean",
                "gap_mean": "mean",
                "violation_pct": "mean",
                "ant_loss": "mean",
                "margin": "first",
            }
        )
        .reset_index()
    )

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Analysis for: {exp_name}")
    print(f"{'='*60}")
    print(f"Total epochs: {epoch_stats['epoch'].max()}")
    print(f"Margin: {epoch_stats['margin'].iloc[0]:.4f}")
    print(f"\nInitial values (epoch 0):")
    print(f"  pos_mean: {epoch_stats['pos_mean'].iloc[0]:.4f}")
    print(f"  neg_mean: {epoch_stats['neg_mean'].iloc[0]:.4f}")
    print(f"  gap_mean: {epoch_stats['gap_mean'].iloc[0]:.4f}")
    print(f"  violation_pct: {epoch_stats['violation_pct'].iloc[0]:.2f}%")
    print(f"  ant_loss: {epoch_stats['ant_loss'].iloc[0]:.4f}")

    print(f"\nFinal values (epoch {epoch_stats['epoch'].max()}):")
    print(f"  pos_mean: {epoch_stats['pos_mean'].iloc[-1]:.4f}")
    print(f"  neg_mean: {epoch_stats['neg_mean'].iloc[-1]:.4f}")
    print(f"  gap_mean: {epoch_stats['gap_mean'].iloc[-1]:.4f}")
    print(f"  violation_pct: {epoch_stats['violation_pct'].iloc[-1]:.2f}%")
    print(f"  ant_loss: {epoch_stats['ant_loss'].iloc[-1]:.4f}")

    # Compute changes
    pos_change = epoch_stats["pos_mean"].iloc[-1] - epoch_stats["pos_mean"].iloc[0]
    neg_change = epoch_stats["neg_mean"].iloc[-1] - epoch_stats["neg_mean"].iloc[0]
    gap_change = epoch_stats["gap_mean"].iloc[-1] - epoch_stats["gap_mean"].iloc[0]
    violation_change = (
        epoch_stats["violation_pct"].iloc[-1] - epoch_stats["violation_pct"].iloc[0]
    )
    loss_change = epoch_stats["ant_loss"].iloc[-1] - epoch_stats["ant_loss"].iloc[0]

    print(f"\nChanges over training:")
    print(
        f"  Δpos_mean: {pos_change:+.4f} ({pos_change/epoch_stats['pos_mean'].iloc[0]*100:+.2f}%)"
    )
    print(
        f"  Δneg_mean: {neg_change:+.4f} ({neg_change/epoch_stats['neg_mean'].iloc[0]*100:+.2f}%)"
    )
    print(
        f"  Δgap_mean: {gap_change:+.4f} ({gap_change/abs(epoch_stats['gap_mean'].iloc[0])*100:+.2f}%)"
    )
    print(f"  Δviolation_pct: {violation_change:+.2f}%")
    print(
        f"  Δant_loss: {loss_change:+.4f} ({loss_change/epoch_stats['ant_loss'].iloc[0]*100:+.2f}%)"
    )

    # Check saturation hypothesis
    print(f"\n{'='*60}")
    print("Saturation Hypothesis Validation:")
    print(f"{'='*60}")

    # Hypothesis: distances improve but gap stays constant
    pos_improved = pos_change > 0.01  # pos should increase
    neg_decreased = neg_change < -0.01  # neg should decrease
    gap_constant = abs(gap_change) < 0.02  # gap should stay relatively constant
    high_violation = epoch_stats["violation_pct"].mean() > 90  # violations stay high

    print(f"✓ Positive similarities improved: {pos_improved} (Δ={pos_change:+.4f})")
    print(f"✓ Negative similarities decreased: {neg_decreased} (Δ={neg_change:+.4f})")
    print(f"✗ Gap remained constant: {gap_constant} (Δ={gap_change:+.4f})")
    print(
        f"✓ Violation rate stayed high: {high_violation} (mean={epoch_stats['violation_pct'].mean():.2f}%)"
    )

    if pos_improved and neg_decreased and gap_constant and high_violation:
        print("\n✓✓✓ HYPOTHESIS CONFIRMED ✓✓✓")
        print("The ANT loss is saturated: distances improve but gap remains constant,")
        print("leading to perpetual margin violations and stagnant ANT loss.")
    else:
        print("\n✗✗✗ HYPOTHESIS REJECTED ✗✗✗")
        print("The data does not support the saturation hypothesis.")

    return epoch_stats


def plot_gap_analysis(epoch_stats, exp_name, output_file):
    """
    Create comprehensive visualization of gap evolution.

    Args:
        epoch_stats: DataFrame with per-epoch statistics
        exp_name: Name of the experiment
        output_file: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"ANT Gap Analysis - {exp_name}", fontsize=16, fontweight="bold")

    epochs = epoch_stats["epoch"]
    margin = epoch_stats["margin"].iloc[0]

    # 1. Positive and Negative Similarities
    ax = axes[0, 0]
    ax.plot(
        epochs,
        epoch_stats["pos_mean"],
        "g-",
        linewidth=2,
        label="Positive (anchor-augmented)",
    )
    ax.plot(
        epochs,
        epoch_stats["neg_mean"],
        "r-",
        linewidth=2,
        label="Negative (anchor-others)",
    )
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Positive vs Negative Similarities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Gap Evolution
    ax = axes[0, 1]
    ax.plot(
        epochs, epoch_stats["gap_mean"], "b-", linewidth=2, label="Gap (q1 - q1_max)"
    )
    ax.axhline(
        y=-margin,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Margin threshold = -{margin}",
    )
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap Value")
    ax.set_title("Gap Evolution vs Margin")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(epochs, -margin, 0, alpha=0.2, color="red", label="Violation zone")

    # 3. Violation Percentage
    ax = axes[0, 2]
    ax.plot(epochs, epoch_stats["violation_pct"], "r-", linewidth=2)
    ax.axhline(y=95, color="orange", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation %")
    ax.set_title("Margin Violation Rate")
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. ANT Loss
    ax = axes[1, 0]
    ax.plot(epochs, epoch_stats["ant_loss"], "purple", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ANT Loss")
    ax.set_title("ANT Loss Evolution")
    ax.grid(True, alpha=0.3)

    # 5. Distance Changes (normalized)
    ax = axes[1, 1]
    pos_norm = (
        (epoch_stats["pos_mean"] - epoch_stats["pos_mean"].iloc[0])
        / epoch_stats["pos_mean"].iloc[0]
        * 100
    )
    neg_norm = (
        (epoch_stats["neg_mean"] - epoch_stats["neg_mean"].iloc[0])
        / epoch_stats["neg_mean"].iloc[0]
        * 100
    )
    gap_norm = (
        (epoch_stats["gap_mean"] - epoch_stats["gap_mean"].iloc[0])
        / abs(epoch_stats["gap_mean"].iloc[0])
        * 100
    )

    ax.plot(epochs, pos_norm, "g-", linewidth=2, label="Positive")
    ax.plot(epochs, neg_norm, "r-", linewidth=2, label="Negative")
    ax.plot(epochs, gap_norm, "b-", linewidth=2, label="Gap")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Change (%)")
    ax.set_title("Normalized Distance Changes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Saturation Check
    ax = axes[1, 2]
    # Show gap distribution relative to margin
    gap_relative = epoch_stats["gap_mean"] / margin
    ax.plot(epochs, gap_relative, "b-", linewidth=2)
    ax.axhline(
        y=-1, color="orange", linestyle="--", linewidth=2, label="Margin threshold"
    )
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap / Margin")
    ax.set_title("Relative Gap Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(epochs, -2, -1, alpha=0.2, color="red", label="Violation zone")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")


def main():
    """Main analysis function."""

    # Define experiments to analyze
    experiments = [
        {
            "name": "exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_v2",
            "log_file": "/home/tiago/TagFex_CVPR2025/logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_v2/exp_matrix_debug0.log",
            "short_name": "antM0.1_Local_v2",
        },
    ]

    all_epoch_stats = {}

    for exp in experiments:
        log_path = Path(exp["log_file"])

        if not log_path.exists():
            print(f"\nWarning: Log file not found: {log_path}")
            print(
                "This is expected if you haven't run a new experiment with the enhanced logging yet."
            )
            continue

        print(f"\nProcessing: {exp['name']}")

        # Parse the log file
        df = parse_ant_distance_stats(log_path)

        if df is None or len(df) == 0:
            print(f"No ANT distance stats found in {log_path}")
            print(
                "Make sure to run a training with the updated code that logs ANT distance statistics."
            )
            continue

        # Analyze gap evolution
        epoch_stats = analyze_gap_evolution(df, exp["short_name"])
        all_epoch_stats[exp["short_name"]] = epoch_stats

        # Create visualization
        output_file = log_path.parent / f"ant_gap_analysis_{exp['short_name']}.png"
        plot_gap_analysis(epoch_stats, exp["short_name"], output_file)

    # Create comparative plot if we have multiple experiments
    if len(all_epoch_stats) > 1:
        create_comparative_plot(all_epoch_stats)


def create_comparative_plot(all_epoch_stats):
    """
    Create a comparative plot across all experiments.

    Args:
        all_epoch_stats: Dictionary mapping experiment names to epoch statistics DataFrames
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Comparative ANT Gap Analysis Across Experiments",
        fontsize=16,
        fontweight="bold",
    )

    colors = ["blue", "green", "red", "purple", "orange"]

    # 1. Gap Evolution Comparison
    ax = axes[0, 0]
    for i, (exp_name, stats) in enumerate(all_epoch_stats.items()):
        margin = stats["margin"].iloc[0]
        ax.plot(
            stats["epoch"],
            stats["gap_mean"],
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{exp_name} (margin={margin})",
        )
        ax.axhline(y=-margin, color=colors[i % len(colors)], linestyle="--", alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap Mean")
    ax.set_title("Gap Evolution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Violation Rate Comparison
    ax = axes[0, 1]
    for i, (exp_name, stats) in enumerate(all_epoch_stats.items()):
        ax.plot(
            stats["epoch"],
            stats["violation_pct"],
            color=colors[i % len(colors)],
            linewidth=2,
            label=exp_name,
        )
    ax.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation %")
    ax.set_title("Violation Rate Comparison")
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ANT Loss Comparison
    ax = axes[1, 0]
    for i, (exp_name, stats) in enumerate(all_epoch_stats.items()):
        ax.plot(
            stats["epoch"],
            stats["ant_loss"],
            color=colors[i % len(colors)],
            linewidth=2,
            label=exp_name,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ANT Loss")
    ax.set_title("ANT Loss Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Positive vs Negative Similarity
    ax = axes[1, 1]
    for i, (exp_name, stats) in enumerate(all_epoch_stats.items()):
        epochs = stats["epoch"]
        # Plot difference between positive and negative
        diff = stats["pos_mean"] - stats["neg_mean"]
        ax.plot(
            epochs, diff, color=colors[i % len(colors)], linewidth=2, label=exp_name
        )
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("pos_mean - neg_mean")
    ax.set_title("Similarity Gap (Positive - Negative)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "/home/tiago/TagFex_CVPR2025/logs/ant_gap_analysis_comparative.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nComparative plot saved to: {output_file}")


if __name__ == "__main__":
    main()
