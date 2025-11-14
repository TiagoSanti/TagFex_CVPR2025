#!/usr/bin/env python3
"""
Simplified ANT gap analysis focused on the key insight:
pos_mean - neg_mean vs margin threshold
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_logs(log_file):
    """Parse ANT distance stats from log file."""
    data = []
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) .* neg_mean: ([\d.]+) .* gap_mean: ([-\d.]+) .* margin: ([\d.]+) .* violation_pct: ([\d.]+)% \| ant_loss: ([\d.]+)"

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
                    }
                )

    return data


def main():
    log_file = Path(
        "/home/tiago/TagFex_CVPR2025/logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_v2/exp_matrix_debug0.log"
    )

    print("Parsing logs...")
    data = parse_logs(log_file)

    if not data:
        print("No data found!")
        return

    print(f"Found {len(data)} data points")

    # Group by epoch
    epochs = {}
    for d in data:
        epoch = d["epoch"]
        if epoch not in epochs:
            epochs[epoch] = []
        epochs[epoch].append(d)

    # Compute epoch averages
    epoch_data = []
    for epoch in sorted(epochs.keys()):
        batch_data = epochs[epoch]
        epoch_data.append(
            {
                "epoch": epoch,
                "pos_mean": np.mean([d["pos_mean"] for d in batch_data]),
                "neg_mean": np.mean([d["neg_mean"] for d in batch_data]),
                "gap_mean": np.mean([d["gap_mean"] for d in batch_data]),
                "margin": batch_data[0]["margin"],
                "violation_pct": np.mean([d["violation_pct"] for d in batch_data]),
                "ant_loss": np.mean([d["ant_loss"] for d in batch_data]),
            }
        )

    # Calculate TRUE gap (pos - neg)
    for d in epoch_data:
        d["true_gap"] = d["pos_mean"] - d["neg_mean"]

    # Print summary
    print("\n" + "=" * 80)
    print("SATURATION ANALYSIS - Key Insight")
    print("=" * 80)
    print(f"Margin required: {epoch_data[0]['margin']:.4f}")
    print(f"\nEpoch 1:")
    print(f"  pos_mean: {epoch_data[0]['pos_mean']:.4f}")
    print(f"  neg_mean: {epoch_data[0]['neg_mean']:.4f}")
    print(f"  TRUE gap (pos-neg): {epoch_data[0]['true_gap']:.4f}")
    print(
        f"  Margin violation: {epoch_data[0]['true_gap']:.4f} < {epoch_data[0]['margin']:.4f} → {'YES' if epoch_data[0]['true_gap'] < epoch_data[0]['margin'] else 'NO'}"
    )
    print(
        f"  Gap is {epoch_data[0]['margin']/epoch_data[0]['true_gap']:.1f}x smaller than required!"
    )

    print(f"\nEpoch {len(epoch_data)}:")
    print(f"  pos_mean: {epoch_data[-1]['pos_mean']:.4f}")
    print(f"  neg_mean: {epoch_data[-1]['neg_mean']:.4f}")
    print(f"  TRUE gap (pos-neg): {epoch_data[-1]['true_gap']:.4f}")
    print(
        f"  Margin violation: {epoch_data[-1]['true_gap']:.4f} < {epoch_data[-1]['margin']:.4f} → {'YES' if epoch_data[-1]['true_gap'] < epoch_data[-1]['margin'] else 'NO'}"
    )

    # Check hypothesis
    print("\n" + "=" * 80)
    print("HYPOTHESIS VALIDATION")
    print("=" * 80)

    # Check if gap stays much smaller than margin
    avg_true_gap = np.mean([d["true_gap"] for d in epoch_data])
    margin = epoch_data[0]["margin"]

    print(f"Average TRUE gap across all epochs: {avg_true_gap:.4f}")
    print(f"Required margin: {margin:.4f}")
    print(f"Gap is {margin/avg_true_gap:.1f}x smaller than required!")

    if avg_true_gap < margin * 0.5:
        print("\n✓✓✓ SATURATION CONFIRMED ✓✓✓")
        print("The gap between positive and negative similarities is systematically")
        print("much smaller than the margin, causing persistent violations.")
    else:
        print("\n✗✗✗ NO SATURATION ✗✗✗")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "ANT Saturation Analysis - antM0.1_Local_v2", fontsize=16, fontweight="bold"
    )

    epochs_list = [d["epoch"] for d in epoch_data]

    # 1. Pos vs Neg similarities
    ax = axes[0, 0]
    ax.plot(
        epochs_list,
        [d["pos_mean"] for d in epoch_data],
        "g-",
        linewidth=2,
        label="Positive (same sample)",
        marker="o",
    )
    ax.plot(
        epochs_list,
        [d["neg_mean"] for d in epoch_data],
        "r-",
        linewidth=2,
        label="Negative (other samples)",
        marker="s",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Positive vs Negative Similarities")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. TRUE gap vs margin
    ax = axes[0, 1]
    ax.plot(
        epochs_list,
        [d["true_gap"] for d in epoch_data],
        "b-",
        linewidth=2,
        label="TRUE gap (pos - neg)",
        marker="o",
    )
    ax.axhline(
        y=margin,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Required margin = {margin}",
    )
    ax.fill_between(
        epochs_list, 0, margin, alpha=0.2, color="red", label="Violation zone"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap Value")
    ax.set_title("TRUE Gap vs Required Margin")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Violation percentage
    ax = axes[1, 0]
    ax.plot(
        epochs_list,
        [d["violation_pct"] for d in epoch_data],
        "purple",
        linewidth=2,
        marker="o",
    )
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Violation %")
    ax.set_title("Margin Violation Rate")
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. ANT Loss
    ax = axes[1, 1]
    ax.plot(
        epochs_list,
        [d["ant_loss"] for d in epoch_data],
        "darkred",
        linewidth=2,
        marker="o",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ANT Loss")
    ax.set_title("ANT Loss Evolution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = log_file.parent / "ant_saturation_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_file}")


if __name__ == "__main__":
    main()
