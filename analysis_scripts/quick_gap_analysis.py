#!/usr/bin/env python3
"""
Quick analysis of gap behavior from current experiment.
Checks if gap maximization loss is working as expected.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_ant_stats(log_file):
    """Parse ANT distance stats from log file."""
    data = []
    # Updated pattern to include new metrics
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


def analyze_gap_progression(data):
    """Analyze if gap is increasing towards target."""
    if not data:
        print("No data found!")
        return

    # Calculate actual gap (pos - neg)
    for d in data:
        d["actual_gap"] = d["pos_mean"] - d["neg_mean"]

    # Group by task and epoch
    tasks = sorted(set(d["task"] for d in data))

    print("=" * 80)
    print("GAP MAXIMIZATION ANALYSIS - CURRENT EXPERIMENT")
    print("=" * 80)
    print(f"\nConfig: gap_target=0.7, gap_beta=0.5, margin=0.1")
    print(f"Total batches logged: {len(data)}")
    print(f"Tasks found: {tasks}")

    for task in tasks:
        task_data = [d for d in data if d["task"] == task]
        epochs = sorted(set(d["epoch"] for d in task_data))

        print(f"\n{'='*80}")
        print(f"TASK {task} - {len(epochs)} epochs")
        print(f"{'='*80}")

        for epoch in epochs[:5] + epochs[-5:]:  # First 5 and last 5 epochs
            epoch_data = [d for d in task_data if d["epoch"] == epoch]

            gaps = [d["actual_gap"] for d in epoch_data]
            gap_means = [d["gap_mean"] for d in epoch_data]
            violations = [d["violation_pct"] for d in epoch_data]
            ant_losses = [d["ant_loss"] for d in epoch_data]
            gap_losses = [d.get("gap_loss", 0.0) for d in epoch_data]
            current_gaps = [d.get("current_gap", d["actual_gap"]) for d in epoch_data]
            total_ant_losses = [
                d.get("total_ant_loss", d["ant_loss"]) for d in epoch_data
            ]

            avg_gap = np.mean(gaps)
            avg_gap_mean = np.mean(gap_means)
            avg_violation = np.mean(violations)
            avg_ant_loss = np.mean(ant_losses)
            avg_gap_loss = np.mean(gap_losses)
            avg_current_gap = np.mean(current_gaps)
            avg_total_ant_loss = np.mean(total_ant_losses)

            print(
                f"\nEpoch {epoch:2d}: gap={avg_current_gap:6.3f} | gap_loss={avg_gap_loss:6.4f} | "
                f"ant_loss={avg_ant_loss:.4f} | total_ant={avg_total_ant_loss:.4f} | violation={avg_violation:5.2f}%"
            )

            if avg_current_gap < 0.7:
                shortfall = 0.7 - avg_current_gap
                expected_gap_loss = max(0, shortfall)
                print(
                    f"          ⚠️  Gap below target! Shortfall: {shortfall:.3f} | Expected gap_loss: {expected_gap_loss:.4f}"
                )
            else:
                print(f"          ✅ Gap meets target (0.7)")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")

    all_gaps = [d.get("current_gap", d["actual_gap"]) for d in data]
    all_ant_losses = [d["ant_loss"] for d in data]
    all_gap_losses = [d.get("gap_loss", 0.0) for d in data]
    all_total_ant_losses = [d.get("total_ant_loss", d["ant_loss"]) for d in data]

    print(f"\nCurrent Gap (pos - neg):")
    print(f"  Mean:   {np.mean(all_gaps):.4f}")
    print(f"  Std:    {np.std(all_gaps):.4f}")
    print(f"  Min:    {np.min(all_gaps):.4f}")
    print(f"  Max:    {np.max(all_gaps):.4f}")
    print(f"  Target: 0.7000")

    print(f"\nANT Loss (base):")
    print(f"  Mean:   {np.mean(all_ant_losses):.4f}")
    print(f"  Std:    {np.std(all_ant_losses):.4f}")
    print(f"  Min:    {np.min(all_ant_losses):.4f}")
    print(f"  Max:    {np.max(all_ant_losses):.4f}")

    if any(d.get("gap_loss") is not None for d in data):
        print(f"\nGap Loss:")
        print(f"  Mean:   {np.mean(all_gap_losses):.4f}")
        print(f"  Std:    {np.std(all_gap_losses):.4f}")
        print(f"  Min:    {np.min(all_gap_losses):.4f}")
        print(f"  Max:    {np.max(all_gap_losses):.4f}")

        print(f"\nTotal ANT Loss (base + gap_beta * gap_loss):")
        print(f"  Mean:   {np.mean(all_total_ant_losses):.4f}")
        print(f"  Std:    {np.std(all_total_ant_losses):.4f}")
        print(f"  Min:    {np.min(all_total_ant_losses):.4f}")
        print(f"  Max:    {np.max(all_total_ant_losses):.4f}")

        avg_gap_contribution = np.mean([d.get("gap_loss", 0.0) * 0.5 for d in data])
        print(f"\nGap Loss Contribution (gap_beta=0.5):")
        print(f"  Mean contribution: {avg_gap_contribution:.4f}")
        print(
            f"  Percentage of total ANT: {100 * avg_gap_contribution / np.mean(all_total_ant_losses):.2f}%"
        )

    # Check if gap is increasing over time
    print(f"\n{'='*80}")
    print("GAP PROGRESSION CHECK")
    print(f"{'='*80}")

    for task in tasks:
        task_data = [d for d in data if d["task"] == task]
        epochs = sorted(set(d["epoch"] for d in task_data))

        if len(epochs) < 2:
            continue

        first_epoch_data = [d for d in task_data if d["epoch"] == epochs[0]]
        last_epoch_data = [d for d in task_data if d["epoch"] == epochs[-1]]

        first_gap = np.mean(
            [d.get("current_gap", d["actual_gap"]) for d in first_epoch_data]
        )
        last_gap = np.mean(
            [d.get("current_gap", d["actual_gap"]) for d in last_epoch_data]
        )

        gap_change = last_gap - first_gap

        print(f"\nTask {task}:")
        print(f"  First epoch gap: {first_gap:.4f}")
        print(f"  Last epoch gap:  {last_gap:.4f}")
        print(f"  Change:          {gap_change:+.4f}")

        if gap_change > 0:
            print(f"  Status:          ✅ Gap increasing (good!)")
        else:
            print(f"  Status:          ❌ Gap decreasing or flat")

    print(f"\n{'='*80}")
    print("KEY OBSERVATIONS")
    print(f"{'='*80}")

    avg_gap = np.mean(all_gaps)
    has_gap_metrics = any(d.get("gap_loss") is not None for d in data)

    if avg_gap < 0.7:
        expected_gap_loss = max(0, 0.7 - avg_gap)
        print(f"\n⚠️  Average gap ({avg_gap:.3f}) is below target (0.7)")
        print(f"   Gap shortfall: {0.7 - avg_gap:.3f}")
        print(
            f"   Expected gap_loss = relu(0.7 - {avg_gap:.3f}) = {expected_gap_loss:.3f}"
        )
        print(f"   Expected contribution (gap_beta=0.5): {0.5 * expected_gap_loss:.3f}")

        if has_gap_metrics:
            actual_gap_loss = np.mean(all_gap_losses)
            print(f"\n   ✅ Actual gap_loss from logs: {actual_gap_loss:.3f}")
            if abs(actual_gap_loss - expected_gap_loss) < 0.01:
                print(f"   ✅ Matches expected! Gap maximization is working correctly.")
            else:
                print(
                    f"   ⚠️  Differs from expected by {abs(actual_gap_loss - expected_gap_loss):.3f}"
                )
    else:
        print(f"\n✅ Gap ({avg_gap:.3f}) meets or exceeds target (0.7)")
        print(f"   Gap maximization loss should be ~0 (no penalty needed)")

        if has_gap_metrics:
            actual_gap_loss = np.mean(all_gap_losses)
            print(f"   Actual gap_loss: {actual_gap_loss:.3f}")

    if has_gap_metrics:
        print(f"\n✅ Gap maximization metrics are being logged correctly!")
    else:
        print(f"\n💡 NOTE: Gap maximization metrics not found in logs.")
        print(
            f"   Run experiment with updated logging to see gap_loss, current_gap, gap_target."
        )


def main():
    log_file = Path(
        "/home/tiago/TagFex_CVPR2025/logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5/exp_matrix_debug0.log"
    )

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return

    print("Parsing logs...")
    data = parse_ant_stats(log_file)

    if not data:
        print("No ANT distance stats found in log!")
        return

    analyze_gap_progression(data)


if __name__ == "__main__":
    main()
