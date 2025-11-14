#!/usr/bin/env python3
"""
Comparative analysis between Baseline (no ANT) and ANT+Gap experiments.
Compares gap evolution, loss components, and final performance.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def parse_ant_stats(log_file):
    """Parse ANT distance stats from log file."""
    data = []
    # Pattern for logs with gap_loss metrics
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) .* neg_mean: ([-\d.]+) .* gap_mean: ([-\d.]+) .* margin: ([\d.]+) .* violation_pct: ([\d.]+)% \| ant_loss: ([\d.]+) \| gap_loss: ([\d.]+) \| current_gap: ([-\d.]+) \| gap_target: ([\d.]+) \| total_ant_loss: ([\d.]+)"

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append({
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
                })

    return data


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
        aggregated.append({
            "task": task,
            "epoch": epoch,
            "current_gap": np.mean([b["current_gap"] for b in batch_data]),
            "gap_loss": np.mean([b["gap_loss"] for b in batch_data]),
            "ant_loss": np.mean([b["ant_loss"] for b in batch_data]),
            "total_ant_loss": np.mean([b["total_ant_loss"] for b in batch_data]),
            "violation_pct": np.mean([b["violation_pct"] for b in batch_data]),
        })
    
    return aggregated


def compare_experiments(baseline_log, ant_gap_log, output_dir):
    """Compare baseline and ANT+Gap experiments."""
    
    print("=" * 80)
    print("COMPARATIVE ANALYSIS: Baseline vs ANT+Gap")
    print("=" * 80)
    
    # Parse logs
    print("\nParsing logs...")
    baseline_data = parse_ant_stats(baseline_log)
    ant_gap_data = parse_ant_stats(ant_gap_log)
    
    if not baseline_data or not ant_gap_data:
        print("❌ Error: Could not parse log files or no data found.")
        return
    
    print(f"  Baseline: {len(baseline_data)} batches")
    print(f"  ANT+Gap:  {len(ant_gap_data)} batches")
    
    # Aggregate by epoch
    baseline_epochs = aggregate_by_epoch(baseline_data)
    ant_gap_epochs = aggregate_by_epoch(ant_gap_data)
    
    # Create visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Gap Evolution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gap comparison
    ax = axes[0, 0]
    baseline_gaps = [e["current_gap"] for e in baseline_epochs]
    ant_gap_gaps = [e["current_gap"] for e in ant_gap_epochs]
    
    ax.plot(range(len(baseline_gaps)), baseline_gaps, 'b-', label='Baseline', linewidth=2)
    ax.plot(range(len(ant_gap_gaps)), ant_gap_gaps, 'r-', label='ANT+Gap', linewidth=2)
    ax.axhline(y=0.7, color='g', linestyle='--', label='Target (0.7)', alpha=0.5)
    ax.set_xlabel('Epoch (across all tasks)', fontsize=12)
    ax.set_ylabel('Gap (pos - neg)', fontsize=12)
    ax.set_title('Gap Evolution: Baseline vs ANT+Gap', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ANT Loss comparison
    ax = axes[0, 1]
    baseline_ant = [e["ant_loss"] for e in baseline_epochs]
    ant_gap_ant = [e["ant_loss"] for e in ant_gap_epochs]
    
    ax.plot(range(len(baseline_ant)), baseline_ant, 'b-', label='Baseline', linewidth=2)
    ax.plot(range(len(ant_gap_ant)), ant_gap_ant, 'r-', label='ANT+Gap', linewidth=2)
    ax.set_xlabel('Epoch (across all tasks)', fontsize=12)
    ax.set_ylabel('ANT Loss (base)', fontsize=12)
    ax.set_title('ANT Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gap Loss (only for ANT+Gap)
    ax = axes[1, 0]
    ant_gap_gap_loss = [e["gap_loss"] for e in ant_gap_epochs]
    
    ax.plot(range(len(ant_gap_gap_loss)), ant_gap_gap_loss, 'r-', linewidth=2)
    ax.set_xlabel('Epoch (across all tasks)', fontsize=12)
    ax.set_ylabel('Gap Loss', fontsize=12)
    ax.set_title('Gap Maximization Loss (ANT+Gap only)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Violation percentage
    ax = axes[1, 1]
    baseline_viol = [e["violation_pct"] for e in baseline_epochs]
    ant_gap_viol = [e["violation_pct"] for e in ant_gap_epochs]
    
    ax.plot(range(len(baseline_viol)), baseline_viol, 'b-', label='Baseline', linewidth=2)
    ax.plot(range(len(ant_gap_viol)), ant_gap_viol, 'r-', label='ANT+Gap', linewidth=2)
    ax.set_xlabel('Epoch (across all tasks)', fontsize=12)
    ax.set_ylabel('Hard Negative Violation %', fontsize=12)
    ax.set_title('Hard Negatives: Baseline vs ANT+Gap', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_evolution.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_dir / 'comparison_evolution.png'}")
    
    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)
    
    baseline_gaps_all = [d["current_gap"] for d in baseline_data]
    ant_gap_gaps_all = [d["current_gap"] for d in ant_gap_data]
    
    print("\n📊 Gap Statistics:")
    print(f"{'Metric':<20} {'Baseline':<15} {'ANT+Gap':<15} {'Improvement':<15}")
    print("-" * 65)
    
    b_mean = np.mean(baseline_gaps_all)
    a_mean = np.mean(ant_gap_gaps_all)
    print(f"{'Mean':<20} {b_mean:<15.4f} {a_mean:<15.4f} {((a_mean - b_mean) / b_mean * 100):>+7.2f}%")
    
    b_std = np.std(baseline_gaps_all)
    a_std = np.std(ant_gap_gaps_all)
    print(f"{'Std Dev':<20} {b_std:<15.4f} {a_std:<15.4f}")
    
    b_max = np.max(baseline_gaps_all)
    a_max = np.max(ant_gap_gaps_all)
    print(f"{'Max':<20} {b_max:<15.4f} {a_max:<15.4f} {((a_max - b_max) / b_max * 100):>+7.2f}%")
    
    b_final = baseline_gaps[-1] if baseline_gaps else 0
    a_final = ant_gap_gaps[-1] if ant_gap_gaps else 0
    print(f"{'Final (last epoch)':<20} {b_final:<15.4f} {a_final:<15.4f} {((a_final - b_final) / b_final * 100):>+7.2f}%")
    
    # Task-by-task comparison
    print("\n📈 Task-by-Task Gap Progression:")
    print(f"{'Task':<10} {'Baseline Start':<18} {'Baseline End':<18} {'ANT+Gap Start':<18} {'ANT+Gap End':<18}")
    print("-" * 82)
    
    for task in sorted(set(e["task"] for e in baseline_epochs)):
        b_task = [e for e in baseline_epochs if e["task"] == task]
        a_task = [e for e in ant_gap_epochs if e["task"] == task]
        
        if b_task and a_task:
            print(f"Task {task:<5} {b_task[0]['current_gap']:<18.4f} {b_task[-1]['current_gap']:<18.4f} "
                  f"{a_task[0]['current_gap']:<18.4f} {a_task[-1]['current_gap']:<18.4f}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    if a_mean > b_mean:
        improvement = ((a_mean - b_mean) / b_mean * 100)
        print(f"\n✅ ANT+Gap achieves {improvement:.2f}% higher average gap than Baseline")
        print(f"   Baseline: {b_mean:.4f} | ANT+Gap: {a_mean:.4f}")
    else:
        print(f"\n⚠️  Baseline has higher average gap than ANT+Gap")
    
    if a_final >= 0.7 and b_final < 0.7:
        print(f"\n✅ ANT+Gap reaches target gap (0.7), Baseline does not")
        print(f"   Baseline final: {b_final:.4f} | ANT+Gap final: {a_final:.4f}")
    elif a_final >= 0.7:
        print(f"\n✅ ANT+Gap reaches target gap (0.7)")
    
    avg_gap_loss = np.mean([d["gap_loss"] for d in ant_gap_data])
    print(f"\n📊 Average gap_loss contribution in ANT+Gap: {avg_gap_loss:.4f}")
    print(f"   With gap_beta=0.5, contributes ~{0.5 * avg_gap_loss:.4f} to total loss")


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline and ANT+Gap experiments')
    parser.add_argument('--baseline', type=str, 
                       default='logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal_gapT0_gapB0/exp_matrix_debug0.log',
                       help='Path to baseline experiment log')
    parser.add_argument('--ant-gap', type=str,
                       default='logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5/exp_matrix_debug0.log',
                       help='Path to ANT+Gap experiment log')
    parser.add_argument('--output', type=str, default='analysis_results',
                       help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    baseline_log = Path(args.baseline)
    ant_gap_log = Path(args.ant_gap)
    
    if not baseline_log.exists():
        print(f"❌ Baseline log not found: {baseline_log}")
        return
    
    if not ant_gap_log.exists():
        print(f"❌ ANT+Gap log not found: {ant_gap_log}")
        return
    
    compare_experiments(baseline_log, ant_gap_log, args.output)
    
    print("\n✅ Analysis complete!")
    print(f"   Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
