#!/usr/bin/env python3
"""
Extract final accuracy metrics from experiment logs
"""
import re
import os
from pathlib import Path


def extract_metrics_from_log(log_path):
    """Extract final metrics from a gist log file"""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        # Look for the last task metrics (handles both 10/10 and 6/6 splits)
        metrics = {}

        # Find the last R0T line to get the final task
        final_task_lines = []
        last_task_pattern = None

        # Search backwards to find the last task
        for i in range(len(lines) - 1, -1, -1):
            if re.search(
                r"R0T\[(\d+)/\1\]", lines[i]
            ):  # Matches R0T[X/X] where both numbers are the same
                final_task_lines = lines[i : min(i + 10, len(lines))]
                break

        if not final_task_lines:
            return {}

        # Extract from the final task lines
        for line in final_task_lines:
            # Extract avg_acc1
            match = re.search(r"avg_acc1\s+([\d.]+)", line)
            if match:
                metrics["avg_acc1"] = float(match.group(1))

            # Extract avg_acc5
            match = re.search(r"avg_acc5\s+([\d.]+)", line)
            if match:
                metrics["avg_acc5"] = float(match.group(1))

            # Extract avg_nme1
            match = re.search(r"avg_nme1\s+([\d.]+)", line)
            if match:
                metrics["avg_nme1"] = float(match.group(1))

            # Extract avg_nme5
            match = re.search(r"avg_nme5\s+([\d.]+)", line)
            if match:
                metrics["avg_nme5"] = float(match.group(1))

            # Extract eval_acc1 (last task accuracy)
            match = re.search(r"eval_acc1\s+([\d.]+)", line)
            if match:
                metrics["last_acc1"] = float(match.group(1))

            # Extract eval_nme1 (last task NME)
            match = re.search(r"eval_nme1\s+([\d.]+)", line)
            if match:
                metrics["last_nme1"] = float(match.group(1))

            # Extract acc1_curve
            if "acc1_curve [" in line:
                match = re.search(r"acc1_curve \[([\d\.\s]+)\]", line)
                if match:
                    curve_values = [float(x) for x in match.group(1).split()]
                    metrics["acc1_curve"] = curve_values
                    metrics["last_acc1_from_curve"] = curve_values[-1]
                    metrics["num_tasks"] = len(curve_values)

            # Extract nme1_curve
            if "nme1_curve [" in line:
                match = re.search(r"nme1_curve \[([\d\.\s]+)\]", line)
                if match:
                    curve_values = [float(x) for x in match.group(1).split()]
                    metrics["nme1_curve"] = curve_values
                    metrics["last_nme1_from_curve"] = curve_values[-1]

        return metrics
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return {}
        return {}


# Define experiments to analyze
experiments = {
    "CIFAR-100 10-10": [
        ("done_exp_cifar100_10-10", "Baseline Original"),
        (
            "done_exp_cifar100_10-10_baseline_tagfex_original",
            "Baseline TagFex Original",
        ),
        ("done_exp_cifar100_10-10_infonce_local_anchor", "InfoNCE Local Anchor"),
        (
            "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.1_antGlobal",
            "ANT β=0.5, m=0.1, Global",
        ),
        (
            "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.1_antLocal",
            "ANT β=0.5, m=0.1, Local",
        ),
        (
            "done_exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5",
            "ANT β=1.0, m=0.1, Local, Gap",
        ),
        (
            "done_exp_cifar100_10-10_antB1_nceA1_antM0.3_antLocal",
            "ANT β=1.0, m=0.3, Local",
        ),
        (
            "done_exp_cifar100_10-10_antB1_nceA1_antM0.5_antLocal",
            "ANT β=1.0, m=0.5, Local",
        ),
        (
            "exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal",
            "ANT β=0.5, m=0.5, Local [NEW]",
        ),
        (
            "exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal",
            "ANT β=1.0, m=0.1, Local [NEW]",
        ),
    ],
    "CIFAR-100 50-10": [
        ("done_exp_cifar100_50-10_antB0_nceA1_antGlobal", "Baseline Global (β=0)"),
        ("done_exp_cifar100_50-10_antB0_nceA1_antLocal", "Baseline Local (β=0)"),
        (
            "exp_cifar100_50-10_antB1_nceA1_antM0.5_antLocal",
            "ANT β=1.0, m=0.5, Local [NEW]",
        ),
    ],
    "ImageNet-100 10-10": [
        ("done_exp_imagenet100_10-10_antB0_nceA1_antLocal", "Baseline Local (β=0)"),
    ],
}

logs_dir = Path("/home/tiago/TagFex_CVPR2025/logs")

print("=" * 80)
print("FINAL ACCURACY METRICS EXTRACTION")
print("=" * 80)
print()

for category, exp_list in experiments.items():
    print(f"\n{'=' * 80}")
    print(f"{category}")
    print(f"{'=' * 80}\n")

    for exp_dir, description in exp_list:
        log_path = logs_dir / exp_dir / "exp_gistlog.log"

        if not log_path.exists():
            print(f"❌ {description}")
            print(f"   Directory: {exp_dir}")
            print(f"   Status: Log file not found")
            print()
            continue

        metrics = extract_metrics_from_log(log_path)

        if not metrics:
            print(f"❌ {description}")
            print(f"   Directory: {exp_dir}")
            print(f"   Status: Could not extract metrics")
            print()
            continue

        print(f"✓ {description}")
        print(f"  Directory: {exp_dir}")

        avg_acc1 = metrics.get("avg_acc1")
        last_acc1 = metrics.get("last_acc1")
        avg_nme1 = metrics.get("avg_nme1")
        last_nme1 = metrics.get("last_nme1")

        print(
            f"  Final Avg Acc@1:  {avg_acc1:.2f}%"
            if avg_acc1 is not None
            else "  Final Avg Acc@1:  N/A"
        )
        print(
            f"  Final Last Acc@1: {last_acc1:.2f}%"
            if last_acc1 is not None
            else "  Final Last Acc@1: N/A"
        )
        print(
            f"  Final Avg NME@1:  {avg_nme1:.2f}%"
            if avg_nme1 is not None
            else "  Final Avg NME@1:  N/A"
        )
        print(
            f"  Final Last NME@1: {last_nme1:.2f}%"
            if last_nme1 is not None
            else "  Final Last NME@1: N/A"
        )

        if "acc1_curve" in metrics:
            curve_str = " ".join([f"{v:.2f}" for v in metrics["acc1_curve"]])
            print(f"  Acc@1 Curve: [{curve_str}]")

        print()

print("\n" + "=" * 80)
print("SUMMARY TABLE - CIFAR-100 10-10")
print("=" * 80)
print(f"{'Experiment':<45} {'Avg Acc@1':<12} {'Last Acc@1':<12} {'Avg NME@1':<12}")
print("-" * 80)

for exp_dir, description in experiments["CIFAR-100 10-10"]:
    log_path = logs_dir / exp_dir / "exp_gistlog.log"
    if log_path.exists():
        metrics = extract_metrics_from_log(log_path)
        if metrics:
            avg_acc = metrics.get("avg_acc1", 0)
            last_acc = metrics.get("last_acc1", 0)
            avg_nme = metrics.get("avg_nme1", 0)
            print(
                f"{description:<45} {avg_acc:>10.2f}%  {last_acc:>10.2f}%  {avg_nme:>10.2f}%"
            )

print("\n" + "=" * 80)
print("SUMMARY TABLE - CIFAR-100 50-10")
print("=" * 80)
print(f"{'Experiment':<45} {'Avg Acc@1':<12} {'Last Acc@1':<12} {'Avg NME@1':<12}")
print("-" * 80)

for exp_dir, description in experiments["CIFAR-100 50-10"]:
    log_path = logs_dir / exp_dir / "exp_gistlog.log"
    if log_path.exists():
        metrics = extract_metrics_from_log(log_path)
        if metrics:
            avg_acc = metrics.get("avg_acc1", 0)
            last_acc = metrics.get("last_acc1", 0)
            avg_nme = metrics.get("avg_nme1", 0)
            print(
                f"{description:<45} {avg_acc:>10.2f}%  {last_acc:>10.2f}%  {avg_nme:>10.2f}%"
            )

print("\n" + "=" * 80)
print("SUMMARY TABLE - ImageNet-100 10-10")
print("=" * 80)
print(f"{'Experiment':<45} {'Avg Acc@1':<12} {'Last Acc@1':<12} {'Avg NME@1':<12}")
print("-" * 80)

for exp_dir, description in experiments["ImageNet-100 10-10"]:
    log_path = logs_dir / exp_dir / "exp_gistlog.log"
    if log_path.exists():
        metrics = extract_metrics_from_log(log_path)
        if metrics:
            avg_acc = metrics.get("avg_acc1", 0)
            last_acc = metrics.get("last_acc1", 0)
            avg_nme = metrics.get("avg_nme1", 0)
            print(
                f"{description:<45} {avg_acc:>10.2f}%  {last_acc:>10.2f}%  {avg_nme:>10.2f}%"
            )

print("\n")
