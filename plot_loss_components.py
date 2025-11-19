#!/usr/bin/env python3
"""
Script to plot loss components from matrix debug log file.
Generates line plots showing raw losses, weighted losses, and total loss per epoch.
Separates plots by task and filters out the last batch of each epoch (often smaller).
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_loss_components(log_file):
    """
    Parse loss components from the matrix debug log file.

    Returns:
        dict: Dictionary with lists of loss values for each component, organized by task
    """
    data = {"contrast": {}, "kd": {}}

    # Pattern to match loss component lines with context
    # Example: "[T1 E5 B10] Loss components: contrast_infoNCE_nll: 5.5234 | ..."
    pattern = re.compile(
        r"\[T(\d+) E(\d+) B(\d+)\] Loss components: (\w+)_infoNCE_nll: ([\d.]+) \| "
        r"\4_infoNCE_ant_loss: ([\d.]+) \| "
        r"\4_infoNCE_nce_weighted: ([\d.]+) \| "
        r"\4_infoNCE_ant_weighted: ([\d.]+) \| "
        r"\4_infoNCE_total: ([\d.]+)"
    )

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                task = int(match.group(1))
                epoch = int(match.group(2))
                batch = int(match.group(3))
                prefix = match.group(4)  # 'contrast' or 'kd'
                nll = float(match.group(5))
                ant_loss = float(match.group(6))
                nce_weighted = float(match.group(7))
                ant_weighted = float(match.group(8))
                total = float(match.group(9))

                if prefix in data:
                    # Initialize task data if needed
                    if task not in data[prefix]:
                        data[prefix][task] = {
                            "nll": [],
                            "ant_loss": [],
                            "nce_weighted": [],
                            "ant_weighted": [],
                            "total": [],
                            "epochs": [],
                            "batches": [],
                        }

                    data[prefix][task]["nll"].append(nll)
                    data[prefix][task]["ant_loss"].append(ant_loss)
                    data[prefix][task]["nce_weighted"].append(nce_weighted)
                    data[prefix][task]["ant_weighted"].append(ant_weighted)
                    data[prefix][task]["total"].append(total)
                    data[prefix][task]["epochs"].append(epoch)
                    data[prefix][task]["batches"].append(batch)

    return data


def aggregate_by_epoch(task_data, filter_last_batch=True):
    """
    Aggregate loss data by epoch (mean across batches).

    Args:
        task_data: Dictionary with batch-level data
        filter_last_batch: If True, exclude the last batch of each epoch (often smaller)

    Returns:
        Dictionary with epoch-level aggregated data
    """
    epoch_data = {}

    # Group by epoch
    unique_epochs = sorted(set(task_data["epochs"]))

    for epoch in unique_epochs:
        # Get all data for this epoch
        epoch_mask = np.array(task_data["epochs"]) == epoch
        epoch_batches = np.array(task_data["batches"])[epoch_mask]

        # Find indices for this epoch
        indices = np.where(epoch_mask)[0]

        # Optionally filter out the last batch (often smaller)
        if filter_last_batch and len(indices) > 1:
            max_batch = epoch_batches.max()
            last_batch_idx = np.where(epoch_batches == max_batch)[0]
            if len(last_batch_idx) > 0:
                # Remove the last batch index
                indices = indices[epoch_batches != max_batch]

        if len(indices) == 0:
            continue

        # Calculate means for this epoch
        epoch_data[epoch] = {
            "nll": np.mean([task_data["nll"][i] for i in indices]),
            "ant_loss": np.mean([task_data["ant_loss"][i] for i in indices]),
            "nce_weighted": np.mean([task_data["nce_weighted"][i] for i in indices]),
            "ant_weighted": np.mean([task_data["ant_weighted"][i] for i in indices]),
            "total": np.mean([task_data["total"][i] for i in indices]),
            "num_batches": len(indices),
        }

    return epoch_data


def plot_loss_components_by_task(data, output_dir, prefix="contrast"):
    """
    Create plots for loss components, one plot per task.
    Plots loss values averaged per epoch instead of per batch.

    Args:
        data: Dictionary with loss component data organized by task
        output_dir: Directory to save plots
        prefix: 'contrast' or 'kd'
    """
    if not data[prefix]:
        print(f"No data found for {prefix} losses")
        return

    tasks = sorted(data[prefix].keys())

    for task in tasks:
        task_data = data[prefix][task]

        # Aggregate by epoch (filtering last batch)
        epoch_data = aggregate_by_epoch(task_data, filter_last_batch=True)

        if not epoch_data:
            print(f"No epoch data for task {task}")
            continue

        epochs = sorted(epoch_data.keys())
        epoch_nums = np.array(epochs)

        # Extract aggregated values
        nll = np.array([epoch_data[e]["nll"] for e in epochs])
        ant_loss = np.array([epoch_data[e]["ant_loss"] for e in epochs])
        nce_weighted = np.array([epoch_data[e]["nce_weighted"] for e in epochs])
        ant_weighted = np.array([epoch_data[e]["ant_weighted"] for e in epochs])
        total = np.array([epoch_data[e]["total"] for e in epochs])

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(
            f"Task {task} - {prefix.upper()} Loss Components (Per Epoch)",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Raw losses (NLL and ANT)
        ax1 = axes[0]
        ax1.plot(
            epoch_nums,
            nll,
            label="NLL (raw)",
            linewidth=2,
            marker="o",
            markersize=5,
            alpha=0.7,
        )
        ax1.plot(
            epoch_nums,
            ant_loss,
            label="ANT Loss (raw)",
            linewidth=2,
            marker="s",
            markersize=5,
            alpha=0.7,
        )
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss Value", fontsize=11)
        ax1.set_title("Raw Loss Components", fontsize=13, fontweight="bold")
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Weighted losses
        ax2 = axes[1]
        ax2.plot(
            epoch_nums,
            nce_weighted,
            label="NLL (weighted)",
            linewidth=2,
            marker="o",
            markersize=5,
            alpha=0.7,
            color="tab:blue",
        )
        ax2.plot(
            epoch_nums,
            ant_weighted,
            label="ANT Loss (weighted)",
            linewidth=2,
            marker="s",
            markersize=5,
            alpha=0.7,
            color="tab:orange",
        )
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Loss Value", fontsize=11)
        ax2.set_title("Weighted Loss Components", fontsize=13, fontweight="bold")
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Total loss and components stacked
        ax3 = axes[2]
        ax3.plot(
            epoch_nums,
            total,
            label="Total Loss",
            linewidth=2.5,
            marker="d",
            markersize=6,
            alpha=0.8,
            color="tab:red",
        )
        ax3.plot(
            epoch_nums,
            nce_weighted,
            label="NLL (weighted)",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )
        ax3.plot(
            epoch_nums,
            ant_weighted,
            label="ANT Loss (weighted)",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
        )
        ax3.set_xlabel("Epoch", fontsize=11)
        ax3.set_ylabel("Loss Value", fontsize=11)
        ax3.set_title("Total Loss vs Components", fontsize=13, fontweight="bold")
        ax3.legend(loc="best", fontsize=10)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = output_dir / f"{prefix}_loss_task{task}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_file}")
        plt.close()


def plot_all_tasks_combined(data, output_dir, prefix="contrast"):
    """
    Create a plot showing all tasks combined (per epoch).

    Args:
        data: Dictionary with loss component data organized by task
        output_dir: Directory to save plots
        prefix: 'contrast' or 'kd'
    """
    if not data[prefix]:
        print(f"No data found for {prefix} losses")
        return

    tasks = sorted(data[prefix].keys())

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{prefix.upper()} Loss Components - All Tasks (Per Epoch)",
        fontsize=16,
        fontweight="bold",
    )

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Plot each component type
    components = [
        ("nll", "NLL (raw)"),
        ("ant_loss", "ANT Loss (raw)"),
        ("nce_weighted", "NLL (weighted)"),
        ("ant_weighted", "ANT Loss (weighted)"),
    ]

    for idx, (comp_name, comp_label) in enumerate(components):
        ax = axes[idx]

        for task in tasks:
            task_data = data[prefix][task]
            epoch_data = aggregate_by_epoch(task_data, filter_last_batch=True)

            if not epoch_data:
                continue

            epochs = sorted(epoch_data.keys())
            values = [epoch_data[e][comp_name] for e in epochs]

            ax.plot(
                epochs,
                values,
                label=f"Task {task}",
                linewidth=2,
                marker="o",
                markersize=4,
                alpha=0.7,
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss Value", fontsize=11)
        ax.set_title(comp_label, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"{prefix}_loss_all_tasks_combined.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved combined plot to {output_file}")
    plt.close()

    # Create total loss comparison across tasks
    fig, ax = plt.subplots(figsize=(14, 8))

    for task in tasks:
        task_data = data[prefix][task]
        epoch_data = aggregate_by_epoch(task_data, filter_last_batch=True)

        if not epoch_data:
            continue

        epochs = sorted(epoch_data.keys())
        total_values = [epoch_data[e]["total"] for e in epochs]

        ax.plot(
            epochs,
            total_values,
            label=f"Task {task}",
            linewidth=2.5,
            marker="d",
            markersize=5,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title(
        f"{prefix.upper()} Total Loss - All Tasks (Per Epoch)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{prefix}_total_loss_all_tasks.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved total loss comparison to {output_file}")
    plt.close()


def print_statistics(data, prefix="contrast"):
    """Print statistics about the loss components for each task (per epoch averages)."""
    if not data[prefix]:
        return

    tasks = sorted(data[prefix].keys())

    for task in tasks:
        task_data = data[prefix][task]
        epoch_data = aggregate_by_epoch(task_data, filter_last_batch=True)

        if not epoch_data:
            continue

        epochs = sorted(epoch_data.keys())
        num_epochs = len(epochs)

        print(f"\n{'='*70}")
        print(f"Statistics for {prefix.upper()} losses - Task {task}:")
        print(f"{'='*70}")
        print(f"Number of epochs: {num_epochs}")
        print(
            f"Batches per epoch (excluding last): ~{epoch_data[epochs[0]]['num_batches']}"
        )
        print(f"\n{'Component':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"{'-'*70}")

        components = ["nll", "ant_loss", "nce_weighted", "ant_weighted", "total"]
        labels = [
            "NLL (raw)",
            "ANT Loss (raw)",
            "NLL (weighted)",
            "ANT Loss (weighted)",
            "Total Loss",
        ]

        for comp, label in zip(components, labels):
            values = np.array([epoch_data[e][comp] for e in epochs])
            print(
                f"{label:<25} {values.mean():>10.4f} {values.std():>10.4f} {values.min():>10.4f} {values.max():>10.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss components from matrix debug log file (per epoch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the matrix debug log file (e.g., exp_debug0.log)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as log file)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["contrast", "kd", "both"],
        default="both",
        help="Type of losses to plot",
    )

    args = parser.parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_file.parent / "loss_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Parse log file
    print(f"Parsing log file: {log_file}")
    data = parse_loss_components(log_file)

    # Plot losses
    if args.type in ["contrast", "both"]:
        plot_loss_components_by_task(data, output_dir, prefix="contrast")
        plot_all_tasks_combined(data, output_dir, prefix="contrast")
        print_statistics(data, prefix="contrast")

    if args.type in ["kd", "both"]:
        plot_loss_components_by_task(data, output_dir, prefix="kd")
        plot_all_tasks_combined(data, output_dir, prefix="kd")
        print_statistics(data, prefix="kd")

    print(f"\n{'='*70}")
    print("Done! Plots saved to:", output_dir)
    print("Note: Last batch of each epoch is filtered out (often smaller)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
