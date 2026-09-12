#!/usr/bin/env python3
"""
Comparative analysis of debug loss/ANT metrics across methods and datasets.

Coverage (3 seeds × 4 datasets × 3 methods):
  Methods : β=0 aGlobal nGlobal | β=0.5 aLocal nLocal | β=0.5 aSymFull nLocal
  Datasets: cifar100_10-10 | cifar100_50-10 | tiny_imagenet_20-20 | tiny_imagenet_100-20

Outputs: figures in analysis/results/debug_metrics/
"""

import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOGS_DIR   = REPO_ROOT / "logs"
OUT_DIR    = REPO_ROOT / "analysis" / "results" / "debug_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

METHODS = {
    "antB0_nceA1_antGlobal_nceGlobal":          "β=0 Baseline",
    "antB0.5_nceA1_antM0.5_antLocal_nceLocal":  "β=0.5 Local/Local",
    "antB0.5_nceA1_antM0.5_antSymmetricFull_nceLocal": "β=0.5 SymFull/Local",
}

DATASETS = {
    "cifar100_10-10":      "CIFAR-100 10-10",
    "cifar100_50-10":      "CIFAR-100 50-10",
    "tiny_imagenet_20-20": "TIN 20-20",
    "tiny_imagenet_100-20":"TIN 100-20",
}

SEEDS = [1993, 1994, 1995]

COLORS = {
    "β=0 Baseline":        "#4477aa",
    "β=0.5 Local/Local":   "#ee6677",
    "β=0.5 SymFull/Local": "#228833",
}

# ── Log parsing ────────────────────────────────────────────────────────────────

_RE_ANT = re.compile(
    r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats:.*?"
    r"gap_mean: ([-\d.]+).*?"
    r"violation_pct: ([\d.]+)%.*?"
    r"ant_loss: ([\d.]+)"
)
_RE_LOSS = re.compile(
    r"\[T(\d+) E(\d+) B(\d+)\] Loss components: "
    r"contrast_nll: ([\d.]+) \| "
    r"contrast_ant_loss: ([\d.]+) \| "
    r"contrast_nce_weighted: ([\d.]+) \| "
    r"contrast_ant_weighted: ([\d.]+) \| "
    r"contrast_total: ([\d.]+)"
)


def parse_debug(log_path: Path) -> dict:
    """Return dict: (task, epoch) → {metric: list_of_batch_values}."""
    rows: dict = defaultdict(lambda: defaultdict(list))

    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = _RE_ANT.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                rows[key]["gap_mean"].append(float(m.group(4)))
                rows[key]["violation_pct"].append(float(m.group(5)))
                rows[key]["ant_loss"].append(float(m.group(6)))
                continue
            m = _RE_LOSS.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                rows[key]["loss_nll"].append(float(m.group(4)))
                rows[key]["loss_ant_loss"].append(float(m.group(5)))
                rows[key]["loss_nce_weighted"].append(float(m.group(6)))
                rows[key]["loss_ant_weighted"].append(float(m.group(7)))
                rows[key]["loss_total"].append(float(m.group(8)))

    # Average over batches
    result: dict = {}
    for (task, epoch), metrics in rows.items():
        result[(task, epoch)] = {k: float(np.mean(v)) for k, v in metrics.items()}
    return result


def load_experiment(dataset: str, config: str, seeds: list) -> dict:
    """Load and aggregate debug data across seeds for one (dataset, config).

    Returns: seed_data[seed] = {(task,epoch): {metric: value}}
    """
    seed_data = {}
    for seed in seeds:
        name = f"debug_exp_{dataset}_{config}_s{seed}"
        log = LOGS_DIR / name / "exp_debug0.log"
        if not log.exists():
            continue
        seed_data[seed] = parse_debug(log)
    return seed_data


def aggregate_seeds(seed_data: dict) -> tuple:
    """Return (mean_dict, std_dict) aggregated across seeds, keyed by (task,epoch)."""
    if not seed_data:
        return {}, {}

    all_keys = set()
    for d in seed_data.values():
        all_keys.update(d.keys())

    mean_d, std_d = {}, {}
    for key in sorted(all_keys):
        metric_vals: dict = defaultdict(list)
        for d in seed_data.values():
            if key in d:
                for m, v in d[key].items():
                    metric_vals[m].append(v)
        mean_d[key] = {m: float(np.mean(vs)) for m, vs in metric_vals.items()}
        std_d[key]  = {m: float(np.std(vs, ddof=0)) for m, vs in metric_vals.items()}
    return mean_d, std_d


def to_task_epoch_series(agg: dict, metric: str):
    """Flatten (task,epoch) dict to sorted arrays of (x_index, mean, std)."""
    keys   = sorted(agg[0].keys())
    x      = list(range(len(keys)))
    mean   = [agg[0].get(k, {}).get(metric, np.nan) for k in keys]
    std    = [agg[1].get(k, {}).get(metric, np.nan) for k in keys]
    # task boundary indices
    tasks  = [k[0] for k in keys]
    bounds = [i for i in range(1, len(tasks)) if tasks[i] != tasks[i-1]]
    return np.array(x), np.array(mean), np.array(std), bounds


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_metrics_per_dataset(dataset: str, ds_label: str):
    """4-panel figure per dataset: violation_pct, gap_mean, loss_nll, loss_total."""
    metrics = [
        ("violation_pct",   "Violation % ↓",   True),
        ("gap_mean",         "Gap mean ↑",       False),
        ("loss_nll",         "InfoNCE loss ↓",   False),
        ("loss_total",       "Total loss ↓",     False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    fig.suptitle(f"Debug metrics — {ds_label}", fontsize=13, fontweight="bold")

    for ax_i, (metric, ylabel, invert) in enumerate(metrics):
        ax = axes[ax_i]
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Epoch index (across tasks)", fontsize=8)

        for config_key, method_label in METHODS.items():
            seed_data = load_experiment(dataset, config_key, SEEDS)
            if not seed_data:
                continue
            agg = aggregate_seeds(seed_data)
            x, mean, std, bounds = to_task_epoch_series(agg, metric)

            valid = ~np.isnan(mean)
            if not valid.any():
                continue

            color = COLORS[method_label]
            ax.plot(x[valid], mean[valid], label=method_label, color=color, lw=1.6)
            ax.fill_between(x[valid], mean[valid] - std[valid],
                            mean[valid] + std[valid], alpha=0.15, color=color)

            # Task boundary markers
            for b in bounds:
                ax.axvline(b, color="gray", lw=0.5, ls="--", alpha=0.4)

        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(axis="y", lw=0.4, alpha=0.4)

    plt.tight_layout()
    path = OUT_DIR / f"{dataset}_debug_metrics.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_cross_dataset_metric(metric: str, ylabel: str):
    """1×N figure comparing metric across all 4 datasets."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    fig.suptitle(f"{ylabel} — all datasets", fontsize=12, fontweight="bold")

    for ax, (ds_key, ds_label) in zip(axes, DATASETS.items()):
        ax.set_title(ds_label, fontsize=9)
        ax.set_xlabel("Epoch index", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(ylabel, fontsize=9)

        for config_key, method_label in METHODS.items():
            seed_data = load_experiment(ds_key, config_key, SEEDS)
            if not seed_data:
                continue
            agg = aggregate_seeds(seed_data)
            x, mean, std, bounds = to_task_epoch_series(agg, metric)
            valid = ~np.isnan(mean)
            if not valid.any():
                continue
            color = COLORS[method_label]
            ax.plot(x[valid], mean[valid], label=method_label, color=color, lw=1.4)
            ax.fill_between(x[valid], mean[valid]-std[valid],
                            mean[valid]+std[valid], alpha=0.12, color=color)
            for b in bounds:
                ax.axvline(b, color="gray", lw=0.5, ls="--", alpha=0.35)

        ax.legend(fontsize=6.5, framealpha=0.7)
        ax.grid(axis="y", lw=0.4, alpha=0.4)

    plt.tight_layout()
    path = OUT_DIR / f"cross_dataset_{metric}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_ant_contribution(dataset: str, ds_label: str):
    """Show proportion of ANT loss in total loss per epoch for ANT methods."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"ANT weighted proportion of total loss — {ds_label}", fontsize=10)
    ax.set_ylabel("ant_weighted / total (%)", fontsize=9)
    ax.set_xlabel("Epoch index", fontsize=8)

    for config_key, method_label in METHODS.items():
        if "Baseline" in method_label:
            continue
        seed_data = load_experiment(dataset, config_key, SEEDS)
        if not seed_data:
            continue
        agg = aggregate_seeds(seed_data)
        x, mean_ant, _, bounds = to_task_epoch_series(agg, "loss_ant_weighted")
        _, mean_tot, _, _       = to_task_epoch_series(agg, "loss_total")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio = 100 * mean_ant / mean_tot
        valid = ~np.isnan(ratio)
        if not valid.any():
            continue
        color = COLORS[method_label]
        ax.plot(x[valid], ratio[valid], label=method_label, color=color, lw=1.6)
        for b in bounds:
            ax.axvline(b, color="gray", lw=0.5, ls="--", alpha=0.4)

    ax.legend(fontsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    plt.tight_layout()
    path = OUT_DIR / f"{dataset}_ant_proportion.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Output directory: {OUT_DIR}\n")

    # Per-dataset 4-panel figures
    print("=== Per-dataset figures ===")
    for ds_key, ds_label in DATASETS.items():
        print(f"  {ds_label}...")
        fig_metrics_per_dataset(ds_key, ds_label)
        fig_ant_contribution(ds_key, ds_label)

    # Cross-dataset single-metric figures
    print("\n=== Cross-dataset figures ===")
    cross_metrics = [
        ("violation_pct", "Violation % ↓"),
        ("gap_mean",      "Gap mean ↑"),
        ("loss_nll",      "InfoNCE loss ↓"),
        ("loss_total",    "Total loss ↓"),
    ]
    for metric, ylabel in cross_metrics:
        fig_cross_dataset_metric(metric, ylabel)

    print(f"\nAll outputs saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
