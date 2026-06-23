"""
Results charts — TagFex paper style.

Generates one PNG per dataset, each with two side-by-side subplots:
  left  → Acc1 curve per incremental task
  right → NME1 curve per incremental task

Multi-seed experiments (3 seeds) are shown as solid lines.
Single-seed experiments are shown as dashed lines.
Delta annotations (vs baseline) are shown for the best non-baseline series.

Usage:
    python charts_results.py          # saves all PNGs in figure/
    python charts_results.py --show   # also opens interactive windows
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Visual style (aligned with chart.py) ─────────────────────────────────────

SERIES_STYLE = {
    # 3-seed experiments
    "β=0 aGlobal nGlobal":   dict(color="tab:blue",   marker="o", markersize=7,  linestyle="-",  label="β=0 aGlobal nGlobal (baseline)"),
    "β=0.5 aSymFull nLocal": dict(color="tab:red",    marker="*", markersize=10, linestyle="-",  label="β=0.5 aSymFull nLocal"),
    "β=0.5 aLocal nLocal":   dict(color="tab:purple", marker="p", markersize=9,  linestyle="-",  label="β=0.5 aLocal nLocal"),
    "β=0 aLocal nLocal":     dict(color="tab:green",  marker="s", markersize=7,  linestyle="-",  label="β=0 aLocal nLocal"),
    # Single-seed experiments (dashed, no band)
    "β=0.5 aGlobal nGlobal": dict(color="tab:orange", marker="^", markersize=7,  linestyle="--", label="β=0.5 aGlobal nGlobal (s1993)"),
    "β=0.5 aGlobal nLocal":  dict(color="tab:brown",  marker="D", markersize=7,  linestyle="--", label="β=0.5 aGlobal nLocal (s1993)"),
    "β=0.5 aLocal nGlobal":  dict(color="tab:cyan",   marker="v", markersize=7,  linestyle="--", label="β=0.5 aLocal nGlobal (s1993)"),
}

# ── All data (from results_report.md) ────────────────────────────────────────
# Structure: { dataset_key: { metric: { exp_name: (mean_list, std_list_or_None) } } }

DATA = {

    # ── CIFAR-100 10×10 ──────────────────────────────────────────────────────
    "cifar100_10x10": {
        "classes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "title":   "CIFAR-100 10×10",
        "acc1": {
            "β=0 aGlobal nGlobal":   ([92.50, 85.27, 83.83, 81.09, 78.56, 77.08, 75.75, 73.07, 70.75, 69.67],
                                      [0.83,  0.22,  0.67,  0.47,  0.87,  0.83,  0.67,  0.85,  0.75,  0.35]),
            "β=0.5 aSymFull nLocal": ([92.50, 85.65, 84.00, 80.92, 79.09, 77.40, 75.81, 73.54, 71.30, 70.18],
                                      [0.83,  0.57,  0.48,  0.55,  0.60,  0.62,  0.47,  0.52,  0.59,  0.58]),
            "β=0.5 aLocal nLocal":   ([92.50, 85.72, 84.29, 81.45, 79.17, 77.30, 75.96, 73.32, 71.24, 70.58],
                                      [0.83,  0.34,  0.80,  0.66,  0.34,  0.36,  0.33,  0.21,  0.40,  0.44]),
            "β=0 aLocal nLocal":     ([92.50, 85.10, 83.61, 81.28, 79.02, 77.38, 75.64, 73.44, 71.18, 70.19],
                                      [0.83,  0.50,  0.74,  0.77,  0.88,  0.84,  0.56,  0.52,  0.43,  0.11]),
            "β=0.5 aGlobal nGlobal": ([91.40, 84.90, 83.47, 80.65, 78.22, 76.13, 75.29, 72.29, 70.49, 69.62], None),
            "β=0.5 aGlobal nLocal":  ([91.40, 84.70, 82.90, 80.05, 77.68, 76.13, 74.91, 72.85, 70.61, 69.52], None),
            "β=0.5 aLocal nGlobal":  ([91.40, 84.55, 82.77, 80.35, 77.98, 76.52, 75.09, 72.30, 70.24, 69.32], None),
        },
        "nme1": {
            "β=0 aGlobal nGlobal":   ([92.20, 85.38, 82.76, 78.55, 75.53, 73.27, 71.14, 67.56, 65.16, 63.88],
                                      [0.86,  0.84,  0.47,  0.44,  0.28,  0.31,  0.48,  0.58,  0.53,  0.42]),
            "β=0.5 aSymFull nLocal": ([92.20, 85.20, 82.72, 79.08, 76.20, 73.44, 71.00, 67.92, 65.45, 64.16],
                                      [0.86,  0.74,  0.77,  0.90,  0.68,  0.76,  0.53,  0.52,  0.68,  0.66]),
            "β=0.5 aLocal nLocal":   ([92.20, 85.55, 82.81, 78.92, 76.03, 73.36, 70.99, 67.66, 65.31, 64.11],
                                      [0.86,  0.78,  0.27,  0.24,  0.26,  0.30,  0.14,  0.39,  0.30,  0.53]),
            "β=0 aLocal nLocal":     ([92.20, 85.02, 82.70, 79.02, 75.83, 73.25, 71.18, 67.75, 65.29, 64.03],
                                      [0.86,  0.70,  0.64,  0.52,  0.42,  0.63,  0.37,  0.33,  0.50,  0.45]),
            "β=0.5 aGlobal nGlobal": ([91.00, 84.20, 82.60, 78.10, 75.34, 72.32, 70.27, 67.39, 64.63, 63.50], None),
            "β=0.5 aGlobal nLocal":  ([91.00, 84.05, 82.17, 77.72, 75.86, 72.17, 70.21, 66.58, 64.57, 63.13], None),
            "β=0.5 aLocal nGlobal":  ([91.00, 84.60, 82.07, 78.35, 75.10, 72.38, 70.19, 67.16, 64.36, 63.20], None),
        },
    },

    # ── CIFAR-100 50+10×5 ────────────────────────────────────────────────────
    "cifar100_50-10x5": {
        "classes": [50, 60, 70, 80, 90, 100],
        "title":   "CIFAR-100 50+10×5",
        "acc1": {
            "β=0 aGlobal nGlobal":   ([84.03, 80.16, 77.91, 75.25, 72.95, 71.63],
                                      [0.32,  0.78,  0.76,  0.49,  0.65,  0.30]),
            "β=0.5 aSymFull nLocal": ([84.03, 80.47, 78.26, 75.37, 73.08, 71.83],
                                      [0.32,  0.30,  0.60,  0.04,  0.14,  0.20]),
            "β=0.5 aLocal nLocal":   ([84.03, 80.22, 78.12, 75.27, 72.93, 71.45],
                                      [0.32,  0.39,  0.58,  0.49,  0.27,  0.34]),
            "β=0 aLocal nLocal":     ([84.03, 80.19, 77.99, 75.10, 72.86, 71.43],
                                      [0.32,  0.48,  0.66,  0.34,  0.29,  0.35]),
            "β=0.5 aGlobal nGlobal": ([84.24, 79.97, 77.56, 74.81, 72.47, 71.22], None),
            "β=0.5 aLocal nGlobal":  ([84.24, 79.73, 77.59, 74.50, 71.99, 70.61], None),
        },
        "nme1": {
            "β=0 aGlobal nGlobal":   ([84.23, 80.79, 78.49, 74.38, 71.60, 69.65],
                                      [0.31,  0.41,  0.23,  0.39,  0.55,  0.51]),
            "β=0.5 aSymFull nLocal": ([84.23, 81.14, 78.48, 74.44, 71.81, 70.12],
                                      [0.31,  0.29,  0.32,  0.32,  0.45,  0.51]),
            "β=0.5 aLocal nLocal":   ([84.23, 80.99, 78.47, 74.50, 71.61, 69.92],
                                      [0.31,  0.25,  0.26,  0.51,  0.54,  0.54]),
            "β=0 aLocal nLocal":     ([84.23, 81.00, 78.56, 74.50, 71.63, 69.71],
                                      [0.31,  0.27,  0.42,  0.62,  0.71,  0.61]),
            "β=0.5 aGlobal nGlobal": ([84.38, 80.55, 78.39, 74.03, 71.46, 69.44], None),
            "β=0.5 aLocal nGlobal":  ([84.38, 80.37, 78.40, 74.16, 71.17, 68.91], None),
        },
    },

    # ── Tiny-ImageNet 100+20×5 ───────────────────────────────────────────────
    "tinyimagenet_100-20x5": {
        "classes": [100, 120, 140, 160, 180, 200],
        "title":   "Tiny-ImageNet 100+20×5",
        "acc1": {
            "β=0 aGlobal nGlobal":   ([67.85, 62.33, 60.03, 58.30, 56.12, 53.35],
                                      [0.50,  0.95,  0.26,  0.52,  0.42,  0.29]),
            "β=0.5 aSymFull nLocal": ([68.38, 64.80, 62.37, 60.27, 57.47, 55.05],
                                      [0.27,  0.26,  0.32,  0.38,  0.23,  0.20]),
            "β=0.5 aLocal nLocal":   ([67.79, 64.26, 61.99, 60.25, 57.32, 54.81],
                                      [0.43,  0.50,  0.39,  0.45,  0.37,  0.13]),
            "β=0 aLocal nLocal":     ([67.85, 62.56, 60.35, 58.07, 55.73, 53.55],
                                      [0.50,  0.65,  0.44,  0.43,  0.43,  0.34]),
        },
        "nme1": {
            "β=0 aGlobal nGlobal":   ([67.82, 63.38, 61.42, 58.58, 54.86, 51.55],
                                      [0.34,  0.29,  0.43,  0.25,  0.41,  0.26]),
            "β=0.5 aSymFull nLocal": ([68.18, 64.77, 62.56, 59.56, 55.37, 51.54],
                                      [0.26,  0.19,  0.28,  0.19,  0.15,  0.15]),
            "β=0.5 aLocal nLocal":   ([67.74, 64.66, 62.51, 59.36, 55.22, 51.34],
                                      [0.35,  0.35,  0.28,  0.22,  0.07,  0.02]),
            "β=0 aLocal nLocal":     ([67.82, 63.58, 61.59, 58.51, 54.81, 51.67],
                                      [0.34,  0.26,  0.32,  0.20,  0.02,  0.14]),
        },
    },

    # ── Tiny-ImageNet 20×10 ──────────────────────────────────────────────────
    "tinyimagenet_20x10": {
        "classes": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        "title":   "Tiny-ImageNet 20×10",
        "acc1": {
            "β=0 aGlobal nGlobal":   ([71.57, 68.25, 66.56, 62.85, 60.69, 58.77, 58.23, 56.17, 54.07, 52.03],
                                      [1.26,  0.86,  0.65,  0.63,  0.57,  0.81,  0.97,  0.60,  0.26,  0.24]),
            "β=0 aLocal nLocal":     ([71.57, 68.62, 66.92, 62.98, 60.89, 59.23, 58.42, 56.35, 54.34, 52.49],
                                      [1.26,  0.92,  0.48,  0.60,  0.54,  0.69,  0.33,  0.34,  0.15,  0.32]),
            "β=0.5 aLocal nLocal":   ([71.57, 69.13, 67.04, 63.34, 61.09, 58.80, 58.32, 56.34, 54.20, 51.92],
                                      [1.26,  0.52,  0.46,  0.59,  0.65,  0.50,  0.43,  0.16,  0.09,  0.16]),
            "β=0.5 aSymFull nLocal": ([71.57, 68.27, 66.67, 62.68, 60.58, 58.66, 58.09, 56.13, 54.22, 52.02],
                                      [1.26,  0.64,  1.22,  0.83,  0.41,  0.30,  0.17,  0.27,  0.06,  0.19]),
            "β=0.5 aLocal nGlobal":  ([69.80, 68.00, 65.97, 62.85, 60.42, 58.35, 58.14, 56.10, 53.38, 51.85], None),
            "β=0.5 aGlobal nLocal":  ([69.80, 67.60, 65.73, 61.70, 60.26, 58.47, 58.10, 56.29, 54.24, 52.02], None),
            "β=0.5 aGlobal nGlobal": ([69.80, 67.20, 65.90, 62.58, 59.84, 58.27, 56.96, 55.59, 53.82, 51.78], None),
        },
        "nme1": {
            "β=0 aGlobal nGlobal":   ([71.70, 68.83, 65.41, 60.54, 57.25, 54.14, 52.88, 50.13, 46.69, 44.54],
                                      [1.07,  0.44,  0.93,  0.83,  0.64,  0.65,  0.48,  0.12,  0.14,  0.26]),
            "β=0 aLocal nLocal":     ([71.70, 69.05, 65.93, 60.76, 57.31, 54.24, 53.31, 50.67, 47.20, 44.99],
                                      [1.07,  0.88,  0.60,  0.17,  0.45,  0.31,  0.03,  0.42,  0.24,  0.51]),
            "β=0.5 aLocal nLocal":   ([71.70, 69.42, 65.89, 60.78, 57.88, 54.55, 53.26, 50.46, 46.93, 44.72],
                                      [1.07,  0.51,  0.65,  0.31,  0.29,  0.49,  0.17,  0.42,  0.25,  0.52]),
            "β=0.5 aSymFull nLocal": ([71.70, 68.37, 65.40, 60.33, 57.31, 54.27, 52.91, 50.56, 47.17, 44.90],
                                      [1.07,  0.45,  1.00,  0.74,  0.74,  0.60,  0.35,  0.24,  0.11,  0.13]),
            "β=0.5 aLocal nGlobal":  ([70.20, 67.40, 65.33, 61.25, 57.46, 53.95, 53.14, 50.84, 47.37, 44.66], None),
            "β=0.5 aGlobal nLocal":  ([70.20, 67.90, 65.17, 59.88, 56.72, 54.10, 52.93, 50.36, 47.07, 44.74], None),
            "β=0.5 aGlobal nGlobal": ([70.20, 68.15, 64.87, 60.03, 57.08, 53.92, 52.89, 50.51, 47.43, 45.16], None),
        },
    },
}

# ── Chart helpers ─────────────────────────────────────────────────────────────

# Baseline key — delta annotations are computed relative to this series
BASELINE_KEY = "β=0 aGlobal nGlobal"


def _y_limits(metric_data: dict, margin: float = 0.8) -> tuple[float, float]:
    """Compute tight y-axis limits from mean values only (no std expansion)."""
    all_vals = []
    for mean, _std in metric_data.values():
        all_vals.extend(mean)
    lo = min(all_vals) - margin
    hi = max(all_vals) + margin
    # round to nearest 0.5 for cleaner ticks
    lo = np.floor(lo * 2) / 2
    hi = np.ceil(hi * 2) / 2
    return lo, hi


def _nice_yticks(lo: float, hi: float, target_n: int = 7) -> np.ndarray:
    """Return evenly-spaced ticks covering [lo, hi] with a ~target_n count."""
    span = hi - lo
    step_candidates = [0.5, 1.0, 2.0, 2.5, 5.0]
    step = min(step_candidates, key=lambda s: abs(round(span / s) - target_n))
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + step * 0.5, step)


def plot_metric(ax: plt.Axes, classes: list, metric_data: dict, metric_name: str, title: str):
    """Plot one metric (Acc1 or NME1) on the given axes."""
    x = np.array(classes)

    # draw multi-seed series first (solid), then single-seed (dashed)
    order = [k for k in SERIES_STYLE if k in metric_data and metric_data[k][1] is not None]
    order += [k for k in SERIES_STYLE if k in metric_data and metric_data[k][1] is None]

    for name in order:
        if name not in metric_data:
            continue
        mean, _std = metric_data[name]
        style = SERIES_STYLE[name]
        mu = np.array(mean)
        ax.plot(
            x, mu,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            zorder=3,
        )

    # ── Delta annotations: all non-baseline multi-seed series ──────────────
    if BASELINE_KEY in metric_data:
        baseline_curve = np.array(metric_data[BASELINE_KEY][0])
        # collect non-baseline multi-seed series in legend order
        annotated = [
            k for k in SERIES_STYLE
            if k in metric_data and k != BASELINE_KEY and metric_data[k][1] is not None
        ]
        # vertical stagger: each series gets an extra offset proportional to its rank
        v_step = 1.5  # pts between stacked labels
        for rank, name in enumerate(annotated):
            curve  = np.array(metric_data[name][0])
            color  = SERIES_STYLE[name]["color"]
            v_off  = 0.4 + rank * v_step
            for i, class_num in enumerate(x):
                delta = curve[i] - baseline_curve[i]
                sign  = "+" if delta >= 0 else "-"
                ax.text(
                    class_num,
                    curve[i] + v_off,
                    f"{sign}{abs(delta):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.9,
                    ),
                )

    n_annotated = sum(
        1 for k in metric_data
        if k != BASELINE_KEY and metric_data[k][1] is not None
    )
    lo, hi = _y_limits(metric_data, margin=0.8 + max(0, n_annotated - 1) * 1.5)
    ax.set_ylim(lo, hi)
    ax.set_yticks(_nice_yticks(lo, hi))
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Number of Classes", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"{title} — {metric_name}", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_charts(show: bool = False):
    output_dir = Path("figure")
    output_dir.mkdir(exist_ok=True)

    for dataset_key, dataset in DATA.items():
        classes = dataset["classes"]
        title   = dataset["title"]
        acc1    = dataset["acc1"]
        nme1    = dataset["nme1"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

        plot_metric(axes[0], classes, acc1, "Acc@1", title)
        plot_metric(axes[1], classes, nme1, "NME@1", title)

        fig.tight_layout()
        out_path = output_dir / f"{dataset_key}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
        print(f"Saved: {out_path}")

        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate results charts for all datasets.")
    parser.add_argument("--show", action="store_true", help="Open interactive window after saving.")
    args = parser.parse_args()
    build_charts(show=args.show)
