"""
CIFAR-100 10-10 NME@1 accuracy-vs-classes chart — TagFex paper style.

For local methods (TagFex, ANT) the nme1_curve is read from experiment log
files and averaged across all available seeds.  Reference baselines (iCaRL,
DyTox, DER) use hardcoded paper values since no local logs exist for them.

Usage:
    python chart.py                  # generates cifar100_10-10.png
    python chart.py --show           # also opens interactive window
"""
import re
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parent / "logs"
OUTPUT_FILE = "cifar100_10-10.png"

# Number-of-classes axis (one value per incremental task)
CLASSES = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Reference baselines from papers — no local experiment logs for these.
REFERENCE_DATA = {
    "iCaRL": [90.4, 78.15, 74.37, 68.82, 64.1, 60.03, 57.71, 53.79, 51.49, 49.11],
    "DyTox": [92.4, 86.05, 79.13, 75.35, 74.0, 72.2, 68.49, 66.06, 64.09, 60.39],
    "DER":   [90.2, 80.3, 78.73, 75.48, 73.86, 71.47, 70.1, 66.94, 65.22, 64.35],
}

# Local experiment series: list the log-directory name (under LOGS_DIR) for
# each seed that should be included in the average.  Directories that do not
# exist yet are silently skipped, so the chart can be regenerated at any time
# as seeds complete.
EXPERIMENT_SERIES = {
    # TagFex: single-seed paper reference (s1993 only — the original published run)
    "TagFex": [
        "done_exp_cifar100_10-10_baseline_tagfex_original_s1993",
    ],
    # Baseline Local: 5-seed average of our reproduced TagFex (antBeta=0, local anchor)
    "Baseline Local": [
        "done_exp_cifar100_10-10_baseline_tagfex_original_s1993",
        "done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1994",
        "done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1995",
        "done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1996",
        "done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1997",
    ],
    # ANT (β=0.5, m=0.5, Local) — uncomment when multi-seed runs complete
    # "ANT": [
    #     "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1993",
    #     "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1994",
    #     "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1995",
    #     "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1996",
    #     "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1997",
    # ],
}

# Visual style for every series (order matters for the legend)
SERIES_STYLE = {
    "iCaRL":          dict(marker="^", color="tab:blue",   markersize=8,  linestyle="-"),
    "DyTox":          dict(marker="s", color="tab:cyan",   markersize=8,  linestyle="-"),
    "DER":            dict(marker="o", color="tab:green",  markersize=8,  linestyle="-"),
    "TagFex":         dict(marker="*", color="tab:red",    markersize=10, linestyle="-"),
    "Baseline Local": dict(marker="p", color="tab:purple", markersize=10, linestyle="-"),
    "ANT":            dict(marker="p", color="tab:purple", markersize=10, linestyle="-"),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_nme1_curve(log_dir: Path) -> list[float] | None:
    """Return the nme1_curve list from *exp_gistlog.log* inside *log_dir*.

    Returns None if the file is missing or the curve is not yet written.
    Also accepts in-progress experiments (exp_* dirs that are not done yet).
    """
    # Try done_exp first, then exp_ (in-progress), scanning for any gist log
    candidates = list(log_dir.glob("exp_gistlog.log"))
    if not candidates:
        return None

    gist_log = candidates[0]
    pattern = re.compile(r"nme1_curve \[([\d\.\s]+)\]")

    last_curve = None
    try:
        with open(gist_log) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    last_curve = [float(v) for v in m.group(1).split()]
    except OSError:
        return None

    return last_curve


def load_series_mean(dir_names: list[str]) -> tuple[np.ndarray | None, int]:
    """Load nme1_curve from all existing seed directories and return the mean.

    Only fully-completed runs (curve length == len(CLASSES)) are included.
    Returns (mean_array, n_seeds_found).  mean_array is None when no logs exist.
    """
    n_tasks = len(CLASSES)
    curves = []
    for name in dir_names:
        log_dir = LOGS_DIR / name
        # Also accept in-progress dirs (without 'done_' prefix) but only if complete
        if not log_dir.exists():
            in_progress = LOGS_DIR / name.replace("done_exp_", "exp_", 1)
            if in_progress.exists():
                log_dir = in_progress
            else:
                continue

        curve = _read_nme1_curve(log_dir)
        if curve is None:
            print(f"  [skip] {log_dir.name}  (log not ready)")
        elif len(curve) != n_tasks:
            print(f"  [skip] {log_dir.name}  (incomplete: {len(curve)}/{n_tasks} tasks done)")
        else:
            curves.append(curve)
            print(f"  [ok]   {log_dir.name}  →  avg={np.mean(curve):.2f}")

    if not curves:
        return None, 0

    arr = np.array(curves)  # shape: (n_seeds, n_tasks)
    return arr.mean(axis=0), len(curves)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_chart(show: bool = False):
    """Build and save the accuracy-vs-classes chart."""

    # Resolve curve data for every series
    series_data: dict[str, np.ndarray] = {}

    # Reference baselines (hardcoded)
    for name, values in REFERENCE_DATA.items():
        series_data[name] = np.array(values)

    # Local experiments (loaded from logs)
    for name, dir_names in EXPERIMENT_SERIES.items():
        print(f"\nLoading {name}:")
        mean_curve, n = load_series_mean(dir_names)
        if mean_curve is not None:
            series_data[name] = mean_curve
            print(f"  => mean over {n} seed(s): avg_nme1={mean_curve.mean():.2f}")
        else:
            print(f"  => no logs found, skipping {name}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, style in SERIES_STYLE.items():
        if name not in series_data:
            continue
        ax.plot(CLASSES, series_data[name], label=name, **style)

    # Delta annotations: top series vs. best of all other plotted methods
    # Priority: ANT > Baseline Local (whichever is present)
    highlight_key = next((k for k in ("ANT", "Baseline Local") if k in series_data), None)
    if highlight_key:
        top_curve = series_data[highlight_key]
        others = [v for k, v in series_data.items() if k != highlight_key]
        if others:
            best_other = np.max(np.stack(others), axis=0)
            for i, class_num in enumerate(CLASSES):
                delta = top_curve[i] - best_other[i]
                sign = "+" if delta >= 0 else "-"
                ax.text(
                    class_num,
                    top_curve[i] + 1.5,
                    f"{sign} {abs(delta):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="tab:purple",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor="tab:purple",
                        alpha=1,
                    ),
                )

    ax.set_xlabel("Number of Classes", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(50, 100, 10))
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=1)

    fig.savefig(OUTPUT_FILE, bbox_inches="tight", pad_inches=0.05)
    print(f"\nSalvo em: {OUTPUT_FILE}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Open interactive window after saving")
    args = parser.parse_args()
    build_chart(show=args.show)
