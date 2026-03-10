"""
Parse and visualize training progress from similarity_debug.log.

The log is ~11GB with per-batch detailed similarity matrices.
This script extracts only the structured summary lines (3-6 per batch):
  [T{t} E{e} B{b}] Contrastive stats: ...
  [T{t} E{e} B{b}] ANT distance stats: ...
  [T{t} E{e} B{b}] Loss components: contrast_* | kd_* | ...

Each batch can have two triples (contrast + kd from task 2 onward).
A context buffer ties stats to the correct loss type.

Usage:
    # Fast: grep summary lines first (~2 min), then parse the small file:
    grep "[T[0-9]" <log> > /tmp/sim_summary.log
    python analysis/scripts/parse_similarity_log.py --log /tmp/sim_summary.log

    # Re-plot from existing CSV without re-parsing:
    python analysis/scripts/parse_similarity_log.py \\
        --from-csv analysis/results/similarity_progress/batch_metrics.csv
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Regex patterns — signed floats to handle negative similarities (task 2+)
# ---------------------------------------------------------------------------
_F = r"(-?[\d.]+)"

RE_CONTRASTIVE = re.compile(
    r"\[T(\d+) E(\d+) B(\d+)\] Contrastive stats: "
    rf"pos_mean: {_F} \| pos_std: {_F} \| neg_mean: {_F} \| neg_std: {_F} \| current_gap: {_F}"
)
RE_ANT = re.compile(
    r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: "
    rf"pos_min: {_F} \| pos_max: {_F} \| neg_min: {_F} \| neg_max: {_F} \| "
    rf"gap_mean: {_F} \| gap_std: {_F} \| gap_min: {_F} \| gap_max: {_F} \| "
    rf"margin: {_F} \| violation_pct: {_F}% \| ant_loss: {_F}"
)
RE_LOSS_CONTRAST = re.compile(
    r"\[T(\d+) E(\d+) B(\d+)\] Loss components: "
    rf"contrast_nll: {_F} \| contrast_ant_loss: {_F} \| "
    rf"contrast_nce_weighted: {_F} \| contrast_ant_weighted: {_F} \| contrast_total: {_F}"
)
RE_LOSS_KD = re.compile(
    r"\[T(\d+) E(\d+) B(\d+)\] Loss components: "
    rf"kd_nll: {_F} \| kd_ant_loss: {_F} \| "
    rf"kd_nce_weighted: {_F} \| kd_ant_weighted: {_F} \| kd_total: {_F}"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log(log_path: Path) -> pd.DataFrame:
    """Stream-parse the log extracting structured summary lines.

    Each batch has up to two triples of:
      (Contrastive stats -> ANT stats -> Loss components)
    one per loss type (contrast=task1, contrast+kd=task2+).

    Returns a DataFrame with one row per (batch x loss_type).
    Columns: task, epoch, batch, loss_type, pos_mean, neg_mean, gap,
             violation_pct, ant_loss, nll, ant_loss_w, total
    """
    rows: list[dict] = []
    pending: dict = {}

    print(f"Parsing {log_path} ...", flush=True)
    with open(log_path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0 and i > 0:
                print(f"  {i // 1_000_000}M lines, {len(rows)} records", flush=True)

            m = RE_CONTRASTIVE.search(line)
            if m:
                # New triple begins — start fresh buffer
                pending = {
                    "task": int(m[1]),
                    "epoch": int(m[2]),
                    "batch": int(m[3]),
                    "pos_mean": float(m[4]),
                    "pos_std": float(m[5]),
                    "neg_mean": float(m[6]),
                    "neg_std": float(m[7]),
                    "gap": float(m[8]),
                }
                continue

            m = RE_ANT.search(line)
            if m and pending:
                pending.update(
                    {
                        "gap_mean": float(m[8]),
                        "gap_std": float(m[9]),
                        "gap_min": float(m[10]),
                        "gap_max": float(m[11]),
                        "violation_pct": float(m[12]),
                        "ant_loss": float(m[13]),
                    }
                )
                continue

            m = RE_LOSS_CONTRAST.search(line)
            if m and pending:
                pending.update(
                    {
                        "loss_type": "contrast",
                        "nll": float(m[4]),
                        "ant_loss_w": float(m[7]),
                        "total": float(m[8]),
                    }
                )
                rows.append(pending.copy())
                pending = {}
                continue

            m = RE_LOSS_KD.search(line)
            if m and pending:
                pending.update(
                    {
                        "loss_type": "kd",
                        "nll": float(m[4]),
                        "ant_loss_w": float(m[7]),
                        "total": float(m[8]),
                    }
                )
                rows.append(pending.copy())
                pending = {}

    df = pd.DataFrame(rows)
    df = df.sort_values(["task", "epoch", "batch", "loss_type"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Average all metrics per (task, epoch, loss_type)."""
    agg = (
        df.groupby(["task", "epoch", "loss_type"])
        .agg(
            pos_mean=("pos_mean", "mean"),
            neg_mean=("neg_mean", "mean"),
            gap=("gap", "mean"),
            violation_pct=("violation_pct", "mean"),
            ant_loss=("ant_loss", "mean"),
            nll=("nll", "mean"),
            ant_loss_w=("ant_loss_w", "mean"),
            total=("total", "mean"),
            batches=("batch", "count"),
        )
        .reset_index()
    )
    agg = agg.sort_values(["task", "epoch", "loss_type"]).reset_index(drop=True)
    # Sequential index per loss_type for plotting across tasks
    agg["run_idx"] = agg.groupby("loss_type").cumcount()
    return agg


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _add_task_separators(axes, epoch_df: pd.DataFrame) -> None:
    """Draw vertical dashed lines at task boundaries on all axes."""
    for task in sorted(epoch_df["task"].unique())[1:]:
        mask = epoch_df["task"] == task
        if not mask.any():
            continue
        x = epoch_df.loc[mask, "run_idx"].iloc[0]
        for ax in axes:
            ax.axvline(x, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)


def plot_progress(epoch_df: pd.DataFrame, out_dir: Path) -> None:
    """Generate multi-panel figures for loss and similarity evolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(epoch_df["task"].unique())
    loss_types = sorted(epoch_df["loss_type"].unique())
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for lt in loss_types:
        sub_lt = epoch_df[epoch_df["loss_type"] == lt]

        # ---- Loss components ----------------------------------------
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"[{lt}] Loss Components per Epoch (avg over batches)", fontsize=13
        )

        for i, task in enumerate(tasks):
            sub = sub_lt[sub_lt["task"] == task]
            if sub.empty:
                continue
            c = colors[i % len(colors)]
            lbl = f"T{int(task)}"
            axes[0].plot(sub["run_idx"], sub["nll"], color=c, label=lbl, linewidth=1.2)
            axes[1].plot(
                sub["run_idx"], sub["ant_loss_w"], color=c, label=lbl, linewidth=1.2
            )
            axes[2].plot(
                sub["run_idx"], sub["total"], color=c, label=lbl, linewidth=1.2
            )

        axes[0].set_ylabel("NCE loss (nll)")
        axes[1].set_ylabel("ANT loss (weighted)")
        axes[2].set_ylabel("Total loss")
        axes[2].set_xlabel("Epoch index (sequential across tasks)")
        for ax in axes:
            ax.legend(fontsize=8, ncol=max(1, len(tasks) // 3))
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        _add_task_separators(axes, sub_lt)
        fig.tight_layout()
        fname = out_dir / f"loss_components_{lt}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")

        # ---- Similarity diagnostics ------------------------------------
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"[{lt}] Similarity Diagnostics per Epoch", fontsize=13)

        for i, task in enumerate(tasks):
            sub = sub_lt[sub_lt["task"] == task]
            if sub.empty:
                continue
            c = colors[i % len(colors)]
            lbl = f"T{int(task)}"
            axes[0].plot(
                sub["run_idx"],
                sub["pos_mean"],
                color=c,
                label=f"{lbl} pos",
                linewidth=1.2,
            )
            axes[0].plot(
                sub["run_idx"],
                sub["neg_mean"],
                color=c,
                label=f"{lbl} neg",
                linewidth=1.2,
                linestyle="--",
            )
            axes[1].plot(sub["run_idx"], sub["gap"], color=c, label=lbl, linewidth=1.2)
            axes[2].plot(
                sub["run_idx"], sub["violation_pct"], color=c, label=lbl, linewidth=1.2
            )

        axes[0].set_ylabel("Mean similarity\n(pos solid, neg dashed)")
        axes[1].set_ylabel("Gap (pos_mean − neg_mean)")
        axes[1].axhline(0, color="red", linestyle=":", linewidth=0.8, alpha=0.7)
        axes[2].set_ylabel("Margin violation %")
        axes[2].axhline(0, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
        axes[2].set_xlabel("Epoch index (sequential across tasks)")
        for ax in axes:
            ax.legend(fontsize=7, ncol=max(1, len(tasks) // 3))
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        _add_task_separators(axes, sub_lt)
        fig.tight_layout()
        fname = out_dir / f"similarity_diagnostics_{lt}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")

        # ---- Per-task detail -------------------------------------------
        n = len(tasks)
        fig, axes_grid = plt.subplots(n, 2, figsize=(14, 3 * n), squeeze=False)
        fig.suptitle(f"[{lt}] Per-Task Loss & Gap Evolution", fontsize=13)

        for row, task in enumerate(tasks):
            sub = sub_lt[sub_lt["task"] == task]
            ax_l, ax_g = axes_grid[row][0], axes_grid[row][1]

            if not sub.empty:
                ax_l.plot(sub["epoch"], sub["nll"], label="NCE", linewidth=1.2)
                ax_l.plot(
                    sub["epoch"], sub["ant_loss_w"], label="ANT (w)", linewidth=1.2
                )
                ax_l.plot(
                    sub["epoch"],
                    sub["total"],
                    label="Total",
                    linewidth=1.4,
                    linestyle="--",
                )
                ax_g.plot(
                    sub["epoch"],
                    sub["gap"],
                    label="gap",
                    color="tab:blue",
                    linewidth=1.2,
                )
                ax_g2 = ax_g.twinx()
                ax_g2.plot(
                    sub["epoch"],
                    sub["violation_pct"],
                    label="violation %",
                    color="tab:red",
                    linewidth=1.0,
                    alpha=0.7,
                )
                ax_g2.set_ylabel("violation %", color="tab:red", fontsize=8)
                ax_g.axhline(0, color="red", linestyle=":", linewidth=0.8)
                ax_g2.legend(fontsize=8, loc="upper right")

            ax_l.set_title(f"Task {int(task)}")
            ax_l.set_xlabel("Epoch")
            ax_l.legend(fontsize=8)
            ax_l.grid(True, alpha=0.3)
            ax_g.set_title(f"Task {int(task)} — Gap & Violations")
            ax_g.set_xlabel("Epoch")
            ax_g.set_ylabel("pos−neg gap")
            ax_g.legend(fontsize=8, loc="upper left")
            ax_g.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = out_dir / f"per_task_detail_{lt}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse similarity_debug.log and plot training progress."
    )
    parser.add_argument(
        "--log",
        default="logs/debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal/similarity_debug.log",
        help="Path to similarity_debug.log or pre-extracted summary lines",
    )
    parser.add_argument(
        "--out",
        default="analysis/results/similarity_progress",
        help="Output directory for CSV and plots",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only parse and save CSV, skip plotting",
    )
    parser.add_argument(
        "--from-csv",
        metavar="CSV",
        default=None,
        help="Skip parsing; load existing batch CSV and regenerate plots",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        print(f"Loading batch CSV from {args.from_csv}")
        batch_df = pd.read_csv(args.from_csv)
    else:
        batch_df = parse_log(Path(args.log))
        csv_path = out_dir / "batch_metrics.csv"
        batch_df.to_csv(csv_path, index=False)
        print(f"Saved {len(batch_df)} batch records to {csv_path}")

    epoch_df = aggregate_by_epoch(batch_df)
    epoch_csv = out_dir / "epoch_metrics.csv"
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"Saved {len(epoch_df)} epoch records to {epoch_csv}")

    print("\n=== Summary ===")
    for (task, lt), sub in epoch_df.groupby(["task", "loss_type"]):
        sub = sub.sort_values("epoch")
        first, last = sub.iloc[0], sub.iloc[-1]
        vf = (
            f"{first['violation_pct']:.1f}%"
            if not np.isnan(first["violation_pct"])
            else "n/a"
        )
        vl = (
            f"{last['violation_pct']:.1f}%"
            if not np.isnan(last["violation_pct"])
            else "n/a"
        )
        print(
            f"  Task {int(task):2d} [{lt:8s}]: epochs {int(sub['epoch'].min()):3d}–{int(sub['epoch'].max()):3d} "
            f"| total {first['total']:.3f} → {last['total']:.3f} "
            f"| gap {first['gap']:.4f} → {last['gap']:.4f} "
            f"| violation {vf} → {vl}"
        )

    if not args.csv_only:
        plot_progress(epoch_df, out_dir)


if __name__ == "__main__":
    main()
