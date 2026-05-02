#!/usr/bin/env python3
"""
generate_report.py — Two-section markdown report for TagFex experiments.

Section 1 (Results):  Comparative acc/NME tables per dataset.
                      Best values in **bold**.  Forgetting metric included.
Section 2 (Debug):    Loss dynamics, ANT distance stats per experiment.
                      Reads exp_debug0.log from raw file or debug_logs.zip.
                      Only shown when debug data is available.

Reads:   LOGS_DIR
Writes:  RESULTS_MD
"""

import io
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

LOGS_DIR   = "/mnt/raid/home/tiago/logs"
RESULTS_MD = "./results_report.md"

_SKIP = {"auto_experiments"}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment name utilities
# ─────────────────────────────────────────────────────────────────────────────

def infer_dataset(exp_name: str) -> str:
    m = re.match(r"(?:debug_exp_|exp_)(.+?)_antB", exp_name)
    return m.group(1) if m else "unknown"


_MODE_SHORT = {
    "antGlobal":       "aGlobal",
    "antLocal":        "aLocal",
    "antSymmetricFull":"aSymFull",
    "nceGlobal":       "nGlobal",
    "nceLocal":        "nLocal",
}

def short_name(exp_name: str) -> str:
    """
    debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_s1993
    → β=0.5 aLocal nLocal
    debug_exp_cifar100_10-10_antB0_nceA1_antGlobal_nceGlobal_s1993
    → β=0 nGlobal ★ Baseline
    """
    m = re.search(
        r"antB([\d.]+)(?:_nceA[\d.]+)?(?:_antM[\d.]+)?_(ant\w+)_(nce\w+?)(?:_s\d+)?$",
        exp_name,
    )
    if not m:
        return exp_name
    beta, ant_mode, nce_mode = m.groups()
    name = f"β={beta} {_MODE_SHORT.get(ant_mode, ant_mode)} {_MODE_SHORT.get(nce_mode, nce_mode)}"
    if is_true_baseline(exp_name):
        name += " ★"
    return name


_DATASET_LABELS = {
    "cifar100_10-10":       "CIFAR-100 10×10",
    "cifar100_50-10":       "CIFAR-100 50+10×5",
    "tiny_imagenet_20-20":  "Tiny-ImageNet 20×10",
    "tiny_imagenet_100-20": "Tiny-ImageNet 100+20×5",
}

def dataset_label(dataset: str) -> str:
    return _DATASET_LABELS.get(dataset, dataset)


def is_true_baseline(exp_name: str) -> bool:
    """β=0 antGlobal nceGlobal — the pure InfoNCE baseline with no ANT."""
    return bool(re.search(r"antB0(?:_nceA[\d.]+)?_antGlobal_nceGlobal", exp_name))


def extract_seed(exp_name: str):
    m = re.search(r"_s(\d+)$", exp_name)
    return int(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Gistlog parsing
# ─────────────────────────────────────────────────────────────────────────────

def _arr(s: str) -> list:
    return [float(x) for x in s.strip().split()] if s.strip() else []


def parse_gistlog(path: str) -> dict:
    """Parse exp_gistlog.log and return metric dict, or None on failure."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None

    acc_curves = re.findall(r"\bacc1_curve \[([^\]]+)\]", text)
    nme_curves = re.findall(r"\bnme1_curve \[([^\]]+)\]", text)
    avg_acc    = re.findall(r"\bavg_acc1 ([\d.]+)", text)
    avg_nme    = re.findall(r"\bavg_nme1 ([\d.]+)", text)
    per_acc    = re.findall(r"eval_acc1_per_task \[([^\]]+)\]", text)
    per_nme    = re.findall(r"eval_nme1_per_task \[([^\]]+)\]", text)

    if not acc_curves:
        return None

    return {
        "acc_curve":       _arr(acc_curves[-1]),
        "nme_curve":       _arr(nme_curves[-1]) if nme_curves else [],
        "avg_acc":         float(avg_acc[-1]) if avg_acc else np.nan,
        "avg_nme":         float(avg_nme[-1]) if avg_nme else np.nan,
        "per_acc_history": [_arr(m) for m in per_acc],
        "per_nme_history": [_arr(m) for m in per_nme],
    }


def avg_forgetting(history: list):
    """
    forgetting_j = max(task j history) − final value for task j
    Returns (mean_forgetting_across_tasks, per_task_list).
    Last task is excluded from the mean (it has never been "forgotten" yet).
    """
    if not history:
        return np.nan, []
    n = len(history[-1])
    ftg = []
    for j in range(n):
        hist = [v[j] for v in history if len(v) > j]
        ftg.append(0.0 if len(hist) <= 1 else max(hist) - hist[-1])
    valid = ftg[:-1] if len(ftg) > 1 else ftg
    return (float(np.mean(valid)) if valid else np.nan), ftg


# ─────────────────────────────────────────────────────────────────────────────
# Debug log parsing  (raw file or inside debug_logs.zip)
# ─────────────────────────────────────────────────────────────────────────────

_CTX_RE  = re.compile(r"\[T(\d+) E(\d+) B(\d+)\]")
_KV_RE   = re.compile(r"([\w]+):\s+([-\d.eE+]+)%?")
_MARKERS = {
    "Loss components:":    "loss",
    "ANT distance stats:": "ant",
}


def _open_debug(exp_path: str):
    """Return (file-like, source_label). Raw file takes priority over zip."""
    raw = os.path.join(exp_path, "exp_debug0.log")
    if os.path.exists(raw):
        return open(raw, encoding="utf-8", errors="replace"), "file"
    zp = os.path.join(exp_path, "debug_logs.zip")
    if os.path.exists(zp):
        try:
            zf = zipfile.ZipFile(zp, "r")
            if "exp_debug0.log" in zf.namelist():
                return io.TextIOWrapper(
                    zf.open("exp_debug0.log"), encoding="utf-8", errors="replace"
                ), "zip"
            zf.close()
        except Exception:
            pass
    return None, None


def _strip_loss(kvs: dict) -> dict:
    return {f"loss_{k.split('_', 1)[1] if '_' in k else k}": v for k, v in kvs.items()}


def parse_debug_log(exp_path: str):
    """Return (DataFrame, source_label). DataFrame is empty if no debug data."""
    fh, label = _open_debug(exp_path)
    if fh is None:
        return pd.DataFrame(), "none"

    records = {}
    with fh:
        for line in fh:
            m_type = None
            m_str_used = None
            for m_str, mt in _MARKERS.items():
                if m_str in line:
                    m_type, m_str_used = mt, m_str
                    break
            if m_type is None:
                continue
            ctx = _CTX_RE.search(line)
            if not ctx:
                continue
            key = (int(ctx[1]), int(ctx[2]), int(ctx[3]))
            after = line[line.index(m_str_used) + len(m_str_used):]
            kvs = {k: float(v) for k, v in _KV_RE.findall(after)}
            if m_type == "loss":
                kvs = _strip_loss(kvs)
            if key not in records:
                records[key] = {"task": key[0], "epoch": key[1], "batch": key[2]}
            records[key].update(kvs)

    df = pd.DataFrame(list(records.values())) if records else pd.DataFrame()
    return df, label


# ─────────────────────────────────────────────────────────────────────────────
# Result collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_results(logs_dir: str) -> pd.DataFrame:
    rows = []
    for exp in sorted(os.listdir(logs_dir)):
        if exp in _SKIP:
            continue
        exp_path = os.path.join(logs_dir, exp)
        if not os.path.isdir(exp_path):
            continue
        gistlog = os.path.join(exp_path, "exp_gistlog.log")
        if not os.path.exists(gistlog):
            continue
        parsed = parse_gistlog(gistlog)
        if not parsed:
            continue

        fgt_acc, fgt_acc_list = avg_forgetting(parsed["per_acc_history"])
        fgt_nme, fgt_nme_list = avg_forgetting(parsed["per_nme_history"])

        row = {
            "dataset":     infer_dataset(exp),
            "exp":         exp,
            "label":       short_name(exp),
            "seed":        extract_seed(exp),
            "is_baseline": is_true_baseline(exp),
            "avg_acc":     parsed["avg_acc"],
            "avg_nme":     parsed["avg_nme"],
            "fgt_acc":     fgt_acc,
            "fgt_nme":     fgt_nme,
            "num_tasks":   len(parsed["acc_curve"]),
        }
        for i, v in enumerate(parsed["acc_curve"], 1):
            row[f"acc_T{i}"] = v
        for i, v in enumerate(parsed["nme_curve"], 1):
            row[f"nme_T{i}"] = v
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def filter_complete(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only experiments that finished all tasks (no missing acc curves)."""
    parts = []
    for _, g in df.groupby("dataset"):
        n = int(g["num_tasks"].max())
        acc_cols = [f"acc_T{i}" for i in range(1, n + 1)]
        g = g.copy()
        for c in acc_cols:
            if c not in g.columns:
                g[c] = np.nan
        parts.append(g[g["num_tasks"] == n].dropna(subset=acc_cols))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Markdown table builder with best-value highlighting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_col(series, higher_is_better: bool, digits: int = 2) -> list:
    """Format a numeric column; best value gets **bold**."""
    arr = np.array(series, dtype=float)
    valid = arr[~np.isnan(arr)]
    best_val = (valid.max() if higher_is_better else valid.min()) if valid.size else None

    fmt = f"{{:.{digits}f}}"
    out = []
    for v in arr:
        if not pd.notna(v):
            out.append("")
            continue
        s = fmt.format(v)
        out.append(f"**{s}**" if best_val is not None and abs(v - best_val) < 1e-9 else s)
    return out


def df_to_md(df: pd.DataFrame, col_specs: list) -> str:
    """
    col_specs: list of (col_name, header, higher_is_better_or_None, digits)
    Numeric columns with higher_is_better != None get their best value bolded.
    """
    present = [(c, h, hib, dg) for c, h, hib, dg in col_specs if c in df.columns]
    if not present:
        return "*No data.*"

    cols_fmt = {}
    for col, hdr, hib, dg in present:
        s = df[col]
        if hib is not None and pd.api.types.is_numeric_dtype(s):
            cols_fmt[col] = _fmt_col(s.values, hib, dg)
        else:
            cols_fmt[col] = s.astype(str).tolist()

    hdr_row = "| " + " | ".join(h for _, h, _, _ in present) + " |"
    sep_row = "| " + " | ".join(":---" for _ in present) + " |"
    body = [
        "| " + " | ".join(cols_fmt[col][i] for col, _, _, _ in present) + " |"
        for i in range(len(df))
    ]
    return "\n".join([hdr_row, sep_row] + body)


def simple_md(df: pd.DataFrame, cols: list, digits: int = 4) -> str:
    """Plain markdown table (no highlighting), float columns formatted."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return "*No data.*"
    tmp = df[present].copy()
    for c in tmp.columns:
        if pd.api.types.is_float_dtype(tmp[c]):
            tmp[c] = tmp[c].map(lambda v: f"{v:.{digits}f}" if pd.notna(v) else "")
        else:
            tmp[c] = tmp[c].astype(str)
    return tmp.to_markdown(index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Results
# ─────────────────────────────────────────────────────────────────────────────

def _results_section(df: pd.DataFrame) -> list:
    lines = []
    lines.append("# Section 1 — Results\n")
    lines.append(
        "> Metrics sourced from `exp_gistlog.log`.  "
        "**Bold** = best value in the group.  \n"
        "> Forgetting = `max(per-task history) − final value`, mean across all tasks "
        "except the last (which has never been forgotten yet).\n"
    )

    for dataset in sorted(df["dataset"].unique()):
        g = df[df["dataset"] == dataset].copy()
        n  = int(g["num_tasks"].max())
        dl = dataset_label(dataset)

        # ── Compute delta vs true baseline (mean across seeds) ────────────────
        baseline_rows = g[g["is_baseline"]]
        base_mean_acc = baseline_rows["avg_acc"].mean() if not baseline_rows.empty else np.nan
        base_mean_nme = baseline_rows["avg_nme"].mean() if not baseline_rows.empty else np.nan
        g["delta_acc"] = np.where(g["is_baseline"], np.nan, g["avg_acc"] - base_mean_acc)
        g["delta_nme"] = np.where(g["is_baseline"], np.nan, g["avg_nme"] - base_mean_nme)

        # ── Sort: baselines pinned to top, then others by avg_acc desc ────────
        g["_sort"] = (~g["is_baseline"]).astype(int)
        g = (
            g.sort_values(["_sort", "avg_acc"], ascending=[True, False])
            .drop(columns=["_sort"])
            .reset_index(drop=True)
        )

        lines.append(f"## {dl}\n")
        n_base = int(g["is_baseline"].sum())
        n_exp  = len(g) - n_base
        base_info = (
            f" | True baseline (β=0 nGlobal): **{n_base}** runs, "
            f"mean Avg Acc1 = **{base_mean_acc:.2f}%**"
        ) if not np.isnan(base_mean_acc) else ""
        lines.append(
            f"Complete experiments: **{len(g)}** ({n_base} baseline + {n_exp} variants) "
            f"| Tasks per session: **{n}**{base_info}\n"
        )

        # ── Summary: avg metrics + forgetting + delta ─────────────────────────
        lines.append("### Accuracy & Forgetting Summary\n")
        delta_spec = [("delta_acc", "∆ Acc vs Baseline", True, 2)] if not np.isnan(base_mean_acc) else []
        lines.append(df_to_md(g, [
            ("label",   "Experiment",      None,  0),
            ("avg_acc", "Avg Acc1 (%)",    True,  2),
            ("avg_nme", "Avg NME1 (%)",    True,  2),
            ("fgt_acc", "Forgetting Acc",  False, 2),
            ("fgt_nme", "Forgetting NME",  False, 2),
        ] + delta_spec))
        lines.append("")

        # ── Acc1 curve per task ───────────────────────────────────────────────
        lines.append("### Acc1 Curve (final accuracy per task after all sessions)\n")
        acc_specs = [(f"acc_T{i}", f"T{i}", True, 2) for i in range(1, n + 1) if f"acc_T{i}" in g.columns]
        lines.append(df_to_md(g, [("label", "Experiment", None, 0)] + acc_specs))
        lines.append("")

        # ── NME1 curve per task ───────────────────────────────────────────────
        nme_specs = [(f"nme_T{i}", f"T{i}", True, 2) for i in range(1, n + 1) if f"nme_T{i}" in g.columns]
        if nme_specs:
            lines.append("### NME1 Curve\n")
            lines.append(df_to_md(g, [("label", "Experiment", None, 0)] + nme_specs))
            lines.append("")

        # ── Best callouts ─────────────────────────────────────────────────────
        lines.append("### Highlights\n")
        non_base = g[~g["is_baseline"]]
        cmp_g    = non_base if not non_base.empty else g
        best_acc_row = cmp_g.sort_values("avg_acc", ascending=False).iloc[0]
        best_nme_row = cmp_g.sort_values("avg_nme", ascending=False).iloc[0]
        best_fgt_row = cmp_g.sort_values(["fgt_acc", "avg_acc"], ascending=[True, False]).iloc[0]
        lines.append(f"| Metric | Best Experiment | Value |")
        lines.append(f"| :--- | :--- | ---: |")
        lines.append(f"| ⭐ Highest Avg Acc1 | `{best_acc_row['label']}` | **{best_acc_row['avg_acc']:.2f}%** |")
        lines.append(f"| ⭐ Highest Avg NME1 | `{best_nme_row['label']}` | **{best_nme_row['avg_nme']:.2f}%** |")
        lines.append(f"| ⭐ Lowest Forgetting | `{best_fgt_row['label']}` | **{best_fgt_row['fgt_acc']:.2f}%** |")
        if not np.isnan(base_mean_acc) and not non_base.empty:
            best_delta_row = non_base.sort_values("delta_acc", ascending=False).iloc[0]
            lines.append(
                f"| 📈 Best ∆ vs Baseline | `{best_delta_row['label']}` "
                f"| **{best_delta_row['delta_acc']:+.2f}%** "
                f"(baseline mean: {base_mean_acc:.2f}%) |"
            )
        lines.append("")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Debug
# ─────────────────────────────────────────────────────────────────────────────

_LOSS_COLS = [
    "loss_total", "loss_nll", "loss_ant_loss",
    "loss_nce_weighted", "loss_ant_weighted",
]
_ANT_COLS = ["violation_pct", "gap_mean", "gap_min", "gap_max", "ant_loss"]


def _debug_exp_section(exp: str, exp_path: str) -> list:
    lines = []
    print(f"  {exp} … ", end="", flush=True)

    debug_df, source = parse_debug_log(exp_path)
    if debug_df.empty:
        print("no debug data.")
        return ["*No debug data available for this experiment.*\n"]

    # Aggregate: mean per (task, epoch)
    num_cols = [c for c in debug_df.columns if c not in ("task", "epoch", "batch")]
    epoch_df = debug_df.groupby(["task", "epoch"])[num_cols].mean().reset_index()

    n_rec   = len(debug_df)
    n_tasks = epoch_df["task"].nunique()
    print(f"ok ({n_rec:,} records, {n_tasks} tasks, source={source})")

    loss_cols = [c for c in _LOSS_COLS if c in epoch_df.columns]
    ant_cols  = [c for c in _ANT_COLS  if c in epoch_df.columns]
    tasks     = sorted(epoch_df["task"].unique())

    # ── Final epoch summary ─────────────────────────────────────────────────
    final_rows = []
    for t in tasks:
        t_df   = epoch_df[epoch_df["task"] == t]
        last   = t_df["epoch"].max()
        row    = t_df[t_df["epoch"] == last].iloc[0].to_dict()
        row["task"]     = int(t)
        row["last_ep"]  = int(last)
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows)

    lines.append("#### Final Epoch Summary (per task)\n")
    summary_cols = (
        ["task", "last_ep"]
        + loss_cols
        + ant_cols
    )
    lines.append(simple_md(final_df, summary_cols, digits=4))
    lines.append("")

    # ── Evolution by task (sampled every 10 epochs) ─────────────────────────
    evo_cols = loss_cols + [c for c in ["violation_pct", "gap_mean"] if c in epoch_df.columns and c not in loss_cols]
    lines.append("#### Training Evolution (sampled every 10 epochs)\n")

    for t in tasks:
        t_df  = epoch_df[epoch_df["task"] == t].sort_values("epoch")
        mn    = int(t_df["epoch"].min())
        mx    = int(t_df["epoch"].max())
        n_bat = (
            int(debug_df[debug_df["task"] == t]["batch"].max())
            if "batch" in debug_df.columns else "?"
        )
        sample_eps = {mn} | set(range(10, mx + 1, 10)) | {mx}
        sampled    = t_df[t_df["epoch"].isin(sample_eps)].copy()

        lines.append(f"**Task {t}** — epochs {mn}–{mx}, ~{n_bat} batches/epoch\n")
        lines.append(simple_md(sampled, ["epoch"] + evo_cols, digits=4))
        lines.append("")

    return lines


def _debug_section(df: pd.DataFrame, logs_dir: str) -> list:
    lines = []
    lines.append("# Section 2 — Debug Metrics\n")
    lines.append(
        "> Per-batch stats from `exp_debug0.log` (raw or extracted from `debug_logs.zip`).  \n"
        "> Values are **batch-averages per epoch**.  \n"
        "> Only experiments with debug data are shown.\n"
    )

    for dataset in sorted(df["dataset"].unique()):
        g  = df[df["dataset"] == dataset].sort_values("avg_acc", ascending=False)
        dl = dataset_label(dataset)
        lines.append(f"## {dl}\n")

        for _, row in g.iterrows():
            exp      = row["exp"]
            lbl      = row["label"]
            exp_path = os.path.join(logs_dir, exp)
            lines.append(f"### `{lbl}`\n")
            lines.append(f"<details>\n<summary>Experiment: <code>{exp}</code></summary>\n")
            lines.extend(_debug_exp_section(exp, exp_path))
            lines.append("</details>\n")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Scanning: {LOGS_DIR}")
    raw_df = collect_results(LOGS_DIR)

    if raw_df.empty:
        print("No experiments with gistlog found.")
        return

    df = filter_complete(raw_df)
    print(f"Experiments: {len(raw_df)} found, {len(df)} complete\n")

    if df.empty:
        print("No complete experiments after filtering.")
        return

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    print("=== Section 1: Results ===")
    sec1 = _results_section(df)

    print("\n=== Section 2: Debug ===")
    sec2 = _debug_section(df, LOGS_DIR)

    report = "\n".join([
        f"<!-- Generated {ts} -->\n",
        *sec1,
        "---\n",
        *sec2,
    ])

    with open(RESULTS_MD, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"\nReport saved → {RESULTS_MD}")

    # Quick preview
    print("\n=== Quick Preview ===")
    for dataset in sorted(df["dataset"].unique()):
        g = df[df["dataset"] == dataset].copy()
        baseline_rows = g[g["is_baseline"]]
        base_mean = baseline_rows["avg_acc"].mean() if not baseline_rows.empty else float("nan")
        # baselines first, then by avg_acc desc
        g["_s"] = (~g["is_baseline"]).astype(int)
        g = g.sort_values(["_s", "avg_acc"], ascending=[True, False]).drop(columns=["_s"])
        label = dataset_label(dataset)
        base_str = f"  baseline mean={base_mean:.2f}" if not np.isnan(base_mean) else ""
        print(f"\n[{label}]{base_str}")
        for _, row in g.iterrows():
            delta_str = ""
            if not row["is_baseline"] and not np.isnan(base_mean):
                delta_str = f"  Δ={row['avg_acc'] - base_mean:+.2f}"
            print(f"  {row['label']:35s}  avg_acc={row['avg_acc']:.2f}  avg_nme={row['avg_nme']:.2f}  fgt={row['fgt_acc']:.2f}{delta_str}")


if __name__ == "__main__":
    main()
