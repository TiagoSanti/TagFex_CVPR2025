#!/usr/bin/env python3
"""
generate_html_pdf_report.py — HTML + PDF report for TagFex experiments.

Section 0 (Overview): Cross-dataset Δ Acc / NME / Fgt table with HTML rowspan.
Section 1 (Results):  Comparative acc/NME tables per dataset.
                      Best values in **bold**.  Forgetting metric included.
Section 2 (Debug):    Loss dynamics, ANT distance stats per experiment.
                      Reads exp_debug0.log from raw file or debug_logs.zip.
                      Skipped when --short flag is passed.

Reads:   LOGS_DIR  (env var, default ./logs)
Writes:  RESULTS_HTML / RESULTS_PDF  (env vars, default ./results_report.html/pdf)

Usage:
  python generate_html_pdf_report.py            # full report (all 3 sections)
  python generate_html_pdf_report.py --short    # skip Section 2, faster PDF
"""

import argparse
import io
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

LOGS_DIR   = os.environ.get("TAGFEX_LOGS_DIR", "./logs")
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
    """Convert an experiment directory name to a compact human-readable label.

    Handles baseline/ANT/avgK/SBS variants. Examples:
        debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_s1993
            → β=0.5 aLocal nLocal
        exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_avgK3_s1993
            → β=0.5 aLocal nLocal avgK3
        exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_sbsQ0.20S0.20_s1993
            → β=0.5 aLocal nLocal SBS
    """
    # Use [a-zA-Z0-9]+ for mode tokens so we don't accidentally capture
    # optional suffix tokens that start with underscore (avgK, sbsQ, seed).
    m = re.search(
        r"antB([\d.]+)"
        r"(?:_nceA[\d.]+)?(?:_antM[\d.]+)?"
        r"_(ant[a-zA-Z0-9]+)"
        r"_(nce[a-zA-Z0-9]+)"
        r"((?:_avgK\d+)?)"
        r"((?:_sbs[^_]+)?)"
        r"(?:_s\d+(?:_v\d+)?)?$",
        exp_name,
    )
    if not m:
        return exp_name
    beta, ant_mode, nce_mode, avgk_part, sbs_part = m.groups()
    name = f"β={beta} {_MODE_SHORT.get(ant_mode, ant_mode)} {_MODE_SHORT.get(nce_mode, nce_mode)}"
    if avgk_part:
        name += f" {avgk_part.lstrip('_')}"   # e.g. "avgK3"
    if sbs_part:
        name += " SBS"
    return name


_DATASET_LABELS = {
    "cifar100_10-10":       "CIFAR-100 10×10",
    "cifar100_50-10":       "CIFAR-100 50+10×5",
    "tiny_imagenet_20-20":  "Tiny-ImageNet 20×10",
    "tiny_imagenet_100-20": "Tiny-ImageNet 100+20×5",
}

# Canonical dataset order for cross-dataset tables (easier → harder)
_DATASET_ORDER = [
    "cifar100_10-10",
    "cifar100_50-10",
    "tiny_imagenet_20-20",
    "tiny_imagenet_100-20",
]

_DATASET_SHORT = {
    "cifar100_10-10":       "C100 10-10",
    "cifar100_50-10":       "C100 50-10",
    "tiny_imagenet_20-20":  "TIN 20-20",
    "tiny_imagenet_100-20": "TIN 100-20",
}

def dataset_label(dataset: str) -> str:
    return _DATASET_LABELS.get(dataset, dataset)


def is_true_baseline(exp_name: str) -> bool:
    """β=0 antGlobal nceGlobal — the pure InfoNCE baseline with no ANT."""
    return bool(re.search(r"antB0(?:_nceA[\d.]+)?_antGlobal_nceGlobal", exp_name))


def extract_seed(exp_name: str):
    m = re.search(r"_s(\d+)(?:_v\d+)?$", exp_name)
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
    for exp in tqdm(sorted(os.listdir(logs_dir)), desc="Scanning logs", unit="exp"):
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


def df_to_md(df: pd.DataFrame, col_specs: list, baseline_mask: list = None) -> str:
    """
    col_specs: list of (col_name, header, higher_is_better_or_None, digits)
    Numeric columns with higher_is_better != None get their best value bolded.
    baseline_mask: optional list of bool (same length as df); baseline rows are underlined.
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
    body = []
    for i in range(len(df)):
        cells = [cols_fmt[col][i] for col, _, _, _ in present]
        if baseline_mask is not None and baseline_mask[i]:
            cells = [f"<u>{c}</u>" for c in cells]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([hdr_row, sep_row] + body)


def _mean_from_cell(s: str):
    """Extract the leading number from a cell like '79.04 ± 0.56', '+0.43', '-0.51'.
    Returns None for non-numeric cells (e.g. '—', empty string).
    """
    m = re.match(r'\s*([+-]?[\d.]+)', s.strip())
    return float(m.group(1)) if m else None


def _dict_table_md(rows: list, baseline_labels: set, highlight_cols: dict = None) -> str:
    """Build a markdown table from a list of dicts.
    Underlines baseline rows; bolds the best value per highlighted column.

    highlight_cols: dict mapping column name -> higher_is_better (True/False).
    """
    if not rows:
        return "*No data.*"
    cols = list(rows[0].keys())

    # Find the best row index for each highlighted column
    best_idx: dict = {}
    if highlight_cols:
        for col, higher in highlight_cols.items():
            if col not in cols:
                continue
            parsed = [(_mean_from_cell(str(row.get(col, ""))), i) for i, row in enumerate(rows)]
            valid = [(v, i) for v, i in parsed if v is not None]
            if valid:
                best_idx[col] = (max if higher else min)(valid, key=lambda x: x[0])[1]

    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(":---" for _ in cols) + " |"
    body = []
    for i, row in enumerate(rows):
        cells = []
        for c in cols:
            cell = str(row.get(c, ""))
            if best_idx.get(c) == i:
                cell = f"**{cell}**"
            cells.append(cell)
        if row.get("Experiment", "") in baseline_labels:
            cells = [f"<u>{c}</u>" for c in cells]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([hdr, sep] + body)


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
# Section 0 — Cross-Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────

def _cross_dataset_section(df: pd.DataFrame) -> list:
    """Build a cross-dataset overview table with Acc Δ, NME Δ, and Fgt sub-rows.

    Three rows per method: Acc Δ, NME Δ, and Fgt (absolute).
    Δ = method mean − true baseline mean for that dataset.
    Baseline rows show absolute values (not Δ).
    Sorted by mean Acc Δ (unweighted across tested datasets), descending.
    """
    lines = []
    lines.append("# Section 0 — Cross-Dataset Overview\n")
    lines.append(
        "> Each method has 3 sub-rows: **Acc Δ** / **NME Δ** (vs β=0 aGlobal nGlobal baseline) and **Fgt** (absolute, lower is better).  \n"
        "> `mean ± std` shown when tested with ≥2 seeds (sample std, ddof=1); single value for 1 seed.  \n"
        "> `—` = method not tested on that dataset.  \n"
        "> Mean = unweighted mean of Δ (for Acc/NME) or mean absolute Fgt across tested datasets.  \n"
        "> Sorted by mean Acc Δ descending.  \n"
        "> <u>Underlined rows</u> = true baseline; shows absolute Avg Acc1 / NME1 (reference).\n"
    )

    # Baseline mean per dataset (acc, nme, fgt)
    base_means: dict = {}
    for ds in _DATASET_ORDER:
        bl = df[(df["dataset"] == ds) & df["is_baseline"]]
        base_means[ds] = {
            "acc": float(bl["avg_acc"].mean()) if not bl.empty else np.nan,
            "nme": float(bl["avg_nme"].mean()) if not bl.empty else np.nan,
            "fgt": float(bl["fgt_acc"].mean()) if not bl.empty else np.nan,
        }

    # Identify the true baseline label
    bl_rows = df[df["is_baseline"]]
    baseline_label = bl_rows["label"].mode().iloc[0] if not bl_rows.empty else None

    # Collect unique labels in a stable order (first appearance across datasets)
    seen: set = set()
    all_labels: list = []
    for ds in _DATASET_ORDER:
        for lbl in df[df["dataset"] == ds]["label"].tolist():
            if lbl not in seen:
                all_labels.append(lbl)
                seen.add(lbl)

    # Build per-cell data storing raw stats for all three metrics
    # cell[lbl][ds] = {"is_baseline", "n",
    #                  "acc_mean", "acc_std", "nme_mean", "nme_std",
    #                  "fgt_mean", "fgt_std"}
    cell: dict = {}
    label_seeds: dict = {}
    for lbl in all_labels:
        cell[lbl] = {}
        all_seeds: set = set()
        for ds in _DATASET_ORDER:
            sub = df[(df["dataset"] == ds) & (df["label"] == lbl)]
            if sub.empty:
                cell[lbl][ds] = None
                continue
            is_base = bool(sub["is_baseline"].all())
            all_seeds.update(sub["seed"].dropna().astype(int).tolist())

            def _raw(col: str):
                arr = sub[col].dropna().values.astype(float)
                if len(arr) == 0:
                    return np.nan, np.nan
                return float(np.mean(arr)), (float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan)

            acc_m, acc_s = _raw("avg_acc")
            nme_m, nme_s = _raw("avg_nme")
            fgt_m, fgt_s = _raw("fgt_acc")
            cell[lbl][ds] = {
                "is_baseline": is_base,
                "n": len(sub),
                "acc_mean": acc_m, "acc_std": acc_s,
                "nme_mean": nme_m, "nme_std": nme_s,
                "fgt_mean": fgt_m, "fgt_std": fgt_s,
            }
        label_seeds[lbl] = sorted(all_seeds)

    # Sort non-baseline labels by unweighted mean Acc Δ (desc)
    def _mean_delta_acc(lbl: str) -> float:
        vals = []
        for ds in _DATASET_ORDER:
            d = cell[lbl].get(ds)
            bm = base_means[ds]["acc"]
            if d and not d["is_baseline"] and not np.isnan(d["acc_mean"]) and not np.isnan(bm):
                vals.append(d["acc_mean"] - bm)
        return float(np.mean(vals)) if vals else np.nan

    non_baseline = [l for l in all_labels if l != baseline_label]
    non_baseline.sort(key=_mean_delta_acc, reverse=True)
    ordered = ([baseline_label] if baseline_label else []) + non_baseline

    ds_cols = [_DATASET_SHORT.get(ds, ds) for ds in _DATASET_ORDER]
    # Internal columns (prefixed with "_") are used for logic only, not rendered
    cols        = ["Experiment", "Seeds", "Metric"] + ds_cols + ["Mean"]
    render_cols = cols  # all rendered

    def _fmt_val(mean: float, std: float, n: int, signed: bool) -> str:
        if np.isnan(mean):
            return "—"
        prefix = ("+" if mean >= 0 else "") if signed else ""
        if n > 1 and not np.isnan(std):
            return f"{prefix}{mean:.2f} ± {std:.2f}"
        return f"{prefix}{mean:.2f}"

    # Build rows — 3 sub-rows per method (Acc Δ, NME Δ, Fgt)
    rows = []  # each dict also carries "_lbl" and "_metric" for logic
    for lbl in ordered:
        seeds_str = ", ".join(str(s) for s in label_seeds.get(lbl, []))
        is_base   = (lbl == baseline_label)

        row_acc = {"Experiment": lbl, "Seeds": seeds_str,
                   "Metric": "Acc" if is_base else "Acc Δ",
                   "_lbl": lbl, "_metric": "acc"}
        row_nme = {"Experiment": "", "Seeds": "",
                   "Metric": "NME" if is_base else "NME Δ",
                   "_lbl": lbl, "_metric": "nme"}
        row_fgt = {"Experiment": "", "Seeds": "",
                   "Metric": "Fgt",
                   "_lbl": lbl, "_metric": "fgt"}

        acc_delta_vals, nme_delta_vals, fgt_vals = [], [], []

        for ds, col in zip(_DATASET_ORDER, ds_cols):
            d = cell[lbl].get(ds)
            if d is None:
                row_acc[col] = row_nme[col] = row_fgt[col] = "—"
                continue

            n = d["n"]
            bm_acc = base_means[ds]["acc"]
            bm_nme = base_means[ds]["nme"]

            if is_base:
                row_acc[col] = _fmt_val(d["acc_mean"], d["acc_std"], n, signed=False)
                row_nme[col] = _fmt_val(d["nme_mean"], d["nme_std"], n, signed=False)
            else:
                if not np.isnan(bm_acc) and not np.isnan(d["acc_mean"]):
                    delta = d["acc_mean"] - bm_acc
                    row_acc[col] = _fmt_val(delta, d["acc_std"], n, signed=True)
                    acc_delta_vals.append(delta)
                else:
                    row_acc[col] = "—"
                if not np.isnan(bm_nme) and not np.isnan(d["nme_mean"]):
                    delta = d["nme_mean"] - bm_nme
                    row_nme[col] = _fmt_val(delta, d["nme_std"], n, signed=True)
                    nme_delta_vals.append(delta)
                else:
                    row_nme[col] = "—"

            # Fgt: always absolute
            row_fgt[col] = _fmt_val(d["fgt_mean"], d["fgt_std"], n, signed=False)
            if not np.isnan(d["fgt_mean"]):
                fgt_vals.append(d["fgt_mean"])

        # Mean column
        if is_base:
            row_acc["Mean"] = row_nme["Mean"] = "—"
        else:
            def _mean_str(vals, signed=True):
                if not vals:
                    return "—"
                mv = float(np.mean(vals))
                return f"{'+' if mv >= 0 else ''}{mv:.2f}" if signed else f"{mv:.2f}"
            row_acc["Mean"] = _mean_str(acc_delta_vals)
            row_nme["Mean"] = _mean_str(nme_delta_vals)
        row_fgt["Mean"] = f"{np.mean(fgt_vals):.2f}" if fgt_vals else "—"

        rows.extend([row_acc, row_nme, row_fgt])

    # Best value per (dataset col, metric) — higher for Acc/NME Δ, lower for Fgt
    best_idx: dict = {}
    for col in ds_cols + ["Mean"]:
        for metric_key, higher in [("Acc Δ", True), ("NME Δ", True), ("Fgt", False)]:
            bv, bi = None, None
            for i, row in enumerate(rows):
                if row["_lbl"] == baseline_label:
                    continue
                if row["_metric"] != metric_key.replace(" Δ", "").lower():
                    continue
                val = _mean_from_cell(row.get(col, ""))
                if val is not None and (bv is None or (higher and val > bv) or (not higher and val < bv)):
                    bv, bi = val, i
            if bi is not None:
                best_idx[(col, metric_key)] = bi

    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(":---" for _ in cols) + " |"
    body = []
    for i, row in enumerate(rows):
        is_base_row = (row["_lbl"] == baseline_label)
        cells = [str(row.get(c, "")) for c in cols]
        metric_str = row["_metric"]  # "acc", "nme", "fgt"
        # Map internal metric key to lookup key used in best_idx
        _lookup = {"acc": "Acc Δ", "nme": "NME Δ", "fgt": "Fgt"}
        lookup_key = _lookup[metric_str]
        for j, col in enumerate(cols):
            if col in ds_cols + ["Mean"] and not is_base_row:
                if best_idx.get((col, lookup_key)) == i:
                    cells[j] = f"**{cells[j]}**"
        if is_base_row:
            cells = [f"<u>{c}</u>" for c in cells]
        body.append("| " + " | ".join(cells) + " |")
    lines.append("\n".join([hdr, sep] + body))
    lines.append("")

    # Summary: best method per dataset (by Acc Δ)
    lines.append("### Melhor método por dataset\n")
    summ_hdr = "| Dataset | Baseline Acc | Best Method | Best Acc Δ |"
    summ_sep = "| :--- | :--- | :--- | :--- |"
    summ_body = []
    for ds, col in zip(_DATASET_ORDER, ds_cols):
        bm_acc = base_means[ds]["acc"]
        bm_str = f"{bm_acc:.2f}%" if not np.isnan(bm_acc) else "—"
        best_lbl, best_delta = None, None
        for lbl in non_baseline:
            d = cell[lbl].get(ds)
            if d and not d["is_baseline"] and not np.isnan(d["acc_mean"]) and not np.isnan(bm_acc):
                delta = d["acc_mean"] - bm_acc
                if best_delta is None or delta > best_delta:
                    best_delta, best_lbl = delta, lbl
        best_str = f"+{best_delta:.2f}" if best_delta is not None and best_delta >= 0 else (f"{best_delta:.2f}" if best_delta is not None else "—")
        summ_body.append(f"| {col} | {bm_str} | {best_lbl or '—'} | {best_str} |")
    lines.append("\n".join([summ_hdr, summ_sep] + summ_body))
    lines.append("")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Results
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std_str(vals, digits: int = 2) -> str:
    """Format a list of floats as 'mean ± std' (or just 'mean' for a single value).

    Uses sample std (ddof=1, Bessel-corrected) as is standard in papers.
    """
    arr = np.array([v for v in vals if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return ""
    if arr.size == 1:
        return f"{arr[0]:.{digits}f}"
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}"


def _aggregated_section(g: pd.DataFrame, n: int, base_mean_acc: float,
                         base_mean_nme: float) -> list:
    """
    Build a mean±std summary table and per-task curve tables, grouped by variant label.
    Variants appear in the same sort order as the input df (first occurrence).
    """
    lines = []
    n_base_seeds = int(g["is_baseline"].sum())

    # Build label order: baselines pinned first, then non-baselines sorted by
    # mean avg_acc descending (best-performing variant first).
    _baseline_lbl_set = set(g[g["is_baseline"]]["label"].unique())
    seen_set: set = set()
    _all_seen: list = []
    for lbl in g["label"]:
        if lbl not in seen_set:
            _all_seen.append(lbl)
            seen_set.add(lbl)
    _baseline_seen = [l for l in _all_seen if l in _baseline_lbl_set]
    _non_baseline_seen = [l for l in _all_seen if l not in _baseline_lbl_set]
    _non_baseline_seen.sort(
        key=lambda lbl: g[g["label"] == lbl]["avg_acc"].mean(), reverse=True
    )
    seen: list = _baseline_seen + _non_baseline_seen

    baseline_labels = set(g[g["is_baseline"]]["label"].unique())

    agg_rows = []
    for lbl in seen:
        sub = g[g["label"] == lbl]
        seeds = sorted(sub["seed"].dropna().astype(int).tolist())
        seed_str = ", ".join(str(s) for s in seeds)
        is_base  = bool(sub["is_baseline"].all())
        row: dict = {
            "Experiment": lbl,
            "Seeds":      seed_str,
            "Avg Acc1 (mean ± std)": _mean_std_str(sub["avg_acc"].tolist()),
            "Avg NME1 (mean ± std)": _mean_std_str(sub["avg_nme"].tolist()),
            "Fgt Acc (mean ± std)":  _mean_std_str(sub["fgt_acc"].tolist()),
        }
        if not np.isnan(base_mean_acc):
            if is_base:
                row[f"∆ Acc vs baseline mean ({n_base_seeds} seeds)"] = "—"
            else:
                delta_vals = sub["avg_acc"].values - base_mean_acc
                row[f"∆ Acc vs baseline mean ({n_base_seeds} seeds)"] = _mean_std_str(delta_vals.tolist())
        agg_rows.append(row)

    agg_highlight = {
        "Avg Acc1 (mean ± std)": True,
        "Avg NME1 (mean ± std)": True,
        "Fgt Acc (mean ± std)": False,
    }
    if not np.isnan(base_mean_acc):
        agg_highlight[f"∆ Acc vs baseline mean ({n_base_seeds} seeds)"] = True
    lines.append(_dict_table_md(agg_rows, baseline_labels, highlight_cols=agg_highlight))
    lines.append("")

    # ── Per-task Acc1 curve (mean ± std) ─────────────────────────────────────
    acc_task_cols = [f"acc_T{i}" for i in range(1, n + 1) if f"acc_T{i}" in g.columns]
    if acc_task_cols:
        lines.append("#### Acc1 curve per task (mean ± std across seeds)\n")
        curve_rows = []
        for lbl in seen:
            sub = g[g["label"] == lbl]
            seeds = sorted(sub["seed"].dropna().astype(int).tolist())
            row = {"Experiment": lbl, "Seeds": ", ".join(str(s) for s in seeds)}
            for col in acc_task_cols:
                row[col.replace("acc_", "")] = _mean_std_str(sub[col].tolist())
            curve_rows.append(row)
        acc_highlight = {col.replace("acc_", ""): True for col in acc_task_cols}
        lines.append(_dict_table_md(curve_rows, baseline_labels, highlight_cols=acc_highlight))
        lines.append("")

    # ── Per-task NME1 curve (mean ± std) ─────────────────────────────────────
    nme_task_cols = [f"nme_T{i}" for i in range(1, n + 1) if f"nme_T{i}" in g.columns]
    if nme_task_cols:
        lines.append("#### NME1 curve per task (mean ± std across seeds)\n")
        curve_rows = []
        for lbl in seen:
            sub = g[g["label"] == lbl]
            seeds = sorted(sub["seed"].dropna().astype(int).tolist())
            row = {"Experiment": lbl, "Seeds": ", ".join(str(s) for s in seeds)}
            for col in nme_task_cols:
                row[col.replace("nme_", "")] = _mean_std_str(sub[col].tolist())
            curve_rows.append(row)
        nme_highlight = {col.replace("nme_", ""): True for col in nme_task_cols}
        lines.append(_dict_table_md(curve_rows, baseline_labels, highlight_cols=nme_highlight))
        lines.append("")

    return lines


def _results_section(df: pd.DataFrame) -> list:
    lines = []
    lines.append("# Section 1 — Results\n")
    lines.append(
        "> Metrics sourced from `exp_gistlog.log`.  "
        "**Bold** = best individual-run value in the group.  \n"
        "> Forgetting = `max(per-task history) − final value`, mean across all tasks "
        "except the last (which has never been forgotten yet).  \n"
        "> ∆ Acc vs Baseline is computed against the **mean Avg Acc1 across all baseline seeds**.  \n"
        "> Aggregated tables show **mean ± std across seeds** (seeds 1993, 1994, 1995).  \n"
        "> <u>Underlined rows</u> = true baseline (β=0 aGlobal nGlobal).\n"
    )

    for dataset in sorted(df["dataset"].unique()):
        g = df[df["dataset"] == dataset].copy()
        n  = int(g["num_tasks"].max())
        dl = dataset_label(dataset)

        # ── Compute delta vs true baseline (mean across seeds) ────────────────
        baseline_rows = g[g["is_baseline"]]
        base_mean_acc = baseline_rows["avg_acc"].mean() if not baseline_rows.empty else np.nan
        base_mean_nme = baseline_rows["avg_nme"].mean() if not baseline_rows.empty else np.nan
        n_base_seeds  = int(g["is_baseline"].sum())
        g["delta_acc"] = np.where(g["is_baseline"], np.nan, g["avg_acc"] - base_mean_acc)
        g["delta_nme"] = np.where(g["is_baseline"], np.nan, g["avg_nme"] - base_mean_nme)

        # ── Sort: β=0 aGlobal nGlobal pinned first, then others by label asc ──
        g["_sort"] = (~g["is_baseline"]).astype(int)
        g = (
            g.sort_values(["_sort", "label"], ascending=[True, True])
            .drop(columns=["_sort"])
            .reset_index(drop=True)
        )

        lines.append(f"## {dl}\n")
        n_base = int(g["is_baseline"].sum())
        n_exp  = len(g) - n_base
        base_info = (
            f" | True baseline (β=0 nGlobal): **{n_base}** seeds, "
            f"mean Avg Acc1 = **{base_mean_acc:.2f}%** "
            f"± {baseline_rows['avg_acc'].std():.2f}%"
        ) if not np.isnan(base_mean_acc) else ""
        lines.append(
            f"Complete experiments: **{len(g)}** individual runs "
            f"({n_base} baseline + {n_exp} variant runs) "
            f"| Tasks per session: **{n}**{base_info}\n"
        )

        # ── Aggregated: mean ± std per variant ────────────────────────────────
        lines.append("### Aggregated Results (mean ± std across seeds)\n")
        if not np.isnan(base_mean_acc):
            lines.append(
                f"> ∆ Acc computed against baseline mean = **{base_mean_acc:.2f}%** "
                f"(mean of {n_base_seeds} seed runs).\n"
            )
        lines.extend(_aggregated_section(g, n, base_mean_acc, base_mean_nme))

        # ── Per-seed detail (collapsible) ─────────────────────────────────────
        lines.append("<details>")
        lines.append("<summary>Per-seed individual runs</summary>\n")

        delta_hdr = f"∆ Acc vs Baseline mean ({n_base_seeds} seeds)" if not np.isnan(base_mean_acc) else None
        delta_spec = [("delta_acc", delta_hdr, True, 2)] if delta_hdr else []

        lines.append("#### Accuracy & Forgetting (per seed)\n")
        baseline_mask = g["is_baseline"].tolist()

        lines.append(df_to_md(g, [
            ("label",   "Experiment",      None,  0),
            ("seed",    "Seed",            None,  0),
            ("avg_acc", "Avg Acc1 (%)",    True,  2),
            ("avg_nme", "Avg NME1 (%)",    True,  2),
            ("fgt_acc", "Forgetting Acc",  False, 2),
            ("fgt_nme", "Forgetting NME",  False, 2),
        ] + delta_spec, baseline_mask=baseline_mask))
        lines.append("")

        acc_specs = [(f"acc_T{i}", f"T{i}", True, 2) for i in range(1, n + 1) if f"acc_T{i}" in g.columns]
        lines.append("#### Acc1 Curve per seed\n")
        lines.append(df_to_md(g, [("label", "Experiment", None, 0), ("seed", "Seed", None, 0)] + acc_specs, baseline_mask=baseline_mask))
        lines.append("")

        nme_specs = [(f"nme_T{i}", f"T{i}", True, 2) for i in range(1, n + 1) if f"nme_T{i}" in g.columns]
        if nme_specs:
            lines.append("#### NME1 Curve per seed\n")
            lines.append(df_to_md(g, [("label", "Experiment", None, 0), ("seed", "Seed", None, 0)] + nme_specs, baseline_mask=baseline_mask))
            lines.append("")

        lines.append("</details>\n")

        """
        # ── Best callouts ─────────────────────────────────────────────────────
        lines.append("### Highlights\n")
        non_base = g[~g["is_baseline"]]
        cmp_g    = non_base if not non_base.empty else g
        best_acc_row = cmp_g.sort_values("avg_acc", ascending=False).iloc[0]
        best_nme_row = cmp_g.sort_values("avg_nme", ascending=False).iloc[0]
        best_fgt_row = cmp_g.sort_values(["fgt_acc", "avg_acc"], ascending=[True, False]).iloc[0]
        lines.append(f"| Metric | Best Experiment (seed) | Value |")
        lines.append(f"| :--- | :--- | ---: |")
        lines.append(f"| ⭐ Highest Avg Acc1 | `{best_acc_row['label']}` (s{int(best_acc_row['seed'])}) | **{best_acc_row['avg_acc']:.2f}%** |")
        lines.append(f"| ⭐ Highest Avg NME1 | `{best_nme_row['label']}` (s{int(best_nme_row['seed'])}) | **{best_nme_row['avg_nme']:.2f}%** |")
        lines.append(f"| ⭐ Lowest Forgetting | `{best_fgt_row['label']}` (s{int(best_fgt_row['seed'])}) | **{best_fgt_row['fgt_acc']:.2f}%** |")
        if not np.isnan(base_mean_acc) and not non_base.empty:
            best_delta_row = non_base.sort_values("delta_acc", ascending=False).iloc[0]
            lines.append(
                f"| 📈 Best ∆ vs Baseline | `{best_delta_row['label']}` (s{int(best_delta_row['seed'])}) "
                f"| **{best_delta_row['delta_acc']:+.2f}%** "
                f"(baseline mean: {base_mean_acc:.2f}%) |"
            )
        lines.append("")
        """

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
# HTML/PDF rendering
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_HTML = os.environ.get("TAGFEX_RESULTS_HTML", "./results_report.html")
RESULTS_PDF  = os.environ.get("TAGFEX_RESULTS_PDF",  "./results_report.pdf")


def _html_escape(value) -> str:
    import html
    return html.escape(str(value), quote=True)


def _cross_dataset_section_html(df: pd.DataFrame) -> str:
    """Render Section 0 as a real HTML table with rowspan cells.

    Experiment and Seeds span the three metric rows, so internal horizontal
    separators start only at the Metric column.
    """
    # Baseline means per dataset
    base_means = {}
    for ds in _DATASET_ORDER:
        bl = df[(df["dataset"] == ds) & df["is_baseline"]]
        base_means[ds] = {
            "acc": float(bl["avg_acc"].mean()) if not bl.empty else np.nan,
            "nme": float(bl["avg_nme"].mean()) if not bl.empty else np.nan,
            "fgt": float(bl["fgt_acc"].mean()) if not bl.empty else np.nan,
        }

    bl_rows = df[df["is_baseline"]]
    baseline_label = bl_rows["label"].mode().iloc[0] if not bl_rows.empty else None

    seen, labels = set(), []
    for ds in _DATASET_ORDER:
        for lbl in df[df["dataset"] == ds]["label"].tolist():
            if lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)

    cell, label_seeds = {}, {}
    for lbl in labels:
        cell[lbl] = {}
        seeds = set()
        for ds in _DATASET_ORDER:
            sub = df[(df["dataset"] == ds) & (df["label"] == lbl)]
            if sub.empty:
                cell[lbl][ds] = None
                continue
            seeds.update(sub["seed"].dropna().astype(int).tolist())

            def raw(col):
                arr = sub[col].dropna().to_numpy(dtype=float)
                if len(arr) == 0:
                    return np.nan, np.nan
                return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else np.nan

            acc_m, acc_s = raw("avg_acc")
            nme_m, nme_s = raw("avg_nme")
            fgt_m, fgt_s = raw("fgt_acc")
            cell[lbl][ds] = {
                "is_baseline": bool(sub["is_baseline"].all()),
                "n": len(sub),
                "acc_mean": acc_m, "acc_std": acc_s,
                "nme_mean": nme_m, "nme_std": nme_s,
                "fgt_mean": fgt_m, "fgt_std": fgt_s,
            }
        label_seeds[lbl] = sorted(seeds)

    def mean_delta_acc(lbl):
        vals = []
        for ds in _DATASET_ORDER:
            d = cell[lbl].get(ds)
            bm = base_means[ds]["acc"]
            if d and not d["is_baseline"] and pd.notna(d["acc_mean"]) and pd.notna(bm):
                vals.append(d["acc_mean"] - bm)
        return float(np.mean(vals)) if vals else -np.inf

    non_baseline = [lbl for lbl in labels if lbl != baseline_label]
    non_baseline.sort(key=mean_delta_acc, reverse=True)
    ordered = ([baseline_label] if baseline_label else []) + non_baseline

    def fmt(mean, std, n, signed=False):
        if pd.isna(mean):
            return "—"
        sign = "+" if signed and mean >= 0 else ""
        if n > 1 and pd.notna(std):
            return f"{sign}{mean:.2f} ± {std:.2f}"
        return f"{sign}{mean:.2f}"

    rows = []
    for lbl in ordered:
        is_base = lbl == baseline_label
        metrics = {"acc": {}, "nme": {}, "fgt": {}}
        acc_vals, nme_vals, fgt_vals = [], [], []
        for ds in _DATASET_ORDER:
            d = cell[lbl].get(ds)
            if d is None:
                metrics["acc"][ds] = metrics["nme"][ds] = metrics["fgt"][ds] = "—"
                continue
            n = d["n"]
            if is_base:
                metrics["acc"][ds] = fmt(d["acc_mean"], d["acc_std"], n)
                metrics["nme"][ds] = fmt(d["nme_mean"], d["nme_std"], n)
            else:
                da = d["acc_mean"] - base_means[ds]["acc"]
                dn = d["nme_mean"] - base_means[ds]["nme"]
                metrics["acc"][ds] = fmt(da, d["acc_std"], n, signed=True)
                metrics["nme"][ds] = fmt(dn, d["nme_std"], n, signed=True)
                acc_vals.append(da); nme_vals.append(dn)
            metrics["fgt"][ds] = fmt(d["fgt_mean"], d["fgt_std"], n)
            if pd.notna(d["fgt_mean"]):
                fgt_vals.append(d["fgt_mean"])

        metrics["acc"]["Mean"] = "—" if is_base else f"{np.mean(acc_vals):+.2f}"
        metrics["nme"]["Mean"] = "—" if is_base else f"{np.mean(nme_vals):+.2f}"
        metrics["fgt"]["Mean"] = f"{np.mean(fgt_vals):.2f}" if fgt_vals else "—"
        rows.append({
            "label": lbl,
            "seeds": ", ".join(map(str, label_seeds.get(lbl, []))),
            "baseline": is_base,
            "metrics": metrics,
        })

    # Determine best non-baseline values per metric/column.
    best = set()
    columns = _DATASET_ORDER + ["Mean"]
    for metric, higher in (("acc", True), ("nme", True), ("fgt", False)):
        for col in columns:
            candidates = []
            for ri, row in enumerate(rows):
                if row["baseline"]:
                    continue
                val = _mean_from_cell(row["metrics"][metric][col])
                if val is not None:
                    candidates.append((val, ri))
            if candidates:
                target = (max if higher else min)(candidates, key=lambda x: x[0])[0]
                for val, ri in candidates:
                    if abs(val - target) < 1e-9:
                        best.add((ri, metric, col))

    out = [
        '<section class="report-section cross-dataset">',
        '<h1>Section 0 — Cross-Dataset Overview</h1>',
        '<div class="callout">'
        'Each method has three sub-rows: <strong>Acc Δ</strong>, '
        '<strong>NME Δ</strong>, and <strong>Fgt</strong>. '
        'Experiment and Seeds use <code>rowspan="3"</code>, so internal separators '
        'start at the Metric column. Mean ± std is shown for two or more seeds.'
        '</div>',
        '<div class="table-wrap"><table class="overview-table">',
        '<colgroup><col class="experiment"><col class="seeds"><col class="metric">'
        '<col class="dataset"><col class="dataset"><col class="dataset">'
        '<col class="dataset"><col class="mean"></colgroup>',
        '<thead><tr><th>Experiment</th><th>Seeds</th><th>Metric</th>'
        + ''.join(f'<th>{_html_escape(_DATASET_SHORT.get(ds, ds))}</th>' for ds in _DATASET_ORDER)
        + '<th>Mean</th></tr></thead><tbody>'
    ]

    metric_labels = {
        "acc": ("Acc" if baseline_label else "Acc", "Acc Δ"),
        "nme": ("NME", "NME Δ"),
        "fgt": ("Fgt", "Fgt"),
    }
    for ri, row in enumerate(rows):
        row_class = " baseline" if row["baseline"] else ""
        for mi, metric in enumerate(("acc", "nme", "fgt")):
            classes = f'group-start{row_class}' if mi == 0 else f'subrow{row_class}'
            out.append(f'<tr class="{classes.strip()}">')
            if mi == 0:
                out.append(f'<td class="experiment" rowspan="3">{_html_escape(row["label"])}</td>')
                out.append(f'<td class="seeds" rowspan="3">{_html_escape(row["seeds"])}</td>')
            label = metric_labels[metric][0 if row["baseline"] else 1]
            out.append(f'<td class="metric">{_html_escape(label)}</td>')
            for col in columns:
                value = _html_escape(row["metrics"][metric][col])
                is_best = (ri, metric, col) in best
                if is_best:
                    value = f'<strong>{value}</strong>'
                td_class = ' class="best"' if is_best else ''
                out.append(f'<td{td_class}>{value}</td>')
            out.append('</tr>')
    out.append('</tbody></table></div>')

    out.extend([
        '<h3>Melhor método por dataset</h3>',
        '<div class="table-wrap"><table class="summary-table">',
        '<thead><tr><th>Dataset</th><th>Baseline Acc</th><th>Best Method</th><th>Best Acc Δ</th></tr></thead><tbody>'
    ])
    for ds in _DATASET_ORDER:
        bm = base_means[ds]["acc"]
        best_lbl, best_delta = None, None
        for lbl in non_baseline:
            d = cell[lbl].get(ds)
            if d and pd.notna(d["acc_mean"]) and pd.notna(bm):
                delta = d["acc_mean"] - bm
                if best_delta is None or delta > best_delta:
                    best_lbl, best_delta = lbl, delta
        out.append(
            '<tr>'
            f'<td>{_html_escape(_DATASET_SHORT.get(ds, ds))}</td>'
            f'<td>{bm:.2f}%</td>'
            f'<td>{_html_escape(best_lbl or "—")}</td>'
            f'<td>{best_delta:+.2f}</td>'
            '</tr>'
        )
    out.append('</tbody></table></div></section>')
    return ''.join(out)


def _markdown_to_html(markdown_text: str) -> str:
    try:
        import mistune
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'mistune'. Install with: pip install mistune"
        ) from exc

    renderer = mistune.HTMLRenderer(escape=False)
    md = mistune.create_markdown(renderer=renderer, plugins=["table", "strikethrough"])
    body = md(markdown_text)

    # PDF engines do not consistently implement interactive <details>.
    # Render every collapsible section expanded while keeping a visible header.
    body = body.replace("<details>", '<section class="details-block">')
    body = body.replace("</details>", "</section>")
    body = body.replace("<summary>", '<div class="details-summary">')
    body = body.replace("</summary>", "</div>")
    # Strip inline text-align styles added by mistune so CSS rules take effect.
    # mistune marks every <td>/<th> with style="text-align:..." based on the
    # :--- / ---: separator, overriding our stylesheet centering.
    body = re.sub(r' style="text-align:[^"]*"', '', body)
    # Highlight best-value cells: mistune renders **bold** inside <td> as <td style="..."><strong>…</strong></td>.
    # After stripping inline styles above, best cells are now <td><strong>….
    # Baseline cells are wrapped in <u> so they become <td><u><strong>… and are not matched.
    # Skip cells that already have a class attribute (e.g. the overview table cells tagged earlier).
    body = re.sub(r'<td(?![^>]*class)([^>]*)>(<strong>)', r'<td\1 class="best">\2', body)
    return body


def _report_css() -> str:
    return r"""
@page {
  size: A4 landscape;
  margin: 12mm 10mm 14mm;
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: 8pt; color: #64748b; }
}
:root {
  --border: #94a3b8;
  --border-soft: #cbd5e1;
  --header: #cfe3f1;
  --group: #f8fafc;
  --baseline: #fff7d6;
  --text: #0f172a;
  --muted: #475569;
  --accent: #0f5132;
}
* { box-sizing: border-box; }
html { font-size: 10pt; }
body {
  margin: 0;
  color: var(--text);
  font-family: "DejaVu Sans", Arial, sans-serif;
  line-height: 1.35;
}
h1, h2, h3, h4 { page-break-after: avoid; break-after: avoid; color: #172554; }
h1 { font-size: 20pt; margin: 0 0 8pt; border-bottom: 2px solid #93c5fd; padding-bottom: 5pt; }
h2 { font-size: 16pt; margin: 18pt 0 7pt; }
h3 { font-size: 13pt; margin: 14pt 0 6pt; }
h4 { font-size: 11pt; margin: 11pt 0 5pt; }
p { margin: 4pt 0 7pt; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f1f5f9; padding: 7pt; border: 1px solid #cbd5e1; }
blockquote, .callout {
  margin: 7pt 0 10pt;
  padding: 7pt 9pt;
  background: #eff6ff;
  border-left: 4px solid #60a5fa;
  color: var(--muted);
}
hr { border: 0; border-top: 1px solid #94a3b8; margin: 16pt 0; }
.table-wrap { width: 100%; margin: 7pt 0 11pt; }
table {
  width: auto;
  max-width: 100%;
  border-collapse: collapse;
  table-layout: auto;
  font-size: 7.4pt;
  break-inside: auto;
}
.overview-table { width: 100%; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th, td {
  border: 0.6pt solid var(--border);
  padding: 3.2pt 4pt;
  text-align: center;
  vertical-align: middle;
  overflow-wrap: anywhere;
}
th { background: var(--header); font-weight: 700; }
td:first-child, th:first-child { text-align: left; }
.overview-table col.experiment { width: 21%; }
.overview-table col.seeds { width: 12%; }
.overview-table col.metric { width: 7%; }
.overview-table col.dataset { width: 12.5%; }
.overview-table col.mean { width: 10%; }
.overview-table td.experiment { text-align: left; font-weight: 600; background: var(--group); }
.overview-table td.seeds { background: var(--group); white-space: nowrap; }
.overview-table td.metric { font-weight: 700; background: #fbfdff; }
.overview-table tr.group-start td { border-top-width: 1.2pt; }
.overview-table tr.baseline td { background: var(--baseline); text-decoration: underline; }
td.best { background-color: #dcfce7; }
td.best strong { color: #166534; }
strong { color: var(--accent); }
.details-block {
  display: block;
  border: 0.7pt solid var(--border-soft);
  margin: 8pt 0 12pt;
  padding: 6pt;
  break-inside: auto;
}
.details-summary {
  display: block;
  font-weight: 700;
  background: #e2e8f0;
  padding: 5pt 7pt;
  margin: -6pt -6pt 7pt;
}
.report-section { break-before: page; }
.report-section:first-child { break-before: auto; }
.cross-dataset { break-before: auto; }
.generated-at { font-size: 8pt; color: #64748b; margin-bottom: 8pt; }
@media screen {
  body { max-width: 1600px; margin: 0 auto; padding: 28px; background: white; }
  html { background: #e5e7eb; }
  .table-wrap { overflow-x: auto; }
  table { font-size: 12px; }
}
"""


def _build_html_document(section0_html: str, remaining_markdown: str, generated_at: str) -> str:
    body = section0_html + _markdown_to_html(remaining_markdown)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TagFex Experiment Report</title>
<style>{_report_css()}</style>
</head>
<body>
<div class="generated-at">Generated {generated_at}</div>
{body}
</body>
</html>
"""


def _write_pdf(html_path: str, pdf_path: str) -> None:
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'weasyprint'. Install with: pip install weasyprint"
        ) from exc

    base_url = os.path.abspath(os.path.dirname(html_path) or ".")
    HTML(filename=html_path, base_url=base_url).write_pdf(pdf_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate TagFex HTML + PDF experiment report."
    )
    parser.add_argument(
        "--short", action="store_true",
        help="Skip Section 2 (Debug Metrics) for a faster, lighter report.",
    )
    args = parser.parse_args()

    results_html = RESULTS_HTML
    results_pdf  = RESULTS_PDF
    if args.short:
        base, ext = os.path.splitext(RESULTS_HTML)
        results_html = f"{base}_short{ext}"
        base, ext = os.path.splitext(RESULTS_PDF)
        results_pdf  = f"{base}_short{ext}"

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

    print("=== Section 0: Cross-Dataset Overview (HTML rowspan table) ===")
    sec0_html = _cross_dataset_section_html(df)

    print("\n=== Section 1: Results ===")
    sec1 = _results_section(df)

    if args.short:
        print("\n[--short] Skipping Section 2 — Debug Metrics")
        sec2 = []
    else:
        print("\n=== Section 2: Debug ===")
        sec2 = _debug_section(df, LOGS_DIR)

    remaining_md = "\n".join([
        "---\n",
        *sec1,
        *([] if not sec2 else ["---\n", *sec2]),
    ])
    html_doc = _build_html_document(sec0_html, remaining_md, ts)

    with open(results_html, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    print(f"\nHTML report saved -> {results_html}")

    _write_pdf(results_html, results_pdf)
    print(f"PDF report saved  -> {results_pdf}")

    print("\n=== Quick Preview ===")
    for dataset in sorted(df["dataset"].unique()):
        g = df[df["dataset"] == dataset].copy()
        baseline_rows = g[g["is_baseline"]]
        base_mean = baseline_rows["avg_acc"].mean() if not baseline_rows.empty else float("nan")
        g["_s"] = (~g["is_baseline"]).astype(int)
        g = g.sort_values(["_s", "avg_acc"], ascending=[True, False]).drop(columns=["_s"])
        label = dataset_label(dataset)
        base_str = f"  baseline mean={base_mean:.2f}" if not np.isnan(base_mean) else ""
        print(f"\n[{label}]{base_str}")
        for _, row in g.iterrows():
            delta_str = ""
            if not row["is_baseline"] and not np.isnan(base_mean):
                delta_str = f"  Δ={row['avg_acc'] - base_mean:+.2f}"
            print(
                f"  {row['label']:35s}  avg_acc={row['avg_acc']:.2f}  "
                f"avg_nme={row['avg_nme']:.2f}  fgt={row['fgt_acc']:.2f}{delta_str}"
            )


if __name__ == "__main__":
    main()
