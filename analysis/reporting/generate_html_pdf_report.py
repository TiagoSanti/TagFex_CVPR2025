#!/usr/bin/env python3
"""
generate_html_pdf_report.py — HTML + PDF report for TagFex experiments.

Section 0 (Overview): Cross-dataset Δ Acc / NME / Fgt table with HTML rowspan.
Section 1 (Results):  Comparative acc/NME tables per dataset.
                      Best values in **bold**.  Forgetting metric included.
Section 2 (Debug):    Loss dynamics, ANT distance stats per experiment.
                      Reads exp_debug0.log from raw file or debug_logs.zip.
                      Skipped by default; included when --full is passed.

Reads:   LOGS_DIR  (env var, default ./logs)
Writes:  RESULTS_HTML / RESULTS_PDF  (env vars, default ./results_report.html/pdf)

Usage:
  python generate_html_pdf_report.py            # short report (Sections 0 and 1)
  python generate_html_pdf_report.py --short    # same as the default
  python generate_html_pdf_report.py --full     # include Section 2 (Debug)
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research_identity import guard_architecture_groups

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.provenance import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_hash,
    file_record,
    git_record,
    records_hash,
    utc_now,
)

LOGS_DIR   = os.environ.get("TAGFEX_LOGS_DIR", "./logs")

# Prefixes that mark quarantined or superseded experiment directories
_SKIP_PREFIXES = ("QUARANTINE_", "OLD_MEM_")
_SKIP = {"auto_experiments"}

_EXPECTED_TASKS = {
    "cifar100_10-10": 10,
    "cifar100_50-10": 6,
    "tiny_imagenet_20-20": 10,
    "tiny_imagenet_100-20": 6,
    "cub200_20-20": 10,
    "cub200_100-20": 6,
}

# ─────────────────────────────────────────────────────────────────────────────
# Experiment name utilities
# ─────────────────────────────────────────────────────────────────────────────

def infer_dataset(exp_name: str) -> str:
    m = re.match(r"(?:debug_exp_|exp_)(.+?)_antB", exp_name)
    return m.group(1) if m else "unknown"


_MODE_SHORT = {
    "antGlobal":       "ANT-IV-GR",
    "antLocal":        "ANT-IV-AR",
    "antSymmetricFull":"ANT-FS-AR",
    "antSymmetricFullGlobal":"ANT-FS-GR",
}

_LEGACY_ANT_TAG = " [legado: ref. conectada]"


def _ant_variant_metadata(exp_name: str) -> tuple[str, str]:
    """Return the report generation and reference-gradient semantics."""
    beta_match = re.search(r"(?:^|_)antB([\d.]+)(?:_|$)", exp_name)
    if beta_match is None or float(beta_match.group(1)) == 0:
        return "baseline", "not_applicable"
    if "_refDetached" in exp_name:
        return "canonical", "detached"
    return "legacy", "connected"

def short_name(exp_name: str) -> str:
    """Convert an experiment directory name to a compact human-readable label.

    Handles baseline/ANT/avgK/SBS variants.  Detached-reference ANT is the
    canonical report identity; historical connected-reference runs receive an
    explicit legacy tag.  The nGlobal/nLocal token is omitted because both
    implement the same InfoNCE.
    Examples:
        debug_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_refDetached_s1993
            → ANT-IV-AR (β=0.5)
        exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_refDetached_avgK3_s1993
            → ANT-IV-AR (β=0.5) + TeacherAvg-3
        exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_sbsQ0.20S0.20_s1993
            → ANT-IV-AR (β=0.5) [legado: ref. conectada] + SBS
    """
    # Use [a-zA-Z0-9]+ for mode tokens so we don't accidentally capture
    # optional suffix tokens that start with underscore (avgK, sbsQ, seed).
    m = re.search(
        r"antB([\d.]+)"
        r"(?:_nceA[\d.]+)?(?:_antM[\d.]+)?"
        r"_(ant[a-zA-Z0-9]+)"
        r"(?:_(nce(?:Global|Local)))?"
        r"((?:_refDetached)?)"
        r"((?:_avgK\d+)?)"
        r"((?:_sbs[^_]+)?)"
        r"(?:_s\d+(?:_v\d+)?)?$",
        exp_name,
    )
    if not m:
        return exp_name
    beta, ant_mode, nce_mode, ref_part, avgk_part, sbs_part = m.groups()
    if float(beta) == 0:
        name = "Baseline InfoNCE"
        if avgk_part:
            k = re.search(r"\d+", avgk_part).group()
            name += f" + TeacherAvg-{k}"
        return name
    name = f"{_MODE_SHORT.get(ant_mode, ant_mode)} (β={beta})"
    if not ref_part:
        name += _LEGACY_ANT_TAG
    if avgk_part:
        k = re.search(r"\d+", avgk_part).group()
        name += f" + TeacherAvg-{k}"
    if sbs_part:
        name += " + SBS"
    return name


_DATASET_LABELS = {
    "cifar100_10-10":       "CIFAR-100 10×10",
    "cifar100_50-10":       "CIFAR-100 50+10×5",
    "tiny_imagenet_20-20":  "Tiny-ImageNet 20×10",
    "tiny_imagenet_100-20": "Tiny-ImageNet 100+20×5",
    "cub200_20-20":         "CUB-200 20×10",
    "cub200_100-20":        "CUB-200 100+20×5",
}

# Canonical dataset order for cross-dataset tables (easier → harder)
_DATASET_ORDER = [
    "cifar100_10-10",
    "cifar100_50-10",
    "tiny_imagenet_20-20",
    "tiny_imagenet_100-20",
    "cub200_20-20",
    "cub200_100-20",
]

_DATASET_SHORT = {
    "cifar100_10-10":       "C100 10-10",
    "cifar100_50-10":       "C100 50-10",
    "tiny_imagenet_20-20":  "TIN 20-20",
    "tiny_imagenet_100-20": "TIN 100-20",
    "cub200_20-20":         "CUB 20-20",
    "cub200_100-20":        "CUB 100-20",
}

def dataset_label(dataset: str) -> str:
    return _DATASET_LABELS.get(dataset, dataset)


def is_true_baseline(exp_name: str) -> bool:
    """Identify only pure InfoNCE, excluding β=0 TeacherAvg ablations."""
    return "_avgK" not in exp_name and bool(re.search(
        r"antB0(?:_nceA[\d.]+)?_ant(?:Global|Local|SymmetricFull)",
        exp_name,
    ))


def _infonce_source_priority(exp_name: str) -> int:
    """Prefer new canonical names, then legacy nGlobal, then nLocal fallback."""
    if "_nceLocal" in exp_name:
        return 2
    if "_nceGlobal" in exp_name:
        return 1
    return 0


def _infonce_equivalence_key(exp_name: str) -> str:
    """Identity after erasing obsolete InfoNCE and inactive β=0 ANT choices."""
    key = re.sub(r"^debug_exp_|^exp_", "", exp_name)
    key = re.sub(r"_nce(?:Global|Local)", "", key)
    if re.search(r"_antB0(?:_|$)", key):
        key = re.sub(r"_ant(?:Global|Local|SymmetricFull)", "_baselineInfoNCE", key)
    return re.sub(r"_s\d+(?:_v\d+)?$", "", key)


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
        data = Path(path).read_bytes()
        text = data.decode("utf-8", errors="replace")
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
        "gistlog_sha256": hashlib.sha256(data).hexdigest(),
        "gistlog_size_bytes": len(data),
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
        if exp.startswith(_SKIP_PREFIXES):
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
            "ant_generation": _ant_variant_metadata(exp)[0],
            "ant_reference_gradient": _ant_variant_metadata(exp)[1],
            "avg_acc":     parsed["avg_acc"],
            "avg_nme":     parsed["avg_nme"],
            "fgt_acc":     fgt_acc,
            "fgt_nme":     fgt_nme,
            "num_tasks":   len(parsed["acc_curve"]),
            "gistlog_path": str(Path(gistlog).resolve()),
            "gistlog_sha256": parsed["gistlog_sha256"],
            "gistlog_size_bytes": parsed["gistlog_size_bytes"],
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
    for dataset, g in df.groupby("dataset"):
        expected = _EXPECTED_TASKS.get(dataset)
        n = expected if expected is not None else int(g["num_tasks"].max())
        acc_cols = [f"acc_T{i}" for i in range(1, n + 1)]
        g = g.copy()
        for c in acc_cols:
            if c not in g.columns:
                g[c] = np.nan
        parts.append(g[g["num_tasks"] == n].dropna(subset=acc_cols))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def deduplicate_equivalent_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one run per logical method/dataset/seed after normalization.

    Historical nGlobal is preferred because it matches the original InfoNCE.
    Historical nLocal is retained as a fallback when no canonical run exists.
    At beta zero, inactive ANT reference/coverage metadata is also erased.
    This prevents duplicate seeds from being counted as independent samples.
    """
    if df.empty:
        return df
    if 'gistlog_path' in df.columns:
        guard_architecture_groups([
            ((row['dataset'], _infonce_equivalence_key(row['exp'])), row['gistlog_path'])
            for _, row in df.iterrows()
        ])
    out = df.copy()
    out["_infonce_key"] = out["exp"].map(_infonce_equivalence_key)
    out["_infonce_priority"] = out["exp"].map(_infonce_source_priority)
    out["_debug_priority"] = out["exp"].str.startswith("debug_").astype(int)
    out = out.sort_values(
        [
            "dataset",
            "_infonce_key",
            "seed",
            "_infonce_priority",
            "_debug_priority",
            "exp",
        ],
        kind="stable",
    )
    out = out.drop_duplicates(
        subset=["dataset", "_infonce_key", "seed"], keep="first"
    )
    return out.drop(
        columns=["_infonce_key", "_infonce_priority", "_debug_priority"]
    ).reset_index(drop=True)


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
        cells = [str(cols_fmt[col][i]) for col, _, _, _ in present]
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

    Three rows per method: Acc Δ, NME Δ, and Fgt Δ.
    Δ = method mean − true baseline mean for that dataset.
    Baseline rows show absolute values (not Δ). Variants are ordered first by
    coverage and then by mean Acc Δ.
    """
    lines = []
    lines.append("# Section 0 — Cross-Dataset Overview\n")
    lines.append(
        "> Each method has 3 sub-rows: **Acc Δ** / **NME Δ** / **Fgt Δ** (vs Baseline InfoNCE; lower Fgt Δ is better).  \n"
        "> Historical nGlobal/nLocal runs share one method identity; beta-zero runs without TeacherAvg are Baseline InfoNCE, while beta-zero TeacherAvg runs remain separate ablations. Duplicate dataset/seed pairs use the canonical source and historical alternatives only as fallback.  \n"
        "> `mean ± std` shown when tested with ≥2 seeds (sample std, ddof=1); single value for 1 seed.  \n"
        "> `—` = method not tested on that dataset.  \n"
        "> **n/ds** = seed count per dataset column (left → right); `—` = not tested.  \n"
        "> Mean = unweighted mean across tested datasets: absolute metrics for the baseline and Δ for variants.  \n"
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
    # Primary sort: more seeds first; secondary: higher Acc Δ
    def _total_seeds_md(lbl):
        return sum(cell[lbl][ds]["n"] for ds in _DATASET_ORDER if cell[lbl].get(ds))
    non_baseline.sort(key=lambda lbl: (_total_seeds_md(lbl), _mean_delta_acc(lbl)), reverse=True)
    ordered = ([baseline_label] if baseline_label else []) + non_baseline

    ds_cols = [_DATASET_SHORT.get(ds, ds) for ds in _DATASET_ORDER]
    # Internal columns (prefixed with "_") are used for logic only, not rendered
    cols        = ["Experiment", "n/ds", "Metric"] + ds_cols + ["Mean"]
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
        # Per-dataset seed count (left→right, matching ds_cols order)
        _counts = [str(cell[lbl][ds]["n"]) if cell[lbl].get(ds) else "—" for ds in _DATASET_ORDER]
        seeds_str = "/".join(_counts)
        is_base   = (lbl == baseline_label)

        row_acc = {"Experiment": lbl, "n/ds": seeds_str,
                   "Metric": "Acc" if is_base else "Acc Δ",
                   "_lbl": lbl, "_metric": "acc"}
        row_nme = {"Experiment": "", "n/ds": "",
                   "Metric": "NME" if is_base else "NME Δ",
                   "_lbl": lbl, "_metric": "nme"}
        row_fgt = {"Experiment": "", "n/ds": "",
                   "Metric": "Fgt" if is_base else "Fgt Δ",
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
                if not np.isnan(d["acc_mean"]):
                    acc_delta_vals.append(d["acc_mean"])
                if not np.isnan(d["nme_mean"]):
                    nme_delta_vals.append(d["nme_mean"])
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

            # Fgt: absolute for baseline row; Δ vs baseline for all others
            if is_base:
                row_fgt[col] = _fmt_val(d["fgt_mean"], d["fgt_std"], n, signed=False)
                if not np.isnan(d["fgt_mean"]):
                    fgt_vals.append(d["fgt_mean"])
            else:
                bm_fgt = base_means[ds]["fgt"]
                if not np.isnan(d["fgt_mean"]) and not np.isnan(bm_fgt):
                    delta_fgt = d["fgt_mean"] - bm_fgt
                    row_fgt[col] = _fmt_val(delta_fgt, d["fgt_std"], n, signed=True)
                    fgt_vals.append(delta_fgt)
                else:
                    row_fgt[col] = "—"

        # Mean column
        if is_base:
            row_acc["Mean"] = (
                f"{np.mean(acc_delta_vals):.2f}" if acc_delta_vals else "—"
            )
            row_nme["Mean"] = (
                f"{np.mean(nme_delta_vals):.2f}" if nme_delta_vals else "—"
            )
        else:
            def _mean_str(vals, signed=True):
                if not vals:
                    return "—"
                mv = float(np.mean(vals))
                return f"{'+' if mv >= 0 else ''}{mv:.2f}" if signed else f"{mv:.2f}"
            row_acc["Mean"] = _mean_str(acc_delta_vals)
            row_nme["Mean"] = _mean_str(nme_delta_vals)
        if is_base:
            row_fgt["Mean"] = f"{np.mean(fgt_vals):.2f}" if fgt_vals else "—"
        else:
            mv = float(np.mean(fgt_vals)) if fgt_vals else None
            row_fgt["Mean"] = f"{mv:+.2f}" if mv is not None else "—"

        rows.extend([row_acc, row_nme, row_fgt])

    # Best value per (dataset col, metric) — higher for Acc/NME Δ, lower for Fgt Δ
    best_idx: dict = {}
    for col in ds_cols + ["Mean"]:
        for metric_key, higher in [("Acc Δ", True), ("NME Δ", True), ("Fgt Δ", False)]:
            bv, bi = None, None
            for i, row in enumerate(rows):
                if row["_metric"] != metric_key.replace(" Δ", "").lower():
                    continue
                displayed = _mean_from_cell(row.get(col, ""))
                val = (
                    0.0
                    if row["_lbl"] == baseline_label and displayed is not None
                    else displayed
                )
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
        _lookup = {"acc": "Acc Δ", "nme": "NME Δ", "fgt": "Fgt Δ"}
        lookup_key = _lookup[metric_str]
        for j, col in enumerate(cols):
            if col in ds_cols + ["Mean"]:
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
        "> <u>Underlined rows</u> = Baseline InfoNCE (β=0).\n"
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

        # ── Sort: Baseline InfoNCE pinned first, then others by label asc ──────────
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
            f" | Baseline InfoNCE (β=0): **{n_base}** seeds, "
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
# Section 0 — Ranking subsection (Friedman + combined metric)
# ─────────────────────────────────────────────────────────────────────────────

# Reference datasets for statistical ranking (established benchmark, complete data)
_RANK_DATASETS = [
    "cifar100_10-10", "cifar100_50-10",
    "tiny_imagenet_20-20", "tiny_imagenet_100-20",
]

# Exploratory combinations remain excluded only from the legacy all-method
# ranking. A ranking explicitly scoped to the complementary group may include
# them when they satisfy the same complete-observation contract.
_RANK_EXCLUDED_METHODS = {
    f"ANT-IV-AR (β=0.5){_LEGACY_ANT_TAG} + SBS",
}


_COMPLEMENTARY_METHOD_MARKERS = (
    " + TeacherAvg-",
    " + SBS",
)

_COMPLEMENTARY_METHOD_TYPE_ORDER = (
    " + TeacherAvg-",
    " + SBS",
)


def _is_complementary_method(label: str) -> bool:
    """Return whether a report label is a complementary method variant."""
    return any(marker in label for marker in _COMPLEMENTARY_METHOD_MARKERS)


def _is_legacy_method(label: str) -> bool:
    """Return whether a report label denotes connected-reference ANT."""
    return _LEGACY_ANT_TAG in label


def _natural_text_sort_key(text: str) -> tuple:
    """Return a case-insensitive key with numeric tokens ordered numerically."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", text)
        if part
    )


def _complementary_method_sort_key(label: str) -> tuple:
    """Sort complementary methods by ablation type, then experiment name."""
    type_index = next(
        (
            index
            for index, marker in enumerate(_COMPLEMENTARY_METHOD_TYPE_ORDER)
            if marker in label
        ),
        len(_COMPLEMENTARY_METHOD_TYPE_ORDER),
    )
    return type_index, _natural_text_sort_key(label)


def _ranking_figure_svg(
    labels: list[str], ranks: np.ndarray, baseline_flags: list[bool],
) -> str:
    """Render a compact vector chart for the Combined mean-rank ordering."""
    row_height = 28
    width, label_width, plot_width = 760, 300, 390
    height = 62 + row_height * len(labels)
    rank_scale = max(float(len(labels)), max(float(rank) for rank in ranks))
    out = [
        '<figure class="ranking-figure">',
        '<figcaption><strong>Figura - ranking por Mean Rank Combined.</strong> '
        'Quanto menor o rank medio, melhor; barras maiores indicam melhor posicao.</figcaption>',
        f'<svg role="img" aria-label="Ranking Combined" viewBox="0 0 {width} {height}" '
        'style="width:100%;height:auto;border:1px solid #cbd5e1;border-radius:6px;background:#f8fafc">',
        f'<text x="{label_width}" y="24" fill="#334155" font-size="13" font-weight="700">Mean Rank Combined (menor = melhor)</text>',
    ]
    for index, (label, rank, is_baseline) in enumerate(zip(labels, ranks, baseline_flags)):
        y = 40 + index * row_height
        bar_width = plot_width * (rank_scale - float(rank) + 1.0) / rank_scale
        fill = '#2563eb' if is_baseline else '#0f766e'
        out.extend([
            f'<text x="12" y="{y + 16}" fill="#1e293b" font-size="12">{_html_escape(label)}</text>',
            f'<rect x="{label_width}" y="{y}" width="{plot_width}" height="18" rx="4" fill="#e2e8f0"/>',
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" height="18" rx="4" fill="{fill}"/>',
            f'<text x="{label_width + plot_width + 10}" y="{y + 14}" fill="#1e293b" font-size="12">{float(rank):.2f}</text>',
        ])
    out.append('</svg></figure>')
    return ''.join(out)


def _ranking_html(
    df: pd.DataFrame,
    method_labels: set[str] | None = None,
    required_method_labels: set[str] | None = None,
    title: str = "Ranking estatístico — Friedman test",
) -> str:
    """Build a Friedman-test ranking subsection for Section 0.

    Rows = (dataset × seed) observations.  Columns = methods.
    Ranks computed per observation (rank 1 = best):
      • ACC / NME : descending (higher = better)
      • Fgt       : ascending  (lower  = better)
    Combined rank = mean of the three per-observation ranks.
    """
    import warnings
    try:
        from scipy.stats import friedmanchisquare, rankdata
    except ImportError:
        return ""

    ref_df = df[df["dataset"].isin(_RANK_DATASETS)].copy()
    if method_labels is not None:
        ref_df = ref_df[ref_df["label"].isin(method_labels)].copy()
    if ref_df.empty:
        return ""

    # Build obs_data: (dataset, seed) → {label: {acc, nme, fgt}}
    obs_data: dict = {}
    for _, row in ref_df.iterrows():
        if pd.isna(row["avg_acc"]) or pd.isna(row["avg_nme"]) or pd.isna(row["fgt_acc"]):
            continue
        key = (row["dataset"], int(row["seed"]) if pd.notna(row["seed"]) else -1)
        obs_data.setdefault(key, {})[row["label"]] = {
            "acc": float(row["avg_acc"]),
            "nme": float(row["avg_nme"]),
            "fgt": float(row["fgt_acc"]),
        }

    if not obs_data:
        return ""

    all_obs_keys = sorted(obs_data.keys())
    all_labels: set = set()
    for v in obs_data.values():
        all_labels.update(v.keys())

    # Start with methods that cover every available observation. Explicitly
    # required methods may have partial coverage; in that case the paired
    # analysis is restricted to the complete observation intersection shared
    # by every compared method. This preserves the Friedman block design.
    complete_labels = {
        lbl for lbl in all_labels
        if method_labels is not None or lbl not in _RANK_EXCLUDED_METHODS
        if all(
            lbl in obs_data[k]
            and np.isfinite(obs_data[k][lbl]["acc"])
            and np.isfinite(obs_data[k][lbl]["nme"])
            and np.isfinite(obs_data[k][lbl]["fgt"])
            for k in all_obs_keys
        )
    }
    required_present = set(required_method_labels or ()) & all_labels
    compared_labels = complete_labels | required_present
    obs_keys = [
        key for key in all_obs_keys
        if all(
            label in obs_data[key]
            and np.isfinite(obs_data[key][label]["acc"])
            and np.isfinite(obs_data[key][label]["nme"])
            and np.isfinite(obs_data[key][label]["fgt"])
            for label in compared_labels
        )
    ]
    complete = sorted(compared_labels)

    n_m, n_o = len(complete), len(obs_keys)
    if n_m < 2 or n_o < 3:
        return ""

    # Build value matrices (obs × method)
    acc_v = np.array([[obs_data[k][l]["acc"] for l in complete] for k in obs_keys])
    nme_v = np.array([[obs_data[k][l]["nme"] for l in complete] for k in obs_keys])
    fgt_v = np.array([[obs_data[k][l]["fgt"] for l in complete] for k in obs_keys])

    # Rank per observation (rank 1 = best)
    def _rank_rows(mat: np.ndarray, ascending: bool) -> np.ndarray:
        sign = 1 if ascending else -1
        return np.array([rankdata(sign * mat[i]) for i in range(n_o)], dtype=float)

    r_acc = _rank_rows(acc_v, ascending=False)  # higher acc → rank 1
    r_nme = _rank_rows(nme_v, ascending=False)  # higher nme → rank 1
    r_fgt = _rank_rows(fgt_v, ascending=True)   # lower  fgt → rank 1
    r_com = (r_acc + r_nme + r_fgt) / 3.0       # combined (lower = better)

    mr_acc = r_acc.mean(axis=0)
    mr_nme = r_nme.mean(axis=0)
    mr_fgt = r_fgt.mean(axis=0)
    mr_com = r_com.mean(axis=0)

    # Baseline identity and per-dataset means for context deltas.
    bl_rows = ref_df[ref_df["is_baseline"]]
    baseline_acc_by_dataset = (
        bl_rows.groupby("dataset")["avg_acc"].mean().to_dict()
        if not bl_rows.empty
        else {}
    )

    # Friedman tests
    def _friedman(mat: np.ndarray):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
            return stat, p
        except Exception:
            return np.nan, np.nan

    results_f = {
        "ACC":      _friedman(r_acc),
        "NME":      _friedman(r_nme),
        "Fgt":      _friedman(r_fgt),
        "Combined": _friedman(r_com),
    }

    def _sig(p: float) -> str:
        if np.isnan(p):  return "—"
        if p < 0.001:    return f"p={p:.3f} ✦✦✦"
        if p < 0.01:     return f"p={p:.3f} ✦✦"
        if p < 0.05:     return f"p={p:.3f} ✦"
        return f"p={p:.3f} (NS)"

    # Sort by combined mean rank (ascending = best first)
    order = np.argsort(mr_com)
    sorted_labels = [complete[i] for i in order]
    sorted_mr_acc = mr_acc[order]
    sorted_mr_nme = mr_nme[order]
    sorted_mr_fgt = mr_fgt[order]
    sorted_mr_com = mr_com[order]

    # Compute n_obs per label (can differ when CUB etc. added later)
    n_obs_per = {
        lbl: sum(1 for k in obs_keys if lbl in obs_data[k]) for lbl in complete
    }

    best_acc = int(np.argmin(sorted_mr_acc))
    best_nme = int(np.argmin(sorted_mr_nme))
    best_fgt = int(np.argmin(sorted_mr_fgt))
    best_com = int(np.argmin(sorted_mr_com))  # always 0 since already sorted

    is_baseline = [lbl == (bl_rows["label"].mode().iloc[0] if not bl_rows.empty else None)
                   for lbl in sorted_labels]

    # Median of per-run Δacc against the matching dataset baseline mean.
    delta_acc_med: dict = {}
    for lbl in sorted_labels:
        deltas = []
        for _, row in ref_df[ref_df["label"] == lbl].iterrows():
            baseline = baseline_acc_by_dataset.get(row["dataset"])
            if baseline is not None and pd.notna(row["avg_acc"]):
                deltas.append(float(row["avg_acc"]) - float(baseline))
        delta_acc_med[lbl] = float(np.median(deltas)) if deltas else np.nan

    # ── Build HTML ────────────────────────────────────────────────────────────
    out = [
        f'<h3>{_html_escape(title)}</h3>',
        '<div class="callout">',
        f'Método de ranking: ranks por observação (dataset × seed), {n_o} observações × {n_m} métodos. '
        'Rank 1 = melhor em cada observação (ACC/NME: maior é melhor; Fgt: menor é melhor). '
        '<strong>Combined MR</strong> = média dos três ranks por observação. '
        'Conjuntos de referência: CIFAR-100 10-10, CIFAR-100 50-10, TIN 20-20, TIN 100-20. '
        'Quando um método requerido ainda tem cobertura parcial, a comparação usa somente a '
        'interseção completa de observações compartilhada por todos os métodos.',
        '</div>',
        _ranking_figure_svg(sorted_labels, sorted_mr_com, is_baseline),
        '<div class="table-wrap"><table class="ranking-table" style="width:100%">',
        '<thead><tr>',
        '<th>#</th><th>Método</th><th>n obs</th>',
        '<th title="Mean rank on ACC (1=best)">MR ACC↑</th>',
        '<th title="Mean rank on NME (1=best)">MR NME↑</th>',
        '<th title="Mean rank on Fgt (1=best=lowest)">MR Fgt↓</th>',
        '<th title="Mean of three per-observation ranks (lower=better overall)"><strong>MR Combined</strong></th>',
        '<th title="Median of per-run Δ Acc against each matching dataset baseline mean">Median Δ Acc</th>',
        '</tr></thead><tbody>',
    ]

    for rank_i, lbl in enumerate(sorted_labels, 1):
        i = rank_i - 1
        bl_row = is_baseline[i]
        tr_cls = ' class="baseline"' if bl_row else ""
        delta_str = "—" if bl_row else (
            f"{delta_acc_med[lbl]:+.2f}" if not np.isnan(delta_acc_med.get(lbl, np.nan)) else "—"
        )

        def _cell(val: float, is_best: bool) -> str:
            s = f"{val:.2f}"
            return f'<td class="best"><strong>{s}</strong></td>' if is_best else f"<td>{s}</td>"

        out.append(f'<tr{tr_cls}>')
        out.append(f'<td>{rank_i}</td>')
        out.append(f'<td style="text-align:left">{_html_escape(lbl)}</td>')
        out.append(f'<td>{n_obs_per[lbl]}</td>')
        out.append(_cell(sorted_mr_acc[i], i == best_acc))
        out.append(_cell(sorted_mr_nme[i], i == best_nme))
        out.append(_cell(sorted_mr_fgt[i], i == best_fgt))
        out.append(_cell(sorted_mr_com[i], i == best_com))
        out.append(f'<td>{delta_str}</td>')
        out.append('</tr>')

    out.append('</tbody></table></div>')

    # Friedman summary
    out.append('<p style="font-size:8pt;margin:4pt 0">')
    for metric, (stat, p) in results_f.items():
        arrow = "↑" if metric in ("ACC", "NME") else ("↓" if metric == "Fgt" else "")
        out.append(
            f'<strong>Friedman {metric}{arrow}:</strong> χ²={stat:.3f}, {_sig(p)}'
            '&nbsp;&nbsp;'
        )
    out.append('</p>')
    out.append(
        '<p style="font-size:7.5pt;color:#475569;margin:2pt 0">'
        'NS = não significativo (α=0.05). ✦ p&lt;0.05 ✦✦ p&lt;0.01 ✦✦✦ p&lt;0.001. '
        f'Classificação baseada em {n_o} observações ({len(_RANK_DATASETS)} datasets × seeds disponíveis).'
        '</p>'
    )

    return ''.join(out)


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
                "seeds": sorted(sub["seed"].dropna().astype(int).tolist()),
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
    # Primary sort: more seeds first; secondary: higher Acc Δ
    def total_seeds_html(lbl):
        return sum(cell[lbl][ds]["n"] for ds in _DATASET_ORDER if cell[lbl].get(ds))
    non_baseline.sort(key=lambda lbl: (total_seeds_html(lbl), mean_delta_acc(lbl)), reverse=True)
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
                if pd.notna(d["acc_mean"]):
                    acc_vals.append(d["acc_mean"])
                if pd.notna(d["nme_mean"]):
                    nme_vals.append(d["nme_mean"])
            else:
                da = d["acc_mean"] - base_means[ds]["acc"]
                dn = d["nme_mean"] - base_means[ds]["nme"]
                metrics["acc"][ds] = fmt(da, d["acc_std"], n, signed=True)
                metrics["nme"][ds] = fmt(dn, d["nme_std"], n, signed=True)
                acc_vals.append(da); nme_vals.append(dn)
            if is_base:
                metrics["fgt"][ds] = fmt(d["fgt_mean"], d["fgt_std"], n)
                if pd.notna(d["fgt_mean"]):
                    fgt_vals.append(d["fgt_mean"])
            else:
                bm_fgt = base_means[ds]["fgt"]
                if pd.notna(d["fgt_mean"]) and pd.notna(bm_fgt):
                    delta_fgt = d["fgt_mean"] - bm_fgt
                    metrics["fgt"][ds] = fmt(delta_fgt, d["fgt_std"], n, signed=True)
                    fgt_vals.append(delta_fgt)
                else:
                    metrics["fgt"][ds] = "—"

        metrics["acc"]["Mean"] = (
            f"{np.mean(acc_vals):.2f}" if is_base and acc_vals
            else (f"{np.mean(acc_vals):+.2f}" if acc_vals else "—")
        )
        metrics["nme"]["Mean"] = (
            f"{np.mean(nme_vals):.2f}" if is_base and nme_vals
            else (f"{np.mean(nme_vals):+.2f}" if nme_vals else "—")
        )
        if is_base:
            metrics["fgt"]["Mean"] = f"{np.mean(fgt_vals):.2f}" if fgt_vals else "—"
        else:
            metrics["fgt"]["Mean"] = f"{float(np.mean(fgt_vals)):+.2f}" if fgt_vals else "—"
        _counts = [str(cell[lbl][ds]["n"]) if cell[lbl].get(ds) else "—" for ds in _DATASET_ORDER]
        _tooltip = "\n".join(
            f"{_DATASET_SHORT.get(ds, ds)}: {', '.join(map(str, cell[lbl][ds]['seeds'])) if cell[lbl].get(ds) else '—'}"
            for ds in _DATASET_ORDER
        )
        rows.append({
            "label": lbl,
            "seeds": "/".join(_counts),
            "seeds_tooltip": _tooltip,
            "baseline": is_base,
            "metrics": metrics,
        })

    columns = _DATASET_ORDER + ["Mean"]
    out = [
        '<section class="report-section cross-dataset">',
        '<h1>Section 0 — Cross-Dataset Overview</h1>',
        '<div class="callout">'
        'Each method has three sub-rows: <strong>Acc Δ</strong>, '
        '<strong>NME Δ</strong>, and <strong>Fgt Δ</strong>. '
        'Experiment and <strong>n/ds</strong> use <code>rowspan="3"</code>. '
        'Historical nGlobal/nLocal runs share one method identity, and all '
        'beta-zero runs without TeacherAvg are Baseline InfoNCE, while '
        'beta-zero TeacherAvg runs remain separate ablations. Duplicate dataset/seed pairs '
        'prefer the canonical source and use historical alternatives only as fallback. '
        '<strong>n/ds</strong> = seed count per dataset column (left→right in dataset order); '
        'hover the cell to see seed IDs per dataset. '
        'Mean ± std shown when ≥2 seeds. The <strong>Mean</strong> column is '
        'the unweighted mean across tested datasets: absolute metrics for the '
        'baseline and deltas for variants. Methods are separated into canonical '
        'detached-reference ANT, complementary ablations, and legacy '
        'connected-reference ANT. The baseline is repeated as a table-header '
        'reference in every displayed group.'
        '</div>',
    ]

    metric_labels = {
        "acc": ("Acc", "Acc Δ"),
        "nme": ("NME", "NME Δ"),
        "fgt": ("Fgt", "Fgt Δ"),
    }
    row_by_label = {row["label"]: row for row in rows}
    central_labels = [
        label for label in non_baseline
        if not _is_complementary_method(label) and not _is_legacy_method(label)
    ]
    complementary_labels = [
        label for label in non_baseline
        if _is_complementary_method(label) and not _is_legacy_method(label)
    ]
    legacy_labels = [label for label in non_baseline if _is_legacy_method(label)]
    complementary_labels.sort(key=_complementary_method_sort_key)
    legacy_labels.sort(key=_complementary_method_sort_key)

    def render_method_group(
        *,
        title: str,
        description: str,
        labels: list[str],
        css_class: str,
        ranking_title: str,
        summary_title: str,
    ) -> None:
        group_rows = []
        if baseline_label and baseline_label in row_by_label:
            group_rows.append(row_by_label[baseline_label])
        group_rows.extend(row_by_label[label] for label in labels)

        # Best cells are computed independently inside each research group.
        # The baseline participates as delta zero, despite displaying absolutes.
        best = set()
        for metric, higher in (("acc", True), ("nme", True), ("fgt", False)):
            for col in columns:
                candidates = []
                for row in group_rows:
                    displayed = _mean_from_cell(row["metrics"][metric][col])
                    value = (
                        0.0
                        if row["baseline"] and displayed is not None
                        else displayed
                    )
                    if value is not None:
                        candidates.append((value, row["label"]))
                if not candidates:
                    continue
                target = (max if higher else min)(
                    candidates, key=lambda item: item[0]
                )[0]
                for value, label in candidates:
                    if abs(value - target) < 1e-9:
                        best.add((label, metric, col))

        out.extend([
            f'<section class="method-group {css_class}">',
            f'<h2>{_html_escape(title)}</h2>',
            f'<div class="callout">{description}</div>',
            '<div class="table-wrap"><table class="overview-table">',
            '<colgroup><col class="experiment"><col class="seeds"><col class="metric">'
            + ''.join('<col class="dataset">' for _ in _DATASET_ORDER)
            + '<col class="mean"></colgroup>',
            f'<thead><tr><th>Experiment</th><th title="seed count per dataset (left→right: {"/".join(_DATASET_SHORT.get(ds, ds) for ds in _DATASET_ORDER)})">n/ds</th><th>Metric</th>'
            + ''.join(
                f'<th>{_html_escape(_DATASET_SHORT.get(ds, ds))}</th>'
                for ds in _DATASET_ORDER
            )
            + '<th>Mean</th></tr>',
        ])

        def append_rows(render_rows: list[dict]) -> None:
            for row in render_rows:
                row_class = " baseline" if row["baseline"] else ""
                for metric_index, metric in enumerate(("acc", "nme", "fgt")):
                    classes = (
                        f'group-start{row_class}'
                        if metric_index == 0
                        else f'subrow{row_class}'
                    )
                    out.append(f'<tr class="{classes.strip()}">')
                    if metric_index == 0:
                        out.append(
                            f'<td class="experiment" rowspan="3">'
                            f'{_html_escape(row["label"])}</td>'
                        )
                        out.append(
                            f'<td class="seeds" rowspan="3" '
                            f'title="{_html_escape(row["seeds_tooltip"])}">'
                            f'{_html_escape(row["seeds"])}</td>'
                        )
                    label = metric_labels[metric][0 if row["baseline"] else 1]
                    out.append(f'<td class="metric">{_html_escape(label)}</td>')
                    for col in columns:
                        value = _html_escape(row["metrics"][metric][col])
                        is_best = (row["label"], metric, col) in best
                        if is_best:
                            value = f'<strong>{value}</strong>'
                        td_class = ' class="best"' if is_best else ''
                        out.append(f'<td{td_class}>{value}</td>')
                    out.append('</tr>')

        # Keeping the baseline inside THEAD makes it the fixed comparison
        # header and allows PDF engines to repeat it after a page break.
        baseline_rows = [row for row in group_rows if row["baseline"]]
        append_rows(baseline_rows)
        out.append('</thead><tbody>')
        append_rows([row for row in group_rows if not row["baseline"]])
        out.append('</tbody></table></div>')

        out.extend([
            f'<h3>{_html_escape(summary_title)}</h3>',
            '<div class="table-wrap"><table class="summary-table">',
            '<thead><tr><th>Dataset</th><th>Baseline Acc</th>'
            '<th>Best Method</th><th>Best Acc Δ</th></tr></thead><tbody>',
        ])
        for ds in _DATASET_ORDER:
            baseline_mean = base_means[ds]["acc"]
            best_label, best_delta = (
                (baseline_label, 0.0)
                if pd.notna(baseline_mean)
                else (None, None)
            )
            for label in labels:
                data = cell[label].get(ds)
                if data and pd.notna(data["acc_mean"]) and pd.notna(baseline_mean):
                    delta = data["acc_mean"] - baseline_mean
                    if best_delta is None or delta > best_delta:
                        best_label, best_delta = label, delta
            baseline_text = (
                f"{baseline_mean:.2f}%" if pd.notna(baseline_mean) else "—"
            )
            delta_text = f"{best_delta:+.2f}" if best_delta is not None else "—"
            out.append(
                '<tr>'
                f'<td>{_html_escape(_DATASET_SHORT.get(ds, ds))}</td>'
                f'<td>{baseline_text}</td>'
                f'<td>{_html_escape(best_label or "—")}</td>'
                f'<td>{delta_text}</td>'
                '</tr>'
            )
        out.append('</tbody></table></div>')
        out.append(
            _ranking_html(
                df,
                method_labels={row["label"] for row in group_rows},
                required_method_labels=(
                    {"ANT-FS-GR (β=0.5)"}
                    if css_class == "central-methods"
                    else (
                        set(labels)
                        if css_class == "complementary-methods"
                        else None
                    )
                ),
                title=ranking_title,
            )
        )
        out.append('</section>')

    render_method_group(
        title="Métodos centrais da pesquisa",
        description=(
            'Formulações canônicas de ANT, nas quais a referência da margem é '
            'destacada do grafo de gradientes, sem TeacherAvg ou SBS. '
            'O Baseline InfoNCE permanece fixado no cabeçalho da tabela como '
            'referência para todos os deltas.'
        ),
        labels=central_labels,
        css_class="central-methods",
        ranking_title="Ranking estatístico — métodos centrais",
        summary_title="Melhor método central por dataset",
    )
    if complementary_labels:
        render_method_group(
            title="Métodos complementares",
            description=(
                'Ablações e extensões canônicas com TeacherAvg ou SBS. O mesmo '
                'Baseline InfoNCE é repetido no cabeçalho para preservar a comparação '
                'direta, sem misturar estes resultados com os métodos centrais.'
            ),
            labels=complementary_labels,
            css_class="complementary-methods",
            ranking_title="Ranking estatístico — métodos complementares",
            summary_title="Melhor método complementar por dataset",
        )
    if legacy_labels:
        render_method_group(
            title="ANT legado — referência conectada",
            description=(
                'Resultados da formulação anterior, na qual a referência da margem '
                'também recebia gradientes. Eles permanecem disponíveis para '
                'rastreabilidade, mas não definem mais a identidade canônica de ANT.'
            ),
            labels=legacy_labels,
            css_class="legacy-methods",
            ranking_title="Ranking estatístico — ANT legado",
            summary_title="Melhor método legado por dataset",
        )
    out.append('</section>')
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
.overview-table { width: 100%; table-layout: fixed; }
.overview-table th, .overview-table td {
  white-space: nowrap;
  overflow-wrap: normal;
  word-break: normal;
}
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
.overview-table col.experiment { width: 27%; }
.overview-table col.seeds { width: 7%; }
.overview-table col.metric { width: 6%; }
.overview-table col.dataset { width: 9%; }
.overview-table col.mean { width: 6%; }
.overview-table td.experiment { text-align: center; font-weight: 600; background: var(--group); }
.overview-table td:first-child, .overview-table th:first-child { text-align: center; }
.overview-table td.seeds { background: var(--group); white-space: nowrap; }
.overview-table td.metric { font-weight: 700; background: #fbfdff; }
.overview-table tr.group-start td { border-top-width: 1.2pt; }
.overview-table tr.baseline td { background: var(--baseline); text-decoration: underline; }
.overview-table tr.baseline td.best { background-color: #dcfce7; }
.ranking-table tr.baseline td { background: var(--baseline); text-decoration: underline; }
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
.provenance-strip {
  font-size: 8pt;
  color: #334155;
  background: #f8fafc;
  border: 1px solid var(--border-soft);
  padding: 6pt 8pt;
  margin: 0 0 10pt;
}
.provenance-strip code { overflow-wrap: anywhere; }
.provenance-details { margin: 5pt 0 0; }
.provenance-index {
  border: 1px solid var(--border-soft);
  padding: 6pt 8pt;
  margin: 0 0 12pt;
}
.provenance-index summary { cursor: pointer; font-weight: 700; color: #172554; }
.provenance-index table { width: 100%; table-layout: fixed; }
.provenance-index col.dataset { width: 14%; }
.provenance-index col.method { width: 25%; }
.provenance-index col.seed { width: 6%; }
.provenance-index col.experiment { width: 28%; }
.provenance-index col.hash { width: 14%; }
.provenance-index col.manifest { width: 13%; }
.provenance-index th, .provenance-index td { text-align: center; }
.provenance-index th:nth-child(-n+3),
.provenance-index td:nth-child(-n+3),
.provenance-index th:nth-child(5),
.provenance-index td:nth-child(5),
.provenance-index th:nth-child(6),
.provenance-index td:nth-child(6) {
  white-space: nowrap;
  overflow-wrap: normal;
  word-break: normal;
}
@media print { .provenance-index { display: none; } }
@media screen {
  body { max-width: 1600px; margin: 0 auto; padding: 28px; background: white; }
  html { background: #e5e7eb; }
  .table-wrap { overflow-x: auto; }
  table { font-size: 14px; }
  .cross-dataset .table-wrap, .provenance-index .table-wrap { overflow-x: visible; }
  .cross-dataset .table-wrap, .provenance-index .table-wrap {
    container-type: inline-size;
  }
  .overview-table {
    font-size: clamp(11px, 0.95cqw, 15px);
  }
  .overview-table th, .overview-table td {
    padding: clamp(3px, 0.3cqw, 6px);
  }
  .provenance-index table {
    font-size: clamp(12px, 0.9cqw, 15px);
  }
  .provenance-index th, .provenance-index td {
    padding: clamp(3px, 0.3cqw, 6px);
  }
}
"""


def _build_html_document(
    section0_html: str,
    remaining_markdown: str,
    generated_at: str,
    provenance: dict | None = None,
) -> str:
    body = section0_html + _markdown_to_html(remaining_markdown)
    provenance_html = ""
    if provenance:
        lineage_hash = _html_escape(provenance["lineage_hash"])
        sidecar = _html_escape(provenance["sidecar_name"])
        selected = int(provenance["selected_experiment_count"])
        legacy = int(provenance["selected_without_run_manifest_count"])
        mode = _html_escape(provenance["mode"])
        provenance_html = (
            '<aside class="provenance-strip" id="report-provenance" '
            f'data-lineage-hash="{lineage_hash}">'
            f'<strong>Proveniência:</strong> {selected} experimentos selecionados; '
            f'modo <code>{mode}</code>; '
            f'<code>lineage {lineage_hash}</code>. '
            f'Manifesto: <code>{sidecar}</code>.'
            + (
                f'<div class="provenance-details">Compatibilidade histórica: '
                f'{legacy} experimento(s) não possuem manifesto interno de execução; '
                'seus logs permanecem identificados por SHA-256 no sidecar.</div>'
                if legacy
                else '<div class="provenance-details">Todos os experimentos selecionados possuem manifesto interno de execução.</div>'
            )
            + '</aside>'
        )
        rows = []
        for item in provenance.get("selected_experiments", []):
            manifest_state = "sim" if item["run_manifests"] else "legado"
            gistlog = item["gistlog"]
            rows.append(
                "<tr>"
                f"<td>{_html_escape(item['dataset'])}</td>"
                f"<td>{_html_escape(item['method'])}</td>"
                f"<td>{_html_escape(item['seed'])}</td>"
                f"<td><code>{_html_escape(item['experiment'])}</code></td>"
                f"<td><code>{_html_escape(gistlog['sha256'][:16])}…</code></td>"
                f"<td>{manifest_state}</td>"
                "</tr>"
            )
        provenance_html += (
            '<details class="provenance-index">'
            '<summary>Experimentos que compõem este relatório</summary>'
            '<p>A relação integral, os hashes completos, as decisões de exclusão '
            'e os manifestos-pais estão no sidecar indicado acima.</p>'
            '<div class="table-wrap"><table><colgroup>'
            '<col class="dataset"><col class="method"><col class="seed">'
            '<col class="experiment"><col class="hash"><col class="manifest">'
            '</colgroup><thead><tr>'
            '<th>Dataset</th><th>Método</th><th>Seed</th><th>Experimento</th>'
            '<th>SHA-256 do gistlog</th><th>Manifesto da execução</th>'
            '</tr></thead><tbody>'
            + "\n".join(rows)
            + '</tbody></table></div></details>'
        )
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="tagfex-lineage-sha256" content="{_html_escape(provenance['lineage_hash']) if provenance else ''}">
<title>TagFex Experiment Report</title>
<style>{_report_css()}</style>
</head>
<body>
<div class="generated-at">Generated {generated_at}</div>
{provenance_html}
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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_pdf(html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{pdf_path.name}.", suffix=".tmp", dir=pdf_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_pdf(str(html_path), str(temporary))
        temporary.replace(pdf_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _relative_or_absolute(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _parent_run_manifests(log_path: Path) -> list[Path]:
    provenance_dir = log_path.parent / "provenance"
    if not provenance_dir.is_dir():
        return []
    return sorted(provenance_dir.glob("run-*.json"), key=lambda path: path.name)


def _selection_records(
    raw_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    repo: Path,
    logs_dir: Path,
    mode: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    selected_names = set(selected_df["exp"])
    complete_names = set(complete_df["exp"])
    selected_records = []
    parent_paths = set()
    for _, row in selected_df.sort_values(["dataset", "label", "seed", "exp"]).iterrows():
        log_path = Path(row["gistlog_path"]).resolve()
        log_record = file_record(log_path, root=repo)
        if (
            log_record["sha256"] != row["gistlog_sha256"]
            or log_record["size_bytes"] != int(row["gistlog_size_bytes"])
        ):
            raise RuntimeError(
                f"Input changed after it was parsed; rerun the report: {log_path}"
            )
        parents = _parent_run_manifests(log_path)
        parent_paths.update(parents)
        debug_record = None
        if mode == "full":
            raw_debug = log_path.parent / "exp_debug0.log"
            zipped_debug = log_path.parent / "debug_logs.zip"
            if raw_debug.is_file():
                debug_record = file_record(raw_debug, root=repo)
            elif zipped_debug.is_file():
                try:
                    with zipfile.ZipFile(zipped_debug, "r") as archive:
                        if "exp_debug0.log" in archive.namelist():
                            debug_record = file_record(zipped_debug, root=repo)
                except (OSError, zipfile.BadZipFile):
                    pass
        selected_records.append(
            {
                "experiment": row["exp"],
                "dataset": row["dataset"],
                "method": row["label"],
                "ant_generation": row.get("ant_generation"),
                "ant_reference_gradient": row.get("ant_reference_gradient"),
                "seed": int(row["seed"]) if pd.notna(row["seed"]) else None,
                "num_tasks": int(row["num_tasks"]),
                "metrics": {
                    "avg_acc1": float(row["avg_acc"]),
                    "avg_nme1": float(row["avg_nme"]),
                    "forgetting_acc": float(row["fgt_acc"]),
                    "forgetting_nme": float(row["fgt_nme"]),
                },
                "gistlog": log_record,
                "debug_input": debug_record,
                "run_manifests": [
                    _relative_or_absolute(path, repo) for path in parents
                ],
            }
        )
    parent_records = [
        file_record(path, root=repo) for path in sorted(parent_paths, key=str)
    ]

    raw_rows = {row["exp"]: row for _, row in raw_df.iterrows()}
    decisions = []
    for directory in sorted(
        (path for path in logs_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        experiment = directory.name
        row = raw_rows.get(experiment)
        if experiment in _SKIP:
            reason = "excluded_auxiliary_directory"
        elif experiment.startswith(_SKIP_PREFIXES):
            reason = "excluded_quarantined_or_superseded_prefix"
        elif not (directory / "exp_gistlog.log").is_file():
            reason = "excluded_missing_exp_gistlog"
        elif row is None:
            reason = "excluded_unparseable_gistlog"
        elif experiment in selected_names:
            reason = "selected"
        elif experiment not in complete_names:
            reason = "incomplete_expected_task_count"
        else:
            reason = "lower_priority_methodologically_equivalent_run"
        decisions.append(
            {
                "experiment": experiment,
                "dataset": row["dataset"] if row is not None else None,
                "seed": (
                    int(row["seed"])
                    if row is not None and pd.notna(row["seed"])
                    else None
                ),
                "num_tasks": int(row["num_tasks"]) if row is not None else None,
                "expected_tasks": (
                    _EXPECTED_TASKS.get(row["dataset"])
                    if row is not None
                    else None
                ),
                "decision": reason,
                "gistlog_path": (
                    _relative_or_absolute(Path(row["gistlog_path"]), repo)
                    if row is not None
                    else None
                ),
            }
        )
    return selected_records, parent_records, decisions


def _report_provenance(
    *,
    raw_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    repo: Path,
    mode: str,
    strict: bool,
    html_path: Path,
    pdf_path: Path,
    sidecar_path: Path,
    generated_at: str,
    logs_dir: Path,
) -> dict:
    selected, parents, decisions = _selection_records(
        raw_df, complete_df, selected_df, repo, logs_dir, mode
    )
    missing = [item["experiment"] for item in selected if not item["run_manifests"]]
    if strict and missing:
        raise RuntimeError(
            f"Strict provenance rejected {len(missing)} selected experiments "
            "without run manifests"
        )
    source_paths = [Path(__file__).resolve(), *[repo / "utils" / name for name in ("provenance.py", "research_identity.py", "argument.py")]]
    source_records = [file_record(path, root=repo) for path in source_paths]
    parameters = {
        "mode": mode,
        "strict_run_manifests": strict,
        "expected_tasks": _EXPECTED_TASKS,
        "skip_names": sorted(_SKIP),
        "skip_prefixes": list(_SKIP_PREFIXES),
        "equivalence": (
            "remove obsolete nceGlobal/nceLocal; erase inactive ANT mode at beta=0"
        ),
        "ant_report_identity": (
            "refDetached is canonical ANT; connected-reference ANT is legacy; "
            "underlying experiment directory names remain unchanged"
        ),
        "deduplication_priority": [
            "canonical InfoNCE name",
            "nceGlobal",
            "nceLocal",
            "non-debug directory",
            "lexical directory name",
        ],
    }
    package_versions = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "mistune",
        "weasyprint",
        "tabulate",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    environment = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions,
    }
    lineage_payload = {
        "source_hash": records_hash(source_records),
        "selected_experiments_hash": canonical_hash(selected),
        "parent_run_manifests_hash": records_hash(parents),
        "selection_parameters_hash": canonical_hash(parameters),
        "environment_hash": canonical_hash(environment),
    }
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "experiment_report",
        "status": "building",
        "generated_at": generated_at,
        "working_directory": str(repo),
        "command": [sys.executable, *sys.argv],
        "mode": mode,
        "source": {
            "git": git_record(repo),
            "files": source_records,
            "source_hash": lineage_payload["source_hash"],
        },
        "environment": environment,
        "selection": {
            "discovered_count": len(raw_df),
            "scanned_directory_count": len(decisions),
            "complete_count": len(complete_df),
            "selected_count": len(selected_df),
            "selected_experiments": selected,
            "selected_experiments_hash": lineage_payload[
                "selected_experiments_hash"
            ],
            "decisions": decisions,
            "decisions_hash": canonical_hash(decisions),
            "decision_counts": dict(
                sorted(pd.Series([item["decision"] for item in decisions]).value_counts().to_dict().items())
            ),
            "parameters": parameters,
            "parameters_hash": lineage_payload["selection_parameters_hash"],
            "selected_without_run_manifests": missing,
        },
        "parents": {
            "run_manifests": parents,
            "run_manifests_hash": lineage_payload[
                "parent_run_manifests_hash"
            ],
        },
        "lineage_payload": lineage_payload,
        "lineage_hash": canonical_hash(lineage_payload),
        "outputs": {
            "html": _relative_or_absolute(html_path, repo),
            "pdf": _relative_or_absolute(pdf_path, repo),
            "sidecar": _relative_or_absolute(sidecar_path, repo),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TagFex HTML + PDF experiment report."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--short",
        dest="short",
        action="store_true",
        default=True,
        help="skip Section 2 (Debug Metrics); this is the default",
    )
    mode.add_argument(
        "--full",
        dest="short",
        action="store_false",
        help="include Section 2 (Debug Metrics)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(LOGS_DIR),
        help="experiment log root (default: TAGFEX_LOGS_DIR or ./logs)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path(RESULTS_HTML),
        help="base HTML output; --short inserts _short before the suffix",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(RESULTS_PDF),
        help="base PDF output; --short inserts _short before the suffix",
    )
    parser.add_argument(
        "--require-run-manifests",
        action="store_true",
        help="reject selected experiments without in-run provenance manifests",
    )
    parser.add_argument(
        "--verbose-preview",
        action="store_true",
        help="print every selected per-seed run after generation",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()

    results_html = args.html
    results_pdf = args.pdf
    if args.short:
        results_html = results_html.with_name(
            f"{results_html.stem}_short{results_html.suffix}"
        )
        results_pdf = results_pdf.with_name(
            f"{results_pdf.stem}_short{results_pdf.suffix}"
        )
    results_html = results_html.resolve()
    results_pdf = results_pdf.resolve()
    sidecar_path = results_html.with_suffix(".provenance.json")
    repo = ROOT
    logs_dir = args.logs_dir.resolve()

    print(f"Scanning: {logs_dir}")
    raw_df = collect_results(str(logs_dir))

    if raw_df.empty:
        raise RuntimeError("No experiments with parseable gistlog found")

    complete_df = filter_complete(raw_df)
    df = deduplicate_equivalent_runs(complete_df)
    print(
        f"Experiments: {len(raw_df)} found, {len(complete_df)} complete, "
        f"{len(df)} after methodological-equivalence deduplication\n"
    )
    if df.empty:
        raise RuntimeError("No complete experiments after filtering")

    generated_iso = utc_now()
    generated_display = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mode = "short" if args.short else "full"
    provenance = _report_provenance(
        raw_df=raw_df,
        complete_df=complete_df,
        selected_df=df,
        repo=repo,
        mode=mode,
        strict=args.require_run_manifests,
        html_path=results_html,
        pdf_path=results_pdf,
        sidecar_path=sidecar_path,
        generated_at=generated_iso,
        logs_dir=logs_dir,
    )

    print("=== Section 0: Cross-Dataset Overview (HTML rowspan table) ===")
    sec0_html = _cross_dataset_section_html(df)

    print("\n=== Section 1: Results ===")
    sec1 = _results_section(df)

    if args.short:
        print("\n[--short] Skipping Section 2 — Debug Metrics")
        sec2 = []
    else:
        print("\n=== Section 2: Debug ===")
        sec2 = _debug_section(df, str(logs_dir))

    remaining_md = "\n".join([
        "---\n",
        *sec1,
        *([] if not sec2 else ["---\n", *sec2]),
    ])
    html_provenance = {
        "lineage_hash": provenance["lineage_hash"],
        "sidecar_name": sidecar_path.name,
        "selected_experiment_count": provenance["selection"]["selected_count"],
        "selected_without_run_manifest_count": len(
            provenance["selection"]["selected_without_run_manifests"]
        ),
        "selected_experiments": provenance["selection"]["selected_experiments"],
        "mode": mode,
    }
    html_doc = _build_html_document(
        sec0_html,
        remaining_md,
        generated_display,
        html_provenance,
    )

    results_html.parent.mkdir(parents=True, exist_ok=True)
    results_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_descriptor, html_temporary_name = tempfile.mkstemp(
        prefix=f".{results_html.name}.", suffix=".tmp", dir=results_html.parent
    )
    os.close(html_descriptor)
    pdf_descriptor, pdf_temporary_name = tempfile.mkstemp(
        prefix=f".{results_pdf.name}.", suffix=".tmp", dir=results_pdf.parent
    )
    os.close(pdf_descriptor)
    html_temporary = Path(html_temporary_name)
    pdf_temporary = Path(pdf_temporary_name)
    try:
        html_temporary.write_text(html_doc, encoding="utf-8")
        _write_pdf(str(html_temporary), str(pdf_temporary))
        pdf_temporary.replace(results_pdf)
        html_temporary.replace(results_html)
        results_html.chmod(0o644)
        results_pdf.chmod(0o644)
    finally:
        with contextlib.suppress(FileNotFoundError):
            html_temporary.unlink()
        with contextlib.suppress(FileNotFoundError):
            pdf_temporary.unlink()

    output_records = [
        file_record(results_html, root=repo),
        file_record(results_pdf, root=repo),
    ]
    provenance.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "artifacts": {
                "files": output_records,
                "artifacts_hash": records_hash(output_records),
            },
        }
    )
    atomic_write_json(sidecar_path, provenance)
    print(f"\nHTML report saved -> {results_html}")
    print(f"PDF report saved  -> {results_pdf}")
    print(f"Provenance saved  -> {sidecar_path}")
    print(f"Lineage hash      -> {provenance['lineage_hash']}")

    print("\n=== Selection Summary ===")
    for dataset in sorted(df["dataset"].unique()):
        g = df[df["dataset"] == dataset].copy()
        baseline_rows = g[g["is_baseline"]]
        base_mean = baseline_rows["avg_acc"].mean() if not baseline_rows.empty else float("nan")
        g["_s"] = (~g["is_baseline"]).astype(int)
        g = g.sort_values(["_s", "avg_acc"], ascending=[True, False]).drop(columns=["_s"])
        label = dataset_label(dataset)
        base_str = f"  baseline mean={base_mean:.2f}" if not np.isnan(base_mean) else ""
        print(f"[{label}] runs={len(g)}{base_str}")
        if not args.verbose_preview:
            continue
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
