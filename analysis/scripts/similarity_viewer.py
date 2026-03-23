"""Interactive Streamlit UI for exploring TagFex similarity matrices.

Two main views:
  - Training Overview : epoch-level charts from exp_debug0.log
  - Batch Inspector   : per-batch heatmap + per-anchor stats from similarity_debug.log

Usage (from project root):
    streamlit run analysis/scripts/similarity_viewer.py
"""

import re
import json
import pickle
from pathlib import Path
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Default paths  (relative to project root, where streamlit is invoked from)
# ---------------------------------------------------------------------------
_LOGS_ROOT = Path("logs")


def _discover_experiments() -> dict[str, Path]:
    """Return {label: dir_path} for every log dir that has both log files."""
    candidates: dict[str, Path] = {}
    if not _LOGS_ROOT.exists():
        return candidates
    for d in sorted(_LOGS_ROOT.iterdir()):
        if (
            d.is_dir()
            and (d / "exp_debug0.log").exists()
            and (d / "similarity_debug.log").exists()
        ):
            candidates[d.name] = d
    return candidates


_EXPERIMENTS = _discover_experiments()
_DEFAULT_KEY = next((k for k in _EXPERIMENTS if k.startswith("debug_")), None) or next(
    iter(_EXPERIMENTS), None
)
_DEFAULT_DIR = (
    _EXPERIMENTS[_DEFAULT_KEY] if _DEFAULT_KEY else Path("logs/demo_synthetic")
)
DEFAULT_MAIN_LOG = str(_DEFAULT_DIR / "exp_debug0.log")
DEFAULT_SIM_LOG = str(_DEFAULT_DIR / "similarity_debug.log")

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
_TS_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ?")
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
RE_DEBUG_HEADER = re.compile(r"SIMILARITY MATRIX DEBUG - T(\d+)_E(\d+)_B(\d+)_(\w+)")
RE_MATRIX_ROW = re.compile(r"^\s*(\d+)\s*\|(.+)$")
# Extracts (value_str, marker) pairs from a raw matrix row in one pass.
# Marker is one of: * (self), + (positive pair), ! (margin violation), "" (none)
RE_CELL = re.compile(r"(-?\d+\.\d+)([*+!]?)")


# ---------------------------------------------------------------------------
# Compact marker encoding  (uint8 saves ~98 % disk/memory vs dtype=object)
# ---------------------------------------------------------------------------
_MARKER_ENC: dict[str, int] = {"": 0, "*": 1, "+": 2, "!": 3}
_MARKER_DEC = np.array(["", "*", "+", "!"])  # index by uint8 value


def _encode_markers(marker_rows: list[list[str]]) -> np.ndarray:
    """Encode a list-of-lists of marker chars into a compact uint8 ndarray."""
    arr = np.array(marker_rows, dtype=object)
    return np.vectorize(_MARKER_ENC.get)(arr, 0).astype(np.uint8)


def _decode_markers(markers: np.ndarray) -> np.ndarray:
    """Convert a compact uint8 marker array back to a string ndarray.

    No-op when *markers* is already a string/object array (backward-compatible
    with old pickle caches that stored dtype=object arrays).
    """
    if markers.dtype == np.uint8:
        return _MARKER_DEC[markers]
    return markers


# ---------------------------------------------------------------------------
# Main log parsing  (structured [T E B] lines from exp_debug0.log)
# ---------------------------------------------------------------------------


def _ensure_pending(pending: dict, task: int, epoch: int, batch: int) -> dict:
    """Return existing pending if it matches (T,E,B), otherwise start fresh."""
    if (
        pending.get("task") == task
        and pending.get("epoch") == epoch
        and pending.get("batch") == batch
    ):
        return pending
    return {"task": task, "epoch": epoch, "batch": batch}


@st.cache_data(show_spinner="Parsing training log…")
def parse_main_log(log_path: str) -> pd.DataFrame:
    """Stream-parse exp_debug0.log; return one row per (batch × loss_type).

    Robust to missing Contrastive stats lines — any [T E B] regex can
    independently start or continue a pending entry.
    RE_ANT groups: 1-3=T,E,B  4=pos_min 5=pos_max 6=neg_min 7=neg_max
                   8=gap_mean 9=gap_std 10=gap_min 11=gap_max
                   12=margin  13=violation_pct  14=ant_loss
    """
    rows: list[dict] = []
    pending: dict = {}
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = RE_CONTRASTIVE.search(line)
            if m:
                t, e, b = int(m[1]), int(m[2]), int(m[3])
                pending = _ensure_pending(pending, t, e, b)
                pending.update(
                    {
                        "pos_mean": float(m[4]),
                        "pos_std": float(m[5]),
                        "neg_mean": float(m[6]),
                        "neg_std": float(m[7]),
                        "gap": float(m[8]),
                    }
                )
                continue

            m = RE_ANT.search(line)
            if m:
                t, e, b = int(m[1]), int(m[2]), int(m[3])
                pending = _ensure_pending(pending, t, e, b)
                pending.update(
                    {
                        "pos_min": float(m[4]),
                        "pos_max": float(m[5]),
                        "neg_min": float(m[6]),
                        "neg_max": float(m[7]),
                        "gap_mean": float(m[8]),
                        "gap_std": float(m[9]),
                        "gap_min": float(m[10]),
                        "gap_max": float(m[11]),
                        "ant_margin": float(m[12]),
                        "violation_pct": float(m[13]),
                        "ant_loss": float(m[14]),
                    }
                )
                continue

            m = RE_LOSS_CONTRAST.search(line)
            if m:
                t, e, b = int(m[1]), int(m[2]), int(m[3])
                pending = _ensure_pending(pending, t, e, b)
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
            if m:
                t, e, b = int(m[1]), int(m[2]), int(m[3])
                pending = _ensure_pending(pending, t, e, b)
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

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Derive gap when Contrastive stats lines are absent
    if "gap" not in df.columns:
        if "pos_mean" in df.columns and "neg_mean" in df.columns:
            df["gap"] = df["pos_mean"] - df["neg_mean"]
        elif "pos_min" in df.columns:
            df["pos_mean"] = (df["pos_min"] + df["pos_max"]) / 2
            df["neg_mean"] = (df["neg_min"] + df["neg_max"]) / 2
            df["gap"] = df["pos_mean"] - df["neg_mean"]
    df = df.sort_values(["task", "epoch", "batch", "loss_type"]).reset_index(drop=True)
    return df


def aggregate_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Average all numeric metrics per (task, epoch, loss_type)."""
    num_cols = [
        c
        for c in [
            "pos_mean",
            "pos_std",
            "neg_mean",
            "neg_std",
            "gap",
            "gap_mean",
            "gap_std",
            "gap_min",
            "gap_max",
            "violation_pct",
            "ant_loss",
            "nll",
            "ant_loss_w",
            "total",
        ]
        if c in df.columns
    ]
    agg = (
        df.groupby(["task", "epoch", "loss_type"])
        .agg(**{c: (c, "mean") for c in num_cols}, batches=("batch", "count"))
        .reset_index()
    )
    agg = agg.sort_values(["task", "epoch", "loss_type"]).reset_index(drop=True)
    agg["run_idx"] = agg.groupby("loss_type").cumcount()
    return agg


# ---------------------------------------------------------------------------
# Similarity debug log  — indexing & entry parsing
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Building entry index (one-time scan)…")
def build_or_load_debug_index(log_path: str) -> dict:
    """
    Returns {entry_key: byte_offset, ...} for every SIMILARITY MATRIX DEBUG header.
    entry_key = "{task}_{epoch}_{batch}_{type}"
    The index is cached to a JSON file alongside the log so repeated runs are instant.
    """
    path = Path(log_path)
    cache_path = path.parent / f".{path.name}.index.json"

    if cache_path.exists():
        with open(cache_path, "r") as fh:
            return json.load(fh)

    index: dict[str, int] = {}
    with open(log_path, "rb") as fh:
        offset = 0
        for raw in fh:
            line = raw.decode("utf-8", errors="replace")
            m = RE_DEBUG_HEADER.search(line)
            if m:
                key = f"{m[1]}_{m[2]}_{m[3]}_{m[4]}"
                index[key] = offset
            offset += len(raw)

    with open(cache_path, "w") as fh:
        json.dump(index, fh)
    return index


@st.cache_data(show_spinner=False)
def _build_entries_df(log_path: str) -> pd.DataFrame:
    """Build and cache the entries DataFrame from the similarity debug log index.

    Cached at the Streamlit session level — avoids rebuilding on every rerun
    (each button click / slider move triggers a full page re-execution).
    """
    _idx = build_or_load_debug_index(log_path)
    _parsed = []
    for key in _idx:
        parts = key.split("_")
        if len(parts) == 4:
            _parsed.append(
                {
                    "task": int(parts[0]),
                    "epoch": int(parts[1]),
                    "batch": int(parts[2]),
                    "type": parts[3],
                    "key": key,
                }
            )
    return (
        pd.DataFrame(_parsed)
        .sort_values(["task", "epoch", "batch", "type"])
        .reset_index(drop=True)
    )


def _entry_disk_cache_path(log_path: str, offset: int) -> Path:
    """Returns the pickle cache path for a parsed entry (next to the log file)."""
    return Path(log_path).parent / ".sim_entry_cache" / f"{offset}.pkl"


@st.cache_data(show_spinner="Loading matrix entry…", max_entries=300)
def parse_debug_entry(log_path: str, offset: int) -> dict:
    """
    Parse one entry from similarity_debug.log starting at byte offset.
    Returns a rich dict with: context, margin, max_strategy, shape,
    local_maxs, anchors, overall, matrix (ndarray), markers (ndarray of str).
    """
    result: dict = {
        "context": "",
        "margin": None,
        "max_strategy": "",
        "shape": (32, 32),
        "local_maxs": None,
        "anchors": [],
        "overall": {},
        "matrix": None,
        "markers": None,
    }

    # Check disk cache first — avoids re-parsing 1.5 GB log on every cache miss
    _disk_cache = _entry_disk_cache_path(log_path, offset)
    if _disk_cache.exists():
        try:
            with open(_disk_cache, "rb") as _fh:
                cached = pickle.load(_fh)
            # Migrate old dtype=object markers to compact uint8
            mk = cached.get("markers")
            needs_resave = False
            if mk is not None and mk.dtype != np.uint8:
                cached["markers"] = np.vectorize(_MARKER_ENC.get)(mk, 0).astype(
                    np.uint8
                )
                mk = cached["markers"]
                needs_resave = True
            # Derive violation_pct from ! markers when absent (e.g. stale cache or antLocal log)
            if mk is not None and "violation_pct" not in cached.get("overall", {}):
                _N = cached.get("shape", (0,))[0]
                _bang = (
                    int(np.sum(mk == 3))
                    if mk.dtype == np.uint8
                    else int(np.sum(mk == "!"))
                )
                _total_neg = _N * (_N - 2)
                if _total_neg > 0:
                    cached.setdefault("overall", {})["violation_pct"] = (
                        _bang / _total_neg * 100.0
                    )
                    cached["overall"]["violations"] = _bang
                    cached["overall"]["total_neg"] = _total_neg
                    needs_resave = True
            if needs_resave:
                try:
                    with open(_disk_cache, "wb") as _fh:
                        pickle.dump(cached, _fh, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
            return cached
        except Exception:
            pass  # corrupted cache → fall through to re-parse

    # Read lines from offset until next entry header
    lines: list[str] = []
    with open(log_path, "rb") as fh:
        fh.seek(offset)
        for raw in fh:
            line = raw.decode("utf-8", errors="replace")
            msg = _TS_PREFIX.sub("", line).rstrip()
            # Stop when the *second* entry header is encountered
            if "SIMILARITY MATRIX DEBUG - T" in msg and lines:
                break
            lines.append(msg)

    # --- State ---
    in_matrix = False
    matrix_rows: list[list[float]] = []
    marker_rows: list[list[str]] = []
    cur_anchor: int | None = None
    in_above = False
    in_below = False
    anchors: dict[int, dict] = {}

    for msg in lines:
        # Skip pure separator lines
        stripped = msg.strip()
        if not stripped or stripped.startswith("=") or stripped.startswith("-" * 10):
            if in_matrix and not RE_MATRIX_ROW.match(msg):
                pass  # separators inside matrix section are fine to skip
            cur_anchor = None if stripped.startswith("=") else cur_anchor
            continue

        # Entry header
        m = RE_DEBUG_HEADER.search(msg)
        if m:
            result["context"] = f"T{m[1]}_E{m[2]}_B{m[3]}_{m[4]}"
            continue

        # Matrix section header
        if "COMPLETE SIMILARITY MATRIX" in msg:
            in_matrix = True
            continue

        # Legend lines — skip
        if (
            "Legend:" in msg
            or "= Self-similarity" in msg
            or "= Positive pair" in msg
            or "= Margin" in msg
        ):
            continue

        # ------------------------------------------------------------------
        # Parse matrix rows  (must check before other patterns to avoid
        # accidentally matching numeric content in verbose section)
        # ------------------------------------------------------------------
        if in_matrix:
            m = RE_MATRIX_ROW.match(msg)
            if m:
                # One findall call per row instead of 128 individual re.search calls
                pairs = RE_CELL.findall(m[2])
                if pairs:
                    matrix_rows.append([float(v) for v, _ in pairs])
                    marker_rows.append([mk for _, mk in pairs])
            continue  # don't fall through to verbose parsers

        # ------------------------------------------------------------------
        # Verbose section (before matrix)
        # ------------------------------------------------------------------

        # Config lines
        if "Matrix shape:" in msg:
            sh = re.search(r"\((\d+),\s*(\d+)\)", msg)
            if sh:
                result["shape"] = (int(sh[1]), int(sh[2]))
            continue

        if "ANT margin:" in msg:
            m = re.search(rf"ANT margin: {_F}, Max strategy: (\w+)", msg)
            if m:
                result["margin"] = float(m[1])
                result["max_strategy"] = m[2]
            continue

        if "Local max per anchor:" in msg:
            m = re.search(rf"min={_F}, max={_F}, mean={_F}", msg)
            if m:
                result["local_maxs"] = {
                    "min": float(m[1]),
                    "max": float(m[2]),
                    "mean": float(m[3]),
                }
            continue

        if "Global max:" in msg:
            m = re.search(rf"Global max: {_F}", msg)
            if m:
                result["global_max"] = float(m[1])
            continue

        if "Threshold (max - margin):" in msg:
            m = re.search(rf"Threshold.*: {_F}", msg)
            if m:
                result["global_threshold"] = float(m[1])
            continue

        # Anchor section header  "--- Anchor N ---"
        m = re.match(r"---\s*Anchor\s*(\d+)\s*---", stripped)
        if m:
            cur_anchor = int(m[1])
            in_above = in_below = False
            anchors[cur_anchor] = {"idx": cur_anchor, "above": [], "below": []}
            continue

        # Overall statistics section
        if "Overall Statistics" in msg:
            cur_anchor = None
            in_above = in_below = False
            continue

        # Per-anchor details
        if cur_anchor is not None:
            if "Positive pair (idx" in msg:
                m = re.search(rf"Positive pair \(idx (\d+)\): {_F}", msg)
                if m:
                    anchors[cur_anchor]["pos_idx"] = int(m[1])
                    anchors[cur_anchor]["pos_sim"] = float(m[2])
                continue

            if "Anchor max:" in msg:
                m = re.search(rf"Anchor max: {_F}, Threshold: {_F}", msg)
                if m:
                    anchors[cur_anchor]["anchor_max"] = float(m[1])
                    anchors[cur_anchor]["threshold"] = float(m[2])
                continue

            if "Above threshold:" in msg and "Values" not in msg:
                m = re.search(r"Above threshold: (\d+), Below threshold: (\d+)", msg)
                if m:
                    anchors[cur_anchor]["above_count"] = int(m[1])
                    anchors[cur_anchor]["below_count"] = int(m[2])
                continue

            if "Values ABOVE threshold" in msg:
                in_above, in_below = True, False
                continue

            if "Top values BELOW threshold" in msg:
                in_above, in_below = False, True
                continue

            if "idx" in msg and "gap:" in msg:
                m = re.search(rf"idx (\d+): {_F} \(gap: ([+-]{_F})\)", msg)
                if m:
                    item = {"idx": int(m[1]), "sim": float(m[2]), "gap": float(m[3])}
                    if in_above:
                        anchors[cur_anchor]["above"].append(item)
                    elif in_below:
                        anchors[cur_anchor]["below"].append(item)
                continue

            # End of anchor block (non-anchor "---" separator)
            if stripped.startswith("---"):
                cur_anchor = None
                in_above = in_below = False
            continue

        # Overall statistics (cur_anchor is None here)
        if "Positive pairs:" in msg:
            m = re.search(rf"min={_F}, max={_F}, mean={_F}, std={_F}", msg)
            if m:
                result["overall"].update(
                    {
                        "pos_min": float(m[1]),
                        "pos_max": float(m[2]),
                        "pos_mean": float(m[3]),
                        "pos_std": float(m[4]),
                    }
                )
            continue

        if "Negative pairs:" in msg:
            m = re.search(rf"min={_F}, max={_F}, mean={_F}, std={_F}", msg)
            if m:
                result["overall"].update(
                    {
                        "neg_min": float(m[1]),
                        "neg_max": float(m[2]),
                        "neg_mean": float(m[3]),
                        "neg_std": float(m[4]),
                    }
                )
            continue

        if "Gap (pos_mean - neg_mean):" in msg:
            m = re.search(rf": {_F}", msg)
            if m:
                result["overall"]["gap"] = float(m[1])
            continue

        if "Margin violations:" in msg:
            cnt = re.search(r"(\d+) / (\d+)", msg)
            pct = re.search(rf"\({_F}%\)", msg)
            if cnt:
                result["overall"]["violations"] = int(cnt[1])
                result["overall"]["total_neg"] = int(cnt[2])
            if pct:
                result["overall"]["violation_pct"] = float(pct[1])
            continue

    # Finalise
    result["anchors"] = sorted(anchors.values(), key=lambda a: a["idx"])

    if matrix_rows:
        try:
            # Fast path: matrix is always NxN so all rows have the same length
            result["matrix"] = np.array(matrix_rows, dtype=np.float32)
            result["markers"] = _encode_markers(marker_rows)
        except ValueError:
            # Fallback for ragged rows (shouldn't happen with well-formed logs)
            max_cols = max(len(r) for r in matrix_rows)
            mat = np.zeros((len(matrix_rows), max_cols), dtype=np.float32)
            mk = np.full((len(matrix_rows), max_cols), "", dtype=object)
            for i, (rv, rm) in enumerate(zip(matrix_rows, marker_rows)):
                mat[i, : len(rv)] = rv
                mk[i, : len(rm)] = rm
            result["matrix"] = mat
            result["markers"] = np.vectorize(_MARKER_ENC.get)(mk, 0).astype(np.uint8)

    # Derive violation_pct from ! markers when the log omits the "Margin violations:" line
    # (e.g. antLocal strategy which doesn't log that summary line)
    if result["markers"] is not None and "violation_pct" not in result["overall"]:
        _mk = result["markers"]
        _N = result["shape"][0]
        _bang = (
            int(np.sum(_mk == 3)) if _mk.dtype == np.uint8 else int(np.sum(_mk == "!"))
        )
        # Each of the N anchors has N-2 potential negatives (excluding self + positive pair)
        _total_neg = _N * (_N - 2)
        if _total_neg > 0:
            result["overall"]["violation_pct"] = _bang / _total_neg * 100.0
            result["overall"]["violations"] = _bang
            result["overall"]["total_neg"] = _total_neg

    # Persist to disk so subsequent sessions skip re-parsing
    try:
        _disk_cache.parent.mkdir(exist_ok=True)
        with open(_disk_cache, "wb") as _fh:
            pickle.dump(result, _fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # non-fatal — disk cache is best-effort

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_TASK_COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def _task_color(task: int):
    return _TASK_COLORS[(task - 1) % len(_TASK_COLORS)]


_METRIC_PANELS: dict[str, list[str]] = {
    "Similarity (pos vs neg)": ["pos_mean", "neg_mean"],
    "Pos-Neg Gap": ["gap"],
    "Violation %": ["violation_pct"],
    "NCE Loss": ["nll"],
    "ANT Loss (weighted)": ["ant_loss_w"],
    "Total Loss": ["total"],
}


def plot_overview(epoch_df: pd.DataFrame, loss_type: str, metric: str) -> plt.Figure:
    """Return a matplotlib figure for the selected epoch-level metric."""
    sub = epoch_df[epoch_df["loss_type"] == loss_type]
    tasks = sorted(sub["task"].unique())
    cols = _METRIC_PANELS.get(metric, ["gap"])
    is_sim_panel = metric == "Similarity (pos vs neg)"

    n_axes = 1
    fig, axes_raw = plt.subplots(n_axes, 1, figsize=(13, 4), squeeze=False)
    ax = axes_raw[0][0]
    fig.suptitle(f"[{loss_type}]  {metric}  —  epoch avg", fontsize=11)

    for task in tasks:
        s = sub[sub["task"] == task]
        c = _task_color(task)
        if is_sim_panel:
            ax.plot(
                s["run_idx"], s["pos_mean"], color=c, lw=1.4, label=f"T{int(task)} pos"
            )
            ax.plot(
                s["run_idx"],
                s["neg_mean"],
                color=c,
                lw=1.0,
                ls="--",
                alpha=0.75,
                label=f"T{int(task)} neg",
            )
        else:
            for col in cols:
                if col in s.columns:
                    ax.plot(
                        s["run_idx"], s[col], color=c, lw=1.4, label=f"T{int(task)}"
                    )

    if metric == "Pos-Neg Gap":
        ax.axhline(0, color="tomato", ls=":", lw=0.9, alpha=0.8)
    if metric == "Violation %":
        ax.axhline(0, color="limegreen", ls=":", lw=0.9, alpha=0.8)

    # Task boundary separators
    for task in tasks[1:]:
        mask = sub["task"] == task
        if mask.any():
            ax.axvline(
                sub.loc[mask, "run_idx"].iloc[0],
                color="gray",
                ls=":",
                lw=0.8,
                alpha=0.6,
            )

    ax.set_xlabel("Epoch index (sequential across tasks)")
    ax.legend(fontsize=7, ncol=max(1, len(tasks) // 3))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def plot_per_task(epoch_df: pd.DataFrame, loss_type: str) -> plt.Figure:
    """4-panel (loss + gap) grid, one column pair per task."""
    sub = epoch_df[epoch_df["loss_type"] == loss_type]
    tasks = sorted(sub["task"].unique())
    n = len(tasks)

    fig, axes_grid = plt.subplots(n, 2, figsize=(13, 3 * n), squeeze=False)
    fig.suptitle(f"[{loss_type}]  Per-task detail", fontsize=11)

    for row, task in enumerate(tasks):
        s = sub[sub["task"] == task]
        ax_l, ax_g = axes_grid[row][0], axes_grid[row][1]
        if not s.empty:
            ax_l.plot(s["epoch"], s["nll"], lw=1.2, label="NCE")
            ax_l.plot(s["epoch"], s["ant_loss_w"], lw=1.2, label="ANT (w)")
            ax_l.plot(s["epoch"], s["total"], lw=1.5, ls="--", label="Total")
            ax_g.plot(s["epoch"], s["gap"], lw=1.2, color="tab:blue", label="gap")
            ax_g.axhline(0, color="tomato", ls=":", lw=0.8)
            if "violation_pct" in s.columns:
                ax_g2 = ax_g.twinx()
                ax_g2.plot(
                    s["epoch"], s["violation_pct"], lw=1.0, color="tab:red", alpha=0.7
                )
                ax_g2.set_ylabel("violation %", color="tab:red", fontsize=8)
        ax_l.set_title(f"Task {int(task)}", fontsize=9)
        ax_l.set_xlabel("Epoch")
        ax_l.legend(fontsize=8)
        ax_l.grid(True, alpha=0.3)
        ax_g.set_title(f"Task {int(task)}  gap & viol.", fontsize=9)
        ax_g.set_xlabel("Epoch")
        ax_g.set_ylabel("pos−neg gap")
        ax_g.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def render_matrix_heatmap(
    matrix: np.ndarray,
    markers: np.ndarray | None,
    margin: float | None,
    max_strategy: str,
    title: str = "",
) -> plt.Figure:
    """32×32 similarity heatmap with threshold-violation overlays."""
    N = matrix.shape[0]
    batch_size = N // 2
    if markers is not None:
        markers = _decode_markers(markers)
    cell_px = max(0.35, min(0.55, 12 / N))
    fig_size = min(N * cell_px + 2.5, 16)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 0.5))

    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        matrix,
        cmap=cmap,
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.15,
        linecolor="#cccccc",
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.65},
        ax=ax,
        annot=False,
    )

    # Overlay markers
    if markers is not None:
        for i in range(min(N, markers.shape[0])):
            for j in range(min(N, markers.shape[1])):
                mk = markers[i, j]
                if mk == "*":
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "·",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="#333333",
                    )
                elif mk == "+":
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "+",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#00cc44",
                        fontweight="bold",
                    )
                elif mk == "!":
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "!",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#cc0000",
                        fontweight="bold",
                    )

    # Quadrant divider
    ax.axhline(batch_size, color="royalblue", lw=1.8, ls="--", alpha=0.8)
    ax.axvline(batch_size, color="royalblue", lw=1.8, ls="--", alpha=0.8)

    strat = max_strategy or "?"
    m_str = f"margin={margin}" if margin is not None else ""
    ax.set_title(f"{title}  [{m_str}, max={strat}]", fontsize=9, pad=6)
    ax.set_xlabel("Column index", fontsize=8)
    ax.set_ylabel("Row index", fontsize=8)

    step = max(1, N // 16)
    ticks = list(range(0, N, step))
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks], fontsize=6)
    ax.set_yticks([t + 0.5 for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks], fontsize=6)

    fig.tight_layout()
    return fig


def render_anchor_chart(anchors: list[dict], N: int) -> plt.Figure:
    """Two-panel chart: similarity values + threshold-violation counts per anchor."""
    if not anchors:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            "No anchor data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        return fig

    batch_size = N // 2
    ancs = sorted(anchors, key=lambda a: a["idx"])
    idxs = [a["idx"] for a in ancs]
    pos_sims = [a.get("pos_sim", float("nan")) for a in ancs]
    anchor_maxs = [a.get("anchor_max", float("nan")) for a in ancs]
    thresholds = [a.get("threshold", float("nan")) for a in ancs]
    above_cnts = [a.get("above_count", 0) for a in ancs]

    x = np.arange(len(idxs))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(idxs) * 0.9), 6))
    fig.suptitle(f"Per-anchor statistics  (showing {len(ancs)} anchors)", fontsize=10)

    ax1.bar(x - w, pos_sims, w, label="pos_sim", color="#2ca02c", alpha=0.85)
    ax1.bar(x, anchor_maxs, w, label="anchor_max", color="#ff7f0e", alpha=0.85)
    ax1.bar(x + w, thresholds, w, label="threshold", color="#d62728", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"A{i}" for i in idxs])
    ax1.set_ylabel("Cosine similarity")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3, axis="y")

    bar_colors = ["#d62728" if c > 0 else "#2ca02c" for c in above_cnts]
    ax2.bar(x, above_cnts, color=bar_colors, alpha=0.85)
    ax2.axhline(
        batch_size - 1,
        color="gray",
        ls="--",
        lw=0.9,
        label=f"max negatives ({batch_size - 1})",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"A{i}" for i in idxs])
    ax2.set_ylabel("Negatives above threshold")
    ax2.set_ylim(0, batch_size)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def render_matrix_heatmap_plotly(
    matrix: np.ndarray,
    markers: np.ndarray | None,
    margin: float | None,
    max_strategy: str,
    title: str = "",
    show_violation_markers: bool = False,
) -> go.Figure:
    """Interactive Plotly heatmap — hover shows sim value per cell, zoom/pan enabled.

    Set show_violation_markers=False to hide the visual '!' overlays.
    """
    N = matrix.shape[0]
    batch_size = N // 2

    mk_labels = {"*": "self-sim", "+": "positive pair", "!": "ANT violation"}
    # Decode compact uint8 markers (no-op for old object-array caches)
    if markers is not None:
        markers = _decode_markers(markers)

    # Build customdata for marker labels (lightweight vs N² hover strings)
    if markers is not None:
        _mk = markers[:N, :N]
        mk_text = np.vectorize(lambda m: mk_labels.get(str(m), "negative"))(_mk)
    else:
        mk_text = np.full((N, N), "negative", dtype=object)

    strat = max_strategy or "?"
    m_str = f"margin={margin}" if margin is not None else ""
    fig = px.imshow(
        matrix,
        color_continuous_scale="RdYlGn",
        zmin=0.0,
        zmax=1.0,
        aspect="equal",
        labels={"color": "Cosine Sim"},
        title=f"{title}  [{m_str}, max={strat}]",
    )
    fig.update_traces(
        customdata=mk_text,
        hovertemplate=(
            "row %{y} / col %{x}<br>"
            "sim = %{z:.4f}<br>"
            "<i>%{customdata}</i>"
            "<extra></extra>"
        ),
    )

    # Quadrant divider (view-1 / view-2 boundary)
    fig.add_hline(
        y=batch_size - 0.5,
        line_dash="dash",
        line_color="royalblue",
        line_width=2,
        annotation_text="view boundary",
        annotation_position="right",
        annotation_font_size=9,
    )
    fig.add_vline(
        x=batch_size - 0.5, line_dash="dash", line_color="royalblue", line_width=2
    )

    # Marker overlays as scatter traces — O(1) traces vs O(N) layout annotations.
    # For 256×256 matrices with ~16K violations, this cuts render time from ~10s to <0.5s.
    # White is used for ! (violations) so they remain visible on red cells (low similarity).
    if markers is not None:
        _mk = markers[:N, :N]
        overlay_markers = [("+", "#00cc44")]
        if show_violation_markers:
            overlay_markers.append(("!", "#ffffff"))
        for sym, color in overlay_markers:
            coords = np.argwhere(_mk == sym)
            if len(coords) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=coords[:, 1].tolist(),
                        y=coords[:, 0].tolist(),
                        mode="text",
                        text=[sym] * len(coords),
                        textfont=dict(color=color, size=8, family="Arial Black"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    size = max(420, min(N * 18 + 120, 860))
    fig.update_layout(
        height=size,
        margin=dict(l=40, r=60, t=60, b=40),
        coloraxis_colorbar=dict(title="Cosine Sim", thickness=14, len=0.75),
    )
    return fig


def render_anchor_chart_plotly(anchors: list[dict], N: int) -> go.Figure:
    """Interactive two-panel Plotly chart: similarities + violation counts per anchor."""
    if not anchors:
        fig = go.Figure()
        fig.add_annotation(
            text="No anchor data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        return fig

    batch_size = N // 2
    ancs = sorted(anchors, key=lambda a: a["idx"])
    labels = [f"A{a['idx']}" for a in ancs]
    pos_sims = [a.get("pos_sim", None) for a in ancs]
    anchor_maxs = [a.get("anchor_max", None) for a in ancs]
    thresholds = [a.get("threshold", None) for a in ancs]
    above_cnts = [a.get("above_count", 0) for a in ancs]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Similarity per anchor",
            "Negatives above threshold (violations)",
        ),
        vertical_spacing=0.2,
    )
    fig.add_trace(
        go.Bar(
            name="pos_sim",
            x=labels,
            y=pos_sims,
            marker_color="#2ca02c",
            hovertemplate="<b>%{x}</b><br>pos_sim = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="anchor_max",
            x=labels,
            y=anchor_maxs,
            marker_color="#ff7f0e",
            hovertemplate="<b>%{x}</b><br>anchor_max = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="threshold",
            x=labels,
            y=thresholds,
            marker_color="#d62728",
            opacity=0.75,
            hovertemplate="<b>%{x}</b><br>threshold = %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Cosine similarity", range=[0.0, 1.05], row=1, col=1)

    bar_cols = ["#d62728" if c > 0 else "#2ca02c" for c in above_cnts]
    fig.add_trace(
        go.Bar(
            name="violations",
            x=labels,
            y=above_cnts,
            marker_color=bar_cols,
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>violations = %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=batch_size - 1,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"max negatives = {batch_size - 1}",
        annotation_position="right",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Violações (acima do threshold)", range=[0, batch_size], row=2, col=1
    )
    fig.update_layout(
        height=480,
        barmode="group",
        margin=dict(l=55, r=20, t=80, b=40),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0.0),
    )
    # Push the first subplot title down so it doesn't collide with the legend
    fig.layout.annotations[0].update(y=fig.layout.annotations[0].y - 0.03)
    return fig


def _is_baseline_experiment(exp_label: str) -> bool:
    """Return True when experiment name indicates baseline ANT weight (antB0)."""
    return bool(re.search(r"(?:^|_)antB0(?:[._]|$)", exp_label))


def _apply_margin_override(entry: dict, margin_override: float | None) -> dict:
    """Return a copy of *entry* with recomputed thresholds/violations for a new margin.

    This enables post-hoc comparison of ANT-violation behavior (especially for
    baseline ant_beta=0 runs) without rerunning training.
    """
    if margin_override is None:
        return entry

    matrix = entry.get("matrix")
    if matrix is None or matrix.size == 0:
        out = dict(entry)
        out["margin"] = float(margin_override)
        return out

    N = int(matrix.shape[0])
    batch_size = N // 2
    max_strat = (entry.get("max_strategy") or "").lower()
    use_global = max_strat == "global"

    cos_sim_anchors = matrix[:batch_size, :batch_size]
    mask_diag = np.eye(batch_size, dtype=bool)
    neg_only = np.where(mask_diag, -np.inf, cos_sim_anchors)

    if use_global:
        anchor_maxs = np.full(batch_size, float(np.max(neg_only)), dtype=np.float32)
    else:
        anchor_maxs = np.max(neg_only, axis=1).astype(np.float32)
    thresholds = anchor_maxs - float(margin_override)

    # Rebuild marker matrix using the same convention as debug logger:
    # only anchor rows from first half mark negative violations in first-half columns.
    new_markers = np.zeros((N, N), dtype=np.uint8)
    np.fill_diagonal(new_markers, 1)  # *
    for i in range(batch_size):
        new_markers[i, i + batch_size] = 2  # +
        row_vals = cos_sim_anchors[i]
        above = (row_vals >= thresholds[i]) & (~mask_diag[i])
        new_markers[i, :batch_size][above] = 3  # !

    violations = int(np.sum(new_markers == 3))
    total_neg = int(batch_size * max(0, batch_size - 1))
    violation_pct = 100.0 * violations / total_neg if total_neg > 0 else None

    # Recompute per-anchor summary tables (for anchors present in parsed logs)
    new_anchors = []
    for a in entry.get("anchors", []):
        i = int(a.get("idx", -1))
        if i < 0 or i >= batch_size:
            new_anchors.append(dict(a))
            continue

        row = cos_sim_anchors[i]
        pos_idx = i + batch_size
        pos_sim = float(matrix[i, pos_idx])
        anchor_max = float(anchor_maxs[i])
        anchor_threshold = float(thresholds[i])

        above_vals: list[tuple[int, float]] = []
        below_vals: list[tuple[int, float]] = []
        for j in range(batch_size):
            if j == i:
                continue
            v = float(row[j])
            if v >= anchor_threshold:
                above_vals.append((j, v))
            else:
                below_vals.append((j, v))

        above_vals.sort(key=lambda x: x[1], reverse=True)
        below_vals.sort(key=lambda x: x[1], reverse=True)

        new_anchors.append(
            {
                **a,
                "pos_idx": pos_idx,
                "pos_sim": pos_sim,
                "anchor_max": anchor_max,
                "threshold": anchor_threshold,
                "above_count": len(above_vals),
                "below_count": len(below_vals),
                "above": [
                    {
                        "idx": j,
                        "sim": v,
                        "gap": v - anchor_threshold,
                    }
                    for j, v in above_vals[:5]
                ],
                "below": [
                    {
                        "idx": j,
                        "sim": v,
                        "gap": anchor_threshold - v,
                    }
                    for j, v in below_vals[:3]
                ],
            }
        )

    new_local_maxs = None
    if not use_global:
        new_local_maxs = {
            "min": float(np.min(anchor_maxs)),
            "max": float(np.max(anchor_maxs)),
            "mean": float(np.mean(anchor_maxs)),
        }

    new_overall = dict(entry.get("overall", {}))
    new_overall["violations"] = violations
    new_overall["total_neg"] = total_neg
    if violation_pct is not None:
        new_overall["violation_pct"] = violation_pct

    return {
        **entry,
        "margin": float(margin_override),
        "markers": new_markers,
        "anchors": new_anchors,
        "local_maxs": new_local_maxs,
        "overall": new_overall,
    }


@st.cache_data(show_spinner=False, max_entries=1000)
def _get_batch_loss_components(
    main_log_path: str,
    task: int,
    epoch: int,
    batch: int,
    loss_type_hint: str | None = None,
) -> dict:
    """Return loss components for one (task, epoch, batch), preferring loss_type_hint."""
    if not Path(main_log_path).exists():
        return {}

    df = parse_main_log(main_log_path)
    if df.empty:
        return {}

    sub = df[(df["task"] == task) & (df["epoch"] == epoch) & (df["batch"] == batch)]
    if sub.empty:
        return {}

    if loss_type_hint is not None and "loss_type" in sub.columns:
        hinted = sub[sub["loss_type"] == loss_type_hint]
        if not hinted.empty:
            row = hinted.iloc[0]
        else:
            row = sub.iloc[0]
    else:
        row = sub.iloc[0]

    out = {"loss_type": row.get("loss_type")}
    for c in ["nll", "ant_loss", "ant_loss_w", "total"]:
        if c in row and pd.notna(row[c]):
            out[c] = float(row[c])
    return out


# ---------------------------------------------------------------------------
# Cached figure builders  — keyed on (log_path, offset) to avoid recomputing
# on every Streamlit rerun.  Each button click triggers a full page re-execution;
# without this cache, heatmap + anchor chart are rebuilt from scratch every time.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=500)
def _get_heatmap_figure(
    log_path: str,
    offset: int,
    show_violation_markers: bool = False,
    margin_override: float | None = None,
) -> "go.Figure | None":
    """Build and cache the Plotly similarity heatmap for the entry at *offset*."""
    entry = _apply_margin_override(parse_debug_entry(log_path, offset), margin_override)
    matrix = entry.get("matrix")
    if matrix is None or matrix.size == 0:
        return None
    return render_matrix_heatmap_plotly(
        matrix,
        entry.get("markers"),
        entry.get("margin"),
        entry.get("max_strategy", "?"),
        entry.get("context", ""),
        show_violation_markers=show_violation_markers,
    )


@st.cache_data(show_spinner=False, max_entries=500)
def _get_anchor_figure(
    log_path: str, offset: int, margin_override: float | None = None
) -> go.Figure:
    """Build and cache the Plotly anchor chart for the entry at *offset*."""
    entry = _apply_margin_override(parse_debug_entry(log_path, offset), margin_override)
    return render_anchor_chart_plotly(entry.get("anchors", []), entry["shape"][0])


@st.cache_data(show_spinner=False, max_entries=200)
def _get_diff_heatmap(
    log_a: str, off_a: int, log_b: str, off_b: int, label_a: str, label_b: str
) -> "go.Figure | None":
    """Build a difference heatmap (A − B) for two matched entries."""
    ea = parse_debug_entry(log_a, off_a)
    eb = parse_debug_entry(log_b, off_b)
    ma, mb = ea.get("matrix"), eb.get("matrix")
    if ma is None or mb is None or ma.shape != mb.shape:
        return None
    diff = ma - mb
    N = ma.shape[0]
    batch_size = N // 2
    abs_max = max(float(np.abs(diff).max()), 0.01)
    hover = [
        [
            f"row {i} / col {j}<br>Δ = {diff[i, j]:+.4f}<br>"
            f"A = {ma[i, j]:.4f} · B = {mb[i, j]:.4f}"
            for j in range(N)
        ]
        for i in range(N)
    ]
    fig = px.imshow(
        diff,
        color_continuous_scale="RdBu_r",
        zmin=-abs_max,
        zmax=abs_max,
        aspect="equal",
        labels={"color": "Δ Sim"},
        title=f"Δ ({label_a} − {label_b})",
    )
    fig.update_traces(text=hover, hovertemplate="%{text}<extra></extra>")
    fig.add_hline(
        y=batch_size - 0.5,
        line_dash="dash",
        line_color="royalblue",
        line_width=2,
    )
    fig.add_vline(
        x=batch_size - 0.5, line_dash="dash", line_color="royalblue", line_width=2
    )
    size = max(420, min(N * 18 + 120, 860))
    fig.update_layout(
        height=size,
        margin=dict(l=40, r=60, t=60, b=40),
        coloraxis_colorbar=dict(title="Δ Sim", thickness=14, len=0.75),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def _fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def main() -> None:
    st.set_page_config(
        page_title="TagFex · Similarity Viewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 TagFex — Similarity Matrix Viewer")
    st.caption(
        "Explore InfoNCE + ANT similarity matrices captured during Class-Incremental Learning training."
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Experimento")
        if _EXPERIMENTS:
            exp_key = st.selectbox(
                "Diretório",
                options=list(_EXPERIMENTS.keys()),
                index=(
                    list(_EXPERIMENTS.keys()).index(_DEFAULT_KEY) if _DEFAULT_KEY else 0
                ),
            )
            _sel_dir = _EXPERIMENTS[exp_key]
            main_log = str(_sel_dir / "exp_debug0.log")
            sim_log = str(_sel_dir / "similarity_debug.log")
        else:
            main_log = st.text_input(
                "Main log (exp_debug0.log)", value=DEFAULT_MAIN_LOG
            )
            sim_log = st.text_input("Similarity debug log", value=DEFAULT_SIM_LOG)
        st.divider()
        st.markdown(
            """**Matrix legend**
- **·** self-similarity (diagonal)
- **+** positive pair
- **!** margin violation
- **——** view-1 / view-2 boundary"""
        )
        st.caption(
            "**Violação de margin (!)** — ocorre quando um par *negativo* tem similaridade "
            "acima do threshold ANT = `anchor_max − margin`. "
            "O loss ANT penaliza esses pares, forçando-os abaixo do threshold e criando "
            "uma margem de segurança entre o negativo mais hard e a âncora. "
            "Quanto maior `violation_pct`, mais pares negativos são difíceis para o modelo."
        )

        st.divider()
        st.markdown("**Post-hoc margin (baseline)**")
        baseline_margin_override_enabled = st.checkbox(
            "Forçar margin=0.50 no baseline (sem re-treinar)", value=True
        )
        baseline_margin_override_value = st.number_input(
            "Margin exibida nos títulos",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help=(
                "Recalcula threshold, violações (!) e violation % para experimentos "
                "baseline (antB0) diretamente no viewer."
            ),
        )

    tab_overview, tab_inspector, tab_comparison = st.tabs(
        ["📈  Training Overview", "🔍  Batch Inspector", "🔄  Batch Comparison"]
    )

    # ==================================================================
    # TAB 1 — Training Overview
    # ==================================================================
    with tab_overview:
        if not Path(main_log).exists():
            st.warning(f"Log not found: `{main_log}`")
            st.stop()

        df = parse_main_log(main_log)
        if df.empty:
            st.error("No structured `[T E B]` records found. Check the log path.")
            st.stop()

        epoch_df = aggregate_by_epoch(df)

        st.success(
            f"Loaded **{len(df):,}** batch records · "
            f"**{int(df['task'].max())}** tasks · "
            f"epochs up to **{int(df['epoch'].max())}**"
        )

        # Controls
        c_lt, c_metric, c_tasks = st.columns([1, 2, 3])
        loss_type = c_lt.selectbox("Loss type", sorted(df["loss_type"].unique()))
        metric = c_metric.selectbox("Metric", list(_METRIC_PANELS.keys()))
        all_tasks = sorted(df["task"].unique().astype(int).tolist())
        sel_tasks = c_tasks.multiselect("Tasks", all_tasks, default=all_tasks)

        # Charts
        sub_epoch = epoch_df[
            (epoch_df["loss_type"] == loss_type) & (epoch_df["task"].isin(sel_tasks))
        ]
        if sub_epoch.empty:
            st.info("No data for the selected filters.")
        else:
            fig = plot_overview(sub_epoch, loss_type, metric)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption(
                f"Média por época de **{metric}** para o loss `{loss_type}`. "
                "Cada linha representa uma task; linhas tracejadas verticais cinzas marcam "
                "as fronteiras entre tasks. O eixo X é um índice sequencial global de época."
            )

            with st.expander("📊 Per-task detail (loss + gap × every task)"):
                fig2 = plot_per_task(sub_epoch, loss_type)
                st.pyplot(fig2, width="stretch")
                plt.close(fig2)
                st.caption(
                    "Detalhe por task: (esq.) loss NCE, ANT ponderada e total por época; "
                    "(dir.) gap pos−neg em azul e % de violações em vermelho (eixo secundário)."
                )

        with st.expander("📋 Epoch-average table"):
            show_cols = [
                c
                for c in [
                    "task",
                    "epoch",
                    "pos_mean",
                    "neg_mean",
                    "gap",
                    "violation_pct",
                    "ant_loss",
                    "nll",
                    "total",
                    "batches",
                ]
                if c in sub_epoch.columns
            ]
            st.dataframe(sub_epoch[show_cols].round(5), width="stretch")

    # ==================================================================
    # TAB 2 — Batch Inspector
    # ==================================================================
    with tab_inspector:
        if not Path(sim_log).exists():
            st.warning(f"Similarity log not found: `{sim_log}`")
            st.stop()

        index = build_or_load_debug_index(sim_log)

        if not index:
            st.error("No similarity matrix entries found in the log.")
            st.stop()

        # Build sorted entries list (cached — avoids rebuilding on every rerun)
        entries_df = _build_entries_df(sim_log)
        total_entries = len(entries_df)

        st.success(
            f"Index: **{total_entries:,}** entries · "
            f"**{entries_df['task'].nunique()}** tasks · "
            f"types: {', '.join(sorted(entries_df['type'].unique()))}"
        )

        # ---- Session-state index (single source of truth) ----
        if "inspector_idx" not in st.session_state:
            st.session_state.inspector_idx = 0

        idx = int(st.session_state.inspector_idx)
        idx = max(0, min(idx, total_entries - 1))
        cur_row = entries_df.iloc[idx]
        cur_task = int(cur_row["task"])
        cur_epoch = int(cur_row["epoch"])
        cur_batch = int(cur_row["batch"])
        cur_type = cur_row["type"]

        # ---- Navigation helpers ----
        def _find_idx(task, epoch, batch, etype):
            mask = (
                (entries_df["task"] == task)
                & (entries_df["epoch"] == epoch)
                & (entries_df["batch"] == batch)
                & (entries_df["type"] == etype)
            )
            m = entries_df[mask].index.tolist()
            return m[0] if m else idx

        def _prev_task_idx():
            prev = [t for t in sorted(entries_df["task"].unique()) if t < cur_task]
            if not prev:
                return idx
            return int(entries_df[entries_df["task"] == prev[-1]].index[0])

        def _next_task_idx():
            nxt = [t for t in sorted(entries_df["task"].unique()) if t > cur_task]
            if not nxt:
                return idx
            return int(entries_df[entries_df["task"] == nxt[0]].index[0])

        def _prev_epoch_idx():
            sub = entries_df[
                (entries_df["task"] == cur_task) & (entries_df["epoch"] < cur_epoch)
            ]
            if sub.empty:
                return idx
            ep = sub["epoch"].max()
            sub2 = sub[sub["epoch"] == ep]
            same = sub2[sub2["type"] == cur_type]
            return int((same if not same.empty else sub2).index[0])

        def _next_epoch_idx():
            sub = entries_df[
                (entries_df["task"] == cur_task) & (entries_df["epoch"] > cur_epoch)
            ]
            if sub.empty:
                return idx
            ep = sub["epoch"].min()
            sub2 = sub[sub["epoch"] == ep]
            same = sub2[sub2["type"] == cur_type]
            return int((same if not same.empty else sub2).index[0])

        # ---- Navigation bar ----
        nb = st.columns([1, 1, 1, 4, 1, 1, 1])
        if nb[0].button("⏮ Task", help="Primeira entrada da task anterior"):
            st.session_state.inspector_idx = _prev_task_idx()
            st.rerun()
        if nb[1].button("◀ Epoch", help="Época anterior (mesma task)  [↓]"):
            st.session_state.inspector_idx = _prev_epoch_idx()
            st.rerun()
        if nb[2].button("◀ Batch", help="Batch anterior  [←]"):
            st.session_state.inspector_idx = max(0, idx - 1)
            st.rerun()
        nb[3].markdown(
            f"<div style='text-align:center;padding-top:8px;font-size:0.9em'>"
            f"Entrada <b>{idx + 1}</b>&nbsp;/&nbsp;<b>{total_entries}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"T{cur_task} · E{cur_epoch} · B{cur_batch} · {cur_type}"
            f"</div>",
            unsafe_allow_html=True,
        )
        if nb[4].button("Batch ▶", help="Próximo batch  [→]"):
            st.session_state.inspector_idx = min(total_entries - 1, idx + 1)
            st.rerun()
        if nb[5].button("Epoch ▶", help="Próxima época (mesma task)  [↑]"):
            st.session_state.inspector_idx = _next_epoch_idx()
            st.rerun()
        if nb[6].button("Task ▶⏭", help="Primeira entrada da próxima task"):
            st.session_state.inspector_idx = _next_task_idx()
            st.rerun()

        # ---- Slider ----
        def _fmt_entry(i: int) -> str:
            r = entries_df.iloc[i]
            return (
                f"T{int(r['task'])}·E{int(r['epoch'])}·B{int(r['batch'])}·{r['type']}"
            )

        slider_val = st.select_slider(
            "Entrada",
            options=list(range(total_entries)),
            value=idx,
            format_func=_fmt_entry,
            label_visibility="collapsed",
        )
        if slider_val != idx:
            st.session_state.inspector_idx = slider_val
            st.rerun()

        # ---- Jump to selectors ----
        with st.expander("🔍 Ir para entrada específica", expanded=False):
            avail_tasks_j = sorted(entries_df["task"].unique().tolist())
            # cascade: reset children when parent changes
            if (
                "j_task" not in st.session_state
                or st.session_state.j_task not in avail_tasks_j
            ):
                st.session_state.j_task = cur_task
            jc1, jc2, jc3, jc4, jc5 = st.columns([1, 1, 1, 1, 0.6])
            j_task = jc1.selectbox("Task", avail_tasks_j, key="j_task")

            avail_epochs_j = sorted(
                entries_df[entries_df["task"] == j_task]["epoch"].unique().tolist()
            )
            if (
                "j_epoch" not in st.session_state
                or st.session_state.j_epoch not in avail_epochs_j
            ):
                st.session_state.j_epoch = avail_epochs_j[0]
            j_epoch = jc2.selectbox("Época", avail_epochs_j, key="j_epoch")

            avail_batches_j = sorted(
                entries_df[
                    (entries_df["task"] == j_task) & (entries_df["epoch"] == j_epoch)
                ]["batch"]
                .unique()
                .tolist()
            )
            if (
                "j_batch" not in st.session_state
                or st.session_state.j_batch not in avail_batches_j
            ):
                st.session_state.j_batch = avail_batches_j[0]
            j_batch = jc3.selectbox("Batch", avail_batches_j, key="j_batch")

            avail_types_j = sorted(
                entries_df[
                    (entries_df["task"] == j_task)
                    & (entries_df["epoch"] == j_epoch)
                    & (entries_df["batch"] == j_batch)
                ]["type"]
                .unique()
                .tolist()
            )
            if (
                "j_type" not in st.session_state
                or st.session_state.j_type not in avail_types_j
            ):
                st.session_state.j_type = avail_types_j[0]
            j_type = jc4.selectbox("Tipo", avail_types_j, key="j_type")

            jc5.markdown("&nbsp;", unsafe_allow_html=True)
            if jc5.button("Ir ↵"):
                st.session_state.inspector_idx = _find_idx(
                    j_task, j_epoch, j_batch, j_type
                )
                st.rerun()

        # ---- Keyboard navigation (← → ↑ ↓) ----
        components.html(
            """
<script>
(function() {
    const doc = window.parent.document;
    function clickBtn(label) {
        const b = [...doc.querySelectorAll('button')].find(el => el.innerText.trim() === label);
        if (b) b.click();
    }
    doc.addEventListener('keydown', function(e) {
        const tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (e.key === 'ArrowRight') { e.preventDefault(); clickBtn('Batch \u25b6'); }
        if (e.key === 'ArrowLeft')  { e.preventDefault(); clickBtn('\u25c4 Batch'); }
        if (e.key === 'ArrowUp')    { e.preventDefault(); clickBtn('Epoch \u25b6'); }
        if (e.key === 'ArrowDown')  { e.preventDefault(); clickBtn('\u25c4 Epoch'); }
    }, false);
})();
</script>
""",
            height=0,
        )

        # ---- Parse entry ----
        entry_key = cur_row["key"]
        inspector_margin_override = None
        if baseline_margin_override_enabled and _EXPERIMENTS:
            if _is_baseline_experiment(exp_key):
                inspector_margin_override = float(baseline_margin_override_value)

        entry = _apply_margin_override(
            parse_debug_entry(sim_log, index[entry_key]), inspector_margin_override
        )

        overall = entry.get("overall", {})
        margin = entry.get("margin")
        max_strat = entry.get("max_strategy", "?")
        local_maxs = entry.get("local_maxs")
        matrix = entry.get("matrix")
        markers_arr = entry.get("markers")
        anchors = entry.get("anchors", [])
        N = entry["shape"][0]

        # ---- KPI row ----
        st.subheader(f"`{entry['context']}`")
        if inspector_margin_override is not None:
            st.caption(f"Baseline exibido com margin = {inspector_margin_override:.2f}")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("pos mean", _fmt(overall.get("pos_mean")))
        k2.metric("neg mean", _fmt(overall.get("neg_mean")))
        k3.metric("gap", _fmt(overall.get("gap")))

        viol = overall.get("violation_pct")
        if viol is None and "violations" in overall and "total_neg" in overall:
            viol = 100.0 * overall["violations"] / max(1, overall["total_neg"])
        k4.metric("violation %", f"{viol:.1f}%" if viol is not None else "—")
        k5.metric("ANT margin", f"{margin}" if margin is not None else "—")
        k6.metric("max strategy", max_strat)

        _batch_loss = _get_batch_loss_components(
            main_log,
            cur_task,
            cur_epoch,
            cur_batch,
            loss_type_hint=cur_type,
        )
        if _batch_loss:
            st.caption(
                "Loss do batch: "
                f"tipo={_batch_loss.get('loss_type', '—')} · "
                f"NCE={_fmt(_batch_loss.get('nll'))} · "
                f"ANT={_fmt(_batch_loss.get('ant_loss'))} · "
                f"ANT×β={_fmt(_batch_loss.get('ant_loss_w'))} · "
                f"Total={_fmt(_batch_loss.get('total'))}"
            )

        if local_maxs:
            lm1, lm2 = st.columns(2)
            lm1.metric("local max (mean)", _fmt(local_maxs.get("mean")))
            lm2.metric(
                "local max (min → max)",
                f"{_fmt(local_maxs.get('min'), 3)} → {_fmt(local_maxs.get('max'), 3)}",
            )

        st.divider()

        # ---- Loss components evolution (current task) ----
        with st.expander("📉 Loss components — task evolution", expanded=True):
            if Path(main_log).exists():
                _loss_df = parse_main_log(main_log)
                if not _loss_df.empty:
                    _epoch_agg = aggregate_by_epoch(_loss_df)
                    _task_loss = _epoch_agg[_epoch_agg["task"] == cur_task].copy()
                    if not _task_loss.empty:
                        _cols = [
                            c
                            for c in ["nll", "ant_loss_w", "total"]
                            if c in _task_loss.columns
                        ]
                        _color = {
                            "contrast": {
                                "nll": "#1f77b4",
                                "ant_loss_w": "#ff7f0e",
                                "total": "#2ca02c",
                            },
                            "kd": {
                                "nll": "#9467bd",
                                "ant_loss_w": "#8c564b",
                                "total": "#e377c2",
                            },
                        }
                        _dash = {"contrast": "solid", "kd": "dash"}
                        _label = {"nll": "NCE", "ant_loss_w": "ANT×β", "total": "Total"}

                        fig_lc = go.Figure()
                        for _ltype in sorted(_task_loss["loss_type"].unique()):
                            _sub = _task_loss[_task_loss["loss_type"] == _ltype]
                            for _col in _cols:
                                fig_lc.add_trace(
                                    go.Scatter(
                                        x=_sub["epoch"],
                                        y=_sub[_col],
                                        mode="lines",
                                        name=f"{_ltype} · {_label.get(_col, _col)}",
                                        line=dict(
                                            color=_color.get(_ltype, {}).get(_col),
                                            dash=_dash.get(_ltype, "solid"),
                                        ),
                                        hovertemplate=(
                                            f"epoch=%{{x}}<br>{_label.get(_col, _col)}"
                                            f"=%{{y:.4f}}<extra>{_ltype}</extra>"
                                        ),
                                    )
                                )
                        fig_lc.add_vline(
                            x=cur_epoch,
                            line_dash="dot",
                            line_color="red",
                            annotation_text=f" E{cur_epoch}",
                            annotation_position="top right",
                        )
                        fig_lc.update_layout(
                            height=250,
                            margin=dict(l=40, r=20, t=30, b=30),
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="left",
                                x=0,
                            ),
                            title=dict(
                                text=f"T{cur_task} — componentes da loss por época",
                                font=dict(size=12),
                            ),
                        )
                        st.plotly_chart(fig_lc, use_container_width=True)
                        st.caption(
                            "**NCE**: perda InfoNCE (NLL bruto)  ·  "
                            "**ANT×β**: `ant_beta × ANT_loss` (componente penalizadora)  ·  "
                            "**Total**: soma ponderada das componentes.  "
                            "Linha tracejada **vermelha**: época inspeccionada — use ← → para navegar."
                        )
                    else:
                        st.info(f"Sem dados de loss para T{cur_task} no main log.")
            else:
                st.info(f"Main log não encontrado: `{main_log}`")

        # ---- Heatmap + anchor chart ----
        col_heat, col_anchor = st.columns([5, 3])

        with col_heat:
            _heat_title = "**Similarity Heatmap**"
            if inspector_margin_override is not None:
                _heat_title = (
                    f"**Similarity Heatmap — margin={inspector_margin_override:.2f}**"
                )
            st.markdown(_heat_title)
            fig_h = _get_heatmap_figure(
                sim_log,
                index[entry_key],
                margin_override=inspector_margin_override,
            )
            if fig_h is not None:
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption(
                    "Similaridades coseno entre todos os embeddings do batch. "
                    "**Eixos**: índices no batch (linha = âncora, coluna = comparado). "
                    "**Verde** = alta similaridade · **Vermelho** = baixa. "
                    "**Linha azul tracejada**: separa embeddings da view-1 (0…N/2−1) dos da view-2 (N/2…N−1). "
                    "**+** par positivo · **!** violação ANT. "
                    "Use a barra de ferramentas Plotly (canto sup. direito) para zoom, pan e leitura de valores por célula."
                )

                with st.expander("Raw matrix values (first 8×8)"):
                    st.dataframe(
                        pd.DataFrame(
                            matrix[:8, :8],
                            columns=[f"c{j}" for j in range(min(8, matrix.shape[1]))],
                            index=[f"r{i}" for i in range(min(8, matrix.shape[0]))],
                        ).round(4),
                        width="stretch",
                    )
            else:
                st.info(
                    "Full matrix not available in this entry.  "
                    "The log may have been written before the matrix section."
                )

        with col_anchor:
            _anc_title = "**Per-anchor statistics**"
            if inspector_margin_override is not None:
                _anc_title = f"**Per-anchor statistics — margin={inspector_margin_override:.2f}**"
            st.markdown(_anc_title)
            if anchors:
                fig_a = _get_anchor_figure(
                    sim_log,
                    index[entry_key],
                    margin_override=inspector_margin_override,
                )
                st.plotly_chart(fig_a, use_container_width=True)
                st.caption(
                    "**Painel superior** — para cada âncora: "
                    "`pos_sim` similaridade com o par positivo; "
                    "`anchor_max` a maior similaridade âncora↔negativo (define o threshold ANT); "
                    "`threshold = anchor_max − margin` (limite abaixo do qual os negativos são aceitáveis). "
                    "**Painel inferior** — nº de negativos *acima* do threshold; "
                    "barras **vermelhas** = violações activas · **verdes** = âncora sem violações."
                )

                with st.expander("Anchor detail table"):
                    st.caption(
                        "**anchor** índice do embedding âncora no batch  ·  "
                        "**pos_idx** índice do par positivo  ·  "
                        "**pos_sim** similaridade coseno âncora↔positivo  ·  "
                        "**anchor_max** maior similaridade âncora↔negativo (hard negative)  ·  "
                        "**threshold** `anchor_max − margin` (negativos acima disto são violações)  ·  "
                        "**above_count** nº de negativos acima do threshold  ·  "
                        "**below_count** nº de negativos abaixo do threshold"
                    )
                    rows_anc = []
                    for a in anchors:
                        rows_anc.append(
                            {
                                "anchor": a["idx"],
                                "pos_sim": _fmt(a.get("pos_sim")),
                                "pos_idx": a.get("pos_idx", ""),
                                "anchor_max": _fmt(a.get("anchor_max")),
                                "threshold": _fmt(a.get("threshold")),
                                "above_count": a.get("above_count", ""),
                                "below_count": a.get("below_count", ""),
                            }
                        )
                    st.dataframe(pd.DataFrame(rows_anc), width="stretch")

                    # Top violations table
                    viol_rows = []
                    for a in anchors:
                        for item in a.get("above", []):
                            viol_rows.append(
                                {
                                    "anchor": a["idx"],
                                    "neg_idx": item["idx"],
                                    "sim": item["sim"],
                                    "gap": item["gap"],
                                }
                            )
                    if viol_rows:
                        st.markdown("**Top threshold violations**")
                        st.caption(
                            "**anchor** âncora que originou a violação  ·  "
                            "**neg_idx** índice do negativo infractor  ·  "
                            "**sim** similaridade coseno âncora↔negativo  ·  "
                            "**gap** `sim − threshold` (positivo = quanto ultrapassa o limite; "
                            "quanto maior, mais severa a violação)"
                        )
                        st.dataframe(
                            pd.DataFrame(viol_rows)
                            .sort_values("sim", ascending=False)
                            .reset_index(drop=True),
                            width="stretch",
                        )
            else:
                st.info("No per-anchor data found in this entry.")

        # ---- Prefetch adjacent entries (figures + parse data) for faster navigation ----
        # Covers ±3 batches (ArrowLeft/Right) and adjacent epochs (ArrowUp/Down + buttons)
        _pf_indices: set[int] = set()
        for _pf_delta in (-3, -2, -1, +1, +2, +3):
            _pf_i = idx + _pf_delta
            if 0 <= _pf_i < total_entries:
                _pf_indices.add(_pf_i)
        for _ep_fn in (_prev_epoch_idx, _next_epoch_idx):
            _ep_i = _ep_fn()
            if _ep_i != idx:
                _pf_indices.add(_ep_i)
        for _pf_i in _pf_indices:
            _pf_key = entries_df.iloc[_pf_i]["key"]
            _get_heatmap_figure(
                sim_log,
                index[_pf_key],
                margin_override=inspector_margin_override,
            )
            _get_anchor_figure(
                sim_log,
                index[_pf_key],
                margin_override=inspector_margin_override,
            )

    # ==================================================================
    # TAB 3 — Batch Comparison  (side-by-side two experiments)
    # ==================================================================
    with tab_comparison:
        if len(_EXPERIMENTS) < 2:
            st.warning(
                "Precisa de pelo menos **2 experimentos debug** disponíveis "
                "em `logs/` (cada um com `exp_debug0.log` e `similarity_debug.log`)."
            )
            st.stop()

        exp_keys = list(_EXPERIMENTS.keys())
        debug_keys = [k for k in exp_keys if k.startswith("debug_")]
        _default_a = debug_keys[0] if debug_keys else exp_keys[0]
        _default_b = (
            debug_keys[1]
            if len(debug_keys) > 1
            else exp_keys[min(1, len(exp_keys) - 1)]
        )

        c_sel_a, c_sel_b = st.columns(2)
        with c_sel_a:
            cmp_exp_a = st.selectbox(
                "Experimento A",
                exp_keys,
                index=exp_keys.index(_default_a),
                key="cmp_exp_a",
            )
        with c_sel_b:
            cmp_exp_b = st.selectbox(
                "Experimento B",
                exp_keys,
                index=exp_keys.index(_default_b),
                key="cmp_exp_b",
            )

        dir_a = _EXPERIMENTS[cmp_exp_a]
        dir_b = _EXPERIMENTS[cmp_exp_b]
        main_log_a = str(dir_a / "exp_debug0.log")
        main_log_b = str(dir_b / "exp_debug0.log")
        sim_log_a = str(dir_a / "similarity_debug.log")
        sim_log_b = str(dir_b / "similarity_debug.log")

        idx_a = build_or_load_debug_index(sim_log_a)
        idx_b = build_or_load_debug_index(sim_log_b)

        common_keys = sorted(set(idx_a.keys()) & set(idx_b.keys()))

        if not common_keys:
            st.error("Nenhuma entrada em comum entre os dois experimentos.")
            st.stop()

        # Build a lightweight DataFrame for the common entries
        _cmp_parsed = []
        for key in common_keys:
            parts = key.split("_")
            if len(parts) == 4:
                _cmp_parsed.append(
                    {
                        "task": int(parts[0]),
                        "epoch": int(parts[1]),
                        "batch": int(parts[2]),
                        "type": parts[3],
                        "key": key,
                    }
                )
        cmp_df = (
            pd.DataFrame(_cmp_parsed)
            .sort_values(["task", "epoch", "batch", "type"])
            .reset_index(drop=True)
        )
        total_cmp = len(cmp_df)

        st.success(
            f"**{total_cmp:,}** entradas em comum · "
            f"**{cmp_df['task'].nunique()}** tasks · "
            f"tipos: {', '.join(sorted(cmp_df['type'].unique()))}"
        )

        # ---- Session-state comparison index ----
        if "cmp_idx" not in st.session_state:
            st.session_state.cmp_idx = 0

        cmp_i = int(st.session_state.cmp_idx)
        cmp_i = max(0, min(cmp_i, total_cmp - 1))
        cmp_row = cmp_df.iloc[cmp_i]
        cmp_task = int(cmp_row["task"])
        cmp_epoch = int(cmp_row["epoch"])
        cmp_batch = int(cmp_row["batch"])
        cmp_type = cmp_row["type"]

        # ---- Navigation helpers ----
        def _cmp_prev_epoch():
            sub = cmp_df[(cmp_df["task"] == cmp_task) & (cmp_df["epoch"] < cmp_epoch)]
            if sub.empty:
                return cmp_i
            ep = sub["epoch"].max()
            sub2 = sub[sub["epoch"] == ep]
            same = sub2[sub2["type"] == cmp_type]
            return int((same if not same.empty else sub2).index[0])

        def _cmp_next_epoch():
            sub = cmp_df[(cmp_df["task"] == cmp_task) & (cmp_df["epoch"] > cmp_epoch)]
            if sub.empty:
                return cmp_i
            ep = sub["epoch"].min()
            sub2 = sub[sub["epoch"] == ep]
            same = sub2[sub2["type"] == cmp_type]
            return int((same if not same.empty else sub2).index[0])

        def _cmp_prev_task():
            prev = [t for t in sorted(cmp_df["task"].unique()) if t < cmp_task]
            if not prev:
                return cmp_i
            return int(cmp_df[cmp_df["task"] == prev[-1]].index[0])

        def _cmp_next_task():
            nxt = [t for t in sorted(cmp_df["task"].unique()) if t > cmp_task]
            if not nxt:
                return cmp_i
            return int(cmp_df[cmp_df["task"] == nxt[0]].index[0])

        # ---- Navigation bar ----
        cn = st.columns([1, 1, 1, 4, 1, 1, 1])
        if cn[0].button("⏮ Task", key="cmp_pt"):
            st.session_state.cmp_idx = _cmp_prev_task()
            st.rerun()
        if cn[1].button("◀ Epoch", key="cmp_pe"):
            st.session_state.cmp_idx = _cmp_prev_epoch()
            st.rerun()
        if cn[2].button("◀ Batch", key="cmp_pb"):
            st.session_state.cmp_idx = max(0, cmp_i - 1)
            st.rerun()
        cn[3].markdown(
            f"<div style='text-align:center;padding-top:8px;font-size:0.9em'>"
            f"Entrada <b>{cmp_i + 1}</b>&nbsp;/&nbsp;<b>{total_cmp}</b>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"T{cmp_task} · E{cmp_epoch} · B{cmp_batch} · {cmp_type}"
            f"</div>",
            unsafe_allow_html=True,
        )
        if cn[4].button("Batch ▶", key="cmp_nb"):
            st.session_state.cmp_idx = min(total_cmp - 1, cmp_i + 1)
            st.rerun()
        if cn[5].button("Epoch ▶", key="cmp_ne"):
            st.session_state.cmp_idx = _cmp_next_epoch()
            st.rerun()
        if cn[6].button("Task ⏭▶", key="cmp_nt"):
            st.session_state.cmp_idx = _cmp_next_task()
            st.rerun()

        # ---- Slider ----
        def _cmp_fmt(i: int) -> str:
            r = cmp_df.iloc[i]
            return (
                f"T{int(r['task'])}·E{int(r['epoch'])}·B{int(r['batch'])}·{r['type']}"
            )

        cmp_slider = st.select_slider(
            "Entrada",
            options=list(range(total_cmp)),
            value=cmp_i,
            format_func=_cmp_fmt,
            label_visibility="collapsed",
            key="cmp_slider",
        )
        if cmp_slider != cmp_i:
            st.session_state.cmp_idx = cmp_slider
            st.rerun()

        # ---- Keyboard navigation ----
        components.html(
            """
<script>
(function() {
    const doc = window.parent.document;
    function clickBtn(label) {
        const b = [...doc.querySelectorAll('button')].find(el => el.innerText.trim() === label);
        if (b) b.click();
    }
    doc.addEventListener('keydown', function(e) {
        const tag = e.target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (e.key === 'ArrowRight') { e.preventDefault(); clickBtn('Batch \\u25b6'); }
        if (e.key === 'ArrowLeft')  { e.preventDefault(); clickBtn('\\u25c4 Batch'); }
        if (e.key === 'ArrowUp')    { e.preventDefault(); clickBtn('Epoch \\u25b6'); }
        if (e.key === 'ArrowDown')  { e.preventDefault(); clickBtn('\\u25c4 Epoch'); }
    }, false);
})();
</script>
""",
            height=0,
        )

        # ---- Parse both entries ----
        entry_key = cmp_row["key"]
        cmp_margin_override_a = None
        cmp_margin_override_b = None
        if baseline_margin_override_enabled:
            if _is_baseline_experiment(cmp_exp_a):
                cmp_margin_override_a = float(baseline_margin_override_value)
            if _is_baseline_experiment(cmp_exp_b):
                cmp_margin_override_b = float(baseline_margin_override_value)

        entry_a = _apply_margin_override(
            parse_debug_entry(sim_log_a, idx_a[entry_key]), cmp_margin_override_a
        )
        entry_b = _apply_margin_override(
            parse_debug_entry(sim_log_b, idx_b[entry_key]), cmp_margin_override_b
        )

        # ---- Short labels for headers ----
        _short_a = cmp_exp_a.replace("debug_exp_cifar100_10-10_", "").replace(
            "_s1993", ""
        )
        _short_b = cmp_exp_b.replace("debug_exp_cifar100_10-10_", "").replace(
            "_s1993", ""
        )

        # ---- KPI comparison ----
        _cmp_title = f"`T{cmp_task}_E{cmp_epoch}_B{cmp_batch}_{cmp_type}`"
        st.subheader(_cmp_title)
        ov_a = entry_a.get("overall", {})
        ov_b = entry_b.get("overall", {})
        loss_a = _get_batch_loss_components(
            main_log_a,
            cmp_task,
            cmp_epoch,
            cmp_batch,
            loss_type_hint=cmp_type,
        )
        loss_b = _get_batch_loss_components(
            main_log_b,
            cmp_task,
            cmp_epoch,
            cmp_batch,
            loss_type_hint=cmp_type,
        )

        kpi_col_a, kpi_col_b = st.columns(2)
        with kpi_col_a:
            _label_a = f"**A** — `{_short_a}`"
            if cmp_margin_override_a is not None:
                _label_a += f" · **margin={cmp_margin_override_a:.2f}**"
            st.markdown(_label_a)
            ka1, ka2, ka3, ka4 = st.columns(4)
            ka1.metric("pos mean", _fmt(ov_a.get("pos_mean")))
            ka2.metric("neg mean", _fmt(ov_a.get("neg_mean")))
            ka3.metric("gap", _fmt(ov_a.get("gap")))
            viol_a = ov_a.get("violation_pct")
            if viol_a is None and "violations" in ov_a and "total_neg" in ov_a:
                viol_a = 100.0 * ov_a["violations"] / max(1, ov_a["total_neg"])
            ka4.metric("violation %", f"{viol_a:.1f}%" if viol_a is not None else "—")
            if loss_a:
                st.caption(
                    "Loss batch A: "
                    f"tipo={loss_a.get('loss_type', '—')} · "
                    f"NCE={_fmt(loss_a.get('nll'))} · "
                    f"ANT={_fmt(loss_a.get('ant_loss'))} · "
                    f"ANT×β={_fmt(loss_a.get('ant_loss_w'))} · "
                    f"Total={_fmt(loss_a.get('total'))}"
                )
        with kpi_col_b:
            _label_b = f"**B** — `{_short_b}`"
            if cmp_margin_override_b is not None:
                _label_b += f" · **margin={cmp_margin_override_b:.2f}**"
            st.markdown(_label_b)
            kb1, kb2, kb3, kb4 = st.columns(4)
            kb1.metric("pos mean", _fmt(ov_b.get("pos_mean")))
            kb2.metric("neg mean", _fmt(ov_b.get("neg_mean")))
            kb3.metric("gap", _fmt(ov_b.get("gap")))
            viol_b = ov_b.get("violation_pct")
            if viol_b is None and "violations" in ov_b and "total_neg" in ov_b:
                viol_b = 100.0 * ov_b["violations"] / max(1, ov_b["total_neg"])
            kb4.metric("violation %", f"{viol_b:.1f}%" if viol_b is not None else "—")
            if loss_b:
                st.caption(
                    "Loss batch B: "
                    f"tipo={loss_b.get('loss_type', '—')} · "
                    f"NCE={_fmt(loss_b.get('nll'))} · "
                    f"ANT={_fmt(loss_b.get('ant_loss'))} · "
                    f"ANT×β={_fmt(loss_b.get('ant_loss_w'))} · "
                    f"Total={_fmt(loss_b.get('total'))}"
                )

        st.divider()

        # ---- Side-by-side heatmaps ----
        heat_a, heat_b = st.columns(2)
        with heat_a:
            _cmp_heat_a = "**A** — Similarity Heatmap"
            if cmp_margin_override_a is not None:
                _cmp_heat_a += f" — margin={cmp_margin_override_a:.2f}"
            st.markdown(_cmp_heat_a)
            fig_ha = _get_heatmap_figure(
                sim_log_a,
                idx_a[entry_key],
                margin_override=cmp_margin_override_a,
            )
            if fig_ha is not None:
                st.plotly_chart(fig_ha, use_container_width=True, key="cmp_heat_a")
            else:
                st.info("Matriz não disponível para A.")
        with heat_b:
            _cmp_heat_b = "**B** — Similarity Heatmap"
            if cmp_margin_override_b is not None:
                _cmp_heat_b += f" — margin={cmp_margin_override_b:.2f}"
            st.markdown(_cmp_heat_b)
            fig_hb = _get_heatmap_figure(
                sim_log_b,
                idx_b[entry_key],
                margin_override=cmp_margin_override_b,
            )
            if fig_hb is not None:
                st.plotly_chart(fig_hb, use_container_width=True, key="cmp_heat_b")
            else:
                st.info("Matriz não disponível para B.")

        # ---- Difference heatmap ----
        with st.expander("Δ Mapa de diferença (A − B)", expanded=False):
            fig_diff = _get_diff_heatmap(
                sim_log_a,
                idx_a[entry_key],
                sim_log_b,
                idx_b[entry_key],
                _short_a,
                _short_b,
            )
            if fig_diff is not None:
                st.plotly_chart(fig_diff, use_container_width=True, key="cmp_diff")
                st.caption(
                    "**Vermelho** = A tem similaridade maior que B · "
                    "**Azul** = B tem similaridade maior que A. "
                    "Valores próximos de zero (branco) indicam concordância entre os experimentos."
                )
            else:
                st.info(
                    "Não foi possível calcular a diferença "
                    "(matrizes com shapes incompatíveis ou ausentes)."
                )

        st.divider()

        # ---- Side-by-side anchor charts ----
        anc_a, anc_b = st.columns(2)
        with anc_a:
            _cmp_anc_a = "**A** — Per-anchor statistics"
            if cmp_margin_override_a is not None:
                _cmp_anc_a += f" — margin={cmp_margin_override_a:.2f}"
            st.markdown(_cmp_anc_a)
            fig_aa = _get_anchor_figure(
                sim_log_a,
                idx_a[entry_key],
                margin_override=cmp_margin_override_a,
            )
            st.plotly_chart(fig_aa, use_container_width=True, key="cmp_anc_a")
        with anc_b:
            _cmp_anc_b = "**B** — Per-anchor statistics"
            if cmp_margin_override_b is not None:
                _cmp_anc_b += f" — margin={cmp_margin_override_b:.2f}"
            st.markdown(_cmp_anc_b)
            fig_ab = _get_anchor_figure(
                sim_log_b,
                idx_b[entry_key],
                margin_override=cmp_margin_override_b,
            )
            st.plotly_chart(fig_ab, use_container_width=True, key="cmp_anc_b")

        # ---- Prefetch adjacent entries for both experiments ----
        for _delta in (-1, +1):
            _ni = cmp_i + _delta
            if 0 <= _ni < total_cmp:
                _nkey = cmp_df.iloc[_ni]["key"]
                _get_heatmap_figure(
                    sim_log_a,
                    idx_a[_nkey],
                    margin_override=cmp_margin_override_a,
                )
                _get_heatmap_figure(
                    sim_log_b,
                    idx_b[_nkey],
                    margin_override=cmp_margin_override_b,
                )
                _get_anchor_figure(
                    sim_log_a,
                    idx_a[_nkey],
                    margin_override=cmp_margin_override_a,
                )
                _get_anchor_figure(
                    sim_log_b,
                    idx_b[_nkey],
                    margin_override=cmp_margin_override_b,
                )


if __name__ == "__main__":
    main()
