#!/usr/bin/env python
"""
Generate synthetic TagFex debug logs for Similarity Viewer demonstration.

Simulates 3 tasks of CIFAR-100 10-10 with realistic learning dynamics:
  T1 — init task (contrast only, 20 epochs)
  T2 — incremental (contrast + kd, 17 epochs)
  T3 — incremental (contrast + kd, 17 epochs)

Outputs (run from project root):
  logs/demo_synthetic/exp_debug0.log
  logs/demo_synthetic/similarity_debug.log

Usage:
    python analysis/scripts/generate_synthetic_debug.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEMO_DIR = Path("logs/demo_synthetic")

NUM_TASKS = 3
TASK_EPOCHS = {1: 20, 2: 17, 3: 17}  # abbreviated (real: 200 / 170)
BATCHES_PER_EPOCH = 5  # abbreviated for compact demo log
BS = 64  # samples per GPU  (128 // 2 GPUs)
N = BS * 2  # cosine-sim matrix dimension (128 × 128)
TEMP = 0.2  # InfoNCE temperature
NCE_ALPHA = 1.0
ANT_BETA = 0.5
ANT_MARGIN = 0.5
NUM_CLASSES_PER_TASK = 10
EMBED_DIM = 128

_T0 = datetime(2026, 3, 12, 8, 0, 0)
_BATCH_SECONDS = 2


def _ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Embedding / similarity simulation
# ---------------------------------------------------------------------------


def _make_cosim(
    task: int, epoch: int, max_epoch: int, batch: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a [N×N] cosine-sim matrix reflecting training progress for contrast loss."""
    progress = epoch / max_epoch
    num_seen = NUM_CLASSES_PER_TASK * task

    rng_proto = np.random.default_rng(task * 10_007 + 17)
    prototypes = rng_proto.normal(0, 1, (num_seen, EMBED_DIM))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8

    # signal increases, noise decreases with training progress
    signal = 0.15 + 0.70 * progress
    noise = 0.50 * (1.0 - progress) + 0.08

    task_cls = np.arange((task - 1) * NUM_CLASSES_PER_TASK, task * NUM_CLASSES_PER_TASK)
    labels = rng.choice(task_cls, size=BS, replace=True)

    embeds = np.empty((N, EMBED_DIM))
    for i in range(BS):
        cls = labels[i]
        v1 = prototypes[cls] * signal + rng.normal(0, noise, EMBED_DIM)
        v1 /= np.linalg.norm(v1) + 1e-8
        embeds[i] = v1
        v2 = prototypes[cls] * signal + rng.normal(0, noise * 0.85, EMBED_DIM)
        v2 /= np.linalg.norm(v2) + 1e-8
        embeds[BS + i] = v2

    return (embeds @ embeds.T).astype(np.float32)


def _make_kd_cosim(
    task: int, epoch: int, max_epoch: int, batch: int, rng: np.random.Generator
) -> np.ndarray:
    """KD cosim: current vs old-model features — lower pos-sim early, converges."""
    progress = epoch / max_epoch
    num_seen = NUM_CLASSES_PER_TASK * task

    rng_proto = np.random.default_rng(task * 10_007 + 17)
    prototypes = rng_proto.normal(0, 1, (num_seen, EMBED_DIM))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8

    # Old-model signal remains fixed; new-model signal grows → alignment improves
    signal_new = 0.10 + 0.55 * progress
    signal_old = 0.30
    noise = 0.45 * (1.0 - progress) + 0.12

    task_cls = np.arange((task - 1) * NUM_CLASSES_PER_TASK, task * NUM_CLASSES_PER_TASK)
    labels = rng.choice(task_cls, size=BS, replace=True)

    embeds = np.empty((N, EMBED_DIM))
    for i in range(BS):
        cls = labels[i]
        # "predicted" = current model
        v1 = prototypes[cls] * signal_new + rng.normal(0, noise, EMBED_DIM)
        v1 /= np.linalg.norm(v1) + 1e-8
        embeds[i] = v1
        # "old" = frozen old model (less noise, closer to prototype)
        v2 = prototypes[cls] * signal_old + rng.normal(0, noise * 0.6, EMBED_DIM)
        v2 /= np.linalg.norm(v2) + 1e-8
        embeds[BS + i] = v2

    return (embeds @ embeds.T).astype(np.float32)


def _compute_stats(cos_sim: np.ndarray, prefix: str = "") -> dict:
    """Compute per-anchor and overall statistics from a cosine-similarity matrix."""
    bs = cos_sim.shape[0] // 2
    pos = np.array([cos_sim[i, bs + i] for i in range(bs)])

    q1 = cos_sim[:bs, :bs].copy()
    np.fill_diagonal(q1, -np.inf)
    local_maxs = q1.max(axis=1)
    thresholds = local_maxs - ANT_MARGIN

    neg_flat = q1[q1 != -np.inf].flatten()
    per_gap = pos - local_maxs

    viol, total = 0, 0
    for i in range(bs):
        for j in range(bs):
            if j == i:
                continue
            total += 1
            if cos_sim[i, j] >= thresholds[i]:
                viol += 1
    viol_pct = 100.0 * viol / max(1, total)

    # ANT loss: E[relu(local_max - pos + margin)]
    ant_raw = float(np.mean(np.maximum(0.0, local_maxs - pos + ANT_MARGIN)))

    # InfoNCE NLL (numerically stable log-softmax)
    nll_acc = 0.0
    for i in range(bs):
        row = (cos_sim[i] / TEMP).astype(np.float64)
        row[i] = -1e9  # mask self
        c = row.max()
        log_denom = c + np.log(np.sum(np.exp(row - c)))
        nll_acc += -(cos_sim[i, bs + i] / TEMP) + log_denom
    nll = float(nll_acc / bs)

    nce_w = NCE_ALPHA * nll
    ant_w = ANT_BETA * ant_raw
    total_loss = nce_w + ant_w

    return {
        "contrastive": {
            "pos_mean": float(pos.mean()),
            "pos_std": float(pos.std()),
            "neg_mean": float(neg_flat.mean()),
            "neg_std": float(neg_flat.std()),
            "gap": float(pos.mean() - neg_flat.mean()),
        },
        "ant": {
            "pos_min": float(pos.min()),
            "pos_max": float(pos.max()),
            "neg_min": float(neg_flat.min()),
            "neg_max": float(neg_flat.max()),
            "gap_mean": float(per_gap.mean()),
            "gap_std": float(per_gap.std()),
            "gap_min": float(per_gap.min()),
            "gap_max": float(per_gap.max()),
            "margin": ANT_MARGIN,
            "violation_pct": viol_pct,
            "ant_loss": ant_raw,
        },
        "loss": {
            "nll": nll,
            "ant_loss": ant_raw,
            "nce_weighted": nce_w,
            "ant_weighted": ant_w,
            "total": total_loss,
        },
        "local_maxs": local_maxs,
        "thresholds": thresholds,
        "pos": pos,
    }


# ---------------------------------------------------------------------------
# exp_debug0.log writer
# ---------------------------------------------------------------------------


def write_main_log(out: Path) -> None:
    """Write exp_debug0.log with [T E B] structured stats for every batch."""
    rng = np.random.default_rng(2025)
    t = _T0

    with out.open("w") as f:

        def line(msg: str) -> None:
            f.write(f"{_ts(t)} | {msg}\n")

        for task in range(1, NUM_TASKS + 1):
            max_epoch = TASK_EPOCHS[task]
            loss_types = ["contrast"] if task == 1 else ["contrast", "kd"]

            for epoch in range(1, max_epoch + 1):
                for batch in range(1, BATCHES_PER_EPOCH + 1):
                    context = f"[T{task} E{epoch} B{batch}]"

                    for ltype in loss_types:
                        cos_sim = (
                            _make_cosim(task, epoch, max_epoch, batch, rng)
                            if ltype == "contrast"
                            else _make_kd_cosim(task, epoch, max_epoch, batch, rng)
                        )
                        s = _compute_stats(cos_sim)

                        c = s["contrastive"]
                        line(
                            f"{context} Contrastive stats: "
                            f"pos_mean: {c['pos_mean']:.4f} | pos_std: {c['pos_std']:.4f} | "
                            f"neg_mean: {c['neg_mean']:.4f} | neg_std: {c['neg_std']:.4f} | "
                            f"current_gap: {c['gap']:.4f}"
                        )

                        a = s["ant"]
                        line(
                            f"{context} ANT distance stats: "
                            f"pos_min: {a['pos_min']:.4f} | pos_max: {a['pos_max']:.4f} | "
                            f"neg_min: {a['neg_min']:.4f} | neg_max: {a['neg_max']:.4f} | "
                            f"gap_mean: {a['gap_mean']:.4f} | gap_std: {a['gap_std']:.4f} | "
                            f"gap_min: {a['gap_min']:.4f} | gap_max: {a['gap_max']:.4f} | "
                            f"margin: {a['margin']:.4f} | violation_pct: {a['violation_pct']:.4f}% | "
                            f"ant_loss: {a['ant_loss']:.4f}"
                        )

                        lv = s["loss"]
                        p = ltype
                        line(
                            f"{context} Loss components: "
                            f"{p}_nll: {lv['nll']:.4f} | {p}_ant_loss: {lv['ant_loss']:.4f} | "
                            f"{p}_nce_weighted: {lv['nce_weighted']:.4f} | "
                            f"{p}_ant_weighted: {lv['ant_weighted']:.4f} | "
                            f"{p}_total: {lv['total']:.4f}"
                        )

                    t += timedelta(seconds=_BATCH_SECONDS)

    print(f"  wrote {out}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# similarity_debug.log helpers and writer
# ---------------------------------------------------------------------------


def write_sim_log(out: Path) -> None:
    """Write similarity_debug.log with 1 entry per (epoch × loss_type)."""
    rng = np.random.default_rng(2025 + 1)
    t = _T0
    index: dict[str, int] = {}

    with out.open("wb") as fb:

        def L(msg: str = "") -> None:
            fb.write(f"{_ts(t)} | {msg}\n".encode())

        def offset() -> int:
            return fb.tell()

        for task in range(1, NUM_TASKS + 1):
            max_epoch = TASK_EPOCHS[task]
            loss_types = ["contrast"] if task == 1 else ["contrast", "kd"]

            for epoch in range(1, max_epoch + 1):
                # Only first batch per epoch (debug_similarity_batches_per_epoch: 1)
                batch = 1
                rng_entry = np.random.default_rng(task * 100_000 + epoch * 100 + batch)

                for ltype in loss_types:
                    cos_sim = (
                        _make_cosim(task, epoch, max_epoch, batch, rng_entry)
                        if ltype == "contrast"
                        else _make_kd_cosim(task, epoch, max_epoch, batch, rng_entry)
                    )
                    s = _compute_stats(cos_sim)
                    context = f"T{task}_E{epoch}_B{batch}_{ltype}"
                    sep = "=" * 80

                    L()
                    L(sep)

                    # Record index offset pointing directly at the header line so
                    # parse_debug_entry can seek here and find the header first.
                    key = f"{task}_{epoch}_{batch}_{ltype}"
                    index[key] = offset()
                    L(f"SIMILARITY MATRIX DEBUG - {context}")
                    L(f"Matrix shape: ({N}, {N})")
                    L(f"ANT margin: {ANT_MARGIN}, Max strategy: Local")
                    L(sep)
                    L()

                    _write_verbose_section(fb, cos_sim, s, context, _ts(t))
                    _write_matrix(fb, cos_sim, s, _ts(t))

                    t += timedelta(seconds=_BATCH_SECONDS * BATCHES_PER_EPOCH)

    # Write index cache
    cache = out.parent / f".{out.name}.index.json"
    cache.write_text(json.dumps(index))
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  wrote {out}  ({size_mb:.1f} MB, {len(index)} entries)")
    print(f"  wrote {cache}")


def _write_verbose_section(
    fb, cos_sim: np.ndarray, s: dict, context: str, ts: str
) -> None:
    def L(msg: str = "") -> None:
        fb.write(f"{ts} | {msg}\n".encode())

    local_maxs = s["local_maxs"]
    thresholds = s["thresholds"]
    pos = s["pos"]

    L(
        f"Local max per anchor: min={local_maxs.min():.4f}, "
        f"max={local_maxs.max():.4f}, mean={local_maxs.mean():.4f}"
    )
    L()

    for ai in range(min(8, BS)):
        L(f"--- Anchor {ai} ---")
        L(f"  Positive pair (idx {BS + ai}): {pos[ai]:.4f}")
        L(f"  Anchor max: {local_maxs[ai]:.4f}, Threshold: {thresholds[ai]:.4f}")

        above, below = [], []
        for j in range(BS):
            if j == ai:
                continue
            v = float(cos_sim[ai, j])
            (above if v >= thresholds[ai] else below).append((j, v))

        above.sort(key=lambda x: x[1], reverse=True)
        below.sort(key=lambda x: x[1], reverse=True)

        L(f"  Above threshold: {len(above)}, Below threshold: {len(below)}")
        if above:
            L("  Values ABOVE threshold:")
            for j, v in above[:5]:
                L(f"    idx {j}: {v:.4f} (gap: +{v - thresholds[ai]:.4f})")
        if below:
            L("  Top values BELOW threshold:")
            for j, v in below[:3]:
                L(f"    idx {j}: {v:.4f} (gap: -{thresholds[ai] - v:.4f})")
        L()

    a = s["ant"]
    L("--- Overall Statistics ---")
    L(
        f"Positive pairs: min={a['pos_min']:.4f}, max={a['pos_max']:.4f}, "
        f"mean={pos.mean():.4f}, std={pos.std():.4f}"
    )
    L(
        f"Negative pairs: min={a['neg_min']:.4f}, max={a['neg_max']:.4f}, "
        f"mean={s['contrastive']['neg_mean']:.4f}, std={s['contrastive']['neg_std']:.4f}"
    )
    L(f"Gap (pos_mean - neg_mean): {s['contrastive']['gap']:.4f}")
    total_neg = BS * (BS - 1)
    viol_cnt = round(a["violation_pct"] / 100 * total_neg)
    L(f"Margin violations: {viol_cnt} / {total_neg} ({a['violation_pct']:.2f}%)")
    L()


def _write_matrix(fb, cos_sim: np.ndarray, s: dict, ts: str) -> None:
    def L(msg: str = "") -> None:
        fb.write(f"{ts} | {msg}\n".encode())

    thresholds = s["thresholds"]
    sep = "=" * 80

    L()
    L(sep)
    L(f"COMPLETE SIMILARITY MATRIX ({N}x{N})")
    L(sep)
    L()

    header = "     |" + "".join(f"  {j:2d}  |" for j in range(N))
    L(header)
    L("-" * len(header))

    for i in range(N):
        row = f" {i:2d}  |"
        is_anc = i < BS
        athr = s["thresholds"][i] if is_anc else None
        for j in range(N):
            v = float(cos_sim[i, j])
            mk = ""
            if i == j:
                mk = "*"
            elif is_anc and j == i + BS:
                mk = "+"
            elif is_anc and j < BS and j != i and v >= athr:
                mk = "!"
            row += f" {v:5.2f}{mk}|"
        L(row)

    L()
    L("Legend:")
    L("  * = Self-similarity (diagonal, always 1.0)")
    L("  + = Positive pair (augmentation)")
    L("  ! = Margin violation (negative above threshold)")
    L()
    L(sep)
    L()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic debug logs in: {DEMO_DIR.resolve()}")

    main_log = DEMO_DIR / "exp_debug0.log"
    sim_log = DEMO_DIR / "similarity_debug.log"

    print("  [1/2] Writing exp_debug0.log …")
    write_main_log(main_log)

    print("  [2/2] Writing similarity_debug.log …")
    write_sim_log(sim_log)

    print()
    print("Done. Launch the Similarity Viewer with:")
    print("  streamlit run analysis/scripts/similarity_viewer.py")


if __name__ == "__main__":
    main()
