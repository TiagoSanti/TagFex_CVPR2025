#!/usr/bin/env python3
"""Reproducible Friedman/Nemenyi analysis over current TagFex logs.

Legacy logs remain valid inputs.  The analysis records every discovered
candidate, the deterministic selection decision, hashes of all selected logs,
the exact observation matrix, and hashes of every generated artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache/tagfex/matplotlib"))

from utils.research_identity import guard_architecture_groups

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, create_report, latex_table, plot_stats

from utils.provenance import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    artifact_record,
    atomic_write_json,
    canonical_hash,
    file_record,
    git_record,
    records_hash,
    utc_now,
)

warnings.filterwarnings("ignore")


CONFIG_NAME = {
    "antB0_nceA1_baselineInfoNCE": "Baseline InfoNCE",
    "antB0_nceA1_baselineInfoNCE_avgK3": "Baseline InfoNCE + TAvg-3",
    "antB0_nceA1_baselineInfoNCE_avgK5": "Baseline InfoNCE + TAvg-5",
    "antB0_nceA1_baselineInfoNCE_avgK10": "Baseline InfoNCE + TAvg-10",
    "antB0.5_nceA1_antM0.5_antGlobal": "ANT-IV-GR [legado: ref. conectada]",
    "antB0.5_nceA1_antM0.5_antGlobal_refDetached": "ANT-IV-GR",
    "antB0.5_nceA1_antM0.5_antLocal": "ANT-IV-AR [legado: ref. conectada]",
    "antB0.5_nceA1_antM0.5_antLocal_refDetached": "ANT-IV-AR",
    "antB0.5_nceA1_antM0.5_antLocal_avgK3": "ANT-IV-AR [legado: ref. conectada] + TAvg-3",
    "antB0.5_nceA1_antM0.5_antLocal_avgK5": "ANT-IV-AR [legado: ref. conectada] + TAvg-5",
    "antB0.5_nceA1_antM0.5_antLocal_avgK10": "ANT-IV-AR [legado: ref. conectada] + TAvg-10",
    "antB0.5_nceA1_antM0.5_antSymmetricFull": "ANT-FS-AR [legado: ref. conectada]",
    "antB0.5_nceA1_antM0.5_antSymmetricFullGlobal": "ANT-FS-GR [legado: ref. conectada]",
    "antB0.5_nceA1_antM0.5_antSymmetricFullGlobal_refDetached": "ANT-FS-GR",
    "antB0.5_nceA1_antM0.5_antSymmetricFull_refDetached": "ANT-FS-AR",
    "antB0.5_nceA1_antM0.5_antSymmetricFull_avgK3": "ANT-FS-AR [legado: ref. conectada] + TAvg-3",
    "antB0.5_nceA1_antM0.5_antSymmetricFull_avgK5": "ANT-FS-AR [legado: ref. conectada] + TAvg-5",
    "antB0.5_nceA1_antM0.5_antSymmetricFull_avgK10": "ANT-FS-AR [legado: ref. conectada] + TAvg-10",
}

DATASET_NAME = {
    "cifar100_10-10": "C100 10-10",
    "cifar100_50-10": "C100 50-10",
    "tiny_imagenet_20-20": "TIN 20-20",
    "tiny_imagenet_100-20": "TIN 100-20",
}
EXPECTED_TASKS = {
    "cifar100_10-10": 10,
    "cifar100_50-10": 6,
    "tiny_imagenet_20-20": 10,
    "tiny_imagenet_100-20": 6,
}
DATASETS_ORDER = list(DATASET_NAME)
SEEDS = ("s1993", "s1994", "s1995")
ALPHA = 0.05

DIR_RE = re.compile(
    r"^(?P<debug>debug_)?exp_"
    r"(?P<dataset>cifar100_\d+-\d+|tiny_imagenet_\d+-\d+)_"
    r"(?P<config>.+?)_(?P<seed>s\d+)(?:_v\d+)?$"
)
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass
class Candidate:
    directory: str
    log_path: str
    dataset: str
    raw_config: str
    canonical_config: str
    seed: str
    avg_acc1: float | None
    task_count: int
    expected_task_count: int
    complete: bool
    eligible_method: bool
    source_priority: int
    debug_priority: int
    sha256: str
    size_bytes: int
    selected: bool = False
    exclusion_reason: str = ""


def _parse_final_metrics(log_path: Path) -> tuple[float | None, int]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, 0
    avg_matches = re.findall(rf"\bavg_acc1 ({NUMBER_RE})", text)
    curve_matches = re.findall(r"\bacc1_curve \[([^\]]+)\]", text)
    if not avg_matches or not curve_matches:
        return None, 0
    values = re.findall(NUMBER_RE, curve_matches[-1])
    return float(avg_matches[-1]), len(values)


def _canonical_config(config: str) -> str:
    config = re.sub(r"_nce(?:Global|Local)", "", config)
    if re.search(r"^antB0(?:_|$)", config):
        config = re.sub(
            r"_ant(?:Global|Local|SymmetricFull)",
            "_baselineInfoNCE",
            config,
        )
    return config


def _source_priority(config: str) -> int:
    if "_nceLocal" in config:
        return 2
    if "_nceGlobal" in config:
        return 1
    return 0


def discover_candidates(logs_dir: Path, repo: Path) -> list[Candidate]:
    candidates = []
    architecture_groups = []
    for directory in sorted(logs_dir.iterdir(), key=lambda path: path.name):
        if not directory.is_dir():
            continue
        match = DIR_RE.match(directory.name)
        if not match:
            continue
        log_path = directory / "exp_gistlog.log"
        if not log_path.is_file():
            continue
        dataset = match.group("dataset")
        config = match.group("config")
        canonical = _canonical_config(config)
        architecture_groups.append(((dataset, canonical), log_path))
        avg_acc1, task_count = _parse_final_metrics(log_path)
        expected = EXPECTED_TASKS[dataset]
        record = file_record(log_path, root=repo)
        candidates.append(
            Candidate(
                directory=directory.name,
                log_path=record["path"],
                dataset=dataset,
                raw_config=config,
                canonical_config=canonical,
                seed=match.group("seed"),
                avg_acc1=avg_acc1,
                task_count=task_count,
                expected_task_count=expected,
                complete=avg_acc1 is not None and task_count == expected,
                eligible_method=canonical in CONFIG_NAME,
                source_priority=_source_priority(config),
                debug_priority=1 if match.group("debug") else 0,
                sha256=record["sha256"],
                size_bytes=record["size_bytes"],
            )
        )
    guard_architecture_groups(architecture_groups)
    return candidates


def select_candidates(candidates: list[Candidate]) -> dict[tuple[str, str, str], Candidate]:
    grouped: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.eligible_method:
            candidate.exclusion_reason = "method_not_in_analysis"
            continue
        if not candidate.complete:
            candidate.exclusion_reason = "incomplete_or_unparseable"
            continue
        key = (candidate.dataset, candidate.canonical_config, candidate.seed)
        grouped[key].append(candidate)

    selected = {}
    for key, values in sorted(grouped.items()):
        ordered = sorted(
            values,
            key=lambda item: (
                item.source_priority,
                item.debug_priority,
                item.directory,
            ),
        )
        winner = ordered[0]
        winner.selected = True
        selected[key] = winner
        for candidate in ordered[1:]:
            candidate.exclusion_reason = "lower_priority_equivalent_run"
    return selected


def complete_configurations(
    selected: dict[tuple[str, str, str], Candidate],
) -> list[str]:
    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    for dataset, config, seed in selected:
        coverage[(dataset, config)].add(seed)
    configs = [
        config
        for config in CONFIG_NAME
        if all(
            coverage[(dataset, config)] >= set(SEEDS)
            for dataset in DATASETS_ORDER
        )
    ]
    return sorted(configs, key=lambda config: CONFIG_NAME[config])


def observation_frame(
    selected: dict[tuple[str, str, str], Candidate], configs: list[str]
) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS_ORDER:
        for seed in SEEDS:
            row = {"observation": f"{DATASET_NAME[dataset]} {seed}"}
            for config in configs:
                row[CONFIG_NAME[config]] = selected[(dataset, config, seed)].avg_acc1
            rows.append(row)
    return pd.DataFrame(rows).set_index("observation")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame, *, index: bool) -> None:
    _atomic_write_text(path, frame.to_csv(index=index, lineterminator="\n"))


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("autorank", "numpy", "pandas", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_analysis(
    logs_dir: Path,
    output_dir: Path,
    repo: Path,
    *,
    require_run_manifests: bool = False,
) -> dict:
    started_at = utc_now()
    candidates = discover_candidates(logs_dir, repo)
    selected = select_candidates(candidates)
    configs = complete_configurations(selected)
    if len(configs) < 2:
        raise RuntimeError(
            f"Only {len(configs)} method(s) have complete 4-dataset × 3-seed coverage"
        )
    for candidate in candidates:
        if candidate.selected and candidate.canonical_config not in configs:
            candidate.selected = False
            candidate.exclusion_reason = "incomplete_cross_dataset_seed_coverage"
    analysis_candidates = [
        candidate
        for candidate in candidates
        if candidate.selected and candidate.canonical_config in configs
    ]
    missing_parent_manifests = []
    for candidate in analysis_candidates:
        log_path = Path(candidate.log_path)
        if not log_path.is_absolute():
            log_path = repo / log_path
        provenance_dir = log_path.parent / "provenance"
        if not provenance_dir.is_dir() or not any(provenance_dir.glob("run-*.json")):
            missing_parent_manifests.append(candidate.log_path)
    if require_run_manifests and missing_parent_manifests:
        raise RuntimeError(
            "Strict provenance rejected "
            f"{len(missing_parent_manifests)} selected logs without run manifests"
        )
    frame = observation_frame(selected, configs)
    result = autorank(frame, alpha=ALPHA, verbose=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.csv"
    selection_path = output_dir / "selection.csv"
    report_path = output_dir / "report.txt"
    latex_path = output_dir / "table.tex"
    cd_path = output_dir / "cd_diagram.png"
    cd_pdf_path = output_dir / "cd_diagram.pdf"
    heat_path = output_dir / "heatmap.png"

    _write_csv(observations_path, frame, index=True)
    selection_frame = pd.DataFrame(asdict(candidate) for candidate in candidates)
    _write_csv(selection_path, selection_frame, index=False)

    report_buffer = io.StringIO()
    report_buffer.write(f"autorank statistical report — avg_acc1, α={ALPHA}\n")
    report_buffer.write(
        f"Observations: {frame.shape[0]}  "
        f"({len(DATASETS_ORDER)} datasets × {len(SEEDS)} seeds)\n"
    )
    report_buffer.write(f"Methods: {frame.shape[1]}\n")
    report_buffer.write("Selection: canonical > nGlobal > nLocal; full-name > debug-name\n\n")
    report_buffer.write(str(result) + "\n\n")
    with contextlib.redirect_stdout(report_buffer):
        create_report(result)
    _atomic_write_text(report_path, report_buffer.getvalue())

    latex_buffer = io.StringIO()
    with contextlib.redirect_stdout(latex_buffer):
        latex_table(result)
    _atomic_write_text(latex_path, latex_buffer.getvalue())

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_stats(result, ax=ax, allow_insignificant=True)
    ax.set_title(
        f"Diagrama de diferença crítica — AvgAcc\n"
        f"{frame.shape[1]} métodos × {frame.shape[0]} blocos pareados | α={ALPHA}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(cd_path, dpi=300, bbox_inches="tight")
    fig.savefig(cd_pdf_path, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(max(8, len(configs) * 0.9), 6))
    ranked_cols = result.rankdf.sort_values("meanrank").index.tolist()
    heat_data = frame[ranked_cols]
    col_norm = (heat_data - heat_data.min()) / (
        heat_data.max() - heat_data.min() + 1e-9
    )
    image = ax2.imshow(
        col_norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    for row_index in range(len(heat_data)):
        for column_index in range(len(ranked_cols)):
            value = heat_data.values[row_index, column_index]
            normalized = col_norm.values[row_index, column_index]
            color = "white" if normalized < 0.25 or normalized > 0.85 else "black"
            ax2.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )
    ax2.set_xticks(range(len(ranked_cols)))
    ax2.set_yticks(range(len(heat_data)))
    ax2.set_xticklabels(ranked_cols, fontsize=8, rotation=40, ha="right")
    ax2.set_yticklabels(heat_data.index, fontsize=7)
    ax2.set_title(
        "avg_acc1 per paired block (sorted by mean rank; column-normalised)",
        fontsize=10,
    )
    plt.colorbar(image, ax=ax2, fraction=0.02, pad=0.01, label="normalised acc")
    fig2.tight_layout()
    fig2.savefig(heat_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    selected_candidates = sorted(
        (candidate for candidate in candidates if candidate.selected),
        key=lambda item: (item.dataset, item.canonical_config, item.seed),
    )
    selected_file_records = [
        {
            "path": candidate.log_path,
            "size_bytes": candidate.size_bytes,
            "sha256": candidate.sha256,
        }
        for candidate in selected_candidates
        if candidate.canonical_config in configs
    ]
    parent_manifest_paths = []
    for candidate in selected_candidates:
        if candidate.canonical_config not in configs:
            continue
        log_path = Path(candidate.log_path)
        if not log_path.is_absolute():
            log_path = repo / log_path
        provenance_dir = log_path.parent / "provenance"
        if provenance_dir.is_dir():
            parent_manifest_paths.extend(provenance_dir.glob("run-*.json"))
    parent_manifest_records = [
        file_record(path, root=repo)
        for path in sorted(set(parent_manifest_paths), key=lambda item: str(item))
    ]
    selection_parameters = {
        "metric": "final avg_acc1",
        "alpha": ALPHA,
        "datasets": DATASETS_ORDER,
        "seeds": list(SEEDS),
        "expected_tasks": EXPECTED_TASKS,
        "configuration_names": {
            config: CONFIG_NAME[config] for config in configs
        },
        "equivalence": "remove obsolete nceGlobal/nceLocal; erase inactive ANT mode at beta=0",
        "ant_report_identity": (
            "refDetached is canonical ANT; connected-reference ANT is legacy; "
            "underlying experiment directory names remain unchanged"
        ),
        "priority": ["canonical", "nceGlobal", "nceLocal", "non-debug name", "lexical path"],
        "coverage_requirement": "all 4 datasets and all 3 seeds",
        "require_run_manifests": require_run_manifests,
        "selected_logs_without_run_manifests": missing_parent_manifests,
    }
    source_files = [
        file_record(Path(__file__).resolve(), root=repo),
        *[file_record((repo / "utils" / name).resolve(), root=repo)
          for name in ("provenance.py", "research_identity.py", "argument.py")],
    ]
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "statistical_analysis",
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd().resolve()),
        "source": {
            "git": git_record(repo),
            "files": source_files,
            "source_hash": records_hash(source_files),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "packages": _package_versions(),
        },
        "inputs": {
            "logs_dir": str(logs_dir.resolve()),
            "discovered_candidate_count": len(candidates),
            "selected_file_count": len(selected_file_records),
            "selected_files": selected_file_records,
            "selected_files_hash": records_hash(selected_file_records),
            "parent_run_manifests": parent_manifest_records,
            "parent_run_manifests_hash": records_hash(parent_manifest_records),
            "selection_parameters": selection_parameters,
            "selection_parameters_hash": canonical_hash(selection_parameters),
            "analysis_input_hash": canonical_hash(
                {
                    "files": selected_file_records,
                    "parent_run_manifests": parent_manifest_records,
                    "parameters": selection_parameters,
                }
            ),
        },
        "result": {
            "observation_count": int(frame.shape[0]),
            "method_count": int(frame.shape[1]),
            "methods": list(frame.columns),
            "omnibus": str(result.omnibus),
            "posthoc": str(result.posthoc),
            "pvalue": float(result.pvalue),
            "critical_difference": float(result.cd),
            "observation_matrix_hash": canonical_hash(
                {
                    "index": list(frame.index),
                    "columns": list(frame.columns),
                    "values": frame.values.tolist(),
                }
            ),
        },
    }
    manifest["artifacts"] = artifact_record(output_dir)
    atomic_write_json(output_dir / "provenance.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an auditable Friedman/Nemenyi analysis over TagFex logs."
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/statistics/current"),
    )
    parser.add_argument(
        "--require-run-manifests",
        action="store_true",
        help="reject selected logs that lack an in-run provenance manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = ROOT
    manifest = run_analysis(
        args.logs_dir.resolve(),
        args.output_dir.resolve(),
        repo,
        require_run_manifests=args.require_run_manifests,
    )
    result = manifest["result"]
    print(
        f"Completed: {result['method_count']} methods × "
        f"{result['observation_count']} paired blocks"
    )
    print(f"p={result['pvalue']:.12g}; CD={result['critical_difference']:.12g}")
    print(f"Input hash: {manifest['inputs']['analysis_input_hash']}")
    print(f"Artifacts: {args.output_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
