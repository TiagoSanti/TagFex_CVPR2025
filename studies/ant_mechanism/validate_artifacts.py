#!/usr/bin/env python3
"""Integrity checks for a completed ANT study output directory."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from studies.ant_mechanism.schema import ALL_ANT_VARIANTS, validate_metric_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expect-runs", type=int, default=9)
    parser.add_argument("--require-kd", action="store_true")
    parser.add_argument("--require-complete-contract", action="store_true")
    args = parser.parse_args()

    manifests = sorted(args.root.glob("**/manifest.json"))
    if len(manifests) != args.expect_runs:
        raise SystemExit(f"expected {args.expect_runs} runs, found {len(manifests)}")

    expected_variants = {variant.name for variant in ALL_ANT_VARIANTS}
    total_records = 0
    total_bytes = 0
    failures = []
    required_streams = {
        "training_objective.jsonl.gz",
        "parameter_gradients.jsonl.gz",
        "parameter_gradient_alignment.jsonl.gz",
        "total_parameter_gradients.jsonl.gz",
        "parameter_updates.jsonl.gz",
        "evaluations.jsonl.gz",
    }
    required_snapshot_keys = {
        "cos_sim",
        "nce_grad",
        "nce_probabilities",
        "nce_positive_mask",
        "nce_negative_mask",
        "nce_loss_per_anchor",
        "meta__absolute_index",
        "meta__continual_target",
    }
    for manifest in manifests:
        run_dir = manifest.parent
        geometry_path = run_dir / "geometry_metrics.jsonl.gz"
        if not geometry_path.is_file():
            failures.append(f"{run_dir.name}: geometry metrics missing")
            continue
        with gzip.open(geometry_path, "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        for record in records:
            validate_metric_record(record)
        total_records += len(records)
        branches = {record["branch"] for record in records}
        variants = {record["variant"] for record in records if record["mode"] == "shadow"}
        actual_count = sum(record["mode"] == "actual" for record in records)
        if "current" not in branches:
            failures.append(f"{run_dir.name}: current branch missing")
        if args.require_kd and "kd" not in branches:
            failures.append(f"{run_dir.name}: KD branch missing")
        observed_variants = variants | {r["variant"] for r in records}
        if not expected_variants.issubset(observed_variants):
            missing = sorted(expected_variants - observed_variants)
            failures.append(f"{run_dir.name}: shadow variants missing: {missing}")
        if actual_count == 0:
            failures.append(f"{run_dir.name}: no actual-condition records")
        snapshots = list((run_dir / "snapshots").glob("*.npz"))
        if not snapshots:
            failures.append(f"{run_dir.name}: snapshots missing")
        if args.require_complete_contract:
            if not (run_dir / "dataset_metadata.json").is_file():
                failures.append(f"{run_dir.name}: dataset metadata missing")
            for stream_name in sorted(required_streams):
                if not (run_dir / stream_name).is_file():
                    failures.append(f"{run_dir.name}: {stream_name} missing")
            current_snapshots = [path for path in snapshots if path.stem.endswith("_current")]
            kd_snapshots = [path for path in snapshots if path.stem.endswith("_kd")]
            if not current_snapshots:
                failures.append(f"{run_dir.name}: current snapshot missing")
            else:
                with np.load(current_snapshots[0]) as snapshot:
                    missing = required_snapshot_keys.difference(snapshot.files)
                    missing.update(
                        key for key in ("payload__view1", "payload__view2", "payload__embedding")
                        if key not in snapshot.files
                    )
                if missing:
                    failures.append(
                        f"{run_dir.name}: current snapshot keys missing: {sorted(missing)}"
                    )
            if args.require_kd:
                if not kd_snapshots:
                    failures.append(f"{run_dir.name}: KD snapshot missing")
                else:
                    with np.load(kd_snapshots[0]) as snapshot:
                        missing = {
                            "payload__student_prediction",
                            "payload__teacher_projection",
                        }.difference(snapshot.files)
                    if missing:
                        failures.append(
                            f"{run_dir.name}: KD snapshot keys missing: {sorted(missing)}"
                        )
        total_bytes += sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"OK runs={len(manifests)} records={total_records} "
        f"size_mib={total_bytes / 1024**2:.2f}"
    )


if __name__ == "__main__":
    main()
