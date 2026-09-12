#!/usr/bin/env python3
"""Aggregate structured ANT study records into presentation-ready CSV files."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from studies.ant_mechanism.schema import validate_metric_record


def read_jsonl_gz(paths: list[Path], validate=False) -> pd.DataFrame:
    rows = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if validate:
                    validate_metric_record(row)
                row["run_dir"] = path.parent.name
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.root / "aggregated"
    output.mkdir(parents=True, exist_ok=True)

    geometry_paths = sorted(args.root.glob("**/geometry_metrics.jsonl.gz"))
    if not geometry_paths:
        raise SystemExit(f"no geometry_metrics.jsonl.gz below {args.root}")
    geometry = read_jsonl_gz(geometry_paths, validate=True)
    geometry.to_csv(output / "geometry_records.csv.gz", index=False, compression="gzip")

    numeric = geometry.select_dtypes(include="number").columns.tolist()
    keys = ["run_dir", "branch", "variant", "mode", "task", "epoch"]
    numeric = [name for name in numeric if name not in {"task", "epoch", "batch"}]
    summary = geometry.groupby(keys, dropna=False)[numeric].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}__{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "geometry_by_epoch.csv", index=False)

    for stream_name in (
        "training_objective",
        "parameter_gradients",
        "parameter_gradient_alignment",
        "parameter_updates",
        "total_parameter_gradients",
        "evaluations",
    ):
        paths = sorted(args.root.glob(f"**/{stream_name}.jsonl.gz"))
        if paths:
            read_jsonl_gz(paths).to_csv(
                output / f"{stream_name}.csv.gz", index=False, compression="gzip"
            )

    print(f"runs={geometry['run_dir'].nunique()} records={len(geometry)} output={output}")


if __name__ == "__main__":
    main()
