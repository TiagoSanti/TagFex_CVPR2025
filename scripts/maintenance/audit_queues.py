#!/usr/bin/env python3
"""Inspect queue/config dependencies without importing training or writing outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.configuration import load_configs, queue_training_config
from utils.experiment_paths import experiment_log_dir, expected_task_count


def audit_queue(queue: Path, root: Path = ROOT) -> dict:
    queue = queue if queue.is_absolute() else root / queue
    rows = []
    for number, line in enumerate(queue.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 3 or not all(field.strip() for field in fields):
            raise ValueError(f"{queue}:{number}: expected configs|description|seed")
        spec, description, seed = (field.strip() for field in fields)
        seed = int(seed)
        dependencies = []
        paths = []
        for name in spec.split(","):
            name = name.strip()
            if not name:
                raise ValueError(f"{queue}:{number}: empty config path")
            path = root / name
            paths.append(path)
            data = path.read_bytes()
            loaded = yaml.safe_load(data)
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}: expected YAML mapping")
            dependencies.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
        config = {"seed": int(seed)}
        config.update(load_configs(paths))
        train_config = queue_training_config(paths, seed)
        log_dir = experiment_log_dir(train_config)
        rows.append({"line": number, "description": description, "queue_seed": int(seed),
                     "configs": dependencies, "effective_config": config,
                     "experiment_log_dir": str(log_dir) if log_dir is not None else None,
                     "expected_tasks": expected_task_count(train_config),
                     "training_paths": {k: str(train_config[k]) if train_config[k] is not None else None
                                        for k in ("dataset_root", "log_dir", "ckpt_dir")},
                     "configured_paths": {k: v for k, v in config.items()
                                          if k.endswith(("_dir", "_root", "_path"))}})
    return {"queue": str(queue.relative_to(root)) if queue.is_relative_to(root) else str(queue),
            "count": len(rows), "entries": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, action="append", help="Relative to repository root; repeatable")
    parser.add_argument("--json", action="store_true", help="Print full effective configurations to stdout")
    args = parser.parse_args()
    queues = args.queue or sorted(ROOT.glob("experiments/*/queues/*.txt"))
    try:
        reports = [audit_queue(queue) for queue in queues]
    except (ValueError, TypeError, IndexError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"{exc}\n")
    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for report in reports:
            print(f"{report['count']:3d} entries  {report['queue']}")
        print(f"{sum(r['count'] for r in reports)} entries in {len(reports)} queues")
        print("Experiment identities are computed without allocating versions or checking logs, hosts or datasets.")


if __name__ == "__main__":
    main()
