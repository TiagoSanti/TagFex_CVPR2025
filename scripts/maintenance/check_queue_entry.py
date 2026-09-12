#!/usr/bin/env python3
"""Read-only queue resumption check: 0 complete, 1 incomplete, 2 invalid/error."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from utils.configuration import queue_training_config
from utils.experiment_paths import completed_run, experiment_log_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_spec", help="Comma-separated YAMLs, relative to repository root")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    try:
        paths = [ROOT / p.strip() for p in args.config_spec.split(",")]
        config = queue_training_config(paths, args.seed)
        # Resolve against root only AFTER computing the historical identity.
        if completed_run(config, root=ROOT) is not None:
            return 0
        base = experiment_log_dir(config)
        if base is not None:
            base = ROOT / base
            if base.exists() or any(base.parent.glob(base.name + '_v*')):
                parser.exit(2, 'Existing run is incomplete, incompatible or lacks proven provenance; manual review required.\n')
        return 1
    except (OSError, ValueError, TypeError, KeyError, IndexError, yaml.YAMLError) as exc:
        parser.exit(2, f"Cannot check queue entry: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
