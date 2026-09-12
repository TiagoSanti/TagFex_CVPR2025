#!/usr/bin/env python3
"""Read a path from a host overlay; never create or probe the target directory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from utils.configuration import load_yaml


def profile_path(profile, key="log_dir", parent=False):
    config = load_yaml(ROOT / profile)
    allowed = {"log_dir", "dataset_root", "ckpt_dir"}
    if not isinstance(config, dict) or not set(config) <= allowed:
        raise ValueError("Host overlays must contain only operational paths")
    value = config[key]
    if not isinstance(value, str) or not value or any(c in value for c in "\n\r\0"):
        raise ValueError(f"Invalid path for {key}")
    # Do not resolve symlinks: a Wolverine path is inspected on Xavier too.
    path = ROOT / Path(value).expanduser()
    return path.parent if parent else path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--key", choices=("log_dir", "dataset_root", "ckpt_dir"), default="log_dir")
    parser.add_argument("--parent", action="store_true", help="Print the campaign root above its logs directory")
    args = parser.parse_args()
    try:
        print(profile_path(args.profile, args.key, args.parent))
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"Invalid host profile: {exc}\n")


if __name__ == "__main__":
    main()
