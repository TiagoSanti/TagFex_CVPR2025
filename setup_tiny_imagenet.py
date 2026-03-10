#!/usr/bin/env python3
"""
Download and prepare the Tiny ImageNet dataset for use with TagFex.

Usage:
    python setup_tiny_imagenet.py [--dest ~/data/datasets/tiny-imagenet-200]

After running, the directory will be structured so the TinyImageNet dataset
class can read it directly without any further pre-processing.
"""

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


DOWNLOAD_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_DEST = os.path.expanduser("~/data/datasets/tiny-imagenet-200")


def _reporthook(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    pct = min(pct, 100)
    bar = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct:3d}%", end="", flush=True)


def download_and_extract(dest: str):
    dest = Path(dest)
    zip_path = dest.parent / "tiny-imagenet-200.zip"

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet from {DOWNLOAD_URL} ...")
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path, _reporthook)
        print()
    else:
        print(f"Found existing archive at {zip_path}, skipping download.")

    if not dest.exists():
        print(f"Extracting to {dest.parent} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest.parent)
        print("Extraction complete.")
    else:
        print(f"Destination {dest} already exists, skipping extraction.")


def verify(dest: str):
    dest = Path(dest)
    train_dir = dest / "train"
    val_ann = dest / "val" / "val_annotations.txt"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_ann.is_file():
        raise FileNotFoundError(f"Missing val annotations: {val_ann}")

    classes = sorted(os.listdir(train_dir))
    print(f"\nVerification:")
    print(f"  Classes found : {len(classes)} (expected 200)")
    total_train = sum(
        len(os.listdir(train_dir / c / "images")) for c in classes
    )
    print(f"  Train images  : {total_train} (expected 100 000)")
    with open(val_ann) as f:
        val_count = sum(1 for _ in f)
    print(f"  Val images    : {val_count} (expected 10 000)")

    if len(classes) != 200:
        print("WARNING: Expected 200 classes, got", len(classes))
    if total_train != 100_000:
        print("WARNING: Expected 100 000 training images, got", total_train)
    if val_count != 10_000:
        print("WARNING: Expected 10 000 val images, got", val_count)

    print("\nDataset ready at:", dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help="Destination directory (default: %(default)s)",
    )
    args = parser.parse_args()

    download_and_extract(args.dest)
    verify(args.dest)


if __name__ == "__main__":
    main()
