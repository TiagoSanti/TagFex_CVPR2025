#!/usr/bin/env python3
"""
Download and prepare the CUB-200-2011 dataset for use with TagFex.

Usage:
    python setup_cub200.py [--dest ~/data/datasets/CUB_200_2011]

After running, the directory will be structured so the CUB200 dataset
class can read it directly without any further pre-processing.

Expected layout::

    root/  (CUB_200_2011/)
    ├── images.txt
    ├── image_class_labels.txt
    ├── train_test_split.txt
    ├── classes.txt
    └── images/
        └── 001.Black_footed_Albatross/
            └── *.jpg
"""

import argparse
import os
import tarfile
import urllib.request
from pathlib import Path


DOWNLOAD_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
DEFAULT_DEST = os.path.expanduser("~/data/datasets/CUB_200_2011")

EXPECTED_CLASSES = 200
EXPECTED_TRAIN = 5994
EXPECTED_TEST = 5794


def _reporthook(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    pct = min(pct, 100)
    bar = "#" * (pct // 2)
    print(f"\r  [{bar:<50}] {pct:3d}%", end="", flush=True)


def download_and_extract(dest: str):
    dest = Path(dest)
    archive_path = dest.parent / "CUB_200_2011.tgz"

    dest.parent.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading CUB-200-2011 from {DOWNLOAD_URL} ...")
        print("(approx. 1.1 GB, this may take a while)")
        urllib.request.urlretrieve(DOWNLOAD_URL, archive_path, _reporthook)
        print()
    else:
        print(f"Found existing archive at {archive_path}, skipping download.")

    if not dest.exists():
        print(f"Extracting to {dest.parent} ...")
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest.parent)
        print("Extraction complete.")
    else:
        print(f"Destination {dest} already exists, skipping extraction.")


def verify(dest: str):
    dest = Path(dest)

    required_files = [
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "classes.txt",
    ]
    for fname in required_files:
        path = dest / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    with open(dest / "classes.txt") as f:
        classes = [line.strip() for line in f if line.strip()]
    num_classes = len(classes)

    with open(dest / "train_test_split.txt") as f:
        splits = [int(line.strip().split()[-1]) for line in f if line.strip()]
    num_train = sum(splits)
    num_test = len(splits) - num_train

    images_dir = dest / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    print("\nVerification:")
    print(f"  Classes found  : {num_classes} (expected {EXPECTED_CLASSES})")
    print(f"  Train images   : {num_train} (expected {EXPECTED_TRAIN})")
    print(f"  Test images    : {num_test} (expected {EXPECTED_TEST})")

    if num_classes != EXPECTED_CLASSES:
        print(f"WARNING: Expected {EXPECTED_CLASSES} classes, got {num_classes}")
    if num_train != EXPECTED_TRAIN:
        print(f"WARNING: Expected {EXPECTED_TRAIN} training images, got {num_train}")
    if num_test != EXPECTED_TEST:
        print(f"WARNING: Expected {EXPECTED_TEST} test images, got {num_test}")

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
