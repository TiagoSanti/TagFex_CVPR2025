#!/usr/bin/env python3
"""Create source, run, and artifact manifests for the controlled validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "logs",
    "checkpoints",
    "validation_runtime",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def source_files(repo: Path) -> list[Path]:
    files = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        if relative.as_posix() == "snapshot_manifest.json":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(repo).as_posix())


def artifact_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )


def file_records(files: list[Path], root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]


def load_effective_config(paths: list[Path]) -> dict:
    effective: dict = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if not isinstance(content, dict):
            raise TypeError(f"Configuration is not a mapping: {path}")
        effective.update(content)
    return effective


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=20
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def environment_record() -> dict[str, object]:
    record: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import numpy
        import torch
        import torchvision

        record.update(
            {
                "numpy": numpy.__version__,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "visible_gpu_count": torch.cuda.device_count(),
                "visible_gpus": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "uuid": str(torch.cuda.get_device_properties(index).uuid),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as error:  # manifest must record environment failures
        record["torch_environment_error"] = repr(error)
    return record


def dataset_record(effective: dict) -> dict[str, object]:
    root_value = effective.get("dataset_root")
    if not root_value:
        return {"error": "dataset_root is absent from effective configuration"}
    root = Path(os.path.expanduser(str(root_value))).resolve()
    if not root.is_dir():
        return {"root": str(root), "error": "dataset root does not exist"}

    key_files = []
    for relative in ("wnids.txt", "words.txt", "val/val_annotations.txt"):
        path = root / relative
        if path.is_file():
            key_files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    train_root = root / "train"
    classes = []
    if train_root.is_dir():
        for class_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
            images = class_dir / "images"
            names = sorted(path.name for path in images.iterdir() if path.is_file())
            classes.append(
                {
                    "class": class_dir.name,
                    "image_count": len(names),
                    "image_names_hash": canonical_hash(names),
                }
            )

    val_images = root / "val" / "images"
    val_names = (
        sorted(path.name for path in val_images.iterdir() if path.is_file())
        if val_images.is_dir()
        else []
    )
    identity = {
        "root": str(root),
        "key_files": key_files,
        "train_classes": classes,
        "validation_image_count": len(val_names),
        "validation_image_names_hash": canonical_hash(val_names),
    }
    identity["dataset_hash"] = canonical_hash(identity)
    return identity


def snapshot(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    records = file_records(source_files(repo), repo)
    manifest = {
        "schema_version": 1,
        "created_at": now(),
        "base_commit": args.base_commit,
        "source_files": records,
        "source_hash": canonical_hash(records),
    }
    write_json(repo / "snapshot_manifest.json", manifest)
    print(manifest["source_hash"])


def prepare(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    manifest_path = Path(args.manifest).resolve()
    if manifest_path.exists():
        raise FileExistsError(f"Run manifest already exists: {manifest_path}")
    snapshot_path = repo / "snapshot_manifest.json"
    snapshot_manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config_paths = [Path(path).resolve() for path in args.config]
    effective = load_effective_config(config_paths)
    config_records = file_records(config_paths, repo)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "condition": args.condition,
        "phase": args.phase,
        "seed": args.seed,
        "status": "running",
        "started_at": now(),
        "command": args.command,
        "output_dir": str(Path(args.output_dir).resolve()),
        "base_commit": snapshot_manifest["base_commit"],
        "source_hash": snapshot_manifest["source_hash"],
        "configuration_files": config_records,
        "effective_configuration": effective,
        "effective_configuration_hash": canonical_hash(effective),
        "dataset": dataset_record(effective),
        "environment": environment_record(),
    }
    write_json(manifest_path, manifest)
    print(manifest_path)


def finalize(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(manifest["output_dir"])
    records = file_records(artifact_files(output_dir), output_dir)
    manifest.update(
        {
            "status": "completed" if args.exit_code == 0 else "failed",
            "exit_code": args.exit_code,
            "finished_at": now(),
            "artifacts": records,
            "artifacts_hash": canonical_hash(records),
        }
    )
    write_json(manifest_path, manifest)
    print(manifest["status"])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="action", required=True)

    snapshot_parser = commands.add_parser("snapshot")
    snapshot_parser.add_argument("--repo", required=True)
    snapshot_parser.add_argument("--base-commit", required=True)
    snapshot_parser.set_defaults(func=snapshot)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--repo", required=True)
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--condition", required=True)
    prepare_parser.add_argument("--phase", required=True)
    prepare_parser.add_argument("--seed", required=True, type=int)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--command", required=True)
    prepare_parser.add_argument("--config", required=True, nargs="+")
    prepare_parser.set_defaults(func=prepare)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--manifest", required=True)
    finalize_parser.add_argument("--exit-code", required=True, type=int)
    finalize_parser.set_defaults(func=finalize)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
