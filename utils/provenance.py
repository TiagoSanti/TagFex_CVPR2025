"""Content-addressed provenance for TagFex experiments and analyses.

The module is deliberately independent from the logging format.  Experiment
manifests live below ``<log_dir>/provenance/`` and therefore do not change any
historical ``exp_*.log`` consumer.  All public ``safe_*`` helpers are
best-effort: provenance failures must never abort a training run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_NAME = "tagfex.provenance"
SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
DEFAULT_DATASET_HASH_MODE = "full"

_SOURCE_ROOTS = (
    "main.py",
    "methods",
    "modules",
    "utils",
    "loggers",
)
_SOURCE_SUFFIXES = {".py"}
_SOURCE_EXTRA_FILES = {"requirements.txt", "pyproject.toml", "uv.lock"}
_IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_json(value: Any) -> Any:
    """Convert common runtime values to deterministic JSON-compatible data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, set):
        return sorted((normalize_json(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            normalize_json(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _display_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return str(path.resolve())


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"file changed while it was being hashed: {path}")
    return {
        "path": _display_path(path, root),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    stable = [
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in records
    ]
    return canonical_hash(stable)


def _run_output(command: Sequence[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            list(command),
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def git_record(repo: Path) -> dict[str, Any]:
    commit = _run_output(["git", "rev-parse", "HEAD"], repo)
    status = _run_output(["git", "status", "--porcelain=v1", "--untracked-files=no"], repo)
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status),
        "tracked_status_sha256": canonical_hash(status or ""),
    }


def _source_paths(repo: Path) -> list[Path]:
    paths: set[Path] = set()
    for entry in _SOURCE_ROOTS:
        candidate = repo / entry
        if candidate.is_file():
            paths.add(candidate)
        elif candidate.is_dir():
            for path in candidate.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix in _SOURCE_SUFFIXES
                    and not any(part in _IGNORED_PARTS for part in path.parts)
                ):
                    paths.add(path)
    for name in _SOURCE_EXTRA_FILES:
        candidate = repo / name
        if candidate.is_file():
            paths.add(candidate)
    return sorted(paths, key=lambda path: path.relative_to(repo).as_posix())


def source_record(repo: Path) -> dict[str, Any]:
    records = [file_record(path, root=repo) for path in _source_paths(repo)]
    return {
        "git": git_record(repo),
        "files": records,
        "file_count": len(records),
        "source_hash": records_hash(records),
    }


def configuration_record(
    config_paths: Iterable[Path], effective_configuration: Mapping[str, Any], repo: Path
) -> dict[str, Any]:
    records = [
        file_record(Path(path).expanduser().resolve(), root=repo)
        for path in config_paths
    ]
    effective = normalize_json(effective_configuration)
    return {
        "merge_order": records,
        "configuration_files_hash": records_hash(records),
        "effective": effective,
        "effective_hash": canonical_hash(effective),
    }


def _dataset_paths(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part in _IGNORED_PARTS for part in path.parts)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _dataset_cache_path(cache_dir: Path, root: Path) -> Path:
    key = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
    return cache_dir / "datasets" / f"{key}.json"


def dataset_record(
    root_value: str | Path | None,
    *,
    cache_dir: Path,
    mode: str = DEFAULT_DATASET_HASH_MODE,
) -> dict[str, Any]:
    """Fingerprint dataset structure or full content.

    The stable identity excludes absolute paths and mtimes.  Mtimes are used
    only to decide whether a cached full-content hash can be reused.
    """
    if root_value is None:
        return {"status": "not_configured", "hash_mode": mode}
    root = Path(os.path.expandvars(os.path.expanduser(str(root_value)))).resolve()
    if not root.is_dir():
        return {
            "status": "unavailable",
            "root": str(root),
            "hash_mode": mode,
            "error": "dataset root is not a directory",
        }
    if mode not in {"full", "cached-full", "structure", "none"}:
        raise ValueError(
            "dataset hash mode must be: full, cached-full, structure, or none"
        )
    if mode == "none":
        return {"status": "skipped", "root": str(root), "hash_mode": mode}

    paths = _dataset_paths(root)
    metadata_records = []
    structure_records = []
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        total_bytes += stat.st_size
        structure_records.append({"path": relative, "size_bytes": stat.st_size})
        metadata_records.append(
            {
                "path": relative,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    structure_hash = canonical_hash(structure_records)
    metadata_hash = canonical_hash(metadata_records)
    result: dict[str, Any] = {
        "status": "verified",
        "root": str(root),
        "hash_mode": mode,
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "structure_hash": structure_hash,
    }
    if mode == "structure":
        result["dataset_hash"] = structure_hash
        return result

    cache_path = _dataset_cache_path(cache_dir, root)
    cached: dict[str, Any] | None = None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if mode == "cached-full" and cached and cached.get("metadata_hash") == metadata_hash:
        content_hash = cached.get("content_hash")
        if isinstance(content_hash, str):
            result.update(
                {
                    "dataset_hash": content_hash,
                    "content_hash": content_hash,
                    "cache_reused": True,
                    "cache_validation": "path+size+mtime_ns+ctime_ns",
                }
            )
            return result

    digest = hashlib.sha256()
    for path, structure in zip(paths, structure_records):
        record = {**structure, "sha256": sha256_file(path)}
        digest.update(canonical_bytes(record))
        digest.update(b"\n")
    content_hash = digest.hexdigest()
    atomic_write_json(
        cache_path,
        {
            "schema_version": SCHEMA_VERSION,
            "root": str(root),
            "metadata_hash": metadata_hash,
            "structure_hash": structure_hash,
            "content_hash": content_hash,
            "file_count": len(paths),
            "total_bytes": total_bytes,
            "created_at": utc_now(),
        },
    )
    result.update(
        {
            "dataset_hash": content_hash,
            "content_hash": content_hash,
            "cache_reused": False,
            "cache_validation": None if mode == "full" else "path+size+mtime_ns+ctime_ns",
        }
    )
    return result


def environment_record() -> dict[str, Any]:
    packages = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "pandas", "pyyaml"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    record: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        record.update(
            {
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
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
    except Exception as error:
        record["torch_environment_error"] = repr(error)
    return record


def artifact_record(output_dir: Path) -> dict[str, Any]:
    paths = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and "provenance" not in path.relative_to(output_dir).parts
            and path.relative_to(output_dir).as_posix() != "provenance.json"
        ),
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    records = [file_record(path, root=output_dir) for path in paths]
    return {
        "files": records,
        "file_count": len(records),
        "artifacts_hash": records_hash(records),
    }


@dataclass
class ExperimentProvenance:
    manifest_path: Path
    manifest: dict[str, Any]

    def finalize(self, status: str, error: BaseException | None = None) -> None:
        self.manifest.update(
            {
                "status": status,
                "finished_at": utc_now(),
                "artifacts": artifact_record(Path(self.manifest["output_dir"])),
            }
        )
        if error is not None:
            self.manifest["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
        atomic_write_json(self.manifest_path, self.manifest)


def start_experiment_provenance(
    *,
    repo: Path,
    output_dir: Path,
    effective_configuration: Mapping[str, Any],
    config_paths: Iterable[Path],
    command: Sequence[str],
) -> ExperimentProvenance:
    repo = repo.resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    started_at = utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-p{os.getpid()}"
    manifest_path = output_dir / "provenance" / f"run-{run_id}.json"
    mode = os.environ.get("TAGFEX_DATASET_HASH_MODE", DEFAULT_DATASET_HASH_MODE)
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "kind": "experiment_run",
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "command": list(command),
        "working_directory": str(Path.cwd().resolve()),
        "output_dir": str(output_dir),
        "source": source_record(repo),
        "configuration": configuration_record(
            config_paths, effective_configuration, repo
        ),
        "dataset": dataset_record(
            effective_configuration.get("dataset_root"),
            cache_dir=repo / ".provenance" / "cache",
            mode=mode,
        ),
        "environment": environment_record(),
    }
    atomic_write_json(manifest_path, manifest)
    return ExperimentProvenance(manifest_path=manifest_path, manifest=manifest)


def safe_start_experiment(**kwargs: Any) -> ExperimentProvenance | None:
    try:
        return start_experiment_provenance(**kwargs)
    except Exception as error:
        print(
            f"WARNING: provenance start failed; training will continue: {error!r}",
            file=sys.stderr,
            flush=True,
        )
        return None


def safe_finalize_experiment(
    tracker: ExperimentProvenance | None,
    status: str,
    error: BaseException | None = None,
) -> None:
    if tracker is None:
        return
    try:
        tracker.finalize(status, error)
    except Exception as provenance_error:
        print(
            "WARNING: provenance finalization failed; training result is unchanged: "
            f"{provenance_error!r}",
            file=sys.stderr,
            flush=True,
        )
