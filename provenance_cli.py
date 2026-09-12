#!/usr/bin/env python3
"""Inventory legacy logs and verify TagFex provenance manifests."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.provenance import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_hash,
    file_record,
    git_record,
    records_hash,
    sha256_file,
    utc_now,
)


CORE_LOG_RE = re.compile(r"(?:gistlog|stdlog\d*)\.log$")


def _inventory_paths(logs_dir: Path, scope: str) -> list[Path]:
    paths = []
    for path in logs_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(logs_dir)
        if scope == "full":
            paths.append(path)
        elif CORE_LOG_RE.search(path.name) or (
            "provenance" in relative.parts and path.suffix == ".json"
        ):
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(logs_dir).as_posix())


def _repo_display_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def create_inventory(
    *,
    logs_dir: Path,
    output: Path,
    scope: str,
    repo: Path,
    settle_seconds: int = 600,
) -> dict[str, Any]:
    paths = _inventory_paths(logs_dir, scope)
    cutoff_ns = time.time_ns() - settle_seconds * 1_000_000_000
    hot_experiments = {
        path.relative_to(logs_dir).parts[0]
        for path in paths
        if path.stat().st_mtime_ns > cutoff_ns
    }
    records = []
    excluded = []
    for path in paths:
        experiment = path.relative_to(logs_dir).parts[0]
        if experiment in hot_experiments:
            excluded.append(
                {
                    "path": _repo_display_path(path, repo),
                    "reason": f"experiment modified within {settle_seconds} seconds",
                }
            )
            continue
        try:
            records.append(file_record(path, root=repo))
        except RuntimeError as error:
            excluded.append(
                {"path": _repo_display_path(path, repo), "reason": str(error)}
            )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        relative = Path(record["path"])
        try:
            experiment = relative.relative_to(logs_dir.relative_to(repo)).parts[0]
        except (ValueError, IndexError):
            experiment = relative.parent.as_posix()
        grouped[experiment].append(record)
    experiment_records = [
        {
            "experiment": experiment,
            "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
            "files_hash": records_hash(files),
        }
        for experiment, files in sorted(grouped.items())
    ]
    source_files = [
        file_record(Path(__file__).resolve(), root=repo),
        file_record((repo / "utils" / "provenance.py").resolve(), root=repo),
    ]
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "legacy_log_inventory",
        "status": "completed",
        "created_at": utc_now(),
        "provenance_level": "artifact-only backfill",
        "limitation": (
            "Hashes prove the current legacy artifacts. They cannot recover "
            "the exact historical code, configuration files, environment, or dataset."
        ),
        "scope": scope,
        "scope_definition": (
            "all files below logs/" if scope == "full" else
            "result-bearing gist/std logs and any future in-run provenance manifests"
        ),
        "working_directory": str(repo.resolve()),
        "logs_dir": str(logs_dir.resolve()),
        "source": {
            "git": git_record(repo),
            "files": source_files,
            "source_hash": records_hash(source_files),
        },
        "files": records,
        "candidate_file_count": len(paths),
        "file_count": len(records),
        "settle_seconds": settle_seconds,
        "excluded_unstable": excluded,
        "excluded_unstable_count": len(excluded),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "files_hash": records_hash(records),
        "experiments": experiment_records,
        "experiments_hash": canonical_hash(experiment_records),
    }
    atomic_write_json(output, manifest)
    return manifest


def _expand_files(paths: Iterable[Path], excluded: set[Path]) -> list[Path]:
    files = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path not in excluded:
                files.add(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.resolve() not in excluded:
                files.add(child.resolve())
    return sorted(files, key=str)


def create_bundle(
    *,
    bundle_kind: str,
    inputs: Iterable[Path],
    artifacts: Iterable[Path],
    parents: Iterable[Path],
    output: Path,
    repo: Path,
) -> dict[str, Any]:
    excluded = {output.resolve()}
    input_records = [
        file_record(path, root=repo)
        for path in _expand_files(inputs, excluded)
    ]
    artifact_records = [
        file_record(path, root=repo)
        for path in _expand_files(artifacts, excluded)
    ]
    parent_records = [
        file_record(path, root=repo)
        for path in _expand_files(parents, excluded)
    ]
    source_files = [
        file_record(Path(__file__).resolve(), root=repo),
        file_record((Path(__file__).resolve().parent / "utils" / "provenance.py"), root=repo),
    ]
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "lineage_bundle",
        "bundle_kind": bundle_kind,
        "status": "completed",
        "created_at": utc_now(),
        "working_directory": str(repo.resolve()),
        "source": {
            "git": git_record(repo),
            "files": source_files,
            "source_hash": records_hash(source_files),
        },
        "inputs": {
            "files": input_records,
            "files_hash": records_hash(input_records),
        },
        "parents": {
            "manifests": parent_records,
            "manifests_hash": records_hash(parent_records),
        },
        "artifacts": {
            "files": artifact_records,
            "artifacts_hash": records_hash(artifact_records),
        },
    }
    manifest["lineage_hash"] = canonical_hash(
        {
            "bundle_kind": bundle_kind,
            "inputs_hash": manifest["inputs"]["files_hash"],
            "parents_hash": manifest["parents"]["manifests_hash"],
            "artifacts_hash": manifest["artifacts"]["artifacts_hash"],
        }
    )
    atomic_write_json(output, manifest)
    return manifest


def _resolve(record: Mapping[str, Any], base: Path) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else base / path


def _verify_records(
    records: Iterable[Mapping[str, Any]], base: Path, label: str
) -> list[str]:
    errors = []
    for record in records:
        path = _resolve(record, base)
        if not path.is_file():
            errors.append(f"{label}: missing: {path}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            errors.append(f"{label}: size mismatch: {path}")
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            errors.append(f"{label}: sha256 mismatch: {path}")
    return errors


def _verify_source(manifest: Mapping[str, Any], working: Path) -> list[str]:
    source = manifest.get("source", {})
    records = source.get("files", [])
    if records:
        errors = _verify_records(records, working, "manifest source")
        if records_hash(records) != source.get("source_hash"):
            errors.append("manifest source: aggregate hash mismatch")
        return errors
    script = source.get("script")
    return _verify_records([script], working, "manifest source") if script else []


def verify_manifest(path: Path) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    kind = manifest.get("kind")
    working = Path(manifest.get("working_directory", ".")).resolve()
    errors = []

    if kind == "legacy_log_inventory":
        records = manifest.get("files", [])
        errors.extend(_verify_records(records, working, "inventory"))
        if records_hash(records) != manifest.get("files_hash"):
            errors.append("inventory: aggregate files_hash mismatch")
        errors.extend(_verify_source(manifest, working))
    elif kind == "statistical_analysis":
        records = manifest.get("inputs", {}).get("selected_files", [])
        errors.extend(_verify_records(records, working, "analysis input"))
        if records_hash(records) != manifest.get("inputs", {}).get("selected_files_hash"):
            errors.append("analysis input: aggregate hash mismatch")
        parents = manifest.get("inputs", {}).get("parent_run_manifests", [])
        errors.extend(_verify_records(parents, working, "parent run manifest"))
        if records_hash(parents) != manifest.get("inputs", {}).get("parent_run_manifests_hash"):
            errors.append("parent run manifest: aggregate hash mismatch")
        errors.extend(_verify_source(manifest, working))
        artifact_base = path.parent
        artifacts = manifest.get("artifacts", {}).get("files", [])
        errors.extend(_verify_records(artifacts, artifact_base, "analysis artifact"))
        if records_hash(artifacts) != manifest.get("artifacts", {}).get("artifacts_hash"):
            errors.append("analysis artifact: aggregate hash mismatch")
    elif kind == "experiment_report":
        errors.extend(_verify_source(manifest, working))
        selected = manifest.get("selection", {}).get("selected_experiments", [])
        gistlogs = [item["gistlog"] for item in selected]
        debug_inputs = [
            item["debug_input"]
            for item in selected
            if item.get("debug_input") is not None
        ]
        errors.extend(_verify_records(gistlogs, working, "report gistlog"))
        errors.extend(_verify_records(debug_inputs, working, "report debug input"))
        if canonical_hash(selected) != manifest.get("selection", {}).get("selected_experiments_hash"):
            errors.append("report selection: selected_experiments_hash mismatch")
        decisions = manifest.get("selection", {}).get("decisions", [])
        if canonical_hash(decisions) != manifest.get("selection", {}).get("decisions_hash"):
            errors.append("report selection: decisions_hash mismatch")
        parameters = manifest.get("selection", {}).get("parameters", {})
        if canonical_hash(parameters) != manifest.get("selection", {}).get("parameters_hash"):
            errors.append("report selection: parameters_hash mismatch")
        parents = manifest.get("parents", {}).get("run_manifests", [])
        errors.extend(_verify_records(parents, working, "report parent manifest"))
        if records_hash(parents) != manifest.get("parents", {}).get("run_manifests_hash"):
            errors.append("report parent manifest: aggregate hash mismatch")
        artifacts = manifest.get("artifacts", {}).get("files", [])
        errors.extend(_verify_records(artifacts, working, "report artifact"))
        if records_hash(artifacts) != manifest.get("artifacts", {}).get("artifacts_hash"):
            errors.append("report artifact: aggregate hash mismatch")
        expected_lineage_payload = {
            "source_hash": manifest.get("source", {}).get("source_hash"),
            "selected_experiments_hash": manifest.get("selection", {}).get("selected_experiments_hash"),
            "parent_run_manifests_hash": manifest.get("parents", {}).get("run_manifests_hash"),
            "selection_parameters_hash": manifest.get("selection", {}).get("parameters_hash"),
            "environment_hash": canonical_hash(manifest.get("environment", {})),
        }
        if expected_lineage_payload != manifest.get("lineage_payload"):
            errors.append("report: lineage_payload mismatch")
        if canonical_hash(expected_lineage_payload) != manifest.get("lineage_hash"):
            errors.append("report: lineage_hash mismatch")
        html_records = [
            record
            for record in artifacts
            if str(record.get("path", "")).endswith(".html")
        ]
        for record in html_records:
            html_path = _resolve(record, working)
            try:
                html_text = html_path.read_text(encoding="utf-8")
            except OSError:
                continue
            if manifest.get("lineage_hash") not in html_text:
                errors.append(f"report artifact: lineage hash absent from HTML: {html_path}")
    elif kind == "experiment_run":
        source = manifest.get("source", {}).get("files", [])
        errors.extend(_verify_records(source, working, "experiment source"))
        if records_hash(source) != manifest.get("source", {}).get("source_hash"):
            errors.append("experiment source: aggregate hash mismatch")
        configs = manifest.get("configuration", {}).get("merge_order", [])
        errors.extend(_verify_records(configs, working, "experiment configuration"))
        if records_hash(configs) != manifest.get("configuration", {}).get("configuration_files_hash"):
            errors.append("experiment configuration: aggregate hash mismatch")
        artifact_base = Path(manifest["output_dir"])
        artifacts = manifest.get("artifacts", {}).get("files", [])
        errors.extend(_verify_records(artifacts, artifact_base, "experiment artifact"))
        if records_hash(artifacts) != manifest.get("artifacts", {}).get("artifacts_hash"):
            errors.append("experiment artifact: aggregate hash mismatch")
    elif kind == "lineage_bundle":
        inputs = manifest.get("inputs", {}).get("files", [])
        errors.extend(_verify_records(inputs, working, "bundle input"))
        if records_hash(inputs) != manifest.get("inputs", {}).get("files_hash"):
            errors.append("bundle input: aggregate hash mismatch")
        parents = manifest.get("parents", {}).get("manifests", [])
        errors.extend(_verify_records(parents, working, "bundle parent"))
        if records_hash(parents) != manifest.get("parents", {}).get("manifests_hash"):
            errors.append("bundle parent: aggregate hash mismatch")
        artifacts = manifest.get("artifacts", {}).get("files", [])
        errors.extend(_verify_records(artifacts, working, "bundle artifact"))
        if records_hash(artifacts) != manifest.get("artifacts", {}).get("artifacts_hash"):
            errors.append("bundle artifact: aggregate hash mismatch")
        expected_lineage_hash = canonical_hash(
            {
                "bundle_kind": manifest.get("bundle_kind"),
                "inputs_hash": manifest.get("inputs", {}).get("files_hash"),
                "parents_hash": manifest.get("parents", {}).get("manifests_hash"),
                "artifacts_hash": manifest.get("artifacts", {}).get("artifacts_hash"),
            }
        )
        if expected_lineage_hash != manifest.get("lineage_hash"):
            errors.append("bundle: lineage_hash mismatch")
        errors.extend(_verify_source(manifest, working))
    else:
        errors.append(f"unsupported manifest kind: {kind!r}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="hash legacy log artifacts")
    inventory.add_argument("--logs-dir", type=Path, default=Path("logs"))
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--scope", choices=("core", "full"), default="core")
    inventory.add_argument(
        "--settle-seconds",
        type=int,
        default=600,
        help="exclude an entire experiment if a selected file changed this recently",
    )

    bundle = commands.add_parser(
        "bundle", help="connect arbitrary inputs, parent manifests, and artifacts"
    )
    bundle.add_argument("--kind", required=True, dest="bundle_kind")
    bundle.add_argument("--input", type=Path, action="append", default=[])
    bundle.add_argument("--artifact", type=Path, action="append", default=[])
    bundle.add_argument("--parent-manifest", type=Path, action="append", default=[])
    bundle.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify files referenced by a manifest")
    verify.add_argument("manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent
    if args.command == "inventory":
        manifest = create_inventory(
            logs_dir=args.logs_dir.resolve(),
            output=args.output.resolve(),
            scope=args.scope,
            repo=repo,
            settle_seconds=args.settle_seconds,
        )
        print(f"Files: {manifest['file_count']}")
        print(f"Bytes: {manifest['total_bytes']}")
        print(f"Excluded unstable: {manifest['excluded_unstable_count']}")
        print(f"Hash: {manifest['files_hash']}")
        print(f"Manifest: {args.output}")
        return
    if args.command == "bundle":
        manifest = create_bundle(
            bundle_kind=args.bundle_kind,
            inputs=args.input,
            artifacts=args.artifact,
            parents=args.parent_manifest,
            output=args.output.resolve(),
            repo=repo,
        )
        print(f"Lineage hash: {manifest['lineage_hash']}")
        print(f"Manifest: {args.output}")
        return

    errors = verify_manifest(args.manifest.resolve())
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALID")


if __name__ == "__main__":
    main()
