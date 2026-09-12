"""Behavioral checks for queue equivalence and isolated shell entry points."""
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_queues", ROOT / "scripts/maintenance/audit_queues.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
MOVES = json.loads((ROOT / "docs/RESTRUCTURING_MOVES.json").read_text())


def test_queue_compatibility_preserves_content_and_config_order():
    count = 0
    for old, new in MOVES.items():
        if not old.endswith(".txt"):
            continue
        assert (ROOT / old).is_symlink()
        assert (ROOT / old).read_bytes() == (ROOT / new).read_bytes()
        before, after = audit.audit_queue(Path(old)), audit.audit_queue(Path(new))
        assert before["entries"] == after["entries"]
        count += after["count"]
    assert count == 334


def test_snapshot_blocks_all_shell_entry_points_from_other_cwd():
    if not (ROOT / ".snapshot-isolated").exists():
        pytest.skip("Snapshot-only operational guard")
    with tempfile.TemporaryDirectory() as cwd:
        for old, new in MOVES.items():
            if not old.endswith(".sh"):
                continue
            for script in (old, new):
                result = subprocess.run(["bash", str(ROOT / script)], cwd=cwd,
                                        text=True, capture_output=True, timeout=5)
                assert result.returncode == 78, (script, result.stdout, result.stderr)
                assert "Isolated snapshot" in result.stderr
        assert not list(Path(cwd).iterdir())


def test_audit_from_other_cwd_and_legacy_queue_path():
    with tempfile.TemporaryDirectory() as cwd:
        result = subprocess.run([sys.executable, str(ROOT / "scripts/maintenance/audit_queues.py"),
                                 "--queue", "configs/queue_ant_central_extra_seeds_local.txt", "--json"],
                                cwd=cwd, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)[0]["count"] == 17
        assert not list(Path(cwd).iterdir())


def test_local_launcher_dispatches_to_canonical_executor_without_training():
    # A disposable fixture replaces the executor; no locks, GPUs or network calls.
    old = "run_ant_central_extra_seeds_local.sh"
    new = MOVES[old]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in (old, new, "scripts/lib/paths.sh", "scripts/maintenance/profile_path.py",
                     "utils/configuration.py", "configs/hosts/xavier/ant_central_extra_seeds/storage.yaml"):
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        executor = root / "scripts/execution/run_sbs_queue.sh"
        executor.parent.mkdir(parents=True)
        executor.write_text('printf "%s\\n" "$QUEUE" "$ALLOWED_GPU_IDS" "$UPDATE_REPORT"\n')
        result = subprocess.run(["bash", str(root / old)], cwd=root.parent,
                                capture_output=True, text=True, check=True, timeout=5,
                                env={**os.environ, "TAGFEX_PYTHON": sys.executable})
        assert result.stdout.splitlines() == [
            str(root / "experiments/ant_central_extra_seeds/queues/queue_ant_central_extra_seeds_local.txt"),
            "0", "0"]


def test_shallow_merge_yaml_seed_precedence_and_missing_file():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "base.yaml").write_text("seed: 42\nnested: {a: 1, b: 2}\n")
        (root / "override.yaml").write_text("nested: {c: 3}\n")
        queue = root / "queue.txt"
        queue.write_text("base.yaml,override.yaml|sample|1997\n")
        row = audit.audit_queue(queue, root)["entries"][0]
        assert row["queue_seed"] == 1997
        assert row["effective_config"] == {"seed": 42, "nested": {"c": 3}}
        (root / "override.yaml").unlink()
        with pytest.raises(FileNotFoundError):
            audit.audit_queue(queue, root)


@pytest.mark.parametrize("line", ["base.yaml|missing seed", "base.yaml|sample|invalid", "|sample|1997", "a.yaml,|sample|1997"])
def test_invalid_queue_is_rejected(line):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "a.yaml").write_text("seed: 1\n")
        queue = root / "queue.txt"
        queue.write_text(line)
        with pytest.raises(ValueError):
            audit.audit_queue(queue, root)
