import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from utils.experiment_paths import completed_run, expected_task_count, experiment_log_dir
from scripts.maintenance.audit_queues import audit_queue
from scripts.maintenance.profile_path import profile_path
from utils.configuration import queue_training_config

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((ROOT / "tests/fixtures/queue_contract_20260911.json").read_text()) + json.loads((ROOT / "tests/fixtures/queue_contract_cub100_20260912.json").read_text())


def test_all_309_effective_configs_and_names_match_frozen_original():
    reports = {p: audit_queue(Path(p)) for p in {row["queue"] for row in CONTRACTS}}
    actual = {(p, row["line"]): row for p, report in reports.items() for row in report["entries"]}
    assert len(actual) == len(CONTRACTS) == 334
    for expected in CONTRACTS:
        row = actual[expected["queue"], expected["line"]]
        config = row["effective_config"]
        assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == expected["config_sha256"]
        assert row["queue_seed"] == expected["queue_seed"]
        assert row["description"] == expected["description"]
        train_config = queue_training_config([ROOT / dep["path"] for dep in row["configs"]], row["queue_seed"])
        assert str(experiment_log_dir(train_config)) == expected["log_dir"]
        assert row["experiment_log_dir"] == expected["log_dir"]
        assert row["expected_tasks"] is not None


def test_original_naming_edge_cases_and_no_config_mutation():
    examples = json.loads((ROOT / "tests/fixtures/naming_contract_20260911.json").read_text())
    for example in examples:
        config = copy.deepcopy(example["config"])
        result = experiment_log_dir(config)
        assert (str(result) if result is not None else None) == example["log_dir"]
        assert config == example["config"]


@pytest.fixture
def config(tmp_path):
    return {"log_dir": str(tmp_path / "logs"), "dataset_name": "cifar100", "scenario": "cil 2-2",
            "class_order": list(range(6)), "seed": 1993}


def write_gist(path, tasks, config=None):
    path.mkdir(parents=True)
    (path / "exp_gistlog.log").write_text("avg_nme1: 70\n" * tasks)
    if config is not None:
        from utils.provenance import canonical_hash
        (path / 'provenance').mkdir()
        (path / 'provenance/run-fixture.json').write_text(json.dumps({'kind':'experiment_run','status':'completed','configuration':{'effective':config,'effective_hash':canonical_hash(config)}}))


def test_completion_requires_full_exact_identity_or_valid_version(config):
    base = experiment_log_dir(config)
    assert completed_run(config) is None
    # Prefix globbing used to admit unrelated seeds and optional suffixes.
    for suffix in ("0", "_cf2", "_v1", "_v02", "_v2_extra"):
        write_gist(base.with_name(base.name + suffix), 3)
    write_gist(base, 2)
    assert completed_run(config) is None
    version = base.with_name(base.name + "_v10")
    write_gist(version, 3, config)
    assert completed_run(config) == version


def test_unknown_or_invalid_protocol_does_not_skip(config):
    write_gist(experiment_log_dir(config), 1)
    config.pop("class_order")
    assert expected_task_count(config) is None
    assert completed_run(config) is None
    config["class_order"] = list(range(5))
    assert expected_task_count(config) is None
    config["scenario"] = "cil joint"
    assert expected_task_count(config) == 1


def test_completion_uses_configured_log_prefix(config):
    base = experiment_log_dir(config)
    write_gist(base, 3, config)
    config['output_file_prefix'] = 'custom'
    assert completed_run(config) is None
    (base / 'exp_gistlog.log').rename(base / 'custom_gistlog.log')
    assert completed_run(config) == base


def test_learner_keeps_version_allocation(config):
    from methods.tagfex.tagfex import TagFex
    learner = TagFex.__new__(TagFex)
    learner.configs = config.copy()
    base = experiment_log_dir(config)
    base.mkdir(parents=True)
    base.with_name(base.name + "_v2").mkdir()
    learner._adjust_log_dir_with_loss_params()
    assert learner.configs["log_dir"] == base.with_name(base.name + "_v3")
    assert not learner.configs["log_dir"].exists()


def test_cli_seed_precedence_and_error_status(config, tmp_path):
    import yaml
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    command = [sys.executable, str(ROOT / "scripts/maintenance/check_queue_entry.py"), str(cfg), "1997"]
    assert subprocess.run(command, cwd=tmp_path).returncode == 1
    write_gist(experiment_log_dir(config), 3, config)
    assert subprocess.run(command, cwd=tmp_path).returncode == 0  # YAML's 1993 wins
    cfg.write_text("not: [valid")
    assert subprocess.run(command, cwd=tmp_path, capture_output=True).returncode == 2


def test_host_profiles_only_contain_paths_and_compatibility_links():
    import yaml
    moves = json.loads((ROOT / "docs/RESTRUCTURING_CONFIG_MOVES.json").read_text())
    for old, new in moves.items():
        assert (ROOT / old).is_symlink()
        assert (ROOT / old).resolve() == ROOT / new
        config = yaml.safe_load((ROOT / new).read_text())
        if new.startswith("configs/hosts/"):
            assert set(config) <= {"dataset_root", "log_dir", "ckpt_dir"}
        else:
            assert not set(config) & {"dataset_root", "log_dir", "ckpt_dir"}
    assert str(profile_path(Path("configs/hosts/wolverine/ant_central_extra_seeds/dataset_cub.yaml"), "dataset_root")) == "/var/tmp/tiago/ANT_detach_cub20_20260904/datasets/CUB_200_2011"


def test_relative_output_identity_is_resolved_after_naming(tmp_path):
    config = {"log_dir": ".", "scenario": "cil joint"}
    name = experiment_log_dir(config)
    write_gist(tmp_path / name, 1, config)
    assert completed_run(config, root=tmp_path) == tmp_path / name
