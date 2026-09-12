import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from utils.experiment_paths import experiment_log_dir

ROOT = Path(__file__).resolve().parents[1]


def test_private_lock_directory_is_shared_across_checkouts(tmp_path):
    locks = tmp_path / "locks"
    locks.mkdir(mode=0o700)
    env = {**os.environ, "TAGFEX_LOCK_DIR": str(locks)}
    helper = ROOT / "scripts/lib/paths.sh"
    command = ['bash', '-c', 'source "$1"; tagfex_validate_lock_dir; exec 9>"$TAGFEX_LOCK_DIR/test.lock"; flock -n 9', 'test', str(helper)]
    with (locks / "test.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert subprocess.run(command, env=env, cwd=tmp_path).returncode == 1
    assert subprocess.run(command, env=env, cwd=tmp_path).returncode == 0
    locks.chmod(0o755)
    assert subprocess.run(command, env=env, capture_output=True).returncode == 78


def test_historical_lock_names_and_upstream_pairs_preserved():
    dependencies = json.loads((ROOT / "docs/RESTRUCTURING_LOCK_DEPENDENCIES.json").read_text())
    moves = json.loads((ROOT / "docs/RESTRUCTURING_MOVES.json").read_text())
    for script in moves.values():
        if script.endswith('.sh'):
            assert '/tmp/tagfex_' not in (ROOT / script).read_text()
    for script, names in dependencies.items():
        content = (ROOT / script).read_text()
        for name in names:
            assert f'$TAGFEX_LOCK_DIR/{name}' in content
        assert '/tmp/tagfex_' not in content
    pairs = [
        ('experiments/ant_fs_gr/run_ant_fs_gr_after_current.sh', 'experiments/baseline_teacheravg/run_baseline_teacheravg_queue.sh', 'tagfex_baseline_teacheravg.lock'),
        ('experiments/cub_iv_gr/run_cub_iv_gr_after_ant_fs_gr.sh', 'experiments/ant_fs_gr/run_ant_fs_gr_queue.sh', 'tagfex_ant_fs_gr.lock'),
        ('experiments/ant_detach_factorial/run_ant_detach_factorial_wolverine_recovery.sh', 'experiments/ant_detach_factorial/run_ant_detach_factorial_wolverine.sh', 'tagfex_ant_detach_factorial_wolverine.lock'),
    ]
    for waiter, producer, name in pairs:
        assert name in dependencies[waiter] and name in dependencies[producer]


@pytest.mark.parametrize('script,gpu', [
    ('run_ant_central_extra_seeds_wolverine_gpu0.sh', '0'),
    ('run_ant_central_extra_seeds_wolverine_gpu1.sh', '1'),
    ('run_ant_central_extra_seeds_wolverine_gpu1_recovery.sh', '1'),
    ('sync_wolverine_central_extra_seeds.sh', None),
])
def test_wolverine_launchers_and_collector_share_profile_without_remote_io(script, gpu, tmp_path):
    fixture = tmp_path / 'repo'
    moves = json.loads((ROOT / 'docs/RESTRUCTURING_MOVES.json').read_text())
    for name in (moves[script], 'scripts/lib/paths.sh', 'scripts/maintenance/profile_path.py',
                 'utils/configuration.py', 'configs/hosts/wolverine/ant_central_extra_seeds/storage.yaml',
                 'configs/hosts/xavier/ant_central_extra_seeds/storage.yaml'):
        dest = fixture / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    engine = fixture / ('scripts/sync/sync_wolverine_refdetach_results.sh' if gpu is None else 'scripts/execution/run_sbs_queue.sh')
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_text('printf "%s\\n" "$ORCH_LOG" "$ALLOWED_GPU_IDS" "$REMOTE_ROOT" "$REMOTE_ORCHESTRATOR" "$LOCAL_RESULTS"\n')
    env = {k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR', 'PYTHONDONTWRITEBYTECODE') if k in os.environ}
    env['TAGFEX_PYTHON'] = sys.executable
    result = subprocess.run(['bash', str(fixture / moves[script])], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True, timeout=5)
    fields = result.stdout.splitlines()
    remote = '/var/tmp/tiago/ANT_central_extra_seeds_20260906'
    if gpu is None:
        assert fields[2:] == [remote, remote + '/orchestrator', str(fixture / 'logs')]
    else:
        assert fields[0].startswith(remote + '/orchestrator/')
        assert fields[1] == gpu


def test_cli_defaults_and_explicit_null_paths_are_distinct(tmp_path):
    from utils.configuration import queue_training_config
    config = tmp_path / 'config.yaml'
    config.write_text('debug: true\n')
    loaded = queue_training_config([config], 1997)
    assert loaded['log_dir'] == Path('logs')
    assert loaded['seed'] == 1997
    assert loaded['debug'] is True
    config.write_text('log_dir: null\nforce_no_debug: true\ndebug: true\nseed: 42\n')
    loaded = queue_training_config([config], 1997)
    assert loaded['log_dir'] is None
    assert loaded['seed'] == 42
    assert loaded['debug'] is False


@pytest.mark.parametrize('executor', ['run_sbs_queue.sh', 'run_avgk_queue.sh'])
@pytest.mark.parametrize('state', ['completed', 'invalid_yaml', 'missing_config'])
def test_executor_skips_or_fails_without_launching_gpu(executor, state, tmp_path):
    import yaml
    # Full shell executor in a disposable checkout; GPU launcher is replaced.
    fixture = tmp_path / "repo"
    files = [f'scripts/execution/{executor}', 'scripts/lib/paths.sh',
             'scripts/maintenance/check_queue_entry.py', 'utils/configuration.py',
             'utils/argument.py', 'utils/experiment_paths.py', 'utils/research_identity.py', 'utils/provenance.py']
    for name in files:
        dest = fixture / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    config = {'dataset_name': 'cifar100', 'log_dir': str(fixture / 'logs'), 'scenario': 'cil 2-2', 'class_order': list(range(4)), 'seed': 42}
    path = fixture / 'config.yaml'
    if state == 'completed':
        path.write_text(yaml.safe_dump(config))
        output = experiment_log_dir(config)
        output.mkdir(parents=True)
        (output / 'exp_gistlog.log').write_text('avg_nme1: 1\navg_nme1: 2\n')
        from utils.provenance import canonical_hash
        (output / 'provenance').mkdir()
        (output / 'provenance/run-fixture.json').write_text(json.dumps({'kind':'experiment_run','status':'completed','configuration':{'effective':config,'effective_hash':canonical_hash(config)}}))
    elif state == 'invalid_yaml':
        path.write_text('invalid: [yaml')
    queue = fixture / 'queue.txt'
    queue.write_text('config.yaml|example|1997\n')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    python = bin_dir / 'python3'
    python.write_text('#!/usr/bin/env bash\nif [[ "$1" == *auto_run_on_free_gpu.py ]]; then\n  touch "$LAUNCH_MARKER"\n  exit 99\nfi\nexec "$REAL_PYTHON" "$@"\n')
    python.chmod(0o700)
    locks = tmp_path / 'locks'
    locks.mkdir(mode=0o700)
    marker = tmp_path / 'launched'
    env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'],
           'REAL_PYTHON': sys.executable, 'LAUNCH_MARKER': str(marker),
           'TAGFEX_LOCK_DIR': str(locks), 'QUEUE': str(queue), 'STOP_ON_FAILURE': '1'}
    # Do not inherit campaign-specific output overrides from an operator's shell.
    for key in ('LOCKFILE', 'ORCH_LOG', 'CONSOLE_DIR'):
        env.pop(key, None)
    result = subprocess.run(['bash', str(fixture / 'scripts/execution' / executor)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if state == 'completed' else 1), result.stdout + result.stderr
    assert not marker.exists()
    assert ('SKIP' if state == 'completed' else 'failed') in result.stdout
