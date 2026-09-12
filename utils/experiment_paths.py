"""Experiment identities shared by training, queue resumption and inspection.

experiment_log_dir is pure: preserve historical suffix formatting and path
semantics; version allocation stays in the learner. No directories are created.
"""
from pathlib import Path
import re
import json
from utils.research_identity import scientific_hash
from utils.provenance import canonical_hash


def expected_task_count(config):
    """Return the CIL task count, or None when it cannot be proven offline."""
    try:
        task, scenario = config["scenario"].lower().split()
        if task != "cil":
            return None
        if scenario == "joint":
            return 1
        total = len(config["class_order"])
        initial, increment = map(int, scenario.split("-"))
        if initial <= 0 or increment <= 0 or total < initial:
            return None
        if (total - initial) % increment:
            return None
        return 1 + (total - initial) // increment
    except (KeyError, TypeError, ValueError):
        return None


def completed_run(config, root=None):
    """Find a matching gist with all expected task summaries; read-only heuristic.

    Accept only the exact identity and its learner-generated _v2, _v3, ...
    variants. Unknown task counts are never treated as completed.
    """
    base = experiment_log_dir(config)
    tasks = expected_task_count(config)
    if base is None or tasks is None:
        return None
    if root is not None:
        base = Path(root) / base
    pattern = re.compile(re.escape(base.name) + r"(?:_v(?:[2-9]|[1-9][0-9]+))?\Z")
    try:
        candidates = sorted(base.parent.iterdir())
    except FileNotFoundError:
        return None
    for directory in candidates:
        if not pattern.fullmatch(directory.name):
            continue
        try:
            prefix = config.get("output_file_prefix", "exp")
            text = (directory / f"{prefix}_gistlog.log").read_text(errors="replace")
        except (FileNotFoundError, NotADirectoryError):
            continue
        if len(re.findall(r"\bavg_nme1\b", text)) >= tasks:
            manifests = list((directory / 'provenance').glob('run-*.json'))
            # A unique completed manifest is required; ambiguity never means done.
            if len(manifests) != 1:
                continue
            try:
                manifest = json.loads(manifests[0].read_text())
                recorded = manifest['configuration']['effective']
                if manifest.get('kind') != 'experiment_run' or manifest.get('status') != 'completed':
                    continue
                if manifest['configuration'].get('effective_hash') != canonical_hash(recorded):
                    continue
                if scientific_hash(recorded) != scientific_hash(config):
                    continue
            except (OSError, ValueError, TypeError, KeyError):
                continue
            return directory
    return None

def experiment_log_dir(config):
    """Return the historical unversioned log path without changing config."""
    log_dir = config.get("log_dir")
    if log_dir is None:
        return

    log_dir = Path(log_dir)

    # Build suffix based on parameters that differ from defaults
    suffix_parts = []

    # ANT parameters
    ant_beta = config.get("ant_beta", 0.0)
    suffix_parts.append(f"antB{ant_beta:.3f}".rstrip("0").rstrip("."))

    nce_alpha = config.get("nce_alpha", 1.0)
    suffix_parts.append(f"nceA{nce_alpha:.3f}".rstrip("0").rstrip("."))

    # Only include ant_margin if ant_beta > 0 (ANT is active)
    if ant_beta > 0:
        ant_margin = config.get("ant_margin", 0.1)
        suffix_parts.append(f"antM{ant_margin:.3f}".rstrip("0").rstrip("."))

    ant_max_global = config.get("ant_max_global", True)
    infonce_max_global = config.get("infonce_max_global", ant_max_global)

    # Check if symmetric_full is enabled
    ant_symmetric_full = config.get("ant_symmetric_full", False)
    if ant_symmetric_full:
        # Preserve historical FS-AR directory names while giving FS-GR a
        # distinct identity. Without this split, both variants collide.
        suffix_parts.append(
            "antSymmetricFullGlobal" if ant_max_global else "antSymmetricFull"
        )
    else:
        suffix_parts.append("antGlobal" if ant_max_global else "antLocal")
    suffix_parts.append("nceGlobal" if infonce_max_global else "nceLocal")

    if ant_beta > 0 and config.get("ant_detach_reference", False):
        suffix_parts.append("refDetached")

    # ANT loss formulation (only if non-default)
    ant_formulation = config.get("ant_formulation", "logsumexp")
    if ant_formulation != "logsumexp":
        suffix_parts.append(f"form{ant_formulation}")

    # Avg-K teacher (only if enabled)
    avg_last_k = config.get("avg_last_k", 0)
    if avg_last_k > 0:
        suffix_parts.append(f"avgK{avg_last_k}")

    # SBS (only if enabled)
    sbs_q = config.get("sbs_q", 0.0)
    sbs_s = config.get("sbs_s", 0.0)
    if sbs_q > 0 or sbs_s > 0:
        suffix_parts.append(f"sbsQ{sbs_q:.2f}S{sbs_s:.2f}".rstrip("0").rstrip("."))

    # Always append seed so every run has a unique, traceable directory
    seed = config.get("seed", 1993)
    suffix_parts.append(f"s{seed}")

    # Contrast factors (optional - you can enable these if needed)
    if config.get("include_contrast_in_logdir", False):
        contrast_factor = config.get("contrast_factor", 1.0)
        if contrast_factor != 1.0:
            suffix_parts.append(f"cf{contrast_factor:.2f}".rstrip("0").rstrip("."))

        contrast_kd_factor = config.get("contrast_kd_factor", 2.0)
        if contrast_kd_factor != 2.0:
            suffix_parts.append(
                f"ckf{contrast_kd_factor:.2f}".rstrip("0").rstrip(".")
            )

    # Build new log directory path
    if suffix_parts:
        suffix = "_" + "_".join(suffix_parts)

        # If log_dir is just a base directory (like './logs'), create experiment subdirectory
        # Only a final component exactly named 'logs' is treated as a root.
        if log_dir.name == "logs" or str(log_dir) in ["./logs", "logs"]:
            # Create experiment name from dataset and scenario
            dataset_name = config.get("dataset_name", "dataset")
            scenario = config.get("scenario", "").split()[-1]
            exp_name = f"exp_{dataset_name}_{scenario}{suffix}"
            new_log_dir = log_dir / exp_name
        else:
            # Append suffix to existing experiment name
            new_log_dir = log_dir.parent / (log_dir.name + suffix)

        return new_log_dir
