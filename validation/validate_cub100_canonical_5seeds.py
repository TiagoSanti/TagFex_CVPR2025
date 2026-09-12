#!/usr/bin/env python3
"""Strict preflight for the local CUB-200 100+20 five-seed campaign."""

from __future__ import annotations

import re
import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.experiment_paths import experiment_log_dir
QUEUE = ROOT / "experiments/cub100_canonical_5seeds/queues/queue_cub100_canonical_5seeds_local.txt"
BASE = "configs/all_in_one/cub200_100-20_antB0_nceA1_antGlobal_nceGlobal_resnet18.yaml"
COMMON = "experiments/cub100_canonical_5seeds/configs/common.yaml"
STORAGE = "configs/hosts/xavier/cub100_canonical_5seeds/storage.yaml"
SEEDS = tuple(range(1993, 1998))
METHOD_ORDER = (
    "Baseline InfoNCE",
    "ANT-FS-GR",
    "ANT-IV-GR",
    "ANT-IV-AR",
    "ANT-FS-AR",
)
METHOD_OVERLAYS = {
    "Baseline InfoNCE": "experiments/cub100_canonical_5seeds/configs/infonce.yaml",
    "ANT-FS-GR": "experiments/cub100_canonical_5seeds/configs/fs_gr.yaml",
    "ANT-IV-GR": "experiments/cub100_canonical_5seeds/configs/iv_gr.yaml",
    "ANT-IV-AR": "experiments/cub100_canonical_5seeds/configs/iv_ar.yaml",
    "ANT-FS-AR": "experiments/cub100_canonical_5seeds/configs/fs_ar.yaml",
}


def load_config(paths: list[str]) -> dict:
    merged: dict = {}
    for relative in paths:
        path = ROOT / relative
        assert path.is_file(), f"config ausente: {relative}"
        with path.open(encoding="utf-8") as stream:
            merged.update(yaml.safe_load(stream) or {})
    return merged


def method_name(config: dict) -> str:
    if float(config["ant_beta"]) == 0:
        return "Baseline InfoNCE"
    return {
        (True, True): "ANT-FS-GR",
        (False, True): "ANT-IV-GR",
        (False, False): "ANT-IV-AR",
        (True, False): "ANT-FS-AR",
    }[(bool(config["ant_symmetric_full"]), bool(config["ant_max_global"]))]


def output_basename(config: dict, seed: int) -> str:
    return experiment_log_dir({**config, 'seed': seed}).name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-logs", action="store_true", help="Inspect operational state; never delete partial runs")
    args = parser.parse_args()
    assert QUEUE.is_file(), f"fila ausente: {QUEUE}"
    active = [
        line.strip()
        for line in QUEUE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(active) == 25, f"fila tem {len(active)} entradas; esperado: 25"

    identities: list[tuple[str, int]] = []
    basenames: set[str] = set()
    existing_complete = []
    existing_partial = []

    for line_number, line in enumerate(active, 1):
        fields = line.split("|")
        assert len(fields) == 3, f"linha {line_number}: esperado três campos"
        config_spec, description, seed_text = fields
        assert re.fullmatch(r"\d+", seed_text), f"linha {line_number}: seed inválida"
        seed = int(seed_text)
        paths = config_spec.split(",")
        assert len(paths) == 4, f"linha {line_number}: esperado quatro YAMLs"
        assert paths == [BASE, COMMON, paths[2], STORAGE], (
            f"linha {line_number}: ordem de overlays inválida"
        )
        config = load_config(paths)
        method = method_name(config)
        assert paths[2] == METHOD_OVERLAYS[method], (
            f"linha {line_number}: overlay não corresponde a {method}"
        )
        assert method in description, f"linha {line_number}: descrição divergente"
        identities.append((method, seed))

        assert config["dataset_name"] == "cub200"
        assert config["scenario"].lower() == "cil 100-20"
        assert len(config["class_order"]) == 200
        assert sorted(config["class_order"]) == list(range(200))
        assert config["trainloader_params"]["batch_size"] == 64
        assert config["testloader_params"]["batch_size"] == 192
        assert config["num_aug"] == 2
        assert config["init_epochs"] == 200
        assert config["inc_epochs"] == 170
        assert config["memory_configs"] == {"fixed_size": True, "memory_size": 2000}
        assert config["backbone_configs"] == {
            "name": "resnet18",
            "params": {"small_base": True},
        }
        assert config["infonce_max_global"] is True
        assert config["ant_detach_reference"] is True
        assert config["ant_formulation"] == "logsumexp"
        assert config["avg_last_k"] == 0
        assert config["debug_similarity"] is False
        assert str(Path(config["log_dir"])) == "logs"
        for forbidden in (
            "study_max_tasks",
            "study_max_epochs",
            "study_max_batches_per_epoch",
        ):
            assert forbidden not in config, f"{method}/{seed}: limite de piloto presente"

        if method == "Baseline InfoNCE":
            assert float(config["ant_beta"]) == 0
        else:
            assert float(config["ant_beta"]) == 0.5
            assert float(config["ant_margin"]) == 0.5
            assert "detach" in description.lower()

        basename = output_basename(config, seed)
        assert basename not in basenames, f"basename duplicado: {basename}"
        basenames.add(basename)
        assert "nceLocal" not in basename
        if method == "Baseline InfoNCE":
            assert "refDetached" not in basename
        else:
            assert basename.endswith(f"_nceGlobal_refDetached_s{seed}")

        matches = sorted((ROOT / "logs").glob(f"{basename}*")) if args.inspect_logs else []
        for match in matches:
            gist = match / "exp_gistlog.log"
            task_count = (
                gist.read_text(encoding="utf-8", errors="replace").count("avg_nme1")
                if gist.is_file()
                else 0
            )
            target = existing_complete if task_count >= 6 else existing_partial
            target.append(match.name)

    expected = [(method, seed) for method in METHOD_ORDER for seed in SEEDS]
    assert identities == expected, "ordem/cobertura não segue baseline + ranking + seeds"
    assert len(set(identities)) == 25, "identidades duplicadas"
    if existing_partial:
        print(f"REVIEW: diretórios parciais preservados: {existing_partial}")

    print("OK: 25 identidades únicas = 5 métodos x 5 seeds")
    print("OK: ordem = Baseline, FS-GR, IV-GR, IV-AR, FS-AR")
    print("OK: CUB-200 100+20, class_order fixa, 200/170 épocas")
    print("OK: small_base=true, batch=64, num_aug=2, memória=2000")
    print("OK: ANT canônico detached; InfoNCE nceGlobal; sem TeacherAvg/SBS")
    print("OK: log_dir=./logs; configuração estática validada")
    if args.inspect_logs:
        print(f"INFO: candidatos com seis tarefas: {len(set(existing_complete))}; a retomada exige proveniência compatível")
    else:
        print("INFO: logs não inspecionados; validação estática somente")


if __name__ == "__main__":
    main()
