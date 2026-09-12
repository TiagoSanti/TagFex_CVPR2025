#!/usr/bin/env python3
"""Preflight for the 1996/1997 baseline + detached ANT campaign."""

from __future__ import annotations

import re
from math import isclose
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
QUEUES = {
    "xavier_gpu0": (
        ROOT / "configs/queue_ant_central_extra_seeds_local.txt",
        "configs/hosts/xavier/ant_central_extra_seeds/storage.yaml",
        "logs",
        17,
    ),
    "wolverine_gpu0": (
        ROOT / "configs/queue_ant_central_extra_seeds_wolverine_gpu0.txt",
        "configs/hosts/wolverine/ant_central_extra_seeds/storage.yaml",
        "/var/tmp/tiago/ANT_central_extra_seeds_20260906/logs",
        16,
    ),
    "wolverine_gpu1": (
        ROOT / "configs/queue_ant_central_extra_seeds_wolverine_gpu1.txt",
        "configs/hosts/wolverine/ant_central_extra_seeds/storage.yaml",
        "/var/tmp/tiago/ANT_central_extra_seeds_20260906/logs",
        17,
    ),
}
METHODS = {"Baseline InfoNCE", "ANT-IV-GR", "ANT-IV-AR", "ANT-FS-GR", "ANT-FS-AR"}
PROTOCOLS = {
    "cifar100_10-10",
    "cifar100_50-10",
    "tiny_imagenet_20-20",
    "tiny_imagenet_100-20",
    "cub200_20-20",
}
SEEDS = {1996, 1997}
EXPECTED_CLASS_COUNTS = {"cifar100": 100, "tiny_imagenet": 200, "cub200": 200}
WEIGHTS = {
    "cifar100_10-10": 5.45,
    "cifar100_50-10": 3.85,
    "tiny_imagenet_20-20": 9.75,
    "tiny_imagenet_100-20": 6.53,
    "cub200_20-20": 5.12,
}


def load_config(paths: list[str]) -> dict:
    merged: dict = {}
    for relpath in paths:
        path = ROOT / relpath
        assert path.is_file(), f"config ausente: {relpath}"
        with path.open(encoding="utf-8") as stream:
            merged.update(yaml.safe_load(stream) or {})
    return merged


def method_name(config: dict) -> str:
    if float(config["ant_beta"]) == 0:
        return "Baseline InfoNCE"
    full = bool(config["ant_symmetric_full"])
    global_reference = bool(config["ant_max_global"])
    return {
        (False, True): "ANT-IV-GR",
        (False, False): "ANT-IV-AR",
        (True, True): "ANT-FS-GR",
        (True, False): "ANT-FS-AR",
    }[(full, global_reference)]


def protocol_name(config: dict) -> str:
    return f'{config["dataset_name"]}_{config["scenario"].split()[-1]}'


def output_basename(config: dict, seed: int) -> str:
    beta = float(config["ant_beta"])
    parts = [f"antB{beta:g}", f'nceA{float(config["nce_alpha"]):g}']
    if beta > 0:
        parts.append(f'antM{float(config["ant_margin"]):g}')
    if config["ant_symmetric_full"]:
        parts.append("antSymmetricFullGlobal" if config["ant_max_global"] else "antSymmetricFull")
    else:
        parts.append("antGlobal" if config["ant_max_global"] else "antLocal")
    parts.append("nceGlobal" if config["infonce_max_global"] else "nceLocal")
    if beta > 0 and config["ant_detach_reference"]:
        parts.append("refDetached")
    parts.append(f"s{seed}")
    return f'exp_{protocol_name(config)}_{"_".join(parts)}'


def main() -> None:
    identities: dict[tuple[str, str, int], str] = {}
    class_orders: dict[str, list[int]] = {}
    host_hours: dict[str, float] = {}

    for host, (queue, expected_storage, expected_log_dir, expected_count) in QUEUES.items():
        assert queue.is_file(), f"fila ausente: {queue}"
        active = [
            line.strip()
            for line in queue.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert len(active) == expected_count, (
            f"{host}: {len(active)} entradas, esperado {expected_count}"
        )
        host_hours[host] = 0.0

        for line_number, line in enumerate(active, 1):
            fields = line.split("|")
            assert len(fields) == 3, f"{host}:{line_number}: formato inválido"
            config_paths, description, seed_text = fields
            assert re.fullmatch(r"\d+", seed_text), f"seed inválida: {seed_text!r}"
            seed = int(seed_text)
            assert seed in SEEDS, f"seed fora do desenho: {seed}"
            paths = config_paths.split(",")
            assert paths[-1] == expected_storage, f"{host}: storage não é o último overlay"
            config = load_config(paths)

            protocol = protocol_name(config)
            method = method_name(config)
            identity = (protocol, method, seed)
            assert protocol in PROTOCOLS, f"protocolo inesperado: {protocol}"
            assert method in METHODS, f"método inesperado: {method}"
            assert identity not in identities, (
                f"identidade duplicada entre {identities[identity]} e {host}: {identity}"
            )
            identities[identity] = host
            host_hours[host] += WEIGHTS[protocol]

            assert config["infonce_max_global"] is True, f"{identity}: não é nceGlobal"
            assert config["ant_detach_reference"] is True, f"{identity}: detach inativo"
            assert config["ant_formulation"] == "logsumexp", f"{identity}: formulação divergente"
            assert config["avg_last_k"] == 0, f"{identity}: TeacherAvg inesperado"
            assert config["debug_similarity"] is False, f"{identity}: debug inesperado"
            assert str(Path(config["log_dir"])) == expected_log_dir, (
                f"{identity}: log_dir={config['log_dir']!r}, esperado {expected_log_dir!r}"
            )
            if host.startswith("wolverine") and protocol == "cub200_20-20":
                assert paths[-2] == "configs/hosts/wolverine/ant_central_extra_seeds/dataset_cub.yaml"
                assert config["dataset_root"] == (
                    "/var/tmp/tiago/ANT_detach_cub20_20260904/datasets/CUB_200_2011"
                )
            basename = output_basename(config, seed)
            assert "nceLocal" not in basename, f"nome legado produzido: {basename}"
            if method == "Baseline InfoNCE":
                assert "refDetached" not in basename, f"detach não deve nomear beta=0: {basename}"
            else:
                assert basename.endswith(f"_nceGlobal_refDetached_s{seed}"), basename
                assert "detach" in description.lower(), f"descrição omite detach: {description}"

            dataset = config["dataset_name"]
            order = config.get("class_order")
            assert isinstance(order, list), f"{identity}: class_order não está fixada"
            assert len(order) == EXPECTED_CLASS_COUNTS[dataset], f"{identity}: class_order incompleta"
            assert len(set(order)) == len(order), f"{identity}: class_order possui repetição"
            if dataset in class_orders:
                assert class_orders[dataset] == order, f"{dataset}: protocolos usam ordens distintas"
            else:
                class_orders[dataset] = order

    expected = {
        (protocol, method, seed)
        for protocol in PROTOCOLS
        for method in METHODS
        for seed in SEEDS
    }
    assert set(identities) == expected, (
        f"cobertura inválida: faltam={sorted(expected - set(identities))}; "
        f"sobram={sorted(set(identities) - expected)}"
    )
    assert isclose(
        sum(host_hours.values()),
        sum(WEIGHTS.values()) * len(METHODS) * len(SEEDS),
        abs_tol=1e-9,
    )

    print("OK: 50 identidades únicas = 5 protocolos x 5 métodos x 2 seeds")
    for host in QUEUES:
        count = sum(assigned == host for assigned in identities.values())
        print(f"OK: {host}: {count} execuções, {host_hours[host]:.2f} GPU-h estimadas")
    print("OK: detach ativo nas quatro variantes ANT; baseline beta=0")
    print("OK: nomes nceGlobal; armazenamento e class_order validados")


if __name__ == "__main__":
    main()
