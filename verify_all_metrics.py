#!/usr/bin/env python3
"""
Extrai métricas finais de todos os experimentos para revisão completa.
"""
import re
import numpy as np
from pathlib import Path


def extract_metrics(log_path):
    """Extrai acc1_curve e nme1_curve do log."""
    try:
        with open(log_path, "r") as f:
            content = f.read()

        # Procurar pelas últimas curvas completas
        acc1_pattern = r"acc1_curve \[([\d\.\s]+)\]"
        nme1_pattern = r"nme1_curve \[([\d\.\s]+)\]"

        acc1_matches = re.findall(acc1_pattern, content)
        nme1_matches = re.findall(nme1_pattern, content)

        if not acc1_matches:
            return None, None, None, None, None

        # Pegar a última curva completa
        last_acc1 = acc1_matches[-1]
        acc1_values = np.array([float(x) for x in last_acc1.split()])

        # Para NME, pegar a última também
        nme1_values = None
        if nme1_matches:
            last_nme1 = nme1_matches[-1]
            nme1_values = np.array([float(x) for x in last_nme1.split()])

        avg_acc1 = acc1_values.mean()
        last_acc1_val = acc1_values[-1]
        avg_nme1 = nme1_values.mean() if nme1_values is not None else None

        return avg_acc1, last_acc1_val, avg_nme1, acc1_values, nme1_values

    except Exception as e:
        print(f"Erro ao processar {log_path}: {e}")
        return None, None, None, None, None


# Diretórios para verificar
experiments = [
    # CIFAR-100 10-10
    ("done_exp_cifar100_10-10", "CIFAR-100 10-10", "Baseline duplicata"),
    (
        "done_exp_cifar100_10-10_baseline_tagfex_original",
        "CIFAR-100 10-10",
        "Baseline TagFex Original",
    ),
    (
        "done_exp_cifar100_10-10_infonce_local_anchor",
        "CIFAR-100 10-10",
        "InfoNCE Local Anchor",
    ),
    (
        "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.1_antGlobal",
        "CIFAR-100 10-10",
        "ANT β=0.5, m=0.1, Global",
    ),
    (
        "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.1_antLocal",
        "CIFAR-100 10-10",
        "ANT β=0.5, m=0.1, Local",
    ),
    (
        "exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal",
        "CIFAR-100 10-10",
        "ANT β=0.5, m=0.5, Local",
    ),
    (
        "exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal",
        "CIFAR-100 10-10",
        "ANT β=1.0, m=0.1, Local",
    ),
    (
        "done_exp_cifar100_10-10_antB1_nceA1_antM0.3_antLocal",
        "CIFAR-100 10-10",
        "ANT β=1.0, m=0.3, Local",
    ),
    (
        "done_exp_cifar100_10-10_antB1_nceA1_antM0.5_antLocal",
        "CIFAR-100 10-10",
        "ANT β=1.0, m=0.5, Local",
    ),
    # CIFAR-100 50-10
    (
        "done_exp_cifar100_50-10_antB0_nceA1_antGlobal",
        "CIFAR-100 50-10",
        "Baseline β=0, Global",
    ),
    (
        "done_exp_cifar100_50-10_antB0_nceA1_antLocal",
        "CIFAR-100 50-10",
        "Baseline β=0, Local",
    ),
    (
        "exp_cifar100_50-10_antB1_nceA1_antM0.5_antLocal",
        "CIFAR-100 50-10",
        "ANT β=1.0, m=0.5, Local",
    ),
    # ImageNet-100 10-10
    (
        "done_exp_imagenet100_10-10_antB0_nceA1_antLocal",
        "ImageNet-100 10-10",
        "Baseline β=0, Local",
    ),
]

print("=" * 100)
print("REVISÃO COMPLETA DE MÉTRICAS - TODOS OS EXPERIMENTOS")
print("=" * 100)

logs_dir = Path("logs")
results = []

for exp_dir, dataset, description in experiments:
    log_file = logs_dir / exp_dir / "exp_gistlog.log"

    if not log_file.exists():
        print(f"\n❌ {exp_dir}")
        print(f"   Arquivo não encontrado: {log_file}")
        continue

    avg_acc1, last_acc1, avg_nme1, acc1_curve, nme1_curve = extract_metrics(log_file)

    if avg_acc1 is None:
        print(f"\n❌ {exp_dir}")
        print(f"   Não foi possível extrair métricas")
        continue

    print(f"\n✅ {description}")
    print(f"   Dataset: {dataset}")
    print(f"   Diretório: {exp_dir}")
    print(f"   Avg Acc@1: {avg_acc1:.2f}%")
    print(f"   Last Acc@1: {last_acc1:.2f}%")
    if avg_nme1 is not None:
        print(f"   Avg NME@1: {avg_nme1:.2f}%")
    print(f"   Tarefas: {len(acc1_curve)}")
    print(f"   Curva Acc@1: {acc1_curve}")

    results.append(
        {
            "dir": exp_dir,
            "dataset": dataset,
            "description": description,
            "avg_acc1": avg_acc1,
            "last_acc1": last_acc1,
            "avg_nme1": avg_nme1,
            "acc1_curve": acc1_curve,
            "nme1_curve": nme1_curve,
            "num_tasks": len(acc1_curve),
        }
    )

# Identificar o melhor resultado para CIFAR-100 10-10
print("\n" + "=" * 100)
print("ANÁLISE: MELHOR RESULTADO EM CIFAR-100 10-10")
print("=" * 100)

cifar_10_10 = [r for r in results if r["dataset"] == "CIFAR-100 10-10"]
if cifar_10_10:
    baseline = next(
        (r for r in cifar_10_10 if "Baseline TagFex Original" in r["description"]), None
    )

    if baseline:
        print(f"\nBaseline: {baseline['avg_acc1']:.2f}% Avg Acc@1")

        # Ordenar por Avg Acc@1
        sorted_results = sorted(cifar_10_10, key=lambda x: x["avg_acc1"], reverse=True)

        print("\n🏆 TOP 5 RESULTADOS:")
        for i, r in enumerate(sorted_results[:5], 1):
            delta = r["avg_acc1"] - baseline["avg_acc1"]
            marker = "⭐" * min(3, int(abs(delta) / 0.1))
            print(
                f"{i}. {r['description']}: {r['avg_acc1']:.2f}% (Δ {delta:+.2f}%) {marker}"
            )

        best = sorted_results[0]
        print(f"\n🎯 MELHOR CONFIGURAÇÃO:")
        print(f"   {best['description']}")
        print(f"   Avg Acc@1: {best['avg_acc1']:.2f}%")
        print(f"   Last Acc@1: {best['last_acc1']:.2f}%")
        print(f"   Avg NME@1: {best['avg_nme1']:.2f}%")
        print(
            f"   Melhoria vs Baseline: +{best['avg_acc1'] - baseline['avg_acc1']:.2f}%"
        )
        print(f"   Curva: {list(best['acc1_curve'])}")

print("\n" + "=" * 100)
