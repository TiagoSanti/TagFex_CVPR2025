#!/usr/bin/env python3
"""
Análise do impacto Local vs Global no desempenho do TagFex/ANT.
Compara pares de experimentos que diferem apenas no escopo (Local vs Global).
"""

import re
from pathlib import Path


def parse_nme1_curve(log_file):
    """Parse the nme1_curve from exp_gistlog.log"""
    nme1_curve = None

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"nme1_curve \[([\d.\s]+)\]", line)
            if match:
                values_str = match.group(1)
                nme1_curve = [float(x) for x in values_str.split()]

    return nme1_curve


def analyze_local_vs_global():
    """Analisa o impacto de Local vs Global anchor."""

    logs_dir = Path("logs")

    # Definir pares de experimentos (Local vs Global)
    pairs = [
        {
            "name": "ant_β=0.5",
            "local": "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.1_antLocal",
            "global": "done_exp_cifar100_10-10_antB0.5_nceA1_antM0.1_antGlobal",
        },
        {
            "name": "ant_β=1.0",
            "local": "idone_exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal",
            "global": "idone_exp_cifar100_10-10_antB1_nceA1_antM0.1_antGlobal",
        },
    ]

    print("=" * 100)
    print("ANÁLISE: IMPACTO DE LOCAL VS GLOBAL ANCHOR")
    print("=" * 100)
    print()

    # Baseline para referência
    baseline_path = logs_dir / "exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal"
    baseline_curve = parse_nme1_curve(baseline_path / "exp_gistlog.log")
    print(f"BASELINE (InfoNCE puro):")
    print(f"  Curva completa: {baseline_curve}")
    print(f"  Performance final (Task 10): {baseline_curve[-1]:.2f}%")
    print(f"  Performance média: {sum(baseline_curve)/len(baseline_curve):.2f}%")
    print(f"  Forgetting (Task 1→10): {baseline_curve[0] - baseline_curve[-1]:.2f}%")
    print()
    print("-" * 100)
    print()

    overall_local_wins = 0
    overall_global_wins = 0
    overall_ties = 0

    for pair in pairs:
        print(f"COMPARAÇÃO: {pair['name']}")
        print("=" * 100)

        # Parse Local
        local_path = logs_dir / pair["local"]
        local_curve = parse_nme1_curve(local_path / "exp_gistlog.log")

        # Parse Global
        global_path = logs_dir / pair["global"]
        global_curve = parse_nme1_curve(global_path / "exp_gistlog.log")

        print()
        print(f"LOCAL:  {local_curve}")
        print(f"GLOBAL: {global_curve}")
        print()

        # Comparar task por task
        min_len = min(len(local_curve), len(global_curve))
        local_wins = 0
        global_wins = 0
        ties = 0

        print("COMPARAÇÃO POR TASK:")
        print(
            f"{'Task':<8} {'Local':<10} {'Global':<10} {'Diferença':<12} {'Vencedor':<10}"
        )
        print("-" * 60)

        for i in range(min_len):
            local_val = local_curve[i]
            global_val = global_curve[i]
            diff = local_val - global_val

            if abs(diff) < 0.1:
                winner = "Empate"
                ties += 1
            elif diff > 0:
                winner = "LOCAL"
                local_wins += 1
            else:
                winner = "GLOBAL"
                global_wins += 1

            print(
                f"Task {i+1:<3} {local_val:<10.2f} {global_val:<10.2f} "
                f"{diff:+10.2f}% {winner:<10}"
            )

        print("-" * 60)
        print()

        # Estatísticas do par
        print("RESUMO DO PAR:")
        print(
            f"  Local vence em:  {local_wins}/{min_len} tasks ({100*local_wins/min_len:.1f}%)"
        )
        print(
            f"  Global vence em: {global_wins}/{min_len} tasks ({100*global_wins/min_len:.1f}%)"
        )
        print(f"  Empates:         {ties}/{min_len} tasks ({100*ties/min_len:.1f}%)")
        print()

        # Métricas agregadas
        local_avg = sum(local_curve[:min_len]) / min_len
        global_avg = sum(global_curve[:min_len]) / min_len

        local_final = local_curve[min_len - 1]
        global_final = global_curve[min_len - 1]

        local_forgetting = local_curve[0] - local_curve[min_len - 1]
        global_forgetting = global_curve[0] - global_curve[min_len - 1]

        print("MÉTRICAS AGREGADAS:")
        print(
            f"  Performance Média:  Local={local_avg:.2f}%  Global={global_avg:.2f}%  "
            f"(Δ={local_avg-global_avg:+.2f}%)"
        )
        print(
            f"  Performance Final:  Local={local_final:.2f}%  Global={global_final:.2f}%  "
            f"(Δ={local_final-global_final:+.2f}%)"
        )
        print(
            f"  Forgetting:         Local={local_forgetting:.2f}%  Global={global_forgetting:.2f}%  "
            f"(Δ={local_forgetting-global_forgetting:+.2f}%)"
        )
        print()

        # Comparação com baseline
        print("COMPARAÇÃO COM BASELINE:")
        baseline_avg = sum(baseline_curve[:min_len]) / min_len
        baseline_final = baseline_curve[min_len - 1]
        baseline_forgetting = baseline_curve[0] - baseline_curve[min_len - 1]

        print(f"  Local vs Baseline:")
        print(f"    Média:  {local_avg-baseline_avg:+.2f}%")
        print(f"    Final:  {local_final-baseline_final:+.2f}%")
        print(f"    Forgetting: {local_forgetting-baseline_forgetting:+.2f}%")
        print()
        print(f"  Global vs Baseline:")
        print(f"    Média:  {global_avg-baseline_avg:+.2f}%")
        print(f"    Final:  {global_final-baseline_final:+.2f}%")
        print(f"    Forgetting: {global_forgetting-baseline_forgetting:+.2f}%")
        print()

        overall_local_wins += local_wins
        overall_global_wins += global_wins
        overall_ties += ties

        print()
        print("=" * 100)
        print()

    # Resumo geral
    total_comparisons = overall_local_wins + overall_global_wins + overall_ties

    print()
    print("=" * 100)
    print("CONCLUSÃO GERAL: LOCAL VS GLOBAL")
    print("=" * 100)
    print()
    print(f"Total de comparações: {total_comparisons} tasks")
    print()
    print(
        f"LOCAL vence em:  {overall_local_wins} tasks ({100*overall_local_wins/total_comparisons:.1f}%)"
    )
    print(
        f"GLOBAL vence em: {overall_global_wins} tasks ({100*overall_global_wins/total_comparisons:.1f}%)"
    )
    print(
        f"Empates:         {overall_ties} tasks ({100*overall_ties/total_comparisons:.1f}%)"
    )
    print()

    if overall_local_wins > overall_global_wins:
        advantage = overall_local_wins - overall_global_wins
        print(
            f"✓ HIPÓTESE CONFIRMADA: LOCAL tem vantagem de {advantage} tasks sobre GLOBAL"
        )
        print(f"  ({100*advantage/total_comparisons:.1f}% de vantagem)")
    elif overall_global_wins > overall_local_wins:
        advantage = overall_global_wins - overall_local_wins
        print(
            f"✗ HIPÓTESE REFUTADA: GLOBAL tem vantagem de {advantage} tasks sobre LOCAL"
        )
        print(f"  ({100*advantage/total_comparisons:.1f}% de vantagem)")
    else:
        print(f"⚠ RESULTADO INCONCLUSIVO: LOCAL e GLOBAL empatam em número de vitórias")

    print()
    print("=" * 100)


if __name__ == "__main__":
    analyze_local_vs_global()
