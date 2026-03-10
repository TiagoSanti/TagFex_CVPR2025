#!/usr/bin/env python3
"""
Compare CIFAR-100 50-10 InfoNCE experiments: Global vs Local anchors.

This script compares two baseline experiments (ant_beta=0, ant_margin=0):
- Global anchors: exp_cifar100_50-10_antB0_nceA1_antM0_antGlobal
- Local anchors: exp_cifar100_50-10_antB0_nceA1_antM0_antLocal

Both experiments use InfoNCE without ANT loss (ant_beta=0), differing only
in anchor type. This validates whether local anchors provide benefit even
without explicit ANT regularization for 50-10 scenario (harder than 10-10).

Analyzes:
1. Gap evolution (pos_mean - neg_mean)
2. Distance statistics (pos/neg mean, std, min, max)
3. Violation percentage trends
4. NME1 performance across tasks

Usage:
    python compare_50-10_global_vs_local.py
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd


def parse_ant_stats(log_file):
    """Parse ANT distance stats from exp_debug0.log."""
    data = []
    # Pattern for logs with all distance stats
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) .* pos_std: ([\d.]+) .* pos_min: ([\d.]+) .* pos_max: ([\d.]+) .* neg_mean: ([-\d.]+) .* neg_std: ([\d.]+) .* neg_min: ([-\d.]+) .* neg_max: ([-\d.]+) .* gap_mean: ([-\d.]+) .* gap_std: ([\d.]+) .* gap_min: ([-\d.]+) .* gap_max: ([-\d.]+) .* margin: ([\d.]+) .* violation_pct: ([\d.]+)%"

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                task = int(match.group(1))
                epoch = int(match.group(2))
                batch = int(match.group(3))
                pos_mean = float(match.group(4))
                neg_mean = float(match.group(8))

                data.append(
                    {
                        "task": task,
                        "epoch": epoch,
                        "batch": batch,
                        "pos_mean": pos_mean,
                        "pos_std": float(match.group(5)),
                        "pos_min": float(match.group(6)),
                        "pos_max": float(match.group(7)),
                        "neg_mean": neg_mean,
                        "neg_std": float(match.group(9)),
                        "neg_min": float(match.group(10)),
                        "neg_max": float(match.group(11)),
                        "gap_mean": float(match.group(12)),
                        "gap_std": float(match.group(13)),
                        "gap_min": float(match.group(14)),
                        "gap_max": float(match.group(15)),
                        "violation_pct": float(match.group(17)),
                        "current_gap": pos_mean - neg_mean,  # Calculate gap
                    }
                )

    return data


def parse_nme1_scores(log_file):
    """Parse NME1 evaluation scores from exp_gistlog.log."""
    scores = {}
    # Pattern for nme1_curve which contains cumulative scores
    pattern = r"nme1_curve \[([\d.\s]+)\]"

    with open(log_file, "r") as f:
        content = f.read()
        matches = re.findall(pattern, content)

        # Get the last (most complete) match
        if matches:
            last_curve = matches[-1]
            values = [float(x) for x in last_curve.split()]
            for task_idx, score in enumerate(values, start=1):
                scores[task_idx] = score

    return scores


def aggregate_by_task(data):
    """Aggregate statistics by task (average across all epochs/batches)."""
    if not data:
        return {}

    df = pd.DataFrame(data)

    # Group by task and compute means
    task_stats = {}
    for task in sorted(df["task"].unique()):
        task_data = df[df["task"] == task]
        task_stats[task] = {
            "pos_mean": task_data["pos_mean"].mean(),
            "pos_std": task_data["pos_std"].mean(),
            "neg_mean": task_data["neg_mean"].mean(),
            "neg_std": task_data["neg_std"].mean(),
            "current_gap": task_data["current_gap"].mean(),
            "gap_mean": task_data["gap_mean"].mean(),
            "gap_std": task_data["gap_std"].mean(),
            "violation_pct": task_data["violation_pct"].mean(),
        }

    return task_stats


def aggregate_by_epoch(data):
    """Aggregate statistics by task and epoch."""
    if not data:
        return {}

    df = pd.DataFrame(data)

    # Group by task and epoch
    epoch_stats = {}
    for task in sorted(df["task"].unique()):
        epoch_stats[task] = {}
        task_data = df[df["task"] == task]

        for epoch in sorted(task_data["epoch"].unique()):
            epoch_data = task_data[task_data["epoch"] == epoch]
            epoch_stats[task][epoch] = {
                "pos_mean": epoch_data["pos_mean"].mean(),
                "neg_mean": epoch_data["neg_mean"].mean(),
                "current_gap": epoch_data["current_gap"].mean(),
                "violation_pct": epoch_data["violation_pct"].mean(),
            }

    return epoch_stats


def plot_comparison(global_stats, local_stats, global_nme1, local_nme1, output_dir):
    """Create comprehensive comparison plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate data by task
    global_task = aggregate_by_task(global_stats)
    local_task = aggregate_by_task(local_stats)

    # Extract tasks
    global_tasks = sorted(global_task.keys())
    local_tasks = sorted(local_task.keys()) if local_task else []
    all_tasks = sorted(set(global_tasks) | set(local_tasks))

    # Helper function to plot with winner highlighting
    def plot_with_winners(
        ax,
        global_tasks,
        global_vals,
        local_tasks,
        local_vals,
        higher_is_better=True,
        ylabel="",
        title="",
        show_deltas=True,
    ):
        """Plot comparison with winner highlighting and delta labels."""
        from matplotlib.lines import Line2D

        # Plot lines first
        if local_tasks:
            ax.plot(
                local_tasks,
                local_vals,
                "-",
                color="#2ecc71",
                linewidth=2.5,
                alpha=0.6,
                zorder=2,
            )
        ax.plot(
            global_tasks,
            global_vals,
            "--",
            color="#e74c3c",
            linewidth=2,
            alpha=0.5,
            zorder=1,
        )

        # Plot markers with highlighting for winners and ties
        if local_tasks and len(local_vals) == len(global_vals):
            for i, task in enumerate(local_tasks):
                delta = local_vals[i] - global_vals[i]

                # Determine threshold for "tie" based on metric scale
                tie_threshold = 0.001 if abs(global_vals[i]) < 1.0 else 0.01

                if abs(delta) < tie_threshold:  # Tie
                    # Plot both markers with equal emphasis
                    ax.plot(
                        task,
                        local_vals[i],
                        "o",
                        color="#95a5a6",
                        markersize=10,
                        markeredgewidth=2,
                        markeredgecolor="#7f8c8d",
                        zorder=4,
                    )
                    ax.plot(
                        task,
                        global_vals[i],
                        "s",
                        color="#95a5a6",
                        markersize=10,
                        markeredgewidth=2,
                        markeredgecolor="#7f8c8d",
                        zorder=4,
                    )
                elif (higher_is_better and delta > 0) or (
                    not higher_is_better and delta < 0
                ):  # Local is better
                    # Highlight local (winner)
                    ax.plot(
                        task,
                        local_vals[i],
                        "o",
                        color="#2ecc71",
                        markersize=12,
                        markeredgewidth=2.5,
                        markeredgecolor="#27ae60",
                        zorder=5,
                    )
                    # De-emphasize global (loser)
                    ax.plot(
                        task,
                        global_vals[i],
                        "s",
                        color="#e74c3c",
                        markersize=8,
                        alpha=0.4,
                        zorder=3,
                    )
                else:  # Global is better
                    # Highlight global (winner)
                    ax.plot(
                        task,
                        global_vals[i],
                        "s",
                        color="#e74c3c",
                        markersize=12,
                        markeredgewidth=2.5,
                        markeredgecolor="#c0392b",
                        zorder=5,
                    )
                    # De-emphasize local (loser)
                    ax.plot(
                        task,
                        local_vals[i],
                        "o",
                        color="#2ecc71",
                        markersize=8,
                        alpha=0.4,
                        zorder=3,
                    )

                # Add delta label if requested
                if show_deltas and abs(delta) >= tie_threshold:
                    y_pos = (
                        max(local_vals[i], global_vals[i])
                        + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    )
                    ax.text(
                        task,
                        y_pos,
                        f"{delta:+.3f}",
                        fontsize=8,
                        ha="center",
                        color="#34495e",
                        fontweight="bold",
                    )

        # Legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="#2ecc71",
                marker="o",
                linestyle="-",
                linewidth=2.5,
                markersize=8,
                label="Local",
            ),
            Line2D(
                [0],
                [0],
                color="#e74c3c",
                marker="s",
                linestyle="--",
                linewidth=2,
                markersize=7,
                label="Global",
            ),
            Line2D(
                [0],
                [0],
                color="#95a5a6",
                marker="o",
                linestyle="none",
                linewidth=0,
                markersize=8,
                markeredgecolor="#7f8c8d",
                markeredgewidth=2,
                label="Empate",
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc="best")
        ax.set_xlabel("Task", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Reset matplotlib to default settings
    plt.rcdefaults()
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    fig.suptitle(
        "CIFAR-100 50-10: Global vs Local Anchor Comparison (InfoNCE Only)",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Gap Evolution (pos_mean - neg_mean)
    ax = axes[0, 0]
    global_gaps = [global_task[t]["current_gap"] for t in global_tasks]
    local_gaps = (
        [local_task[t]["current_gap"] for t in local_tasks] if local_tasks else []
    )
    plot_with_winners(
        ax,
        global_tasks,
        global_gaps,
        local_tasks,
        local_gaps,
        higher_is_better=True,
        ylabel="Gap (pos_mean - neg_mean)",
        title="Gap Evolution Across Tasks",
    )

    # 2. Positive Similarities
    ax = axes[0, 1]
    global_pos = [global_task[t]["pos_mean"] for t in global_tasks]
    local_pos = [local_task[t]["pos_mean"] for t in local_tasks] if local_tasks else []
    plot_with_winners(
        ax,
        global_tasks,
        global_pos,
        local_tasks,
        local_pos,
        higher_is_better=True,
        ylabel="Positive Mean Similarity",
        title="Positive Similarities",
    )

    # 3. Negative Similarities
    ax = axes[1, 0]
    global_neg = [global_task[t]["neg_mean"] for t in global_tasks]
    local_neg = [local_task[t]["neg_mean"] for t in local_tasks] if local_tasks else []
    plot_with_winners(
        ax,
        global_tasks,
        global_neg,
        local_tasks,
        local_neg,
        higher_is_better=False,  # Lower (more negative) is better
        ylabel="Negative Mean Similarity",
        title="Negative Similarities",
    )

    # 4. NME1 Performance
    ax = axes[1, 1]
    if global_nme1:
        global_nme1_tasks = sorted(global_nme1.keys())
        global_nme1_scores = [global_nme1[t] * 100 for t in global_nme1_tasks]
    else:
        global_nme1_tasks = []
        global_nme1_scores = []

    if local_nme1:
        local_nme1_tasks = sorted(local_nme1.keys())
        local_nme1_scores = [local_nme1[t] * 100 for t in local_nme1_tasks]
    else:
        local_nme1_tasks = []
        local_nme1_scores = []

    plot_with_winners(
        ax,
        global_nme1_tasks,
        global_nme1_scores,
        local_nme1_tasks,
        local_nme1_scores,
        higher_is_better=True,
        ylabel="NME1 Accuracy (%)",
        title="Evaluation Performance",
    )
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(
        output_dir / "cifar100_50-10_global_vs_local_comparison.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close("all")  # Close all figures to avoid memory issues

    print(
        f"\n✓ Comparison plot saved to: {output_dir / 'cifar100_50-10_global_vs_local_comparison.png'}"
    )

    # Generate markdown report
    generate_markdown_report(
        global_task, local_task, global_nme1, local_nme1, output_dir
    )


def generate_markdown_report(
    global_task, local_task, global_nme1, local_nme1, output_dir
):
    """Generate a comprehensive markdown report with comparison tables."""

    output_dir = Path(output_dir)
    report_path = output_dir / "cifar100_50-10_comparison_report.md"

    global_tasks = sorted(global_task.keys())
    local_tasks = sorted(local_task.keys()) if local_task else []

    with open(report_path, "w") as f:
        # Header
        f.write("# CIFAR-100 50-10: Global vs Local Anchor Comparison\n\n")
        f.write(
            f"**Gerado em**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Sumário Executivo\n\n")

        if global_nme1 and local_nme1:
            global_final = global_nme1[max(global_nme1.keys())] * 100
            local_final = local_nme1[max(local_nme1.keys())] * 100
            delta_nme1 = local_final - global_final

            f.write(f"**NME1 Accuracy Final**:\n")
            f.write(f"- Global: **{global_final:.2f}%**\n")
            f.write(f"- Local: **{local_final:.2f}%**\n")
            f.write(f"- Melhoria: **{delta_nme1:+.2f}%** ")
            if delta_nme1 > 0:
                f.write("✅ (Local é melhor)\n\n")
            elif delta_nme1 < 0:
                f.write("⚠️ (Global é melhor)\n\n")
            else:
                f.write("(Empate)\n\n")

        if global_task and local_task:
            global_avg_gap = np.mean(
                [global_task[t]["current_gap"] for t in global_tasks]
            )
            local_avg_gap = np.mean([local_task[t]["current_gap"] for t in local_tasks])
            delta_gap = local_avg_gap - global_avg_gap

            f.write(f"**Gap Médio (pos_mean - neg_mean)**:\n")
            f.write(f"- Global: **{global_avg_gap:.4f}**\n")
            f.write(f"- Local: **{local_avg_gap:.4f}**\n")
            f.write(f"- Diferença: **{delta_gap:+.4f}**\n\n")

        f.write("---\n\n")

        # Gap Evolution Table
        f.write("## 1. Evolução do Gap (pos_mean - neg_mean)\n\n")
        f.write(
            "O gap representa a separação entre similaridades positivas e negativas. "
        )
        f.write("**Valores maiores indicam melhor discriminação de features**.\n\n")

        f.write("| Task | Global | Local | Δ (Local-Global) | Vencedor |\n")
        f.write("|------|--------|-------|------------------|----------|\n")

        for task in global_tasks:
            if task in local_tasks:
                global_gap = global_task[task]["current_gap"]
                local_gap = local_task[task]["current_gap"]
                delta = local_gap - global_gap
                winner = (
                    "🟢 Local"
                    if delta > 0
                    else "🔴 Global" if delta < 0 else "➖ Empate"
                )
                f.write(
                    f"| {task} | {global_gap:.4f} | {local_gap:.4f} | {delta:+.4f} | {winner} |\n"
                )

        f.write("\n**Resumo**: ")
        if local_tasks:
            wins_local = sum(
                1
                for t in local_tasks
                if local_task[t]["current_gap"] > global_task[t]["current_gap"]
            )
            wins_global = sum(
                1
                for t in local_tasks
                if local_task[t]["current_gap"] < global_task[t]["current_gap"]
            )
            f.write(
                f"Local vence em {wins_local}/{len(local_tasks)} tasks, Global vence em {wins_global}/{len(local_tasks)} tasks.\n\n"
            )

        f.write("---\n\n")

        # Positive Similarities Table
        f.write("## 2. Similaridades Positivas\n\n")
        f.write("Similaridade de cosseno média entre pares positivos (mesma classe). ")
        f.write("**Valores maiores indicam maior coesão intra-classe**.\n\n")

        f.write("| Task | Global | Local | Δ (Local-Global) | Vencedor |\n")
        f.write("|------|--------|-------|------------------|----------|\n")

        for task in global_tasks:
            if task in local_tasks:
                global_pos = global_task[task]["pos_mean"]
                local_pos = local_task[task]["pos_mean"]
                delta = local_pos - global_pos
                winner = (
                    "🟢 Local"
                    if delta > 0
                    else "🔴 Global" if delta < 0 else "➖ Empate"
                )
                f.write(
                    f"| {task} | {global_pos:.4f} | {local_pos:.4f} | {delta:+.4f} | {winner} |\n"
                )

        f.write("\n**Resumo**: ")
        if local_tasks:
            wins_local = sum(
                1
                for t in local_tasks
                if local_task[t]["pos_mean"] > global_task[t]["pos_mean"]
            )
            wins_global = sum(
                1
                for t in local_tasks
                if local_task[t]["pos_mean"] < global_task[t]["pos_mean"]
            )
            f.write(
                f"Local vence em {wins_local}/{len(local_tasks)} tasks, Global vence em {wins_global}/{len(local_tasks)} tasks.\n\n"
            )

        f.write("---\n\n")

        # Negative Similarities Table
        f.write("## 3. Similaridades Negativas\n\n")
        f.write(
            "Similaridade de cosseno média entre pares negativos (classes diferentes). "
        )
        f.write(
            "**Valores menores (mais negativos) indicam melhor separação inter-classe**.\n\n"
        )

        f.write("| Task | Global | Local | Δ (Local-Global) | Vencedor |\n")
        f.write("|------|--------|-------|------------------|----------|\n")

        for task in global_tasks:
            if task in local_tasks:
                global_neg = global_task[task]["neg_mean"]
                local_neg = local_task[task]["neg_mean"]
                delta = local_neg - global_neg
                winner = (
                    "🟢 Local"
                    if delta < 0
                    else "🔴 Global" if delta > 0 else "➖ Empate"
                )
                f.write(
                    f"| {task} | {global_neg:.4f} | {local_neg:.4f} | {delta:+.4f} | {winner} |\n"
                )

        f.write("\n**Resumo**: ")
        if local_tasks:
            wins_local = sum(
                1
                for t in local_tasks
                if local_task[t]["neg_mean"] < global_task[t]["neg_mean"]
            )
            wins_global = sum(
                1
                for t in local_tasks
                if local_task[t]["neg_mean"] > global_task[t]["neg_mean"]
            )
            f.write(
                f"Local vence em {wins_local}/{len(local_tasks)} tasks (menor é melhor), Global vence em {wins_global}/{len(local_tasks)} tasks.\n\n"
            )

        f.write("---\n\n")

        # NME1 Performance Table
        f.write("## 4. Performance NME1 (Acurácia de Avaliação)\n\n")
        f.write("Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. ")
        f.write(
            "**Valores maiores indicam melhor performance geral e retenção de conhecimento**.\n\n"
        )

        f.write("| Task | Global | Local | Δ (Local-Global) | Vencedor |\n")
        f.write("|------|--------|-------|------------------|----------|\n")

        if global_nme1 and local_nme1:
            for task in sorted(global_nme1.keys()):
                if task in local_nme1:
                    global_score = global_nme1[task] * 100
                    local_score = local_nme1[task] * 100
                    delta = local_score - global_score
                    winner = (
                        "🟢 Local"
                        if delta > 0
                        else "🔴 Global" if delta < 0 else "➖ Empate"
                    )
                    f.write(
                        f"| {task} | {global_score:.2f}% | {local_score:.2f}% | {delta:+.2f}% | {winner} |\n"
                    )

            f.write("\n**Resumo**: ")
            wins_local = sum(
                1
                for t in local_nme1.keys()
                if t in global_nme1 and local_nme1[t] > global_nme1[t]
            )
            wins_global = sum(
                1
                for t in local_nme1.keys()
                if t in global_nme1 and local_nme1[t] < global_nme1[t]
            )
            f.write(
                f"Local vence em {wins_local}/{len(local_nme1)} tasks, Global vence em {wins_global}/{len(local_nme1)} tasks.\n\n"
            )

            # Final performance highlight
            global_final = global_nme1[max(global_nme1.keys())] * 100
            local_final = local_nme1[max(local_nme1.keys())] * 100
            f.write(
                f"**Performance na Tarefa Final**: Global={global_final:.2f}%, Local={local_final:.2f}% ({local_final-global_final:+.2f}%)\n\n"
            )

        f.write("---\n\n")

        # Statistical Summary
        f.write("## Resumo Estatístico\n\n")

        if global_task and local_task:
            f.write("### Estatísticas de Gap\n\n")
            global_gaps = [global_task[t]["current_gap"] for t in global_tasks]
            local_gaps = [local_task[t]["current_gap"] for t in local_tasks]

            f.write(f"- **Global**:\n")
            f.write(f"  - Média: {np.mean(global_gaps):.4f}\n")
            f.write(f"  - Desvio Padrão: {np.std(global_gaps):.4f}\n")
            f.write(f"  - Mínimo: {np.min(global_gaps):.4f}\n")
            f.write(f"  - Máximo: {np.max(global_gaps):.4f}\n\n")

            f.write(f"- **Local**:\n")
            f.write(f"  - Média: {np.mean(local_gaps):.4f}\n")
            f.write(f"  - Desvio Padrão: {np.std(local_gaps):.4f}\n")
            f.write(f"  - Mínimo: {np.min(local_gaps):.4f}\n")
            f.write(f"  - Máximo: {np.max(local_gaps):.4f}\n\n")

            f.write("### Estatísticas de Similaridade Positiva\n\n")
            global_pos = [global_task[t]["pos_mean"] for t in global_tasks]
            local_pos = [local_task[t]["pos_mean"] for t in local_tasks]

            f.write(
                f"- **Global**: Média={np.mean(global_pos):.4f}, Desvio Padrão={np.std(global_pos):.4f}\n"
            )
            f.write(
                f"- **Local**: Média={np.mean(local_pos):.4f}, Desvio Padrão={np.std(local_pos):.4f}\n\n"
            )

            f.write("### Estatísticas de Similaridade Negativa\n\n")
            global_neg = [global_task[t]["neg_mean"] for t in global_tasks]
            local_neg = [local_task[t]["neg_mean"] for t in local_tasks]

            f.write(
                f"- **Global**: Média={np.mean(global_neg):.4f}, Desvio Padrão={np.std(global_neg):.4f}\n"
            )
            f.write(
                f"- **Local**: Média={np.mean(local_neg):.4f}, Desvio Padrão={np.std(local_neg):.4f}\n\n"
            )

        if global_nme1 and local_nme1:
            f.write("### Estatísticas de NME1 Accuracy\n\n")
            global_nme1_vals = [v * 100 for v in global_nme1.values()]
            local_nme1_vals = [v * 100 for v in local_nme1.values()]

            f.write(
                f"- **Global**: Média={np.mean(global_nme1_vals):.2f}%, Desvio Padrão={np.std(global_nme1_vals):.2f}%\n"
            )
            f.write(
                f"- **Local**: Média={np.mean(local_nme1_vals):.2f}%, Desvio Padrão={np.std(local_nme1_vals):.2f}%\n\n"
            )

        f.write("---\n\n")

        # Conclusions
        f.write("## Conclusões\n\n")

        if global_nme1 and local_nme1:
            global_final = global_nme1[max(global_nme1.keys())] * 100
            local_final = local_nme1[max(local_nme1.keys())] * 100
            delta_nme1 = local_final - global_final

            if delta_nme1 > 0.1:
                f.write(
                    f"✅ **Local apresenta melhoria significativa** (+{delta_nme1:.2f}% NME1 final)\n\n"
                )
            elif delta_nme1 < -0.1:
                f.write(
                    f"⚠️ **Global apresenta melhor performance** ({delta_nme1:.2f}% NME1 final)\n\n"
                )
            else:
                f.write(
                    f"➖ **Ambos os experimentos apresentam performance comparável** (Δ={delta_nme1:+.2f}% NME1 final)\n\n"
                )

        if global_task and local_task:
            global_avg_gap = np.mean(
                [global_task[t]["current_gap"] for t in global_tasks]
            )
            local_avg_gap = np.mean([local_task[t]["current_gap"] for t in local_tasks])

            if abs(local_avg_gap - global_avg_gap) < 0.01:
                f.write(
                    "- Os valores de gap são **praticamente idênticos** entre os experimentos, sugerindo separação de features similar.\n"
                )
            elif local_avg_gap > global_avg_gap:
                f.write("- Local alcança **gap ligeiramente melhor** em média.\n")
            else:
                f.write("- Global alcança **gap ligeiramente melhor** em média.\n")

        f.write("\n---\n\n")
        f.write(
            "*Relatório gerado automaticamente por `compare_50-10_global_vs_local.py`*\n"
        )

    print(f"✓ Markdown report saved to: {report_path}")


def generate_report(
    global_task,
    local_task,
    global_nme1,
    local_nme1,
    global_final_nme1,
    local_final_nme1,
    nme1_diff,
    global_avg_gap,
    local_avg_gap,
    gap_diff,
    global_avg_viol,
    local_avg_viol,
    output_dir,
):
    """Legacy function for backward compatibility - calls generate_markdown_report."""
    # This function is kept for compatibility but is deprecated
    pass


def main():
    """Main execution."""
    # Define experiment directories (relative to script location)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent / "logs"

    global_dir = base_dir / "exp_cifar100_50-10_antB0_nceA1_antM0_antGlobal"
    local_dir = base_dir / "exp_cifar100_50-10_antB0_nceA1_antM0_antLocal"

    # Output directory
    output_dir = script_dir.parent / "results" / "cifar100_50-10_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CIFAR-100 50-10: Global vs Local Anchor Comparison")
    print("=" * 80)
    print(f"\nGlobal experiment: {global_dir}")
    print(f"Local experiment:  {local_dir}")
    print(f"Output directory:  {output_dir}\n")

    # Parse data
    print("Parsing ANT statistics...")
    global_stats = parse_ant_stats(global_dir / "exp_matrix_debug0.log")
    local_stats = parse_ant_stats(local_dir / "exp_matrix_debug0.log")
    print(f"  Global: {len(global_stats)} records")
    print(f"  Local:  {len(local_stats)} records")

    print("\nParsing NME1 scores...")
    global_nme1 = parse_nme1_scores(global_dir / "exp_gistlog.log")
    local_nme1 = parse_nme1_scores(local_dir / "exp_gistlog.log")
    print(f"  Global: {len(global_nme1)} tasks")
    print(f"  Local:  {len(local_nme1)} tasks")

    # Generate comparison plot and report
    print("\nGenerating comparison visualizations...")
    plot_comparison(global_stats, local_stats, global_nme1, local_nme1, output_dir)

    print("\n" + "=" * 80)
    print("✓ Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
