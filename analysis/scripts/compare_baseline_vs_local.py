#!/usr/bin/env python3
"""
Compare InfoNCE with Local Anchor across different experiments.

Analyzes:
1. Gap evolution (pos_mean - neg_mean)
2. Distance statistics (pos/neg mean, std, min, max)
3. Violation percentage trends
4. NME1 performance across tasks

Usage:
    python compare_baseline_vs_local.py
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


def plot_comparison(baseline_stats, local_stats, baseline_nme1, local_nme1, output_dir):
    """Create comprehensive comparison plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract tasks
    baseline_tasks = sorted(baseline_stats.keys())
    local_tasks = sorted(local_stats.keys()) if local_stats else []
    all_tasks = sorted(set(baseline_tasks) | set(local_tasks))

    # Helper function to plot with winner highlighting
    def plot_with_winners(
        ax,
        baseline_tasks,
        baseline_vals,
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
            baseline_tasks,
            baseline_vals,
            "--",
            color="#3498db",
            linewidth=2,
            alpha=0.5,
            zorder=1,
        )

        # Plot markers with highlighting for winners and ties
        if local_tasks and len(local_vals) == len(baseline_vals):
            for i, task in enumerate(local_tasks):
                delta = local_vals[i] - baseline_vals[i]

                # Determine threshold for "tie" based on metric scale
                tie_threshold = 0.001 if abs(baseline_vals[i]) < 1.0 else 0.01

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
                        baseline_vals[i],
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
                    ax.plot(
                        task,
                        local_vals[i],
                        "o",
                        color="#2ecc71",
                        markersize=12,
                        markeredgewidth=2,
                        markeredgecolor="#27ae60",
                        zorder=4,
                    )
                    ax.plot(
                        task,
                        baseline_vals[i],
                        "s",
                        color="#3498db",
                        markersize=7,
                        alpha=0.5,
                        zorder=3,
                    )
                else:  # Baseline is better
                    ax.plot(
                        task,
                        baseline_vals[i],
                        "s",
                        color="#3498db",
                        markersize=12,
                        markeredgewidth=2,
                        markeredgecolor="#2980b9",
                        zorder=4,
                    )
                    ax.plot(
                        task,
                        local_vals[i],
                        "o",
                        color="#2ecc71",
                        markersize=7,
                        alpha=0.5,
                        zorder=3,
                    )

                # Add delta label if requested
                if show_deltas and abs(delta) >= tie_threshold:
                    # Position label above the higher point
                    y_pos = max(local_vals[i], baseline_vals[i])
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_offset = y_range * 0.05  # 5% offset for better spacing

                    # Format delta based on value magnitude
                    if abs(delta) < 0.01:
                        delta_text = f"Δ={delta:+.4f}"
                    elif abs(delta) < 1:
                        delta_text = f"Δ={delta:+.3f}"
                    else:
                        delta_text = f"Δ={delta:+.1f}"

                    # Color based on winner
                    text_color = (
                        "#27ae60"
                        if (
                            (higher_is_better and delta > 0)
                            or (not higher_is_better and delta < 0)
                        )
                        else "#2980b9"
                    )

                    ax.text(
                        task,
                        y_pos + y_offset,
                        delta_text,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor=text_color,
                            alpha=0.8,
                            linewidth=1,
                        ),
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
                label="Âncora Local",
            ),
            Line2D(
                [0],
                [0],
                color="#3498db",
                marker="s",
                linestyle="--",
                linewidth=2,
                markersize=7,
                label="Baseline",
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
        ax.grid(True, alpha=0.3)  # Set style

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 10)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "InfoNCE + Local Anchor: Experiment Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Gap Evolution (pos_mean - neg_mean)
    ax = axes[0, 0]
    baseline_gaps = [baseline_stats[t]["current_gap"] for t in baseline_tasks]
    local_gaps = (
        [local_stats[t]["current_gap"] for t in local_tasks] if local_tasks else []
    )
    plot_with_winners(
        ax,
        baseline_tasks,
        baseline_gaps,
        local_tasks,
        local_gaps,
        higher_is_better=True,
        ylabel="Gap (pos_mean - neg_mean)",
        title="Gap Evolution Across Tasks",
    )

    # 2. Positive Similarities
    ax = axes[0, 1]
    baseline_pos = [baseline_stats[t]["pos_mean"] for t in baseline_tasks]
    local_pos = [local_stats[t]["pos_mean"] for t in local_tasks] if local_tasks else []
    plot_with_winners(
        ax,
        baseline_tasks,
        baseline_pos,
        local_tasks,
        local_pos,
        higher_is_better=True,
        ylabel="Positive Mean Similarity",
        title="Positive Similarities",
    )

    # 3. Negative Similarities
    ax = axes[1, 0]
    baseline_neg = [baseline_stats[t]["neg_mean"] for t in baseline_tasks]
    local_neg = [local_stats[t]["neg_mean"] for t in local_tasks] if local_tasks else []
    plot_with_winners(
        ax,
        baseline_tasks,
        baseline_neg,
        local_tasks,
        local_neg,
        higher_is_better=False,  # Lower (more negative) is better
        ylabel="Negative Mean Similarity",
        title="Negative Similarities",
    )

    # 4. NME1 Performance
    ax = axes[1, 1]
    if baseline_nme1:
        baseline_nme1_tasks = sorted(baseline_nme1.keys())
        baseline_nme1_scores = [baseline_nme1[t] for t in baseline_nme1_tasks]
    else:
        baseline_nme1_tasks = []
        baseline_nme1_scores = []

    if local_nme1:
        local_nme1_tasks = sorted(local_nme1.keys())
        local_nme1_scores = [local_nme1[t] for t in local_nme1_tasks]
    else:
        local_nme1_tasks = []
        local_nme1_scores = []

    plot_with_winners(
        ax,
        baseline_nme1_tasks,
        baseline_nme1_scores,
        local_nme1_tasks,
        local_nme1_scores,
        higher_is_better=True,
        ylabel="NME1 Accuracy (%)",
        title="Evaluation Performance",
    )
    ax.set_ylim([55, 100])

    plt.tight_layout()
    plt.savefig(
        output_dir / "baseline_vs_local_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"\n✓ Comparison plot saved to: {output_dir / 'baseline_vs_local_comparison.png'}"
    )

    # Generate markdown report
    generate_markdown_report(
        baseline_stats, local_stats, baseline_nme1, local_nme1, output_dir
    )


def print_summary_table(baseline_stats, local_stats, baseline_nme1, local_nme1):
    """Print detailed comparison table."""


def generate_markdown_report(
    baseline_stats, local_stats, baseline_nme1, local_nme1, output_dir
):
    """Generate a comprehensive markdown report with comparison tables."""

    output_dir = Path(output_dir)
    report_path = output_dir / "comparison_report.md"

    baseline_tasks = sorted(baseline_stats.keys())
    local_tasks = sorted(local_stats.keys()) if local_stats else []

    with open(report_path, "w") as f:
        # Header
        f.write("# InfoNCE + Local Anchor: Comparação de Experimentos\n\n")
        f.write(
            f"**Gerado em**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Sumário Executivo\n\n")

        if baseline_nme1 and local_nme1:
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            local_final = local_nme1[max(local_nme1.keys())]
            delta_nme1 = local_final - baseline_final

            f.write(f"**NME1 Accuracy Final**:\n")
            f.write(f"- Baseline: **{baseline_final:.2f}%**\n")
            f.write(f"- Âncora Local: **{local_final:.2f}%**\n")
            f.write(f"- Melhoria: **{delta_nme1:+.2f}%** ")
            if delta_nme1 > 0:
                f.write("✅ (Âncora Local é melhor)\n\n")
            elif delta_nme1 < 0:
                f.write("⚠️ (Baseline é melhor)\n\n")
            else:
                f.write("(Empate)\n\n")

        if baseline_stats and local_stats:
            baseline_avg_gap = np.mean(
                [baseline_stats[t]["current_gap"] for t in baseline_tasks]
            )
            local_avg_gap = np.mean(
                [local_stats[t]["current_gap"] for t in local_tasks]
            )
            delta_gap = local_avg_gap - baseline_avg_gap

            f.write(f"**Gap Médio (pos_mean - neg_mean)**:\n")
            f.write(f"- Baseline: **{baseline_avg_gap:.4f}**\n")
            f.write(f"- Âncora Local: **{local_avg_gap:.4f}**\n")
            f.write(f"- Diferença: **{delta_gap:+.4f}**\n\n")

        f.write("---\n\n")

        # Gap Evolution Table
        f.write("## 1. Evolução do Gap (pos_mean - neg_mean)\n\n")
        f.write(
            "O gap representa a separação entre similaridades positivas e negativas. "
        )
        f.write("**Valores maiores indicam melhor discriminação de features**.\n\n")

        f.write("| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |\n")
        f.write("|------|----------|--------------|----------------|----------|\n")

        for task in baseline_tasks:
            if task in local_tasks:
                prev_gap = baseline_stats[task]["current_gap"]
                new_gap = local_stats[task]["current_gap"]
                delta = new_gap - prev_gap
                winner = (
                    "🟢 Local" if delta > 0 else "🔵 Base" if delta < 0 else "➖ Empate"
                )
                f.write(
                    f"| {task} | {prev_gap:.4f} | {new_gap:.4f} | {delta:+.4f} | {winner} |\n"
                )

        f.write("\n**Resumo**: ")
        if local_tasks:
            wins_new = sum(
                1
                for t in local_tasks
                if local_stats[t]["current_gap"] > baseline_stats[t]["current_gap"]
            )
            wins_prev = sum(
                1
                for t in local_tasks
                if local_stats[t]["current_gap"] < baseline_stats[t]["current_gap"]
            )
            f.write(
                f"Âncora Local vence em {wins_new}/{len(local_tasks)} tasks, Baseline vence em {wins_prev}/{len(local_tasks)} tasks.\n\n"
            )

        f.write("---\n\n")

        # Positive Similarities Table
        f.write("## 2. Similaridades Positivas\n\n")
        f.write("Similaridade de cosseno média entre pares positivos (mesma classe). ")
        f.write("**Valores maiores indicam maior coesão intra-classe**.\n\n")

        f.write("| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |\n")
        f.write("|------|----------|--------------|----------------|----------|\n")

        for task in baseline_tasks:
            if task in local_tasks:
                prev_pos = baseline_stats[task]["pos_mean"]
                new_pos = local_stats[task]["pos_mean"]
                delta = new_pos - prev_pos
                winner = (
                    "🟢 Local" if delta > 0 else "🔵 Base" if delta < 0 else "➖ Empate"
                )
                f.write(
                    f"| {task} | {prev_pos:.4f} | {new_pos:.4f} | {delta:+.4f} | {winner} |\n"
                )

        f.write("\n**Resumo**: ")
        if local_tasks:
            wins_new = sum(
                1
                for t in local_tasks
                if local_stats[t]["pos_mean"] > baseline_stats[t]["pos_mean"]
            )
            wins_prev = sum(
                1
                for t in local_tasks
                if local_stats[t]["pos_mean"] < baseline_stats[t]["pos_mean"]
            )
            f.write(
                f"Âncora Local vence em {wins_new}/{len(local_tasks)} tasks, Baseline vence em {wins_prev}/{len(local_tasks)} tasks.\n\n"
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

        f.write("| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |\n")
        f.write("|------|----------|--------------|----------------|----------|\n")

        for task in baseline_tasks:
            if task in local_tasks:
                prev_neg = baseline_stats[task]["neg_mean"]
                new_neg = local_stats[task]["neg_mean"]
                delta = new_neg - prev_neg
                winner = (
                    "🟢 Local" if delta < 0 else "🔵 Base" if delta > 0 else "➖ Empate"
                )
                f.write(
                    f"| {task} | {prev_neg:.4f} | {new_neg:.4f} | {delta:+.4f} | {winner} |\n"
                )

        f.write("\n**Resumo**: ")
        if local_tasks:
            wins_new = sum(
                1
                for t in local_tasks
                if local_stats[t]["neg_mean"] < baseline_stats[t]["neg_mean"]
            )
            wins_prev = sum(
                1
                for t in local_tasks
                if local_stats[t]["neg_mean"] > baseline_stats[t]["neg_mean"]
            )
            f.write(
                f"Âncora Local vence em {wins_new}/{len(local_tasks)} tasks (menor é melhor), Baseline vence em {wins_prev}/{len(local_tasks)} tasks.\n\n"
            )

        f.write("---\n\n")

        # NME1 Performance Table
        f.write("## 4. Performance NME1 (Acurácia de Avaliação)\n\n")
        f.write("Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. ")
        f.write(
            "**Valores maiores indicam melhor performance geral e retenção de conhecimento**.\n\n"
        )

        f.write("| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |\n")
        f.write("|------|----------|--------------|----------------|----------|\n")

        if baseline_nme1 and local_nme1:
            for task in sorted(baseline_nme1.keys()):
                if task in local_nme1:
                    prev_nme1 = baseline_nme1[task]
                    new_nme1 = local_nme1[task]
                    delta = new_nme1 - prev_nme1
                    winner = (
                        "🟢 Local"
                        if delta > 0
                        else "🔵 Base" if delta < 0 else "➖ Empate"
                    )
                    f.write(
                        f"| {task} | {prev_nme1:.2f}% | {new_nme1:.2f}% | {delta:+.2f}% | {winner} |\n"
                    )

            f.write("\n**Resumo**: ")
            wins_new = sum(
                1
                for t in local_nme1.keys()
                if t in baseline_nme1 and local_nme1[t] > baseline_nme1[t]
            )
            wins_prev = sum(
                1
                for t in local_nme1.keys()
                if t in baseline_nme1 and local_nme1[t] < baseline_nme1[t]
            )
            f.write(
                f"Âncora Local vence em {wins_new}/{len(local_nme1)} tasks, Baseline vence em {wins_prev}/{len(local_nme1)} tasks.\n\n"
            )

            # Final performance highlight
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            local_final = local_nme1[max(local_nme1.keys())]
            f.write(
                f"**Performance na Tarefa Final**: Baseline={baseline_final:.2f}%, Âncora Local={local_final:.2f}% ({local_final-baseline_final:+.2f}%)\n\n"
            )

        f.write("---\n\n")

        # Statistical Summary
        f.write("## Resumo Estatístico\n\n")

        if baseline_stats and local_stats:
            f.write("### Estatísticas de Gap\n\n")
            baseline_gaps = [baseline_stats[t]["current_gap"] for t in baseline_tasks]
            local_gaps = [local_stats[t]["current_gap"] for t in local_tasks]

            f.write(f"- **Baseline**:\n")
            f.write(f"  - Média: {np.mean(baseline_gaps):.4f}\n")
            f.write(f"  - Desvio Padrão: {np.std(baseline_gaps):.4f}\n")
            f.write(f"  - Mínimo: {np.min(baseline_gaps):.4f}\n")
            f.write(f"  - Máximo: {np.max(baseline_gaps):.4f}\n\n")

            f.write(f"- **Âncora Local**:\n")
            f.write(f"  - Média: {np.mean(local_gaps):.4f}\n")
            f.write(f"  - Desvio Padrão: {np.std(local_gaps):.4f}\n")
            f.write(f"  - Mínimo: {np.min(local_gaps):.4f}\n")
            f.write(f"  - Máximo: {np.max(local_gaps):.4f}\n\n")

            f.write("### Estatísticas de Similaridade Positiva\n\n")
            baseline_pos = [baseline_stats[t]["pos_mean"] for t in baseline_tasks]
            local_pos = [local_stats[t]["pos_mean"] for t in local_tasks]

            f.write(
                f"- **Baseline**: Média={np.mean(baseline_pos):.4f}, Desvio Padrão={np.std(baseline_pos):.4f}\n"
            )
            f.write(
                f"- **Âncora Local**: Média={np.mean(local_pos):.4f}, Desvio Padrão={np.std(local_pos):.4f}\n\n"
            )

            f.write("### Estatísticas de Similaridade Negativa\n\n")
            baseline_neg = [baseline_stats[t]["neg_mean"] for t in baseline_tasks]
            local_neg = [local_stats[t]["neg_mean"] for t in local_tasks]

            f.write(
                f"- **Baseline**: Média={np.mean(baseline_neg):.4f}, Desvio Padrão={np.std(baseline_neg):.4f}\n"
            )
            f.write(
                f"- **Âncora Local**: Média={np.mean(local_neg):.4f}, Desvio Padrão={np.std(local_neg):.4f}\n\n"
            )

        if baseline_nme1 and local_nme1:
            f.write("### Estatísticas de NME1 Accuracy\n\n")
            baseline_nme1_vals = list(baseline_nme1.values())
            local_nme1_vals = list(local_nme1.values())

            f.write(
                f"- **Baseline**: Média={np.mean(baseline_nme1_vals):.2f}%, Desvio Padrão={np.std(baseline_nme1_vals):.2f}%\n"
            )
            f.write(
                f"- **Âncora Local**: Média={np.mean(local_nme1_vals):.2f}%, Desvio Padrão={np.std(local_nme1_vals):.2f}%\n\n"
            )

        f.write("---\n\n")

        # Conclusions
        f.write("## Conclusões\n\n")

        if baseline_nme1 and local_nme1:
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            local_final = local_nme1[max(local_nme1.keys())]
            delta_nme1 = local_final - baseline_final

            if delta_nme1 > 0.1:
                f.write(
                    f"✅ **Âncora Local apresenta melhoria significativa** (+{delta_nme1:.2f}% NME1 final)\n\n"
                )
            elif delta_nme1 < -0.1:
                f.write(
                    f"⚠️ **Baseline apresenta melhor performance** ({delta_nme1:.2f}% NME1 final)\n\n"
                )
            else:
                f.write(
                    f"➖ **Ambos os experimentos apresentam performance comparável** (Δ={delta_nme1:+.2f}% NME1 final)\n\n"
                )

        if baseline_stats and local_stats:
            baseline_avg_gap = np.mean(
                [baseline_stats[t]["current_gap"] for t in baseline_tasks]
            )
            local_avg_gap = np.mean(
                [local_stats[t]["current_gap"] for t in local_tasks]
            )

            if abs(local_avg_gap - baseline_avg_gap) < 0.01:
                f.write(
                    "- Os valores de gap são **praticamente idênticos** entre os experimentos, sugerindo separação de features similar.\n"
                )
            elif local_avg_gap > baseline_avg_gap:
                f.write(
                    "- Âncora Local alcança **gap ligeiramente melhor** em média.\n"
                )
            else:
                f.write("- Baseline alcança **gap ligeiramente melhor** em média.\n")

        f.write("\n---\n\n")
        f.write(
            "*Relatório gerado automaticamente por `compare_baseline_vs_local.py`*\n"
        )

    print(f"✓ Markdown report saved to: {report_path}")


def print_summary_table(baseline_stats, local_stats, baseline_nme1, local_nme1):
    """Print detailed comparison table."""

    print("\n" + "=" * 100)
    print("INFONCE + LOCAL ANCHOR: EXPERIMENT COMPARISON")
    print("=" * 100)

    baseline_tasks = sorted(baseline_stats.keys())
    local_tasks = sorted(local_stats.keys()) if local_stats else []
    all_tasks = sorted(set(baseline_tasks) | set(local_tasks))

    # Header
    print(
        f"\n{'Task':<6} {'Metric':<20} {'Previous':<15} {'New':<15} {'Δ (New-Prev)':<15} {'Winner':<10}"
    )
    print("-" * 100)

    metrics = [
        ("Gap", "current_gap", "{:.4f}"),
        ("Pos Mean", "pos_mean", "{:.4f}"),
        ("Neg Mean", "neg_mean", "{:.4f}"),
        ("Violation %", "violation_pct", "{:.2f}%"),
        ("NME1", "nme1", "{:.2f}%"),
    ]

    for task in all_tasks:
        baseline_data = baseline_stats.get(task, {})
        local_data = local_stats.get(task, {}) if local_stats else {}

        for i, (label, key, fmt) in enumerate(metrics):
            if key == "nme1":
                baseline_val = baseline_nme1.get(task)
                local_val = local_nme1.get(task) if local_nme1 else None
            else:
                baseline_val = baseline_data.get(key)
                local_val = local_data.get(key)

            baseline_str = (
                fmt.format(baseline_val) if baseline_val is not None else "N/A"
            )
            local_str = fmt.format(local_val) if local_val is not None else "N/A"

            # Calculate delta and winner
            if baseline_val is not None and local_val is not None:
                delta = local_val - baseline_val
                if key == "nme1" or key == "current_gap" or key == "pos_mean":
                    # Higher is better
                    winner = "New ✓" if delta > 0 else "Prev ✓" if delta < 0 else "Tie"
                elif key == "neg_mean":
                    # Lower is better (more negative)
                    winner = "New ✓" if delta < 0 else "Prev ✓" if delta > 0 else "Tie"
                elif key == "violation_pct":
                    # Lower is better
                    winner = "New ✓" if delta < 0 else "Prev ✓" if delta > 0 else "Tie"
                else:
                    winner = ""

                if "%" in fmt:
                    delta_str = f"{delta:+.2f}%"
                else:
                    delta_str = f"{delta:+.4f}"
            else:
                delta_str = "N/A"
                winner = ""

            task_str = f"{task}" if i == 0 else ""
            print(
                f"{task_str:<6} {label:<20} {baseline_str:<15} {local_str:<15} {delta_str:<15} {winner:<10}"
            )

        print("-" * 100)

    # Overall statistics
    print("\nOVERALL STATISTICS")
    print("=" * 100)

    if local_stats and baseline_stats:
        # Average gaps
        baseline_avg_gap = np.mean(
            [baseline_stats[t]["current_gap"] for t in baseline_tasks]
        )
        local_avg_gap = np.mean([local_stats[t]["current_gap"] for t in local_tasks])
        print(
            f"Average Gap:          Previous={baseline_avg_gap:.4f}  |  New={local_avg_gap:.4f}  |  Δ={local_avg_gap-baseline_avg_gap:+.4f}"
        )

        # Average violations
        baseline_avg_viol = np.mean(
            [baseline_stats[t]["violation_pct"] for t in baseline_tasks]
        )
        local_avg_viol = np.mean([local_stats[t]["violation_pct"] for t in local_tasks])
        print(
            f"Average Violations:   Previous={baseline_avg_viol:.2f}%  |  New={local_avg_viol:.2f}%  |  Δ={local_avg_viol-baseline_avg_viol:+.2f}%"
        )

    if baseline_nme1 and local_nme1:
        baseline_final = baseline_nme1[max(baseline_nme1.keys())]
        local_final = local_nme1[max(local_nme1.keys())] if local_nme1 else None
        if local_final:
            print(
                f"Final NME1:           Previous={baseline_final:.2f}%  |  New={local_final:.2f}%  |  Δ={local_final-baseline_final:+.2f}%"
            )

    print("=" * 100)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two experiments with InfoNCE analysis"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="../../logs/done_exp_cifar100_10-10_baseline_tagfex_original",
        help="Path to baseline experiment directory",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="../../logs/done_exp_cifar100_10-10_antB0_nceA1_antM0_antLocal",
        help="Path to experiment directory to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/baseline_vs_local",
        help="Output directory for results",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="InfoNCE Experiment Comparison",
        help="Title for the comparison",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline",
        help="Label for baseline experiment",
    )
    parser.add_argument(
        "--experiment-label",
        type=str,
        default="Âncora Local",
        help="Label for experiment being compared",
    )

    args = parser.parse_args()

    # Paths
    baseline_dir = Path(args.baseline)
    local_dir = Path(args.experiment)
    output_dir = Path(args.output)

    print("=" * 80)
    print(args.title)
    print("=" * 80)
    print(f"\n{args.baseline_label}: {baseline_dir}")
    print(f"{args.experiment_label}: {local_dir}")
    print(f"Output:             {output_dir}\n")

    # Parse baseline data
    print("\n[1/4] Parsing baseline logs...")
    baseline_debug = baseline_dir / "exp_debug0.log"
    baseline_gist = baseline_dir / "exp_gistlog.log"

    if not baseline_debug.exists():
        print(f"❌ Error: {baseline_debug} not found")
        return

    baseline_data = parse_ant_stats(baseline_debug)
    baseline_stats = aggregate_by_task(baseline_data)
    baseline_nme1 = parse_nme1_scores(baseline_gist) if baseline_gist.exists() else {}
    print(f"   Parsed {len(baseline_data)} entries, {len(baseline_stats)} tasks")

    # Parse local anchor data (if available)
    print(f"\n[2/4] Parsing {args.experiment_label.lower()} logs...")
    local_debug = local_dir / "exp_debug0.log"
    local_gist = local_dir / "exp_gistlog.log"

    if local_debug.exists():
        local_data = parse_ant_stats(local_debug)
        local_stats = aggregate_by_task(local_data)
        local_nme1 = parse_nme1_scores(local_gist) if local_gist.exists() else {}
        print(f"   Parsed {len(local_data)} entries, {len(local_stats)} tasks")
    else:
        print(f"   ⚠ {args.experiment_label} logs not found yet: {local_debug}")
        print(f"   Will show baseline data only")
        local_stats = {}
        local_nme1 = {}

    # Generate plots
    print("\n[3/4] Generating comparison plots...")
    plot_comparison(baseline_stats, local_stats, baseline_nme1, local_nme1, output_dir)

    # Print summary
    print("\n[4/4] Generating summary table...")
    print_summary_table(baseline_stats, local_stats, baseline_nme1, local_nme1)

    print(f"\n{'='*80}")
    print(f"✓ Analysis complete! Results saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
