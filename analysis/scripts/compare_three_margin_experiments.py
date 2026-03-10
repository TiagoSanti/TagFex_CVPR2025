#!/usr/bin/env python3
"""
Compare three experiments with different ANT margins:
- Baseline (antM=0.1)
- antM=0.3
- antM=0.5

Analyzes:
1. Gap evolution (pos_mean - neg_mean or gap_mean)
2. Distance statistics (gap mean, std, min, max)
3. Violation percentage trends
4. NME1 performance across tasks

Usage:
    python compare_three_margin_experiments.py
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd


def parse_ant_stats_baseline(log_file):
    """Parse ANT distance stats from baseline logs (with pos_mean and neg_mean)."""
    data = []
    # Pattern for logs with pos_mean and neg_mean (old format - all in one line)
    # Note: current_gap in these logs appears to be always 0.0 (gap loss disabled in baseline)
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) \| pos_std: ([\d.]+) \| pos_min: ([\d.]+) \| pos_max: ([\d.]+) \| neg_mean: ([-\d.]+) \| neg_std: ([\d.]+) \| neg_min: ([-\d.]+) \| neg_max: ([-\d.]+) \| gap_mean: ([-\d.]+) \| gap_std: ([\d.]+) \| gap_min: ([-\d.]+) \| gap_max: ([-\d.]+) \| margin: ([\d.]+) \| violation_pct: ([\d.]+)%"

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                task = int(match.group(1))
                epoch = int(match.group(2))
                batch = int(match.group(3))
                pos_mean = float(match.group(4))
                neg_mean = float(match.group(8))
                # Calculate current_gap from pos_mean - neg_mean
                current_gap = pos_mean - neg_mean

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
                        "gap_mean": float(
                            match.group(12)
                        ),  # Gap from margin (ANT-specific)
                        "gap_std": float(match.group(13)),
                        "gap_min": float(match.group(14)),
                        "gap_max": float(match.group(15)),
                        "violation_pct": float(match.group(17)),
                        "current_gap": current_gap,  # Natural gap (pos_mean - neg_mean)
                    }
                )

    return data


def parse_ant_stats_new(log_file):
    """Parse ANT distance stats from new logs (simplified format without Contrastive stats)."""
    data = []

    # Pattern for ANT distance stats (contains gap_mean, violation_pct, etc.)
    # Note: gap_mean here is the distance from margin, NOT pos_mean - neg_mean
    ant_pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_min: ([-\d.]+) \| pos_max: ([\d.]+) \| neg_min: ([-\d.]+) \| neg_max: ([\d.]+) \| gap_mean: ([-\d.]+) \| gap_std: ([\d.]+) \| gap_min: ([-\d.]+) \| gap_max: ([-\d.]+) \| margin: ([\d.]+) \| violation_pct: ([\d.]+)%"

    with open(log_file, "r") as f:
        for line in f:
            match = re.search(ant_pattern, line)
            if match:
                task = int(match.group(1))
                epoch = int(match.group(2))
                batch = int(match.group(3))

                pos_min = float(match.group(4))
                pos_max = float(match.group(5))
                neg_min = float(match.group(6))
                neg_max = float(match.group(7))
                gap_mean = float(match.group(8))  # Gap from margin (ANT-specific)
                gap_std = float(match.group(9))
                gap_min = float(match.group(10))
                gap_max = float(match.group(11))
                margin = float(match.group(12))
                violation_pct = float(match.group(13))

                # Estimate pos_mean and neg_mean from min/max
                # This is not perfect but gives a rough approximation
                pos_mean_approx = (pos_min + pos_max) / 2
                neg_mean_approx = (neg_min + neg_max) / 2

                data.append(
                    {
                        "task": task,
                        "epoch": epoch,
                        "batch": batch,
                        "pos_mean": pos_mean_approx,  # Approximation
                        "pos_min": pos_min,
                        "pos_max": pos_max,
                        "neg_mean": neg_mean_approx,  # Approximation
                        "neg_min": neg_min,
                        "neg_max": neg_max,
                        "gap_mean": gap_mean,  # Gap from margin (ANT-specific)
                        "gap_std": gap_std,
                        "gap_min": gap_min,
                        "gap_max": gap_max,
                        "violation_pct": violation_pct,
                        "current_gap": pos_mean_approx
                        - neg_mean_approx,  # Approximated gap
                        "margin": margin,
                    }
                )

    return data


def parse_nme1_scores(log_file):
    """Parse NME1 evaluation scores from exp_gistlog.log."""
    scores = {}
    pattern = r"nme1_curve \[([\d.\s]+)\]"

    with open(log_file, "r") as f:
        content = f.read()
        matches = re.findall(pattern, content)

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

    task_stats = {}
    for task in sorted(df["task"].unique()):
        task_data = df[df["task"] == task]
        task_stats[task] = {
            "pos_mean": task_data["pos_mean"].mean(),
            "neg_mean": task_data["neg_mean"].mean(),
            "current_gap": task_data["current_gap"].mean(),
            "gap_mean": task_data["gap_mean"].mean(),
            "gap_std": task_data["gap_std"].mean(),
            "violation_pct": task_data["violation_pct"].mean(),
        }

    return task_stats


def plot_comparison(
    baseline_stats,
    exp03_stats,
    exp05_stats,
    baseline_nme1,
    exp03_nme1,
    exp05_nme1,
    output_dir,
):
    """Create comprehensive comparison plots for three experiments."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract tasks
    baseline_tasks = sorted(baseline_stats.keys())
    exp03_tasks = sorted(exp03_stats.keys()) if exp03_stats else []
    exp05_tasks = sorted(exp05_stats.keys()) if exp05_stats else []
    all_tasks = sorted(set(baseline_tasks) | set(exp03_tasks) | set(exp05_tasks))

    # Helper function to determine winner and plot
    def plot_three_way_comparison(
        ax,
        tasks,
        baseline_vals,
        exp03_vals,
        exp05_vals,
        higher_is_better=True,
        ylabel="",
        title="",
        show_deltas=False,
    ):
        """Plot comparison for three experiments with winner highlighting."""
        from matplotlib.lines import Line2D

        # Define colors
        colors = {
            "baseline": "#3498db",  # Blue
            "exp03": "#e74c3c",  # Red
            "exp05": "#2ecc71",  # Green
        }

        # Plot lines
        if baseline_vals:
            ax.plot(
                tasks,
                baseline_vals,
                "-",
                color=colors["baseline"],
                linewidth=2,
                alpha=0.6,
                zorder=1,
            )
        if exp03_vals:
            ax.plot(
                tasks,
                exp03_vals,
                "-",
                color=colors["exp03"],
                linewidth=2,
                alpha=0.6,
                zorder=2,
            )
        if exp05_vals:
            ax.plot(
                tasks,
                exp05_vals,
                "-",
                color=colors["exp05"],
                linewidth=2,
                alpha=0.6,
                zorder=3,
            )

        # Plot markers with highlighting for winner and ties
        if baseline_vals and exp03_vals and exp05_vals:
            for i, task in enumerate(tasks):
                vals = [baseline_vals[i], exp03_vals[i], exp05_vals[i]]
                labels = ["baseline", "exp03", "exp05"]

                # Check if all values are exactly equal (tie)
                is_tie = vals[0] == vals[1] == vals[2]

                if is_tie:
                    # All tied - plot all in gray
                    for j, (val, label) in enumerate(zip(vals, labels)):
                        marker = (
                            "o"
                            if label == "baseline"
                            else "s" if label == "exp03" else "^"
                        )
                        ax.plot(
                            task,
                            val,
                            marker,
                            color="#95a5a6",
                            markersize=10,
                            markeredgewidth=2,
                            markeredgecolor="#7f8c8d",
                            zorder=4,
                        )
                else:
                    # Determine winner
                    if higher_is_better:
                        winner_idx = np.argmax(vals)
                    else:
                        winner_idx = np.argmin(vals)

                    winner = labels[winner_idx]
                    winner_val = vals[winner_idx]
                    baseline_val = vals[0]

                    # Plot all markers
                    for j, (val, label) in enumerate(zip(vals, labels)):
                        marker = (
                            "o"
                            if label == "baseline"
                            else "s" if label == "exp03" else "^"
                        )
                        if label == winner:
                            # Highlight winner
                            ax.plot(
                                task,
                                val,
                                marker,
                                color=colors[label],
                                markersize=12,
                                markeredgewidth=2.5,
                                markeredgecolor=colors[label],
                                zorder=5,
                            )
                        else:
                            # Non-winner
                            ax.plot(
                                task,
                                val,
                                marker,
                                color=colors[label],
                                markersize=8,
                                alpha=0.5,
                                zorder=4,
                            )

                    # Add delta label if winner is not baseline
                    if show_deltas and winner != "baseline":
                        delta = winner_val - baseline_val

                        # Position label above the highest point
                        y_pos = max(vals)
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
                        text_color = colors[winner]

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
                color=colors["baseline"],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=8,
                label="Baseline (M=0.1)",
            ),
            Line2D(
                [0],
                [0],
                color=colors["exp03"],
                marker="s",
                linestyle="-",
                linewidth=2,
                markersize=8,
                label="antM=0.3",
            ),
            Line2D(
                [0],
                [0],
                color=colors["exp05"],
                marker="^",
                linestyle="-",
                linewidth=2,
                markersize=8,
                label="antM=0.5",
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

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 10)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Comparação de Margens ANT: Baseline (M=0.1) vs antM=0.3 vs antM=0.5",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Gap Evolution
    ax = axes[0, 0]
    baseline_gaps = [baseline_stats[t]["current_gap"] for t in baseline_tasks]
    exp03_gaps = (
        [exp03_stats[t]["current_gap"] for t in exp03_tasks] if exp03_tasks else []
    )
    exp05_gaps = (
        [exp05_stats[t]["current_gap"] for t in exp05_tasks] if exp05_tasks else []
    )
    plot_three_way_comparison(
        ax,
        baseline_tasks,
        baseline_gaps,
        exp03_gaps,
        exp05_gaps,
        higher_is_better=True,
        ylabel="Gap (pos_mean - neg_mean)",
        title="Evolução do Gap Entre Similaridades",
        show_deltas=True,
    )

    # 2. Violation Percentage
    ax = axes[0, 1]
    baseline_viol = [baseline_stats[t]["violation_pct"] for t in baseline_tasks]
    exp03_viol = (
        [exp03_stats[t]["violation_pct"] for t in exp03_tasks] if exp03_tasks else []
    )
    exp05_viol = (
        [exp05_stats[t]["violation_pct"] for t in exp05_tasks] if exp05_tasks else []
    )
    plot_three_way_comparison(
        ax,
        baseline_tasks,
        baseline_viol,
        exp03_viol,
        exp05_viol,
        higher_is_better=False,
        ylabel="Violation Percentage (%)",
        title="Percentual de Violações da Margem",
        show_deltas=True,
    )

    # 3. Gap Standard Deviation
    ax = axes[1, 0]
    baseline_gap_std = [baseline_stats[t]["gap_std"] for t in baseline_tasks]
    exp03_gap_std = (
        [exp03_stats[t]["gap_std"] for t in exp03_tasks] if exp03_tasks else []
    )
    exp05_gap_std = (
        [exp05_stats[t]["gap_std"] for t in exp05_tasks] if exp05_tasks else []
    )
    plot_three_way_comparison(
        ax,
        baseline_tasks,
        baseline_gap_std,
        exp03_gap_std,
        exp05_gap_std,
        higher_is_better=False,
        ylabel="Gap Standard Deviation",
        title="Desvio Padrão do Gap",
        show_deltas=True,
    )

    # 4. NME1 Performance
    ax = axes[1, 1]
    # Get all available tasks (union of all experiments)
    all_nme1_tasks = sorted(
        set(baseline_nme1.keys() if baseline_nme1 else [])
        | set(exp03_nme1.keys() if exp03_nme1 else [])
        | set(exp05_nme1.keys() if exp05_nme1 else [])
    )

    # Build aligned score lists (only for tasks present in all three experiments)
    common_nme1_tasks = sorted(
        set(baseline_nme1.keys() if baseline_nme1 else [])
        & set(exp03_nme1.keys() if exp03_nme1 else [])
        & set(exp05_nme1.keys() if exp05_nme1 else [])
    )

    if common_nme1_tasks:
        baseline_nme1_scores = [baseline_nme1[t] for t in common_nme1_tasks]
        exp03_nme1_scores = [exp03_nme1[t] for t in common_nme1_tasks]
        exp05_nme1_scores = [exp05_nme1[t] for t in common_nme1_tasks]

        plot_three_way_comparison(
            ax,
            common_nme1_tasks,
            baseline_nme1_scores,
            exp03_nme1_scores,
            exp05_nme1_scores,
            higher_is_better=True,
            ylabel="NME1 Accuracy (%)",
            title="Performance de Avaliação",
            show_deltas=True,
        )
        ax.set_ylim([55, 100])
    else:
        ax.text(
            0.5,
            0.5,
            "No common tasks with NME1 scores",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    output_file = output_dir / "three_margin_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Comparison plot saved to: {output_file}")

    # Generate markdown report
    generate_markdown_report(
        baseline_stats,
        exp03_stats,
        exp05_stats,
        baseline_nme1,
        exp03_nme1,
        exp05_nme1,
        output_dir,
    )


def generate_markdown_report(
    baseline_stats,
    exp03_stats,
    exp05_stats,
    baseline_nme1,
    exp03_nme1,
    exp05_nme1,
    output_dir,
):
    """Generate comprehensive markdown report comparing three experiments."""

    output_dir = Path(output_dir)
    report_path = output_dir / "three_margin_comparison_report.md"

    baseline_tasks = sorted(baseline_stats.keys())
    exp03_tasks = sorted(exp03_stats.keys()) if exp03_stats else []
    exp05_tasks = sorted(exp05_stats.keys()) if exp05_stats else []

    with open(report_path, "w") as f:
        # Header
        f.write("# Comparação de Margens ANT: Baseline vs antM=0.3 vs antM=0.5\n\n")
        f.write(
            f"**Gerado em**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("---\n\n")

        # Executive Summary
        f.write("## Sumário Executivo\n\n")

        if baseline_nme1 and exp03_nme1 and exp05_nme1:
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            exp03_final = exp03_nme1[max(exp03_nme1.keys())]
            exp05_final = exp05_nme1[max(exp05_nme1.keys())]

            f.write(f"**NME1 Accuracy Final**:\n")
            f.write(f"- Baseline (M=0.1): **{baseline_final:.2f}%**\n")
            f.write(
                f"- antM=0.3: **{exp03_final:.2f}%** ({exp03_final-baseline_final:+.2f}%)\n"
            )
            f.write(
                f"- antM=0.5: **{exp05_final:.2f}%** ({exp05_final-baseline_final:+.2f}%)\n\n"
            )

            # Determine winner
            winner_val = max(baseline_final, exp03_final, exp05_final)
            if winner_val == exp03_final:
                winner_name = "antM=0.3"
            elif winner_val == exp05_final:
                winner_name = "antM=0.5"
            else:
                winner_name = "Baseline"

            f.write(f"🏆 **Vencedor**: {winner_name} com {winner_val:.2f}%\n\n")

        if baseline_stats and exp03_stats and exp05_stats:
            baseline_avg_gap = np.mean(
                [baseline_stats[t]["current_gap"] for t in baseline_tasks]
            )
            exp03_avg_gap = np.mean(
                [exp03_stats[t]["current_gap"] for t in exp03_tasks]
            )
            exp05_avg_gap = np.mean(
                [exp05_stats[t]["current_gap"] for t in exp05_tasks]
            )

            f.write(f"**Gap Médio (pos_mean - neg_mean)**:\n")
            f.write(f"- Baseline (M=0.1): **{baseline_avg_gap:.4f}**\n")
            f.write(
                f"- antM=0.3: **{exp03_avg_gap:.4f}** ({exp03_avg_gap-baseline_avg_gap:+.4f})\n"
            )
            f.write(
                f"- antM=0.5: **{exp05_avg_gap:.4f}** ({exp05_avg_gap-baseline_avg_gap:+.4f})\n\n"
            )

        f.write("---\n\n")

        # Gap Evolution Table
        f.write("## 1. Evolução do Gap (pos_mean - neg_mean)\n\n")
        f.write(
            "O gap representa a separação entre similaridades positivas e negativas. "
        )
        f.write("**Valores maiores indicam melhor discriminação de features**.\n\n")

        f.write("| Task | Baseline | antM=0.3 | antM=0.5 | Vencedor |\n")
        f.write("|------|----------|----------|----------|----------|\n")

        for task in baseline_tasks:
            if task in exp03_tasks and task in exp05_tasks:
                base_gap = baseline_stats[task]["current_gap"]
                exp03_gap = exp03_stats[task]["current_gap"]
                exp05_gap = exp05_stats[task]["current_gap"]

                winner_val = max(base_gap, exp03_gap, exp05_gap)
                if winner_val == exp03_gap:
                    winner = "🔴 M=0.3"
                elif winner_val == exp05_gap:
                    winner = "🟢 M=0.5"
                else:
                    winner = "🔵 Base"

                f.write(
                    f"| {task} | {base_gap:.4f} | {exp03_gap:.4f} | {exp05_gap:.4f} | {winner} |\n"
                )

        f.write("\n")

        # Violation Percentage Table
        f.write("## 2. Percentual de Violações da Margem\n\n")
        f.write("Percentual de pares que violam a margem mínima. ")
        f.write("**Valores menores indicam melhor conformidade com a margem**.\n\n")

        f.write("| Task | Baseline | antM=0.3 | antM=0.5 | Vencedor |\n")
        f.write("|------|----------|----------|----------|----------|\n")

        for task in baseline_tasks:
            if task in exp03_tasks and task in exp05_tasks:
                base_viol = baseline_stats[task]["violation_pct"]
                exp03_viol = exp03_stats[task]["violation_pct"]
                exp05_viol = exp05_stats[task]["violation_pct"]

                winner_val = min(base_viol, exp03_viol, exp05_viol)
                if winner_val == exp03_viol:
                    winner = "🔴 M=0.3"
                elif winner_val == exp05_viol:
                    winner = "🟢 M=0.5"
                else:
                    winner = "🔵 Base"

                f.write(
                    f"| {task} | {base_viol:.2f}% | {exp03_viol:.2f}% | {exp05_viol:.2f}% | {winner} |\n"
                )

        f.write("\n")

        # NME1 Performance Table
        f.write("## 3. Performance NME1 (Acurácia de Avaliação)\n\n")
        f.write("Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. ")
        f.write(
            "**Valores maiores indicam melhor performance geral e retenção de conhecimento**.\n\n"
        )

        f.write("| Task | Baseline | antM=0.3 | antM=0.5 | Vencedor |\n")
        f.write("|------|----------|----------|----------|----------|\n")

        if baseline_nme1 and exp03_nme1 and exp05_nme1:
            for task in sorted(baseline_nme1.keys()):
                if task in exp03_nme1 and task in exp05_nme1:
                    base_nme1 = baseline_nme1[task]
                    exp03_nme1_val = exp03_nme1[task]
                    exp05_nme1_val = exp05_nme1[task]

                    winner_val = max(base_nme1, exp03_nme1_val, exp05_nme1_val)
                    if winner_val == exp03_nme1_val:
                        winner = "🔴 M=0.3"
                    elif winner_val == exp05_nme1_val:
                        winner = "🟢 M=0.5"
                    else:
                        winner = "🔵 Base"

                    f.write(
                        f"| {task} | {base_nme1:.2f}% | {exp03_nme1_val:.2f}% | {exp05_nme1_val:.2f}% | {winner} |\n"
                    )

        f.write("\n")

        # Statistical Summary
        f.write("## Resumo Estatístico\n\n")

        if baseline_stats and exp03_stats and exp05_stats:
            f.write("### Estatísticas de Gap\n\n")
            baseline_gaps = [baseline_stats[t]["current_gap"] for t in baseline_tasks]
            exp03_gaps = [exp03_stats[t]["current_gap"] for t in exp03_tasks]
            exp05_gaps = [exp05_stats[t]["current_gap"] for t in exp05_tasks]

            f.write(
                f"- **Baseline**: Média={np.mean(baseline_gaps):.4f}, Std={np.std(baseline_gaps):.4f}\n"
            )
            f.write(
                f"- **antM=0.3**: Média={np.mean(exp03_gaps):.4f}, Std={np.std(exp03_gaps):.4f}\n"
            )
            f.write(
                f"- **antM=0.5**: Média={np.mean(exp05_gaps):.4f}, Std={np.std(exp05_gaps):.4f}\n\n"
            )

            f.write("### Estatísticas de Violação\n\n")
            baseline_viols = [
                baseline_stats[t]["violation_pct"] for t in baseline_tasks
            ]
            exp03_viols = [exp03_stats[t]["violation_pct"] for t in exp03_tasks]
            exp05_viols = [exp05_stats[t]["violation_pct"] for t in exp05_tasks]

            f.write(
                f"- **Baseline**: Média={np.mean(baseline_viols):.2f}%, Std={np.std(baseline_viols):.2f}%\n"
            )
            f.write(
                f"- **antM=0.3**: Média={np.mean(exp03_viols):.2f}%, Std={np.std(exp03_viols):.2f}%\n"
            )
            f.write(
                f"- **antM=0.5**: Média={np.mean(exp05_viols):.2f}%, Std={np.std(exp05_viols):.2f}%\n\n"
            )

        if baseline_nme1 and exp03_nme1 and exp05_nme1:
            f.write("### Estatísticas de NME1 Accuracy\n\n")
            baseline_nme1_vals = list(baseline_nme1.values())
            exp03_nme1_vals = list(exp03_nme1.values())
            exp05_nme1_vals = list(exp05_nme1.values())

            f.write(
                f"- **Baseline**: Média={np.mean(baseline_nme1_vals):.2f}%, Std={np.std(baseline_nme1_vals):.2f}%\n"
            )
            f.write(
                f"- **antM=0.3**: Média={np.mean(exp03_nme1_vals):.2f}%, Std={np.std(exp03_nme1_vals):.2f}%\n"
            )
            f.write(
                f"- **antM=0.5**: Média={np.mean(exp05_nme1_vals):.2f}%, Std={np.std(exp05_nme1_vals):.2f}%\n\n"
            )

        f.write("---\n\n")

        # Conclusions
        f.write("## Conclusões\n\n")

        if baseline_nme1 and exp03_nme1 and exp05_nme1:
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            exp03_final = exp03_nme1[max(exp03_nme1.keys())]
            exp05_final = exp05_nme1[max(exp05_nme1.keys())]

            improvements = {
                "antM=0.3": exp03_final - baseline_final,
                "antM=0.5": exp05_final - baseline_final,
            }

            f.write("### Melhorias Relativas ao Baseline\n\n")
            for exp_name, improvement in improvements.items():
                if improvement > 0.1:
                    f.write(f"✅ **{exp_name}**: Melhoria de **+{improvement:.2f}%**\n")
                elif improvement < -0.1:
                    f.write(f"⚠️ **{exp_name}**: Degradação de **{improvement:.2f}%**\n")
                else:
                    f.write(
                        f"➖ **{exp_name}**: Performance comparável ({improvement:+.2f}%)\n"
                    )

            f.write("\n")

            # Overall winner
            winner_val = max(baseline_final, exp03_final, exp05_final)
            if winner_val == exp03_final:
                winner_name = "antM=0.3"
            elif winner_val == exp05_final:
                winner_name = "antM=0.5"
            else:
                winner_name = "Baseline"

            f.write(
                f"### Vencedor Geral\n\n🏆 **{winner_name}** apresenta a melhor performance final com **{winner_val:.2f}% NME1**\n\n"
            )

        f.write("---\n\n")
        f.write(
            "*Relatório gerado automaticamente por `compare_three_margin_experiments.py`*\n"
        )

    print(f"✓ Markdown report saved to: {report_path}")


def print_summary_table(
    baseline_stats,
    exp03_stats,
    exp05_stats,
    baseline_nme1,
    exp03_nme1,
    exp05_nme1,
):
    """Print detailed comparison table."""

    print("\n" + "=" * 120)
    print("COMPARAÇÃO DE MARGENS ANT: BASELINE (M=0.1) vs antM=0.3 vs antM=0.5")
    print("=" * 120)

    baseline_tasks = sorted(baseline_stats.keys())

    # Header
    print(
        f"\n{'Task':<6} {'Metric':<20} {'Baseline':<15} {'antM=0.3':<15} {'antM=0.5':<15} {'Winner':<10}"
    )
    print("-" * 120)

    metrics = [
        ("Gap", "current_gap", "{:.4f}", True),
        ("Violation %", "violation_pct", "{:.2f}%", False),
        ("Gap Std", "gap_std", "{:.4f}", False),
        ("NME1", "nme1", "{:.2f}%", True),
    ]

    for task in baseline_tasks:
        baseline_data = baseline_stats.get(task, {})
        exp03_data = exp03_stats.get(task, {}) if exp03_stats else {}
        exp05_data = exp05_stats.get(task, {}) if exp05_stats else {}

        for i, (label, key, fmt, higher_is_better) in enumerate(metrics):
            if key == "nme1":
                baseline_val = baseline_nme1.get(task)
                exp03_val = exp03_nme1.get(task) if exp03_nme1 else None
                exp05_val = exp05_nme1.get(task) if exp05_nme1 else None
            else:
                baseline_val = baseline_data.get(key)
                exp03_val = exp03_data.get(key)
                exp05_val = exp05_data.get(key)

            baseline_str = (
                fmt.format(baseline_val) if baseline_val is not None else "N/A"
            )
            exp03_str = fmt.format(exp03_val) if exp03_val is not None else "N/A"
            exp05_str = fmt.format(exp05_val) if exp05_val is not None else "N/A"

            # Determine winner
            if (
                baseline_val is not None
                and exp03_val is not None
                and exp05_val is not None
            ):
                vals = [baseline_val, exp03_val, exp05_val]
                if higher_is_better:
                    winner_idx = np.argmax(vals)
                else:
                    winner_idx = np.argmin(vals)

                winners = ["Base ✓", "M=0.3 ✓", "M=0.5 ✓"]
                winner = winners[winner_idx]
            else:
                winner = ""

            task_str = f"{task}" if i == 0 else ""
            print(
                f"{task_str:<6} {label:<20} {baseline_str:<15} {exp03_str:<15} {exp05_str:<15} {winner:<10}"
            )

        print("-" * 120)

    # Overall statistics
    print("\nESTATÍSTICAS GERAIS")
    print("=" * 120)

    if exp03_stats and exp05_stats and baseline_stats:
        # Average gaps
        baseline_avg_gap = np.mean(
            [baseline_stats[t]["current_gap"] for t in baseline_tasks]
        )
        exp03_avg_gap = np.mean(
            [exp03_stats[t]["current_gap"] for t in exp03_stats.keys()]
        )
        exp05_avg_gap = np.mean(
            [exp05_stats[t]["current_gap"] for t in exp05_stats.keys()]
        )
        print(
            f"Gap Médio:       Baseline={baseline_avg_gap:.4f}  |  M=0.3={exp03_avg_gap:.4f}  |  M=0.5={exp05_avg_gap:.4f}"
        )

        # Average violations
        baseline_avg_viol = np.mean(
            [baseline_stats[t]["violation_pct"] for t in baseline_tasks]
        )
        exp03_avg_viol = np.mean(
            [exp03_stats[t]["violation_pct"] for t in exp03_stats.keys()]
        )
        exp05_avg_viol = np.mean(
            [exp05_stats[t]["violation_pct"] for t in exp05_stats.keys()]
        )
        print(
            f"Violações Médias: Baseline={baseline_avg_viol:.2f}%  |  M=0.3={exp03_avg_viol:.2f}%  |  M=0.5={exp05_avg_viol:.2f}%"
        )

    if baseline_nme1 and exp03_nme1 and exp05_nme1:
        baseline_final = baseline_nme1[max(baseline_nme1.keys())]
        exp03_final = exp03_nme1[max(exp03_nme1.keys())]
        exp05_final = exp05_nme1[max(exp05_nme1.keys())]
        print(
            f"NME1 Final:      Baseline={baseline_final:.2f}%  |  M=0.3={exp03_final:.2f}%  |  M=0.5={exp05_final:.2f}%"
        )

    print("=" * 120)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare three ANT margin experiments")
    parser.add_argument(
        "--baseline",
        type=str,
        default="logs/done_exp_cifar100_10-10_baseline_tagfex_original",
        help="Path to baseline experiment directory",
    )
    parser.add_argument(
        "--exp03",
        type=str,
        default="logs/exp_cifar100_10-10_antB1_nceA1_antM0.3_antLocal",
        help="Path to antM=0.3 experiment directory",
    )
    parser.add_argument(
        "--exp05",
        type=str,
        default="logs/exp_cifar100_10-10_antB1_nceA1_antM0.5_antLocal",
        help="Path to antM=0.5 experiment directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/results/three_experiments_comparison",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Paths
    baseline_dir = Path(args.baseline)
    exp03_dir = Path(args.exp03)
    exp05_dir = Path(args.exp05)
    output_dir = Path(args.output)

    print("=" * 120)
    print("COMPARAÇÃO DE MARGENS ANT")
    print("=" * 120)
    print(f"\nBaseline (M=0.1): {baseline_dir}")
    print(f"antM=0.3:         {exp03_dir}")
    print(f"antM=0.5:         {exp05_dir}")
    print(f"Output:           {output_dir}\n")

    # Parse baseline data
    print("\n[1/4] Parsing baseline logs...")
    baseline_debug = baseline_dir / "exp_debug0.log"
    baseline_gist = baseline_dir / "exp_gistlog.log"

    if not baseline_debug.exists():
        print(f"❌ Error: {baseline_debug} not found")
        return

    baseline_data = parse_ant_stats_baseline(baseline_debug)
    baseline_stats = aggregate_by_task(baseline_data)
    baseline_nme1 = parse_nme1_scores(baseline_gist) if baseline_gist.exists() else {}
    print(f"   Parsed {len(baseline_data)} entries, {len(baseline_stats)} tasks")

    # Parse antM=0.3 data
    print("\n[2/4] Parsing antM=0.3 logs...")
    exp03_debug = exp03_dir / "exp_debug0.log"
    exp03_gist = exp03_dir / "exp_gistlog.log"

    if exp03_debug.exists():
        exp03_data = parse_ant_stats_new(exp03_debug)
        exp03_stats = aggregate_by_task(exp03_data)
        exp03_nme1 = parse_nme1_scores(exp03_gist) if exp03_gist.exists() else {}
        print(f"   Parsed {len(exp03_data)} entries, {len(exp03_stats)} tasks")
    else:
        print(f"   ⚠ antM=0.3 logs not found: {exp03_debug}")
        exp03_stats = {}
        exp03_nme1 = {}

    # Parse antM=0.5 data
    print("\n[3/4] Parsing antM=0.5 logs...")
    exp05_debug = exp05_dir / "exp_debug0.log"
    exp05_gist = exp05_dir / "exp_gistlog.log"

    if exp05_debug.exists():
        exp05_data = parse_ant_stats_new(exp05_debug)
        exp05_stats = aggregate_by_task(exp05_data)
        exp05_nme1 = parse_nme1_scores(exp05_gist) if exp05_gist.exists() else {}
        print(f"   Parsed {len(exp05_data)} entries, {len(exp05_stats)} tasks")
    else:
        print(f"   ⚠ antM=0.5 logs not found: {exp05_debug}")
        exp05_stats = {}
        exp05_nme1 = {}

    # Generate plots
    print("\n[4/4] Generating comparison plots...")
    plot_comparison(
        baseline_stats,
        exp03_stats,
        exp05_stats,
        baseline_nme1,
        exp03_nme1,
        exp05_nme1,
        output_dir,
    )

    # Print summary
    print("\n[5/5] Generating summary table...")
    print_summary_table(
        baseline_stats, exp03_stats, exp05_stats, baseline_nme1, exp03_nme1, exp05_nme1
    )

    print(f"\n{'='*120}")
    print(f"✓ Analysis complete! Results saved to: {output_dir}/")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
