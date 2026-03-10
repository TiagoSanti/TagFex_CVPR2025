#!/usr/bin/env python3
"""
Análise do comportamento das distâncias no TagFex baseline (InfoNCE puro).

Visualiza a evolução de:
1. Distribuição de distâncias positivas e negativas
2. Gap entre positivos e negativos ao longo das tasks
3. InfoNCE loss e suas componentes
4. Violações de margem (casos problemáticos)
5. Estatísticas agregadas por task

Ideal para apresentação: gráficos claros mostrando como o método aprende.
"""

import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import seaborn as sns

# Configurar estilo
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def parse_ant_stats(log_file):
    """Parse ANT distance statistics from exp_debug0.log."""
    data = []
    pattern = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) \| pos_std: ([\d.]+) \| pos_min: ([\d.]+) \| pos_max: ([\d.]+) \| neg_mean: ([-\d.]+) \| neg_std: ([\d.]+) \| neg_min: ([-\d.]+) \| neg_max: ([\d.]+) \| gap_mean: ([-\d.]+) \| gap_std: ([\d.]+) \| gap_min: ([-\d.]+) \| gap_max: ([\d.]+) \| margin: ([\d.]+) \| violation_pct: ([\d.]+)%"

    print(f"Parsing distance statistics from {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append(
                    {
                        "task": int(match.group(1)),
                        "epoch": int(match.group(2)),
                        "batch": int(match.group(3)),
                        "pos_mean": float(match.group(4)),
                        "pos_std": float(match.group(5)),
                        "pos_min": float(match.group(6)),
                        "pos_max": float(match.group(7)),
                        "neg_mean": float(match.group(8)),
                        "neg_std": float(match.group(9)),
                        "neg_min": float(match.group(10)),
                        "neg_max": float(match.group(11)),
                        "gap_mean": float(match.group(12)),
                        "gap_std": float(match.group(13)),
                        "gap_min": float(match.group(14)),
                        "gap_max": float(match.group(15)),
                        "margin": float(match.group(16)),
                        "violation_pct": float(match.group(17)),
                    }
                )

    print(f"  Found {len(data)} entries")
    return data


def parse_loss_components(log_file):
    """Parse InfoNCE loss components."""
    data = []
    pattern = (
        r"\[T(\d+) E(\d+) B(\d+)\] Loss components: contrast_infoNCE_nll: ([\d.]+)"
    )

    print(f"Parsing loss components from {log_file}...")
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                data.append(
                    {
                        "task": int(match.group(1)),
                        "epoch": int(match.group(2)),
                        "batch": int(match.group(3)),
                        "nll": float(match.group(4)),
                    }
                )

    print(f"  Found {len(data)} loss entries")
    return data


def aggregate_by_task_epoch(data):
    """Aggregate data by task and epoch."""
    task_epoch_data = defaultdict(lambda: defaultdict(list))

    for entry in data:
        task = entry["task"]
        epoch = entry["epoch"]
        task_epoch_data[task][epoch].append(entry)

    # Average across batches
    aggregated = defaultdict(list)
    for task in sorted(task_epoch_data.keys()):
        for epoch in sorted(task_epoch_data[task].keys()):
            entries = task_epoch_data[task][epoch]
            avg_entry = {"task": task, "epoch": epoch}

            # Average numeric fields
            numeric_fields = [
                k for k in entries[0].keys() if k not in ["task", "epoch", "batch"]
            ]
            for field in numeric_fields:
                values = [e[field] for e in entries]
                avg_entry[field] = np.mean(values)
                avg_entry[f"{field}_std_batch"] = np.std(values)

            aggregated[task].append(avg_entry)

    return aggregated


def plot_distance_evolution(stats_by_task, output_dir):
    """Plot evolution of positive and negative distances across tasks."""
    print("\nGenerating distance evolution plot...")

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "TagFex Baseline: Evolução das Distâncias ao Longo das Tasks",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    tasks_to_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for idx, task in enumerate(tasks_to_plot):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        if task not in stats_by_task or not stats_by_task[task]:
            continue

        data = stats_by_task[task]
        epochs = [d["epoch"] for d in data]

        # Plot pos_mean and neg_mean
        pos_means = [d["pos_mean"] for d in data]
        neg_means = [d["neg_mean"] for d in data]

        ax.plot(epochs, pos_means, "g-", linewidth=2, label="Pos Mean", alpha=0.8)
        ax.plot(epochs, neg_means, "r-", linewidth=2, label="Neg Mean", alpha=0.8)

        # Fill area showing gap
        ax.fill_between(
            epochs, pos_means, neg_means, alpha=0.2, color="blue", label="Gap"
        )

        # Add std as shaded area
        pos_stds = [d["pos_std"] for d in data]
        neg_stds = [d["neg_std"] for d in data]

        ax.fill_between(
            epochs,
            np.array(pos_means) - np.array(pos_stds),
            np.array(pos_means) + np.array(pos_stds),
            alpha=0.15,
            color="green",
        )
        ax.fill_between(
            epochs,
            np.array(neg_means) - np.array(neg_stds),
            np.array(neg_means) + np.array(neg_stds),
            alpha=0.15,
            color="red",
        )

        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Cosine Similarity", fontsize=9)
        ax.set_title(f"Task {task}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.0)

    plt.savefig(
        output_dir / "distance_evolution_all_tasks.png", dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: {output_dir / 'distance_evolution_all_tasks.png'}")
    plt.close()


def plot_gap_progression(stats_by_task, output_dir):
    """Plot gap progression: início vs fim de cada task."""
    print("\nGenerating gap progression plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Progressão do Gap: Início vs Fim de Cada Task", fontsize=16, fontweight="bold"
    )

    tasks = sorted(stats_by_task.keys())

    # Coletar dados de início e fim de cada task
    gap_start = []
    gap_end = []
    pos_start = []
    pos_end = []
    neg_start = []
    neg_end = []
    violation_start = []
    violation_end = []

    for task in tasks:
        data = stats_by_task[task]
        if len(data) < 2:
            continue

        # Primeiro epoch (início)
        start = data[0]
        # Último epoch (fim)
        end = data[-1]

        gap_start.append(start["pos_mean"] - start["neg_mean"])
        gap_end.append(end["pos_mean"] - end["neg_mean"])
        pos_start.append(start["pos_mean"])
        pos_end.append(end["pos_mean"])
        neg_start.append(start["neg_mean"])
        neg_end.append(end["neg_mean"])
        violation_start.append(start["violation_pct"])
        violation_end.append(end["violation_pct"])

    # Plot 1: Gap evolution
    ax = axes[0, 0]
    x = list(range(1, len(gap_start) + 1))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x], gap_start, width, label="Início da Task", alpha=0.8
    )
    ax.bar([i + width / 2 for i in x], gap_end, width, label="Fim da Task", alpha=0.8)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Gap (pos_mean - neg_mean)", fontsize=12)
    ax.set_title("Gap: Início vs Fim", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)

    # Plot 2: Positive means
    ax = axes[0, 1]
    ax.plot(x, pos_start, "o-", linewidth=2, markersize=8, label="Início", alpha=0.8)
    ax.plot(x, pos_end, "s-", linewidth=2, markersize=8, label="Fim", alpha=0.8)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Positive Mean", fontsize=12)
    ax.set_title("Similaridade Positiva", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    # Plot 3: Negative means
    ax = axes[1, 0]
    ax.plot(x, neg_start, "o-", linewidth=2, markersize=8, label="Início", alpha=0.8)
    ax.plot(x, neg_end, "s-", linewidth=2, markersize=8, label="Fim", alpha=0.8)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Negative Mean", fontsize=12)
    ax.set_title("Similaridade Negativa", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    # Plot 4: Violation percentage
    ax = axes[1, 1]
    ax.bar(
        [i - width / 2 for i in x], violation_start, width, label="Início", alpha=0.8
    )
    ax.bar([i + width / 2 for i in x], violation_end, width, label="Fim", alpha=0.8)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Violation % (gap < margin)", fontsize=12)
    ax.set_title("Violações de Margem", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(
        output_dir / "gap_progression_start_vs_end.png", dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: {output_dir / 'gap_progression_start_vs_end.png'}")
    plt.close()


def plot_loss_evolution(loss_by_task, output_dir):
    """Plot InfoNCE loss evolution."""
    print("\nGenerating loss evolution plot...")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        "InfoNCE Loss: Evolução ao Longo das Epochs", fontsize=16, fontweight="bold"
    )

    tasks = sorted(loss_by_task.keys())

    for idx, task in enumerate(tasks[:10]):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        data = loss_by_task[task]
        if not data:
            continue

        epochs = [d["epoch"] for d in data]
        nll = [d["nll"] for d in data]

        ax.plot(epochs, nll, "b-", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("NLL", fontsize=9)
        ax.set_title(f"Task {task}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Anotar valores inicial e final
        if len(nll) > 0:
            ax.text(
                0.05,
                0.95,
                f"Início: {nll[0]:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax.text(
                0.05,
                0.85,
                f"Fim: {nll[-1]:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )

    plt.tight_layout()
    plt.savefig(
        output_dir / "loss_evolution_all_tasks.png", dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: {output_dir / 'loss_evolution_all_tasks.png'}")
    plt.close()


def plot_distribution_comparison(stats_by_task, output_dir):
    """Plot box plots comparing distributions across tasks."""
    print("\nGenerating distribution comparison plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Distribuição das Distâncias: Comparação Entre Tasks (Fim do Treinamento)",
        fontsize=16,
        fontweight="bold",
    )

    tasks = sorted(stats_by_task.keys())

    # Coletar dados finais de cada task
    pos_means_final = []
    neg_means_final = []
    gaps_final = []
    violations_final = []

    for task in tasks:
        data = stats_by_task[task]
        if not data:
            continue
        final = data[-1]  # Último epoch
        pos_means_final.append(final["pos_mean"])
        neg_means_final.append(final["neg_mean"])
        gaps_final.append(final["pos_mean"] - final["neg_mean"])
        violations_final.append(final["violation_pct"])

    # Plot 1: Positive means
    ax = axes[0, 0]
    ax.bar(tasks, pos_means_final, color="green", alpha=0.7)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Positive Mean (final)", fontsize=12)
    ax.set_title("Similaridade Positiva Final", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(
        y=np.mean(pos_means_final),
        color="r",
        linestyle="--",
        label=f"Média: {np.mean(pos_means_final):.3f}",
    )
    ax.legend()

    # Plot 2: Negative means
    ax = axes[0, 1]
    ax.bar(tasks, neg_means_final, color="red", alpha=0.7)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Negative Mean (final)", fontsize=12)
    ax.set_title("Similaridade Negativa Final", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(
        y=np.mean(neg_means_final),
        color="b",
        linestyle="--",
        label=f"Média: {np.mean(neg_means_final):.3f}",
    )
    ax.legend()

    # Plot 3: Gaps
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))
    bars = ax.bar(tasks, gaps_final, color=colors, alpha=0.8)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Gap (pos - neg)", fontsize=12)
    ax.set_title("Gap Final por Task", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(
        y=np.mean(gaps_final),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Média: {np.mean(gaps_final):.3f}",
    )
    ax.axhline(y=0.7, color="orange", linestyle=":", linewidth=2, label="Target: 0.70")
    ax.legend()

    # Plot 4: Violations
    ax = axes[1, 1]
    ax.bar(tasks, violations_final, color="orange", alpha=0.7)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Violation % (final)", fontsize=12)
    ax.set_title("Violações de Margem Final", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(
        y=np.mean(violations_final),
        color="r",
        linestyle="--",
        label=f"Média: {np.mean(violations_final):.2f}%",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_dir / "distribution_comparison_across_tasks.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"  Saved: {output_dir / 'distribution_comparison_across_tasks.png'}")
    plt.close()


def plot_summary_statistics(stats_by_task, output_dir):
    """Create a comprehensive summary visualization."""
    print("\nGenerating summary statistics plot...")

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "TagFex Baseline: Resumo Estatístico das Distâncias",
        fontsize=16,
        fontweight="bold",
    )

    tasks = sorted(stats_by_task.keys())

    # Aggregate statistics
    stats_summary = {
        "task": [],
        "pos_mean_avg": [],
        "neg_mean_avg": [],
        "gap_avg": [],
        "pos_std_avg": [],
        "neg_std_avg": [],
        "violation_avg": [],
    }

    for task in tasks:
        data = stats_by_task[task]
        if not data:
            continue

        stats_summary["task"].append(task)
        stats_summary["pos_mean_avg"].append(np.mean([d["pos_mean"] for d in data]))
        stats_summary["neg_mean_avg"].append(np.mean([d["neg_mean"] for d in data]))
        stats_summary["gap_avg"].append(
            np.mean([d["pos_mean"] - d["neg_mean"] for d in data])
        )
        stats_summary["pos_std_avg"].append(np.mean([d["pos_std"] for d in data]))
        stats_summary["neg_std_avg"].append(np.mean([d["neg_std"] for d in data]))
        stats_summary["violation_avg"].append(
            np.mean([d["violation_pct"] for d in data])
        )

    # Plot 1: Average distances over tasks
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        stats_summary["task"],
        stats_summary["pos_mean_avg"],
        "go-",
        linewidth=2,
        markersize=8,
        label="Pos Mean",
    )
    ax.plot(
        stats_summary["task"],
        stats_summary["neg_mean_avg"],
        "ro-",
        linewidth=2,
        markersize=8,
        label="Neg Mean",
    )
    ax.fill_between(
        stats_summary["task"],
        stats_summary["pos_mean_avg"],
        stats_summary["neg_mean_avg"],
        alpha=0.2,
        color="blue",
    )
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Cosine Similarity (avg)", fontsize=11)
    ax.set_title("Distâncias Médias por Task", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gap evolution
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(
        stats_summary["task"],
        stats_summary["gap_avg"],
        "bo-",
        linewidth=3,
        markersize=8,
        label="Gap Médio",
    )
    ax.axhline(y=0.7, color="orange", linestyle="--", linewidth=2, label="Target (0.7)")
    ax.axhline(y=0.1, color="red", linestyle=":", linewidth=2, label="Margin (0.1)")
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Gap (pos - neg)", fontsize=11)
    ax.set_title("Evolução do Gap Entre Tasks", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Standard deviations
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(
        stats_summary["task"],
        stats_summary["pos_std_avg"],
        "g^-",
        linewidth=2,
        markersize=8,
        label="Pos Std",
    )
    ax.plot(
        stats_summary["task"],
        stats_summary["neg_std_avg"],
        "rv-",
        linewidth=2,
        markersize=8,
        label="Neg Std",
    )
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Standard Deviation (avg)", fontsize=11)
    ax.set_title("Variabilidade das Distâncias", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Violations
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(
        stats_summary["task"],
        stats_summary["violation_avg"],
        "o-",
        linewidth=2,
        markersize=8,
        color="orange",
    )
    ax.fill_between(
        stats_summary["task"],
        0,
        stats_summary["violation_avg"],
        alpha=0.3,
        color="orange",
    )
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Violation % (avg)", fontsize=11)
    ax.set_title("Casos Problemáticos (gap < margin)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Plot 5: Distribution heatmap
    ax = fig.add_subplot(gs[1, 1:])

    # Create matrix: tasks x metrics
    metrics_names = ["Pos Mean", "Neg Mean", "Gap", "Pos Std", "Neg Std", "Violations"]
    matrix_data = np.array(
        [
            stats_summary["pos_mean_avg"],
            stats_summary["neg_mean_avg"],
            stats_summary["gap_avg"],
            stats_summary["pos_std_avg"],
            stats_summary["neg_std_avg"],
            [v / 100 for v in stats_summary["violation_avg"]],  # Normalize to 0-1
        ]
    )

    im = ax.imshow(matrix_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.set_yticks(range(len(metrics_names)))
    ax.set_yticklabels(metrics_names)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_title("Heatmap: Todas as Métricas por Task", fontsize=12, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Valor Normalizado", fontsize=10)

    # Add text annotations
    for i in range(len(metrics_names)):
        for j in range(len(tasks)):
            text = ax.text(
                j,
                i,
                f"{matrix_data[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.savefig(output_dir / "summary_statistics.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'summary_statistics.png'}")
    plt.close()


def generate_text_report(stats_by_task, output_dir):
    """Generate detailed text report."""
    print("\nGenerating text report...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ANÁLISE DE DISTÂNCIAS: TAGFEX BASELINE (InfoNCE Puro)")
    report_lines.append("=" * 80)
    report_lines.append("")

    tasks = sorted(stats_by_task.keys())

    # Overall summary
    report_lines.append("## RESUMO GERAL")
    report_lines.append("")

    all_pos_means = []
    all_neg_means = []
    all_gaps = []
    all_violations = []

    for task in tasks:
        data = stats_by_task[task]
        if not data:
            continue
        final = data[-1]
        all_pos_means.append(final["pos_mean"])
        all_neg_means.append(final["neg_mean"])
        all_gaps.append(final["pos_mean"] - final["neg_mean"])
        all_violations.append(final["violation_pct"])

    report_lines.append(f"**Positive Mean (final):**")
    report_lines.append(f"  - Média: {np.mean(all_pos_means):.4f}")
    report_lines.append(f"  - Std: {np.std(all_pos_means):.4f}")
    report_lines.append(
        f"  - Min: {np.min(all_pos_means):.4f} (Task {tasks[np.argmin(all_pos_means)]})"
    )
    report_lines.append(
        f"  - Max: {np.max(all_pos_means):.4f} (Task {tasks[np.argmax(all_pos_means)]})"
    )
    report_lines.append("")

    report_lines.append(f"**Negative Mean (final):**")
    report_lines.append(f"  - Média: {np.mean(all_neg_means):.4f}")
    report_lines.append(f"  - Std: {np.std(all_neg_means):.4f}")
    report_lines.append(
        f"  - Min: {np.min(all_neg_means):.4f} (Task {tasks[np.argmin(all_neg_means)]})"
    )
    report_lines.append(
        f"  - Max: {np.max(all_neg_means):.4f} (Task {tasks[np.argmax(all_neg_means)]})"
    )
    report_lines.append("")

    report_lines.append(f"**Gap (pos - neg, final):**")
    report_lines.append(f"  - Média: {np.mean(all_gaps):.4f}")
    report_lines.append(f"  - Std: {np.std(all_gaps):.4f}")
    report_lines.append(
        f"  - Min: {np.min(all_gaps):.4f} (Task {tasks[np.argmin(all_gaps)]})"
    )
    report_lines.append(
        f"  - Max: {np.max(all_gaps):.4f} (Task {tasks[np.argmax(all_gaps)]})"
    )
    report_lines.append(f"  - Target (0.7): {np.mean(all_gaps)/0.7*100:.1f}% atingido")
    report_lines.append("")

    report_lines.append(f"**Violations (gap < margin=0.1, final):**")
    report_lines.append(f"  - Média: {np.mean(all_violations):.2f}%")
    report_lines.append(f"  - Std: {np.std(all_violations):.2f}%")
    report_lines.append(
        f"  - Min: {np.min(all_violations):.2f}% (Task {tasks[np.argmin(all_violations)]})"
    )
    report_lines.append(
        f"  - Max: {np.max(all_violations):.2f}% (Task {tasks[np.argmax(all_violations)]})"
    )
    report_lines.append("")

    # Per-task details
    report_lines.append("## DETALHAMENTO POR TASK")
    report_lines.append("")

    for task in tasks:
        data = stats_by_task[task]
        if not data:
            continue

        start = data[0]
        end = data[-1]

        report_lines.append(f"### Task {task}")
        report_lines.append("")
        report_lines.append(f"**Início (Epoch {start['epoch']}):**")
        report_lines.append(
            f"  - Pos Mean: {start['pos_mean']:.4f} ± {start['pos_std']:.4f}"
        )
        report_lines.append(
            f"  - Neg Mean: {start['neg_mean']:.4f} ± {start['neg_std']:.4f}"
        )
        report_lines.append(f"  - Gap: {start['pos_mean'] - start['neg_mean']:.4f}")
        report_lines.append(f"  - Violations: {start['violation_pct']:.2f}%")
        report_lines.append("")

        report_lines.append(f"**Fim (Epoch {end['epoch']}):**")
        report_lines.append(
            f"  - Pos Mean: {end['pos_mean']:.4f} ± {end['pos_std']:.4f}"
        )
        report_lines.append(
            f"  - Neg Mean: {end['neg_mean']:.4f} ± {end['neg_std']:.4f}"
        )
        report_lines.append(f"  - Gap: {end['pos_mean'] - end['neg_mean']:.4f}")
        report_lines.append(f"  - Violations: {end['violation_pct']:.2f}%")
        report_lines.append("")

        # Calculate improvements
        gap_improvement = (end["pos_mean"] - end["neg_mean"]) - (
            start["pos_mean"] - start["neg_mean"]
        )
        violation_improvement = start["violation_pct"] - end["violation_pct"]

        report_lines.append(f"**Progresso:**")
        report_lines.append(f"  - Δ Gap: {gap_improvement:+.4f}")
        report_lines.append(f"  - Δ Violations: {violation_improvement:+.2f}%")
        report_lines.append("")

    report_lines.append("=" * 80)

    # Write report
    report_path = output_dir / "baseline_distances_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distance behavior in TagFex baseline"
    )
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Path to experiment logs directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_baseline_distances",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANÁLISE DE DISTÂNCIAS: TAGFEX BASELINE")
    print("=" * 80)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Parse data
    debug_log = log_dir / "exp_debug0.log"

    print("\n[1/2] Parsing data...")
    stats_data = parse_ant_stats(debug_log)
    loss_data = parse_loss_components(debug_log)

    print("\n[2/2] Aggregating by task and epoch...")
    stats_by_task = aggregate_by_task_epoch(stats_data)
    loss_by_task = aggregate_by_task_epoch(loss_data)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_distance_evolution(stats_by_task, output_dir)
    plot_gap_progression(stats_by_task, output_dir)
    plot_loss_evolution(loss_by_task, output_dir)
    plot_distribution_comparison(stats_by_task, output_dir)
    plot_summary_statistics(stats_by_task, output_dir)

    # Generate text report
    print("\n" + "=" * 80)
    print("GENERATING TEXT REPORT")
    print("=" * 80)

    generate_text_report(stats_by_task, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - distance_evolution_all_tasks.png (10 subplots)")
    print("  - gap_progression_start_vs_end.png")
    print("  - loss_evolution_all_tasks.png")
    print("  - distribution_comparison_across_tasks.png")
    print("  - summary_statistics.png")
    print("  - baseline_distances_report.txt")
    print("")


if __name__ == "__main__":
    main()
