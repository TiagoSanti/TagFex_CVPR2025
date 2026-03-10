#!/usr/bin/env python3
"""
Compare three CIFAR-100 10-10 experiments:
- Baseline (TagFex original without ANT)
- ANT Local margin=0.3
- ANT Local margin=0.5

Analyzes:
1. Gap evolution (pos_mean - neg_mean)
2. Distance statistics (pos/neg mean, std, min, max)
3. Violation percentage trends
4. NME1 performance across tasks

Usage:
    python compare_three_experiments.py
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
    
    # Pattern for single-line logs (old format)
    pattern_single = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats: pos_mean: ([\d.]+) .* pos_std: ([\d.]+) .* pos_min: ([\d.]+) .* pos_max: ([\d.]+) .* neg_mean: ([-\d.]+) .* neg_std: ([\d.]+) .* neg_min: ([-\d.]+) .* neg_max: ([-\d.]+) .* gap_mean: ([-\d.]+) .* gap_std: ([\d.]+) .* gap_min: ([-\d.]+) .* gap_max: ([-\d.]+) .* margin: ([\d.]+) .* violation_pct: ([\d.]+)%"
    
    # Pattern for multi-line logs (new format) - just capture the header
    pattern_header = r"\[T(\d+) E(\d+) B(\d+)\] ANT distance stats:"
    
    # Patterns for individual metrics (can appear on subsequent lines)
    pattern_pos_min = r"pos_min: ([-\d.]+)"
    pattern_pos_max = r"pos_max: ([-\d.]+)"
    pattern_neg_min = r"neg_min: ([-\d.]+)"
    pattern_neg_max = r"neg_max: ([-\d.]+)"
    pattern_gap_mean = r"gap_mean: ([-\d.]+)"
    pattern_gap_std = r"gap_std: ([-\d.]+)"
    pattern_gap_min = r"gap_min: ([-\d.]+)"
    pattern_gap_max = r"gap_max: ([-\d.]+)"
    pattern_margin = r"margin: ([\d.]+)"
    pattern_violation = r"violation_pct: ([\d.]+)%"

    with open(log_file, "r") as f:
        content = f.read()
        
        # Try single-line format first
        for line in content.split('\n'):
            match = re.search(pattern_single, line)
            if match:
                task = int(match.group(1))
                epoch = int(match.group(2))
                batch = int(match.group(3))
                pos_mean = float(match.group(4))
                neg_mean = float(match.group(8))

                data.append({
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
                    "current_gap": pos_mean - neg_mean,
                })
        
        # If no single-line matches found, try multi-line format
        if not data:
            # Find all ANT distance stats blocks (might span multiple lines)
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                match_header = re.search(pattern_header, line)
                if match_header:
                    task = int(match_header.group(1))
                    epoch = int(match_header.group(2))
                    batch = int(match_header.group(3))
                    
                    # Collect the entire block (this line + next few lines until we hit another log entry)
                    block = line
                    j = i + 1
                    while j < len(lines) and not re.search(r"\[\d{4}-\d{2}-\d{2}", lines[j]):
                        block += " " + lines[j]
                        j += 1
                        if j - i > 5:  # Safety limit
                            break
                    
                    # Extract all metrics from the block
                    try:
                        pos_min = float(re.search(pattern_pos_min, block).group(1))
                        pos_max = float(re.search(pattern_pos_max, block).group(1))
                        neg_min = float(re.search(pattern_neg_min, block).group(1))
                        neg_max = float(re.search(pattern_neg_max, block).group(1))
                        gap_mean = float(re.search(pattern_gap_mean, block).group(1))
                        gap_std = float(re.search(pattern_gap_std, block).group(1))
                        gap_min = float(re.search(pattern_gap_min, block).group(1))
                        gap_max = float(re.search(pattern_gap_max, block).group(1))
                        margin = float(re.search(pattern_margin, block).group(1))
                        violation_pct = float(re.search(pattern_violation, block).group(1))
                        
                        # Calculate pos_mean and neg_mean from gap and available info
                        # gap_mean = pos_mean - neg_mean
                        # We need to extract or estimate these
                        # For now, use gap_mean as current_gap
                        pos_mean = (pos_min + pos_max) / 2  # Rough estimate
                        neg_mean = (neg_min + neg_max) / 2  # Rough estimate
                        
                        data.append({
                            "task": task,
                            "epoch": epoch,
                            "batch": batch,
                            "pos_mean": pos_mean,
                            "pos_std": gap_std,  # Approximation
                            "pos_min": pos_min,
                            "pos_max": pos_max,
                            "neg_mean": neg_mean,
                            "neg_std": gap_std,  # Approximation
                            "neg_min": neg_min,
                            "neg_max": neg_max,
                            "gap_mean": gap_mean,
                            "gap_std": gap_std,
                            "gap_min": gap_min,
                            "gap_max": gap_max,
                            "violation_pct": violation_pct,
                            "current_gap": gap_mean,  # Use gap_mean as authoritative
                        })
                    except (AttributeError, ValueError) as e:
                        pass  # Skip incomplete blocks
                    
                    i = j
                else:
                    i += 1

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


def plot_comparison(baseline_stats, m03_stats, m05_stats, baseline_nme1, m03_nme1, m05_nme1, output_dir):
    """Create comprehensive comparison plots for three experiments."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Extract tasks
    baseline_tasks = sorted(baseline_stats.keys()) if baseline_stats else []
    m03_tasks = sorted(m03_stats.keys()) if m03_stats else []
    m05_tasks = sorted(m05_stats.keys()) if m05_stats else []
    all_tasks = sorted(set(baseline_tasks) | set(m03_tasks) | set(m05_tasks))

    # Helper function to plot three experiments with winner highlighting
    def plot_three_with_winners(
        ax,
        baseline_tasks,
        baseline_vals,
        m03_tasks,
        m03_vals,
        m05_tasks,
        m05_vals,
        higher_is_better=True,
        ylabel="",
        title="",
        show_deltas=True,
    ):
        """Plot comparison of three experiments with winner highlighting."""
        from matplotlib.lines import Line2D

        # Plot lines first
        if baseline_tasks:
            ax.plot(
                baseline_tasks,
                baseline_vals,
                "--",
                color="#95a5a6",
                linewidth=2,
                alpha=0.6,
                zorder=1,
                label="Baseline"
            )
        if m03_tasks:
            ax.plot(
                m03_tasks,
                m03_vals,
                "-",
                color="#3498db",
                linewidth=2.5,
                alpha=0.7,
                zorder=2,
                label="ANT Local m=0.3"
            )
        if m05_tasks:
            ax.plot(
                m05_tasks,
                m05_vals,
                "-",
                color="#2ecc71",
                linewidth=2.5,
                alpha=0.7,
                zorder=3,
                label="ANT Local m=0.5"
            )

        # Plot markers with highlighting for winners
        if baseline_tasks and m03_tasks and m05_tasks and len(baseline_vals) == len(m03_vals) == len(m05_vals):
            for i, task in enumerate(baseline_tasks):
                base_val = baseline_vals[i]
                m03_val = m03_vals[i]
                m05_val = m05_vals[i]

                # Determine winner
                if higher_is_better:
                    winner_val = max(base_val, m03_val, m05_val)
                else:
                    winner_val = min(base_val, m03_val, m05_val)

                # Determine threshold for "close enough"
                tie_threshold = 0.001 if abs(winner_val) < 1.0 else 0.01

                # Plot baseline
                if abs(base_val - winner_val) < tie_threshold:
                    ax.plot(task, base_val, "s", color="#95a5a6", markersize=11, 
                           markeredgewidth=2.5, markeredgecolor="#7f8c8d", zorder=5)
                else:
                    ax.plot(task, base_val, "s", color="#95a5a6", markersize=7, 
                           alpha=0.4, zorder=4)

                # Plot m=0.3
                if abs(m03_val - winner_val) < tie_threshold:
                    ax.plot(task, m03_val, "^", color="#3498db", markersize=11, 
                           markeredgewidth=2.5, markeredgecolor="#2980b9", zorder=6)
                else:
                    ax.plot(task, m03_val, "^", color="#3498db", markersize=7, 
                           alpha=0.4, zorder=4)

                # Plot m=0.5
                if abs(m05_val - winner_val) < tie_threshold:
                    ax.plot(task, m05_val, "o", color="#2ecc71", markersize=11, 
                           markeredgewidth=2.5, markeredgecolor="#27ae60", zorder=7)
                else:
                    ax.plot(task, m05_val, "o", color="#2ecc71", markersize=7, 
                           alpha=0.4, zorder=4)

                # Add delta label for winner
                if show_deltas:
                    # Find best vs second best
                    vals = [(base_val, "Base"), (m03_val, "0.3"), (m05_val, "0.5")]
                    if higher_is_better:
                        vals.sort(reverse=True, key=lambda x: x[0])
                    else:
                        vals.sort(key=lambda x: x[0])
                    
                    best_val, best_label = vals[0]
                    second_val, _ = vals[1]
                    delta = best_val - second_val

                    if abs(delta) >= tie_threshold:
                        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                        y_offset = y_range * 0.05

                        if abs(delta) < 0.01:
                            delta_text = f"+{delta:.4f}"
                        elif abs(delta) < 1:
                            delta_text = f"+{delta:.3f}"
                        else:
                            delta_text = f"+{delta:.1f}"

                        # Color based on winner
                        if best_label == "0.5":
                            text_color = "#27ae60"
                        elif best_label == "0.3":
                            text_color = "#2980b9"
                        else:
                            text_color = "#7f8c8d"

                        ax.text(
                            task,
                            best_val + y_offset,
                            delta_text,
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color=text_color,
                            fontweight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor=text_color,
                                alpha=0.8,
                                linewidth=1,
                            ),
                        )

        # Legend
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.set_xlabel("Task", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 10)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "CIFAR-100 10-10: Baseline vs ANT Local (m=0.3 vs m=0.5)",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Gap Evolution (pos_mean - neg_mean)
    ax = axes[0, 0]
    baseline_gaps = [baseline_stats[t]["current_gap"] for t in baseline_tasks] if baseline_tasks else []
    m03_gaps = [m03_stats[t]["current_gap"] for t in m03_tasks] if m03_tasks else []
    m05_gaps = [m05_stats[t]["current_gap"] for t in m05_tasks] if m05_tasks else []
    plot_three_with_winners(
        ax,
        baseline_tasks,
        baseline_gaps,
        m03_tasks,
        m03_gaps,
        m05_tasks,
        m05_gaps,
        higher_is_better=True,
        ylabel="Gap (pos_mean - neg_mean)",
        title="Gap Evolution Across Tasks",
    )

    # 2. Positive Similarities
    ax = axes[0, 1]
    baseline_pos = [baseline_stats[t]["pos_mean"] for t in baseline_tasks] if baseline_tasks else []
    m03_pos = [m03_stats[t]["pos_mean"] for t in m03_tasks] if m03_tasks else []
    m05_pos = [m05_stats[t]["pos_mean"] for t in m05_tasks] if m05_tasks else []
    plot_three_with_winners(
        ax,
        baseline_tasks,
        baseline_pos,
        m03_tasks,
        m03_pos,
        m05_tasks,
        m05_pos,
        higher_is_better=True,
        ylabel="Positive Mean Similarity",
        title="Positive Similarities",
    )

    # 3. Negative Similarities
    ax = axes[1, 0]
    baseline_neg = [baseline_stats[t]["neg_mean"] for t in baseline_tasks] if baseline_tasks else []
    m03_neg = [m03_stats[t]["neg_mean"] for t in m03_tasks] if m03_tasks else []
    m05_neg = [m05_stats[t]["neg_mean"] for t in m05_tasks] if m05_tasks else []
    plot_three_with_winners(
        ax,
        baseline_tasks,
        baseline_neg,
        m03_tasks,
        m03_neg,
        m05_tasks,
        m05_neg,
        higher_is_better=False,  # Lower (more negative) is better
        ylabel="Negative Mean Similarity",
        title="Negative Similarities",
    )

    # 4. NME1 Performance
    ax = axes[1, 1]
    baseline_nme1_tasks = sorted(baseline_nme1.keys()) if baseline_nme1 else []
    baseline_nme1_scores = [baseline_nme1[t] for t in baseline_nme1_tasks] if baseline_nme1 else []
    
    m03_nme1_tasks = sorted(m03_nme1.keys()) if m03_nme1 else []
    m03_nme1_scores = [m03_nme1[t] for t in m03_nme1_tasks] if m03_nme1 else []
    
    m05_nme1_tasks = sorted(m05_nme1.keys()) if m05_nme1 else []
    m05_nme1_scores = [m05_nme1[t] for t in m05_nme1_tasks] if m05_nme1 else []

    plot_three_with_winners(
        ax,
        baseline_nme1_tasks,
        baseline_nme1_scores,
        m03_nme1_tasks,
        m03_nme1_scores,
        m05_nme1_tasks,
        m05_nme1_scores,
        higher_is_better=True,
        ylabel="NME1 Accuracy (%)",
        title="Evaluation Performance",
    )
    ax.set_ylim([55, 100])

    plt.tight_layout()
    plt.savefig(
        output_dir / "three_experiments_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"\n✓ Comparison plot saved to: {output_dir / 'three_experiments_comparison.png'}"
    )

    # Generate markdown report
    generate_markdown_report(
        baseline_stats, m03_stats, m05_stats, baseline_nme1, m03_nme1, m05_nme1, output_dir
    )


def generate_markdown_report(
    baseline_stats, m03_stats, m05_stats, baseline_nme1, m03_nme1, m05_nme1, output_dir
):
    """Generate a comprehensive markdown report comparing three experiments."""

    output_dir = Path(output_dir)
    report_path = output_dir / "three_experiments_report.md"

    baseline_tasks = sorted(baseline_stats.keys()) if baseline_stats else []
    m03_tasks = sorted(m03_stats.keys()) if m03_stats else []
    m05_tasks = sorted(m05_stats.keys()) if m05_stats else []
    all_tasks = sorted(set(baseline_tasks) | set(m03_tasks) | set(m05_tasks))

    with open(report_path, "w") as f:
        # Header
        f.write("# CIFAR-100 10-10: Comparação de Três Experimentos\n\n")
        f.write(f"**Gerado em**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Sumário Executivo\n\n")

        if baseline_nme1 and m03_nme1 and m05_nme1:
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            m03_final = m03_nme1[max(m03_nme1.keys())]
            m05_final = m05_nme1[max(m05_nme1.keys())]

            f.write(f"**NME1 Accuracy Final**:\n")
            f.write(f"- Baseline (TagFex original): **{baseline_final:.2f}%**\n")
            f.write(f"- ANT Local m=0.3: **{m03_final:.2f}%** ({m03_final-baseline_final:+.2f}% vs baseline)\n")
            f.write(f"- ANT Local m=0.5: **{m05_final:.2f}%** ({m05_final-baseline_final:+.2f}% vs baseline)\n\n")

            # Determine best
            best_nme1 = max(baseline_final, m03_final, m05_final)
            if best_nme1 == m05_final:
                f.write("🏆 **Vencedor**: ANT Local m=0.5\n\n")
            elif best_nme1 == m03_final:
                f.write("🏆 **Vencedor**: ANT Local m=0.3\n\n")
            else:
                f.write("🏆 **Vencedor**: Baseline\n\n")

        if baseline_stats and m03_stats and m05_stats:
            baseline_avg_gap = np.mean([baseline_stats[t]["current_gap"] for t in baseline_tasks])
            m03_avg_gap = np.mean([m03_stats[t]["current_gap"] for t in m03_tasks])
            m05_avg_gap = np.mean([m05_stats[t]["current_gap"] for t in m05_tasks])

            f.write(f"**Gap Médio (pos_mean - neg_mean)**:\n")
            f.write(f"- Baseline: **{baseline_avg_gap:.4f}**\n")
            f.write(f"- ANT Local m=0.3: **{m03_avg_gap:.4f}** ({m03_avg_gap-baseline_avg_gap:+.4f})\n")
            f.write(f"- ANT Local m=0.5: **{m05_avg_gap:.4f}** ({m05_avg_gap-baseline_avg_gap:+.4f})\n\n")

        f.write("---\n\n")

        # Gap Evolution Table
        f.write("## 1. Evolução do Gap (pos_mean - neg_mean)\n\n")
        f.write("O gap representa a separação entre similaridades positivas e negativas. ")
        f.write("**Valores maiores indicam melhor discriminação de features**.\n\n")

        f.write("| Task | Baseline | ANT m=0.3 | ANT m=0.5 | Melhor Δ | Vencedor |\n")
        f.write("|------|----------|-----------|-----------|----------|----------|\n")

        for task in all_tasks:
            if task in baseline_tasks and task in m03_tasks and task in m05_tasks:
                base_gap = baseline_stats[task]["current_gap"]
                m03_gap = m03_stats[task]["current_gap"]
                m05_gap = m05_stats[task]["current_gap"]
                
                best_gap = max(base_gap, m03_gap, m05_gap)
                delta = best_gap - base_gap
                
                if best_gap == m05_gap:
                    winner = "🟢 m=0.5"
                elif best_gap == m03_gap:
                    winner = "🔵 m=0.3"
                else:
                    winner = "⚪ Base"
                
                f.write(f"| {task} | {base_gap:.4f} | {m03_gap:.4f} | {m05_gap:.4f} | {delta:+.4f} | {winner} |\n")

        f.write("\n---\n\n")

        # NME1 Performance Table
        f.write("## 2. Performance NME1 (Acurácia de Avaliação)\n\n")

        f.write("| Task | Baseline | ANT m=0.3 | ANT m=0.5 | Melhor Δ | Vencedor |\n")
        f.write("|------|----------|-----------|-----------|----------|----------|\n")

        if baseline_nme1 and m03_nme1 and m05_nme1:
            for task in sorted(baseline_nme1.keys()):
                if task in m03_nme1 and task in m05_nme1:
                    base_nme1 = baseline_nme1[task]
                    m03_nme1_val = m03_nme1[task]
                    m05_nme1_val = m05_nme1[task]
                    
                    best_nme1 = max(base_nme1, m03_nme1_val, m05_nme1_val)
                    delta = best_nme1 - base_nme1
                    
                    if best_nme1 == m05_nme1_val:
                        winner = "🟢 m=0.5"
                    elif best_nme1 == m03_nme1_val:
                        winner = "🔵 m=0.3"
                    else:
                        winner = "⚪ Base"
                    
                    f.write(f"| {task} | {base_nme1:.2f}% | {m03_nme1_val:.2f}% | {m05_nme1_val:.2f}% | {delta:+.2f}% | {winner} |\n")

        f.write("\n---\n\n")

        # Statistical Summary
        f.write("## Resumo Estatístico\n\n")

        if baseline_stats and m03_stats and m05_stats:
            f.write("### Comparação de Gaps\n\n")
            
            baseline_gaps = [baseline_stats[t]["current_gap"] for t in baseline_tasks]
            m03_gaps = [m03_stats[t]["current_gap"] for t in m03_tasks]
            m05_gaps = [m05_stats[t]["current_gap"] for t in m05_tasks]

            f.write("| Experimento | Média | Desvio Padrão | Mínimo | Máximo |\n")
            f.write("|-------------|-------|---------------|--------|--------|\n")
            f.write(f"| Baseline | {np.mean(baseline_gaps):.4f} | {np.std(baseline_gaps):.4f} | {np.min(baseline_gaps):.4f} | {np.max(baseline_gaps):.4f} |\n")
            f.write(f"| ANT m=0.3 | {np.mean(m03_gaps):.4f} | {np.std(m03_gaps):.4f} | {np.min(m03_gaps):.4f} | {np.max(m03_gaps):.4f} |\n")
            f.write(f"| ANT m=0.5 | {np.mean(m05_gaps):.4f} | {np.std(m05_gaps):.4f} | {np.min(m05_gaps):.4f} | {np.max(m05_gaps):.4f} |\n\n")

        if baseline_nme1 and m03_nme1 and m05_nme1:
            f.write("### Comparação de NME1 Accuracy\n\n")
            
            baseline_nme1_vals = list(baseline_nme1.values())
            m03_nme1_vals = list(m03_nme1.values())
            m05_nme1_vals = list(m05_nme1.values())

            f.write("| Experimento | Média | Desvio Padrão | Mínimo | Máximo |\n")
            f.write("|-------------|-------|---------------|--------|--------|\n")
            f.write(f"| Baseline | {np.mean(baseline_nme1_vals):.2f}% | {np.std(baseline_nme1_vals):.2f}% | {np.min(baseline_nme1_vals):.2f}% | {np.max(baseline_nme1_vals):.2f}% |\n")
            f.write(f"| ANT m=0.3 | {np.mean(m03_nme1_vals):.2f}% | {np.std(m03_nme1_vals):.2f}% | {np.min(m03_nme1_vals):.2f}% | {np.max(m03_nme1_vals):.2f}% |\n")
            f.write(f"| ANT m=0.5 | {np.mean(m05_nme1_vals):.2f}% | {np.std(m05_nme1_vals):.2f}% | {np.min(m05_nme1_vals):.2f}% | {np.max(m05_nme1_vals):.2f}% |\n\n")

        f.write("---\n\n")

        # Conclusions
        f.write("## Conclusões\n\n")

        if baseline_nme1 and m03_nme1 and m05_nme1:
            baseline_final = baseline_nme1[max(baseline_nme1.keys())]
            m03_final = m03_nme1[max(m03_nme1.keys())]
            m05_final = m05_nme1[max(m05_nme1.keys())]

            best_final = max(baseline_final, m03_final, m05_final)
            
            if best_final == m05_final and m05_final > baseline_final + 0.1:
                f.write(f"✅ **ANT Local m=0.5 apresenta a melhor performance** (+{m05_final-baseline_final:.2f}% vs baseline)\n\n")
            elif best_final == m03_final and m03_final > baseline_final + 0.1:
                f.write(f"✅ **ANT Local m=0.3 apresenta a melhor performance** (+{m03_final-baseline_final:.2f}% vs baseline)\n\n")
            else:
                f.write(f"⚠️ **Baseline ainda é competitivo ou superior**\n\n")

            # Compare m=0.3 vs m=0.5
            if abs(m05_final - m03_final) < 0.1:
                f.write("- Margem 0.3 e 0.5 apresentam **performance similar**\n")
            elif m05_final > m03_final:
                f.write(f"- **Margem 0.5 supera margem 0.3** em {m05_final-m03_final:.2f}%\n")
            else:
                f.write(f"- **Margem 0.3 supera margem 0.5** em {m03_final-m05_final:.2f}%\n")

        f.write("\n---\n\n")
        f.write("*Relatório gerado automaticamente por `compare_three_experiments.py`*\n")

    print(f"✓ Markdown report saved to: {report_path}")


def main():
    """Main execution."""
    
    # Paths
    baseline_dir = Path("../../logs/done_exp_cifar100_10-10_baseline_tagfex_original")
    m03_dir = Path("../../logs/exp_cifar100_10-10_antB1_nceA1_antM0.3_antLocal")
    m05_dir = Path("../../logs/exp_cifar100_10-10_antB1_nceA1_antM0.5_antLocal")
    output_dir = Path("../results/three_experiments_comparison")

    print("=" * 80)
    print("CIFAR-100 10-10: Three Experiments Comparison")
    print("=" * 80)
    print(f"\nBaseline:      {baseline_dir}")
    print(f"ANT m=0.3:     {m03_dir}")
    print(f"ANT m=0.5:     {m05_dir}")
    print(f"Output:        {output_dir}\n")

    # Parse baseline data
    print("\n[1/6] Parsing baseline logs...")
    baseline_debug = baseline_dir / "exp_debug0.log"
    baseline_gist = baseline_dir / "exp_gistlog.log"

    if not baseline_debug.exists():
        print(f"❌ Error: {baseline_debug} not found")
        return

    baseline_data = parse_ant_stats(baseline_debug)
    baseline_stats = aggregate_by_task(baseline_data)
    baseline_nme1 = parse_nme1_scores(baseline_gist) if baseline_gist.exists() else {}
    print(f"   Parsed {len(baseline_data)} entries, {len(baseline_stats)} tasks")

    # Parse ANT m=0.3 data
    print("\n[2/6] Parsing ANT m=0.3 logs...")
    m03_debug = m03_dir / "exp_debug0.log"
    m03_gist = m03_dir / "exp_gistlog.log"

    if m03_debug.exists():
        m03_data = parse_ant_stats(m03_debug)
        m03_stats = aggregate_by_task(m03_data)
        m03_nme1 = parse_nme1_scores(m03_gist) if m03_gist.exists() else {}
        print(f"   Parsed {len(m03_data)} entries, {len(m03_stats)} tasks")
    else:
        print(f"   ⚠ ANT m=0.3 logs not found: {m03_debug}")
        m03_stats = {}
        m03_nme1 = {}

    # Parse ANT m=0.5 data
    print("\n[3/6] Parsing ANT m=0.5 logs...")
    m05_debug = m05_dir / "exp_debug0.log"
    m05_gist = m05_dir / "exp_gistlog.log"

    if m05_debug.exists():
        m05_data = parse_ant_stats(m05_debug)
        m05_stats = aggregate_by_task(m05_data)
        m05_nme1 = parse_nme1_scores(m05_gist) if m05_gist.exists() else {}
        print(f"   Parsed {len(m05_data)} entries, {len(m05_stats)} tasks")
    else:
        print(f"   ⚠ ANT m=0.5 logs not found: {m05_debug}")
        m05_stats = {}
        m05_nme1 = {}

    # Generate plots
    print("\n[4/6] Generating comparison plots...")
    plot_comparison(baseline_stats, m03_stats, m05_stats, baseline_nme1, m03_nme1, m05_nme1, output_dir)

    # Print quick summary
    print("\n[5/6] Summary of results...")
    if baseline_nme1 and m03_nme1 and m05_nme1:
        baseline_final = baseline_nme1[max(baseline_nme1.keys())]
        m03_final = m03_nme1[max(m03_nme1.keys())]
        m05_final = m05_nme1[max(m05_nme1.keys())]
        
        print(f"\nNME1 Final Accuracy:")
        print(f"  Baseline:      {baseline_final:.2f}%")
        print(f"  ANT m=0.3:     {m03_final:.2f}% ({m03_final-baseline_final:+.2f}%)")
        print(f"  ANT m=0.5:     {m05_final:.2f}% ({m05_final-baseline_final:+.2f}%)")

    print(f"\n{'='*80}")
    print(f"✓ Analysis complete! Results saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
