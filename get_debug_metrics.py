import os
import re
import numpy as np
import pandas as pd

LOGS_DIR = "/mnt/raid/home/tiago/logs"
OUTPUT_MD = os.path.join(LOGS_DIR, "experiment_report.md")

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_space_array(s):
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split()]


def infer_dataset(exp_name):
    """
    Extrai o identificador dataset/protocolo do nome do experimento.
    Suporta prefixos 'debug_exp_' e 'exp_'.
    Ex: debug_exp_cifar100_10-10_... → cifar100_10-10
        exp_tiny_imagenet_20-20_...  → tiny_imagenet_20-20
    """
    m = re.match(r"(?:debug_exp_|exp_)(.+?)_antB", exp_name)
    if m:
        return m.group(1)
    return "unknown"


def df_to_markdown_table(df, cols, float_digits=2):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "*sem dados*"
    df_fmt = df[cols].copy()
    for col in df_fmt.columns:
        if pd.api.types.is_float_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(
                lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else ""
            )
    return df_fmt.to_markdown(index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Gistlog parsing  (acurácia / forgetting)
# ──────────────────────────────────────────────────────────────────────────────

def parse_gistlog(path):
    """Extrai acc1_curve, avg_acc1 e eval_acc1_per_task de exp_gistlog.log."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    acc_curve_matches = re.findall(r"acc1_curve \[([^\]]+)\]", text)
    avg_acc_matches   = re.findall(r"avg_acc1 ([0-9]+(?:\.[0-9]+)?)", text)
    per_task_matches  = re.findall(r"eval_acc1_per_task \[([^\]]+)\]", text)

    if not acc_curve_matches or not per_task_matches:
        return None

    return {
        "acc_curve":        parse_space_array(acc_curve_matches[-1]),
        "avg_acc":          float(avg_acc_matches[-1]) if avg_acc_matches else np.nan,
        "per_task_history": [parse_space_array(m) for m in per_task_matches],
    }


def compute_avg_forgetting(per_task_history, exclude_last_task=True):
    """
    forgetting_j = max(histórico da task j) − valor final da task j
    exclude_last_task=True: exclui a última task da média (ainda não foi esquecida).
    """
    if not per_task_history:
        return np.nan, []

    n_tasks = len(per_task_history[-1])
    forgetting = []

    for j in range(n_tasks):
        hist = [v[j] for v in per_task_history if len(v) > j]
        forgetting.append(0.0 if len(hist) <= 1 else max(hist) - hist[-1])

    valid = forgetting[:-1] if exclude_last_task and len(forgetting) > 1 else forgetting
    return (float(np.mean(valid)) if valid else np.nan), forgetting


# ──────────────────────────────────────────────────────────────────────────────
# Debug log parsing  (exp_debug0.log)
# ──────────────────────────────────────────────────────────────────────────────

_CTX_RE  = re.compile(r"\[T(\d+) E(\d+) B(\d+)\]")
_KV_RE   = re.compile(r"([\w]+):\s+([-\d.eE+]+)%?")

_MARKERS = {
    "Loss components:":   "loss",
    "ANT distance stats:": "ant",
    "Contrastive stats:": "contrastive",
}


def _strip_loss_prefix(kvs):
    """'contrast_nll' → 'loss_nll', 'contrast_total' → 'loss_total', etc."""
    cleaned = {}
    for k, v in kvs.items():
        short = k.split("_", 1)[1] if "_" in k else k
        cleaned[f"loss_{short}"] = v
    return cleaned


def parse_debug_log(path):
    """
    Lê exp_debug0.log linha a linha e retorna um DataFrame com colunas:
      task, epoch, batch,
      loss_nll, loss_ant_loss, loss_nce_weighted, loss_ant_weighted, loss_total,
      pos_min, pos_max, neg_min, neg_max,
      gap_mean, gap_std, gap_min, gap_max, margin, violation_pct, ant_loss,
      (opcional) pos_mean, pos_std, neg_mean, neg_std, current_gap
    """
    records = {}  # (task, epoch, batch) → merged dict

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                marker_type = marker_str = None
                for m_str, m_type in _MARKERS.items():
                    if m_str in raw_line:
                        marker_str, marker_type = m_str, m_type
                        break
                if marker_type is None:
                    continue

                ctx = _CTX_RE.search(raw_line)
                if not ctx:
                    continue

                key = (int(ctx[1]), int(ctx[2]), int(ctx[3]))
                after = raw_line[raw_line.index(marker_str) + len(marker_str):]
                kvs = {k: float(v) for k, v in _KV_RE.findall(after)}

                if marker_type == "loss":
                    kvs = _strip_loss_prefix(kvs)

                if key not in records:
                    records[key] = {"task": key[0], "epoch": key[1], "batch": key[2]}
                records[key].update(kvs)

    except FileNotFoundError:
        return pd.DataFrame()

    return pd.DataFrame(list(records.values())) if records else pd.DataFrame()


def aggregate_per_epoch(debug_df):
    """Média dos valores por batch → uma linha por (task, epoch)."""
    if debug_df.empty:
        return pd.DataFrame()
    num_cols = [c for c in debug_df.columns if c not in ("task", "epoch", "batch")]
    return debug_df.groupby(["task", "epoch"])[num_cols].mean().reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Coleta de resultados
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_DIRS = {"auto_experiments"}


def collect_results(logs_dir):
    results = []

    for exp in sorted(os.listdir(logs_dir)):
        if exp in _SKIP_DIRS:
            continue

        exp_path = os.path.join(logs_dir, exp)
        if not os.path.isdir(exp_path):
            continue

        gistlog = os.path.join(exp_path, "exp_gistlog.log")
        if not os.path.exists(gistlog):
            continue

        parsed = parse_gistlog(gistlog)
        if parsed is None:
            continue

        avg_fgt, fgt_per_task = compute_avg_forgetting(parsed["per_task_history"])

        row = {
            "dataset":       infer_dataset(exp),
            "exp":           exp,
            "avg_acc":       parsed["avg_acc"],
            "avg_forgetting": avg_fgt,
            "num_tasks":     len(parsed["acc_curve"]),
        }
        for i, v in enumerate(parsed["acc_curve"], 1):
            row[f"T{i}"] = v
        for i, v in enumerate(fgt_per_task, 1):
            row[f"F_T{i}"] = v

        results.append(row)

    return pd.DataFrame(results)


def drop_incomplete_experiments(df):
    groups = []
    for dataset, group in df.groupby("dataset", sort=True):
        n = int(group["num_tasks"].max())
        tcols = [f"T{i}" for i in range(1, n + 1)]
        group = group.copy()
        for c in tcols:
            if c not in group.columns:
                group[c] = np.nan
        group = group[group["num_tasks"] == n].dropna(subset=tcols)
        groups.append(group)
    return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Seção de debug por experimento
# ──────────────────────────────────────────────────────────────────────────────

# Colunas em ordem de prioridade para cada categoria
_LOSS_COLS = ["loss_total", "loss_nll", "loss_ant_loss", "loss_ant_weighted"]
_ANT_COLS  = ["violation_pct", "gap_mean", "gap_min", "gap_max"]
_SIM_COLS  = ["pos_mean", "pos_std", "neg_mean", "neg_std", "current_gap",
              "pos_min", "pos_max", "neg_min", "neg_max"]


def _present(df, cols):
    return [c for c in cols if c in df.columns]


def _debug_section_for_exp(exp_name, exp_path):
    """Retorna linhas markdown com a análise de debug de um experimento."""
    debug_log = os.path.join(exp_path, "exp_debug0.log")
    if not os.path.exists(debug_log):
        return [f"*`exp_debug0.log` não encontrado.*\n"]

    print(f"  Parsing: {exp_name} ...", end=" ", flush=True)
    debug_df = parse_debug_log(debug_log)
    if debug_df.empty:
        print("vazio.")
        return [f"*Nenhum dado de debug encontrado.*\n"]

    epoch_df = aggregate_per_epoch(debug_df)
    n_tasks   = epoch_df["task"].nunique()
    n_epochs  = epoch_df["epoch"].nunique()
    n_records = len(debug_df)
    print(f"ok ({n_records:,} registros → {n_tasks} tasks × {n_epochs} épocas distintas).")

    loss_cols = _present(epoch_df, _LOSS_COLS)
    ant_cols  = _present(epoch_df, _ANT_COLS)
    sim_cols  = _present(epoch_df, _SIM_COLS)
    sim_evo   = _present(epoch_df, ["pos_mean", "neg_mean", "current_gap"])

    tasks = sorted(epoch_df["task"].unique())
    lines = []

    # ── Tabela resumo: último epoch de cada task ──────────────────────────────
    final_rows = []
    for t in tasks:
        t_df   = epoch_df[epoch_df["task"] == t]
        last   = t_df["epoch"].max()
        row    = t_df[t_df["epoch"] == last].iloc[0].to_dict()
        row["task"]        = int(t)
        row["final_epoch"] = int(last)
        final_rows.append(row)

    final_df     = pd.DataFrame(final_rows)
    summary_cols = ["task", "final_epoch"] + loss_cols + ant_cols + sim_cols

    lines.append("##### Resumo no último epoch por task\n")
    lines.append(df_to_markdown_table(final_df, summary_cols, float_digits=4))
    lines.append("")

    # ── Evolução por task (amostrada a cada 10 epochs) ────────────────────────
    evo_cols = ["epoch"] + loss_cols + ant_cols + sim_evo
    lines.append("##### Evolução ao longo do treinamento (épocas amostradas)\n")

    for t in tasks:
        t_df    = epoch_df[epoch_df["task"] == t].copy().sort_values("epoch")
        min_ep  = int(t_df["epoch"].min())
        max_ep  = int(t_df["epoch"].max())
        n_batch = int(debug_df[debug_df["task"] == t]["batch"].max()) \
                  if "batch" in debug_df.columns else "?"

        sample_set = {min_ep} | set(range(10, max_ep + 1, 10)) | {max_ep}
        sampled    = t_df[t_df["epoch"].isin(sample_set)].copy()

        lines.append(f"**Task {t}** — epochs {min_ep}–{max_ep}, ~{n_batch} batches/epoch\n")
        lines.append(df_to_markdown_table(sampled, evo_cols, float_digits=4))
        lines.append("")

    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Relatório markdown completo
# ──────────────────────────────────────────────────────────────────────────────

def build_markdown_report(df, logs_dir):
    lines = []
    lines.append("# Relatório comparativo de experimentos\n")
    lines.append(
        "Resultados extraídos de `exp_gistlog.log` (acurácia/forgetting) "
        "e `exp_debug0.log` (dinâmica de treinamento: loss, ANT, similaridade).\n"
    )
    lines.append(
        "Forgetting = `max(histórico da task) − valor final`, excluindo a última task da média.\n"
    )

    for dataset in sorted(df["dataset"].unique()):
        group = (
            df[df["dataset"] == dataset]
            .copy()
            .sort_values(["avg_acc", "avg_forgetting"], ascending=[False, True])
            .reset_index(drop=True)
        )

        max_tasks   = int(group["num_tasks"].max())
        task_cols   = [f"T{i}"   for i in range(1, max_tasks + 1) if f"T{i}"   in group.columns]
        forget_cols = [f"F_T{i}" for i in range(1, max_tasks + 1) if f"F_T{i}" in group.columns]

        lines.append(f"## Dataset: `{dataset}`\n")
        lines.append(f"- Experimentos completos: **{len(group)}**")
        lines.append(f"- Número de tasks: **{max_tasks}**\n")

        lines.append("### Resumo de acurácia\n")
        lines.append(df_to_markdown_table(group, ["exp", "avg_acc", "avg_forgetting"] + task_cols))
        lines.append("")

        lines.append("### Forgetting por task\n")
        lines.append(df_to_markdown_table(group, ["exp", "avg_forgetting"] + forget_cols))
        lines.append("")

        best_acc = group.iloc[0]
        best_fgt = group.sort_values(["avg_forgetting", "avg_acc"], ascending=[True, False]).iloc[0]
        lines.append("### Destaques\n")
        lines.append(f"- Melhor `avg_acc`: **{best_acc['exp']}** ({best_acc['avg_acc']:.2f})")
        lines.append(f"- Melhor `avg_forgetting`: **{best_fgt['exp']}** ({best_fgt['avg_forgetting']:.2f})")
        lines.append("")

        # ── Análise de debug por experimento ──────────────────────────────────
        lines.append("### Análise de debug por experimento\n")
        for _, row in group.iterrows():
            exp_name = row["exp"]
            lines.append(f"#### `{exp_name}`\n")
            lines.extend(_debug_section_for_exp(exp_name, os.path.join(logs_dir, exp_name)))

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    df = collect_results(LOGS_DIR)

    if df.empty:
        print("Nenhum experimento válido encontrado.")
        return

    df = drop_incomplete_experiments(df)

    if df.empty:
        print("Nenhum experimento completo encontrado após filtragem.")
        return

    print(f"Experimentos completos encontrados: {len(df)}\n")
    print("Gerando relatório com análise de debug...")
    report_md = build_markdown_report(df, LOGS_DIR)

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nRelatório salvo em: {OUTPUT_MD}")

    print("\n=== Prévia dos datasets ===")
    for dataset in sorted(df["dataset"].unique()):
        subdf = df[df["dataset"] == dataset].sort_values(
            ["avg_acc", "avg_forgetting"], ascending=[False, True]
        )
        print(f"\n[{dataset}]")
        print(subdf[["exp", "avg_acc", "avg_forgetting"]].to_string(index=False))


if __name__ == "__main__":
    main()