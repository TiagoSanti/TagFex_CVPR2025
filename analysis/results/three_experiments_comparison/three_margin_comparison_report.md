# Comparação de Margens ANT: Baseline vs antM=0.3 vs antM=0.5

**Gerado em**: 2025-12-02 10:08:51

---

## Sumário Executivo

**NME1 Accuracy Final**:
- Baseline (M=0.1): **63.51%**
- antM=0.3: **63.81%** (+0.30%)
- antM=0.5: **63.74%** (+0.23%)

🏆 **Vencedor**: antM=0.3 com 63.81%

**Gap Médio (pos_mean - neg_mean)**:
- Baseline (M=0.1): **0.8513**
- antM=0.3: **0.2692** (-0.5821)
- antM=0.5: **0.2580** (-0.5933)

---

## 1. Evolução do Gap (pos_mean - neg_mean)

O gap representa a separação entre similaridades positivas e negativas. **Valores maiores indicam melhor discriminação de features**.

| Task | Baseline | antM=0.3 | antM=0.5 | Vencedor |
|------|----------|----------|----------|----------|
| 1 | 0.7510 | 0.2695 | 0.2523 | 🔵 Base |
| 2 | 0.8160 | 0.2876 | 0.2721 | 🔵 Base |
| 3 | 0.8470 | 0.2690 | 0.2570 | 🔵 Base |
| 4 | 0.8580 | 0.2619 | 0.2496 | 🔵 Base |
| 5 | 0.8673 | 0.2634 | 0.2543 | 🔵 Base |
| 6 | 0.8644 | 0.2774 | 0.2661 | 🔵 Base |
| 7 | 0.8723 | 0.2643 | 0.2565 | 🔵 Base |
| 8 | 0.8762 | 0.2640 | 0.2536 | 🔵 Base |
| 9 | 0.8791 | 0.2652 | 0.2585 | 🔵 Base |
| 10 | 0.8813 | 0.2697 | 0.2600 | 🔵 Base |

## 2. Percentual de Violações da Margem

Percentual de pares que violam a margem mínima. **Valores menores indicam melhor conformidade com a margem**.

| Task | Baseline | antM=0.3 | antM=0.5 | Vencedor |
|------|----------|----------|----------|----------|
| 1 | 4.34% | 9.27% | 23.23% | 🔵 Base |
| 2 | 1.86% | 8.28% | 26.19% | 🔵 Base |
| 3 | 1.77% | 7.97% | 28.27% | 🔵 Base |
| 4 | 1.78% | 8.40% | 30.70% | 🔵 Base |
| 5 | 1.72% | 8.23% | 29.92% | 🔵 Base |
| 6 | 1.79% | 8.96% | 32.07% | 🔵 Base |
| 7 | 1.79% | 9.28% | 32.14% | 🔵 Base |
| 8 | 1.83% | 9.16% | 32.60% | 🔵 Base |
| 9 | 1.77% | 9.12% | 32.51% | 🔵 Base |
| 10 | 1.77% | 9.04% | 32.66% | 🔵 Base |

## 3. Performance NME1 (Acurácia de Avaliação)

Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. **Valores maiores indicam melhor performance geral e retenção de conhecimento**.

| Task | Baseline | antM=0.3 | antM=0.5 | Vencedor |
|------|----------|----------|----------|----------|
| 1 | 93.20% | 93.20% | 93.20% | 🔴 M=0.3 |
| 2 | 85.25% | 85.40% | 85.70% | 🟢 M=0.5 |
| 3 | 82.73% | 82.73% | 82.93% | 🟢 M=0.5 |
| 4 | 79.15% | 78.70% | 79.15% | 🟢 M=0.5 |
| 5 | 75.82% | 76.04% | 76.66% | 🟢 M=0.5 |
| 6 | 73.10% | 73.13% | 73.48% | 🟢 M=0.5 |
| 7 | 70.56% | 70.71% | 71.37% | 🟢 M=0.5 |
| 8 | 67.38% | 67.14% | 67.62% | 🟢 M=0.5 |
| 9 | 64.76% | 65.17% | 65.24% | 🟢 M=0.5 |
| 10 | 63.51% | 63.81% | 63.74% | 🔴 M=0.3 |

## Resumo Estatístico

### Estatísticas de Gap

- **Baseline**: Média=0.8513, Std=0.0381
- **antM=0.3**: Média=0.2692, Std=0.0075
- **antM=0.5**: Média=0.2580, Std=0.0064

### Estatísticas de Violação

- **Baseline**: Média=2.04%, Std=0.77%
- **antM=0.3**: Média=8.77%, Std=0.47%
- **antM=0.5**: Média=30.03%, Std=3.05%

### Estatísticas de NME1 Accuracy

- **Baseline**: Média=75.55%, Std=9.11%
- **antM=0.3**: Média=75.60%, Std=9.03%
- **antM=0.5**: Média=75.91%, Std=9.01%

---

## Conclusões

### Melhorias Relativas ao Baseline

✅ **antM=0.3**: Melhoria de **+0.30%**
✅ **antM=0.5**: Melhoria de **+0.23%**

### Vencedor Geral

🏆 **antM=0.3** apresenta a melhor performance final com **63.81% NME1**

---

*Relatório gerado automaticamente por `compare_three_margin_experiments.py`*
