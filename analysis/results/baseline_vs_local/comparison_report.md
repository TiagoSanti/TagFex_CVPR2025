# InfoNCE + Local Anchor: Comparação de Experimentos

**Gerado em**: 2025-11-19 14:15:04

---

## Sumário Executivo

**NME1 Accuracy Final**:
- Baseline: **63.51%**
- Âncora Local: **63.83%**
- Melhoria: **+0.32%** ✅ (Âncora Local é melhor)

**Gap Médio (pos_mean - neg_mean)**:
- Baseline: **0.8513**
- Âncora Local: **0.8489**
- Diferença: **-0.0024**

---

## 1. Evolução do Gap (pos_mean - neg_mean)

O gap representa a separação entre similaridades positivas e negativas. **Valores maiores indicam melhor discriminação de features**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 0.7510 | 0.7334 | -0.0176 | 🔵 Base |
| 2 | 0.8160 | 0.8130 | -0.0030 | 🔵 Base |
| 3 | 0.8470 | 0.8460 | -0.0010 | 🔵 Base |
| 4 | 0.8580 | 0.8574 | -0.0007 | 🔵 Base |
| 5 | 0.8673 | 0.8667 | -0.0006 | 🔵 Base |
| 6 | 0.8644 | 0.8640 | -0.0004 | 🔵 Base |
| 7 | 0.8723 | 0.8722 | -0.0001 | 🔵 Base |
| 8 | 0.8762 | 0.8764 | +0.0001 | 🟢 Local |
| 9 | 0.8791 | 0.8783 | -0.0008 | 🔵 Base |
| 10 | 0.8813 | 0.8815 | +0.0003 | 🟢 Local |

**Resumo**: Âncora Local vence em 2/10 tasks, Baseline vence em 8/10 tasks.

---

## 2. Similaridades Positivas

Similaridade de cosseno média entre pares positivos (mesma classe). **Valores maiores indicam maior coesão intra-classe**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 0.7682 | 0.7603 | -0.0079 | 🔵 Base |
| 2 | 0.8139 | 0.8108 | -0.0030 | 🔵 Base |
| 3 | 0.8457 | 0.8446 | -0.0011 | 🔵 Base |
| 4 | 0.8570 | 0.8564 | -0.0006 | 🔵 Base |
| 5 | 0.8669 | 0.8664 | -0.0006 | 🔵 Base |
| 6 | 0.8641 | 0.8637 | -0.0004 | 🔵 Base |
| 7 | 0.8723 | 0.8723 | -0.0001 | 🔵 Base |
| 8 | 0.8765 | 0.8764 | -0.0001 | 🔵 Base |
| 9 | 0.8796 | 0.8789 | -0.0007 | 🔵 Base |
| 10 | 0.8818 | 0.8820 | +0.0002 | 🟢 Local |

**Resumo**: Âncora Local vence em 1/10 tasks, Baseline vence em 9/10 tasks.

---

## 3. Similaridades Negativas

Similaridade de cosseno média entre pares negativos (classes diferentes). **Valores menores (mais negativos) indicam melhor separação inter-classe**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 0.0171 | 0.0268 | +0.0097 | 🔵 Base |
| 2 | -0.0021 | -0.0021 | +0.0000 | 🔵 Base |
| 3 | -0.0013 | -0.0014 | -0.0000 | 🟢 Local |
| 4 | -0.0010 | -0.0010 | +0.0001 | 🔵 Base |
| 5 | -0.0004 | -0.0004 | -0.0000 | 🟢 Local |
| 6 | -0.0002 | -0.0003 | -0.0001 | 🟢 Local |
| 7 | 0.0000 | 0.0000 | +0.0000 | 🔵 Base |
| 8 | 0.0003 | 0.0001 | -0.0002 | 🟢 Local |
| 9 | 0.0005 | 0.0007 | +0.0002 | 🔵 Base |
| 10 | 0.0005 | 0.0005 | -0.0000 | 🟢 Local |

**Resumo**: Âncora Local vence em 5/10 tasks (menor é melhor), Baseline vence em 5/10 tasks.

---

## 4. Performance NME1 (Acurácia de Avaliação)

Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. **Valores maiores indicam melhor performance geral e retenção de conhecimento**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 93.20% | 93.20% | +0.00% | ➖ Empate |
| 2 | 85.25% | 85.30% | +0.05% | 🟢 Local |
| 3 | 82.73% | 83.07% | +0.34% | 🟢 Local |
| 4 | 79.15% | 78.88% | -0.27% | 🔵 Base |
| 5 | 75.82% | 76.30% | +0.48% | 🟢 Local |
| 6 | 73.10% | 73.27% | +0.17% | 🟢 Local |
| 7 | 70.56% | 70.79% | +0.23% | 🟢 Local |
| 8 | 67.38% | 67.81% | +0.43% | 🟢 Local |
| 9 | 64.76% | 64.93% | +0.17% | 🟢 Local |
| 10 | 63.51% | 63.83% | +0.32% | 🟢 Local |

**Resumo**: Âncora Local vence em 8/10 tasks, Baseline vence em 1/10 tasks.

**Performance na Tarefa Final**: Baseline=63.51%, Âncora Local=63.83% (+0.32%)

---

## Resumo Estatístico

### Estatísticas de Gap

- **Baseline**:
  - Média: 0.8513
  - Desvio Padrão: 0.0381
  - Mínimo: 0.7510
  - Máximo: 0.8813

- **Âncora Local**:
  - Média: 0.8489
  - Desvio Padrão: 0.0430
  - Mínimo: 0.7334
  - Máximo: 0.8815

### Estatísticas de Similaridade Positiva

- **Baseline**: Média=0.8526, Desvio Padrão=0.0340
- **Âncora Local**: Média=0.8512, Desvio Padrão=0.0363

### Estatísticas de Similaridade Negativa

- **Baseline**: Média=0.0013, Desvio Padrão=0.0053
- **Âncora Local**: Média=0.0023, Desvio Padrão=0.0082

### Estatísticas de NME1 Accuracy

- **Baseline**: Média=75.55%, Desvio Padrão=9.11%
- **Âncora Local**: Média=75.74%, Desvio Padrão=9.01%

---

## Conclusões

✅ **Âncora Local apresenta melhoria significativa** (+0.32% NME1 final)

- Os valores de gap são **praticamente idênticos** entre os experimentos, sugerindo separação de features similar.

---

*Relatório gerado automaticamente por `compare_baseline_vs_local.py`*
