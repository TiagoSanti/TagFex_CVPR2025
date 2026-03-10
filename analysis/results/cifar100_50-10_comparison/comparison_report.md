# InfoNCE + Local Anchor: Comparação de Experimentos

**Gerado em**: 2025-11-20 02:16:53

---

## Sumário Executivo

**NME1 Accuracy Final**:
- Baseline: **70.11%**
- Âncora Local: **69.35%**
- Melhoria: **-0.76%** ⚠️ (Baseline é melhor)

**Gap Médio (pos_mean - neg_mean)**:
- Baseline: **0.8754**
- Âncora Local: **0.8772**
- Diferença: **+0.0018**

---

## 1. Evolução do Gap (pos_mean - neg_mean)

O gap representa a separação entre similaridades positivas e negativas. **Valores maiores indicam melhor discriminação de features**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 0.8590 | 0.8611 | +0.0021 | 🟢 Local |
| 2 | 0.8651 | 0.8674 | +0.0023 | 🟢 Local |
| 3 | 0.8784 | 0.8799 | +0.0015 | 🟢 Local |
| 4 | 0.8819 | 0.8845 | +0.0026 | 🟢 Local |
| 5 | 0.8831 | 0.8842 | +0.0011 | 🟢 Local |
| 6 | 0.8851 | 0.8861 | +0.0009 | 🟢 Local |

**Resumo**: Âncora Local vence em 6/6 tasks, Baseline vence em 0/6 tasks.

---

## 2. Similaridades Positivas

Similaridade de cosseno média entre pares positivos (mesma classe). **Valores maiores indicam maior coesão intra-classe**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 0.8577 | 0.8597 | +0.0020 | 🟢 Local |
| 2 | 0.8644 | 0.8666 | +0.0022 | 🟢 Local |
| 3 | 0.8779 | 0.8795 | +0.0015 | 🟢 Local |
| 4 | 0.8815 | 0.8840 | +0.0025 | 🟢 Local |
| 5 | 0.8831 | 0.8842 | +0.0011 | 🟢 Local |
| 6 | 0.8851 | 0.8861 | +0.0010 | 🟢 Local |

**Resumo**: Âncora Local vence em 6/6 tasks, Baseline vence em 0/6 tasks.

---

## 3. Similaridades Negativas

Similaridade de cosseno média entre pares negativos (classes diferentes). **Valores menores (mais negativos) indicam melhor separação inter-classe**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | -0.0013 | -0.0014 | -0.0001 | 🟢 Local |
| 2 | -0.0007 | -0.0008 | -0.0001 | 🟢 Local |
| 3 | -0.0005 | -0.0005 | +0.0000 | 🔵 Base |
| 4 | -0.0004 | -0.0005 | -0.0001 | 🟢 Local |
| 5 | 0.0000 | 0.0001 | +0.0000 | 🔵 Base |
| 6 | 0.0000 | 0.0001 | +0.0000 | 🔵 Base |

**Resumo**: Âncora Local vence em 3/6 tasks (menor é melhor), Baseline vence em 3/6 tasks.

---

## 4. Performance NME1 (Acurácia de Avaliação)

Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. **Valores maiores indicam melhor performance geral e retenção de conhecimento**.

| Task | Baseline | Âncora Local | Δ (Local-Base) | Vencedor |
|------|----------|--------------|----------------|----------|
| 1 | 83.84% | 83.84% | +0.00% | ➖ Empate |
| 2 | 80.43% | 81.22% | +0.79% | 🟢 Local |
| 3 | 78.24% | 78.46% | +0.22% | 🟢 Local |
| 4 | 74.56% | 74.55% | -0.01% | 🔵 Base |
| 5 | 71.98% | 71.46% | -0.52% | 🔵 Base |
| 6 | 70.11% | 69.35% | -0.76% | 🔵 Base |

**Resumo**: Âncora Local vence em 2/6 tasks, Baseline vence em 3/6 tasks.

**Performance na Tarefa Final**: Baseline=70.11%, Âncora Local=69.35% (-0.76%)

---

## Resumo Estatístico

### Estatísticas de Gap

- **Baseline**:
  - Média: 0.8754
  - Desvio Padrão: 0.0098
  - Mínimo: 0.8590
  - Máximo: 0.8851

- **Âncora Local**:
  - Média: 0.8772
  - Desvio Padrão: 0.0095
  - Mínimo: 0.8611
  - Máximo: 0.8861

### Estatísticas de Similaridade Positiva

- **Baseline**: Média=0.8750, Desvio Padrão=0.0102
- **Âncora Local**: Média=0.8767, Desvio Padrão=0.0100

### Estatísticas de Similaridade Negativa

- **Baseline**: Média=-0.0005, Desvio Padrão=0.0004
- **Âncora Local**: Média=-0.0005, Desvio Padrão=0.0005

### Estatísticas de NME1 Accuracy

- **Baseline**: Média=76.53%, Desvio Padrão=4.78%
- **Âncora Local**: Média=76.48%, Desvio Padrão=5.17%

---

## Conclusões

⚠️ **Baseline apresenta melhor performance** (-0.76% NME1 final)

- Os valores de gap são **praticamente idênticos** entre os experimentos, sugerindo separação de features similar.

---

*Relatório gerado automaticamente por `compare_baseline_vs_local.py`*
