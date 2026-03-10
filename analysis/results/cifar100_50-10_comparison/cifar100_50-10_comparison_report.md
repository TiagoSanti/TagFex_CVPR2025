# CIFAR-100 50-10: Global vs Local Anchor Comparison

**Gerado em**: 2025-11-20 02:09:54

---

## Sumário Executivo

**NME1 Accuracy Final**:
- Global: **7011.00%**
- Local: **6935.00%**
- Melhoria: **-76.00%** ⚠️ (Global é melhor)

**Gap Médio (pos_mean - neg_mean)**:
- Global: **0.8754**
- Local: **0.8772**
- Diferença: **+0.0018**

---

## 1. Evolução do Gap (pos_mean - neg_mean)

O gap representa a separação entre similaridades positivas e negativas. **Valores maiores indicam melhor discriminação de features**.

| Task | Global | Local | Δ (Local-Global) | Vencedor |
|------|--------|-------|------------------|----------|
| 1 | 0.8590 | 0.8611 | +0.0021 | 🟢 Local |
| 2 | 0.8651 | 0.8674 | +0.0023 | 🟢 Local |
| 3 | 0.8784 | 0.8799 | +0.0015 | 🟢 Local |
| 4 | 0.8819 | 0.8845 | +0.0026 | 🟢 Local |
| 5 | 0.8831 | 0.8842 | +0.0011 | 🟢 Local |
| 6 | 0.8851 | 0.8861 | +0.0009 | 🟢 Local |

**Resumo**: Local vence em 6/6 tasks, Global vence em 0/6 tasks.

---

## 2. Similaridades Positivas

Similaridade de cosseno média entre pares positivos (mesma classe). **Valores maiores indicam maior coesão intra-classe**.

| Task | Global | Local | Δ (Local-Global) | Vencedor |
|------|--------|-------|------------------|----------|
| 1 | 0.8577 | 0.8597 | +0.0020 | 🟢 Local |
| 2 | 0.8644 | 0.8666 | +0.0022 | 🟢 Local |
| 3 | 0.8779 | 0.8795 | +0.0015 | 🟢 Local |
| 4 | 0.8815 | 0.8840 | +0.0025 | 🟢 Local |
| 5 | 0.8831 | 0.8842 | +0.0011 | 🟢 Local |
| 6 | 0.8851 | 0.8861 | +0.0010 | 🟢 Local |

**Resumo**: Local vence em 6/6 tasks, Global vence em 0/6 tasks.

---

## 3. Similaridades Negativas

Similaridade de cosseno média entre pares negativos (classes diferentes). **Valores menores (mais negativos) indicam melhor separação inter-classe**.

| Task | Global | Local | Δ (Local-Global) | Vencedor |
|------|--------|-------|------------------|----------|
| 1 | -0.0013 | -0.0014 | -0.0001 | 🟢 Local |
| 2 | -0.0007 | -0.0008 | -0.0001 | 🟢 Local |
| 3 | -0.0005 | -0.0005 | +0.0000 | 🔴 Global |
| 4 | -0.0004 | -0.0005 | -0.0001 | 🟢 Local |
| 5 | 0.0000 | 0.0001 | +0.0000 | 🔴 Global |
| 6 | 0.0000 | 0.0001 | +0.0000 | 🔴 Global |

**Resumo**: Local vence em 3/6 tasks (menor é melhor), Global vence em 3/6 tasks.

---

## 4. Performance NME1 (Acurácia de Avaliação)

Acurácia Nearest Mean Exemplar top-1 em todas as tarefas aprendidas. **Valores maiores indicam melhor performance geral e retenção de conhecimento**.

| Task | Global | Local | Δ (Local-Global) | Vencedor |
|------|--------|-------|------------------|----------|
| 1 | 8384.00% | 8384.00% | +0.00% | ➖ Empate |
| 2 | 8043.00% | 8122.00% | +79.00% | 🟢 Local |
| 3 | 7824.00% | 7846.00% | +22.00% | 🟢 Local |
| 4 | 7456.00% | 7455.00% | -1.00% | 🔴 Global |
| 5 | 7198.00% | 7146.00% | -52.00% | 🔴 Global |
| 6 | 7011.00% | 6935.00% | -76.00% | 🔴 Global |

**Resumo**: Local vence em 2/6 tasks, Global vence em 3/6 tasks.

**Performance na Tarefa Final**: Global=7011.00%, Local=6935.00% (-76.00%)

---

## Resumo Estatístico

### Estatísticas de Gap

- **Global**:
  - Média: 0.8754
  - Desvio Padrão: 0.0098
  - Mínimo: 0.8590
  - Máximo: 0.8851

- **Local**:
  - Média: 0.8772
  - Desvio Padrão: 0.0095
  - Mínimo: 0.8611
  - Máximo: 0.8861

### Estatísticas de Similaridade Positiva

- **Global**: Média=0.8750, Desvio Padrão=0.0102
- **Local**: Média=0.8767, Desvio Padrão=0.0100

### Estatísticas de Similaridade Negativa

- **Global**: Média=-0.0005, Desvio Padrão=0.0004
- **Local**: Média=-0.0005, Desvio Padrão=0.0005

### Estatísticas de NME1 Accuracy

- **Global**: Média=7652.67%, Desvio Padrão=478.49%
- **Local**: Média=7648.00%, Desvio Padrão=516.91%

---

## Conclusões

⚠️ **Global apresenta melhor performance** (-76.00% NME1 final)

- Os valores de gap são **praticamente idênticos** entre os experimentos, sugerindo separação de features similar.

---

*Relatório gerado automaticamente por `compare_50-10_global_vs_local.py`*
