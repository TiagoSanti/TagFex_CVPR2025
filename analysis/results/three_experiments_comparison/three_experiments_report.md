# CIFAR-100 10-10: Comparação de Três Experimentos

**Gerado em**: 2025-11-22 17:11:51

---

## Sumário Executivo

**NME1 Accuracy Final**:
- Baseline (TagFex original): **63.51%**
- ANT Local m=0.3: **70.71%** (+7.20% vs baseline)
- ANT Local m=0.5: **73.48%** (+9.97% vs baseline)

🏆 **Vencedor**: ANT Local m=0.5

**Gap Médio (pos_mean - neg_mean)**:
- Baseline: **0.8513**
- ANT Local m=0.3: **-0.5744** (-1.4257)
- ANT Local m=0.5: **-0.5970** (-1.4483)

---

## 1. Evolução do Gap (pos_mean - neg_mean)

O gap representa a separação entre similaridades positivas e negativas. **Valores maiores indicam melhor discriminação de features**.

| Task | Baseline | ANT m=0.3 | ANT m=0.5 | Melhor Δ | Vencedor |
|------|----------|-----------|-----------|----------|----------|
| 1 | 0.7510 | -0.6472 | -0.6553 | +0.0000 | ⚪ Base |
| 2 | 0.8160 | -0.6036 | -0.6146 | +0.0000 | ⚪ Base |
| 3 | 0.8470 | -0.5750 | -0.5927 | +0.0000 | ⚪ Base |
| 4 | 0.8580 | -0.5576 | -0.5798 | +0.0000 | ⚪ Base |
| 5 | 0.8673 | -0.5644 | -0.5873 | +0.0000 | ⚪ Base |
| 6 | 0.8644 | -0.5466 | -0.5728 | +0.0000 | ⚪ Base |
| 7 | 0.8723 | -0.5495 | -0.5768 | +0.0000 | ⚪ Base |

---

## 2. Performance NME1 (Acurácia de Avaliação)

| Task | Baseline | ANT m=0.3 | ANT m=0.5 | Melhor Δ | Vencedor |
|------|----------|-----------|-----------|----------|----------|
| 1 | 93.20% | 93.20% | 93.20% | +0.00% | 🟢 m=0.5 |
| 2 | 85.25% | 85.40% | 85.70% | +0.45% | 🟢 m=0.5 |
| 3 | 82.73% | 82.73% | 82.93% | +0.20% | 🟢 m=0.5 |
| 4 | 79.15% | 78.70% | 79.15% | +0.00% | 🟢 m=0.5 |
| 5 | 75.82% | 76.04% | 76.66% | +0.84% | 🟢 m=0.5 |
| 6 | 73.10% | 73.13% | 73.48% | +0.38% | 🟢 m=0.5 |

---

## Resumo Estatístico

### Comparação de Gaps

| Experimento | Média | Desvio Padrão | Mínimo | Máximo |
|-------------|-------|---------------|--------|--------|
| Baseline | 0.8513 | 0.0381 | 0.7510 | 0.8813 |
| ANT m=0.3 | -0.5744 | 0.0325 | -0.6472 | -0.5466 |
| ANT m=0.5 | -0.5970 | 0.0270 | -0.6553 | -0.5728 |

### Comparação de NME1 Accuracy

| Experimento | Média | Desvio Padrão | Mínimo | Máximo |
|-------------|-------|---------------|--------|--------|
| Baseline | 75.55% | 9.11% | 63.51% | 93.20% |
| ANT m=0.3 | 79.99% | 7.19% | 70.71% | 93.20% |
| ANT m=0.5 | 81.85% | 6.44% | 73.48% | 93.20% |

---

## Conclusões

✅ **ANT Local m=0.5 apresenta a melhor performance** (+9.97% vs baseline)

- **Margem 0.5 supera margem 0.3** em 2.77%

---

*Relatório gerado automaticamente por `compare_three_experiments.py`*
