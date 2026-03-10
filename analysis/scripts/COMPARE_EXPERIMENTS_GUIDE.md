# 📊 Compare Experiments - Guia Completo

Guia de uso do script `compare_experiments.py` para análise comparativa completa entre experimentos Baseline e ANT+Gap.

---

## 🎯 Overview

O script integra **dois tipos de logs** para análise abrangente:

1. **`exp_matrix_debug0.log`**: Loss components (gap, ANT loss, gap loss, total loss)
2. **`exp_gistlog.log`**: Performance metrics (accuracy, NME, forgetting)

---

## 📋 Usage

### Método 1: Usando Diretórios (Recomendado)

```bash
python analysis_scripts/compare_experiments.py \
    --baseline-dir logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal \
    --ant-gap-dir logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5 \
    --output analysis_results
```

O script automaticamente procura:
- `{dir}/exp_matrix_debug0.log`
- `{dir}/exp_gistlog.log`

### Método 2: Especificando Logs Individuais

```bash
python analysis_scripts/compare_experiments.py \
    --baseline-matrix logs/exp_baseline/exp_matrix_debug0.log \
    --baseline-gist logs/exp_baseline/exp_gistlog.log \
    --ant-gap-matrix logs/exp_ant_gap/exp_matrix_debug0.log \
    --ant-gap-gist logs/exp_ant_gap/exp_gistlog.log \
    --output analysis_results
```

---

## 📈 Outputs

### 1. `comparison_losses.png` (2×3 grid)

**Linha 1:**
- **Gap Evolution**: Baseline vs ANT+Gap (alvo: 0.7)
- **ANT Loss**: Loss base do ANT
- **Gap Loss**: Gap maximization loss (só ANT+Gap)

**Linha 2:**
- **Total Loss**: Loss total (NCE + ANT)
- **Loss Components Breakdown**: Bar chart da composição final
- **Hard Negative Violations**: Percentual de violações

### 2. `comparison_performance.png` (2×2 grid)

**Linha 1:**
- **Task Performance (eval_nme1)** 🥇: NME accuracy por task
- **Forgetting Metric (avg_nme1)** 🏆: Média cumulativa (retenção)

**Linha 2:**
- **Linear Classifier (eval_acc1)**: Accuracy do classificador
- **Final Performance**: Bar chart comparativo

### 3. `comparison_correlation.png` (1×2 grid)

- **Gap vs NME1 Scatter**: Correlação entre gap e performance
- **Gap Loss Impact**: Contribuição do gap loss no forgetting

---

## 🔍 Métricas Explicadas

### Loss Components

| Métrica | Descrição | Range | Objetivo |
|---------|-----------|-------|----------|
| **`current_gap`** | `pos_mean - neg_mean` | [-2, 2] | Maximizar (alvo: 0.7) |
| **`pos_mean`** | Similaridade média entre pares positivos | [-1, 1] | ~0.8-0.9 (alto) |
| **`neg_mean`** | Similaridade média entre pares negativos | [-1, 1] | ~0.2-0.4 (baixo) |
| **`ant_loss`** | Loss base (hard negatives) | [0, ∞) | Minimizar |
| **`gap_loss`** | `relu(gap_target - current_gap)` | [0, ∞) | Minimizar |
| **`total_ant_loss`** | `ant_loss + gap_beta * gap_loss` | [0, ∞) | Minimizar |
| **`violation_pct`** | % de negativos com gap < margin | [0, 100] | Minimizar |

### Performance Metrics

| Métrica | Descrição | Importância | Interpretação |
|---------|-----------|-------------|---------------|
| **`eval_nme1`** 🥇 | Top-1 NME accuracy na task atual | **MAIS IMPORTANTE** | Qualidade do espaço de features |
| **`avg_nme1`** 🏆 | Média cumulativa de eval_nme1 | **Forgetting metric** | Retenção de conhecimento |
| **`eval_acc1`** | Top-1 accuracy (linear classifier) | Secundário | Pode ter viés para classes recentes |
| **`avg_acc1`** | Média cumulativa de eval_acc1 | Secundário | Forgetting com classificador linear |

**Por que NME > ACC?**
- **NME (Nearest Mean Exemplar)**: Classifica por distância às médias de classe
  - Mais robusto a catastrophic forgetting
  - Não requer retreinamento do classificador
  - Melhor indicador da qualidade do embedding
- **Linear Classifier**: Pode ser enviesado para classes recentes

---

## 📊 Interpretação dos Resultados

### Gap Statistics

```
📊 Gap Statistics:
Metric               Baseline        ANT+Gap         Improvement    
-----------------------------------------------------------------
Mean                 -0.3500         0.1200          +434.29%
Max                  0.0000          0.6990          +inf%
Final (last epoch)   -0.0500         0.6990          +1498.00%
```

**✅ Bom sinal:**
- ANT+Gap mean > Baseline mean (gap maior)
- ANT+Gap final ≥ 0.7 (atingiu target)
- Improvement > 100% (melhoria significativa)

**⚠️ Atenção:**
- Gap final < 0.7 (não atingiu target)
- Baseline > ANT+Gap (gap maximization não funcionou)

### Performance Comparison

```
🏁 Final Task (10) Performance:
Metric               Baseline        ANT+Gap         Difference     
-----------------------------------------------------------------
🥇 avg_nme1          75.55           78.20            +2.65
eval_nme1           63.51           66.80            +3.29
avg_acc1            79.04           80.50            +1.46
eval_acc1           70.36           72.10            +1.74
```

**✅ Bom sinal:**
- avg_nme1 (ANT+Gap) > avg_nme1 (Baseline) — Menos forgetting
- eval_nme1 (ANT+Gap) > eval_nme1 (Baseline) — Melhor performance atual
- Diferença positiva em todas métricas

**⚠️ Atenção:**
- avg_nme1 degradou — Mais forgetting com ANT+Gap
- Trade-off: eval_nme1 melhor mas avg_nme1 pior

### Forgetting Analysis

```
📉 Forgetting Analysis (First Task → Last Task):
  Baseline NME1 drop:  93.20 → 63.51 = -29.69
  ANT+Gap NME1 drop:   93.20 → 66.80 = -26.40
  ✅ ANT+Gap reduces forgetting by 11.08%
```

**Cálculo:**
```
forgetting_reduction = (baseline_drop - ant_gap_drop) / baseline_drop * 100
                     = (29.69 - 26.40) / 29.69 * 100
                     = 11.08%
```

**✅ Bom sinal:**
- Redução de forgetting > 5% (melhoria significativa)
- Drop absoluto menor no ANT+Gap

### Correlation Analysis

```
📈 Correlation (Gap vs NME1 in ANT+Gap): 0.850
   ✅ Strong positive correlation - Higher gap → Better performance
```

**Interpretação:**
- **Correlation > 0.7**: Forte correlação positiva — Gap maximization efetivo
- **Correlation 0.4-0.7**: Correlação moderada — Gap ajuda mas não é determinante
- **Correlation 0.2-0.4**: Correlação fraca — Gap tem pouco impacto
- **Correlation < 0.2**: Sem correlação — Gap não relacionado com performance
- **Correlation < 0**: Correlação negativa ⚠️ — Gap maior prejudica performance!

---

## 🎯 Cenários de Análise

### Cenário 1: Gap Maximization Funcionou

```
✅ Gap Improvement: ANT+Gap achieves +450% higher average gap
✅ Target Achievement: ANT+Gap reaches gap target (0.7)
✅ Performance: ANT+Gap improves avg_nme1 by +2.50%
📈 Correlation (Gap vs NME1): 0.75 - Strong positive correlation
```

**Conclusão**: Gap maximization efetivo! Maior separação → Melhor performance e menos forgetting.

### Cenário 2: Gap OK, Performance Degradou

```
✅ Gap Improvement: ANT+Gap achieves +300% higher average gap
✅ Target Achievement: ANT+Gap reaches gap target (0.7)
⚠️ Performance: ANT+Gap degrades avg_nme1 by -1.20%
📈 Correlation (Gap vs NME1): -0.15 - Weak negative correlation
```

**Conclusão**: Gap alto mas performance pior. Possíveis causas:
- `gap_beta` muito alto (dominando loss)
- `gap_target` muito agressivo (forçando overfitting)
- Gap maximization conflitando com InfoNCE

**Ações**:
- Reduzir `gap_beta` (1.0 → 0.5)
- Reduzir `gap_target` (0.7 → 0.5)
- Analisar loss components (gap_loss dominando?)

### Cenário 3: Gap Não Atingiu Target

```
⚠️ Neither experiment reaches gap target (0.7)
   Baseline: -0.05 | ANT+Gap: 0.45
✅ Performance: ANT+Gap improves avg_nme1 by +1.50%
```

**Conclusão**: Gap não atingiu 0.7 mas ainda há melhoria. Considere:
- Aumentar `gap_beta` (0.5 → 1.0) para forçar mais
- Ou reduzir `gap_target` (0.7 → 0.5) para ser mais realista
- Gap 0.45 já pode ser suficiente se performance melhorou

---

## 🔧 Tuning de Hiperparâmetros

### `gap_target`

| Valor | Quando Usar | Trade-off |
|-------|-------------|-----------|
| **0.5** | Gap atual ~ 0.3-0.4 | Conservador, menos agressivo |
| **0.7** | Gap atual ~ 0.5-0.6 | **Recomendado** (padrão atual) |
| **1.0** | Gap atual ≥ 0.7 | Agressivo, pode causar overfitting |

### `gap_beta`

| Valor | Quando Usar | Trade-off |
|-------|-------------|-----------|
| **0.3** | Gap loss dominando total loss | Reduz impacto, foca em InfoNCE |
| **0.5** | **Recomendado** (padrão atual) | Balanceado |
| **1.0** | Gap não crescendo suficiente | Aumenta impacto, pode dominar loss |

### Processo de Tuning

1. **Experimento baseline**: `gap_target=0.7`, `gap_beta=0.5`
2. **Analisar resultados**:
   - Gap final < 0.5? → Aumentar `gap_beta` para 1.0
   - Gap final ≥ 0.7 mas performance pior? → Reduzir `gap_beta` para 0.3
   - Gap final ~0.5-0.6? → OK, manter ou aumentar `gap_target` para 0.8
3. **Revalidar** com novo experimento

---

## 📁 Estrutura de Logs

### `exp_matrix_debug0.log`

```
[T1 E1 B1] ANT distance stats: pos_mean: 0.9019 | neg_mean: 0.8939 | 
gap_mean: -0.0504 | margin: 0.1000 | violation_pct: 86.15% | 
ant_loss: 4.9066 | gap_loss: 0.6920 | current_gap: 0.0080 | 
gap_target: 0.7000 | total_ant_loss: 5.2526

[T1 E1 B1] Loss components: contrast_infoNCE_nll: 5.5234 | 
contrast_infoNCE_ant_loss: 4.9066 | contrast_infoNCE_gap_loss: 0.6920 | 
contrast_infoNCE_total_ant_loss: 5.2526 | 
contrast_infoNCE_nce_weighted: 5.5234 | 
contrast_infoNCE_ant_weighted: 5.2526 | 
contrast_infoNCE_total: 10.7760
```

### `exp_gistlog.log`

```
R0T[1/10]E[200/200] train
├> eval_acc1 93.40 eval_acc5 99.80 eval_nme1 93.20 eval_nme5 99.90 
   avg_acc1 93.40 avg_acc5 99.80 avg_nme1 93.20 avg_nme5 99.90
├> eval_acc1_per_task [93.40]
├> eval_nme1_per_task [93.20]
├> acc1_curve [93.40]
└> nme1_curve [93.20]
```

---

## 🚀 Exemplo Completo

```bash
# 1. Rodar experimentos
python main.py --config configs/all_in_one/cifar100_10-10_tagfex_baseline_resnet18.yaml
python main.py --config configs/all_in_one/cifar100_10-10_tagfex_ant_gap_resnet18.yaml

# 2. Aguardar conclusão (10 tasks)

# 3. Analisar
python analysis_scripts/compare_experiments.py \
    --baseline-dir logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal \
    --ant-gap-dir logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5 \
    --output analysis_results_v1

# 4. Verificar resultados
ls analysis_results_v1/
# comparison_losses.png
# comparison_performance.png
# comparison_correlation.png

# 5. Interpretar console output e gráficos

# 6. Se necessário, ajustar hiperparâmetros e repetir
```

---

## ❓ FAQ

### Q: O experimento ainda não terminou, posso analisar?

**A:** Sim! O script funciona com experimentos parciais. Apenas mostrará análise até as tasks completadas.

### Q: Baseline tem gap negativo, é normal?

**A:** Sim. Sem gap maximization, é comum que `neg_mean` seja maior que `pos_mean` inicialmente, resultando em gap negativo. ANT+Gap força gap positivo.

### Q: Correlation é negativa, o que fazer?

**A:** Correlação negativa indica que gap maior piora performance. Possíveis ações:
1. Reduzir drasticamente `gap_beta` (ex: 1.0 → 0.3)
2. Reduzir `gap_target` (ex: 0.7 → 0.5)
3. Verificar se ANT está conflitando com InfoNCE

### Q: Gap atingiu target mas avg_nme1 pior?

**A:** Trade-off comum. Gap maximization pode estar:
- Forçando embeddings sub-ótimos
- Dominando a loss total (verificar % de gap_loss)
- Considere reduzir `gap_beta` para balancear melhor

### Q: Como saber se gap_target=1.0 é viável?

**A:** Analise o experimento com 0.7 primeiro:
- Se gap final < 0.5: Não tente 1.0 ainda, aumente `gap_beta`
- Se gap final ~0.7 e performance OK: Teste 0.8 ou 0.9 primeiro
- Se gap final ~0.9 e performance excelente: Tente 1.0

---

## 📚 Referências

- `CHECKPOINT_GAP_MAXIMIZATION.md`: Documentação da implementação
- `EXPERIMENTOS_COMPARATIVOS.md`: Guia de configuração de experimentos
- `methods/tagfex/tagfex.py`: Implementação das losses
- `loggers/loguru.py`: Logging de todas métricas
