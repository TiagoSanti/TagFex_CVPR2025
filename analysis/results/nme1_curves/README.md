# Comparação de Performance: NME1 Curves de Todos os Experimentos

Análise comparativa da métrica `nme1_curve` (NME Top-1 Accuracy) de todos os experimentos em `logs/`.

---

## 📊 Visualizações Geradas

### 1. **`nme1_curves_comparison.png`** - Comparação Unificada
- **Todas as curvas em um único gráfico**
- Cada experimento com cor e marcador distintos
- Legenda descritiva com configurações principais
- Ideal para apresentar visão geral comparativa

**Experimentos plotados** (8 total):
1. TagFex Original
2. Baseline (InfoNCE puro, ant_β=0)
3. ant_β=0.5, Local
4. ant_β=0.5, Global
5. ant_β=1.0, Local
6. ant_β=1.0, Global
7. ant_β=1.0, Local, margin=0.2
8. ant_β=1.0, Local, gap_target=0.7, gap_β=0.5

---

### 2. **`nme1_curves_detailed.png`** - Análise Detalhada
- **Subplots individuais** para cada experimento
- Valores anotados em cada ponto
- Performance final destacada com linha tracejada
- Melhor para análise individual e comparação lado a lado

---

### 3. **`nme1_summary.txt`** - Tabela Estatística
- Resumo quantitativo de todos os experimentos
- Métricas: Final, Média, Desvio Padrão, Max Drop
- Análise de forgetting (Task 1 → Final)
- Ranking por performance final

---

## 🏆 Resultados Principais

### Performance Final (Task 10)

| Ranking | Experimento | NME1 Final | Tarefas |
|---------|-------------|------------|---------|
| 🥇 1º | ant_β=1.0, Local | **71.11%** | 7 tasks (incompleto) |
| 🥈 2º | ant_β=1.0, Global | **71.04%** | 7 tasks (incompleto) |
| 🥉 3º | ant_β=1.0, Local, margin=0.2 | **67.00%** | 8 tasks (incompleto) |
| 4º | ant_β=0.5, Local | 63.83% | 10 tasks ✓ |
| 5º | ant_β=0.5, Global | 63.72% | 10 tasks ✓ |
| 6º | TagFex Original | 63.51% | 10 tasks ✓ |
| 7º | Baseline (ant_β=0) | 63.51% | 10 tasks ✓ |
| 8º | ant_β=1.0 + gap_loss | 63.30% | 10 tasks ✓ |

**Nota**: Experimentos com prefixo `idone_` estão incompletos (7-8 tasks), então comparação direta não é totalmente justa.

---

## 📉 Análise de Forgetting

**Forgetting = Task 1 Accuracy - Final Accuracy**

| Experimento | Forgetting | Status |
|-------------|------------|--------|
| ant_β=1.0, Local | **22.09%** | ✅ Menor forgetting (incompleto) |
| ant_β=1.0, Global | **22.16%** | ✅ Menor forgetting (incompleto) |
| ant_β=1.0, margin=0.2 | 26.20% | (incompleto) |
| ant_β=0.5, Local | 29.37% | Experimentos completos (10 tasks) |
| ant_β=0.5, Global | 29.48% | |
| TagFex Original | 29.69% | |
| Baseline (ant_β=0) | 29.69% | |
| ant_β=1.0 + gap_loss | **29.90%** | ⚠️ Maior forgetting |

---

## 🔍 Observações Importantes

### 1. Baseline vs Gap Maximization
```
Baseline (ant_β=0):        63.51% final, 29.69% forgetting
Gap Maximization:          63.30% final, 29.90% forgetting
Diferença:                 -0.21% final, +0.21% forgetting
```
**Conclusão**: Gap maximization não melhora (e ligeiramente piora) a performance final.

### 2. TagFex Original = Baseline
```
Ambos têm exatamente os mesmos resultados (63.51%, 29.69%)
```
**Conclusão**: Confirma que o experimento original tinha ant_β=0 (baseline).

### 3. ant_β=0.5 vs ant_β=1.0
```
ant_β=0.5:  ~63.8% final (completo, 10 tasks)
ant_β=1.0:  ~71.1% final (incompleto, 7 tasks)
```
**Observação**: ant_β=1.0 parece melhor, mas experimentos não completaram as 10 tasks. Difícil comparar diretamente.

### 4. Local vs Global
```
Em ambos ant_β=0.5 e ant_β=1.0:
- Local e Global têm performance muito similar
- Diferença < 0.5% na maioria dos casos
```
**Conclusão**: Escolha entre Local/Global tem impacto mínimo.

### 5. Margin 0.1 vs 0.2
```
ant_β=1.0, margin=0.1:  71.11% (Task 7)
ant_β=1.0, margin=0.2:  67.00% (Task 8)
```
**Tendência**: Margin maior (0.2) pode prejudicar performance.

---

## 💡 Insights para o Professor

### Mensagem Principal
> "Comparamos 8 configurações diferentes. O baseline (InfoNCE puro) tem performance equivalente ou superior a variantes com ANT loss e gap maximization quando comparamos experimentos completos (10 tasks)."

### Pontos de Discussão

1. **Gap Maximization não ajuda**
   - Experimento mais elaborado (ant_β=1.0 + gap) tem a pior performance
   - Confirma análise anterior: InfoNCE já maximiza gap naturalmente

2. **Experimentos incompletos parecem melhores**
   - ant_β=1.0 mostra 71% vs 63% do baseline
   - Mas são apenas 7 tasks vs 10 tasks
   - Tasks finais são mais difíceis (catastrofic forgetting acumula)

3. **Consistência dos resultados**
   - Baseline e TagFex Original idênticos (validação)
   - Local vs Global têm impacto mínimo (robustez)
   - ant_β=0.5 intermediário entre 0.0 e 1.0 (esperado)

4. **Forgetting é o gargalo**
   - Todos os métodos perdem ~30% de Task 1 para Task 10
   - Catastrofic forgetting é o problema principal
   - Modificações no loss têm impacto limitado

---

## 📈 Gráficos Recomendados para Apresentação

### Sequência Sugerida

1. **Slide 1**: Mostre `nme1_curves_comparison.png`
   - "Comparação de 8 configurações diferentes"
   - Destaque: curvas praticamente sobrepostas para experimentos completos

2. **Slide 2**: Mostre `nme1_curves_detailed.png` (foque em 4 principais)
   - Baseline, ant_β=0.5 Local, ant_β=1.0 Local, Gap maximization
   - Mostre valores finais anotados

3. **Slide 3**: Tabela de resumo (do txt)
   - Performance final
   - Forgetting analysis
   - Destacar: baseline competitivo

---

## 🚀 Próximas Análises Sugeridas

1. **Completar experimentos incompletos**
   - Rodar ant_β=1.0 até Task 10
   - Comparação justa com baseline

2. **Plotar outras métricas**
   - acc1_curve (top-1 accuracy CNN)
   - Comparar NME vs CNN accuracy

3. **Análise de estabilidade**
   - Desvio padrão entre tasks
   - Variância da performance

4. **Per-task forgetting**
   - Quanto cada task esquece das anteriores
   - Identificar tasks mais problemáticas

---

## 🔧 Como Regenerar

```bash
cd /home/tiago/TagFex_CVPR2025
source .venv/bin/activate

python analysis_scripts/plot_all_nme1_curves.py \
  --logs-dir logs \
  --output analysis_nme1_comparison
```

---

## 📁 Arquivos

```
analysis_nme1_comparison/
├── nme1_curves_comparison.png      ← Gráfico unificado (todos juntos)
├── nme1_curves_detailed.png        ← Subplots individuais
└── nme1_summary.txt                ← Tabela estatística
```

---

**Última atualização**: 18 de Novembro de 2025  
**Script**: `analysis_scripts/plot_all_nme1_curves.py`
