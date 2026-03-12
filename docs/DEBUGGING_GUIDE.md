# Similarity Matrix Debugging Guide

## Overview

Este recurso permite analisar detalhadamente as matrizes de similaridade durante o treinamento com InfoNCE e ANT Loss. É especialmente útil para:

- Entender o comportamento da loss ANT
- Visualizar como o margin afeta as amostras
- Identificar violações de margem
- Comparar estratégias global vs local de normalização

## Como Habilitar

### 1. Configuração Básica

Adicione os seguintes parâmetros ao seu arquivo de configuração YAML:

```yaml
# Habilitar debug de similaridade
debug_similarity: true

# Controle de sampling: 1 batch por época (uniforme ao longo do treino)
debug_similarity_epoch_interval: 1    # log a cada N épocas (1 = toda época)
debug_similarity_batches_per_epoch: 1 # número de batches a registar por época selecionada
```

O batch de debug usa o **mesmo batch size do treino real** (sem override), garantindo que as matrizes reflitam o comportamento real do modelo.

### 2. Configs de Referência

Dois arquivos de configuração prontos para experimentos de debug em CIFAR-100 10-10:

| Config                                                                              | Descrição                           |
| ----------------------------------------------------------------------------------- | ----------------------------------- |
| `configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_debug_resnet18.yaml` | ANT β=0.5, margin=0.5, anchor local |
| `configs/all_in_one/cifar100_10-10_baseline_local_debug_resnet18.yaml`              | Baseline InfoNCE puro, anchor local |

### 3. Executar Experimento

```bash
# ANT com debug
python main.py --config configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_debug_resnet18.yaml

# Baseline com debug
python main.py --config configs/all_in_one/cifar100_10-10_baseline_local_debug_resnet18.yaml
```

## Outputs Gerados

Quando `debug_similarity: true`, dois tipos de outputs são criados no diretório de logs:

### 1. `similarity_debug.log`

Arquivo de texto com análise detalhada de cada batch selecionado:

```
================================================================================
SIMILARITY MATRIX DEBUG - T1_E1_B1_contrast
Matrix shape: (16, 16)
ANT margin: 0.1, Max strategy: Local
================================================================================

Local max per anchor: min=0.4523, max=0.8912, mean=0.6234

--- Anchor 0 ---
  Positive pair (idx 8): 0.9245
  Anchor max: 0.7832, Threshold: 0.6832
  Above threshold: 3, Below threshold: 6
  Values ABOVE threshold:
    idx 3: 0.7245 (gap: +0.0413)
    idx 5: 0.6945 (gap: +0.0113)
  Top values BELOW threshold:
    idx 2: 0.6521 (gap: -0.0311)
...
```

**Informações por âncora:**
- **Positive pair**: Similaridade com o par positivo (aumento)
- **Anchor max**: Máximo negativo para esta âncora
- **Threshold**: max - margin (valores acima disso violam a margem)
- **Above threshold**: Amostras que violam a margem (contribuem para loss ANT)
- **Below threshold**: Amostras bem separadas (não contribuem significativamente)

### 2. `exp_debug0.log`

Log compacto com estatísticas por batch (todas as épocas, todos os batches):

- `[T E B] Loss components: ...` — componentes individuais da loss (NCE, ANT, total)
- `[T E B] ANT distance stats: ...` — estatísticas de gap e violações
- `[T E B] Contrastive stats: ...` — pos_mean, neg_mean, gap médio

Este log é a fonte do **Training Overview** no Similarity Viewer.

### 3. Logs Padrão

Os logs normais (`exp_stdlog0.log`, `exp_gistlog.log`) contêm apenas mensagens de nível INFO/SUCCESS — mensagens de debug de similaridade são excluídas automaticamente via filtro Loguru (bind `sim_debug=True`).

## Similarity Viewer

A ferramenta de análise interativa está em `analysis/scripts/similarity_viewer.py`:

```bash
streamlit run analysis/scripts/similarity_viewer.py
```

Duas tabs:
- **Training Overview** — curvas de loss, gap, violations por época a partir do `exp_debug0.log`
- **Batch Inspector** — heatmap interativo, stats por âncora e evolução de loss para cada entrada do `similarity_debug.log`

O Batch Inspector suporta navegação por Task/Epoch/Batch com teclado (← → ↑ ↓).

## Análise dos Resultados

### Verificar Comportamento do Margin

**Objetivo**: Confirmar que apenas amostras próximas ao máximo são atualizadas.

1. Abra `similarity_debug.log`
2. Para cada âncora, verifique:
   - Quantas amostras estão "Above threshold" (devem ser poucas)
   - Os gaps das amostras above/below threshold
   - Se o positive pair está bem acima do threshold

**Esperado com ANT funcionando:**
- Poucas violações de margem (< 20% das amostras)
- Gaps positivos pequenos para violations
- Positive pairs com similaridade >> threshold

### Comparar Global vs Local

Execute dois experimentos:

```yaml
# Experimento 1: Global
ant_max_global: true

# Experimento 2: Local
ant_max_global: false
```

**Compare:**
- Número de violações (local deve ter mais balanceamento)
- Distribuição dos gaps
- Convergência da loss

### Analisar Heatmaps

**Padrões Bons:**
- Diagonal bem definida (alta auto-similaridade)
- Pares positivos (quadrante oposto) em verde/amarelo
- Maioria dos negativos em vermelho/laranja
- Poucos valores verdes fora da diagonal/pares positivos

**Padrões Problemáticos:**
- Muitos valores verdes nos negativos → margin muito grande
- Todos valores vermelhos → modelo não está aprendendo
- Pares positivos em vermelho → augmentations muito fortes

## Estimativa de Armazenamento

Com `debug_similarity_epoch_interval: 1` e `debug_similarity_batches_per_epoch: 1` (1 batch por época):

| Arquivo                   | Estimativa (CIFAR-100 10-10, bs=128) |
| ------------------------- | ------------------------------------ |
| `similarity_debug.log`    | ~3.5 GB                              |
| `exp_debug0.log`          | ~855 MB                              |
| `exp_stdlog0.log`         | ~0 MB (filtrado)                     |
| **Total por experimento** | **~4.4 GB**                          |

Para 2 experimentos (ANT + baseline): ~8.8 GB.

## Ajuste de Hiperparâmetros

### Margin (ant_margin)

- **Muito pequeno** (< 0.05): Muitas violações, loss alta
- **Ideal** (0.3 - 0.6): 10-30% de violações
- **Muito grande** (> 0.8): Poucas/nenhuma violação, loss → 0

**Como verificar**: Olhe "violation_pct" no log ou no Training Overview.

### Beta (ant_beta)

Controla o peso da loss ANT:

- **0.0**: Apenas InfoNCE puro
- **0.5**: Balance entre InfoNCE e ANT
- **1.0**: ANT dominante

**Recomendação**: Comece com 0.5 e ajuste baseado nos resultados.

## Troubleshooting

### Log de similaridade vazio

Verificar se `debug_similarity: true` está no YAML e se o diretório de logs tem permissão de escrita.

### Muitos arquivos de heatmap

Os heatmaps em PNG foram removidos do pipeline de debug. O Batch Inspector no Similarity Viewer renderiza os heatmaps interativamente a partir do `similarity_debug.log`, sem gerar ficheiros PNG.

### Debug muito lento

Aumentar `debug_similarity_epoch_interval` para amostrar menos épocas:

```yaml
debug_similarity_epoch_interval: 10  # só log a cada 10 épocas
debug_similarity_batches_per_epoch: 1
```

As primeiras e últimas épocas de cada task são sempre incluídas independentemente do intervalo.

## Exemplos de Uso

### Caso 1: Validar implementação ANT

```yaml
debug_similarity: true
debug_similarity_epoch_interval: 1
debug_similarity_batches_per_epoch: 1
ant_beta: 0.5
ant_margin: 0.5
ant_max_global: false
```

**O que verificar no Similarity Viewer:**
- `violation_pct` no Training Overview — deve ser 10–30%
- Heatmap no Batch Inspector — pares positivos em verde, maioria dos negativos em vermelho
- Loss ANT×β positiva e decrescendo ao longo das épocas

### Caso 2: Comparar ANT vs Baseline

Use os dois configs de debug prontos:

```bash
# ANT
python main.py --config configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_debug_resnet18.yaml

# Baseline
python main.py --config configs/all_in_one/cifar100_10-10_baseline_local_debug_resnet18.yaml
```

Depois abra o Similarity Viewer apontando para cada diretório de logs.

### Caso 3: Ajustar margin

Mude `ant_margin` no config ANT de debug (0.3, 0.5, 0.7) e compare o `violation_pct` médio no Training Overview.

## Limitações

1. **Performance**: Debug adiciona overhead (~10-20% mais lento) devido ao log das matrizes
2. **Espaço em disco**: ~4.4 GB por experimento com configuração padrão (1 batch/época, bs=128)
3. **Apenas treinamento**: Debug não funciona durante avaliação

## Dicas

1. **Use o Similarity Viewer** para navegar — muito mais eficiente do que ler os logs directamente
2. **Compare Training Overview entre experimentos** para ver o efeito do ANT nas curvas de loss
3. **Use `debug_similarity_epoch_interval`** para reduzir o volume de dados se necessário
4. **Desabilite após validação**: `debug_similarity: false` para experimentos de produção

## Próximos Passos

Após validar o comportamento:

1. Desabilite debug: `debug_similarity: false`
2. Execute experimento completo com o config não-debug correspondente
3. Use `analysis/scripts/similarity_viewer.py` para análise post-hoc se necessário
