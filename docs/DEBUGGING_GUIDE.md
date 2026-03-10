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

# Tamanho do batch para debug (cria matrizes 16x16)
debug_similarity_batch_size: 16
```

### 2. Exemplo Completo

Veja o arquivo `configs/exps/debug_similarity_example.yaml` para uma configuração completa.

### 3. Executar Experimento

```bash
python main.py --config configs/exps/debug_similarity_example.yaml
```

## Outputs Gerados

Quando `debug_similarity: true`, três tipos de outputs são criados no diretório de logs:

### 1. `similarity_debug.log`

Arquivo de texto com análise detalhada de cada batch:

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

### 2. `similarity_heatmaps/`

Diretório com visualizações em PNG de cada matriz de similaridade:

- **Nome**: `sim_heatmap_T{task}_E{epoch}_B{batch}_{type}.png`
- **Formato**: Heatmap colorido com valores anotados
- **Elementos visuais**:
  - Cores: Verde (alta similaridade) → Amarelo → Vermelho (baixa similaridade)
  - Valores numéricos em cada célula
  - Linhas azuis tracejadas separando pares originais/aumentados
  - Diagonal destacada (auto-similaridade = 1.0)

**Interpretação do Heatmap:**
- Diagonal principal: sempre 1.0 (auto-similaridade)
- Valores próximos à diagonal: negativos dentro do batch
- Valores no quadrante oposto: pares positivos (augmentations)
- Cores quentes próximas ao threshold: violações de margem

### 3. Logs Padrão

Os logs normais (`loguru_stdlog0.log`, `loguru_debuglog0.log`) continuam contendo:
- Estatísticas básicas de contrastive learning
- Componentes da loss
- Métricas de treinamento

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

## Ajuste de Hiperparâmetros

### Margin (ant_margin)

- **Muito pequeno** (< 0.05): Muitas violações, loss alta
- **Ideal** (0.1 - 0.3): 10-30% de violações
- **Muito grande** (> 0.5): Poucas/nenhuma violação, loss → 0

**Como verificar**: Olhe "violation_pct" no log.

### Beta (ant_beta)

Controla o peso da loss ANT:

- **0.0**: Apenas InfoNCE puro
- **0.5**: Balance entre InfoNCE e ANT
- **1.0**: ANT dominante

**Recomendação**: Comece com 0.5 e ajuste baseado nos resultados.

## Troubleshooting

### Matrizes muito grandes (> 16x16)

**Solução**: Reduza `debug_similarity_batch_size`:

```yaml
debug_similarity_batch_size: 8  # Matrizes 8x8
```

### Muitos arquivos de heatmap

Os heatmaps são salvos para **cada batch**. Em um epoch típico com 391 batches, isso gera 391 imagens.

**Solução 1**: Limitar debug aos primeiros batches (modificar código para condicional)

**Solução 2**: Analisar apenas primeiros epochs:

```yaml
debug: true  # Reduz para 5 epochs
```

### Debug muito lento

**Solução**: 
1. Use `debug_similarity_batch_size: 8` (menor)
2. Rode debug apenas no primeiro task
3. Desabilite após entender o comportamento

## Exemplos de Uso

### Caso 1: Validar implementação ANT

```yaml
debug_similarity: true
debug_similarity_batch_size: 16
ant_beta: 0.5
ant_margin: 0.1
ant_max_global: true
init_epochs: 5  # Apenas alguns epochs para testar
```

**O que verificar:**
- Loss ANT > 0 quando ant_beta > 0
- Violações de margem presentes mas não excessivas
- Heatmaps mostram gradiente de cores

### Caso 2: Comparar estratégias de normalização

Execute dois experimentos lado a lado:

```bash
# Experimento Global
python main.py --config configs/debug_global.yaml

# Experimento Local
python main.py --config configs/debug_local.yaml
```

**Compare os logs:**
```bash
# Verificar distribuição de violations
grep "violation_pct" logs/exp_global/similarity_debug.log
grep "violation_pct" logs/exp_local/similarity_debug.log
```

### Caso 3: Otimizar margin

Teste diferentes valores:

```yaml
# configs/debug_margin_test.yaml
ant_margin: 0.05  # Teste 0.05, 0.1, 0.2, 0.3, 0.5
```

**Analise:** Relação entre margin e violation_pct no log.

## Limitações

1. **Performance**: Debug adiciona overhead (~10-20% mais lento)
2. **Espaço em disco**: Heatmaps podem ocupar muito espaço
3. **Batch size**: Reduzir batch pode afetar convergência
4. **Apenas treinamento**: Debug não funciona durante avaliação

## Dicas

1. **Sempre verifique os primeiros batches**: Comportamento inicial é indicativo
2. **Compare heatmaps entre epochs**: Deve ver evolução (mais separação)
3. **Use grep para extrair estatísticas**: `grep "violation_pct" similarity_debug.log | awk '{print $NF}'`
4. **Desabilite após validação**: Não use em produção/experimentos finais

## Próximos Passos

Após validar o comportamento:

1. Desabilite debug: `debug_similarity: false`
2. Restaure batch size normal
3. Execute experimento completo com configurações otimizadas
