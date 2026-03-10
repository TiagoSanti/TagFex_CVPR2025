# ANT Loss Analysis Scripts

Scripts para análise do comportamento da ANT (Adaptive Negative Thresholding) loss e gap maximization.

## Scripts Disponíveis

### `analyze_ant_gaps_simple.py`
Script simplificado focado na análise do gap (pos_mean - neg_mean) vs margin.

**Uso:**
```bash
python analyze_ant_gaps_simple.py
```

**Funcionalidades:**
- Análise focada do gap entre positivos e negativos
- Comparação gap vs margin threshold
- Visualizações simplificadas
- Útil para análise rápida de experimentos

### `analyze_ant_gaps.py`
Script completo para análise detalhada das estatísticas de distância do ANT.

**Funcionalidades:**
- Parse completo dos logs de debug
- Análise de positivos (mean, std, min, max)
- Análise de negativos (mean, std, min, max)
- Análise de gaps e violações
- Visualizações abrangentes
- Análise multi-experimento
- Validação de hipóteses

### `reference_enhanced_ant_loss.py`
Implementação de referência com estratégias avançadas de ANT loss.

**Inclui:**
- Gap maximization loss
- Adaptive margin
- Hard negative mining
- Documentação completa de cada estratégia

**Nota:** Este é um arquivo de referência. A implementação ativa está em `methods/tagfex/tagfex.py`.

## Configuração Atual

A implementação ativa usa:
- **Gap Maximization Loss**: `gap_loss = F.relu(gap_target - current_gap)`
- **Config parameters**: `gap_target` e `gap_beta` em `configs/all_in_one/cifar100_10-10_tagfex_ant_resnet18.yaml`
- **Logging completo**: Todas as métricas em `loggers/loguru.py`

## Análise de Resultados

Para analisar os resultados de um experimento:

1. Localize o arquivo de log: `logs/<exp_name>/exp_matrix_debug0.log`
2. Execute o script de análise apropriado
3. Analise as métricas:
   - `current_gap`: Gap atual entre positivos e negativos
   - `gap_target`: Alvo configurado
   - `gap_loss`: Penalização quando gap < target
   - `total_ant_loss`: ant_loss + gap_beta * gap_loss

## Métricas Importantes

- **pos_mean**: Similaridade média entre âncora e positivos (quanto maior, melhor)
- **neg_mean**: Similaridade média entre âncora e negativos (quanto menor, melhor)
- **current_gap**: pos_mean - neg_mean (quanto maior, melhor separação)
- **violation_pct**: % de pares que violam a margin (alto = muitos hard negatives)
- **ant_loss**: Loss ANT original (só ativa quando gap < margin)
- **gap_loss**: Loss de maximização (ativa quando gap < gap_target)
