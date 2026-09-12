# Registros históricos e edições de resultados

Os documentos desta coleção preservam evidências e decisões. Não representam
status de execução atual nem receitas para novas campanhas. Blocos citados
mantêm o texto anterior, inclusive conclusões que precisam das ressalvas
editoriais no início de cada documento. Links internos do texto citado são
referências à localização original, não navegação atual.

## Evolução da pesquisa

| Registro | Escopo |
|---|---|
| [README anterior](README_2026-06.md) | Tabelas e estado declarado de junho de 2026, com revisões posteriores |
| [Resultados de 2025](results_2025.md) | Rankings históricos; não constituem recomendações atuais |
| [Hipótese ANT+Gap](ant_gap_hypothesis_2025.md) | Motivação e previsões exploratórias anteriores à avaliação |
| [Resultados ANT+Gap](ant_gap_results_2025.md) | Comparação e correção do bug que apresentava gap zero no baseline |
| [Plano de achatamento da loss](ant_loss_flattening_investigation_plan.md) | Hipóteses, diagnósticos e matriz experimental original |
| [Resultados de maio de 2026](ant_investigation_results.md) | Comparação de formulações com uma seed |

A nota redundante `DESCOBERTA_INFONCE_MAXIMIZA_GAP.md` foi incorporada ao
contexto da comparação ANT+Gap. Nenhuma conclusão de “ANT desnecessária” é
generalizada a outros protocolos. O documento canônico da implementação é
[ANT_METHOD.md](../ANT_METHOD.md).

Registros complementares: [50-10](cifar100_50-10_additional_report.md),
[interpretação NME original](nme1_interpretation_2025.md) e
[edições estatísticas](statistical_editions.md).

## Resultados mantidos nos caminhos existentes

- [Diagnóstico do baseline](../../analysis/results/baseline_metrics/README_APRESENTACAO.md).
- [Comparação histórica baseline/local](../../analysis/results/baseline_vs_local/comparison_report.md).
- [Comparação 50-10 com escala consistente](../../analysis/results/cifar100_50-10_comparison/comparison_report.md).
- [Ablação de margens com dez tarefas](../../analysis/results/three_experiments_comparison/three_margin_comparison_report.md).
- [ANT+Gap](../../analysis/results/experiments_comparison/README.md).
- [Curvas NME históricas](../../analysis/results/nme1_curves/README.md).
- [Estatística sem a seleção RefDetach](../../analysis/results/statistics/current/report.txt).
- [Mecanismo ANT, setembro](../../analysis/results/ant_mechanism_20260909/ANALISE_RESULTADOS.md).

## Erratas e remoções documentais

Em 11/09/2026 foram retirados dois relatórios gerados defeituosos:

1. `cifar100_50-10_comparison_report.md`: apresentava 7011%/6935%, em vez
   de 70,11%/69,35%; a versão de escala consistente permanece no índice acima.
2. `three_experiments_report.md`: o ganho anunciado de +9,97 pp comparava
   baseline em T10 (63,51%) com ANT m=0,5 em T6 (73,48%). A ablação completa
   posterior traz ANT m=0,5 em T10 com 63,74%. Não era um ganho válido entre
   experimentos completos no mesmo horizonte.

`analysis/scripts/STATUS.md` descrevia o diagnóstico antigo em que o gap do
baseline aparecia como zero. A correção foi preservada no registro ANT+Gap;
o estado transitório e a árvore redundante `analysis/STRUCTURE.txt` foram
retirados. Fontes, logs, tabelas numéricas e figuras não foram apagados por
essa revisão documental. Os relatórios rastreados anteriores permanecem no
histórico Git. `results_report.md` era uma saída corrente não rastreada; uma
cópia de segurança foi transferida para `.local/documentation-review/removed/`
na raiz do projeto. Esse diretório local é ignorado pelo Git e restrito ao
proprietário (diretórios 700, arquivos 600); não faz parte da distribuição.
