# Curvas NME — edição histórica de novembro de 2025

Este diretório preserva figuras de curvas e o [resumo numérico](nme1_summary.txt).
A interpretação original está no [arquivo histórico](../../../docs/history/nme1_interpretation_2025.md).

O resumo inclui execuções de 7, 8 e 10 tarefas. Compare somente horizontes
compatíveis; uma última NME maior em execução parcial não demonstra ganho.
O desvio ao longo da curva não estima variabilidade entre seeds, e a queda
entre o primeiro e o último ponto cumulativo não é forgetting por tarefa.
Veja o [contrato de métricas](../../../docs/RESULTS_AND_METRICS.md).

O gerador é `analysis/scripts/plot_all_nme1_curves.py`. Seus caminhos e
seleção são históricos; confira-os antes de criar outra edição. Não
sobrescreva as tabelas preservadas com a seleção corrente de `logs/`.
