# Edições estatísticas e limitações de proveniência

- `analysis/results/statistics/current/`: relatório, observações, seleção,
  tabela e figuras da comparação histórica de nove métodos, p≈0,506.
- `ANT_Monografia/figuras/data/desempenho/autorank_avgacc_principais_refdetach.txt`:
  comparação distinta com RefDetach, p≈0,0976.

As duas seleções não são intercambiáveis. Preserve os arquivos numéricos,
métodos, seeds e entradas associados a cada texto; não “corrija” um p-valor
copiando o de outra seleção.

Na auditoria de 11/09, o manifesto estatístico apontava para outra versão de
`statistical_tests.py`; o inventário de logs de agosto apontava para arquivos
ausentes e outra versão de `provenance_cli.py`. Não foram reescritos hashes,
manifestos ou relatórios para forçar verificação. Uma nova análise deve criar
outra edição, mantendo esta recuperável. Veja [proveniência](../PROVENANCE.md).
