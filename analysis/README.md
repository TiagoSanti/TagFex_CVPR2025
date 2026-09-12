# Análise de resultados

Execute a partir da raiz do repositório, com o ambiente ativado. A interface
principal é `generate_html_pdf_report.py`; a análise estatística usa
`statistical_tests.py`. Os scripts em `analysis/scripts/` incluem leitores
históricos, comparações específicas e visualizações exploratórias.

## Fluxo atual

1. Identificar campanha, protocolo, configuração efetiva e seeds.
2. Conferir completude e selecionar uma observação por identidade.
3. Gerar tabelas/figuras numa edição identificada e preservar entradas.
4. Verificar proveniência e registrar limites antes de citar os resultados.

Consulte [métricas e seleção](../docs/RESULTS_AND_METRICS.md),
[proveniência](../docs/PROVENANCE.md) e
[observabilidade](../docs/DEBUGGING_GUIDE.md).

```bash
python generate_html_pdf_report.py --help
python statistical_tests.py --help
python provenance_cli.py --help
```

## Ferramentas e limites

| Ferramenta | Finalidade |
|---|---|
| `generate_html_pdf_report.py` | Relatório curto/full com seleção e sidecar |
| `statistical_tests.py` | Comparação pareada e artefatos da seleção |
| `analysis/compare_debug_metrics.py` | Figuras de métricas textuais históricas |
| `plot_loss_components.py`, `get_debug_metrics.py` | Leitura de diagnósticos históricos |
| `analysis/scripts/compare_experiments.py` | Comparador legado ANT+Gap |
| `analysis/scripts/similarity_viewer.py` | Viewer Streamlit de matrizes; raiz de logs ainda específica de máquina |
| `studies/ant_mechanism/summarize.py` | Agregação dos streams estruturados |
| `studies/ant_mechanism/validate_artifacts.py` | Validação do contrato do estudo |

O launcher `analysis/run_analysis.sh` exige o nome real do script e muda o
diretório de trabalho para `analysis/scripts/`. Alguns nomes abreviados
mostrados no seu menu não são aliases implementados; prefira a CLI explícita.
`compare_experiments.py --help` permite inspecionar os parâmetros. Os atalhos
de diretório esperam o nome antigo `exp_matrix_debug0.log`; para logs novos,
forneça os quatro caminhos explícitos, como descrito no guia de observabilidade.

Scripts de `local_vs_global`, `baseline_vs_local`, `gap` e
`reference_enhanced_ant_loss.py` são material de investigação histórica.
Não representam uma segunda InfoNCE ativa nem uma implementação canônica de
ANT. Exemplos teóricos antigos de diferença intrínseca nGlobal/nLocal foram
superados; veja a [formulação atual](../docs/ANT_METHOD.md).

## Resultados preservados

- [Índice histórico e erratas](../docs/history/README.md).
- [Mecanismo ANT — setembro de 2026](results/ant_mechanism_20260909/ANALISE_RESULTADOS.md).
- [Estatística histórica](results/statistics/current/report.txt): `current`
  é nome legado; não implica que a edição represente todos os runs presentes.
- [Dados das figuras da monografia](../ANT_Monografia/figuras/data/README.md).

Relatórios atuais são gerados de `logs/`, que recebe escritas locais e coleta
remota. Não renomear esse acervo durante campanhas. Uma figura de comparação
antiga não deve ser reutilizada sem conferir escala, horizonte, fonte e
semântica das métricas.

A retirada dos relatórios defeituosos foi documental. Os geradores históricos
`compare_three_experiments.py` e `compare_50-10_global_vs_local.py` ainda
podem escrever nos nomes antigos; precisam de revisão de seleção/escala antes
de voltarem a alimentar uma edição científica. Não execute esses geradores
por cima dos resultados preservados.

## Implementações canônicas

- `reporting/generate_html_pdf_report.py`: HTML/PDF e seleção de resultados.
- `statistics/statistical_tests.py`: análise estatística reproduzível.

As entradas homônimas na raiz permanecem como compatibilidade de importação e
CLI. ANT detached é canônico e ANT conectado é legado explícito. Edições
estatísticas preservadas ficam em `results/statistics/editions/`; `current` é
um link para a edição de 12/09/2026. A movimentação dos geradores foi validada
sobre logs congelados, sem alterar métricas, seleção ou cobertura analítica.
