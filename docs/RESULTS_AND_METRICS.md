# Métricas, seleção e interpretação dos resultados

Contrato de leitura dos resultados, revisado em 11/09/2026. As tabelas do
documento anterior foram preservadas como [edição histórica de 2025](history/results_2025.md).
Este guia não declara uma configuração universalmente melhor.

## Métricas

| Campo | Significado |
|---|---|
| `eval_acc1` / `acc1_curve` | Acurácia top-1 do classificador; curva cumulativa ao fim de cada tarefa |
| `eval_nme1` / `nme1_curve` | Acurácia top-1 do classificador por médias de exemplares |
| `avg_acc1` | Média temporal dos valores de `acc1_curve` |
| `avg_nme1` | Média temporal dos valores de `nme1_curve`; não é forgetting |
| Última ACC/NME | Último ponto de uma curva completa no protocolo declarado |
| `eval_acc1_per_task` | Acurácias dos blocos de tarefas já vistos, avaliados no mesmo incremento |

No gerador HTML, para cada tarefa antiga j, forgetting é o melhor valor
histórico daquele bloco menos seu valor na avaliação final. A média exclui
a tarefa recém-introduzida. Isso não equivale a subtrair o último ponto do
primeiro em `acc1_curve`, pois os pontos cumulativos avaliam conjuntos de
classes diferentes. A monografia também apresenta curvas de acurácia
balanceada entre tarefas e de forgetting; suas definições estão na
[documentação dos dados](../ANT_Monografia/figuras/data/README.md).

As tabelas apresentam acurácias em porcentagem. Diferenças entre porcentagens
são pontos percentuais (pp). O parser do gerador lê valores numéricos dos
gistlogs sem multiplicá-los automaticamente por 100; valide a escala da
entrada antes de converter. Desvio entre seeds (`ddof=1`) não é desvio entre
tarefas de uma mesma execução. Com uma seed não se estima variabilidade entre seeds.

## Completude e seleção

Para C classes, base I e incremento K, o protocolo regular tem
`1 + (C−I)/K` tarefas. Exemplo: CIFAR-100 50-10 tem seis tarefas, não onze.
Conferir curvas e histórico contra esse contrato; a existência de
`avg_nme1` ou um marcador de conclusão isolado não basta.

O gerador e a análise estatística selecionam fontes canônicas e mantêm
compatibilidade com logs históricos. Tokens `nGlobal/nLocal` não definem
métodos distintos. Há uma observação por método/protocolo/seed; prioridade
de origem favorece nomes canônicos, depois nGlobal, depois nLocal, com
preferência por nomes não-debug no desempate. TeacherAvg permanece uma
ablação mesmo quando β=0. Sempre conferir a seleção produzida: outros
parâmetros e revisões também podem mudar o significado do experimento.

Não comparar última acurácia de runs com números diferentes de tarefas.
Não tratar épocas, batches ou várias cópias de uma seed como réplicas
independentes. Comparações pareadas precisam da interseção das observações
completas dos métodos incluídos. Métodos adicionais podem mudar essa interseção.

## Artefatos e reprodução

```bash
python generate_html_pdf_report.py --help
python statistical_tests.py --help
python provenance_cli.py --help
```

O relatório HTML/PDF usa `logs/` por padrão e produz um sidecar de
proveniência. `statistical_tests.py` produz observações, seleção, relatório,
tabela, figuras e manifesto. Use um diretório de edição novo ao recalcular
estatísticas; não substitua silenciosamente resultados citados. Veja
[proveniência](PROVENANCE.md) para comandos de produção e verificação.

Um rank descritivo não prova superioridade. Registre métodos, protocolos,
seeds, unidade de pareamento, métrica, teste, α e seleção. Não atribua
significância a uma diferença pequena apenas por ela ser positiva.

## Edições existentes

- [Edição de 12/09](../analysis/results/statistics/editions/20260912/report.txt):
  17 métodos em 12 blocos; o diretório `current` aponta para esta edição.
  Mantém quatro datasets e seeds 1993–1995; a nova campanha CUB de cinco seeds
  não amplia automaticamente esse universo estatístico.

- [Estatística histórica de 11/09](../analysis/results/statistics/editions/20260911/report.txt):
  nove métodos em 12 blocos; p≈0,506. É uma edição histórica
  cujo manifesto se refere aos fontes daquela edição.
- [Estatística com RefDetach](../ANT_Monografia/figuras/data/desempenho/autorank_avgacc_principais_refdetach.txt):
  outra seleção de nove métodos, p≈0,0976. Não substituir os valores da edição
  anterior sem mudar conjuntamente seleção, tabela e narrativa.
- [Estudo observacional ANT](../analysis/results/ant_mechanism_20260909/ANALISE_RESULTADOS.md):
  uma seed, três tarefas e nove condições; finalidade mecanística.

Resultados históricos permanecem úteis, inclusive os inconclusivos. O
[índice histórico](history/README.md) identifica erros conhecidos e limites.
