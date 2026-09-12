# Logs e observação dos experimentos

Guia de observabilidade revisado em 11/09/2026. Os treinos atuais usam
serviços de usuário e o launcher com `--no-screen`. Screen é uma opção do
launcher; não é uma propriedade de todos os experimentos.

## Artefatos e consumidores

| Artefato | Conteúdo e uso |
|---|---|
| `exp_gistlog.log` | Curvas e métricas de avaliação; entrada dos relatórios |
| `exp_stdlog<rank>.log` | Configuração e acompanhamento textual do treino |
| `exp_debug<rank>.log` | Componentes de loss e diagnósticos textuais legados |
| `similarity_debug.log`, `similarity_heatmaps/` | Matrizes e visualização detalhada amostrada |
| `<log_dir>/provenance/` | Manifestos de execução, quando capturados com sucesso |
| Log do orquestrador e consoles por run | Lançamento, skip, erro e execução de cada entrada |
| Artefatos do observer ANT | Streams JSONL comprimidos, snapshots NPZ e manifestos próprios |

Nomes e presença variam por revisão e configuração. `exp_matrix_debug0.log`
é encontrado em formatos antigos. Consulte o
[mapa operacional](OPERATIONAL_DEPENDENCIES.md) para caminhos reais; `logs/`
em Xavier contém tanto treino local quanto resultados coletados remotamente.

## Diagnóstico textual e matrizes

Exemplo de opções de um overlay de diagnóstico para uma nova execução:

```yaml
legacy_debug_metrics: true
debug_similarity: true
debug_similarity_epoch_interval: 10
debug_similarity_batches_per_epoch: 1
```

`legacy_debug_metrics` controla o logger textual de métricas históricas e
seus diagnósticos; `debug_similarity` controla a observação detalhada de
matrizes. Desativar somente heatmaps não elimina todo o custo de logging.
O tamanho do batch é o do treinamento; não se deve chamar esse diagnóstico
de smoke sem também limitar o protocolo.

Para criar um run, a sintaxe é `python main.py train --exp-configs` seguida
do YAML base e dos overlays, em ordem. Os exemplos de `main.py --config`
do guia anterior não correspondem à interface atual. Não alterar uma fila
aberta ou seus YAMLs para habilitar diagnóstico durante uma campanha.

## Observação estruturada do mecanismo

O [observer ANT](../studies/ant_mechanism/README.md) registra geometria,
gradientes, atualizações, avaliações e metadados. Suas configs desativam os
diagnósticos textuais legados e amostram streams/snapshots. O documento do
estudo define chaves, frequências, limites de armazenamento e critérios de
validação, incluindo a distinção entre variante treinada e variantes shadow.

O termo ANT bruto tem piso de contagem na formulação logsumexp. Interprete
loss ajustada, proporção ativa, gaps e gradientes em conjunto; plateau não
prova inatividade. Veja a [formulação](ANT_METHOD.md).

## Análise de logs existentes

```bash
python plot_loss_components.py --help
python analysis/scripts/compare_experiments.py --help
```

O comparador é legado de ANT+Gap. Para arquivos modernos, use os quatro
caminhos explícitos: `--baseline-matrix`, `--baseline-gist`,
`--ant-gap-matrix`, `--ant-gap-gist`, além de `--output`.
Os atalhos `--baseline-dir`/`--ant-gap-dir` procuram
`exp_matrix_debug0.log`, apesar do nome moderno mostrado em parte do help.
Não interpretar automaticamente rótulos históricos como métodos atuais.

O viewer é `analysis/scripts/similarity_viewer.py`, executável via Streamlit.
Ele ainda contém uma raiz de logs específica do ambiente legado. Confira-a
antes de usar; migração da ferramenta está pendente. Os scripts de análise
não formam ainda uma única biblioteca de parsing; resultados citados devem
usar seleção explicitamente validada.

## Operação e diagnóstico

Use o [guia de filas](EXPERIMENT_QUEUE_GUIDELINES.md) para localizar serviços,
recursos e falhas. Utilização baixa não prova GPU disponível: memória, outros
processos e política do host também contam. Uma lista de processos obtida em
container/isolamento pode não enxergar o PID mostrado pela GPU.
Não apagar locks, mover diretórios de logs ou reiniciar ambientes para
“limpar” uma campanha; identifique seus produtores e consumidores primeiro.
