> Revisão documental de 11/09/2026. Este diretório contém tanto infraestrutura
> reutilizável quanto a definição da campanha. Os caminhos atuais permanecem
> válidos; sua separação é uma proposta de refatoração. Smoke/short iniciam
> treinamentos e requerem host/GPU alocados. Fera e Quati estão inativas segundo
> o responsável. Veja os [resultados preservados](../../analysis/results/ant_mechanism_20260909/ANALISE_RESULTADOS.md),
> a [formulação](../../docs/ANT_METHOD.md) e o [mapa operacional](../../docs/OPERATIONAL_DEPENDENCIES.md).

# Estudo observacional do mecanismo ANT

Este módulo instrumenta o treinamento original do TagFex sem duplicá-lo. O
observador é estritamente opcional (`ant_study.enabled`) e calcula diagnósticos
analíticos a partir de cópias destacadas das matrizes de similaridade. Somente
as sondagens esparsas de gradientes de parâmetros mantêm referências temporárias
ao grafo, antes do `backward` normal.

## Matriz experimental

O estudo cobre o Baseline InfoNCE e as oito combinações:

| Cobertura | Referência | Conectada | Referência destacada |
|---|---|---|---|
| IV (`Q1`) | GR (global) | ANT-IV-GR | ANT-IV-GR-D |
| IV (`Q1`) | AR (por âncora) | ANT-IV-AR | ANT-IV-AR-D |
| FS (matriz válida completa) | GR (global) | ANT-FS-GR | ANT-FS-GR-D |
| FS (matriz válida completa) | AR (por âncora) | ANT-FS-AR | ANT-FS-AR-D |

Em batches sentinela, as oito variantes são avaliadas como *shadow variants*
sobre a mesma matriz. Apenas a variante indicada no YAML participa do
treinamento. Isso separa diferenças instantâneas do operador de diferenças
acumuladas na trajetória do modelo.

## Contrato de coleta

### Identidade do lote

- índice local e índice absoluto da amostra;
- classe original e classe na ordem incremental;
- view 1 ou view 2;
- origem corrente ou replay;
- tarefa, época, batch e ramo (`current` ou `kd`).

### InfoNCE

- loss média, termo de alinhamento e `logsumexp` do denominador;
- similaridade positiva e negativa;
- probabilidade e rank do positivo;
- proporção de âncoras com negativo acima do positivo;
- massa de probabilidade dos negativos e massa top-5;
- entropia e número efetivo de negativos;
- matriz analítica de `dL_NCE/dS` e sua norma.

### ANT

- cobertura, referência e estado do `detach`;
- quantidade de negativos válidos por âncora;
- referência, quantidade de máximos empatados e limiar implícito;
- violação bruta `v=s-r+gamma`, máscara ativa e percentis 90/95;
- quantidade/proporção ativa e proporção de âncoras com gradiente efetivo;
- loss bruta, piso `log(N)` e loss ajustada;
- gradiente na referência, direção de repelir/aproximar e `dL_ANT/dS`;
- norma ANT, norma combinada, razão ANT/InfoNCE, cosseno e acordo de sinais.

### Objetivo completo e parâmetros

- classificação, contraste corrente, KD, auxiliar, transferência e KL;
- pesos externos efetivos dos ramos corrente e KD;
- normas de gradiente InfoNCE/ANT por `ta_net`, projetor, preditor,
  classificador, ramos TS e demais parâmetros;
- cosseno e razão ANT/InfoNCE por grupo;
- norma absoluta e relativa da atualização aplicada pelo otimizador.

### Snapshots

Em tarefas, épocas e batches configurados são preservados, em NPZ comprimido:

- matriz de similaridades;
- probabilidades, máscaras e loss por âncora da InfoNCE;
- pesos de agregação e loss por âncora da ANT;
- gradientes InfoNCE, ANT e combinados;
- máscaras válida, ativa e de referência;
- violações brutas das oito variantes;
- metadados das amostras;
- views transformadas (limitadas a `snapshot_image_count`), embeddings do ramo
  corrente e, a partir da tarefa 2, projeções de estudante e professor do KD.

### Desempenho incremental e proveniência

- acurácia global e por tarefa nas avaliações de época e de fim de tarefa;
- configuração efetiva de treinamento, seed e identidade da condição;
- nomes, ordem e mapeamento incremental das classes do dataset;
- versão do esquema e SHA-256 dos fontes que implementam a instrumentação.

## Organização dos artefatos

Cada condição escreve um diretório independente contendo:

| Artefato | Conteúdo |
|---|---|
| `manifest.json` | proveniência, configuração efetiva e oito variantes shadow |
| `dataset_metadata.json` | classes, ordem incremental e mapeamento de rótulos |
| `geometry_metrics.jsonl.gz` | escalares InfoNCE/ANT por ramo e variante |
| `training_objective.jsonl.gz` | decomposição da loss total e pesos externos |
| `parameter_gradients.jsonl.gz` | normas das parcelas NCE/ANT por grupo de parâmetros |
| `parameter_gradient_alignment.jsonl.gz` | razão de normas e cosseno NCE × ANT |
| `total_parameter_gradients.jsonl.gz` | gradiente completo antes/depois de clipping |
| `parameter_updates.jsonl.gz` | norma absoluta e relativa do passo do otimizador |
| `evaluations.jsonl.gz` | métricas incrementais durante e após cada tarefa |
| `snapshots/*.npz` | matrizes, máscaras, gradientes, imagens e representações |

Os registros escalares usam JSON Lines comprimido para permitir leitura em
fluxo. Matrizes e representações usam NPZ comprimido; imagens e embeddings são
armazenados em `float16`, enquanto os gradientes analíticos permanecem em
`float32`.

## Frequência e armazenamento

Os escalares são amostrados a cada 10 batches por padrão. Snapshots são
restritos ao primeiro batch de épocas sentinela. O observador interrompe apenas
snapshots pesados quando o diretório alcança 10 GiB ou o disco tem menos de
50 GiB livres; os escalares continuam sendo gravados. Os logs textuais legados
devem permanecer desativados (`legacy_debug_metrics: false`).

## Validação

```bash
.venv/bin/python -m unittest discover -s tests \
  -p 'test_ant_mechanism_*.py' -v
```

A suíte compara, para as oito variantes, o gradiente analítico com o autograd,
valida as cardinalidades IV/FS, prova que o `detach` não muda o forward e
reproduz o exemplo numérico da InfoNCE apresentado na monografia.

## Execução

O runner recebe `smoke` ou `short` e usa a GPU indicada em `GPU`:

```bash
GPU=1 bash studies/ant_mechanism/run_matrix.sh smoke
GPU=1 bash studies/ant_mechanism/run_matrix.sh short
```

`smoke` executa duas tarefas, uma época e dois batches por época. `short`
executa as três primeiras tarefas com o cronograma original, usando seed 1993.

Antes de executar em um host novo:

```bash
GPU=1 bash studies/ant_mechanism/preflight.sh
```

Depois dos nove smokes:

```bash
.venv/bin/python -m studies.ant_mechanism.validate_artifacts \
  study_outputs/ant_mechanism_20260909_smoke --expect-runs 9 --require-kd \
  --require-complete-contract
.venv/bin/python -m studies.ant_mechanism.summarize \
  study_outputs/ant_mechanism_20260909_smoke
```

O smoke só é aceito quando as nove condições possuem manifestos e metadados do
dataset, alcançam os ramos corrente e KD, contêm as oito variantes shadow,
produzem todos os streams estruturados e preservam snapshots com imagens,
embeddings, probabilidades, máscaras e gradientes.
