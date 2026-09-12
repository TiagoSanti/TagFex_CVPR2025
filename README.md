# TagFex — investigação ANT para aprendizado incremental de classes

> Estado da pesquisa e organização revisados em 12/09/2026. Veja o
> [relatório de integração](docs/RESEARCH_INTEGRATION_20260912.md) e os contratos de retomada.

Extensão de pesquisa do [TagFex original](https://github.com/bwnzheng/TagFex_CVPR2025),
associado ao artigo *Task-Agnostic Guided Feature Expansion for Class-Incremental
Learning* (CVPR 2025). O núcleo da investigação é a ANT (*Avoid Non-essential
Tuning*), complementada por TeacherAvg-K e Speed-Based Sampling (SBS).
O nome ANT descreve a hipótese motivadora; a máscara de relações ativas não
comprova, por si só, preservação de conhecimento ou ausência de ajustes nocivos.

## Escopo da pesquisa

- Comparar cobertura intra-view/full-symmetric (IV/FS), referência global/por
  âncora (GR/AR) e referência conectada/destacada (`detach`).
- Medir ACC, NME e esquecimento em protocolos de CIFAR-100, Tiny ImageNet,
  CUB-200 e ImageNet-100, com cobertura experimental específica por campanha.
- Investigar geometria e gradientes, além da acurácia.
- Preservar entradas, seleção e proveniência dos resultados usados nas publicações.

As tabelas antigas não são um painel de execução atual. Veja o
[índice de análises](analysis/README.md), os
[resultados históricos](docs/history/README.md) e o
[estudo do mecanismo ANT](analysis/results/ant_mechanism_20260909/ANALISE_RESULTADOS.md).
Esse estudo cobre uma seed e três tarefas; não estabelece superioridade global.

## Ambiente e dados

Execute os comandos a partir da raiz. Para instalar em uma cópia nova:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

O ambiente usado localmente é Python 3.11 com PyTorch/CUDA. Confira a
compatibilidade do host antes de treinar. `requirements.txt` ainda não declara
todas as ferramentas de análise: `autorank` e SciPy são utilizados na
estatística; `pytest` nos testes e Streamlit no viewer. O arquivo atual não
deve ser tratado como lock completo dessas ferramentas. Não reinstale o
ambiente compartilhado de um treinamento ativo.

- CIFAR-100: carregador com download automático.
- Tiny ImageNet: `python setup_tiny_imagenet.py --help`.
- CUB-200: `python setup_cub200.py --help`.
- ImageNet-100: [preparação e manifests](docs/IMAGENET100_SETUP.md).

Datasets, logs e checkpoints não são entregues pelo clone do código.
Os [submódulos de publicação](.gitmodules) podem ser obtidos, numa cópia nova,
com `git submodule update --init --recursive`. Preserve mudanças locais antes
de atualizar submódulos de uma cópia de trabalho existente.

## Comandos principais

Inspecionar a interface sem iniciar treinamento:

```bash
python main.py train --help
python generate_html_pdf_report.py --help
python statistical_tests.py --help
python provenance_cli.py --help
```

Exemplo de treinamento, a executar somente no host e na GPU alocados à campanha:

```bash
python main.py train \
  --exp-configs configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_resnet18.yaml \
  --seed 1993
```

Os YAMLs sobrepõem os argumentos da CLI e são aplicados da esquerda para a
direita com mesclagem superficial. O nome do primeiro YAML não determina a
variante efetiva. `infonce_max_global` é aceito por compatibilidade e ignorado
no cálculo atual. A interface `evaluate` ainda não implementa avaliação
independente; a avaliação usada na pesquisa ocorre durante o treinamento.

Geração de resultados (escreve novos artefatos, podendo substituir saídas correntes):

```bash
python generate_html_pdf_report.py          # curto: results_report_short.{html,pdf}
python generate_html_pdf_report.py --full   # inclui diagnósticos legados
python provenance_cli.py verify results_report_short.provenance.json
```

Congele entradas e seleções antes de citar uma edição; veja
[métricas](docs/RESULTS_AND_METRICS.md) e [proveniência](docs/PROVENANCE.md).

## Organização existente

| Caminho | Responsabilidade |
|---|---|
| `main.py`, `methods/`, `modules/` | Treinamento, métodos, redes, dados, memória e métricas |
| `utils/`, `loggers/` | Configuração, proveniência e logging |
| `configs/` | Protocolos legados, perfis em `hosts/` e links de compatibilidade |
| `experiments/` | Campanhas, filas, controles científicos migrados e lançadores |
| `scripts/execution/`, `scripts/sync/`, `scripts/maintenance/`, `scripts/lib/` | Operação compartilhada, auditoria e caminhos |
| `run_*`, `sync_*` na raiz | Entradas de compatibilidade |
| `studies/ant_mechanism/` | Instrumentação e ferramentas do estudo ANT |
| `validation/` | Controles experimentais e validação de campanhas |
| `analysis/`, geradores na raiz | Análises, seleção e relatórios |
| `tests/` | Testes automatizados |
| `ANT_Monografia/`, `ANT_WOPFACOM/` | Publicações e apresentações em submódulos |

O agrupamento de campanhas está implementado com
compatibilidade para caminhos antigos. A migração para `src/` e novos destinos
de saídas continua pendente. Veja o [escopo e as validações](docs/RESEARCH_INTEGRATION_20260912.md). Em 11/09/2026,
Xavier e Wolverine têm campanhas em uso; o responsável confirmou Fera e Quati
inativas. Revalide o estado antes de operar: o [mapa de dependências](docs/OPERATIONAL_DEPENDENCIES.md)
registra filas, snapshots, ambientes e coleta, com data de observação.

## Documentação canônica

- [Formulação e variantes ANT](docs/ANT_METHOD.md)
- [Métricas e seleção de resultados](docs/RESULTS_AND_METRICS.md)
- [Logs, matrizes e observação estruturada](docs/DEBUGGING_GUIDE.md)
- [Filas, launcher e operação dos hosts](docs/EXPERIMENT_QUEUE_GUIDELINES.md)
- [Proveniência e preservação de edições](docs/PROVENANCE.md)
- [Campanhas e evidências históricas](docs/history/README.md)
- [Auditoria documental e registro da revisão](docs/DOCUMENTATION_AUDIT.md)

Para validar o código sem treinamento:

```bash
python -m pytest -q tests
```

O [README do estudo](studies/ant_mechanism/README.md) descreve testes analíticos
e smokes de treinamento; smokes também precisam de recursos alocados.
