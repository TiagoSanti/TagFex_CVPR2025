# Proveniência dos experimentos e análises

O sistema é aditivo: os formatos históricos `exp_gistlog.log`,
`exp_stdlog*.log` e `exp_debug*.log` não foram alterados. Consumidores antigos
continuam funcionando sem conhecer manifestos.

## Novas execuções

O treinamento que grava logs tenta criar, no processo principal, um manifesto único em
`<log_dir>/provenance/run-<UTC>-p<PID>.json`. O manifesto registra:

- commit e estado do Git;
- SHA-256 individual dos fontes de treinamento e hash agregado;
- ordem e SHA-256 dos YAMLs, configuração efetiva e seu hash;
- identificação do dataset conforme o modo de hash selecionado;
- Python, dependências, CUDA e GPUs visíveis;
- comando, horários, estado final e eventual exceção;
- SHA-256 individual dos artefatos e hash agregado.

A captura é tolerante a falhas: um erro de proveniência emite `WARNING`, mas
não muda o resultado nem o código de saída do treinamento. O padrão científico
`full` relê todos os bytes a cada execução. Para reduzir o custo de uma
execução exploratória, `TAGFEX_DATASET_HASH_MODE=structure` usa nomes e
tamanhos; `cached-full` reutiliza o hash quando caminho, tamanho, `mtime` e
`ctime` não mudaram; `none` desativa apenas o dataset. Para cobertura integral do conteúdo, usar `full`. As filas atuais de seeds
extras configuram `structure`; seus manifestos não comprovam identidade byte
a byte dos datasets. Preserve essa limitação ao interpretar os resultados,
sem editar overlays de campanhas ativas para mudar a política retroativamente.

## Logs históricos

Logs anteriores não contêm informação suficiente para recuperar retroativamente
o código, ambiente e dataset exatos. É possível, entretanto, congelar os
artefatos existentes:

```bash
python provenance_cli.py inventory \
  --logs-dir logs \
  --scope core \
  --output analysis/results/provenance/edicao-nova/logs-core.json
```

`core` cobre logs de resultado (`gist`) e execução (`std`), além de manifestos
novos. `full` inclui também logs de depuração, mas o volume depende do acervo e deve ser estimado antes da coleta. Por padrão, se qualquer arquivo de um experimento tiver sido alterado
nos últimos dez minutos, toda aquela pasta é relacionada em
`excluded_unstable` e fica fora do hash. Isso impede tratar uma execução ativa
como snapshot; o intervalo pode ser ajustado com `--settle-seconds`.

## Recálculo estatístico

```bash
python statistical_tests.py \
  --logs-dir logs \
  --output-dir analysis/results/statistics/edicao-nova
```

A saída contém a matriz usada (`observations.csv`), todas as decisões de
seleção (`selection.csv`), relatório, figuras, tabela e `provenance.json`.
Para resultados novos, acrescente `--require-run-manifests`: a análise falha
se qualquer log selecionado não estiver ligado ao manifesto da execução. O
modo permissivo existe somente para compatibilidade com o histórico anterior
à implantação deste sistema.

## Verificação

```bash
python provenance_cli.py verify \
  analysis/results/statistics/edicao-nova/provenance.json
```

O verificador retorna código zero somente quando arquivos, tamanhos, hashes
individuais e hashes agregados conferem.

## Relatório HTML/PDF de resultados

O gerador principal publica HTML, PDF e um sidecar como uma única geração
lógica:

```bash
python generate_html_pdf_report.py
python provenance_cli.py verify results_report_short.provenance.json
```

O topo do HTML mostra o `lineage_hash`, o nome do sidecar e uma seção
recolhível com todos os experimentos selecionados. Para cada experimento, o
sidecar registra dataset, método, seed, métricas interpretadas, caminho,
SHA-256 integral do `exp_gistlog.log` e eventuais manifestos da execução.
Também são registradas todas as decisões de exclusão e deduplicação.

No modo integral (`--full`), o hash de entrada inclui ainda o `exp_debug0.log`
ou `debug_logs.zip` efetivamente lido. No modo curto (padrão; `--short` pode
ser informado explicitamente), somente os gistlogs alimentam o conteúdo. O
`lineage_hash` liga seleção, manifestos-pais, parâmetros
do relatório, ambiente e código gerador; os hashes finais do HTML e PDF ficam
no sidecar para evitar autorreferência.

O relatório é produzido primeiro em arquivos temporários. O HTML/PDF anterior
só é substituído depois que ambos foram gerados com sucesso, e o sidecar é
publicado por último como marcador da geração completa. Se um log mudar entre
o parsing e o cálculo de seu hash, a geração é abortada.

Para novos painéis científicos, use:

```bash
python generate_html_pdf_report.py --require-run-manifests
```

Esse modo recusa qualquer experimento selecionado sem manifesto interno. O
modo permissivo mantém os resultados históricos utilizáveis, identificando
cada log legado diretamente por seu SHA-256.

## Manuscrito e outros artefatos derivados

O comando `bundle` conecta entradas, manifestos-pais e saídas sem impor um
formato ao gerador. Por exemplo, depois da compilação da monografia:

```bash
python provenance_cli.py bundle \
  --kind qualification-monograph \
  --input ANT_Monografia \
  --parent-manifest analysis/results/statistics/edicao-nova/provenance.json \
  --artifact ANT_Monografia/monografia.pdf \
  --output ANT_Monografia/provenance.json
```

O manifesto de saída contém um `lineage_hash` que liga os hashes agregados das
fontes, dos manifestos-pais e dos artefatos. O próprio arquivo de saída é
excluído para evitar autorreferência.


## Limitações atuais e transferência entre hosts

Os nomes `edicao-nova` dos exemplos são placeholders: escolha um identificador
novo e não sobrescreva artefatos citados. Geração e hashing leem o acervo;
não foram executados como parte desta revisão documental.

O inventário geral de fontes cobre `main.py`, `methods/`, `modules/`,
`utils/`, `loggers/` e arquivos de dependências selecionados. Ainda não inclui
`studies/`; o observer registra hashes próprios da instrumentação. A cobertura
conjunta precisa ser verificada antes de declarar reprodução completa.

Mover um resultado não deve reescrever seu manifesto original. Registre
origem, destino e hashes num registro de transferência separado; verificação
de manifestos com caminhos absolutos pode exigir a localização original.
O sistema ainda precisa de suporte explícito à relocação.

O inventário de agosto e o manifesto estatístico histórico falharam contra
partes do checkout/acervo de 11/09. Eles foram preservados, não recalculados
por cima das edições citadas. Veja [edições estatísticas](history/statistical_editions.md).
