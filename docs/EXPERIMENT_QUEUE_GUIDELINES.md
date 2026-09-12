> Revisão de 11/09/2026: Xavier e Wolverine são os únicos hosts em uso;
> Fera e Quati estão inativas, conforme confirmação do responsável. Instruções
> e ocorrências datadas de outros hosts permanecem como referência histórica,
> não como filas a relançar. O [mapa operacional](OPERATIONAL_DEPENDENCIES.md)
> contém a fotografia dos serviços efetivos; estado/GPU de uma campanha não
> pode ser inferido somente do nome de um wrapper.

# Diretrizes para filas de experimentos

Este documento é a fonte de verdade para criar, iniciar, interromper, retomar e
coletar filas de experimentos do TagFex. Ele foi consolidado após a auditoria
dos históricos em `logs/auto_experiments/` e deve ser lido antes de alterar
qualquer `configs/queue_*.txt` ou `run_*_queue.sh`.

## Princípios obrigatórios

1. A configuração científica, a política do host e a descrição da fila são
   responsabilidades separadas.
2. Resultados produzidos no host local devem ser escritos diretamente em
   `./logs`, que é a entrada canônica dos reports.
3. Staging fora de `./logs` só é permitido quando o host realmente precisar de
   outro filesystem. Nesse caso, a fila deve ter overlay de armazenamento e
   coletor explícitos para aquele host.
4. O nome final do experimento deve ser produzido pelo código do método. Uma
   fila não deve inventar uma segunda convenção de nomes.
5. Uma execução completa nunca deve ser apagada para simplificar uma retomada.
   Tentativas incompletas só podem ser removidas após confirmar que não há
   processo escrevendo nelas e que o número esperado de tarefas não foi
   atingido.
6. Não instalar `cron @reboot` para uma fila temporária. Use uma unidade
   transitória do `systemd --user` e retome explicitamente após reinicialização.
7. Não editar uma fila que está sendo consumida. O orquestrador lê o arquivo
   progressivamente; uma alteração pode mudar somente as entradas futuras.

## Estrutura canônica

No snapshot da [etapa 2](RESTRUCTURING_STAGE2.md), as campanhas migradas usam
a estrutura abaixo. Os checkouts em operação ainda usam os caminhos anteriores:

```text
experiments/<campanha>/configs/common.yaml        # controles científicos
experiments/<campanha>/configs/<variante>.yaml    # célula fatorial
configs/hosts/<host>/<campanha>/storage.yaml      # armazenamento
configs/hosts/<host>/<campanha>/dataset_cub.yaml  # dataset, quando necessário
experiments/<campanha>/queues/queue_<host>.txt    # ordem, descrição e seed
experiments/<campanha>/run_<host>.sh              # recursos, lock e observabilidade
```

O `common.yaml` não deve conter caminhos específicos de máquina. Em especial,
não deve conter `log_dir`, raiz de dataset, chave SSH ou ID de GPU quando esses
valores diferirem entre hosts.

Exemplo local:

```yaml
# configs/hosts/xavier/<campanha>/storage.yaml
log_dir: ./logs
```

Exemplo de staging remoto:

```yaml
# configs/hosts/wolverine/<campanha>/storage.yaml
log_dir: /var/tmp/tiago/<campanha>/logs
```

Os overlays são aplicados da esquerda para a direita. A política de
armazenamento deve ser a última configuração, para que o destino final fique
explícito na própria entrada:

```text
configs/base.yaml,experiments/campanha/configs/common.yaml,experiments/campanha/configs/iv_gr.yaml,configs/hosts/xavier/campanha/storage.yaml|C100 10-10 ANT-IV-GR detach|1993
```

## Formato da fila

Cada linha ativa tem exatamente três campos separados por `|`:

```text
<config[,overlay,...]>|<descrição>|<seed>
```

Regras:

- comentários ocupam a linha inteira e começam com `#`;
- comentários inline são proibidos;
- a seed contém somente dígitos;
- caminhos são relativos à raiz do repositório;
- não há espaços dentro da lista de configurações;
- uma entrada representa exatamente uma identidade experimental;
- itens concluídos permanecem na fila e são tratados por `is_done`; não se usa
  `# done` no campo da seed;
- a contagem declarada no comentário e no wrapper deve coincidir com o número
  de linhas ativas.

Comentários inline já transformaram `1994 # done` em uma seed inválida e
causaram várias falhas consecutivas. O padrão acima elimina essa ambiguidade.

## Wrapper por campanha

Novas filas devem delegar ao orquestrador composável atualmente implementado
em `scripts/execution/run_sbs_queue.sh`, com wrapper de compatibilidade na raiz. Não copie o corpo do
orquestrador para um novo arquivo.

O wrapper deve carregar `scripts/lib/paths.sh`, verificar a proteção do snapshot
e validar o diretório de locks antes de qualquer escrita. Deve definir, no mínimo:

```bash
QUEUE="$SCRIPT_DIR/experiments/<campanha>/queues/queue_<host>.txt"
LOCKFILE="$TAGFEX_LOCK_DIR/tagfex_<campanha>_<host>.lock"
ORCH_LOG="$SCRIPT_DIR/logs/auto_experiments/<campanha>_<host>_orchestrator.log"
CONSOLE_DIR="$SCRIPT_DIR/logs/auto_experiments/<campanha>_<host>_console"
GPUS=1
MIN_FREE_MB=10000
ALLOWED_GPU_IDS="<ids autorizados>"
STOP_ON_FAILURE=1
```

Cada campanha/host precisa de lock própria. A lock impede duas instâncias da
mesma fila, mas não reserva uma GPU contra outras filas. Filas simultâneas
devem ter conjuntos `ALLOWED_GPU_IDS` disjuntos ou outro mecanismo explícito de
exclusão.

`THRESHOLD=100` sem `MIN_FREE_MB` ou `MEMORY_THRESHOLD` considera qualquer GPU
disponível. Portanto, nunca usar esse valor isoladamente.

Para campanhas novas, `STOP_ON_FAILURE=1` é o padrão. Continuar depois de OOM,
configuração ausente ou erro de dataset tende a produzir várias pastas
incompletas e esconder a causa inicial.

## Nome e detecção de conclusão

A fonte de verdade do nome é
`utils.experiment_paths.experiment_log_dir()`, também chamada pelo learner. A detecção deve considerar todos
os campos que alteram a identidade, incluindo:

- dataset, protocolo e seed;
- `ant_beta`, `nce_alpha` e `ant_margin`;
- cobertura e referência ANT;
- `ant_detach_reference`;
- formulação ANT;
- TeacherAvg;
- SBS;
- demais extensões que venham a ser incorporadas.

Não criar outro `is_done()` copiando parte dessa lista. A divergência histórica
entre `run_avgk_queue.sh` e o produtor omitiu o sufixo SBS, não reconheceu uma
execução já completa e a reiniciou duas vezes.

O resolvedor compartilhado já está extraído no snapshot. Qualquer novo parâmetro
de nome deve ser acrescentado nessa função e coberto pelos contratos de regressão.
A versão do diretório continua sendo alocada pelo learner.

A checagem de retomada considera completa a identidade exata ou sua versão `_vN`
quando o `<output_file_prefix>_gistlog.log` contém resultados para pelo
menos o número esperado de tarefas. A mera existência da pasta, de
`exp_stdlog0.log`, de checkpoint ou de uma ocorrência de `avg_nme1` não basta.

## Preflight obrigatório

Antes de iniciar:

1. Confirmar que nenhum treino usa a GPU destinada à fila:

   ```bash
   gpustat --no-color
   nvidia-smi
   ps -ef | rg 'main.py train|run_.*queue|auto_run_on_free_gpu'
   ```

   Na Xavier, `gpustat` é a checagem concisa preferencial; `nvidia-smi`
   permanece como confirmação do driver, memória e processos.

2. Conferir filesystem e espaço do `log_dir` efetivo:

   ```bash
   df -hT ./logs
   ```

3. Mesclar os YAMLs de cada linha na ordem declarada e validar que todas as
   entradas locais resolvem para `./logs`, ou para o staging documentado no
   host remoto.
4. Validar que cada arquivo existe, cada linha tem três campos e as seeds são
   numéricas.
5. Verificar que `LOCKFILE`, `ORCH_LOG` e `CONSOLE_DIR` são exclusivos.
6. Conferir quantas entradas serão puladas, executadas ou consideradas
   incompletas.
7. Registrar `git status`, commit atual, tamanho da fila e destino dos logs.
8. Executar `bash -n` no wrapper e os testes relacionados ao método.

Não iniciar se a fila, overlays ou scripts relevantes existirem apenas na
máquina de origem e ainda não tiverem sido implantados no host executor.

Para a campanha principal de seeds 1996/1997, executar também:

```bash
.venv/bin/python validation/validate_ant_central_extra_seeds.py
```

Esse preflight exige exatamente 50 identidades únicas, `nceGlobal`, detach nas
quatro variantes ANT, armazenamento correto e a mesma ordem fixa de classes
entre protocolos do mesmo dataset.

## Inicialização e monitoramento

Padrão local:

```bash
systemd-run --user \
  --unit=<campanha>-<host>-<AAAAMMDD> \
  --same-dir --collect \
  bash run_<campanha>_<host>.sh
```

Não redirecionar stdout/stderr para o mesmo `ORCH_LOG`: o orquestrador já usa
`tee -a`. Fazer ambos duplica eventos como `START` e `DONE`, confundindo
coletores e scripts encadeados. A saída da unidade fica no journal; os eventos
estruturados permanecem no orquestrador.

Monitoramento mínimo:

```bash
systemctl --user --no-pager --full status <unidade>.service
journalctl --user -u <unidade>.service -f
tail -F logs/auto_experiments/<campanha>_<host>_orchestrator.log
nvidia-smi
```

Após iniciar, confirmar:

- unidade `active (running)`;
- lock mantida;
- PID de `main.py train` no cgroup da unidade;
- GPU e memória esperadas;
- diretório criado no filesystem correto;
- basename sem `_v2`, salvo quando a repetição foi deliberadamente preservada;
- primeiros epochs sem OOM ou erro de dataset.

## Interrupção e retomada

Interromper pela unidade, não por PIDs avulsos:

```bash
systemctl --user stop <unidade>.service
```

Depois, confirmar que a lock está livre, não há descendentes e os arquivos
pararam de mudar. Classificar cada diretório atingido como completo ou
incompleto.

Para retomar:

- resultados completos ficam onde estão e devem ser pulados;
- resultados incompletos são preservados até a causa da interrupção ser
  entendida;
- se a decisão for reiniciar do zero, remover ou mover somente o caminho exato
  da tentativa incompleta, com a fila parada;
- nunca apagar por glob amplo;
- reiniciar a mesma fila; `is_done` cuidará das entradas completas.

O código cria `_v2`, `_v3`, etc. quando a identidade esperada já existe. Isso
protege dados, mas não é um mecanismo de retomada. Um `_v*` inesperado indica
que o preflight não classificou/limpou uma tentativa incompleta ou que a
detecção de conclusão divergiu do produtor de nomes.

## Armazenamento local e remoto

### Xavier/local

Usar `log_dir: ./logs`. Assim o report, os testes estatísticos e as figuras
encontram os resultados sem coleta intermediária.

### Wolverine

Quando `/home` não comportar os logs, usar um diretório de staging em outro
filesystem, declarado em `storage_wolverine.yaml`. O staging deve ter:

- espaço verificado antes do início;
- diretório exclusivo por campanha;
- coletor `rsync` incremental com `--partial`;
- origem remota preservada até a verificação final;
- registro de origem, destino e hashes;
- sincronização final após a unidade ficar inativa.

O coletor copia somente para o `./logs` canônico de Xavier. O report definitivo
é gerado em Xavier depois da sincronização. Não modificar `common.yaml` para
resolver um problema de disco de um único host.

Manifests de execução podem conter `output_dir` absoluto. Não reescrever esses
manifests depois da coleta. Registrar a transferência separadamente e manter a
origem até validar a proveniência.

### Quati

O Quati não participa da campanha principal de seeds 1996/1997. As instruções
abaixo ficam preservadas apenas para uma eventual campanha futura autorizada;
não relançar filas antigas desse host.

O Quati é acessado por SSH como `tiago@10.87.10.217`, porta `2222`, usando a
chave dedicada `~/.ssh/id_ed25519_quati`:

```bash
ssh -i ~/.ssh/id_ed25519_quati -p 2222 tiago@10.87.10.217
```

O host possui uma RTX 4090 de 24 GB identificada como GPU `0`. A política
inicial para uma fila exclusiva nesse dispositivo é:

```bash
GPUS=1
ALLOWED_GPU_IDS=0
THRESHOLD=100
MIN_FREE_MB=20000
STOP_ON_FAILURE=1
```

O volume `/home` do Quati tem pouco espaço livre e não deve receber novos
logs, checkpoints ou cópias completas de campanha. Antes de cada implantação,
confirmar a situação atual com `df -h /home /mnt/raid`. Código isolado,
resultados e metadados devem ficar no RAID seguindo esta estrutura:

```text
/mnt/raid/home/tiago/experiments/<campanha>/
├── repo/                 # snapshot imutável durante a execução
├── logs/                 # saídas dos experimentos
├── orchestrator/         # log da fila e consoles individuais
└── metadata/             # commit, inventário, hashes e backups operacionais
```

O repositório legado em `/home/tiago/TagFex_CVPR2025` pode estar defasado e
conter alterações locais. Ele deve ser tratado como somente leitura: não usar
`git pull`, `git reset`, limpeza, checkout ou implantação por cima dele. Criar
um snapshot novo em `repo/`, registrar o commit de origem e gerar um manifesto
SHA-256 após a sincronização. Uma `.venv` existente no repositório legado pode
ser referenciada por link simbólico somente depois de validar Python, PyTorch,
CUDA, imports e CLI no snapshot.

Ao copiar o código com `rsync`, ancorar exclusões destinadas apenas à raiz do
repositório. Por exemplo, usar `--exclude=/data/`, nunca
`--exclude=data/`: o segundo padrão também exclui `modules/data/` e produz um
snapshot que só falhará ao importar o gerenciador de datasets. Aplicar a mesma
regra a `.git`, `.venv`, `logs`, checkpoints e demais diretórios volumosos.
Depois da cópia, executar um `rsync -naci` com os mesmos filtros e aceitar
somente diferenças operacionais deliberadas, como permissão do wrapper.

Os datasets atualmente compartilhados pelo host ficam em:

```text
/home/tiago/TagFex_CVPR2025/data/datasets/cifar100
/home/tiago/data/datasets/tiny-imagenet-200
/mnt/raid/home/tiago/data/datasets/CUB_200_2011
```

O download e a preparação do ImageNet-100 exigem as imagens autorizadas do
ILSVRC e os manifests do recorte de 100 classes. Seguir o procedimento
reproduzível em [`IMAGENET100_SETUP.md`](IMAGENET100_SETUP.md), sempre usando o
RAID para download, extração e dataset final no Quati.

Como o CUB-200 está no RAID, filas que o utilizam devem acrescentar um overlay
de `dataset_root` específico do Quati antes do overlay de armazenamento. O
último YAML de toda entrada deve apontar `log_dir` para
`/mnt/raid/home/tiago/experiments/<campanha>/logs`.

Antes do início, além do preflight geral:

1. verificar GPU, processos e espaço em `/home` e `/mnt/raid`;
2. conferir se o diretório da campanha ainda não existe ou classificá-lo antes
   de reutilizá-lo;
3. auditar `crontab -l` e unidades de usuário para impedir que uma fila antiga
   seja retomada junto com a nova;
4. salvar qualquer entrada de autoresume antes de removê-la; não instalar um
   novo `@reboot`;
5. validar no host as combinações dos YAMLs, identidades únicas, seeds,
   número de tarefas e caminhos efetivos;
6. carregar ao menos uma amostra real de cada dataset da campanha;
7. executar testes do método, `bash -n`, importação CUDA e a CLI;
8. confirmar novamente que a GPU está livre e que a lock exclusiva não existe.

O wrapper do Quati deve manter `ORCH_LOG` e `CONSOLE_DIR` no RAID e definir
`UPDATE_REPORT=0`. Reports gerados enquanto a campanha remota está parcial não
são definitivos e não devem varrer uma árvore diferente daquela que contém os
resultados. O report canônico será regenerado em Xavier após a coleta.

Iniciar a fila remotamente com uma unidade transitória e diretório de trabalho
explícito:

```bash
systemd-run --user \
  --unit=<campanha>-quati-<AAAAMMDD> \
  --collect --property=Restart=no \
  --working-directory=/mnt/raid/home/tiago/experiments/<campanha>/repo \
  /usr/bin/bash ./run_<campanha>_quati.sh
```

Monitorar pelo menos a primeira execução até observar epochs progredindo,
alocação estável de memória e ausência de `out of memory`, `CUDA error`,
`Traceback` ou `RuntimeError`. Conferir também que o primeiro diretório surgiu
em `.../<campanha>/logs` sem sufixo `_v*` inesperado.

Durante a campanha, preservar os resultados no RAID. A coleta para o `./logs`
canônico de Xavier deve ter origem e destino explícitos, preservar a origem e
ser finalizada somente depois de a unidade ficar inativa. Evitar publicar no
report diretórios ainda em escrita; após a sincronização final, validar hashes,
completude por número de tarefas e reconhecimento pelo gerador de reports.

## Reports e encerramento

Ao concluir uma entrada local, o diretório deve estar visível sob `./logs`.
Depois de uma sincronização remota, aplicar a mesma condição.

Antes de declarar uma campanha concluída:

1. Todas as entradas precisam estar completas ou explicitamente justificadas.
2. Não pode haver lock ou unidade ativa da campanha.
3. Não pode haver diretório incompleto não classificado.
4. `generate_html_pdf_report.py --short` deve reconhecer as novas execuções.
5. A seleção metodológica e eventuais deduplicações devem ser revisadas.
6. Os testes estatísticos devem reconhecer os rótulos novos.
7. O inventário/proveniência deve ser atualizado.
8. O status final e os comandos de reprodução devem ser registrados.

## Padrões proibidos observados no histórico

- `cron @reboot` apontando para uma fila temporária já encerrada;
- stdout redirecionado ao mesmo arquivo no qual `log()` usa `tee -a`;
- `# done` anexado ao campo de seed;
- caminho de Wolverine dentro de um overlay científico compartilhado;
- `is_done()` reimplementado sem todos os sufixos do produtor;
- considerar uma lock mantida como prova suficiente de treinamento ativo;
- editar a fila enquanto o processo a está lendo;
- continuar dezenas de entradas após uma falha ambiental repetida;
- encadear filas apenas observando que uma lock foi liberada, sem registro de
  conclusão verificável;
- mover um resultado com manifesto absoluto sem política de relocação.

## Dívidas técnicas conhecidas

- Integrar à política compartilhada os consumidores de nomes históricos em reports
  e na coleta remota. Treino, filas SBS/AvgK e auditor já compartilham o resolvedor.
- Renomear ou substituir `run_sbs_queue.sh` por um orquestrador genérico.
- Adicionar comando de preflight/dry-run que liste `skip`, `run`, `partial` e o
  caminho efetivo de cada entrada.
- Registrar no início da unidade o hash da fila e dos overlays, impedindo edição
  silenciosa durante a execução.
- Tornar a verificação de proveniência consciente de transferências entre
  hosts, sem alterar manifests originais.


## Referência do launcher e diagnóstico de recursos

Esta seção incorpora o conteúdo operacional ainda válido dos antigos guias
AUTO_GPU_LAUNCHER, LOGGING_SYSTEM e GPU_MEMORY_GUIDE. O launcher existente é
`auto_run_on_free_gpu.py`; consulte `python auto_run_on_free_gpu.py --help`.

| Opção | Função |
|---|---|
| `--command` | Comando a lançar; obrigatório |
| `--gpus` | Quantidade de GPUs a aguardar; não transforma por si só Python em DDP |
| `--allowed-gpu-ids` | IDs físicos permitidos no host |
| `--threshold` | Limiar de utilização, padrão 1% |
| `--memory-threshold` | Limite percentual de memória ocupada, opcional |
| `--min-free-mb` | Memória livre mínima, opcional |
| `--interval` | Intervalo de consulta, padrão 30 segundos |
| `--no-screen` | Execução direta; usado pelo orquestrador atual |
| `--no-wait` | Não aguardar término; não usar como evidência de conclusão |
| `--log-dir` | Destino do logging do launcher; não substitui o `log_dir` científico dos YAMLs |

Utilização e memória são sinais diferentes. Memória ocupada com utilização
baixa pode pertencer a outro serviço, espera de dados ou intervalo de avaliação.
Não inferir abandono a partir de uma amostra de utilização. Confira o host,
PID, serviço e proprietário; em container/isolamento a lista de processos
pode não mostrar o PID da GPU. Não apagar locks nem reiniciar drivers/ambientes
como procedimento rotineiro de liberação.

Os antigos `start_queue_monitor.sh`, `run_experiments_queue.sh`,
`monitor_screen_sessions.sh`, `test_gpu_monitor.sh`, `check_gpu_processes.sh`
e `diagnose_gpu_memory.sh` não fazem parte do checkout. Logs e observadores
estão descritos no [guia de observabilidade](DEBUGGING_GUIDE.md).

## Campanhas científicas e armazenamento não descartável

Uma campanha concluída pode continuar fornecendo dataset/ambiente para outra.
Na Wolverine, o snapshot de seeds extras usa a `.venv` do checkout antigo e
o dataset CUB da campanha de 04/09. Em Xavier, treino e coletor escrevem no
mesmo `logs/`. Serviços carregam variáveis que podem sobrescrever defaults
dos wrappers. Essas dependências devem ser conferidas antes de reorganizar.
