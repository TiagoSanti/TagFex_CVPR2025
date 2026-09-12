> Revisão de 11/09/2026: plano original preservado; contagens de pendências e
> capacidade dos hosts abaixo são datadas e não descrevem a fila atual.
> As filas observadas hoje pertencem à campanha de seeds extras, com
> recuperação na Wolverine. Veja o [mapa operacional](OPERATIONAL_DEPENDENCIES.md).
> O status científico de cada célula deve ser conferido nos resultados e
> manifestos, não deduzido da conclusão de um wrapper.

# Plano da ablação fatorial do `detach` da referência ANT

## Decisão experimental

Vale completar o fatorial. O `detach` mantém o valor de
`r = max(s)`, o limiar `r - gamma` e o conjunto ativo no *forward*, mas elimina
o caminho de gradiente de `r`. Portanto, ele muda o mecanismo geométrico da
ANT, e não somente uma escolha numérica.

As 12 execuções ANT-FS-AR com `ant_detach_reference: true` já concluídas são
reutilizadas. Restam três variantes por protocolo e semente:

| Variante | `ant_symmetric_full` | `ant_max_global` | Situação |
|---|---:|---:|---|
| ANT-IV-GR + detach | `false` | `true` | enfileirar |
| ANT-IV-AR + detach | `false` | `false` | enfileirar |
| ANT-FS-GR + detach | `true` | `true` | enfileirar |
| ANT-FS-AR + detach | `true` | `false` | 12 execuções concluídas |

O escopo primário usa CIFAR-100 10-10 e 50-10 e Tiny ImageNet 20-20 e
100-20, com sementes 1993, 1994 e 1995. São 36 execuções novas.

## Divisão da carga

| Host | Protocolos | Execuções | Estimativa histórica |
|---|---|---:|---:|
| host local | C100 10-10 + TIN 100-20 | 18 | ~110 GPU-h |
| Wolverine, GPU 1 | C100 50-10 + TIN 20-20 | 18 | ~122 GPU-h |

A divisão mantém cada protocolo inteiro no mesmo host e equilibra melhor o
tempo do que uma divisão apenas pelo número de execuções.

Na auditoria de 2026-08-31, a Wolverine tinha ~718 GB livres na raiz, a GPU 1
estava livre e a GPU 0 ocupada. `/hd2` estava cheio e não deve ser usado. O host
local tinha ~170 GB livres, mas `nvidia-smi` não conseguia acessar o driver;
por isso a fila local deve aguardar a recuperação do driver.

## Arquivos e execução

- `configs/queue_ant_detach_factorial_local.txt`: fila local, 18 entradas.
- `configs/queue_ant_detach_factorial_wolverine.txt`: fila remota, 18 entradas.
- `configs/ant_detach_factorial/common.yaml`: intervenção comum de `detach`.
- `configs/ant_detach_factorial/{iv_gr,iv_ar,fs_gr}.yaml`: células faltantes.
- `configs/ant_detach_factorial/storage_local.yaml`: grava diretamente em
  `./logs`, entrada canônica dos reports locais.
- `configs/ant_detach_factorial/storage_wolverine.yaml`: staging da Wolverine
  fora do filesystem de `/home`, seguido de sincronização para Xavier.
- `run_ant_detach_factorial_local.sh`: lançador local.
- `run_ant_detach_factorial_wolverine.sh`: lançador limitado à GPU 1.

As filas usam a composição já suportada por `--exp-configs`: configuração do
protocolo, overlay comum, overlay da variante e overlay de armazenamento do
host, nessa ordem. `run_sbs_queue.sh` aceita esses caminhos separados por
vírgula e calcula a pasta esperada após mesclar os YAMLs, preservando retomada e
detecção de execuções concluídas.

Os parâmetros científicos compartilhados não definem `log_dir`. Essa separação
impede que a limitação de espaço da Wolverine envie também os resultados locais
para um staging fora de `./logs`.

Depois de implantar os arquivos na cópia de trabalho da Wolverine, a fila pode
ser iniciada com:

```bash
systemd-run --user --unit=ant-detach-factorial-wolverine-20260831 \
  --same-dir --collect bash run_ant_detach_factorial_wolverine.sh
```

No host local, somente depois de `nvidia-smi` voltar a funcionar:

```bash
systemd-run --user --unit=ant-detach-factorial-local-20260831 \
  --same-dir --collect bash run_ant_detach_factorial_local.sh
```

Nenhuma dessas filas deve compartilhar a mesma GPU com outro treino. Os
lançadores aguardam pelo menos 10 GB livres antes de iniciar cada execução.

## Extensão CUB-200 20--20 na Wolverine

Em 4 de setembro de 2026, foi acrescentada uma fila com as quatro variantes
detach, seeds 1993--1995, no CUB-200 20--20. Ela é isolada na GPU 0 da
Wolverine; a fila original permanece na GPU 1. A primeira tentativa, com o
`llama-server` ocupando cerca de 2,8 GiB, falhou por OOM ao iniciar a tarefa 9.
Para o relançamento, o container foi parado de forma reversível e o wrapper
passou a exigir 11.000 MiB livres antes de admitir o treinamento.

Dataset, logs e arquivos de orquestração permanecem em
`/var/tmp/tiago/ANT_detach_cub20_20260904`. A fila possui lock e unidade
independentes e usa `UPDATE_REPORT=0`; os resultados só entram nos reports
canônicos depois da sincronização e validação em Xavier.
