# Reestruturação — etapa 2: configuração e caminhos compartilhados

11/09/2026. Implementação somente no snapshot privado; não implantada em Xavier
ou Wolverine. Complementa a [etapa 1](RESTRUCTURING_STAGE1.md).

## Contratos preservados

- As 309 entradas das 21 filas mantêm ordem, descrição, seed e composição
  científica. Foram alterados caminhos de referência, não valores dos YAMLs.
- Os 17 YAMLs realocados permanecem byte a byte iguais aos originais. Links
  relativos preservam os caminhos anteriores e consumidores de YAML simples.
  Não foi introduzido um formato de includes ou merge recursivo.
- O merge continua superficial e os YAMLs prevalecem sobre a CLI. A resolução
  compartilhada também preserva os defaults reais de `utils/argument.py` e
  `force_no_debug`. Ausência de `log_dir` usa o default da CLI; `log_dir: null`
  permanece distinto e explícito.
- Nomes não versionados foram comparados com a implementação anterior para
  todas as entradas e para 69 casos adicionais. A seleção de `_v2`, `_v3`, etc.
  continua no learner, sem alterar sua política existente.
- Logs locais continuam em `./logs`; destinos remotos e o dataset CUB compartilhado
  com a campanha anterior permanecem com os mesmos valores. Nada foi transferido.
- Os basenames dos locks e pares produtor/espera foram preservados. Os defaults
  continuam compatíveis com processos antigos; não houve migração de locks ativos.

## Organização

| Local | Responsabilidade |
|---|---|
| `experiments/ant_central_extra_seeds/configs/` | Controles e variantes científicas da campanha |
| `experiments/ant_detach_factorial/configs/` | Controles e variantes científicas da campanha |
| `experiments/ant_detach_cub20/configs/` | Variante científica da campanha CUB |
| `configs/hosts/<host>/<campanha>/storage.yaml` | Destinos de resultados e, quando necessário, dataset |
| `configs/hosts/wolverine/ant_central_extra_seeds/dataset_cub.yaml` | Dependência explícita do dataset CUB de outra campanha |
| `utils/configuration.py` | Merge usado pelo treino e composição real de CLI para inspeção |
| `utils/experiment_paths.py` | Identidade de resultados, número de tarefas e checagem de retomada |
| `scripts/maintenance/profile_path.py` | Leitura de caminhos dos perfis, sem criar ou sondar seus destinos |
| `scripts/maintenance/check_queue_entry.py` | Retorno 0 completo, 1 incompleto/incerto, 2 erro |
| `scripts/lib/paths.sh` | Raiz do checkout, acesso a perfis e diretório comum de locks |

São 11 YAMLs científicos e 6 operacionais. Veja o
[mapa de configurações](RESTRUCTURING_CONFIG_MOVES.json) e o
[mapa de consumidores de locks](RESTRUCTURING_LOCK_DEPENDENCIES.json).
O [mapa geral](RESTRUCTURING_MOVES.json) também corrige duas classificações da
primeira etapa: o coletor central pertence à campanha central; o encadeamento
CUB-IV-GR pertence à campanha que ele inicia, e não à campanha pela qual espera.

Os lançadores locais recentes e os lançadores remotos que declaravam
`CAMPAIGN_ROOT` consultam os perfis de armazenamento para obter seus destinos.
O coletor central consulta os mesmos perfis para origem remota e destino local.
Isso elimina a repetição desses caminhos absolutos nesses lançadores.

`configs/all_in_one/`, cenários compartilhados, investigações antigas e o estudo
do mecanismo ainda conservam sua organização e seus caminhos existentes.
Vários YAMLs legados são autocontidos e têm consumidores além das filas.
Sua separação completa fica para um lote próprio, com inventário desses
consumidores; não foram transformados implicitamente em overlays incompletos.

## Correções de funcionamento

A lógica anterior de retomada duplicava o nome do learner e usava um glob por
prefixo. Agora treino, auditor e os executores SBS/AvgK usam a mesma função.
A checagem contempla SBS, fatores opcionais de contraste, defaults da CLI e
seed efetiva após os YAMLs. Só aceita a identidade exata ou uma versão válida
`_v2`, `_v3`, ...; outro sufixo ou uma seed com o mesmo prefixo não completam a fila.

O número esperado de tarefas precisa ser determinável. Configuração insuficiente
ou protocolo inválido não é mais presumido completo depois de uma única tarefa.
O prefixo de arquivo configurado também é respeitado. A checagem permanece uma
heurística baseada em resumos `avg_nme1`, não uma certificação de proveniência,
integridade científica ou término bem-sucedido do processo.

Falhas de leitura/configuração retornam erro e impedem que aquela entrada seja
lançada. Os dois executores agora retornam status diferente de zero quando a
fila acumula falhas, inclusive YAML inválido ou arquivo ausente; anteriormente
o último `echo` podia fazer uma fila malsucedida parecer bem-sucedida ao serviço.
Isso é uma correção operacional deliberada, não uma mudança científica.

## Locks e diretórios privados

`TAGFEX_LOCK_DIR` centraliza o diretório usado pelos 23 consumidores de locks
realocados. Por compatibilidade com filas antigas, seu default histórico ainda
é `/tmp`. Nenhum lock desse diretório foi aberto por esta etapa.

Um diretório privado pode ser selecionado explicitamente: precisa existir,
ser absoluto, pertencer ao usuário e ter modo 0700; symlink no próprio diretório
é rejeitado. Todos os produtores e consumidores do mesmo host precisam receber
o mesmo valor. O diretório deve ser compartilhado entre checkouts do usuário,
e não criado separadamente por checkout, para preservar a exclusão mútua.
Overrides explícitos como `LOCKFILE` continuam válidos e também precisam fazer
parte dessa coordenação. Não basta mudar só a variável de um processo novo.

A troca do default histórico não foi feita porque exigiria uma transição
coordenada dos consumidores ainda ativos. Temporários e artefatos desta revisão
continuam exclusivamente em `.local/restructure-20260911/`, com acesso privado.

## Evidências

Resultado final: **61 testes e 22 subtestes passaram**. A análise sintática de
55 arquivos shell, sete CLIs com `--help`, `git diff --check` e o preflight de
50 identidades da campanha central também passaram.

As saídas desta etapa estão em `../metadata/stage2/`, relativas à raiz do snapshot:
`tests.txt`, `verification.json`, `queues-after.json`, `cli-checks.json` e
`preflight.txt`. O patch da etapa 1 foi preservado em `stage1.patch`. `stage2-only.patch` contém
somente o avanço desta etapa; `cumulative.patch` contém as duas etapas sobre
o commit inicial do snapshot.

Os contratos em `tests/fixtures/` foram congelados usando a configuração anterior
à movimentação, o método de nomes original e os defaults originais da CLI,
antes de comparar com a nova implementação. Não foram gerados pela função nova.
A comparação verifica hashes da composição YAML, seeds, descrições e nomes.

Os testes incluem retomada dos dois executores em fixtures privadas, YAML inválido,
configuração ausente, versões, prefixos ambíguos, overrides, chamadas fora da raiz,
leitura dos perfis Wolverine por lançadores e coletor com executor substituído,
e exclusão mútua real com `flock` em diretório privado. O acesso à GPU e à rede
foi substituído nesses testes; nenhum treino, serviço ou coletor real foi iniciado.

Os 698 arquivos da origem presentes no manifesto inicial foram novamente
verificados por SHA-256. Esta etapa não alterou nenhum arquivo fora do snapshot.
A proteção `.snapshot-isolated` continua ativa para os lançadores movidos;
não cobre arbitrariamente todo script Python ou shell existente.

## Próximo lote

Revisar os consumidores dos YAMLs legados e dos geradores de análise antes de
separar suas configurações e mover esses geradores da raiz. A leitura de nomes
históricos por reports e pelo coletor remoto ainda merece um lote próprio;
ela não foi substituída por esta checagem de filas. A implantação nos hosts,
a retirada de wrappers e a mudança definitiva de locks continuam pendentes.
