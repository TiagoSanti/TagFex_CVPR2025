# Reestruturação — etapa 1 no snapshot isolado

Data: 11/09/2026. Registro do estado concluído na etapa 1; não implantado.

A implementação atual avançou para a [etapa 2](RESTRUCTURING_STAGE2.md).
As descrições abaixo documentam o escopo e os resultados da primeira etapa.

## Escopo e isolamento

Snapshot privado: `/home/tiago/TagFex_CVPR2025/.local/restructure-20260911/repo`.
O diretório pai tem permissão 0700. Arquivos de trabalho, caches e temporários
ficam em `../metadata/` e `../runtime/`, fora de `/tmp`.

Foram copiados 698 arquivos (234.664.151 bytes): conteúdo atual rastreado e
não ignorado, incluindo mudanças pendentes e arquivos novos dos dois submódulos.
Os submódulos foram materializados como diretórios comuns neste Git independente;
`.gitmodules` é uma referência ao arranjo de origem, não deve ser usado para
inicializar submódulos aqui. Ambientes, datasets, logs e demais arquivos ignorados
não foram copiados. Isso é um snapshot de fontes e artefatos selecionados pelo
Git, não um backup completo da pesquisa ou uma reprodução dos hosts.

O commit inicial `06183f617b6f39274bc33abe55d5ca33e2ec8f49` registra a cópia antes
da reorganização. A implementação fica pendente sobre essa base, para revisão
com `git diff HEAD` e `git status`. Os estados e patches do Git original e dos
submódulos estão em `../metadata/`, junto ao manifesto SHA-256 dos arquivos.
Nenhum commit foi feito no repositório original.

Única alteração desta etapa fora do snapshot: regra na `.gitignore` original
para ocultar `/.local/restructure-20260911/`. Código, configurações, logs,
serviços e ambientes originais não foram modificados por esta etapa.

## Organização implementada

| Destino canônico | Conteúdo |
|---|---|
| `experiments/<campanha>/` | Lançadores específicos e encadeamentos |
| `experiments/<campanha>/queues/` | 21 filas, sem alterar linhas ou ordem |
| `experiments/legacy/` | Dois executores antigos com campanhas embutidas |
| `scripts/execution/` | Executores SBS, AvgK e entrada DDP |
| `scripts/sync/` | Implementação compartilhada de coleta Wolverine |
| `scripts/maintenance/` | Instalador cron e novo auditor de filas |
| `scripts/lib/paths.sh` | Raiz independente do diretório de invocação e proteção do snapshot |

O [mapa completo](RESTRUCTURING_MOVES.json) associa os 48 caminhos anteriores
aos novos destinos. Os 27 scripts na raiz agora são wrappers de compatibilidade,
e os 21 caminhos antigos de filas são links simbólicos relativos. Há uma única
implementação de cada script e uma única cópia de cada fila.

A raiz ainda contém os wrappers: esta etapa reduz a dispersão das implementações,
mas a remoção física das entradas antigas depende da atualização dos consumidores
externos. YAMLs, treinamento, estudos, análises e publicações mantêm suas posições.
Nenhuma decisão de excluir resultados ou pesquisa foi aplicada.

## Dependências e limites

Os lançadores passam a referenciar diretamente filas e executores canônicos.
A composição continua superficial, na ordem dos YAMLs. YAMLs continuam tendo
precedência sobre parâmetros de CLI, inclusive seed quando definida no YAML.
O auditor expõe ambas as seeds para permitir detectar divergências.

As dependências operacionais externas permanecem no
[levantamento operacional](OPERATIONAL_DEPENDENCIES.md): caminhos absolutos de
campanhas/datasets em Wolverine, logs, chaves SSH, serviços e locks. O levantamento
é datado e não representa uma nova observação dos processos nesta etapa.
Fera e Quati permanecem fora da migração operacional conforme informado.

Os caminhos históricos de locks `/tmp/tagfex_*.lock` foram preservados no código:
produtores e encadeamentos precisam compartilhar o mesmo lock para se coordenar.
Nenhum deles foi aberto pelos testes. Uma futura mudança para armazenamento
privado deve migrar todos os participantes juntos, incluindo processos antigos.
Não trocar só o lock de um executor enquanto existir consumidor do caminho antigo.

A presença de `.snapshot-isolated` bloqueia os 27 scripts operacionais movidos,
antes de criar diretórios, locks, cron, sincronizações ou iniciar treinamento.
Essa proteção abrange esses lançadores, **não** todo código Python nem scripts
antigos dentro de estudos/configs. Não é um sandbox do sistema operacional.
Não remova o marcador para executar campanhas nesta cópia: YAMLs ainda contêm
destinos reais. Treinamento direto, SSH, cron e sincronização não foram executados.

## Inspeção reproduzível, sem treinamento

A partir desta raiz, usando o interpretador existente somente para leitura:

```bash
export TMPDIR="$PWD/../runtime/tmp"
export MPLCONFIGDIR="$PWD/../runtime/mpl"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
PYTHON=/home/tiago/TagFex_CVPR2025/.venv/bin/python
"$PYTHON" scripts/maintenance/audit_queues.py
"$PYTHON" scripts/maintenance/audit_queues.py --queue configs/queue_ant_central_extra_seeds_local.txt --json
"$PYTHON" -m pytest -q -p no:cacheprovider
"$PYTHON" validation/validate_ant_central_extra_seeds.py
```

O auditor aceita caminhos relativos à raiz mesmo quando chamado de outro diretório.
Ele imprime dependências, hashes, seeds e configuração efetiva. `configured_paths`
contém valores configurados; não calcula o diretório final com sufixos de treino,
não verifica completude de logs, existência de datasets nem disponibilidade de hosts.

## Validações

- Base antes das mudanças: 30 testes e 22 subtestes passaram.
- Suite após a mudança: 39 testes e 22 subtestes passaram; saída em
  `../metadata/tests-after.txt`.
- 21 filas / 309 entradas comparadas diretamente com a origem: mesmos bytes,
  dependências YAML, hashes e configurações efetivas, na mesma ordem.
- Preflight existente: 50 identidades da campanha central, 5 protocolos ×
  5 métodos × 2 seeds; armazenamento, class_order e detach validados.
- `bash -n`: 55 arquivos, incluindo wrappers, implementações e biblioteca.
- Cinco CLIs responderam a `--help`: treino, relatório, estatística, proveniência
  e auditor de filas. `git diff --check` passou.
- Testes de chamadas a partir de outro diretório; rejeição de filas inválidas;
  precedência de seed e merge superficial; bloqueio dos 54 pontos de entrada shell.
- Um executor substituto em fixture privada confirma que o lançador local repassa
  fila canônica, GPU e UPDATE_REPORT corretamente, sem executar treinamento.
- Os 698 arquivos da origem capturados no manifesto permaneceram iguais.

Não foram validados desempenho, treino completo, acesso SSH, GPU, datasets,
serviços, sincronização real ou build das publicações. As dependências instaladas
foram reutilizadas; não foi testada instalação limpa de `requirements.txt`.

## Próxima etapa proposta

1. Revisar esta estrutura e definir o contrato de caminhos de saída, datasets,
   runtime privado e perfis de host, preservando identidades dos resultados.
2. Extrair a resolução do nome final de experimento para uma função compartilhada
   por treino, detecção de completude e auditor; hoje existe lógica duplicada.
3. Separar YAMLs científicos dos overlays operacionais por host, validando
   equivalência de todas as composições antes de mover configurações.
4. Atualizar consumidores externos em janela coordenada e só então aposentar
   wrappers, links antigos e locks históricos. Não implantar sobre filas ativas.
5. Tratar em lotes próprios os geradores da raiz, resultados versionáveis,
   ferramentas legadas e estrutura de publicações; decidir exclusões a partir
   do valor científico e da proveniência, não apenas do tamanho ou idade.
