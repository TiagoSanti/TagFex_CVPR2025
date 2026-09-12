# Integração do estado da pesquisa — 12/09/2026

A implementação concilia a origem atual com as etapas 1/2 de organização,
preservando o suporte ao override do backbone, os relatórios ANT canônico/legado,
a nova campanha CUB e as alterações das publicações. Foi preparada em um clone
privado com submódulos reais e histórico próprio; o checkout operacional não foi
substituído. Publicar a branch não implanta automaticamente os serviços dos hosts.

## Organização e compatibilidade

- Quatro campanhas têm controles científicos em `experiments/<campanha>/configs/`
  e armazenamento em `configs/hosts/<host>/<campanha>/`.
- 22 filas / 334 entradas / 101 YAMLs referenciados; as 25 entradas CUB-200
  100+20 foram adicionadas aos 309 contratos históricos de regressão.
- Os geradores atuais estão em `analysis/reporting/` e `analysis/statistics/`.
  As entradas antigas na raiz mantêm importação e CLI, incluindo funções privadas
  utilizadas pelos testes. A identidade dos fontes na proveniência aponta às
  implementações novas, não só aos wrappers.
- Implementações de treino, analisadores históricos e YAMLs autocontidos legados
  permanecem disponíveis. Não houve mudança global de layout para `src/`.
- Filas e YAMLs antigos mantêm links relativos; scripts mantêm wrappers.

## Retomada e identidade científica

Nomes históricos são preservados, inclusive quando não codificam o stem do
backbone. A retomada só considera concluída uma execução com todas as tarefas
esperadas e um único manifesto `experiment_run`, de status `completed`, cujo
hash de configuração seja válido e cuja configuração científica seja compatível.
Defaults efetivos do backbone são normalizados como no learner. Configurações
com outra seed, arquitetura ou limites de piloto não são equivalentes.

Caminhos de implantação, host, dispositivo e controles apenas de saída não fazem
parte dessa comparação. Ordem de classes, treinamento, memória, transformações,
backbone, rede e limites científicos permanecem na comparação. Essa verificação
não recalcula o conteúdo inteiro do dataset nem substitui uma auditoria completa
dos artefatos ou validação científica do resultado.

Se já existe um diretório e não há prova suficiente, o verificador retorna erro
para revisão, em vez de pular a entrada ou iniciar outra tentativa silenciosamente.
Não apaga, renomeia ou reaproveita resultados. Resultados legados sem manifesto
continuam disponíveis para análise, com a incerteza registrada pela proveniência.

HTML e estatística rejeitam misturas arquiteturais conhecidas para o mesmo
método/protocolo. Ausência de manifesto em resultados históricos não é prova de
igualdade arquitetural; use a opção estrita de proveniência quando a edição exigir.
ANT conectado e destacado mantêm identidades separadas. A análise estatística
continua com quatro datasets e três seeds, sem incorporar CUB automaticamente.

O preflight CUB separa configuração estática de inspeção de logs (`--inspect-logs`).
Ele não remove parciais. A decisão de retomada pertence ao verificador compartilhado.
O diretório de locks é configurável, mantendo os basenames e a compatibilidade
histórica até uma migração coordenada. Não mudar só um produtor ou consumidor.

## Artefatos e versionamento

- Preservadas as edições estatísticas de 11/09 e 12/09 em
  `analysis/results/statistics/editions/`; `current` aponta para 12/09.
- Mantidos fontes, scripts, dados de figuras, resultados científicos históricos,
  pilotos identificados e testes. Arquivos volumosos já rastreados não foram
  descartados apenas por tamanho; continuam sendo evidência da pesquisa.
- Backup `.orig`, estado do editor, ambiente, logs de execução, caches e cópias
  privadas ficam fora dos commits. A proteção `.snapshot-isolated` é local e
  ignorada pelo Git; não bloqueia automaticamente clones de produção.
- Incluídos 23 PDFs finais de figuras da monografia que estavam localmente
  disponíveis, mas ignorados em seus novos diretórios. Os gitlinks devem apontar
  a commits publicados das duas publicações, preservando seu histórico.

## Validação

Foram aprovados 71 testes e 22 subtestes. Monografia e artigo compilaram com
`latexmk -pdf` em diretórios privados. A instalação limpa de dependências e
treinamentos completos em GPU não foram executados.

Os testes automatizados cobrem os contratos antigos e novos, overrides de
backbone, separação de identidades, retomada, locks, preflight, CLI e seleção.
Os dois orquestradores foram executados em fixtures com o lançador de GPU
substituído. Nenhum treino real ou sincronização operacional foi iniciado.

A movimentação dos geradores foi comparada com o código atual da origem sobre
363 gistlogs congelados: igualdade das linhas selecionadas de HTML, tabelas de
resultados, seção comparativa, candidatos estatísticos, matriz de observações,
CSV de seleção, relatório estatístico e tabela LaTeX. PDFs não foram comparados
byte a byte, porque metadados de geração podem variar.

Esta edição preserva o escopo científico das publicações e suas ressalvas sobre
ordem de classes fixa. Três seeds na análise existente e cinco na nova campanha
não são descrições intercambiáveis; não houve substituição global na monografia.

## Operação posterior

Revalidar processos, unidades, fila aberta e coleta em cada host antes de migrar
um checkout operacional. Atualizar a branch remota não autoriza interromper as
campanhas nem mudar seus caminhos em uso. Os registros operacionais anteriores
são observações datadas, não um monitoramento em tempo real.

## Histórico remoto

O histórico de `origin/main` foi incorporado à branch de integração. O gitlink
anterior do artigo apontava para `6e5369c6574afb7360d34e7a7a3f2051105f9c2c`,
que o remoto recusou fornecer (`not our ref`). A nova referência usa um commit
validado descendente do `main` disponível no repositório do artigo.
