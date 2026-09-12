> Revisão aplicada em 11/09/2026 por solicitação do responsável. As tabelas
> abaixo preservam o diagnóstico anterior e seus caminhos de origem; arquivos
> retirados não são links de navegação. Consulte o [índice atual](../README.md)
> e o [histórico/erratas](history/README.md). Foram consolidados os guias,
> retiradas saídas defeituosas e atualizada a documentação dos submódulos.
> Dados, código, filas, serviços e artefatos científicos permaneceram intactos.
> Pendências de implementação/reprodução estão descritas nos guias e no TODO
> da monografia; esta revisão não afirma que elas tenham sido resolvidas.

# Auditoria documental — 11/09/2026

Diagnóstico que fundamentou a revisão aplicada. As recomendações nas tabelas
registram os destinos escolhidos a partir do estado anterior. A avaliação
considera conteúdo, referências locais, correspondência com implementação,
utilidade científica e sobreposição. Não foi feita nova validação externa da
bibliografia nem novo cálculo de todos os resultados históricos. Os manuscritos
LaTeX são produtos da pesquisa e não foram classificados como documentação
descartável nesta auditoria.

O usuário confirmou: somente Xavier e Wolverine estão em uso; Fera e Quati
não estão sendo utilizadas atualmente e não há outros hosts. Informação
operacional atual não transforma resultados anteriores desses hosts em obsoletos.

## Critérios

- **Atualizar:** documento com responsabilidade permanente ou utilidade atual.
- **Consolidar:** extrair conteúdo válido para uma fonte canônica e remover o
  documento redundante somente depois de atualizar seus consumidores.
- **Histórico:** preservar evidências, hipóteses e decisões datadas fora dos
  guias correntes. Acrescentar contexto/errata sem reescrever retrospectivamente
  o que foi observado.
- **Remover:** saída comprovadamente defeituosa, estado transitório superado
  ou inventário redundante sem conteúdo científico exclusivo. Preservar fontes
  e entradas; para arquivos já rastreados, o histórico Git continua disponível.

## Documentação principal e técnica

Os caminhos desta tabela são relativos à raiz.

| Documento | Destino | Fundamentação e ação |
|---|---|---|
| `README.md` | Atualizar | Porta de entrada necessária. Reduzir tabelas históricas e status volátil; corrigir fórmula de mistura, arquitetura, nomenclatura ANT e comandos; apontar para campanhas e resultados datados. |
| `docs/EXPERIMENT_QUEUE_GUIDELINES.md` | Atualizar | Guia operacional central e coerente com filas atuais. Separar políticas permanentes de incidentes, capacidade momentânea de disco e históricos dos hosts. Xavier/Wolverine ativos; Quati/Fera inativos conforme usuário. |
| `docs/OPERATIONAL_DEPENDENCIES.md` | Atualizar | Mapa necessário à manutenção. Preservar observação datada e registrar separadamente confirmação do usuário; renovar somente após novas observações. |
| `docs/PROVENANCE.md` | Atualizar | Contrato necessário para experimentos/análises. Explicitar `structure` usado nas filas atuais, lacuna de `studies/` no hash geral de fontes, relocação e limitações dos manifestos antigos. |
| `docs/IMAGENET100_SETUP.md` | Atualizar | Procedimento de preparação e definição do recorte continuam valiosos mesmo com Quati inativa. Separar instrução reproduzível de caminhos/incidentes específicos da Quati; conferir disponibilidade dos manifests. |
| `docs/DEBUGGING_GUIDE.md` | Atualizar substancialmente | Recurso existe. Exemplos usam `main.py --config` e YAMLs inexistentes; CLI atual é `main.py train --exp-configs`. Separar logs textuais legados, viewer, coleta amostrada e observação estruturada; explicar `legacy_debug_metrics`. |
| `docs/AUTO_GPU_LAUNCHER.md` | Consolidar e remover depois | Preservar referência das opções reais de `auto_run_on_free_gpu.py` no guia operacional. Grande parte descreve scripts inexistentes e uma arquitetura Screen que não representa os serviços atuais. |
| `docs/LOGGING_SYSTEM.md` | Consolidar e remover depois | Afirma que todos os experimentos usam Screen e documenta monitor/filas ausentes. Transferir contrato dos logs para guia de observabilidade e eventos do orquestrador para guia de filas; evitar conservar o texto como descrição do sistema atual. |
| `docs/GPU_MEMORY_GUIDE.md` | Consolidar e remover depois | Extrair diagnóstico de utilização/VRAM para troubleshooting operacional. Remover receitas genéricas de matar processos/reiniciar driver da orientação rotineira: não identificam o proprietário e as dependências dos jobs. Ausência de PID numa visão isolada não prova processo morto. |
| `docs/RESULTS_AND_METRICS.md` | Dividir; retirar versão atual após migração | O README o anuncia como definição de métricas, mas ele é sobretudo um ranking de 2025 e recomendações de “melhor configuração”. Preservar tabelas como snapshot histórico; criar contrato atual de ACC, NME, média temporal, esquecimento e seleção. Não manter afirmação de benefício intrínseco de “InfoNCE Local”. |
| `docs/ANT_simetria.md` | Substituir por documento canônico | Ainda descreve FS como proposta e a implementação como somente Q1. Incorporar cobertura IV/FS, referência GR/AR, detach, máscaras, pesos e testes atuais em um documento de formulação. |
| `docs/ant_simetrico_markdown_corrigido.md` | Consolidar com o anterior e remover | Versão paralela com principalmente correções de Markdown/fórmulas; não justifica duas fontes. Reaproveitar a notação corrigida, mas revisar afirmações sobre a implementação atual. |
| `docs/CROSS_TASK_EMBEDDING_COLLISION.md` | Manter como hipótese de pesquisa | Questões e alternativas não se tornam inválidas por idade. Identificar como hipótese/extensão não implementada; retirar da lista de funcionalidades e do próximo passo obrigatório. |

## Campanhas e análises exploratórias

| Documento | Destino | Fundamentação e ação |
|---|---|---|
| `docs/ANT_DETACH_FACTORIAL_PLAN.md` | Atualizar junto à campanha | Preserva o desenho fatorial e controles. Separar plano original, execução real e extensão CUB; atualizar estados após validação dos resultados, não pela idade dos logs. |
| `docs/ant_loss_flattening_investigation_plan.md` | Histórico | Plano extenso que motivou diagnósticos e ablações já realizados. Vincular a resultados de maio e ao estudo posterior; extrair questões abertas. Não manter a seção “próximo passo imediato” como backlog atual. |
| `docs/ant_investigation_results.md` | Histórico científico relevante | Seis execuções, seed 1995, úteis para a escolha de formulação. Preservar tabelas e configuração original; qualificar afirmações gerais como “não muda performance”/“não prejudica generalização” pela evidência de uma seed. Não reinterpretar flags antigas com a semântica do código atual sem revisão da versão. |
| `studies/ant_mechanism/README.md` | Atualizar e manter | Define contrato dos artefatos, shadow variants, validação e execução smoke/short. Separar biblioteca reutilizável e instrução da campanha quando houver reestruturação. |
| `analysis/results/ant_mechanism_20260909/ANALISE_RESULTADOS.md` | Manter como edição científica | Explicita uma seed, três tarefas e limites de inferência; tem resultados mecanísticos úteis. Acrescentar ligação a fontes/manifests e preservar a edição ao surgirem novos resultados. |
| `docs/analysis/ANALISE_BASELINE_VS_ANT_GAP.md` | Histórico com errata | Guarda uma comparação e a correção do cálculo do gap. Restringir conclusão ao protocolo observado; não interpretar igualdade de médias de gap como prova geral de redundância da ANT. |
| `docs/analysis/DESCOBERTA_INFONCE_MAXIMIZA_GAP.md` | Consolidar no histórico anterior e remover duplicata | Reitera as mesmas evidências, com afirmações mais fortes (“95%+” de confiança) sem procedimento de estimação documentado. Preservar a correção do bug de medição uma única vez. |
| `docs/analysis/TORNAR_ANT_MAIS_RELEVANTE.md` | Histórico de hipótese superada | Recomenda gap maximization e prevê +2–3% sem resultado experimental correspondente. Não atualizar como receita; resumir o que motivou a tentativa e vincular ao resultado posterior. |

## Guias da pasta analysis e resultados antigos

| Documento | Destino | Fundamentação e ação |
|---|---|---|
| `analysis/README.md` | Reescrever como índice atual | Diretório precisa de uma entrada. Substituir descrição da comparação local/global InfoNCE como eixo científico por parsing, seleção, estatística, mecanismos e resultados históricos. |
| `analysis/STRUCTURE.txt` | Remover | Duplica árvore/uso do README, enumera scripts e contagens antigos e referencia `REORGANIZATION_SUMMARY.md` inexistente. Não contém evidência científica exclusiva. |
| `analysis/scripts/README.md` | Consolidar no índice e remover | Diz que gap maximization está ativo e aponta configuração ausente. Extrair descrição dos leitores legados e da implementação experimental de referência. |
| `analysis/scripts/STATUS.md` | Remover após preservar breve nota histórica do bug | Estado de desenvolvimento antigo com runs parciais e interpretação de gap do baseline como zero, contradita pelas análises posteriores. “Aguardando experimentos” não representa o estado atual. |
| `analysis/scripts/COMPARE_EXPERIMENTS_GUIDE.md` | Consolidar como uso de leitor legado | Script ainda existe, mas o guia usa `analysis_scripts/`, formatos antigos e chama `avg_nme1` de forgetting. Preservar somente CLI realmente suportada e semântica dos logs; retirar o guia longo depois. |
| `analysis/results/baseline_metrics/README_APRESENTACAO.md` | Consolidar em README histórico curto | Dados/figuras ainda podem ter valor; narrativa para apresentação de 2025, números para memorizar e checklist estão superados. Manter contexto, entradas, comando e limites; roteiro atual pertence à apresentação atual. |
| `analysis/results/baseline_metrics/baseline_distances_report.txt` | Histórico | Saída quantitativa do diagnóstico; preservar com fontes. Não é documentação corrente a atualizar manualmente. |
| `analysis/results/baseline_vs_local/comparison_report.md` | Histórico com ressalva metodológica | Chama +0,32 de melhora “significativa” sem teste apresentado e trata local/global como métodos distintos. Preservar observações; retirar a conclusão causal e significância da orientação atual. |
| `analysis/results/cifar100_50-10_comparison/cifar100_50-10_comparison_report.md` | Remover | Contém 7011%/6935% e delta −76,00, enquanto `comparison_report.md` traz mesmos dados como 70,11%/69,35% e −0,76. Erro de escala demonstrado por comparação direta. |
| `analysis/results/cifar100_50-10_comparison/comparison_report.md` | Histórico | Versão com escala consistente. Manter como observação datada, com ressalva sobre equivalência InfoNCE e ausência de conclusão causal a partir do rótulo local/global. |
| `analysis/results/cifar100_50-10_comparison/cifar100_50-10_global_vs_local_report.md` | Consolidar no histórico de 50-10 | Repete comparação e objetivo superado, e descreve 50 classes iniciais + 10 tarefas incrementais. Confrontar suas métricas antes de fundir: não presumir identidade de runs somente pelo assunto. |
| `analysis/results/three_experiments_comparison/three_experiments_report.md` | Remover da árvore corrente | Anuncia ganho +9,97 comparando baseline 63,51 em T10 a ANT m=0,5 73,48 em T6; o relatório posterior de margens mostra a mesma sequência completa. Guarda uma comparação de horizontes incompatíveis, não evidência de ganho. Preservar uma errata curta do erro, fontes e entradas. |
| `analysis/results/three_experiments_comparison/three_margin_comparison_report.md` | Histórico | Mostra dez tarefas para os três métodos e ganhos finais de +0,30/+0,23. Preservar como ablação antiga; validar semântica de gap/violação e não generalizar “vencedor” sem múltiplas seeds. |
| `analysis/results/experiments_comparison/README.md` | Consolidar como índice do histórico ANT+Gap | Repete dois documentos em `docs/analysis/`, referencia caminhos antigos e extrapola “InfoNCE é suficiente” para o trabalho todo. Manter identificação das entradas, outputs e correção aplicada. |
| `analysis/results/experiments_comparison/comparison_summary.txt` | Histórico | Saída quantitativa associada à comparação; preservar como edição e ligar às entradas, sem promover a contrato atual do método. |
| `analysis/results/nme1_curves/README.md` | Histórico | Já identifica diferença entre runs de 7/8 e 10 tarefas, uma ressalva útil. Remover narrativa de apresentação e status “próximas análises”; revisar a distinção entre queda T1→Tfinal e forgetting por tarefa. |
| `analysis/results/nme1_curves/nme1_summary.txt` | Histórico | Tabela inclui quantidade de tarefas por run. Preservar essa informação; não usar ranking bruto de última acurácia entre horizontes diferentes. |
| `analysis/results/statistics/current/report.txt` | Manter edição datada | Comparação de nove métodos/12 blocos, p≈0,506. Manifesto não verifica contra fonte atual; preservar edição e recuperar contexto. Não atualizar texto manualmente com resultados de outra seleção. |
| `results_report.md` | Remover saída corrente/continuar fora do Git | Relatório gerado grande e sem papel de guia; fluxo atual usa HTML/PDF. Preservar externamente apenas se identificar uma edição efetivamente citada e não recuperável por outra fonte. |

## Documentação dos submódulos

| Documento | Destino | Fundamentação e ação |
|---|---|---|
| `ANT_Monografia/ROTEIRO.md` | Atualizar | Hierarquia científica e decisões editoriais úteis. Atualizar referência conectada/detach, experimentos e estado de agosto; manter orientação, sem duplicar tabelas voláteis. |
| `ANT_Monografia/TODO_ESCRITA.md` | Atualizar ou migrar tarefas para backlog único | Contém pendências reais. Reconciliar com capítulos e figuras existentes, remover a ambiguidade de `tagfex_clean.py` como implementação canônica e distinguir concluído de ainda não validado. |
| `ANT_Monografia/VALIDACAO_REFERENCIAS.md` | Manter como auditoria datada | Registra alterações bibliográficas e fontes. Não presumir que 29 entradas de agosto cubram acréscimos posteriores; acrescentar novas auditorias sem apagar a anterior. Não foi revalidada externamente neste trabalho. |
| `ANT_Monografia/figuras/README.md` | Atualizar | Documenta instalação, geração e fontes. Conferir caminhos reorganizados, scripts de setup disponíveis e figuras referenciadas no LaTeX. |
| `ANT_Monografia/figuras/data/README.md` | Atualizar | Essencial: distingue imagens didáticas, simulações, dados experimentais e métricas derivadas. Conferir fontes e seleção após novas campanhas. |
| `ANT_Monografia/apresentacao_qualificacao/README.md` | Atualizar | Geradores existem; manter como entrada da apresentação e alinhar inventário de slides, dependências e comandos. |
| `ANT_Monografia/apresentacao_qualificacao/ROTEIRO_NARRATIVO.md` | Atualizar | Explica função e evidência de cada slide. Alinhar números de slides, detach, hipótese e edição dos resultados, sem atualizar números isoladamente. |
| `ANT_Monografia/figuras/data/desempenho/autorank_avgacc_principais_refdetach.txt` | Manter edição datada | Outra seleção de nove métodos, com RefDetach e p≈0,0976. Não conflitar nem substituir automaticamente p≈0,506: métodos comparados são diferentes. |
| `ANT_WOPFACOM/README.md` | Atualizar | Compilação e origem do artigo continuam relevantes. Fixar revisão da monografia/dados usados em vez de “versão corrente” e “48 logs” sem identificação da edição. |
| `ANT_WOPFACOM/apresentacao/README.md` | Reescrever substancialmente | Anuncia `build_presentation.py` e PPTX/PDF/ODP que não existem no diretório. O material real contém figuras, exportadores e roteiro. Descrever o que está disponível e registrar entregas ausentes, sem prometer regeneração que não funciona. |
| `ANT_WOPFACOM/apresentacao/ROTEIRO_5MIN.md` | Manter como edição da apresentação | Tem função própria e ressalvas estatísticas. Vincular à edição do artigo/apresentação; revisar somente quando essa edição mudar. |
| `ANT_WOPFACOM/apresentacao/figuras_png/README.md` | Manter ou incorporar seção técnica no README da apresentação | Explica geração, mosaicos e relação com figuras da monografia; não é redundância meramente por chamar-se README. Preservar detalhes junto aos assets se continuarem sendo distribuídos separadamente. |

`ROTEIRO.md`, `TODO_ESCRITA.md` e `VALIDACAO_REFERENCIAS.md` da monografia
merecem atenção na política de versionamento: regras `*.md` podem esconder
documentos locais de valor. O fato de existirem no disco não garante sua
preservação no submódulo.

## Documentos já excluídos no working tree

- `paper/ROTEIRO.md` e `paper/TODO_ESCRITA.md`: aceitar remoção da cópia antiga
  somente depois de preservar a versão canônica no submódulo; não manter duas
  cópias do roteiro.
- `paper/ANT4CIL/README_MONOGRAFIA.md`: tratar com o arquivo do manuscrito
  ANT4CIL. Não restaurar como guia atual; preservar contexto com a versão
  histórica do artigo antes de concluir a retirada desse conjunto.

## Fontes canônicas propostas

1. README de entrada: propósito, instalação, comandos e navegação.
2. Guia de método: formulação, variantes, semântica histórica e testes.
3. Guia de métricas/análise: contratos, seleção e limites de inferência.
4. Guia de observabilidade: logs legados, matrizes e observer estruturado.
5. Guia de execução: filas, recursos, serviços, recuperação e diagnóstico.
6. Guia de proveniência, com mapa operacional separado por ser datado.
7. Preparação de datasets: procedimentos específicos, inclusive ImageNet-100.
8. README por campanha, por publicação e por gerador distribuído separadamente.

Não criar uma nova cópia de cada assunto em todas as camadas. Campanhas
referenciam os contratos; resultados referenciam campanhas/entradas;
publicações referenciam edições de resultados.

## Ordem sugerida

1. Retirar da navegação corrente os relatórios com erro comprovado e os guias
   operacionais de scripts inexistentes; criar substitutos antes da remoção
   dos guias que ainda possuem conteúdo reutilizável.
2. Consolidar simetria, logs/diagnóstico GPU e navegação de `analysis/`.
3. Separar histórico científico de documentação atual; inserir erratas
   necessárias e referências à investigação que substituiu cada hipótese.
4. Atualizar README, métricas, método, campanhas e apresentações com a mesma
   nomenclatura e a mesma edição de resultados.
5. Conferir links e exemplos estáticos, CLI `--help`, caminhos e artefatos.
   Não executar exemplos de treinamento/recuperação para testar documentação
   enquanto campanhas estão ativas.

As recomendações não autorizam remover dados, logs ou scripts por consequência
de retirar um documento. Cada dependência continua sujeita ao mapa operacional.

## Verificação da revisão aplicada

- 74 links locais da documentação ativa conferidos, sem destinos ausentes.
  Links e comandos dentro de registros históricos citados não são instruções
  atuais; sua localização original é explicitada no arquivo.
- Dez interfaces `--help` executadas com sucesso: treinamento, relatório,
  estatística, proveniência, launcher, preparação Tiny ImageNet/CUB,
  componentes de loss, comparador legado e gerador dos slides de qualificação.
- Hashes de 305 arquivos de código, scripts, filas e configurações mantidos
  iguais ao início da revisão. Nenhum treinamento ou gerador de resultados
  foi iniciado para validar os exemplos.
- `git diff --check` passou no principal e nos dois submódulos.
- O ajuste de ignore se limitou ao relatório Markdown gerado e às exceções
  dos três documentos editoriais/bibliográficos da monografia. Não foram
  registrados commits, alterados ponteiros de submódulos ou modificados dados.

As cópias de trabalho e de segurança da revisão foram transferidas de `/tmp`
para `.local/documentation-review/` na raiz do projeto, com diretórios 700 e
arquivos 600. A transferência foi conferida por SHA-256 e a origem temporária
foi removida. O destino é local, privado e ignorado pelo Git.
