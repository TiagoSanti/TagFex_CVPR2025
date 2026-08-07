# TODO da escrita da qualificação

Estados: `[ ]` pendente, `[~]` em andamento, `[x]` concluído, `[!]` bloqueado.

## Prioridade 0 — consistência técnica

- [ ] Comparar `tagfex_clean.py` e `methods/tagfex/tagfex.py` para fixar a
  implementação canônica descrita na monografia.
- [!] Demonstrar/testar numericamente valores e gradientes de nGlobal/nLocal.
- [ ] Definir operacionalmente relação preservada, violação relevante e ajuste
  não essencial.
- [ ] Auditar sinais, máscaras e pares positivos das equações históricas.
- [ ] Padronizar professor, âncora, positivo, negativo e similaridade em todo o
  texto; manter identificadores de código apenas quando necessário.

## Capítulo 1 — problema e hipótese

- [ ] Redigir contextualização sem antecipar a ANT.
- [ ] Formular problema independente da solução.
- [ ] Fechar uma questão principal e apenas questões secundárias testadas.
- [ ] Escrever H0/H1 e a matriz hipótese → métrica → experimento → critério.
- [ ] Confirmar objetivos específicos contra o plano experimental final.

## Capítulo 2 — TagFex e lacuna

- [ ] Fixar notação e significado exato de task-agnostic no protocolo.
- [ ] Descrever o TagFex passo a passo com equações auditadas.
- [ ] Produzir figura do pipeline e ponto de intervenção da ANT.
- [ ] Escrever pseudocódigo de uma etapa incremental baseline.
- [ ] Auditar e incorporar referências selecionadas de `ANT4CIL/aaai2026.bib`.

## Capítulo 3 — ANT e validação

- [ ] Criar exemplo geométrico pequeno de relações preservadas/violadoras.
- [ ] Formalizar matriz, máscaras, referência, margem, violação e agregação.
- [ ] Produzir figura Q1–Q4 consistente com a implementação atual.
- [ ] Produzir tabela aGlobal/aLocal × intra-view/aSymFull.
- [ ] Consolidar formulações alternativas em uma ablação única.
- [ ] Escrever pseudocódigo TagFex+ANT e marcar diferenças para o baseline.
- [ ] Montar matriz pergunta/hipótese × experimento × dataset × métricas.
- [ ] Confirmar status de cada execução nos logs/reports.
- [ ] Selecionar resultados com múltiplas sementes; marcar resultados parciais.
- [ ] Explicar piso `log(N_valid)` e usar adjusted loss/active ratio/hard negatives.
- [ ] Classificar evidência por hipótese, sem extrapolar diferenças pequenas.
- [ ] Manter Avg-K/SBS depois do núcleo e extensões ao final.

## Capítulo 4 — conclusão do trabalho

- [ ] Confirmar calendário do programa e data-alvo da defesa.
- [ ] Definir conjunto mínimo de experimentos que responde à hipótese principal.
- [ ] Preencher estado atual, atividades, entregáveis e cronograma mensal.
- [ ] Registrar contingência para GPU, variância e resultados inconclusivos.

## Verificações editoriais finais

- [ ] Conferir que nenhuma seção planejada tenha apenas um parágrafo curto.
- [ ] Conferir labels, referências cruzadas, figuras e bibliografia.
- [ ] Compilar e revisar sumário, quebras, páginas órfãs e avisos LaTeX.
- [ ] Conferir extensão aproximada: 6–9 / 12–17 / 22–32 / 3–5 páginas.
