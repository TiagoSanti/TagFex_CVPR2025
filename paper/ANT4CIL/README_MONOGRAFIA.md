# Uso do esboço ANT4CIL na monografia

Este diretório preserva, sem alterações, uma versão anterior do artigo ANT4CIL.
Embora tenha sido descrito como um esboço para ICLR, o pacote extraído utiliza o
template da AAAI 2026. O arquivo de trabalho principal é `main.tex`.

Na monografia atual, a hierarquia obrigatória é: ANT como núcleo; referência,
cobertura e normalização como decisões metodológicas; formulações alternativas
como ablações; Avg-K/SBS como investigações auxiliares; e colisões/protótipos
como extensões. Este material histórico não redefine essa hierarquia.

## Material que pode ser reaproveitado

### Introdução

- progressão entre CIL, esquecimento catastrófico, aprendizado contrastivo e a
  motivação da ANT;
- formulação inicial das contribuições;
- exemplo intuitivo de atualizações que continuam após a separação de pares.

O texto deve ser convertido em problema e hipótese de pesquisa. Afirmações de
que a InfoNCE exacerba diretamente o esquecimento ou de que a ANT supera
consistentemente o estado da arte não devem ser mantidas sem a validação atual.

### Fundamentação Teórica e Trabalhos Relacionados

- revisão das famílias data-centric, model-centric e algorithm-centric;
- notação inicial do cenário CIL;
- descrição conceitual de expansão dinâmica, repetição e destilação;
- bibliografia de `aaai2026.bib`, após conferência de cada entrada.

A seção `Task Agnostic` mistura ausência de identidade de tarefa na inferência
com ausência de fronteiras no treinamento. Esses cenários devem permanecer
separados na monografia.

### Metodologia de referência: TagFex

- equações das perdas de classificação principal e auxiliar;
- motivação para atributos específicos e agnósticos à tarefa;
- descrição de destilação contrastiva, transferência e atenção de fusão;
- hiperparâmetros e protocolos experimentais como ponto de partida.

As equações devem ser auditadas contra `tagfex_clean.py` e
`methods/tagfex/tagfex.py`. O artigo usa uma descrição arquitetural mais antiga
e contém sinais/notação que podem não corresponder ao código executado.

### Proposta ANT

- decomposição da matriz de similaridade e exemplo patológico;
- ideia de margem relativa ao negativo de referência;
- motivação das referências global e local;
- figuras `fig_a.png`, `fig_a_2.png`, `fig_b.png`, `fig_c.png` e `fig_d.png`.

Antes de reutilizar as figuras, conferir se representam a ANT assimétrica
antiga ou a variante simétrica completa atualmente implementada.

### Metodologia experimental

- estrutura de descrição de datasets e divisões incrementais;
- organização de tabela de resultados e estudo de ablação;
- lista inicial de baselines.

Os valores não devem ser copiados para a monografia. A fonte atual deve ser
`results_report.md`, com separação explícita entre uma e múltiplas sementes.

## Incompatibilidades com a implementação atual

1. O artigo descreve ANT somente sobre uma visão; o código também implementa
   `ant_symmetric_full`, usando a matriz completa e removendo self e positivo.
2. O indicador estrito apresentado na equação do artigo não representa todas
   as formulações atuais: `logsumexp`, `expm1`, `softplus`, `topk` e
   `active_only` possuem agregações diferentes.
3. A ANT `logsumexp` atual possui piso de contagem `log(N_valid)`. Isso precisa
   ser explicado ao interpretar a loss bruta.
4. A descrição de `Local Anchor Normalization` atribui gradientes mais
   balanceados à subtração de um máximo por linha. Na implementação atual,
   esse deslocamento ocorre dentro de uma expressão invariável a constantes por
   linha; a alegação deve ser demonstrada ou removida.
5. O texto associa diretamente gradiente residual, parâmetro não essencial e
   esquecimento. Esses três conceitos precisam de definições operacionais e de
   evidência separada.
6. Os resultados e conclusões precedem as campanhas mais recentes e não
   incluem ANT simétrica, Avg-K, SBS, CUB-200 ou as investigações das
   formulações alternativas.

## Problemas editoriais a corrigir ao reaproveitar

- padronizar `Class-Incremental Learning`, `continual learning` e
  `task-agnostic`;
- corrigir `Tunning` para `Tuning` e `logexpsum` para `log-sum-exp`;
- conferir o par positivo nas equações da InfoNCE (`z_i` e sua outra visão),
  evitando representá-lo como autossimilaridade trivial;
- revisar sinais das equações de perda do TagFex;
- substituir linguagem de artigo concluído por linguagem de qualificação:
  questões, hipóteses, evidências preliminares e trabalho restante;
- revisar inglês e converter o conteúdo selecionado para português acadêmico,
  sem tradução literal.

## Ordem recomendada de consulta

1. Usar `main.tex` como fonte histórica de intuições e referências.
2. Confirmar cada mecanismo no código atual.
3. Confirmar cada resultado nos reports atuais.
4. Só então adaptar equações, figuras e argumentos à monografia.
