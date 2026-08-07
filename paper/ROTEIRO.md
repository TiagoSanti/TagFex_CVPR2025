# Roteiro científico da monografia de qualificação

Este documento controla a narrativa e a hierarquia editorial. Os arquivos
`.tex` permanecem como esqueleto: títulos e comentários de escrita, sem redação
definitiva nesta etapa.

## Narrativa central

```text
PROBLEMA
Objetivos de destilação contrastiva continuam produzindo atualizações
sobre relações que podem já estar suficientemente preservadas.
        ↓
PERGUNTA
É possível restringir essas atualizações sem prejudicar a aquisição
de novas classes?
        ↓
HIPÓTESE
Otimizar apenas violações relevantes reduz ajustes não essenciais e
interferência, preservando a plasticidade.
        ↓
BASELINE
TagFex e sua destilação contrastiva.
        ↓
PROPOSTA CENTRAL
ANT — Avoid Non-essential Tuning.
        ↓
DECISÕES DE PROJETO
Referência ANT: aGlobal / aLocal
Cobertura ANT: intra-view / aSymFull
Normalização InfoNCE: nGlobal / nLocal
        ↓
MECANISMO
referência → margem → violações → termos ativos → agregação → gradiente
        ↓
VALIDAÇÃO
accuracy / NME / forgetting
+ adjusted loss / active ratio / violations / hard negatives / gaps
        ↓
INVESTIGAÇÕES AUXILIARES
Avg-K / SBS
        ↓
EXTENSÕES
colisões entre tarefas / negativos baseados em protótipos
```

## Hierarquia das contribuições

1. **Núcleo da dissertação:** ANT.
2. **Decisões da formulação:** referência `aGlobal/aLocal`, cobertura
   `intra-view/aSymFull` e normalização InfoNCE `nGlobal/nLocal`.
3. **Ablações da formulação:** `logsumexp`, `expm1`, `softplus`, `top-k` e
   `active-only`.
4. **Investigações auxiliares:** Avg-K e SBS.
5. **Extensões ainda não consolidadas:** colisão entre tarefas e negativos
   baseados em protótipos antigos.

Avg-K e SBS só devem subir nessa hierarquia se evidência conceitual e
experimental futura justificar contribuição independente.

## Estrutura e extensão planejada

| Capítulo | Função narrativa | Extensão |
|---|---|---:|
| 1. Introdução | problema, perguntas, hipóteses e objetivos | 6–9 páginas |
| 2. Fundamentação Teórica e Trabalhos Relacionados | fundamentos, TagFex e lacuna | 12–17 páginas |
| 3. Proposta Metodológica | ANT, decisões, mecanismo e evidências | 22–32 páginas |
| 4. Plano de Trabalho e Cronograma | trabalho restante e riscos | 3–5 páginas |

Meta total: aproximadamente 40–70 páginas. O Capítulo 3 deve ser o maior.

## Decisões editoriais

- As variantes são derivadas de decisões matemáticas, não apresentadas como uma
  coleção de configurações.
- `aGlobal/aLocal` definem a referência da ANT.
- `intra-view/aSymFull` definem a cobertura da matriz da ANT.
- `nGlobal/nLocal` pertencem à InfoNCE e não são equivalentes às referências ANT.
- Formulações alternativas são ablações da mesma ideia central.
- Resultados seguem hipótese → expectativa → métrica → observação →
  interpretação → limitação.
- Resultados são marcados como completos, parciais, em execução, falhos ou
  exploratórios.
- Conclusões usam evidência forte, tendência, inconclusiva ou não sustentada.

## Fontes técnicas canônicas

- `../tagfex_clean.py`: referência simplificada do fluxo e das perdas.
- `../methods/tagfex/tagfex.py`: implementação usada no projeto; deve ser
  confrontada com a versão simplificada antes das equações definitivas.
- `../results_report.md`: resultados agregados atuais.
- `../docs/ant_investigation_results.md`: investigação do piso da ANT e das
  formulações alternativas.
- `../docs/ANT_simetria.md`: histórico da decisão de cobertura simétrica.
- `ANT4CIL/main.tex`: fonte histórica de intuições/equações, não canônica.
- `ANT4CIL/README_MONOGRAFIA.md`: mapa de reaproveitamento e incompatibilidades.

## Inconsistências que não podem ser resolvidas por suposição

1. **nLocal/nGlobal:** a subtração de constante por linha da implementação
   atual parece algebricamente invariável no objetivo InfoNCE. A narrativa
   histórica de gradientes mais balanceados precisa de prova, teste numérico ou
   revisão.
2. **Definição de ajuste não essencial:** loss residual, gradiente não nulo,
   violação de margem e dano ao conhecimento anterior não são equivalentes.
3. **ANT histórica versus atual:** o artigo antigo descreve cobertura intra-view
   e indicador estrito; o código atual inclui aSymFull e cinco agregações.
4. **Resultados:** diferenças pequenas e execuções de uma única semente não
   sustentam superioridade. Usar sempre o status e a dispersão disponíveis.
5. **Task-agnostic:** ausência de `task-id` na inferência não implica ausência de
   fronteiras de etapa no treinamento.

## Figuras e tabelas estruturantes

- pipeline TagFex com o ponto de intervenção da ANT;
- exemplo geométrico de relação adequada versus violação;
- matriz em blocos Q1–Q4 para intra-view e aSymFull;
- tabela referência × cobertura × granularidade × custo;
- exemplo algébrico nGlobal/nLocal;
- matriz hipótese × comparação × dataset × métricas × status;
- gráfico do piso da raw loss versus adjusted loss/active ratio;
- cronograma mensal com marcos.

## Ordem sugerida de redação

1. Fechar notação, protocolo e pipeline TagFex no Capítulo 2.
2. Auditar e formalizar a ANT base no Capítulo 3.
3. Resolver/documentar nGlobal/nLocal.
4. Fixar perguntas, hipóteses e critérios no Capítulo 1.
5. Construir a matriz de experimentos e selecionar evidências.
6. Preencher o cronograma com datas confirmadas.
