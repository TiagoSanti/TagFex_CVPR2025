# Formulação e variantes ANT

Referência da implementação em `methods/tagfex/tagfex.py`, revisada em
11/09/2026. O nome adotado pela monografia é *Avoid Non-essential Tuning*;
“Adaptive Negative Threshold” aparece em documentos antigos. A hipótese de
preservação deve ser avaliada experimentalmente, não inferida da máscara.

## TagFex e ponto de intervenção

O TagFex mantém um ramo agnóstico à tarefa, projetor contrastivo e ramos
específicos que se expandem ao longo das tarefas. Ramos específicos anteriores
são congelados; cópias anteriores do ramo agnóstico/projetor servem de professor
na destilação. O ramo agnóstico corrente continua sendo treinado. A atenção
participa da transferência para o ramo específico; a classificação usa as
features dos ramos específicos. A descrição antiga de uma única TA congelada
depois da primeira tarefa não representa esse fluxo.

A ANT atua nos termos contrastivos corrente e de destilação. Cada termo usa:

$$L_c=\alpha L_{\mathrm{NCE}}+\beta L_{\mathrm{ANT}}.$$

`nce_alpha` é α e `ant_beta` é β; não há fator implícito `1−β`. O objetivo
completo também contém classificação e, nas etapas pertinentes, auxiliar,
transferência e destilação, com pesos externos definidos no YAML. TeacherAvg-K
modifica o professor; SBS modifica a seleção de exemplares. São ablações
distintas da geometria da ANT.

## Matriz, máscaras e referência

Com duas views de B amostras, N=2B. Na ramificação corrente, S contém
similaridades de cosseno dos embeddings; em KD, compara projeções do estudante
e do professor. A InfoNCE usa a matriz completa, remove a diagonal do
denominador e identifica o positivo pelo deslocamento B. A temperatura divide
os logits da InfoNCE; a margem ANT opera na similaridade antes dessa divisão.

| Dimensão | Opção | Configuração e domínio |
|---|---|---|
| Cobertura | IV | `ant_symmetric_full: false`: bloco superior esquerdo B×B, sem diagonal |
| Cobertura | FS | `ant_symmetric_full: true`: N×N, sem diagonal nem par positivo |
| Referência | GR | `ant_max_global: true`: máximo entre negativos válidos de todo o domínio |
| Referência | AR | `ant_max_global: false`: máximo dos negativos válidos por linha |
| Gradiente | Conectada | `ant_detach_reference: false` |
| Gradiente | Destacada | `ant_detach_reference: true` |

IV implementado usa a primeira view; não significa média dos dois blocos
intra-view. Para IV há B−1 negativos por linha; para FS, 2B−2. As oito
combinações cobertura × referência × detach são condições distintas.

Para negativos válidos j da linha i:

$$v_{ij}=s_{ij}-r_i+\gamma,\qquad
L_{\mathrm{ANT}}=\frac1A\sum_i\log\sum_{j\in\mathcal N_i}
\exp(\max(0,v_{ij})).$$

Essa é a formulação `logsumexp` padrão; A é o número de linhas do domínio.
As posições inválidas são mascaradas depois do ReLU. Negativos válidos
inativos ainda contribuem `exp(0)=1` ao somatório: o piso é `log(|N_i|)`.
Por isso, loss bruta quase constante não demonstra gradiente nulo.

O `detach` mantém referência, máscara ativa e valor forward para a mesma S,
mas remove o caminho de derivação através do máximo. Os testes verificam
mudança do gradiente na referência. Não se deve concluir daí que a variante
destacada melhora sempre a classificação. Outras formulações disponíveis são
`expm1`, `softplus`, `topk` e `active_only`; seus resultados antigos estão no
[registro da investigação](history/ant_investigation_results.md).

## Identidade histórica e interpretação

| Nome histórico | Apresentação |
|---|---|
| `antGlobal` | ANT-IV-GR |
| `antLocal` | ANT-IV-AR |
| `antSymmetricFull` | ANT-FS-AR no esquema atual de nomes |
| `antSymmetricFullGlobal` | ANT-FS-GR |
| `refDetached` | Referência destacada |
| `avgK3/5/10` | TeacherAvg-3/5/10 |

Historicamente, FS nem sempre respeitava a flag GR da mesma forma. Recupere a
revisão e a configuração efetiva antes de reinterpretar um run antigo.
`nGlobal/nLocal` descrevem caminhos históricos da InfoNCE que diferiam por
uma constante por linha, cancelada algebricamente no objetivo e no gradiente
em aritmética exata. O cálculo atual ignora `infonce_max_global`; os nomes são
preservados para rastrear artefatos. Isso não significa identidade bit a bit
das trajetórias de treinamentos antigos.

Com β=0, parâmetros exclusivos da ANT ficam inativos; TeacherAvg e SBS ainda
podem produzir ablações distintas. A seleção deve preservar esses controles,
usar uma observação por método/protocolo/seed e conferir a revisão do código.
Consulte o [contrato de métricas](RESULTS_AND_METRICS.md).

## Evidência e testes

O [estudo do mecanismo](../studies/ant_mechanism/README.md) observa as variantes
sobre as mesmas matrizes e em trajetórias separadas. Seus
[resultados](../analysis/results/ant_mechanism_20260909/ANALISE_RESULTADOS.md)
cobrem CIFAR-100, uma seed e três tarefas. As campanhas de desempenho e
detach têm escopo próprio; consulte o [plano fatorial](ANT_DETACH_FACTORIAL_PLAN.md).

```bash
python -m pytest -q tests/test_ant_reference_detach.py tests/test_ant_mechanism_analytics.py
```

Esses testes não substituem avaliação multi-seed, controle de clipping,
memória, dados ou validação da seleção de resultados.

## Identidade adotada nos relatórios atuais

Desde a revisão de 12/09/2026, ANT com referência destacada é a identidade
canônica dos relatórios. Resultados conectados permanecem como legado explícito,
sem serem fundidos com os destacados. Essa nomenclatura não altera os nomes
históricos de diretórios nem demonstra superioridade estatística.

Na campanha CUB-200 100+20 de cinco seeds, `small_base: true` seleciona o stem
7×7/stride 2. Esse controle arquitetural é científico e deve acompanhar a
proveniência; o basename histórico não distingue esse stem do default anterior.
