> Edição científica preservada. Fontes: streams e snapshots da campanha `ant_mechanism_20260909`; agregador `studies/ant_mechanism/summarize.py`. As tabelas adjacentes não substituem manifestos e artefatos originais. O [contrato do estudo](../../../studies/ant_mechanism/README.md) define reprodução e validação. O responsável confirmou a Fera inativa atualmente; isso não altera a origem histórica destas execuções.

# Análise do estudo observacional do mecanismo ANT

## Escopo e validade

Esta análise usa as nove condições concluídas na Fera para CIFAR-100 10x10,
seed 1993, limitadas às três primeiras tarefas. Foram validados 47.960
registros estruturados (aproximadamente 1,47 GiB incluindo snapshots). As
tabelas agregadas locais ocupam 56 MiB.

As comparações de desempenho abaixo têm somente uma seed e três tarefas. Elas
servem para formular e explicar mecanismos, não para declarar superioridade
estatística. Os registros de batches e épocas são correlacionados e não devem
ser tratados como réplicas independentes.

## Resultado principal

O melhor compromisso observado foi **ANT-IV-AR-D**: referência por âncora,
escopo intra-view e referência destacada. Na terceira tarefa, alcançou 84,70%
de ACC e 83,63% de NME, contra 84,17% e 82,50% do Baseline. Os ganhos foram,
respectivamente, +0,53 e +1,13 ponto percentual.

O resultado mecanístico mais forte é controlado e independe da trajetória de
treinamento: em 315 matrizes nas quais as oito variantes foram recalculadas
sobre exatamente as mesmas similaridades, o `detach` preservou todos os
valores forward, máscaras, violações e referências. Ele alterou somente o
gradiente.

Sem `detach`, a referência recebe gradiente líquido negativo em todas as
matrizes observadas. Como a descida do gradiente subtrai esse valor, o modelo é
induzido a **aumentar** a similaridade do negativo usado como referência. Com
`detach`, resta apenas o gradiente direto positivo: a referência passa a ser
repelida, como os demais negativos. A proporção de referências com direção de
repulsão mudou de 0% para 100% nas quatro formulações.

## Desempenho incremental

Valores em porcentagem. `AIA` é a média das acurácias ao final das tarefas 1,
2 e 3. Todos os métodos obtiveram 93,40% de ACC e 93,20% de NME na primeira
tarefa; as diferenças apareceram nas etapas incrementais.

| Método | ACC T2 | ACC T3 | Δ ACC T3 | ACC AIA | NME T2 | NME T3 | Δ NME T3 | NME AIA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 85,20 | 84,17 | 0,00 | 87,59 | 84,35 | 82,50 | 0,00 | 86,68 |
| ANT-IV-GR | 85,55 | 83,27 | -0,90 | 87,41 | 84,70 | 82,50 | 0,00 | 86,80 |
| ANT-IV-GR-D | 86,00 | 83,67 | -0,50 | 87,69 | 84,75 | 82,37 | -0,13 | 86,77 |
| ANT-IV-AR | 85,60 | 84,53 | +0,37 | 87,84 | 84,80 | 82,90 | +0,40 | 86,97 |
| **ANT-IV-AR-D** | **86,25** | **84,70** | **+0,53** | **88,12** | **85,85** | **83,63** | **+1,13** | **87,56** |
| ANT-FS-GR | 85,60 | 83,80 | -0,37 | 87,60 | 85,55 | 82,93 | +0,43 | 87,23 |
| ANT-FS-GR-D | 85,60 | 83,47 | -0,70 | 87,49 | 84,95 | 82,60 | +0,10 | 86,92 |
| ANT-FS-AR | 85,20 | 83,27 | -0,90 | 87,29 | 84,30 | 82,00 | -0,50 | 86,50 |
| ANT-FS-AR-D | 85,55 | 83,67 | -0,50 | 87,54 | 84,25 | 82,73 | +0,23 | 86,73 |

O `detach` melhorou a ACC final em IV-GR (+0,40), IV-AR (+0,17) e FS-AR
(+0,40), mas piorou FS-GR (-0,33). Portanto, ele é mecanicamente bem definido,
mas seu ganho de classificação não é universal nesta amostra.

### Retenção e plasticidade

| Método | Classes antigas (20) em T3 | Classes novas (10) em T3 | Esquecimento médio T1/T2 |
|---|---:|---:|---:|
| Baseline | 82,30 | 87,90 | 4,90 |
| ANT-IV-AR | **82,75** | 88,10 | **4,60** |
| ANT-IV-AR-D | 82,70 | **88,70** | 5,25 |

ANT-IV-AR-D não venceu por reduzir mais o esquecimento médio que o Baseline;
o principal diferencial foi a plasticidade nas classes novas, mantendo a
acurácia das classes antigas praticamente estável. A versão conectada IV-AR
teve o menor esquecimento médio. Isso recomenda apresentar ACC/NME junto com
retenção por bloco, e não interpretar a ACC final isoladamente.

## O que o `detach` muda no gradiente

As linhas abaixo usam as 315 matrizes controladas. A coluna `||g_D||/||g||`
compara a norma ANT destacada com a conectada. O cosseno é calculado entre os
gradientes ANT e InfoNCE no espaço das similaridades.

| Operador | `||g_D||/||g||` | Redução da norma | Cosseno conectado | Cosseno destacado |
|---|---:|---:|---:|---:|
| IV-GR | 3,39% | 96,61% | -0,008 | +0,042 |
| IV-AR | 13,54% | 86,46% | -0,015 | +0,038 |
| FS-GR | 1,75% | 98,25% | -0,011 | +0,085 |
| FS-AR | 10,37% | 89,63% | -0,025 | +0,077 |

A referência conectada acumula a derivada de todas as violações que dependem
dela. Esse efeito é especialmente extremo em GR, pois uma única referência
global concentra contribuições de muitas âncoras. Ao destacá-la, a ANT se
torna muito mais fraca, porém deixa de competir com o InfoNCE no ponto de
referência. O acordo de sinais nas entradas em que ambas as losses atuam passa
de 97,4--99,9% para 100%.

No espaço de parâmetros, o efeito é menor e depende do Jacobiano da rede. No
ramo corrente de ANT-IV-AR-D, por exemplo, o gradiente ANT equivale em média a
4,25% do gradiente InfoNCE no projetor e 1,08% no `ta_net`; seus cossenos são
+0,395 e +0,104. Na versão conectada, as razões são 6,82% e 6,49%, com
cossenos +0,068 e -0,024. Assim, o `detach` troca uma correção mais forte e
parcialmente conflitante por uma correção menor e mais alinhada.

## AR versus GR e IV versus FS

| Operador | Negativos válidos | Ativos corrente/KD | Ativos da mesma classe (corrente) | Referência da mesma classe (corrente) |
|---|---:|---:|---:|---:|
| IV-GR | 127 | 8,59% / 3,04% | 23,64% | 34,92% |
| IV-AR | 127 | 32,98% / 32,30% | 10,39% | 26,84% |
| FS-GR | 254 | 7,67% / 2,37% | 25,61% | 38,62% |
| FS-AR | 254 | 26,90% / 25,91% | 11,43% | 27,24% |

Somente 5,70% dos negativos válidos pertencem à mesma classe da âncora. GR,
contudo, concentra entre 23,6% e 25,6% de seu conjunto ativo nesses falsos
negativos, uma super-representação de aproximadamente quatro vezes. AR reduz
essa concentração para 10,4--11,4% e mantém a loss ativa para praticamente
todas as âncoras.

Ao longo das tarefas 2 e 3, a taxa ativa corrente de IV-GR caiu de 3,54% nas
épocas iniciais para 2,01% nas finais. Em IV-AR-D, ela aumentou de 27,98% para
35,67%. Isso sugere que GR rapidamente se torna uma restrição esparsa,
dominada por poucos negativos extremos e frequentemente semanticamente
positivos, enquanto AR continua distribuindo a regularização.

FS dobra os negativos de 127 para 254, mas todas as variantes FS ficaram
abaixo do Baseline em ACC na terceira tarefa. O escopo completo também aumenta
a exposição a falsos negativos e relações redundantes entre as duas views. A
evidência atual favorece IV para a formulação principal e FS como ablação.

## Magnitude das losses

A loss ANT bruta não deve ser comparada diretamente com a InfoNCE. Na
formulação atual, `logsumexp(ReLU(v))` possui o piso constante `log(N)`:

- IV: `log(127) = 4,844`; contribuição ponderada pelo beta 0,5 ≈ 2,422;
- FS: `log(254) = 5,537`; contribuição ponderada ≈ 2,769.

Isso explica por que a loss total média nas tarefas incrementais sobe de 4,94
no Baseline para 8,77--9,35 com ANT. A maior parte dessa diferença é constante
e tem gradiente zero. A contribuição ANT ponderada e ajustada pelo piso foi
somente 0,001--0,002 em GR e 0,017--0,024 em AR. Para gráficos e explicações,
usar `ant_loss_adjusted` e as normas de gradiente; a loss bruta é útil apenas
para reproduzir o objetivo realmente registrado.

Apesar da grande mudança no valor numérico da loss total, as atualizações
relativas dos parâmetros mudaram pouco. Em ANT-IV-AR-D contra o Baseline, a
atualização média do `ta_net` variou +0,86%, a do projetor -2,62% e a do
classificador -0,02%. Isso reforça que ANT atua principalmente na direção e na
seleção das relações, não como uma simples amplificação global do passo.

## Geometria do InfoNCE ao final do treinamento

Nas épocas finais das tarefas 2 e 3, ANT-IV-AR-D apresentou, comparado ao
Baseline:

- ramo corrente: InfoNCE 1,801 contra 1,822; probabilidade positiva 0,1955
  contra 0,1918; negativos acima do positivo 7,98% contra 8,34%;
- ramo KD: InfoNCE 2,173 contra 2,208; probabilidade positiva 0,1453 contra
  0,1409; negativos acima do positivo 15,62% contra 16,65%.

FS-GR conectado obteve valores contrastivos ligeiramente melhores em alguns
itens, mas teve ACC final inferior. Portanto, melhorar o surrogate contrastivo
não garante sozinho melhor classificação incremental; retenção, falsos
negativos e a interação com classificação/KD continuam relevantes.

Nas tarefas incrementais, o peso externo do ramo corrente cai de 0,5 em T2
para aproximadamente 0,333 em T3, enquanto o ramo KD cresce de 1,0 para
aproximadamente 1,333. Consequentemente, o comportamento da ANT no KD se torna
progressivamente mais importante e deve aparecer separadamente nas figuras.

## Conclusões e próximos testes

1. O `detach` corrige uma direção indesejada e mensurável: a atração do
   negativo usado como referência.
2. IV-AR-D é a hipótese prioritária: melhor ACC/NME neste estudo, regularização
   distribuída, menor viés para falsos negativos e gradiente mais alinhado.
3. IV-AR conectado continua relevante, pois teve a melhor retenção média;
   `detach` parece favorecer plasticidade mais que estabilidade.
4. FS e GR não devem ser descartados por uma seed, mas os dados explicam por
   que podem falhar: concentração em poucos falsos negativos e/ou duplicação
   do espaço de relações.
5. A confirmação deve usar o protocolo completo e múltiplas seeds. As
   comparações prioritárias são Baseline, IV-AR, IV-AR-D e, como controle do
   escopo, FS-AR-D.
6. Para a apresentação, as figuras mais informativas são: direção do gradiente
   na referência com/sem `detach`; norma ANT/InfoNCE; taxa ativa AR/GR ao longo
   das épocas; enriquecimento de falsos negativos; ACC/NME e estabilidade
   versus plasticidade.
