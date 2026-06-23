# Análise: ANT Simétrico vs Implementação Atual

## 1. Situação atual da loss

Hoje a sua loss **não usa todos os blocos simetricamente em tudo**. Ela mistura duas escolhas diferentes:

1. **InfoNCE**: usa a matriz inteira $2B \times 2B$.
2. **ANT**: usa apenas o bloco superior esquerdo (`Q1 × Q1`, ou seja, `view1` vs `view1`).

O ponto central está neste trecho do código:

```python
pos_start = cos_sim.shape[0] // 2
cos_sim_q1 = cos_sim[:pos_start, :pos_start]
```

Todo o cálculo do ANT é feito exclusivamente sobre esse bloco.

Já a InfoNCE usa a matriz completa:

```python
pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
```

---

## 2. Estrutura da matriz de similaridade

Após concatenar as duas views:

* Total de embeddings: $2B = 256$.
* Matriz: $S \in \mathbb{R}^{256 \times 256}$.

Ela pode ser decomposta em blocos:

$$
S =
\begin{bmatrix}
S_{11} & S_{12} \\
S_{21} & S_{22}
\end{bmatrix}
$$

Onde:

* $S_{11}$: intra-view, `view1` vs `view1`.
* $S_{22}$: intra-view, `view2` vs `view2`.
* $S_{12}$ e $S_{21}$: cross-view.

---

## 3. Formulação atual da loss

### 3.1 InfoNCE (global)

Para cada linha $r$:

$$
L_{\text{InfoNCE}} =
-\operatorname{sim}(r, r^+) + \log \sum_{c} \exp(\operatorname{sim}(r, c))
$$

* Usa toda a matriz.
* Um positivo por linha.
* Todos os outros são negativos.

---

### 3.2 ANT (atual)

Usa apenas:

$$
S_{11}
$$

Com:

$$
\operatorname{ANT} =
\mathbb{E}_{i}
\left[
\log \sum_{j}
\exp\left(
\operatorname{ReLU}(S_{11}[i,j] - m_i + \gamma)
\right)
\right]
$$

Onde:

* $m_i$: máximo global ou por linha.
* $\gamma$: margem.

---

## 4. Versões simétricas da loss

### 4.1 Opção A: ANT simétrico intra-view

Usa ambos os blocos intra-view:

$$
L_{\text{ANT-intra-sym}} =
\frac{1}{2}
\left(
L_{\text{ANT}}(S_{11}) + L_{\text{ANT}}(S_{22})
\right)
$$

#### Implementação conceitual:

```python
B = cos_sim.shape[0] // 2
q1 = cos_sim[:B, :B]
q2 = cos_sim[B:, B:]

mask = torch.eye(B, dtype=bool, device=device)

q1 = q1.masked_fill(mask, 0.0)
q2 = q2.masked_fill(mask, 0.0)

if ant_max_global:
    q1_max = q1[~mask].max()
    q2_max = q2[~mask].max()
    mq1 = F.relu(q1 - q1_max + ant_margin)
    mq2 = F.relu(q2 - q2_max + ant_margin)
else:
    q1_max = q1.max(dim=-1, keepdim=True).values
    q2_max = q2.max(dim=-1, keepdim=True).values
    mq1 = F.relu(q1 - q1_max + ant_margin)
    mq2 = F.relu(q2 - q2_max + ant_margin)

ant_loss = 0.5 * (
    torch.logsumexp(mq1, dim=-1).mean() +
    torch.logsumexp(mq2, dim=-1).mean()
)
```

---

### 4.2 Opção B: ANT na matriz completa (full symmetric)

Usa todos os negativos:

$$
L_{\text{ANT-full}} =
\frac{1}{2B} \sum_{r=1}^{2B}
\log \sum_{c \in N(r)}
\exp\left(
\operatorname{ReLU}(S[r,c] - m_r + \gamma)
\right)
$$

#### Implementação conceitual:

```python
N = cos_sim.shape[0]
self_mask = torch.eye(N, dtype=torch.bool, device=device)
pos_mask = self_mask.roll(shifts=N // 2, dims=0)

neg_mask = ~(self_mask | pos_mask)

neg_scores = cos_sim.masked_fill(~neg_mask, -float("inf"))

if ant_max_global:
    m = neg_scores[neg_mask].max()
    shifted = F.relu(neg_scores - m + ant_margin)
else:
    m = neg_scores.max(dim=-1, keepdim=True).values
    shifted = F.relu(neg_scores - m + ant_margin)

ant_loss = torch.logsumexp(
    shifted.masked_fill(~neg_mask, -float("inf")),
    dim=-1
).mean()
```

---

## 5. Por que isso muda os resultados

### 5.1 Mudança na distribuição de negativos

Atualmente o ANT vê apenas $S_{11}$, ignorando:

* $S_{22}$.
* Cross-view.

Isso altera:

* média dos negativos;
* máximo;
* cauda da distribuição.

---

### 5.2 Eliminação de assimetria artificial

As duas views são equivalentes, mas o ANT usa apenas uma metade.

Consequência:

* dependência da ordem do batch;
* perda de invariância.

---

### 5.3 Mudança no hardest negative

Com ANT global:

* hoje: máximo em $S_{11}$;
* simétrico: máximo em todos os blocos.

Isso impacta diretamente:

* margem efetiva;
* violação (%).

---

### 5.4 Mudança no gradiente

Mais pares passam a contribuir:

* maior cobertura do espaço;
* possível estabilização;
* ou maior penalização, dependendo do cenário.

---

## 6. Expectativas experimentais

### ANT intra-view simétrico

* comportamento mais estável;
* menor viés de batch;
* impacto moderado.

---

### ANT full-matrix

* mais negativos efetivos;
* máximos maiores;
* maior dificuldade da margem;
* possível mudança forte na violação.

---

## 7. Descompasso atual

Hoje:

* InfoNCE → global e simétrica.
* ANT → parcial e assimétrica.

Isso pode gerar inconsistência na geometria da loss.

---

## 8. Plano experimental sugerido

1. Baseline atual.
2. ANT intra-view simétrico.
3. ANT full-matrix.

Permite isolar:

* efeito da assimetria;
* efeito da restrição intra-view.

---

## 9. Conclusão

Ao usar todos os blocos simetricamente, o ANT deixa de depender de um subconjunto arbitrário dos negativos e passa a refletir melhor a geometria completa do batch contrastivo. Isso altera:

* máximos de referência;
* violações de margem;
* gradientes;
* estabilidade estatística.

A versão intra-view simétrica mantém proximidade com o método atual. A versão full-matrix é a mais consistente com a InfoNCE.
