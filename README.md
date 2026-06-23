# TagFex CVPR 2025 — Extensão de Investigação

**Paper**: Task-Agnostic Guided Feature Expansion for Class-Incremental Learning (CVPR 2025)  
**ArXiv**: https://arxiv.org/abs/2503.00823  
**Repositório original**: https://github.com/bwnzheng/TagFex_CVPR2025

Este repositório estende o framework TagFex original com três contribuições investigadas:
**ANT Loss** (Adaptive Negative Threshold), **AvgK Teacher Averaging** e **SBS Speed-Based Sampling**.

---

## Índice

1. [Ambiente de Execução](#ambiente-de-execução)
2. [Estado do Projeto](#estado-do-projeto)
3. [Melhores Resultados](#melhores-resultados)
4. [Experimentos Realizados](#experimentos-realizados)
5. [Contribuições Investigadas](#contribuições-investigadas)
6. [Instalação e Uso](#instalação-e-uso)
7. [Fila de Experimentos](#fila-de-experimentos)
8. [Estrutura do Projeto](#estrutura-do-projeto)
9. [Fundamentos Teóricos](#fundamentos-teóricos)
10. [Implementações — Referência Rápida](#implementações--referência-rápida)
11. [Análise e Relatório](#análise-e-relatório)
12. [Documentação Adicional](#documentação-adicional)
13. [Referências](#referências)

---

## Ambiente de Execução

| Campo | Valor |
|-------|-------|
| **Máquina** | xavier |
| **GPU** | NVIDIA GeForce RTX 3080 Ti · 12 GB · 1 GPU |
| **OS / Python** | Ubuntu · Python 3.11.2 |
| **Gestor de pacotes** | `uv` (`.venv/` na raiz do projeto) |
| **Activar ambiente** | `source .venv/bin/activate` |

```bash
source .venv/bin/activate
nvidia-smi                      # verificar GPU
```

---

## Estado do Projeto

### 🔄 EM CURSO — Junho 2026

**112 runs completos**; **59 experimentos de ablação multi-seed em fila** para máquina externa.

| Fase | Estado | Runs |
|------|--------|------|
| Baseline (β=0, aGlobal nGlobal) | ✅ Concluída | 3 seeds × 4 cenários |
| ANT Local (β=0.5, aLocal nLocal) | ✅ Concluída | 3 seeds × 4 cenários |
| ANT SymFull (β=0.5, aSymFull nLocal) | ✅ Concluída | 3 seeds × 4 cenários |
| AvgK Ablation (K=3,5) | ✅ Concluída | 3 seeds × 4 cenários |
| AvgK Ablation (K=10) + SBS | 🔄 Seed 1993 completo | 4 cenários × 2 variantes; multi-seed em fila |
| Ablação âncora (aGlobal/aLocal × nGlobal/nLocal) | 🔄 Seed 1993 completo | TIN-100-20 em fila local; restantes em fila externa |

**Experimentos em fila** (nova máquina): `configs/queue_ablation_missing.txt` — 59 runs  
**TIN-100-20 em fila local**: `configs/queue_tin100_ablation.txt` — 9 runs  
**Relatório completo**: `results_report.md` (gerado por `generate_md_report.py`)

---

## Melhores Resultados

Médias e desvios padrão amostrais (ddof=1) sobre 3 seeds. Δ relativo ao **mean baseline** (β=0, aGlobal nGlobal, 3 seeds).

### CIFAR-100 10×10

Baseline (β=0 aGlobal nGlobal): **78.76 ± 0.77%** avg Acc@1

| Configuração | Seeds | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 1993–1995 | 78.76 ± 0.77 | 75.54 ± 0.62 | 13.72 | — |
| β=0 aLocal nLocal | 1993–1995 | 78.93 ± 0.73 | 75.62 ± 0.63 | 13.10 | +0.17 |
| **β=0.5 aLocal nLocal** ⭐ | 1993–1995 | **79.15 ± 0.51** | 75.69 ± 0.30 | 12.97 | **+0.39** |
| β=0.5 aSymFull nLocal | 1993–1995 | 79.04 ± 0.69 | **75.74 ± 0.84** | 13.07 | +0.28 |
| β=0.5 aLocal nLocal avgK5 | 1993–1995 | 78.96 ± 0.09 | 75.44 ± 0.23 | 13.05 | +0.20 |
| β=0.5 aSymFull nLocal avgK5 | 1993–1995 | 79.07 ± 0.09 | 75.63 ± 0.10 | 13.33 | +0.31 |

### CIFAR-100 50+10×5

Baseline: **76.99 ± 0.58%**

| Configuração | Seeds | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 1993–1995 | 76.99 ± 0.58 | 76.52 ± 0.42 | 9.31 | — |
| **β=0.5 aSymFull nLocal** ⭐ | 1993–1995 | **77.18 ± 0.27** | **76.70 ± 0.40** | 9.43 | **+0.19** |
| β=0.5 aLocal nLocal | 1993–1995 | 77.01 ± 0.42 | 76.62 ± 0.27 | 9.66 | +0.02 |
| β=0.5 aSymFull nLocal avgK5 | 1993–1995 | 77.17 ± 0.20 | 76.54 ± 0.26 | 9.42 | +0.18 |

### Tiny-ImageNet 100+20×5

Baseline: **59.66 ± 0.58%**

| Configuração | Seeds | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 1993–1995 | 59.66 ± 0.58 | 59.60 ± 0.29 | 9.93 | — |
| **β=0.5 aSymFull nLocal** ⭐ | 1993–1995 | **61.39 ± 0.33** | **60.33 ± 0.14** | 10.79 | **+1.73** |
| β=0.5 aLocal nLocal | 1993–1995 | 61.07 ± 0.45 | 60.14 ± 0.14 | 10.77 | +1.41 |
| β=0.5 aLocal nLocal avgK5 | 1993–1995 | 61.28 ± 0.21 | 60.25 ± 0.11 | 10.37 | +1.62 |
| β=0.5 aSymFull nLocal avgK5 | 1993–1995 | 61.07 ± 0.22 | 60.10 ± 0.15 | 10.65 | +1.41 |

### Tiny-ImageNet 20×10

Baseline: **60.92 ± 0.77%**

| Configuração | Seeds | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 1993–1995 | 60.92 ± 0.77 | 57.21 ± 0.64 | 16.05 | — |
| β=0 aLocal nLocal | 1993–1995 | 61.18 ± 0.59 | 57.52 ± 0.38 | 15.93 | +0.26 |
| **β=0.5 aLocal nLocal** ⭐ | 1993–1995 | **61.18 ± 0.48** | 57.56 ± 0.53 | 16.77 | **+0.26** |
| β=0.5 aLocal nLocal avgK5 | 1993–1995 | 61.40 ± 0.49 | 57.68 ± 0.49 | 16.58 | +0.49 |
| β=0.5 aSymFull nLocal avgK5 | 1993–1995 | 61.30 ± 0.40 | **57.74 ± 0.56** | 16.09 | +0.38 |

> **Nota**: O dataset mais difícil (TIN-100-20) é onde ANT demonstra o maior benefício (+1.73 pp). Em CIFAR-100 os ganhos são modestos (+0.2–0.4 pp) mas consistentes entre seeds.

---

## Experimentos Realizados

### Visão Geral

**112 runs completos** distribuídos por 4 datasets, 3 seeds principais (1993/1994/1995) e ablações adicionais (seed 1993):

| Dataset | Runs | Variantes principais (3 seeds) | Ablações (seed 1993) |
|---------|------|-------------------------------|----------------------|
| CIFAR-100 10-10 | 29 | β=0 aGlobal, β=0 aLocal, β=0.5 aLocal, β=0.5 aSymFull | avgK3/5/10, SBS, aGlobal nLocal, aLocal nGlobal |
| CIFAR-100 50-10 | 28 | β=0 aGlobal, β=0 aLocal, β=0.5 aLocal, β=0.5 aSymFull | avgK3/5/10, SBS, aLocal nGlobal |
| TIN 100-20 | 26 | β=0 aGlobal, β=0 aLocal, β=0.5 aLocal, β=0.5 aSymFull | avgK3/5/10, SBS |
| TIN 20-20 | 29 | β=0 aGlobal, β=0 aLocal, β=0.5 aLocal, β=0.5 aSymFull | avgK3/5/10, SBS, aGlobal nLocal, aLocal nGlobal |

### CIFAR-100 10-10 — resultados completos

| Configuração | n | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 3 | 78.76 ± 0.77 | 75.54 ± 0.62 | 13.72 | — |
| β=0 aLocal nLocal | 3 | 78.93 ± 0.73 | 75.62 ± 0.63 | 13.10 | +0.17 |
| β=0.5 aLocal nLocal **⭐** | 3 | **79.15 ± 0.51** | 75.69 ± 0.30 | 12.97 | **+0.39** |
| β=0.5 aLocal nLocal avgK3 | 3 | 78.98 ± 0.10 | 75.74 ± 0.25 | 13.12 | +0.22 |
| β=0.5 aLocal nLocal avgK5 | 3 | 78.96 ± 0.09 | 75.44 ± 0.23 | 13.05 | +0.20 |
| β=0.5 aLocal nLocal avgK10 | 1 | 78.99 | 75.55 | 13.19 | +0.23 |
| β=0.5 aLocal nLocal SBS | 1 | 79.19 | 75.50 | 13.00 | +0.43 |
| β=0.5 aSymFull nLocal | 3 | 79.04 ± 0.69 | **75.74 ± 0.84** | 13.07 | +0.28 |
| β=0.5 aSymFull nLocal avgK3 | 1 | 78.95 | 75.65 | 13.64 | +0.19 |
| β=0.5 aSymFull nLocal avgK5 | 3 | 79.07 ± 0.09 | 75.63 ± 0.10 | 13.33 | +0.31 |
| β=0.5 aSymFull nLocal avgK10 | 1 | 78.85 | 75.52 | 13.27 | +0.09 |
| β=0.5 aSymFull nLocal SBS | 1 | 78.86 | 75.25 | **12.83** | +0.10 |
| β=0.5 aGlobal nGlobal | 1 | 78.25 | 74.93 | 12.83 | −0.51 |
| β=0.5 aGlobal nLocal | 1 | 78.08 | 74.75 | 13.77 | −0.68 |
| β=0.5 aLocal nGlobal | 1 | 78.05 | 74.84 | 13.81 | −0.71 |

### CIFAR-100 50-10 — resultados completos

| Configuração | n | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 3 | 76.99 ± 0.58 | 76.52 ± 0.42 | 9.31 | — |
| β=0 aLocal nLocal | 3 | 76.94 ± 0.37 | 76.61 ± 0.30 | 9.34 | −0.05 |
| β=0.5 aLocal nLocal | 3 | 77.01 ± 0.42 | 76.62 ± 0.27 | 9.66 | +0.02 |
| β=0.5 aLocal nLocal avgK3 | 3 | 76.65 ± 0.58 | 76.34 ± 0.28 | 9.27 | −0.34 |
| β=0.5 aLocal nLocal avgK5 | 3 | 77.04 ± 0.06 | 76.60 ± 0.12 | 9.10 | +0.05 |
| β=0.5 aLocal nLocal avgK10 | 1 | 77.25 | 76.38 | 9.45 | +0.26 |
| β=0.5 aLocal nLocal SBS | 1 | 76.49 | 76.15 | 9.88 | −0.50 |
| β=0.5 aSymFull nLocal **⭐** | 3 | **77.18 ± 0.27** | **76.70 ± 0.40** | 9.43 | **+0.19** |
| β=0.5 aSymFull nLocal avgK3 | 1 | 77.04 | 76.29 | 9.41 | +0.05 |
| β=0.5 aSymFull nLocal avgK5 | 3 | 77.17 ± 0.20 | 76.54 ± 0.26 | 9.42 | +0.18 |
| β=0.5 aSymFull nLocal avgK10 | 1 | 76.88 | 76.20 | 9.17 | −0.11 |
| β=0.5 aSymFull nLocal SBS | 1 | 76.87 | 76.26 | 9.48 | −0.12 |
| β=0.5 aGlobal nGlobal | 1 | 76.71 | 76.37 | 9.80 | −0.28 |
| β=0.5 aLocal nGlobal | 1 | 76.44 | 76.23 | 9.50 | −0.55 |

### Tiny-ImageNet 100-20 — resultados completos

| Configuração | n | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 3 | 59.66 ± 0.58 | 59.60 ± 0.29 | 9.93 | — |
| β=0 aLocal nLocal | 3 | 59.69 ± 0.48 | 59.66 ± 0.22 | 10.03 | +0.03 |
| β=0.5 aLocal nLocal | 3 | 61.07 ± 0.45 | 60.14 ± 0.14 | 10.77 | +1.41 |
| β=0.5 aLocal nLocal avgK3 | 3 | 61.18 ± 0.28 | 60.21 ± 0.28 | 10.57 | +1.52 |
| β=0.5 aLocal nLocal avgK5 | 3 | 61.28 ± 0.21 | 60.25 ± 0.11 | 10.37 | +1.62 |
| β=0.5 aLocal nLocal avgK10 | 1 | 61.08 | 59.90 | 10.84 | +1.42 |
| β=0.5 aLocal nLocal SBS | 1 | 61.01 | 59.98 | 11.04 | +1.35 |
| β=0.5 aSymFull nLocal **⭐** | 3 | **61.39 ± 0.33** | **60.33 ± 0.14** | 10.79 | **+1.73** |
| β=0.5 aSymFull nLocal avgK3 | 1 | 60.84 | 59.86 | 10.67 | +1.18 |
| β=0.5 aSymFull nLocal avgK5 | 3 | 61.07 ± 0.22 | 60.10 ± 0.15 | 10.65 | +1.41 |
| β=0.5 aSymFull nLocal avgK10 | 1 | 61.09 | 60.31 | 10.25 | +1.43 |
| β=0.5 aSymFull nLocal SBS | 1 | 61.11 | 60.30 | 10.46 | +1.45 |

> TIN-100-20 requer `grad_clip_norm: 5.0` nas variantes β=0.5 — o `trans_cls_loss` explode sem clipping em cenários com 100 classes base.

### Tiny-ImageNet 20-20 — resultados completos

| Configuração | n | Avg Acc@1 | Avg NME@1 | Fgt | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | 3 | 60.92 ± 0.77 | 57.21 ± 0.64 | 16.05 | — |
| β=0 aLocal nLocal | 3 | 61.18 ± 0.59 | 57.52 ± 0.38 | 15.93 | +0.26 |
| β=0.5 aLocal nLocal **⭐** | 3 | **61.18 ± 0.48** | 57.56 ± 0.53 | 16.77 | **+0.26** |
| β=0.5 aLocal nLocal avgK3 | 3 | 61.07 ± 0.54 | 57.29 ± 0.26 | 16.60 | +0.15 |
| β=0.5 aLocal nLocal avgK5 | 3 | 61.40 ± 0.49 | 57.68 ± 0.49 | 16.58 | +0.49 |
| β=0.5 aLocal nLocal avgK10 | 1 | 61.45 | 57.84 | 16.33 | +0.53 |
| β=0.5 aLocal nLocal SBS | 1 | 61.14 | 57.42 | 16.88 | +0.22 |
| β=0.5 aSymFull nLocal | 3 | 60.89 ± 0.58 | 57.29 ± 0.60 | 16.13 | −0.03 |
| β=0.5 aSymFull nLocal avgK3 | 1 | 61.29 | 57.67 | 16.93 | +0.37 |
| β=0.5 aSymFull nLocal avgK5 | 3 | 61.30 ± 0.40 | **57.74 ± 0.56** | 16.09 | +0.38 |
| β=0.5 aSymFull nLocal avgK10 | 1 | 61.38 | 57.91 | 16.23 | +0.46 |
| β=0.5 aSymFull nLocal SBS | 1 | 61.04 | 57.55 | 17.27 | +0.12 |
| β=0.5 aGlobal nGlobal | 1 | 60.17 | 57.02 | 15.91 | −0.75 |
| β=0.5 aGlobal nLocal | 1 | 60.42 | 56.91 | 15.54 | −0.50 |
| β=0.5 aLocal nGlobal | 1 | 60.49 | 57.16 | 16.41 | −0.43 |

---

## Contribuições Investigadas

### 1. ANT Loss — Adaptive Negative Threshold

Formulações completas e ablação em [Fundamentos Teóricos → ANT Loss](#ant-loss--adaptive-negative-threshold).

**Hiperparâmetros óptimos** (validados com 3 seeds):
- `ant_beta: 0.5`, `ant_margin: 0.5`, `ant_max_global: false`, `infonce_max_global: false`

**Conclusão**: ✅ **Benefício claro em TIN-100-20 (+1.41–1.73 pp)**; ganhos modestos mas consistentes em CIFAR-100 (+0.28–0.39 pp). A configuração `aLocal nLocal` é sistematicamente superior a `aGlobal nGlobal` em todos os cenários.

---

### 2. AvgK Teacher Averaging

**Ideia**: construir o teacher para a próxima task como a **média dos pesos dos últimos K epochs** (Stochastic Weight Averaging por task), em vez do snapshot final. Produz um teacher mais estável e generalizado.

**Escopo**: apenas a **TA branch** (`ta_net`) e o **projection head** (`projector`) são médios. A TS branch (task-specific) e o classifier não são incluídos — mantêm o estado do último epoch.

```python
# Dentro de _append_ckpt() — chamado nos últimos K epochs de cada task:
ta_sd   = {k: v.clone() for k, v in net.ta_net.state_dict().items()}
proj_sd = {k: v.clone() for k, v in net.projector.state_dict().items()}
self._ckpt_buf_ta.append(ta_sd)        # deque(maxlen=K) — janela deslizante
self._ckpt_buf_proj.append(proj_sd)

# No início da task seguinte, em vez do snapshot final:
last_ta_net = avg_state_dicts(list(self._ckpt_buf_ta))         # média elemento a elemento
last_projector = avg_state_dicts(list(self._ckpt_buf_proj))
```

Recolha: para `num_epochs=200` e `K=5`, os checkpoints são acumulados nos epochs 196–200 (condição `epoch >= num_epochs - K`).

> Cada K requer re-run completo — o teacher afecta todas as tasks seguintes (efeito em cascata). Não é um pós-processamento.

**Resultados** (média de 3 seeds; Δ relativo à variante sem avgK com 3 seeds):

| Dataset | aLocal K=3 | aLocal K=5 | aLocal K=10 | aSymFull K=3 | aSymFull K=5 | aSymFull K=10 |
|---|---|---|---|---|---|---|
| C100 10-10 | 78.98 (−0.17) | 78.96 (−0.19) | 78.99¹ | 78.95¹ | 79.07 (+0.03) | 78.85¹ |
| C100 50-10 | 76.65 (−0.36) | 77.04 (+0.03) | 77.25¹ | 77.04¹ | 77.17 (−0.01) | 76.88¹ |
| TIN 100-20 | 61.18 (+0.11) | 61.28 (+0.21) | 61.08¹ | 60.84¹ | 61.07 (−0.32) | 61.09¹ |
| TIN 20-20 | 61.07 (−0.11) | 61.40 (+0.22) | 61.45¹ | 61.29¹ | 61.30 (+0.41) | 61.38¹ |

¹ Apenas seed 1993.

**Conclusão**: ⚠️ avgK K=5 mostra ganhos moderados em TIN-20-20 (+0.22–0.41 pp) mas dentro da variância entre seeds para CIFAR-100. K=10 é inconsistente. avgK não traz benefício estatisticamente robusto na maioria dos cenários.

**Config**:
```yaml
avg_last_k: 5   # 0 = desactivado (default)
```

---

### 3. SBS — Speed-Based Sampling

**Ideia**: antes do herding iCaRL, filtrar os candidatos a exemplares com base na velocidade de aprendizagem de cada amostra durante o treino. Remove as triviais (já generalizadas) e as ruidosas (difíceis de aprender), conservando o intervalo intermédio para herding.

**Mérica de velocidade de aprendizagem** por amostra:

$$\text{speed}_i = \frac{\text{n.º de vezes que a amostra } i \text{ foi classificada corretamente}}{\text{n.º de epochs em que apareceu no treino}}$$

**Filtro por banda de percentis**: ordena as amostras de novas classes por `speed`, depois:
- Remove os `s%` mais **lentos** (slow-learned → ruidosos / difíceis)
- Remove os `q%` mais **rápidos** (fast-learned → triviais / já generalizados)
- Herding corre apenas no subconjunto intermédio `[s%, (1−q)%]`

```python
# _sbs_keep_mask(speeds, q, s, min_keep)
order = np.argsort(speeds)                 # ascendente: index 0 = mais lento
drop = set(order[:n_drop_s])              # s% mais lentos
drop |= set(order[n - n_drop_q:])         # q% mais rápidos
mask = np.ones(n, dtype=bool)
mask[list(drop)] = False
```

Guarda de segurança: se o filtro deixasse menos de `min_keep` amostras, usa todos.

**Tracking**: implementado via `WithIndexDataset` wrapper — cada batch emite `(local_idx, aug1, aug2, label)`. Os arrays `_sbs_correct[idx]` e `_sbs_total[idx]` são atualizados após cada batch com base nas predições do modelo no mesmo forward pass.

**Resultados** (seed 1993 apenas; Q=0.20, S=0.20; Δ relativo à variante sem SBS com 3 seeds):

| Dataset | aLocal SBS | aSymFull SBS |
|---|---|---|
| C100 10-10 | 79.19 (+0.04) | 78.86 (−0.18) |
| C100 50-10 | 76.49 (−0.52) | 76.87 (−0.31) |
| TIN 100-20 | 61.01 (−0.06) | 61.11 (+0.04) |
| TIN 20-20 | 61.14 (−0.04) | 61.04 (+0.15) |

**Conclusão**: ⚠️ Resultados mistos com 1 seed — não conclusivos. SBS não mostrou benefício claro com Q=S=0.20 e mem=2000. O paper de referência reporta +0.53 pp com mem=4000; com menos memória o efeito pode ser diferente.

**Bugs corrigidos durante implementação**:
1. `WithIndexDataset.__getitems__`: necessário override explícito — PyTorch ≥2.0 `_MapDatasetFetcher` verifica `__getitems__` antes de `__getitem__`, contornando o prepend de `idx`.
2. `update_memory` herding loop: `actual_idx_without_removal` causava divergência de índices após `np.delete` no array shrinkante.

**Config**:
```yaml
sbs_q: 0.20   # drop 20% mais rápidos
sbs_s: 0.20   # drop 20% mais lentos
```

---

## Instalação e Uso

### Requisitos

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Datasets

```bash
# Tiny ImageNet (necessário fazer download)
python setup_tiny_imagenet.py   # → ~/data/datasets/tiny-imagenet-200/

# CIFAR-100 — download automático na primeira execução
```

### Executar um experimento

```bash
source .venv/bin/activate

python main.py train \
  --exp-configs configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_resnet18.yaml \
  --seed 1993
```

### Gerar relatório

```bash
python generate_md_report.py          # → results_report.md (markdown)
python generate_html_pdf_report.py    # → results_report.{html,pdf}
python generate_html_pdf_report.py --short  # → results_report_short.{html,pdf} (sem debug)
```

---

## Fila de Experimentos

### Scripts disponíveis

| Script | Ficheiro de fila | Estado |
|--------|-----------------|--------|
| `run_multiseed_queue.sh` | `queue_multiseed.txt` | ✅ Concluída |
| `run_sbs_queue.sh` | `queue_sbs.txt` | ✅ Concluída |
| `run_avgk_queue.sh` | `queue_avgk.txt` | ✅ Concluída |
| `run_ablation_missing_queue.sh` | `queue_ablation_missing.txt` | 🔄 59 runs em fila (máquina externa) |
| `run_exp_queue_restart.sh` | — | Legado |

### Formato do ficheiro de fila

```
# configs/queue_multiseed.txt
config_path|descrição|seed
configs/all_in_one/cifar100_10-10_antB0.5_nceA1_antM0.5_antSymmetricFull_avgK5_resnet18.yaml|C100-10-10 aSymFull avgK5|1994
```

### Lançar nova fila

```bash
screen -dmS nome_sessao bash run_sbs_queue.sh

# Monitorar
tail -f logs/auto_experiments/sbs_orchestrator.log
tail -f logs/auto_experiments/sbs_console/<exp_name>.log
```

### Comportamento do orquestrador

- `flock` para evitar corridas paralelas (lockfile `/tmp/tagfex_*.lock`)
- `is_done()` reconstrói o sufixo exato do directório e verifica `exp_gistlog.log`
- Aguarda GPU disponível via `auto_run_on_free_gpu.py` (`--threshold`)
- Gera `results_report.md` automático após cada run bem-sucedido

---

## Estrutura do Projeto

```
TagFex_CVPR2025/
├── main.py                          # Entry point: train / eval
├── requirements.txt
├── setup_tiny_imagenet.py           # Download Tiny ImageNet
├── generate_md_report.py            # Parsing de logs → results_report.md
├── generate_html_pdf_report.py      # results_report.{html,pdf} (suporta --short)
├── charts_results.py                # Gráficos de resultados comparativos
├── results_report.md                # Relatório gerado (gitignored)
├── auto_run_on_free_gpu.py          # Lança experimento quando GPU fica livre
├── run_multiseed_queue.sh           # Fila multiseed (wrapper)
├── run_sbs_queue.sh                 # Fila base (SBS/avgK/multiseed)
├── run_avgk_queue.sh                # Fila avgK ablation
├── run_ablation_missing_queue.sh    # Fila ablações em falta (59 runs)
├── run_exp_queue_restart.sh         # Fila com restart (legado)
├── trainddp.sh                      # Training multi-GPU (DDP)
│
├── figure/                          # Gráficos de resultados (PNG)
│   ├── cifar100_10x10.png
│   ├── cifar100_50-10x5.png
│   ├── tinyimagenet_100-20x5.png
│   └── tinyimagenet_20x10.png
│
├── configs/
│   ├── queue_multiseed.txt          # 24 entradas (3 variantes × 4 cenários × 2 seeds)
│   ├── queue_sbs.txt                # Fila SBS ablation
│   ├── queue_avgk.txt               # Fila avgK ablation
│   ├── queue_tin100_ablation.txt    # TIN-100-20 ablações âncora (9 runs, local)
│   ├── queue_ablation_missing.txt   # Ablações em falta (59 runs, máquina externa)
│   ├── ant_investigation/           # Configs de investigação de formulações ANT
│   └── all_in_one/                  # Configs completos prontos a usar
│       ├── cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_resnet18.yaml ⭐
│       ├── cifar100_10-10_*_avgK[3/5/10]_resnet18.yaml
│       ├── cifar100_10-10_*_sbsQS020_resnet18.yaml
│       ├── cifar100_50-10_* / tiny_imagenet_20-20_* / tiny_imagenet_100-20_*
│       └── tiny_imagenet_100-20_antB0.5_*  (com grad_clip_norm: 5.0)
│
├── methods/
│   └── tagfex/
│       ├── tagfex.py                # TagFex: train loop, ANT, AvgK, SBS, update_memory
│       └── tagfexnet.py             # TagFexNet: dual-branch + merge attention
│
├── modules/
│   ├── learner/memory.py            # HerdingIndicesLearner (iCaRL herding base)
│   ├── data/
│   │   ├── dataset.py               # WithIndexDataset, MultipleAugmentationDataset
│   │   ├── manager.py               # DataManager
│   │   └── augmentation.py          # Transforms por dataset
│   ├── networks/                    # TagFexNet sub-components
│   ├── backbones/resnet.py          # ResNet18 com stem adaptável
│   ├── evaluation.py
│   └── metrics.py
│
├── loggers/
│   ├── loguru.py                    # Logger (gistlog + stdlog + debuglog)
│   └── utils.py
│
├── logs/                            # Directórios de experimento (gitignored)
│   ├── exp_*/debug_exp_*            # 112+ runs completos
│   └── auto_experiments/            # Logs de orquestração (gitignored)
│
├── analysis/
│   ├── scripts/                     # Comparação e análise de experimentos
│   └── results/                     # Plots e CSVs de análises
│
└── docs/                            # Documentação técnica
    ├── RESULTS_AND_METRICS.md
    ├── DEBUGGING_GUIDE.md
    ├── LOGGING_SYSTEM.md
    ├── GPU_MEMORY_GUIDE.md
    ├── AUTO_GPU_LAUNCHER.md
    ├── ANT_simetria.md
    ├── CROSS_TASK_EMBEDDING_COLLISION.md
    └── ant_investigation_results.md
```

### Estrutura de um directório de experimento

```
logs/exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceLocal_avgK3_s1993/
├── exp_gistlog.log     # métricas por task + finais (avg_acc1, avg_nme1, acc1_curve, ...)
├── exp_stdlog0.log     # stdout completo do treino (loss por epoch, eval por 5 epochs)
└── exp_debug0.log      # loss components e ANT distance stats por batch
```

O sufixo do directório é construído por `_adjust_log_dir_with_loss_params()` a partir dos hiperparâmetros activos (`ant_beta`, `nce_alpha`, `ant_margin`, `ant_max_global`, `avg_last_k`, `sbs_q/s`, `seed`).

---

## Fundamentos Teóricos

### O Problema: Catastrophic Forgetting em CIL

**Class-Incremental Learning (CIL)**: um modelo aprende novas classes sequencialmente sem acesso aos dados antigos, mantendo apenas um buffer de memória pequeno (2000 exemplares neste trabalho). Inferência sem task ID.

Desafios principais:
1. **Feature Collision** — features de novas classes colidem com antigas no espaço de embedding
2. **Distribution Shift** — distribuição muda a cada task
3. **Memory Constraints** — buffer pequeno (2000 exemplares total)
4. **Task-Agnostic Inference** — sem task ID durante inferência

---

### TagFex Framework

#### Arquitectura Dual-Branch

```
         ┌──────────────────────────────┐
         │         Input Image          │
         └──────────┬───────────────────┘
                    │
       ┌────────────┴────────────┐
       │                          │
┌──────▼────────┐       ┌────────▼────────┐
│ Task-Agnostic │       │  Task-Specific  │
│  Branch (TA)  │       │   Branch (TS)   │
│  (Frozen)     │       │  (Trainable)    │
└──────┬────────┘       └────────┬────────┘
       │  f_ta                   │  f_ts
       └────────────┬────────────┘
                    │
           ┌────────▼────────┐
           │ Merge Attention  │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │   Classifier     │
           └─────────────────┘
```

**TA Branch**: treinada na task 0 com InfoNCE auto-supervisionado; **frozen** nas tasks incrementais → sem forgetting na TA.  
**TS Branch**: treinada supervisionada; expande para novas classes a cada task.  
**Merge Attention**: combina `f_ta` e `f_ts` adaptativamente com pesos aprendidos.

#### Loss Functions

Task 0 (treino inicial):
```
L = L_cls + λ_contrast · InfoNCE(f_ta)
```

Tasks incrementais (task > 1):
```
L = L_cls + L_aux + L_transfer + λ_contrast · InfoNCE_distill(f_ta^new, f_ta^old)
```

ANT substitui InfoNCE quando `ant_beta > 0`:
```
L_contrast = (1 − ant_beta) · InfoNCE + ant_beta · L_ANT
```

---

### InfoNCE Loss

Maximiza similaridade entre pares positivos (augmentações da mesma imagem) e minimiza com negativos (outras amostras do batch):

$$L_{\text{InfoNCE}}(z_i) = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_j \exp(\text{sim}(z_i, z_j) / \tau)}$$

**Limitação em CIL**: nunca atinge zero mesmo com embeddings perfeitamente descorrelacionados → gradientes desnecessários (Non-Essential Tuning) → mais forgetting.

---

### ANT Loss — Adaptive Negative Threshold

Foca apenas nos **hard negatives** (próximos do positivo). Fórmula base (formulação `logsumexp`, usada em todos os experimentos):

$$m_{ij} = \text{sim}(z_i, z_j) - (\text{ref}_i - \gamma)$$

$$L_{\text{ANT}} = \log \left[ \sum_j \exp(m_{ij}) \cdot \mathbb{1}(m_{ij} > 0) \right]$$

Onde $\text{ref}_i$ é a referência por âncora (máximo global ou local) e $\gamma$ é a margem.

- `m_ij > 0`: negativo dentro da margem → **hard negative** → incluído
- `m_ij ≤ 0`: negativo fácil → **ignorado**

#### Nomenclatura dos experimentos

Cada rótulo de experimento codifica **três dimensões independentes** do ANT:

| Parte | Param YAML | Significado |
|---|---|---|
| `a` | `ant_max_global` + `ant_symmetric_full` | Como se constroem os **negativos ANT** |
| `n` | `infonce_max_global` | Como se normaliza o **InfoNCE** |
| `avgK` | `avg_last_k` | Teacher averaging (K epochs) |
| `SBS` | `sbs_q`, `sbs_s` | Speed-Based Sampling |

#### Dimensão `a` — Negativos ANT

Controla **quais pares entram como negativos** e **qual a referência** do threshold.

| Rótulo | `ant_max_global` | `ant_symmetric_full` | Negativos usados | Referência |
|---|---|---|---|---|
| `aGlobal` | `true` | `false` | Intra-view: B×B excluindo diagonal | Máximo **global** do batch (escalar único) |
| `aLocal` | `false` | `false` | Intra-view: B×B excluindo diagonal | Máximo **por âncora** (vetor [B,1]) |
| `aSymFull` | `—` | `true` | Full symmetric: N×N excluindo self e positivo | Máximo **por âncora** (local) |

- **`aGlobal`**: todas as âncoras partilham a mesma referência. Âncoras com negativos fáceis são sub-penalizadas pela referência alta das âncoras difíceis.
- **`aLocal`**: cada âncora normaliza pelo seu próprio máximo. Gradiente proporcional ao contexto local de cada âncora. ✅ Melhor na maioria dos cenários.
- **`aSymFull`**: usa 2B−2 negativos por âncora (vs B−1 em `aLocal`), incluindo pares cross-view. Ligeiramente superior em TIN-100-20, mas não em CIFAR-100.

```python
# aGlobal / aLocal — bloco intra-view
ant_sim = cos_sim[:B, :B].masked_fill(eye_mask, -inf)      # [B, B]
ref_global = ant_sim.max()                                  # aGlobal
ref_local  = ant_sim.max(dim=-1, keepdim=True).values       # aLocal [B,1]

# aSymFull — matriz completa
ant_sim = cos_sim.masked_fill(self_or_pos_mask, -inf)       # [N, N]
ref_local = ant_sim.max(dim=-1, keepdim=True).values        # sempre local
```

#### Dimensão `n` — Normalização InfoNCE

Controla como o **denominador do InfoNCE** é normalizado (estabilização numérica).

| Rótulo | `infonce_max_global` | Comportamento |
|---|---|---|
| `nGlobal` | `true` | Subtrai o **máximo global** do batch antes do softmax |
| `nLocal` | `false` | Subtrai o **máximo por âncora** (log-sum-exp local, cada âncora independente) |

`nLocal` é consistentemente melhor em combinação com `aLocal`. Para activar, definir explicitamente `infonce_max_global: false` no YAML (quando `ant_beta > 0`, o código não herda automaticamente o valor de `ant_max_global`).

#### Ablação completa de formulação (seed 1993, C100 10-10, β=0.5)

| Rótulo | `ant_max_global` | `infonce_max_global` | `ant_symmetric_full` | Acc@1 | Δ |
|---|---|---|---|---|---|
| β=0 aGlobal nGlobal *(true baseline)* | — | `true` | `false` | 77.87 | — |
| β=0.5 **aGlobal nGlobal** | `true` | `true` | `false` | 78.25 | +0.38 |
| β=0.5 **aGlobal nLocal** | `true` | `false` | `false` | 78.08 | +0.21 |
| β=0.5 **aLocal nGlobal** | `false` | `true` | `false` | 78.05 | +0.18 |
| β=0.5 **aLocal nLocal** ⭐ | `false` | `false` | `false` | **78.57** | **+0.70** |
| β=0.5 **aSymFull nLocal** | — | `false` | `true` | 78.28 | +0.41 |

Local Anchor em ambas as dimensões (ANT + InfoNCE) é claramente o melhor.

#### Formulações ANT disponíveis no código

O código implementa 5 formulações (`ant_formulation`). Todos os experimentos usaram `logsumexp` (default).

| Formulação | Fórmula | Característica |
|---|---|---|
| `logsumexp` *(default)* | $\log(\sum \exp(\text{relu}(m_{ij})))$ | Count-floor = $\log(N_{\text{neg}})$; usado em todos os experimentos |
| `expm1` | $\log(1 + \sum(\exp(\text{relu}(m)) - 1))$ | Zero contribuição de não-violadores; sem floor |
| `softplus` | $\frac{1}{N}\sum \text{softplus}(m_{ij}/\tau)$ | Suave, sem ReLU duro; parâmetro `ant_tau` |
| `topk` | logsumexp nos top-k violadores | Foca só nos $k$ negativos mais difíceis; parâmetro `ant_topk` |
| `active_only` | $\frac{1}{\|\text{act}\|}\sum_{j:m>0} m_{ij}$ | Média directa das violações activas; sem floor |

---

## Implementações — Referência Rápida

### Chaves de config (YAML)

| Chave | Default | Descrição |
|-------|---------|-----------|
| `ant_beta` | 0.0 | Peso ANT loss (0 = desactivado, i.e. InfoNCE puro) |
| `ant_margin` | 0.1 | Margem de hard negatives γ (óptimo: 0.5) |
| `ant_max_global` | true | `false` = Local Anchor (`aLocal`); `true` = Global (`aGlobal`) |
| `infonce_max_global` | `true` quando `ant_beta>0`; senão ≡ `ant_max_global` | `false` = `nLocal`; `true` = `nGlobal` |
| `ant_symmetric_full` | false | `true` = `aSymFull` (full symmetric N×N) |
| `ant_formulation` | `logsumexp` | Formulação ANT: `logsumexp`, `expm1`, `softplus`, `topk`, `active_only` |
| `avg_last_k` | 0 | AvgK teacher averaging (0 = desactivado) |
| `sbs_q` | 0.0 | SBS: drop top q% amostras mais rápidas |
| `sbs_s` | 0.0 | SBS: drop bottom s% amostras mais lentas |
| `grad_clip_norm` | null | Gradient clipping (necessário em TIN-100-20 β=0.5) |
| `memory_size` | 2000 | Total de exemplares no buffer de memória |

### Ficheiros principais

| Ficheiro | Conteúdo |
|----------|----------|
| `methods/tagfex/tagfex.py` | `TagFex`: train loop, ANT, AvgK, SBS, `update_memory` |
| `methods/tagfex/tagfexnet.py` | `TagFexNet`: arquitectura dual-branch |
| `modules/learner/memory.py` | `HerdingIndicesLearner`: base iCaRL herding |
| `modules/data/dataset.py` | `WithIndexDataset` (SBS idx tracking), `MultipleAugmentationDataset` |
| `modules/data/manager.py` | `DataManager`: `get_dataset_by_class_ids` |
| `generate_md_report.py` | Parsing de `exp_gistlog.log`, tabelas mean±std, relatório markdown |
| `generate_html_pdf_report.py` | Converte relatório para HTML/PDF com tabela cross-dataset (rowspan) |

---

## Análise e Relatório

### Gerar relatório completo

```bash
source .venv/bin/activate
python generate_md_report.py          # → results_report.md (~18k linhas, 112 runs)
python generate_html_pdf_report.py    # → results_report.{html,pdf}
python generate_html_pdf_report.py --short  # → results_report_short.{html,pdf} (sem debug)
```

O relatório inclui:
- **Section 0**: visão geral cross-dataset com 3 sub-linhas por variante (Acc Δ / NME Δ / Fgt)
- **Section 1**: tabelas mean±std por variante (ddof=1), Δ vs baseline, curves por task
- **Section 2**: debug metrics (loss components, ANT distance stats) por experimento

### Verificar métricas individuais

```bash
python verify_all_metrics.py
python extract_final_metrics.py
```

### Visualização

```bash
# Loss components ao longo do treino
python plot_loss_components.py logs/<exp>/exp_debug0.log -t contrast

# Comparar dois experimentos
python analysis/scripts/compare_experiments.py \
  --baseline logs/<baseline>/exp_gistlog.log \
  --exp logs/<exp>/exp_gistlog.log
```

---

## Documentação Adicional

| Ficheiro | Conteúdo |
|----------|----------|
| [docs/RESULTS_AND_METRICS.md](docs/RESULTS_AND_METRICS.md) | Definição de métricas e formato dos logs |
| [docs/DEBUGGING_GUIDE.md](docs/DEBUGGING_GUIDE.md) | Debugging e análise de logs |
| [docs/LOGGING_SYSTEM.md](docs/LOGGING_SYSTEM.md) | Sistema de logging (gistlog, stdlog, debuglog) |
| [docs/GPU_MEMORY_GUIDE.md](docs/GPU_MEMORY_GUIDE.md) | Guia de memória GPU e configuração |
| [docs/ant_investigation_results.md](docs/ant_investigation_results.md) | Investigação detalhada de formulações ANT |
| [pace/README.md](pace/README.md) | Stack do professor — resultados de referência (81.52% C100 10-10) |

---

## Referências

### Paper

```bibtex
@inproceedings{tagfex2025,
  title={Task-Agnostic Guided Feature Expansion for Class-Incremental Learning},
  author={Zheng, Bingwen and others},
  booktitle={CVPR},
  year={2025}
}
```

**ArXiv**: https://arxiv.org/abs/2503.00823  
**Repositório original**: https://github.com/bwnzheng/TagFex_CVPR2025

### Referência SBS

Hacohen & Tuytelaars, "Speed-Based Sampling", ICML 2025  
Implementação de referência: `pace/models/tagfexcma_v12_sbs.py`

### Datasets

- **CIFAR-100**: 100 classes · splits 10×10 e 50+10×5
- **Tiny ImageNet**: 200 classes · splits 20×10 e 100+20×5

### Base

[PyCIL](https://github.com/G-U-N/PyCIL) — base do `pace/` do professor.

---

**Última atualização**: Junho 2026  
**Estado**: 🔄 112 experimentos completos · 59+9 ablações multi-seed em fila · Resultados em `results_report.md`
