# TagFex - Task-Agnostic Guided Feature Expansion for Class-Incremental Learning

**Paper**: CVPR 2025  
**ArXiv**: https://arxiv.org/abs/2503.00823

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Fundamentos Teóricos](#fundamentos-teóricos)
3. [Estado Atual do Projeto](#estado-atual-do-projeto)
4. [Melhor Configuração Encontrada](#melhor-configuração-encontrada)
5. [Instalação e Uso](#instalação-e-uso)
6. [Experimentos Realizados](#experimentos-realizados)
7. [Descobertas Principais](#descobertas-principais)
8. [ANT Loss: Conceito e Implementação](#ant-loss-conceito-e-implementação)
9. [Local vs Global Anchor](#local-vs-global-anchor)
10. [Estrutura do Projeto](#estrutura-do-projeto)
11. [Análise e Debugging](#análise-e-debugging)
12. [Documentação Adicional](#documentação-adicional)
13. [Referências](#referências)

---

## 🖥️ Ambientes de Execução

| Máquina | GPU | VRAM | Responsabilidade |
|---------|-----|------|------------------|
| **quati** (ativo) | NVIDIA RTX 4090 | 24 GB · 1 GPU | Tiny ImageNet 20-20 + CIFAR-100 (todas as seeds) |
| **fera** (indisponível) | 2× GPU ~49 GB | ~98 GB · 2 GPUs | ImageNet-100 (quando retornar) |

O script de fila [`run_experiments_queue.sh`](run_experiments_queue.sh) detecta a máquina automaticamente pelo hostname (ou usa `MACHINE="auto"` no topo do script para sobrescrever).

---

## 🎯 Visão Geral

**TagFex** é um framework para Class-Incremental Learning que resolve o problema de **feature collision** através de:

- 🎯 Captura contínua de características task-agnostic
- 🔄 Modelo não supervisionado separado
- 📈 Superioridade sobre métodos expansion-based que treinam do zero

![motivation](papers/tagfex/assets/motivation.svg)
![overview](papers/tagfex/assets/overview.svg)

### Contribuições Principais

1. **TagFex Framework Original**: Feature expansion task-agnostic
2. **ANT Loss (Adaptive Negative Threshold)**: Melhora aprendizado contrastivo focando em hard negatives
3. **Local Anchor Normalization**: Normalização adaptativa por âncora

---

## � Fundamentos Teóricos

### O Problema: Catastrophic Forgetting em Class-Incremental Learning

**Catastrophic Forgetting** ocorre quando um modelo de aprendizado profundo, ao aprender novas tarefas (classes), sobrescreve o conhecimento previamente adquirido. Em Class-Incremental Learning (CIL), este é o desafio central: como adicionar novas classes sem esquecer as antigas?

**Desafios específicos de CIL**:
1. **Feature Collision**: Features de novas classes podem colidir com features de classes antigas
2. **Distribution Shift**: Distribuição de dados muda a cada task
3. **Memory Constraints**: Impossível armazenar todos os dados antigos
4. **Task-Agnostic Setting**: Sem acesso a task labels durante inferência

---

### TagFex Framework: Task-Agnostic Guided Feature Expansion

#### Arquitetura Geral

TagFex utiliza uma arquitetura de **dual-branch**:

```
                    ┌─────────────────┐
                    │   Input Image   │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                  │
    ┌───────▼────────┐              ┌─────────▼────────┐
    │  Task-Agnostic │              │  Task-Specific   │
    │     Branch     │              │     Branch       │
    │   (Frozen)     │              │   (Trainable)    │
    └───────┬────────┘              └─────────┬────────┘
            │                                  │
            │  Features f_ta                   │  Features f_ts
            │                                  │
            └────────────────┬─────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Merge Attention │
                    │   (Adaptive)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Classifier    │
                    └─────────────────┘
```

#### Componentes-Chave

**1. Task-Agnostic Branch** (`f_ta`):
- Treinado de forma **auto-supervisionada** usando InfoNCE
- Captura features **task-independent**
- **Frozen** durante treinamento de novas tasks
- Evita catastrophic forgetting por não ser atualizado

**2. Task-Specific Branch** (`f_ts`):
- Treinado de forma **supervisionada** com labels
- Expande dinamicamente para novas classes
- Especializa-se em features discriminativas
- Atualizado a cada nova task

**3. Merge Attention Module**:
- Combina adaptivamente `f_ta` e `f_ts`
- Aprende pesos de atenção: `α_ta`, `α_ts`
- Features finais: `f_merged = α_ta · f_ta + α_ts · f_ts`

#### Loss Functions do TagFex

**Loss de Classificação** (Task-Specific):
```
L_cls = CrossEntropy(f_ts, labels)
```

**Loss Contrastiva** (Task-Agnostic):
```
L_contrast = InfoNCE(f_ta(x), f_ta(x'))
```
Onde `x'` é uma augmentation de `x`.

**Loss de Distillation** (Knowledge Retention):
```
L_distill = InfoNCE(f_ta^new(x), f_ta^old(x))
```
Mantém consistência com modelo anterior.

**Loss Total**:
```
L_total = L_cls + λ_contrast · L_contrast + λ_distill · L_distill
```

---

### InfoNCE Loss: Contrastive Learning

#### Formulação Original

InfoNCE (Information Noise Contrastive Estimation) é uma loss contrastiva que maximiza a similaridade entre pares positivos e minimiza entre pares negativos:

```
L_InfoNCE(z_i) = -log[ exp(sim(z_i, z_i+) / τ) / Σ_j exp(sim(z_i, z_j) / τ) ]
```

Onde:
- `z_i`: embedding da âncora
- `z_i+`: embedding do par positivo (augmentation de `z_i`)
- `z_j`: embeddings de pares negativos (outras amostras do batch)
- `sim(·,·)`: similaridade coseno
- `τ`: temperatura (tipicamente 0.07-0.2)

#### Reformulação: Shifted Log-Sum-Exp

A forma original pode ser reescrita de maneira mais intuitiva:

```
L_InfoNCE(z_i) = log[ Σ_j exp(m_ij) ]

onde: m_ij = [sim(z_i, z_j) - sim(z_i, z_i)] / τ
```

**Propriedade chave**: `sim(z_i, z_i) = 1` (máximo), então `-1 < m_ij < 0`.

#### Limitação do InfoNCE em CIL

**Problema**: InfoNCE **nunca atinge zero**, mesmo quando o modelo já discrimina perfeitamente:

**Caso Ideal** (embeddings perfeitamente descorrelacionados):
```
S = [1  0  0  ...  0]  ← Matriz de similaridade ideal
    [0  1  0  ...  0]
    [0  0  1  ...  0]
    [⋮  ⋮  ⋮  ⋱  ⋮]
    [0  0  0  ...  1]
```

Mesmo neste caso ideal:
```
L_InfoNCE(z_i) = log(Σ_j exp(0-1)) = log(N × e^(-1)) ≈ log(N) - 1
```

Para N=10 amostras: `L ≈ 1.30` (não zero!)

**Consequência**: O modelo continua **atualizando parâmetros** mesmo quando não é necessário → **Non-Essential Tuning** → Aumenta catastrophic forgetting.

---

### ANT Loss: Adaptive Negative Threshold

#### Motivação

InfoNCE atualiza **todos** os negativos proporcionalmente, mas apenas os **hard negatives** (próximos ao positivo) são realmente informativos. Atualizar negativos fáceis causa:

1. **Gradientes desnecessários** em parâmetros irrelevantes
2. **Maior risco de forgetting** de conhecimento anterior
3. **Convergência mais lenta** e menos estável

#### Formulação Matemática

ANT Loss foca apenas em hard negatives usando uma **margem adaptativa**:

```
ant_m_ij = sim(z_i, z_j) - (max_k sim(z_i, z_k) - margin)

L_ANT(z_i) = log[ Σ_j exp(m_ij) · 𝟙(ant_m_ij > 0) ]
```

Onde:
- `𝟙(·)`: função indicadora (1 se verdadeiro, 0 caso contrário)
- `margin`: threshold de dificuldade (hiperparâmetro, tipicamente 0.1-0.5)
- `max_k sim(z_i, z_k)`: maior similaridade negativa

**Interpretação**:
- Se `ant_m_ij > 0`: negativo está dentro da margem → **hard negative** → incluído
- Se `ant_m_ij ≤ 0`: negativo está longe → **easy negative** → **ignorado**

#### Benefícios do ANT

1. **Selective Updates**: Apenas hard negatives contribuem para gradiente
2. **Avoid Non-Essential Tuning**: Parâmetros irrelevantes não são atualizados
3. **Better Retention**: Menos interferência com conhecimento anterior
4. **Margin Control**: Hyperparâmetro ajustável para controlar "hardness"

#### Exemplo Visual

```
Similaridades com âncora z_i:
    z_i+ (positivo):  0.95  ✅
    z_1 (negativo):   0.70  🔴 Hard negative (dentro da margem)
    z_2 (negativo):   0.50  🟡 Medium negative (fora da margem)
    z_3 (negativo):   0.20  ⚪ Easy negative (fora da margem)

Com margin=0.3 e max_neg=0.70:
    Threshold = 0.70 - 0.3 = 0.40
    
    ant_m_i1 = 0.70 - 0.40 = 0.30 > 0  ✅ Incluído
    ant_m_i2 = 0.50 - 0.40 = 0.10 > 0  ✅ Incluído
    ant_m_i3 = 0.20 - 0.40 = -0.20 ≤ 0 ❌ Ignorado
```

---

### Local Anchor Normalization

#### Problema da Normalização Global

InfoNCE (e ANT) tradicionalmente usam um **máximo global** compartilhado:

```
max_global = max(todas_as_similaridades_negativas_do_batch)
m_ij = sim(z_i, z_j) - max_global
```

**Problema**: Âncoras com negativos fáceis são **penalizadas** pelo max de outras âncoras com negativos difíceis → Gradientes desbalanceados.

#### Solução: Normalização Local

Cada âncora usa seu **próprio máximo local**:

```
max_local_i = max_j (sim(z_i, z_j))  [apenas negativos de z_i]
m_ij^local = sim(z_i, z_j) - max_local_i
```

**Benefícios**:
1. **Adaptive per Anchor**: Cada âncora avaliada em seu próprio contexto
2. **Balanced Gradients**: Todas as âncoras contribuem proporcionalmente
3. **Works Independently**: Funciona sozinha (+0.14%) ou com ANT (+0.31%)

#### Comparação Visual

```
Âncora A: negativos = [0.65, 0.45, 0.25]  → max_local_A = 0.65
Âncora B: negativos = [0.80, 0.75, 0.30]  → max_local_B = 0.80

Global Norm (max=0.80):
    A é normalizada por 0.80 (máximo de B) → Gradiente comprimido ❌
    
Local Norm:
    A normalizada por 0.65 (seu próprio max) → Gradiente adaptado ✅
    B normalizada por 0.80 (seu próprio max) → Gradiente adaptado ✅
```

---

### TagFex vs TagFex+ANT: Diferenças

| Aspecto | TagFex (Baseline) | TagFex + ANT |
|---------|-------------------|--------------|
| **Loss Contrastiva** | InfoNCE puro | InfoNCE + ANT |
| **Negativos Utilizados** | Todos | Apenas hard negatives |
| **Normalização** | Típica: Global | Recomendado: Local |
| **Atualização de Parâmetros** | Todos os parâmetros | Parâmetros essenciais |
| **Catastrophic Forgetting** | Risco moderado | Risco reduzido |
| **Hiperparâmetros** | `τ` (temperatura) | `τ`, `β` (ANT strength), `m` (margin) |
| **Avg NME@1 (CIFAR-100 10-10)** | 75.55% | **76.18%** (+0.63%) |

#### Loss Total Comparada

**TagFex Baseline**:
```
L = L_cls + λ · InfoNCE(f_ta)
```

**TagFex + ANT**:
```
L = L_cls + λ_nce · InfoNCE(f_ta) + λ_ant · ANT(f_ta)

ou combinado:

L = L_cls + λ · [α · InfoNCE + β · ANT]
```

Onde tipicamente: `α=1.0`, `β=0.5` (melhor configuração).

---

### Melhor Configuração Encontrada

Após 15 experimentos, a configuração ótima é:

```yaml
# InfoNCE Base
nce_alpha: 1.0
infonce_temp: 0.2

# ANT Loss
ant_beta: 0.5           # Moderate strength
ant_margin: 0.5         # Medium margin (sweet spot)
ant_max_global: false   # Local normalization
```

**Resultado**: 79.35% Avg Acc@1 (+0.31% vs baseline)

**Interpretação**:
- `β=0.5`: Balança InfoNCE e ANT (mais robusto que β=1.0)
- `margin=0.5`: Captura hard negatives discriminativos (não muito amplo)
- `local=true`: Normalização adaptativa por âncora

---

## 📊 Estado Atual do Projeto

### Última Atualização: 11 Março 2026

**Status**: 🔄 Fase de **reproducibilidade** — rodando 5 seeds para resultados do paper

**Experimentos Totais**: 15 configurações exploradas (seed 1993) → reproduzindo 4 configurações-chave com 5 seeds (1993–1997)

### Convenção de Seeds

Cada run recebe o sufixo `_s{seed}` no diretório de log:
- Seed `1993` — padrão da comunidade (iCaRL, PyCIL, C3Box), já executado
- Seeds `1994–1997` — runs adicionais para média ± desvio padrão no paper
- A `class_order` está fixa no YAML (não varia com seed); apenas inicialização de pesos e augmentation variam

### Filas Ativas (quati · 11 mar 2026)

| Sessão screen | Script | Status |
|---------------|--------|--------|
| `tagfex_queue` | `run_experiments_queue.sh` | 🔄 Tiny ImageNet 20-20 — exp 2/4 em andamento |
| `cifar100_queue` | `run_cifar100_parallel_queue.sh` | 🔄 CIFAR-100 — exp 5/17 em andamento |

**`tagfex_queue`** (sequencial, seed 1993):
| # | Experimento | Status | avg_nme1 |
|---|-------------|--------|----------|
| 1 | Tiny ImageNet 20-20 Baseline Local | ✅ Concluído | **57.14%** |
| 2 | Tiny ImageNet 20-20 Baseline Global | 🔄 Rodando (task 4/10) | — |
| 3 | Tiny ImageNet 20-20 ANT β=0.5 m=0.5 Local | ⏳ | — |
| 4 | Tiny ImageNet 20-20 ANT β=0.5 m=0.5 Global | ⏳ | — |

**`cifar100_queue`** (paralela via `--min-free-mb 8000`):
| # | Experimento | Status | avg_nme1 |
|---|-------------|--------|----------|
| 1–4 | CIFAR-100 10-10 Baseline Local s1994–1997 | ✅ Concluídos | 75.89 / 76.08 / 75.39 / 75.37 |
| 5 | CIFAR-100 10-10 ANT β=0.5 m=0.5 Local s1994 | 🔄 Rodando (task 7/10) | — |
| 6–8 | CIFAR-100 10-10 ANT β=0.5 m=0.5 Local s1995–1997 | ⏳ | — |
| 9–12 | CIFAR-100 50-10 Baseline Local s1994–1997 | ⏳ | — |
| 13–17 | CIFAR-100 50-10 ANT β=0.5 m=0.5 Local s1993–1997 | ⏳ | — |

### Experimentos Concluídos

| Data | Experimento | Seed | avg_nme1 |
|------|-------------|------|----------|
| Mar 11, 2026 | CIFAR-100 10-10 Baseline Local | 1994 | 75.89% |
| Mar 11, 2026 | CIFAR-100 10-10 Baseline Local | 1995 | 76.08% |
| Mar 11, 2026 | CIFAR-100 10-10 Baseline Local | 1996 | 75.39% |
| Mar 11, 2026 | CIFAR-100 10-10 Baseline Local | 1997 | 75.37% |
| Mar 11, 2026 | Tiny ImageNet 20-20 Baseline Local | 1993 | 57.14% |
| Dez 8-10, 2025 | ANT β=0.5, margins 0.5/0.6/0.7, Local | 1993 | 76.18% |
| Dez 3-4, 2025 | ANT β=1.0, m=0.1/0.5, Local | 1993 | — |
| Nov 20-22, 2025 | CIFAR-100 50-10 variations | 1993 | — |
| Nov 19, 2025 | InfoNCE Local Anchor, ImageNet-100 | 1993 | — |
| Nov 12-13, 2025 | Local vs Global comparison | 1993 | — |

---

## 🏆 Melhor Configuração Encontrada

### ANT β=0.5, margin=0.5, Local Anchor

**Experimento**: `done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1993/`

#### Resultados CIFAR-100 10-10 (10 tasks)

| Métrica | Valor | Δ vs Baseline |
|---------|-------|---------------|
| **Avg Acc@1** | **79.35%** ⭐ | **+0.31%** |
| **Last Acc@1** | **70.77%** | **+0.41%** |
| **Avg NME@1** | **76.18%** ⭐⭐ | **+0.63%** |

**Curva de Acurácia por Task**:
```
[93.40, 85.90, 84.57, 81.32, 79.32, 77.48, 75.70, 73.65, 71.40, 70.77]
```

#### Parâmetros da Configuração

```yaml
# Contrastive Learning
nce_alpha: 1.0              # InfoNCE base

# ANT Loss
ant_beta: 0.5               # ⭐ Strength moderada
ant_margin: 0.5             # ⭐ Margem ótima
ant_max_global: false       # ✅ Local anchor normalization

# Gap Maximization (não usado neste experimento)
gap_target: 0.0
gap_beta: 0.0
```

#### Comparação com Baseline TagFex

**Baseline** (InfoNCE puro, global anchor):
- Avg Acc@1: 79.04%
- Last Acc@1: 70.36%
- Avg NME@1: 75.55%

**Melhorias**:
- ✅ +0.31% Avg Acc@1
- ✅ +0.41% Last Acc@1  
- ✅ **+0.63% Avg NME@1** (melhor discriminação de features)

---

## 💻 Instalação e Uso

### Requisitos

```bash
pytorch torchvision torchmetrics loguru tqdm
```

### Instalação

```bash
git clone https://github.com/bwnzheng/TagFex_CVPR2025.git
cd TagFex_CVPR2025

# Recomendado: usar uv (Python 3.12.11)
uv venv .venv --python 3.12.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Alternativa: pip
pip install -r requirements.txt
```

### Download de Datasets

```bash
# Tiny ImageNet (400 MB) — necessário para experimentos na quati
python setup_tiny_imagenet.py  # salva em ~/data/datasets/tiny-imagenet-200/

# CIFAR-100 e ImageNet-100 — download automático via torchvision
```

### Execução

#### Single-GPU (CIFAR-100, Tiny ImageNet — quati)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py train \
  --exp-configs configs/all_in_one/tiny_imagenet_20-20_baseline_local_resnet18.yaml
```

#### Multi-GPU (ImageNet-100 — fera)

```bash
./trainddp.sh 0,1 \
  --exp-configs configs/all_in_one/imagenet100_10-10_baseline_local_resnet18.yaml \
  --log-dir ./logs/exp_imagenet100_10-10
```

#### Auto-Lançamento com Fila de GPUs 🚀

Sistema automático que monitora GPUs e dispara experimentos quando há recursos disponíveis. Suporta **duas filas paralelas** na quati:

```bash
# Fila principal (Tiny ImageNet) — roda em screen
./start_queue_monitor.sh        # cria sessão screen tagfex_queue

# Fila CIFAR-100 paralela — roda em segundo screen simultâneo
screen -dmS cifar100_queue ./run_cifar100_parallel_queue.sh

# Experimento único com espera por memória livre absoluta
# --min-free-mb: útil para paralelismo (não depende de GPU "idle")
python3 auto_run_on_free_gpu.py \
  --command "python main.py train --exp-configs configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml --seed 1994" \
  --min-free-mb 8000

# Verificar processos idle ocupando GPUs
./check_gpu_processes.sh
```

**Perfis de máquina em [`run_experiments_queue.sh`](run_experiments_queue.sh):**

| `MACHINE` | GPUs/exp | Critério de GPU livre | Fila |
|-----------|----------|-----------------------|------|
| `quati` | 1 | 10% VRAM ocupada | Tiny ImageNet 20-20 (seed 1993) |
| `fera` | 2 (torchrun) | 5% VRAM ocupada | ImageNet-100 + CIFAR-100 global |

**[`run_cifar100_parallel_queue.sh`](run_cifar100_parallel_queue.sh)** — fila dedicada CIFAR-100:
- Usa `--min-free-mb 8000` em vez de threshold percentual
- Inicia quando há ≥ 8 GB livres — funciona **em paralelo com Tiny ImageNet (~6.5 GB)**
- 17 runs: CIFAR-100 10-10 e 50-10, seeds 1994–1997 (+ seed 1993 para 50-10 ANT)

**Parâmetros de [`auto_run_on_free_gpu.py`](auto_run_on_free_gpu.py):**

| Flag | Descrição | Uso típico |
|------|-----------|------------|
| `--threshold` | % de utilização máxima | GPU idle sem job |
| `--memory-threshold` | % de VRAM máxima ocupada | GPU com processo leve |
| `--min-free-mb` | MB mínimos **livres** | Execução paralela |

**Vantagens:**
- ✅ Não precisa monitorar manualmente com `gpustat`
- ✅ Dois experimentos em paralelo na mesma GPU (Tiny ImageNet + CIFAR-100)
- ✅ Perfis por máquina — sem precisar ajustar manualmente
- ✅ Suporta fila overnight com logging de progresso `[N/total]`

📖 **Documentação completa**: [AUTO_GPU_LAUNCHER.md](AUTO_GPU_LAUNCHER.md)  
🧠 **GPU com memória ocupada mas 0% uso?** Veja: [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md)

```bash
# Monitorar filas em execução
screen -ls                                              # listar sessões
screen -r tagfex_queue                                 # fila Tiny ImageNet
screen -r cifar100_queue                               # fila CIFAR-100
tail -f logs/auto_experiments/queue_progress.log       # log tagfex_queue
tail -f logs/auto_experiments/cifar100_queue_progress.log  # log cifar100_queue
tail -f logs/exp_.../exp_stdlog0.log                   # log de treino individual
```

### Exemplos de Uso

```bash
# Tiny ImageNet — melhor config ANT (quati)
python main.py train \
  --exp-configs configs/all_in_one/tiny_imagenet_20-20_ant_beta0.5_margin0.5_local_resnet18.yaml

# CIFAR-100 — melhor configuração (ANT β=0.5, m=0.5, Local)
python main.py train \
  --exp-configs configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml

# CIFAR-100 — Baseline com Local Anchor
python main.py train \
  --exp-configs configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml

# Multi-GPU ImageNet-100 (fera)
./trainddp.sh 0,1 \
  --exp-configs configs/all_in_one/imagenet100_10-10_baseline_local_resnet18.yaml

# Debug — validação rápida do pipeline (3/2 épocas)
python main.py train \
  --exp-configs configs/all_in_one/tiny_imagenet_20-20_debug_resnet18.yaml \
  --disable-save-ckpt --terminal-only
```

### Visualização de Loss Components

```bash
python plot_loss_components.py \
  logs/exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal/exp_debug0.log \
  -t contrast
```

---

## 📊 Experimentos Realizados

### CIFAR-100 10-10 (10 tasks, 10 classes cada)

| Configuração | Avg Acc@1 | Last Acc@1 | Avg NME@1 | Δ vs Baseline |
|--------------|-----------|------------|-----------|---------------|
| **ANT β=0.5, m=0.5, Local** | **79.35%** ⭐ | **70.77%** | **76.18%** ⭐ | **+0.31%** |
| ANT β=0.5, m=0.1, Local | 79.32% | 70.64% | 75.81% | +0.27% |
| ANT β=1.0, m=0.5, Local | 79.27% | 70.20% | 75.91% | +0.23% |
| ANT β=0.5, m=0.7, Local | 79.24% | 70.85% | 75.67% | +0.20% |
| InfoNCE Local Anchor | 79.18% | 70.33% | 75.74% | +0.14% |
| ANT β=0.5, m=0.1, Global | 79.16% | 70.64% | 75.72% | +0.12% |
| ANT β=0.5, m=0.6, Local | 79.14% | 70.49% | 75.65% | +0.10% |
| **Baseline TagFex** | 79.04% | 70.36% | 75.55% | -- |
| ANT β=1.0, m=0.3, Local | 78.99% | 70.18% | 75.60% | -0.05% |
| ANT β=1.0, m=0.1, Local | 78.97% | 70.11% | 75.74% | -0.07% |

**Total**: 11 configurações únicas testadas

### CIFAR-100 50-10 (6 tasks: 50+5×10)

| Configuração | Avg Acc@1 | Last Acc@1 | Avg NME@1 | Observação |
|--------------|-----------|------------|-----------|------------|
| Baseline Local | **77.13%** | 71.44% | 76.48% | Melhor Avg |
| Baseline Global | 77.11% | **71.91%** | **76.53%** | Melhor Last |
| ANT β=1.0, m=0.5, Local | 77.08% | 71.38% | 76.16% | Impacto mínimo |

**Observação**: Base task grande (50 classes) → representação já robusta → ANT menos crítico

### ImageNet-100 10-10

| Configuração | Avg Acc@1 | Last Acc@1 | Avg NME@1 |
|--------------|-----------|------------|-----------|
| Baseline Local | **81.28%** | 72.84% | **77.36%** |

### Tiny ImageNet 20-20 (10 tasks × 20 classes) 🆕

200 classes, imagens 64×64. Executado na quati (RTX 4090 24GB, single GPU). Apenas seed 1993 (exploração inicial; 5 seeds para CIFAR-100 são os resultados do paper).

| Configuração | Avg Acc@1 | Last Acc@1 | Avg NME@1 | Status |
|--------------|-----------|------------|-----------|--------|
| Baseline Local | **60.68%** | 52.57% | **57.14%** | ✅ Concluído (s1993) |
| Baseline Global | — | — | — | 🔄 Rodando (task 4/10) |
| ANT β=0.5, m=0.5, Local | — | — | — | ⏳ |
| ANT β=0.5, m=0.5, Global | — | — | — | ⏳ |

> Os experimentos Global vs Local permitem isolar o efeito da normalização de âncora.

### CIFAR-100 — Reproducibilidade Paper (5 seeds)

Resultados do paper: média ± desvio padrão de 5 seeds (1993–1997), seguindo convenção do TagFex original ("5 runs, mean values reported").

| Cenário | Configuração | Seeds concluídas | avg_nme1 (média) | Seeds pendentes |
|---------|--------------|------------------|------------------|-----------------|
| 10-10 | Baseline Local | 1993–1997 ✅ | **75.66%** | — |
| 10-10 | ANT β=0.5 m=0.5 Local | 1993 ✅ | — | 1994–1997 🔄 |
| 50-10 | Baseline Local | 1993 ✅ | — | 1994–1997 ⏳ |
| 50-10 | ANT β=0.5 m=0.5 Local | — | — | 1993–1997 ⏳ |

**Gráfico atualizado**: [`cifar100_10-10.png`](cifar100_10-10.png) — Baseline Local 5 seeds vs. baselines do paper (iCaRL, DyTox, DER, TagFex ref).

---

## 💡 Descobertas Principais

### 1. Local Anchor > Global Anchor

**Sempre** usar normalização local (`ant_max_global: false`):

- Local (β=0.5, m=0.1): **79.32%**
- Global (β=0.5, m=0.1): 79.16% (-0.16%)

**Ganho médio**: +0.16% a +0.27%

### 2. Margin 0.5 é Ótimo para β=0.5

Com ANT β=0.5:

| Margin | Avg Acc@1 | Δ vs m=0.5 |
|--------|-----------|------------|
| **0.5** | **79.35%** | -- |
| 0.7 | 79.24% | -0.11% |
| 0.6 | 79.14% | -0.21% |
| 0.1 | 79.32% | -0.03% |

**Interpretação**: Margin muito grande (0.7) inclui negatives menos discriminativos. Margin 0.5 é o "sweet spot".

### 3. β Moderado (0.5) > β Alto (1.0)

Com margin fixo:

| β | Margin | Avg Acc@1 | Observação |
|---|--------|-----------|------------|
| **0.5** | 0.5 | **79.35%** | Mais estável |
| 1.0 | 0.5 | 79.27% (-0.08%) | OK com margin alta |
| 1.0 | 0.1 | 78.97% (-0.38%) | Instável com margin baixa |

**Conclusão**: β moderado é mais robusto a variações de margin.

### 4. ANT Funciona Melhor com Base Tasks Pequenas

| Split | Base Classes | ANT Impact | Interpretação |
|-------|--------------|------------|---------------|
| **10-10** | 10 | **+0.31%** | Representação inicial fraca → ANT crítico |
| **50-10** | 50 | -0.03% | Representação inicial robusta → ANT desnecessário |

**Implicação**: ANT é mais útil em cenários com menos dados iniciais.

### 5. Local Anchor Funciona Isolado

**InfoNCE Local Anchor** (β=0, local norm): **79.18%** (+0.14% vs baseline)

**Conclusão**: Normalização local **por si só** já melhora o InfoNCE, mesmo sem ANT.

---

## 🔬 ANT Loss: Conceito e Implementação

### O Que é ANT Loss?

**ANT (Adaptive Negative Threshold Loss)** é uma extensão do InfoNCE que foca em **hard negatives** - amostras negativas difíceis de distinguir, próximas da similaridade positiva.

### Motivação

InfoNCE trata todos os negativos igualmente:

```python
loss = -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
```

**Problema**: Negativos fáceis dominam o denominador, mas contribuem pouco para aprendizado discriminativo.

### Solução: ANT

```python
# 1. Identificar hard negatives (dentro de uma margem)
gap = sim_pos - sim_neg_i
is_hard = gap < margin

# 2. Penalizar apenas hard negatives
ant_loss = sum(relu(margin - gap)) se is_hard
```

### Implementação

**Arquivo**: `methods/tagfex/tagfex.py`, função `infoNCE_loss()`

```python
def infoNCE_loss(
    cos_sim,
    temperature,
    nce_alpha=1.0,
    ant_beta=0.0,        # ANT strength
    ant_margin=0.1,      # Margin threshold
    max_global=True,     # Use global or local anchor
    gap_target=0.0,      # Gap maximization target
    gap_beta=0.0,        # Gap maximization strength
):
    # 1. InfoNCE base
    nll = F.cross_entropy(cos_sim / temperature, targets)
    
    # 2. ANT loss (se habilitado)
    if ant_beta > 0:
        pos_sim = cos_sim[pos_mask]
        neg_sim = cos_sim[~pos_mask]
        
        # Gap entre positivo e negativo
        gap = pos_sim.unsqueeze(-1) - neg_sim.reshape(batch_size, -1)
        
        # Penalizar gaps < margin (hard negatives)
        ant_loss = F.relu(ant_margin - gap).mean()
    
    # 3. Loss total
    total_loss = nce_alpha * nll + ant_beta * ant_loss
    
    return total_loss
```

### Hiperparâmetros

| Parâmetro | Valores Testados | Melhor | Descrição |
|-----------|------------------|--------|-----------|
| `ant_beta` | 0.0, 0.5, 1.0 | **0.5** | Strength da ANT loss |
| `ant_margin` | 0.1, 0.3, 0.5, 0.6, 0.7 | **0.5** | Janela de hard negatives |
| `ant_max_global` | True, False | **False** | Local anchor normalization |

---

## 🎯 Local vs Global Anchor

### Diferença Conceitual

**Global Anchor** (`max_global=True`):
- Todas as âncoras compartilham o mesmo máximo de similaridade
- Normalização global: `logits_norm = logits - max(todas_as_negativas)`

**Local Anchor** (`max_global=False`):
- Cada âncora usa seu próprio máximo local
- Normalização adaptativa: `logits_norm = logits - max(negativas_desta_âncora)`

### Por Que Local é Melhor?

1. **Adaptação ao contexto**: Cada âncora tem dificuldade diferente
2. **Evita dominância global**: Negativos difíceis de uma âncora não "comprimem" outras
3. **Gradiente mais balanceado**: Todas as âncoras contribuem proporcionalmente

### Visualização

Ver: [papers/ant/THEORY_LOCAL_ANCHOR.md](papers/ant/THEORY_LOCAL_ANCHOR.md) para detalhes matemáticos e visualizações completas.

**Exemplo simplificado**:

```
Âncora A: positiva=0.92, negativas=[0.65, 0.45, 0.25]
Âncora B: positiva=0.88, negativas=[0.80, 0.75, 0.30]

Global: max = 0.80 (de B) → A é penalizada por max de B
Local:  max_A = 0.65, max_B = 0.80 → cada uma com seu contexto
```

### Implementação

```python
if not max_global:  # Local anchor
    # Encontrar max per âncora
    cos_sim_neg = cos_sim.clone()
    cos_sim_neg[pos_mask] = -float('inf')
    max_neg_per_anchor = cos_sim_neg.max(dim=-1, keepdim=True).values
    
    # Subtrair max local
    cos_sim = cos_sim - max_neg_per_anchor
else:  # Global anchor
    max_neg_global = cos_sim[~pos_mask].max()
    cos_sim = cos_sim - max_neg_global
```

---

## 📁 Estrutura do Projeto

```
TagFex_CVPR2025/
├── README.md                      # Este arquivo
├── main.py                        # Entry point
├── trainddp.sh                    # Multi-GPU training
├── requirements.txt               # Dependências
├── setup_tiny_imagenet.py         # Download e verificação do Tiny ImageNet 🆕
├── run_experiments_queue.sh       # Fila principal (Tiny ImageNet + ImageNet-100)
├── run_cifar100_parallel_queue.sh # Fila CIFAR-100 5-seed (paralelo, --min-free-mb)
├── start_queue_monitor.sh         # Lança fila principal em sessão screen
├── auto_run_on_free_gpu.py        # Monitor GPU: --threshold / --memory-threshold / --min-free-mb
│
├── configs/                       # Configurações
│   ├── all_in_one/               # Configs prontas (all-in-one YAML)
│   │   ├── cifar100_10-10_*                          # CIFAR-100 10-10 (6 configs)
│   │   ├── cifar100_50-10_*                          # CIFAR-100 50-10 (3 configs)
│   │   ├── imagenet100_10-10_*                       # ImageNet-100 10-10 (3 configs)
│   │   ├── imagenet100_50-10_*                       # ImageNet-100 50-10 (3 configs)
│   │   └── tiny_imagenet_20-20_*                     # Tiny ImageNet 20-20 (5 configs) 🆕
│   ├── exps/                     # Configs experimentais
│   └── scenarios/                # Cenários de datasets
│       └── tiny_imagenet.yaml    # Cenário Tiny ImageNet 🆕
│
├── methods/                       # Implementações de métodos
│   └── tagfex/
│       ├── tagfex.py             # TagFex + ANT + Local Anchor
│       └── ...
│
├── modules/                       # Componentes reutilizáveis
│   ├── evaluation.py             # Métricas
│   ├── metrics.py                # Cálculo de métricas
│   ├── data/
│   │   ├── dataset.py            # CIFAR-100, ImageNet-100, Tiny ImageNet, CUB, DomainNet
│   │   ├── augmentation.py       # Transforms por dataset
│   │   └── manager.py
│   └── backbones/
│       └── resnet.py             # ResNet18 com stem adaptável por dataset
│
├── loggers/                       # Sistema de logging
│   ├── loguru.py                 # Logger principal
│   └── utils.py
│
├── logs/                          # Experimentos executados
│   ├── done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1993/        # Baseline Local seed 1993
│   ├── done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1994/        # avg_nme1=75.89
│   ├── done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1995/        # avg_nme1=76.08
│   ├── done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1996/        # avg_nme1=75.39
│   ├── done_exp_cifar100_10-10_antB0_nceA1_antLocal_s1997/        # avg_nme1=75.37
│   ├── done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1993/  # ⭐ Melhor ANT
│   ├── done_exp_tiny_imagenet_20-20_antB0_nceA1_antLocal_s1993/   # avg_nme1=57.14
│   ├── exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_s1994/  # 🔄 Rodando
│   ├── exp_tiny_imagenet_20-20_antB0_nceA1_antGlobal_s1993/       # 🔄 Rodando
│   └── ... (15+ concluídos)
│
├── analysis/                      # Scripts de análise
│   ├── scripts/
│   │   ├── analyze_ant_gaps.py
│   │   ├── compare_experiments.py
│   │   └── ...
│   └── results/                  # Resultados de análises
│
├── docs/                          # Documentação técnica
│   ├── RESULTS_AND_METRICS.md    # Todos os resultados e métricas (Dez 2025)
│   └── DEBUGGING_GUIDE.md        # Guia de debugging e análise
│
└── papers/                        # Artigos e material de pesquisa
    ├── tagfex/                    # TagFex CVPR 2025
    │   ├── Task-Agnostic...pdf
    │   └── assets/                # Figuras do artigo
    ├── ant/                       # ANT Loss (AAAI 2026)
    │   ├── ant.tex
    │   ├── antcil.tex
    │   └── THEORY_LOCAL_ANCHOR.md
    └── c3box/                     # C3Box (arXiv 2601.20852)
```

---

## 🔍 Análise e Debugging

### Scripts de Análise

#### 1. Análise Rápida de Gaps

```bash
python analysis/scripts/quick_gap_analysis.py \
  --log-file logs/exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal/exp_matrix_debug0.log
```

**Output**: Gráficos de evolução de gap, ANT loss, gap loss por task.

#### 2. Comparação de Experimentos

```bash
python analysis/scripts/compare_experiments.py \
  --baseline logs/done_exp_cifar100_10-10_baseline_tagfex_original/exp_matrix_debug0.log \
  --ant-exp logs/done_exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal/exp_matrix_debug0.log \
  --output comparison_results
```

**Output**: 
- Gráficos comparativos side-by-side
- Estatísticas de diferenças
- Análise task-by-task

#### 3. Visualização de Matrizes de Similaridade

```bash
python analysis/scripts/visualize_local_anchor_theory.py
```

**Output** (`analysis/results/infonce_theory/`):
- `infonce_matrices_comparison.png` - Comparação Global vs Local
- `infonce_loss_computation.png` - Cálculo detalhado
- `infonce_gradient_impact.png` - Impacto no gradiente

#### 4. Análise Detalhada de ANT

```bash
python analysis/scripts/analyze_ant_gaps.py \
  --log-file logs/exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal/exp_matrix_debug0.log
```

**Output**: 19 métricas ao longo do treinamento (pos_mean, neg_mean, gap, violation %, etc.)

### Debug de Similaridades

Para habilitar debug detalhado de matrizes de similaridade:

```yaml
# Adicionar ao config YAML
debug_similarity: true
debug_similarity_batch_size: 16
```

Ver: [docs/DEBUGGING_GUIDE.md](docs/DEBUGGING_GUIDE.md) para detalhes completos.

### Visualização de Loss Components

```bash
python plot_loss_components.py \
  logs/exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal/exp_debug0.log \
  -t contrast
```

**Output**: Gráficos de InfoNCE NLL, ANT loss, componentes ponderados, loss total.

---

## 📚 Documentação Adicional

Para informações detalhadas, consulte os documentos em [docs/](docs/):

### 1. [RESULTS_AND_METRICS.md](docs/RESULTS_AND_METRICS.md)

Consolidação completa de todos os resultados experimentais:
- ✅ Métricas de todos os 15 experimentos
- ✅ Comparações detalhadas com baseline
- ✅ Análise de ganhos por task
- ✅ TOP 5 configurações
- ✅ Curvas de acurácia completas
- ✅ Recomendações para o artigo

### 2. [THEORY_LOCAL_ANCHOR.md](papers/ant/THEORY_LOCAL_ANCHOR.md)

Teoria matemática e implementação da normalização de âncora local:
- 📐 Diferença conceitual entre Global vs Local
- 📊 Visualizações de matrizes de similaridade
- 🧮 Cálculo da loss passo a passo
- 📈 Impacto no gradiente
- 🔬 Implementação no código
- ✅ Resultados experimentais
- 💡 Aplicabilidade a outros métodos

### 3. [DEBUGGING_GUIDE.md](docs/DEBUGGING_GUIDE.md)

Guia completo de ferramentas de debugging:
- 🔧 Como habilitar debug de similaridade
- 📄 Outputs gerados (logs, heatmaps)
- 📊 Análise de resultados
- ⚙️ Ajuste de hiperparâmetros
- 🎯 Interpretação de métricas

### 4. [AUTO_GPU_LAUNCHER.md](AUTO_GPU_LAUNCHER.md) 🆕

Sistema de execução automática de experimentos quando GPU fica disponível:
- 🚀 Monitoramento contínuo de GPUs via `nvidia-smi`
- ⚙️ Suporte a threshold de utilização e memória
- 📋 Fila de experimentos para execução overnight
- 📊 Logs detalhados de execução
- 🔧 Scripts de diagnóstico (`test_gpu_monitor.sh`, `diagnose_gpu_memory.sh`)

**Uso rápido:**
```bash
# Executar um experimento quando GPU ficar livre
python3 auto_run_on_free_gpu.py --config configs/xxx.yaml --threshold 5.0

# Executar fila de experimentos
./run_experiments_queue.sh
```

### 5. [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md) 🆕

Guia completo sobre o comportamento de memória vs utilização de GPUs:
- 🧠 Por que GPU mostra 0% utilização mas 87% memória ocupada?
- 🔍 5 causas comuns (processos mortos, idle, Jupyter, multi-GPU, memory leaks)
- 🛠️ Soluções específicas para cada cenário
- ✅ Checklist antes de matar processos
- 📝 Melhores práticas para código GPU-efficient

**Diagnóstico rápido:**
```bash
# Identificar processos mortos/idle
./diagnose_gpu_memory.sh

# Verificar processos ativos
./check_gpu_processes.sh
```

---

## 🎓 Referências

### Papers

#### TagFex Framework

```bibtex
@inproceedings{tagfex2025,
  title={Task-Agnostic Guided Feature Expansion for Class-Incremental Learning},
  author={[Authors]},
  booktitle={CVPR},
  year={2025}
}
```

#### ANT Loss

```bibtex
@article{ant2024,
  title={Adaptive Negative Threshold Loss for Contrastive Learning},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024},
  note={See ant.tex and antcil.tex in project root}
}
```

### Links

- **ArXiv (TagFex)**: https://arxiv.org/abs/2503.00823
- **Repository**: https://github.com/bwnzheng/TagFex_CVPR2025
- **ANT Paper (.tex)**: [ant.tex](ant.tex), [antcil.tex](antcil.tex)

### Código-fonte

- **TagFex Implementation**: [methods/tagfex/tagfex.py](methods/tagfex/tagfex.py)
  - `_compute_contrastive_loss_base()`: Implementação ANT Loss (linha ~804)
  - `infoNCE_loss()`: Implementação InfoNCE (linha ~982)
  - Local vs Global anchor normalization (parâmetro `ant_max_global`)

### Datasets

- **CIFAR-100**: 100 classes, splits 10-10 (10 tasks) e 50-10 (6 tasks)
- **ImageNet-100**: 100 classes, split 10-10 (10 tasks)

### Inspiração

Este repositório foi inspirado por [PyCIL](https://github.com/G-U-N/PyCIL).

---

## 📝 Changelog do Desenvolvimento

### Dezembro 2025 - Fase Final ✅

- ✅ Identificada melhor configuração: **ANT β=0.5, m=0.5, Local**
- ✅ Experimentos com margins variadas (0.1, 0.3, 0.5, 0.6, 0.7)
- ✅ Validação em 3 datasets (CIFAR-100 10-10, 50-10, ImageNet-100)
- ✅ Documentação consolidada em [docs/](docs/)

### Novembro 2025 - Desenvolvimento Intensivo

- 🔬 Implementação de Gap Maximization Loss
- 🔬 Comparação Local vs Global Anchor
- 🔬 Teste de InfoNCE Local Anchor isolado
- 🔬 Experimentos com β=0.5 vs β=1.0
- 📊 Scripts de análise e visualização

### Fundação

- 🏗️ Framework TagFex original
- 🏗️ Implementação InfoNCE
- 🏗️ Estrutura de treinamento multi-GPU

---

## 🤝 Contribuições

Para questões ou sugestões, abra uma issue no repositório.

---

**Última atualização**: Dezembro 2025  
**Status**: ✅ Projeto completo - Melhor configuração identificada e validada
