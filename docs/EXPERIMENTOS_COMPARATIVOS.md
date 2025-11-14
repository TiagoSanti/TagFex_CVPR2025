# Experimentos Comparativos: Baseline vs ANT+Gap

Este guia descreve como executar e analisar os experimentos comparativos entre TagFex baseline (sem ANT) e TagFex com ANT + Gap Maximization.

## 📋 Configurações Disponíveis

### 1. **Baseline (Reprodução do TagFex Original)**
**Arquivo:** `configs/all_in_one/cifar100_10-10_tagfex_baseline_resnet18.yaml`

**Parâmetros:**
```yaml
nce_alpha: 1.0       # InfoNCE ativo
ant_beta: 0.0        # ❌ ANT DESABILITADO
ant_margin: 0.1      # Mantido para logging
ant_max_global: false

gap_target: 0.0      # ❌ Gap maximization DESABILITADO
gap_beta: 0.0        # Sem contribuição de gap loss
```

**Características:**
- ✅ Usa apenas InfoNCE loss (método original)
- ✅ Loga estatísticas de distância ANT (sem afetar treinamento)
- ✅ Permite tracking do gap naturalmente evoluído
- ✅ Baseline para comparação

**Diretório de logs esperado:**
```
logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal_gapT0_gapB0/
```

---

### 2. **ANT + Gap Maximization (Método Completo)**
**Arquivo:** `configs/all_in_one/cifar100_10-10_tagfex_ant_gap_resnet18.yaml`

**Parâmetros:**
```yaml
nce_alpha: 1.0       # InfoNCE base
ant_beta: 1.0        # ✅ ANT ATIVO (contribuição igual)
ant_margin: 0.1      # Margin para filtragem de negativos
ant_max_global: false # Maximum local por âncora

gap_target: 0.7      # ✅ Gap maximization ATIVO (alvo 0.7)
gap_beta: 0.5        # Peso da gap loss
```

**Características:**
- ✅ InfoNCE + ANT loss (foco em hard negatives)
- ✅ Gap maximization loss (força gap >= 0.7)
- ✅ Logging completo de todas as métricas
- ✅ Total loss = nce_loss + ant_loss + gap_beta * gap_loss

**Diretório de logs esperado:**
```
logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5/
```

---

## 🚀 Como Executar os Experimentos

### Opção 1: Script Automatizado (Recomendado)

```bash
# Dar permissão de execução (primeira vez)
chmod +x run_comparison_experiments.sh

# Executar com single GPU
./run_comparison_experiments.sh 0

# Executar com múltiplas GPUs
./run_comparison_experiments.sh 0,1
```

O script perguntará qual experimento executar:
1. Baseline apenas
2. ANT+Gap apenas
3. Ambos (sequencialmente)
4. Sair

### Opção 2: Execução Manual

**Baseline:**
```bash
# Single GPU
python main.py train --exp-configs configs/all_in_one/cifar100_10-10_tagfex_baseline_resnet18.yaml

# Multi-GPU (DDP)
./trainddp.sh 0,1 --exp-configs configs/all_in_one/cifar100_10-10_tagfex_baseline_resnet18.yaml
```

**ANT+Gap:**
```bash
# Single GPU
python main.py train --exp-configs configs/all_in_one/cifar100_10-10_tagfex_ant_gap_resnet18.yaml

# Multi-GPU (DDP)
./trainddp.sh 0,1 --exp-configs configs/all_in_one/cifar100_10-10_tagfex_ant_gap_resnet18.yaml
```

---

## 📊 Análise dos Resultados

### 1. Análise Rápida do Gap (Individual)

```bash
# Analisar experimento baseline
python analysis_scripts/quick_gap_analysis.py

# Será necessário editar o caminho do log no script ou passar como argumento
```

### 2. Análise Comparativa (Baseline vs ANT+Gap)

```bash
# Comparação automática
python analysis_scripts/compare_experiments.py

# Com caminhos customizados
python analysis_scripts/compare_experiments.py \
    --baseline logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal_gapT0_gapB0/exp_matrix_debug0.log \
    --ant-gap logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5/exp_matrix_debug0.log \
    --output comparison_results
```

**Output:**
- `comparison_evolution.png`: Gráficos comparando gap, ANT loss, gap loss, violações
- Estatísticas comparativas no terminal
- Análise task-by-task

### 3. Visualização de Componentes de Loss

```bash
# Baseline
python plot_loss_components.py \
    logs/exp_cifar100_10-10_antB0_nceA1_antM0.1_antLocal_gapT0_gapB0/exp_matrix_debug0.log \
    -t contrast

# ANT+Gap
python plot_loss_components.py \
    logs/exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal_gapT0.7_gapB0.5/exp_matrix_debug0.log \
    -t contrast
```

---

## 📈 Métricas Comparadas

### Durante o Treinamento

| Métrica | Baseline | ANT+Gap | Expectativa |
|---------|----------|---------|-------------|
| **Gap (pos - neg)** | Evolução natural | Forçado para 0.7 | ANT+Gap > Baseline |
| **ANT Loss** | Calculado mas não usado | Ativo, focando hard negatives | Similar |
| **Gap Loss** | 0.0 (desabilitado) | Ativo quando gap < 0.7 | Decrescente |
| **Hard Negative %** | Natural | Reduzido por ANT | ANT+Gap < Baseline |

### Resultados Finais (a serem medidos)

- **Acurácia final** nas 10 tasks
- **Forgetting** médio por task
- **Gap final** atingido
- **Tempo de convergência**

---

## 🔍 O Que Observar

### No Baseline (sem ANT):
- Gap cresce naturalmente apenas pela InfoNCE
- ANT loss é calculado mas não contribui para gradiente
- Violações de margin podem permanecer altas
- Gap provavelmente não atinge 0.7

### No ANT+Gap:
- Gap cresce de forma forçada em direção a 0.7
- ANT loss penaliza hard negatives
- Gap loss ativa quando gap < 0.7
- Violações devem cair rapidamente
- Gap deve atingir ou superar 0.7

---

## 📝 Estrutura de Logs

Ambos os experimentos geram os mesmos arquivos de log:

```
logs/<exp_name>/
├── exp_matrix_debug0.log    # Logs detalhados de ANT distance stats
├── exp_stdlog0.log           # Logs padrão do treinamento
├── exp_gistlog.log           # Logs de performance (rank 0)
└── ckpt/                     # Checkpoints salvos
```

**Conteúdo do exp_matrix_debug0.log:**
- Loss components (InfoNCE NLL, ANT loss, weighted components, total)
- ANT distance stats (pos_mean, neg_mean, gap, violation %, margins)
- Gap maximization metrics (gap_loss, current_gap, gap_target, total_ant_loss)

---

## 🎯 Hipóteses a Validar

1. **Gap Evolution:** ANT+Gap deve atingir gap >= 0.7, enquanto Baseline pode não atingir
2. **Convergência:** ANT+Gap pode convergir mais rápido devido ao foco em hard negatives
3. **Separabilidade:** Features devem ter maior separação classe-a-classe com ANT+Gap
4. **Forgetting:** Menor forgetting esperado com maior gap (melhor separação)
5. **Acurácia:** ANT+Gap deve ter acurácia igual ou superior ao Baseline

---

## 💡 Troubleshooting

### Experimento não está gerando logs de distância?
- Verifique se `debug: true` está no config (não necessário, mas útil)
- Confirme que logger está configurado corretamente

### Gap loss não aparece nos logs do Baseline?
- **Normal!** Baseline tem `gap_beta=0.0`, então gap_loss será sempre 0.0
- ANT distance stats ainda são logged para tracking

### Como saber qual experimento está rodando?
- Cheque o nome do diretório de logs (inclui parâmetros ant_beta e gap_beta)
- Verifique o arquivo de config usado

### Experimentos muito lentos?
- Reduza `num_workers` no config
- Use menos épocas para teste rápido: `init_epochs: 5`, `inc_epochs: 3`

---

## 📚 Referências

- **TagFex Paper:** [Inserir link quando disponível]
- **ANT (Adaptive Negative Thresholding):** Ver `antcil.tex`
- **Gap Maximization:** Ver `docs/analysis/TORNAR_ANT_MAIS_RELEVANTE.md`
