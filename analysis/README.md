# TagFex Analysis

Este diretório contém todos os scripts de análise e resultados dos experimentos TagFex.

## 📁 Estrutura de Diretórios

```
analysis/
├── scripts/                    # Scripts de análise
│   ├── compare_baseline_vs_local.py          # Compara InfoNCE baseline vs âncora local
│   ├── visualize_local_anchor_theory.py      # Visualiza teoria da âncora local
│   ├── compare_tagfex_ant_experiments.py     # Compara experimentos TagFex+ANT
│   ├── compare_experiments.py                # Comparação geral de experimentos
│   ├── plot_all_nme1_curves.py              # Plota curvas NME1
│   ├── analyze_baseline_distances.py         # Analisa distâncias baseline
│   ├── analyze_local_vs_global.py           # Analisa local vs global
│   ├── analyze_ant_gaps.py                   # Análise detalhada de gaps ANT
│   ├── analyze_ant_gaps_simple.py           # Análise simples de gaps ANT
│   ├── quick_gap_analysis.py                # Análise rápida de gaps
│   └── reference_enhanced_ant_loss.py       # Referência da loss ANT aprimorada
│
├── results/                    # Resultados de análises
│   ├── baseline_vs_local/      # Comparação baseline vs âncora local
│   ├── infonce_theory/         # Visualizações teóricas do InfoNCE
│   ├── experiments_comparison/ # Comparações gerais de experimentos
│   ├── nme1_curves/           # Curvas de NME1
│   └── baseline_metrics/      # Métricas baseline
│
└── docs/                       # Documentação adicional (se houver)
```

## 🔧 Scripts Principais

### 1. Comparação Baseline vs Âncora Local
**Script:** `compare_baseline_vs_local.py`

Compara experimentos InfoNCE com normalização global (baseline) vs local (âncora local).

```bash
cd scripts
python compare_baseline_vs_local.py
```

**Output:** `../results/baseline_vs_local/`
- `baseline_vs_local_comparison.png` - Gráficos comparativos
- `comparison_report.md` - Relatório detalhado em Markdown

**Editar diretórios de input:**
Edite as linhas ~856-859 no script para apontar para seus experimentos.

---

### 2. Visualização da Teoria da Âncora Local
**Script:** `visualize_local_anchor_theory.py`

Gera visualizações didáticas explicando a diferença teórica entre InfoNCE original e âncora local.

```bash
cd scripts
python visualize_local_anchor_theory.py
```

**Output:** `../results/infonce_theory/`
- `infonce_matrices_comparison.png` - Comparação de matrizes de similaridade
- `infonce_loss_computation.png` - Cálculo detalhado da loss
- `infonce_gradient_impact.png` - Impacto no gradiente
- `infonce_conceptual_diagram.png` - Diagrama conceitual (se completo)

---

### 3. Comparação de Experimentos TagFex+ANT
**Script:** `compare_tagfex_ant_experiments.py`

Compara múltiplos experimentos TagFex com diferentes configurações ANT.

```bash
cd scripts
python compare_tagfex_ant_experiments.py --baseline-dir ../../logs/exp_baseline --ant-dir ../../logs/exp_ant
```

**Output:** `../results/experiments_comparison/`

---

### 4. Plotar Curvas NME1
**Script:** `plot_all_nme1_curves.py`

Plota curvas de NME1 Accuracy para múltiplos experimentos.

```bash
cd scripts
python plot_all_nme1_curves.py
```

**Output:** `../results/nme1_curves/`

---

## 📊 Estrutura de Logs

Os scripts esperam encontrar logs de experimentos em:

```
../../logs/
├── exp_cifar100_10-10_antB0_nceA1_antM0_antGlobal/    # Baseline global
├── exp_cifar100_10-10_antB0_nceA1_antM0_antLocal/     # Âncora local
├── exp_cifar100_10-10_antB1_nceA1_antM0.1_antLocal/   # TagFex+ANT
└── ...
```

Cada diretório de experimento deve conter:
- `exp_debug0.log` - Log com estatísticas de distância
- `exp_gistlog.log` - Log com resultados de avaliação (NME1, Accuracy)

---

## 🚀 Executando Análises

### Passo 1: Navegar para o diretório de scripts
```bash
cd /home/tiago/TagFex_CVPR2025/analysis/scripts
```

### Passo 2: Executar script desejado
```bash
python compare_baseline_vs_local.py
```

### Passo 3: Verificar resultados
```bash
ls -la ../results/baseline_vs_local/
```

---

## 📝 Notas Importantes

1. **Caminhos Relativos:** Todos os scripts foram atualizados para usar caminhos relativos a partir do diretório `scripts/`.

2. **Logs de Experimentos:** Os scripts procuram logs em `../../logs/` (dois níveis acima de `scripts/`).

3. **Outputs:** Todos os resultados são salvos em `../results/<categoria>/`.

4. **Execução:** Execute sempre os scripts a partir do diretório `scripts/`:
   ```bash
   cd analysis/scripts
   python <script_name>.py
   ```

---

## 🔍 Troubleshooting

### Erro: "FileNotFoundError: logs not found"
- Verifique se os diretórios de logs existem em `../../logs/`
- Edite os caminhos no script conforme necessário

### Erro: "No such file or directory: results/"
- Certifique-se de executar o script a partir de `analysis/scripts/`
- Ou crie os diretórios manualmente:
  ```bash
  mkdir -p ../results/{baseline_vs_local,infonce_theory,experiments_comparison,nme1_curves,baseline_metrics}
  ```

### Gráficos não aparecem
- Certifique-se de que matplotlib está instalado
- Verifique se o ambiente virtual está ativado

---

## 📚 Documentação Adicional

- **Teoria da Âncora Local:** Ver `../../docs/EXPLICACAO_ANCORA_LOCAL.md`
- **Guia de Comparação:** Ver `scripts/COMPARE_EXPERIMENTS_GUIDE.md`
- **Status das Análises:** Ver `scripts/STATUS.md`

---

*Última atualização: 2025-11-19*
