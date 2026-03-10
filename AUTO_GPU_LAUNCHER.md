# 🚀 Auto GPU Launcher - Sistema de Execução Automática de Experimentos

Sistema que monitora o uso das GPUs e dispara experimentos automaticamente quando uma GPU fica disponível.

---

## 📋 Visão Geral

O **Auto GPU Launcher** resolve o problema de GPUs ocupadas por outros projetos. Ele monitora continuamente o uso das GPUs e, quando detecta uma GPU livre (abaixo do threshold configurado), automaticamente inicia o experimento configurado.

### **Características principais:**
- ✅ Monitoramento contínuo via `nvidia-smi`
- ✅ Suporte a threshold de **utilização** e **memória**
- ✅ Detecção automática de GPUs disponíveis
- ✅ Configuração automática de `CUDA_VISIBLE_DEVICES`
- ✅ Log detalhado de execução
- ✅ Fila de experimentos
- ✅ Execução em background

---

## 🛠️ Scripts Disponíveis

### **0. start_queue_monitor.sh** (Recomendado para uso SSH) 🆕
Inicia o monitoramento de GPUs em uma sessão screen, permitindo desconectar do SSH.

```bash
./start_queue_monitor.sh
```

**O que ele faz:**
- ✅ Cria sessão screen chamada `tagfex_gpu_monitor`
- ✅ Executa `run_experiments_queue.sh` dentro da sessão
- ✅ Você pode desconectar do SSH sem interromper o monitoramento
- ✅ Cada experimento também roda em sua própria sessão screen

**Comandos úteis após iniciar:**
- Ver progresso: `screen -r tagfex_gpu_monitor`
- Listar sessões: `screen -ls`
- Desanexar: `Ctrl+A`, depois `D`

---

### **1. auto_run_on_free_gpu.py** (Principal)
Monitora GPUs e lança um experimento quando uma GPU fica livre.

```bash
python3 auto_run_on_free_gpu.py \
  --command "python main.py train --exp-configs configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
  --threshold 5.0 \
  --memory-threshold 10.0 \
  --interval 30
```

**Parâmetros:**
- `--command`: Comando a executar quando GPU ficar disponível (obrigatório)
- `--threshold`: % de utilização máxima para considerar GPU livre (padrão: 1.0)
- `--memory-threshold`: % de memória máxima para considerar GPU livre (padrão: None)
- `--min-free-mb`: MB livres mínimos para considerar GPU disponível (padrão: None). Útil para execução paralela — verifica espaço absoluto, não percentual
- `--interval`: Intervalo de verificação em segundos (padrão: 30)
- `--no-screen`: Não usar sessão screen (executa diretamente)

**🆕 Sessões Screen:**
Por padrão, experimentos são executados em sessões `screen` nomeadas automaticamente:
- **Nome da sessão**: Extraído do arquivo YAML (ex: `cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18`)
- **Vantagens**: Pode desconectar do SSH sem interromper o experimento
- **Comandos úteis**:
  - Listar sessões: `screen -ls`
  - Anexar à sessão: `screen -r <nome>`
  - Desanexar: `Ctrl+A`, depois `D`
  - Matar sessão: `screen -X -S <nome> quit`

---

### **2. run_experiments_queue.sh** (Fila de experimentos)
Executa múltiplos experimentos em sequência, aguardando GPU livre para cada um.

```bash
./run_experiments_queue.sh
```

**Como usar:**
1. Edite o arquivo `run_experiments_queue.sh`
2. Adicione seus experimentos na lista `EXPERIMENTS`
3. Configure `THRESHOLD` e `MEMORY_THRESHOLD`
4. Execute o script

**Exemplo de configuração:**
```bash
THRESHOLD=5.0
MEMORY_THRESHOLD=10.0
CHECK_INTERVAL=30

EXPERIMENTS=(
    "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml"
    "configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml"
    "configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml"
)
```

---

### **3. test_gpu_monitor.sh** (Teste)
Testa o sistema de monitoramento sem executar experimentos.

```bash
./test_gpu_monitor.sh
```

Mostra o status atual das GPUs e qual seria escolhida para executar um experimento.

---

### **4. check_gpu_processes.sh** (Diagnóstico simples)
Mostra processos ativos em cada GPU.

```bash
./check_gpu_processes.sh
```

---

### **5. diagnose_gpu_memory.sh** (Diagnóstico avançado) 🆕
Identifica processos mortos/idle e sugere comandos para liberar memória.

```bash
./diagnose_gpu_memory.sh
```

**O que ele detecta:**
- ✅ Processos mortos (zombies) - seguros de matar
- ✅ Processos idle (sleeping/suspended) - verificar antes de matar
- ✅ Processos ativos (running) - não matar
- ✅ Memória órfã (processo morto mas memória não liberada)

**Exemplo de output:**
```
=== Processo 1234 (python) ===
PID: 1234
Usuário: root
Estado: Z (zombie)
CPU: 0.0%
Memória GPU 0: 42916 MB
DIAGNÓSTICO: ⚠️  PROCESSO MORTO (zombie)
Comando para matar: sudo kill -9 1234
```

---

## � Sessões Screen

Por padrão, todos os experimentos são executados em **sessões screen** nomeadas automaticamente.

### **Por que usar Screen?**
- ✅ Experimento continua rodando mesmo se desconectar do SSH
- ✅ Pode anexar/desanexar à vontade sem interromper o treinamento
- ✅ Nome da sessão reflete os parâmetros do experimento (facilita gerenciamento)
- ✅ Múltiplos experimentos em paralelo com organização clara

### **Nome da Sessão**
O nome é extraído automaticamente do arquivo YAML:
```bash
# Comando:
python main.py train --exp-configs configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml

# Nome da sessão screen:
cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18
```

Se o comando não tiver `--exp-configs` ou `--config`, usa: `exp_gpu<ID>`

### **Comandos Úteis**

**Listar todas as sessões:**
```bash
screen -ls
```

**Anexar a uma sessão (ver output em tempo real):**
```bash
screen -r cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18

# Ou com tab-completion:
screen -r <TAB>
```

**Desanexar (dentro da sessão):**
```
Ctrl+A, depois D
```

**Matar uma sessão:**
```bash
screen -X -S cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18 quit
```

**Matar todas as sessões:**
```bash
screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit
```

### **Desabilitar Screen**
Se preferir execução direta (sem screen):
```bash
python3 auto_run_on_free_gpu.py \
  --command "..." \
  --no-screen
```

---
## 🎭 Arquitetura de 2 Níveis de Screen

O sistema usa **duas camadas de sessões screen** para máxima flexibilidade:

```
┌─────────────────────────────────────────────────────┐
│  Screen Nível 1: tagfex_gpu_monitor                │
│  (Monitoramento e gerenciamento de fila)            │
│                                                      │
│  ┌───────────────────────────────────────────────┐ │
│  │ Screen Nível 2: cifar100_10-10_baseline_...  │ │
│  │ (Experimento 1 rodando)                       │ │
│  └───────────────────────────────────────────────┘ │
│                                                      │
│  ┌───────────────────────────────────────────────┐ │
│  │ Screen Nível 2: cifar100_10-10_ant_beta0.5...│ │
│  │ (Experimento 2 rodando)                       │ │
│  └───────────────────────────────────────────────┘ │
│                                                      │
│  ┌───────────────────────────────────────────────┐ │
│  │ Screen Nível 2: imagenet100_10-10_baseline...│ │
│  │ (Aguardando GPU livre...)                     │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### **Por que 2 níveis?**

**Nível 1 (Monitor):**
- Executa o script `run_experiments_queue.sh`
- Monitora disponibilidade de GPUs
- Gerencia a fila de experimentos
- **Benefício:** Pode desconectar do SSH sem interromper o monitoramento

**Nível 2 (Experimentos):**
- Cada experimento roda em sua própria sessão screen
- Nome descritivo baseado no YAML
- **Benefício:** Pode anexar a qualquer experimento individualmente

### **Workflow Típico**

```bash
# 1. Iniciar monitoramento (cria sessão de nível 1)
./start_queue_monitor.sh

# 2. Ver progresso do monitoramento
screen -r tagfex_gpu_monitor

# 3. Dentro do monitor, ver lista de experimentos em execução
# (Ctrl+A, D para desanexar)

# 4. Anexar a um experimento específico (nível 2)
screen -r cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18

# 5. Ver saída em tempo real do treinamento
# (Ctrl+A, D para desanexar)

# 6. Desconectar do SSH - tudo continua rodando!
exit
```

### **Comandos para Gerenciar Ambos os Níveis**

```bash
# Ver todas as sessões (ambos os níveis)
screen -ls

# Output exemplo:
# 123456.tagfex_gpu_monitor                    (Detached)  ← Nível 1
# 123457.cifar100_10-10_baseline_local_resnet18 (Detached)  ← Nível 2
# 123458.cifar100_10-10_ant_beta0.5_margin0.5... (Detached)  ← Nível 2

# Matar apenas o monitor (experimentos continuam!)
screen -X -S tagfex_gpu_monitor quit

# Matar um experimento específico
screen -X -S cifar100_10-10_baseline_local_resnet18 quit

# Matar tudo (monitor + todos os experimentos)
screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit
```

---
## �🚦 Como Usar

### **Execução única:**
```bash
# Experimento com ANT
python3 auto_run_on_free_gpu.py \
  --config configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml \
  --threshold 5.0 \
  --interval 30
```

### **Execução com fila de experimentos (noturno):**

**Método Recomendado: Screen (pode desconectar do SSH)**
```bash
# 1. Configure os experimentos
nano run_experiments_queue.sh

# 2. Inicie monitoramento em sessão screen
./start_queue_monitor.sh

# 3. Desconecte do SSH quando quiser (monitoramento continua!)

# 4. Depois, para ver progresso:
screen -r tagfex_gpu_monitor
```

**Método Alternativo: nohup**
```bash
# 1. Configure os experimentos
nano run_experiments_queue.sh

# 2. Execute em background (nohup)
nohup ./run_experiments_queue.sh > queue_experiments.log 2>&1 &

# 3. Monitore o progresso
tail -f queue_experiments.log
```

### **Execução com threshold de memória:**
Útil quando há processos idle ocupando memória mas não usando GPU.

```bash
python3 auto_run_on_free_gpu.py \
  --config configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml \
  --threshold 5.0 \
  --memory-threshold 15.0
```

### **Execução paralela (dois jobs simultâneos):**
Use `--min-free-mb` para executar um experimento secundário enquanto outro está rodando.
O parâmetro verifica se há **N MB livres** — ao contrário de `--memory-threshold` (que bloqueia se % usada for alta), `--min-free-mb` garante que há espaço suficiente para mais um job.

```bash
# Primeiro job já rodando (ex: TinyImageNet ~6 GB)
# Segundo job aguarda até haver ≥8 GB livres
python3 auto_run_on_free_gpu.py \
  --config configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml \
  --threshold 100.0 \
  --min-free-mb 8000 \
  --no-screen --wait
```

---

## 📊 Logs e Monitoramento

### **Localização dos logs:**
```
logs/auto_experiments/
├── auto_exp_2024-01-15_14-30-00.log  # Log do experimento
├── auto_exp_2024-01-15_18-45-30.log
└── ...
```

### **Monitorar execução em tempo real:**
```bash
# Ver último log criado
tail -f logs/auto_experiments/$(ls -t logs/auto_experiments/ | head -n1)

# Ver logs do queue
tail -f queue_experiments.log
```

### **Verificar status das GPUs:**
```bash
watch -n 5 nvidia-smi
```

---

## 🐛 Troubleshooting

### **"nvidia-smi: command not found"**
```bash
# Verificar instalação NVIDIA
nvidia-smi --version

# Se não instalado, instalar drivers NVIDIA
sudo apt-get install nvidia-utils
```

### **GPUs com alta memória mas baixa utilização**

**Problema:** GPU mostra 0% util mas 87% memória ocupada.

**Causa:** Processos mortos/idle não liberaram memória CUDA.

**Diagnóstico completo:**
```bash
./diagnose_gpu_memory.sh
```

Este script identifica:
- ✅ Processos mortos (seguros de matar)
- ✅ Processos idle (verificar antes de matar)  
- ✅ Processos ativos (não matar)
- ✅ Comandos específicos para limpar

📖 **Guia completo:** [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md) explica em detalhes por que isso acontece e todas as soluções possíveis.

### **Script não detecta GPU livre**

**Verificar threshold:**
```bash
# Testar com threshold mais alto
python3 auto_run_on_free_gpu.py --config <CONFIG> --threshold 10.0
```

**Verificar se há GPUs disponíveis:**
```bash
./test_gpu_monitor.sh
```

### **Experimento não inicia após detectar GPU livre**

**Verificar permissões:**
```bash
# Tornar scripts executáveis
chmod +x auto_run_on_free_gpu.py
chmod +x run_experiments_queue.sh
```

**Verificar ambiente Python:**
```bash
# Ativar virtual environment
source .venv/bin/activate

# Verificar dependências
pip install -r requirements.txt
```

### **Process killed by system (OOM)**

**Problema:** Sistema mata processo por falta de memória.

**Soluções:**
1. Reduzir `batch_size` no YAML
2. Usar GPU com mais memória
3. Habilitar gradient checkpointing (se disponível)

---

## ⚙️ Configuração Avançada

### **Threshold de Utilização vs Memória**

**Quando usar cada um:**

| Cenário | Threshold Utilização | Threshold Memória | Min Free MB |
|---------|---------------------|-------------------|-------------|
| GPU completamente livre | `--threshold 1.0` | (não precisa) | (não precisa) |
| GPU com processos idle | `--threshold 5.0` | `--memory-threshold 10.0` | (não precisa) |
| GPU com modelo carregado mas não treinando | `--threshold 5.0` | `--memory-threshold 20.0` | (não precisa) |
| Execução paralela (outro job rodando) | `--threshold 100.0` | (não aplicar) | `--min-free-mb 8000` |

**Exemplo real (caso atual):**
```
GPU 0: 0%, 42938/49140 MB (87.4%)
```

Neste caso, usar:
```bash
python3 auto_run_on_free_gpu.py \
  --config configs/xxx.yaml \
  --threshold 5.0 \
  --memory-threshold 20.0
```

### **Ajustar intervalo de verificação**

```bash
# Verificar a cada 10 segundos (mais responsivo)
python3 auto_run_on_free_gpu.py --config <CONFIG> --interval 10

# Verificar a cada 60 segundos (menos overhead)
python3 auto_run_on_free_gpu.py --config <CONFIG> --interval 60
```

### **Forçar uso de GPU específica**

```bash
# Usar apenas GPU 1
CUDA_VISIBLE_DEVICES=1 python3 auto_run_on_free_gpu.py --config <CONFIG>
```

---

## 📈 Melhores Práticas

### **1. Executar experimentos overnight**
```bash
# Criar script com fila de experimentos
nano run_experiments_queue.sh

# Executar em background
nohup ./run_experiments_queue.sh > overnight.log 2>&1 &

# Deslogar (experimentos continuam rodando)
exit
```

### **2. Monitorar progresso remotamente**
```bash
# Via SSH
ssh user@machine "tail -f ~/TagFex_CVPR2025/logs/auto_experiments/*.log"

# Via tmux/screen
tmux new -s experiments
./run_experiments_queue.sh
# Ctrl+B, D para desanexar
```

### **3. Organizar experimentos por prioridade**
```bash
# No run_experiments_queue.sh, ordenar por prioridade
EXPERIMENTS=(
    # Alta prioridade (configs importantes)
    "configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml"
    
    # Média prioridade (ablations)
    "configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml"
    
    # Baixa prioridade (extras)
    "configs/all_in_one/imagenet100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml"
)
```

### **4. Backup de logs periodicamente**
```bash
# Criar backup semanal
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/auto_experiments/
```

---

## 🔍 Verificação Pré-Execução

Antes de executar experimentos overnight, execute:

```bash
# 1. Testar detecção de GPUs
./test_gpu_monitor.sh

# 2. Diagnosticar memória ocupada
./diagnose_gpu_memory.sh

# 3. Verificar configurações
cat run_experiments_queue.sh

# 4. Testar um experimento curto manualmente
python main.py --config configs/all_in_one/cifar100_10-10_baseline_local_resnet18.yaml

# 5. Se tudo OK, lançar fila
nohup ./run_experiments_queue.sh > queue.log 2>&1 &
```

---

## 📚 Arquivos Relacionados

- **README.md**: Documentação principal do projeto
- **GPU_MEMORY_GUIDE.md**: Guia detalhado sobre memória GPU vs utilização
- **main.py**: Script principal de treinamento
- **configs/all_in_one/**: Configurações de experimentos

---

## 🆘 Suporte

Se encontrar problemas:

1. **Verificar logs**: `tail -f logs/auto_experiments/*.log`
2. **Diagnosticar GPUs**: `./diagnose_gpu_memory.sh`
3. **Ler guia de memória**: [GPU_MEMORY_GUIDE.md](GPU_MEMORY_GUIDE.md)
4. **Verificar issues conhecidos** neste documento

---

**Última atualização:** 2024 (commit atual)
