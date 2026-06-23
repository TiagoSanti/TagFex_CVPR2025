# 🔬 GPU Memory vs Utilization - Guia Completo

## 🤔 O Problema: 0% Utilização mas 87% Memória?

Esse é um dos cenários **mais comuns** e **mais confusos** ao trabalhar com GPUs. Vamos entender!

---

## 📊 Diferença entre Utilização e Memória

### **Utilização GPU (Compute Utilization)**
- Mede quanto os **cores CUDA estão computando**
- Indica se a GPU está **processando** cálculos ativamente
- `0%` = GPU ociosa (não está executando kernels CUDA)
- `100%` = GPU saturada (cores CUDA totalmente ocupados)

### **Memória GPU (Memory Usage)**
- Mede quanto da **VRAM está alocada**
- Indica quanto espaço está **reservado** (não necessariamente em uso)
- `87%` = 42GB de 49GB estão alocados (dados/modelos carregados)
- Memória alocada ≠ memória sendo processada

### **Analogia:**
```
Memória = Estante de livros
  └─ Livros ocupam espaço mesmo se você não está lendo

Utilização = Pessoa lendo livros
  └─ Se ninguém está lendo, utilização = 0%
      MAS os livros continuam na estante (memória ocupada)!
```

---

## 🐛 Causas Comuns de Alta Memória + Baixa Utilização

### **1. Processo Morto/Zombie** 🧟 (MAIS COMUM)

**O que aconteceu:**
- Treinamento crashou/foi interrompido (Ctrl+C, kill, OOM)
- Python terminou mas contexto CUDA não foi liberado
- Memória alocada ficou "presa" na GPU

**Como identificar:**
```bash
nvidia-smi  # Mostra PID do processo
ps aux | grep <PID>  # Verificar se processo existe

# Se não aparecer = processo morto, memória órfã
```

**Como resolver:**
```bash
# Opção 1: Matar processo específico
kill -9 <PID>

# Opção 2: Limpar GPU inteira (cuidado!)
sudo fuser -k /dev/nvidia0  # GPU 0
sudo fuser -k /dev/nvidia1  # GPU 1

# Opção 3: Reiniciar driver NVIDIA
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia

# Opção 4: Reiniciar máquina (mais fácil)
sudo reboot
```

---

### **2. Processo Idle/Suspenso** 😴

**O que aconteceu:**
- Script carregou modelo/dados mas está **aguardando**
- Pausado em `input()`, `time.sleep()`, debug breakpoint
- Jupyter notebook com células executadas mas kernel ativo

**Como identificar:**
```bash
ps -p <PID> -o state,cmd,%cpu
# State=S (sleeping), %cpu < 0.1% = idle
```

**Como resolver:**
```bash
# Se for seu processo:
# CUIDADO: Verificar se não é um pause intencional!

# Ver o que está fazendo
strace -p <PID>  # Ver syscalls (onde está travado)

# Se certeza que pode matar:
kill <PID>  # Tentar término gracioso
kill -9 <PID>  # Forçar se não responder
```

**Casos legítimos de idle:**
- Debug com breakpoint ativo
- Script esperando input do usuário
- Servidor esperando requisições (ex: TensorServing)
- Pre-carregamento de modelos para inferência rápida

---

### **3. Jupyter Notebooks** 📓

**O que aconteceu:**
- Célula executou, carregou modelo na GPU
- Kernel ainda está ativo (memória não foi liberada)
- Mesmo fechando o browser, kernel continua rodando

**Como identificar:**
```bash
ps aux | grep jupyter
# Verá: jupyter-notebook ou jupyter-lab
```

**Como resolver:**
```bash
# No Jupyter: Kernel → Restart → Yes

# Ou via terminal:
jupyter notebook list  # Ver notebooks ativos
jupyter notebook stop 8888  # Parar por porta

# Matar todos Jupyter:
pkill -f jupyter
```

---

### **4. Multi-GPU Mal Configurado** 🎯

**O que aconteceu:**
- Script alocou memória em **todas as GPUs**
- Mas só está usando **uma** (ex: GPU 0)
- GPUs 1, 2, 3 têm memória ocupada mas 0% utilização

**Como identificar:**
```bash
nvidia-smi
# GPU 0: 95% util, 40GB memory  ← Trabalhando
# GPU 1: 0% util, 10GB memory   ← Memória alocada mas não usada
# GPU 2: 0% util, 10GB memory   ← Memória alocada mas não usada
```

**Causa no código:**
```python
# ❌ ERRADO: Aloca em todas as GPUs visíveis
import torch
model = nn.DataParallel(model)  # Sem especificar device_ids

# ✅ CORRETO: Especificar GPUs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Só ver GPU 0

# Ou:
model = nn.DataParallel(model, device_ids=[0])
```

**Como resolver:**
```bash
# Fixar antes de executar script:
CUDA_VISIBLE_DEVICES=0 python train.py
```

---

### **5. Memory Leak** 💧

**O que aconteceu:**
- Bug no código acumula tensores na GPU sem liberar
- Gradientes não são zerados entre iterações
- Referências circulares impedem garbage collection

**Como identificar:**
```bash
# Memória aumenta gradualmente durante execução
watch -n 1 nvidia-smi  # Observar memória crescendo
```

**Causas comuns:**
```python
# ❌ Esqueceu de zerar gradientes
for batch in dataloader:
    loss.backward()
    # optimizer.step()  ← SEM .zero_grad()!

# ❌ Acumulando histórico de tensores
losses = []
for batch in dataloader:
    loss = criterion(output, target)
    losses.append(loss)  # ← Mantém grafo computacional!
    # Correto: losses.append(loss.item())

# ❌ Detaching incorreto
hidden = model(x)
hidden = hidden.detach()  # ← Ainda mantém memória!
# Correto: hidden = hidden.detach().cpu()
```

---

## 🛠️ Ferramentas de Diagnóstico

### **1. Script de Diagnóstico Completo** (Novo!)

```bash
./diagnose_gpu_memory.sh
```

**O que faz:**
- ✅ Lista todos os processos em cada GPU
- ✅ Identifica processos mortos/zombie/idle
- ✅ Mostra tempo de execução e CPU usage
- ✅ **Sugere ações específicas** (matar, verificar, ignorar)
- ✅ Gera comandos prontos para copiar/colar

**Exemplo de output:**
```
📊 PID 12345 | 42916 MiB | python train.py
   👤 Usuário: root
   ⏱️  Tempo ativo: 3-07:22:15
   🖥️  CPU usage: 0.0%
   📝 Comando: python train.py --epochs 100
   🔍 Status: 😴 IDLE/SLEEP
   💡 AÇÃO: Processo idle - VERIFICAR antes de matar
   → Pode estar em debug, aguardando input...
   → Se certeza que pode matar: kill 12345
```

### **2. Check GPU Processes** (Simples)

```bash
./check_gpu_processes.sh
```

Versão mais simples, foco em identificar processos idle.

### **3. Comandos Úteis**

```bash
# Ver processos GPU com detalhes
nvidia-smi

# Ver apenas memória por GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Monitorar em tempo real
watch -n 1 nvidia-smi

# GPUs disponíveis (bonito)
gpustat
gpustat --watch

# Processos usando GPU específica
fuser -v /dev/nvidia0

# Ver comando completo de um PID
ps -p <PID> -o cmd=

# Ver estado do processo
ps -p <PID> -o state,cpu,cmd
```

---

## 🚀 Soluções por Cenário

### **Cenário 1: Você é o dono do processo**

```bash
# 1. Diagnosticar
./diagnose_gpu_memory.sh

# 2. Se processo morto → Matar
kill -9 <PID>

# 3. Se processo idle → Verificar primeiro
strace -p <PID>  # Ver onde está travado
# Se pode matar:
kill <PID>
```

### **Cenário 2: Processo de outro usuário (shared server)**

```bash
# 1. Verificar quem é o dono
nvidia-smi
ps -p <PID> -o user,cmd

# 2. Verificar há quanto tempo está rodando
ps -p <PID> -o etime,cmd

# 3. Se > 3 dias sem atividade → Avisar usuário
# Se não responder → Admin pode matar
```

### **Cenário 3: Não pode matar (servidor compartilhado)**

**Use `--memory-threshold` no auto-launcher:**

```bash
# Só dispara se GPU tiver < 20% memória
python3 auto_run_on_free_gpu.py \
    --command "python main.py train ..." \
    --threshold 1.0 \
    --memory-threshold 20.0
```

Ou configure no `run_experiments_queue.sh`:
```bash
# Linha 11
MEMORY_THRESHOLD="--memory-threshold 20.0"
```

### **Cenário 4: Limpar tudo e começar do zero**

```bash
# ⚠️ CUIDADO: Isso mata TODOS os processos CUDA

# Opção 1: Por GPU
sudo fuser -k /dev/nvidia0  # Limpar GPU 0
sudo fuser -k /dev/nvidia1  # Limpar GPU 1

# Opção 2: Reiniciar driver
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia

# Opção 3: Reiniciar máquina (mais garantido)
sudo reboot
```

---

## 📋 Checklist de Verificação

Antes de matar processos, verificar:

- [ ] **É meu processo?** `ps -p <PID> -o user`
- [ ] **Está há quanto tempo?** `ps -p <PID> -o etime`
- [ ] **Está realmente idle?** `ps -p <PID> -o %cpu`
- [ ] **Tem checkpoint recente?** (se for seu treinamento)
- [ ] **É um servidor legítimo?** (TensorServing, FastAPI, etc.)
- [ ] **Avisei o dono?** (se não for meu)

✅ **Só matar se:**
- Processo está morto (não aparece em `ps`)
- É seu processo e você sabe que pode terminar
- É zombie/defunct (state=Z)
- Administrador autorizou (servidor compartilhado)

❌ **NÃO matar se:**
- Processo de outro usuário sem permissão
- Servidor em produção (mesmo que idle)
- Debug ativo (pode estar investigando bug)
- Não tem certeza do que é

---

## 💡 Melhores Práticas

### **Ao executar seus experimentos:**

```bash
# 1. Sempre fixar GPU específica
CUDA_VISIBLE_DEVICES=0 python train.py

# 2. Usar try-finally para liberar recursos
try:
    model = load_model()
    train()
finally:
    torch.cuda.empty_cache()
    del model

# 3. Limpar cache periodicamente
import torch
torch.cuda.empty_cache()

# 4. Usar context managers
with torch.cuda.device(0):
    # Código aqui
    pass
# Memória liberada automaticamente ao sair
```

### **Ao sair de experimentos:**

```bash
# Python: Ctrl+C → Ctrl+C novamente (force quit)
# Isso força limpeza de contexto CUDA

# Ou adicionar signal handler no código:
import signal
def cleanup(sig, frame):
    torch.cuda.empty_cache()
    sys.exit(0)
signal.signal(signal.SIGINT, cleanup)
```

---

## 🎯 Para seu caso específico (GPUs com 87% memória)

Execute o diagnóstico:
```bash
./diagnose_gpu_memory.sh
```

Ele vai te dizer exatamente:
- Quais processos estão causando isso
- Se são processos mortos/idle/ativos
- Comandos específicos para resolver
- Se é seguro usar a GPU mesmo assim

Se não puder limpar a memória, use:
```bash
python3 auto_run_on_free_gpu.py \
    --command "python main.py train --exp-configs configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
    --threshold 1.0 \
    --memory-threshold 30.0
```

Isso só vai usar GPUs com < 30% de memória ocupada.
