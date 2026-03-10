# Sistema de Logging de Experimentos - TagFex

## Visão Geral

O sistema de experimentação automática agora inclui **logging detalhado de progresso** para controlar o ciclo de vida completo dos experimentos.

## Confirmação: Uso de Screen

✅ **SIM, todos os experimentos rodam em sessões screen!**

- Cada experimento é lançado com `screen -dmS <nome>` (modo detached)
- Nome da sessão baseado no arquivo de configuração
- Permite desconectar SSH sem interromper experimentos
- Ambiente virtual propagado automaticamente para as sessões

## Sistema de Logging

### 1. Log de Progresso Centralizado

> **Multi‑GPU:** o script de fila aceita uma variável `GPUS` definida no topo. Se
> `GPUS>1` a execução usa `torchrun --nproc_per_node=$GPUS` para lançar o treino
> com `main.py` (que já suporta dist). Caso contrário, roda em GPU única.

### 1. Log de Progresso Centralizado

Arquivo: `logs/auto_experiments/queue_progress.log`

Registra todos os eventos importantes:
- 🚀 Início da fila de experimentos
- 📋 Enfileiramento de cada experimento
- 🔍 Detecção de GPU livre (com stats de utilização e memória)
- ✅ Disparo de experimento (com nome da sessão screen)
- 📺 Sessões screen criadas
- 🏁 Conclusão do enfileiramento
- ✅ Sessões encerradas (via monitor)

### 2. Log Completo do Console

Arquivo: `logs/auto_experiments/queue_console_YYYYMMDD_HHMMSS.log`

**NOVO!** Captura toda a saída do console, incluindo:
- Todas as mensagens do sistema
- Output do auto_run_on_free_gpu.py
- Códigos de cor preservados
- Timestamps de início/fim
- Erros e avisos completos

Este log é criado automaticamente a cada execução do `run_experiments_queue.sh`.

### 3. Logs Individuais por Experimento

Diretório: `logs/auto_experiments/`

Formato: `auto_gpu{N}_{timestamp}.log`

Cada experimento tem seu próprio log contendo:
- Timestamp de início
- GPU alocada
- Comando executado
- Nome da sessão screen
- Output completo do experimento

### 4. Timestamps em Todos os Eventos

Formato: `[YYYY-MM-DD HH:MM:SS] evento`

Exemplo:
```
[2026-02-25 22:05:30] 🚀 Iniciando fila de experimentos
[2026-02-25 22:05:30] 📋 Enfileirando: ImageNet-100 50-10 Baseline
[2026-02-25 22:05:35] ✅ GPU 0 disponível!
[2026-02-25 22:05:35]    Utilização: 0.0%
[2026-02-25 22:05:35]    Memória: 89.3%
[2026-02-25 22:05:36] ✅ Experimento disparado: ImageNet-100 50-10 Baseline
[2026-02-25 22:05:36]    📺 Sessão screen: imagenet100_50-10_baseline_local_resnet18
```

## Scripts Principais

### 1. start_queue_monitor.sh
Inicia o sistema em sessão screen principal

### 2. run_experiments_queue.sh
Gerencia a fila de experimentos com logging:
- Cria log de progresso
- Registra cada enfileiramento
- Captura nomes de sessões screen
- Conta sessões ativas ao final

### 3. auto_run_on_free_gpu.py
Monitora GPUs e dispara experimentos:
- Loga detecção de GPU livre com stats detalhados
- Registra timestamp de disparo
- Salva nome da sessão screen nos logs
- **Novo**: parâmetro `--gpus N` permite aguardar N GPUs livres e repassa para o comando
  (usado por `run_experiments_queue.sh` com a variável GPUS).

### 4. monitor_screen_sessions.sh (NOVO)
Monitor contínuo de sessões screen:
- Detecta novas sessões criadas
- Loga quando sessões terminam
- Atualiza status em tempo real
- Integrado ao log de progresso

## Como Usar

### Iniciar Fila de Experimentos
```bash
./start_queue_monitor.sh
```

### Monitorar Progresso em Tempo Real
```bash
# Método 1: Log de progresso
tail -f logs/auto_experiments/queue_progress.log

# Método 2: Monitor de sessões screen (em terminal separado)
./monitor_screen_sessions.sh
```

### Verificar Status das GPUs
```bash
gpustat --watch
```

### Listar Sessões Screen
```bash
screen -ls
```

### Anexar a um Experimento Específico  
```bash
screen -r <nome_da_sessao>
# Exemplo: screen -r imagenet100_10-10_ant_beta0.5_margin0.5_local_resnet18
```

### Verificar Todos os Logs
```bash
ls -lh logs/auto_experiments/
```

## Eventos Logados

| Evento | Símbolo | Descrição |
|--------|---------|-----------|
| Início da fila | 🚀 | Queue iniciada |
| Enfileirando | 📋 | Experimento adicionado à fila |
| GPU livre | ✅ | GPU disponível detectada (+ stats) |
| Experimento disparado | ✅ | Experimento iniciado com sucesso |
| Sessão screen | 📺 | Nome da sessão criada |
| Sessão encerrada | ✅ | Experimento concluído |
| Nova sessão | 🆕 | Nova sessão detectada pelo monitor |
| Erro | ❌ | Falha ao disparar experimento |
| Conclusão | 🏁 | Script de enfileiramento concluído |

## Estrutura de Arquivos de Log

```
logs/auto_experiments/
├── queue_progress.log                    # Log centralizado de progresso
├── queue_console_20260225_220530.log    # Log completo do console (com timestamp)
├── queue_console_20260225_230145.log    # Outro log de console (execução diferente)
├── auto_gpu0_20260225_220530.log        # Log do experimento na GPU 0
├── auto_gpu1_20260225_220545.log        # Log do experimento na GPU 1
└── ...
```

**Nota:** Um novo arquivo `queue_console_*.log` é criado a cada execução do sistema.

## Monitoramento Contínuo

Para monitoramento completo durante a execução:

**Terminal 1:** Progresso da fila (eventos importantes)
```bash
tail -f logs/auto_experiments/queue_progress.log
```

**Terminal 2:** Console completo (toda a saída)
```bash
tail -f logs/auto_experiments/queue_console_*.log
# Ou para o mais recente:
tail -f $(ls -t logs/auto_experiments/queue_console_*.log | head -1)
```

**Terminal 3:** GPUs
```bash
gpustat --watch
```

**Terminal 4:** Monitor de sessões (opcional)
```bash
./monitor_screen_sessions.sh 60
```

## Troubleshooting

### Ver experimentos em execução
```bash
screen -ls
ps aux | grep "main.py train"
```

### Verificar último status logado
```bash
tail -20 logs/auto_experiments/queue_progress.log
```

### Anexar à sessão principal do queue
```bash
screen -r tagfex_gpu_monitor
```

### Matar todas as sessões (emergência)
```bash
screen -ls | grep Detached | cut -d. -f1 | xargs -I {} screen -X -S {} quit
```
