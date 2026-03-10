#!/usr/bin/env python3
"""
Auto GPU Experiment Launcher
Monitora GPUs e dispara experimento automaticamente quando uma ficar disponível.

Uso:
    python3 auto_run_on_free_gpu.py \
        --command "python main.py train --exp-configs configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \
        --threshold 95.0 \
        --interval 30 \
        --log-dir ./logs/auto_exp
"""

import subprocess
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
import signal
import os
import re


class GPUMonitor:
    def __init__(
        self, threshold=95.0, memory_threshold=None, interval=30, verbose=True
    ):
        """
        Args:
            threshold: Threshold de utilização GPU (%) para considerar livre
            memory_threshold: Threshold de memória ocupada (%) para considerar livre (None = ignorar)
            interval: Intervalo entre checagens (segundos)
            verbose: Se True, mostra mensagens detalhadas
        """
        self.threshold = threshold
        self.memory_threshold = memory_threshold
        self.interval = interval
        self.verbose = verbose
        self.running = True

        # Registrar handler para Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handler para Ctrl+C"""
        print("\n\n⚠️  Interrompido pelo usuário. Saindo...")
        self.running = False
        sys.exit(0)

    def _log(self, message, force=False):
        """Log com timestamp"""
        if self.verbose or force:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

    def get_gpu_status(self):
        """
        Retorna status de todas as GPUs.

        Returns:
            list: Lista de dicts com {id, util, memory_used, memory_total, free}
        """
        try:
            # Usar nvidia-smi para obter status
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                gpu_id = int(parts[0])
                util = float(parts[1])
                mem_used = float(parts[2])
                mem_total = float(parts[3])
                mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

                # GPU é livre se:
                # 1. Utilização < threshold (se threshold < 100) E/OU
                # 2. Memória < memory_threshold (se especificado)
                # Se threshold >= 100, ignora utilização (monitora apenas memória)
                if self.threshold >= 100:
                    # Modo: só memória
                    if self.memory_threshold is not None:
                        is_free = mem_percent < self.memory_threshold
                    else:
                        # Threshold >= 100 e sem memory_threshold = sempre livre
                        is_free = True
                else:
                    # Modo: utilização + memória (se especificado)
                    is_free = util < self.threshold
                    if self.memory_threshold is not None:
                        is_free = is_free and (mem_percent < self.memory_threshold)

                gpus.append(
                    {
                        "id": gpu_id,
                        "util": util,
                        "memory_used": mem_used,
                        "memory_total": mem_total,
                        "memory_percent": mem_percent,
                        "free": is_free,
                    }
                )

            return gpus

        except subprocess.CalledProcessError as e:
            self._log(f"❌ Erro ao executar nvidia-smi: {e}", force=True)
            return []
        except Exception as e:
            self._log(f"❌ Erro ao obter status das GPUs: {e}", force=True)
            return []

    def find_free_gpus(self, count=1):
        """
        Encontra um conjunto de GPUs livres.

        Args:
            count (int): número mínimo de GPUs livres necessários.

        Returns:
            list: lista de IDs das GPUs livres (tamanho >= count) ou [] se nenhuma combinação disponível
        """
        gpus = self.get_gpu_status()

        if not gpus or count <= 0:
            return []

        free_ids = [gpu["id"] for gpu in gpus if gpu["free"]]
        if len(free_ids) >= count:
            # somente retornar as 'count' primeiras GPUs livres para consistência
            return free_ids[:count]
        else:
            return []

    def display_status(self, gpus):
        """Exibe status das GPUs de forma formatada"""
        self._log("=" * 70)
        self._log("GPU Status:")
        for gpu in gpus:
            status_icon = "✅" if gpu["free"] else "🔴"
            self._log(
                f"  {status_icon} GPU {gpu['id']}: "
                f"{gpu['util']:5.1f}% util | "
                f"{gpu['memory_used']:7.0f} / {gpu['memory_total']:.0f} MB "
                f"({gpu['memory_percent']:4.1f}%)"
            )
        self._log("=" * 70)

    def wait_for_free_gpu(self, count=1):
        """
        Monitora GPUs até um conjunto com pelo menos `count` unidades ficar disponível.

        Args:
            count (int): número de GPUs que precisam estar livres.

        Returns:
            list: IDs das GPUs selecionadas ou lista vazia se monitoramento interrompido
        """
        # Mensagem de critério baseada no modo
        if self.threshold >= 100:
            if self.memory_threshold is not None:
                threshold_msg = f"mem < {self.memory_threshold}% (modo: apenas memória)"
            else:
                threshold_msg = "sempre livre (threshold >= 100, sem memory_threshold)"
        else:
            threshold_msg = f"util < {self.threshold}%"
            if self.memory_threshold is not None:
                threshold_msg += f" AND mem < {self.memory_threshold}%"

        self._log(
            f"🔍 Monitorando GPUs ({threshold_msg}, intervalo: {self.interval}s)",
            force=True,
        )
        self._log("   Pressione Ctrl+C para interromper\n", force=True)

        iteration = 0
        while self.running:
            gpus = self.get_gpu_status()

            if not gpus:
                self._log(
                    "⚠️  Nenhuma GPU encontrada. Tentando novamente...", force=True
                )
                time.sleep(self.interval)
                continue

            # Exibir status a cada 10 iterações ou na primeira
            if iteration % 10 == 0:
                self.display_status(gpus)

            # Verificar se há GPUs livres suficientes
            free_ids = self.find_free_gpus(count)
            if free_ids:
                self._log(f"\n✅ GPUs disponíveis: {free_ids}", force=True)
                for gid in free_ids:
                    gpu = next((g for g in gpus if g["id"] == gid), None)
                    if gpu:
                        self._log(
                            f"   GPU {gid} util: {gpu['util']:.1f}% mem: {gpu['memory_percent']:.1f}%",
                            force=True,
                        )
                return free_ids

            # Aguardar próxima checagem
            if iteration % 10 != 0:
                self._log(
                    f"⏳ Não há {count} GPUs livres. Aguardando {self.interval}s..."
                )

            iteration += 1
            time.sleep(self.interval)

        return []


def extract_config_name(command):
    """
    Extrai o nome do arquivo de configuração do comando.

    Args:
        command: Comando completo

    Returns:
        str: Nome base do arquivo de config (sem extensão) ou None
    """
    # Procurar por --exp-configs ou --config seguido por um caminho
    patterns = [
        r"--exp-configs\s+([^\s]+)",
        r"--config\s+([^\s]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            config_path = match.group(1)
            # Extrair apenas o nome do arquivo sem extensão
            config_name = Path(config_path).stem
            return config_name

    return None


def run_experiment(command, gpu_ids, log_dir=None, use_screen=True):
    """
    Executa experimento nas GPUs especificadas.

    Args:
        command: Comando a executar
        gpu_ids: inteiro ou lista de IDs de GPUs
        log_dir: Diretório para logs (opcional)
        use_screen: Se True, executa dentro de uma sessão screen (default: True)

    Returns:
        subprocess.Popen or None: Processo do experimento (None se usar screen)
    """
    # normalizar gpu_ids para lista
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]

    # Configurar variável de ambiente
    env = os.environ.copy()
    # caso o comando envolva torchrun (multi-gpu), não restrinja CUDA_VISIBLE_DEVICES
    if "torchrun" not in command:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

    # Extrair nome da configuração para nome da sessão screen
    config_name = extract_config_name(command)
    if len(gpu_ids) == 1:
        screen_session_name = config_name if config_name else f"exp_gpu{gpu_ids[0]}"
    else:
        first = gpu_ids[0]
        screen_session_name = (
            config_name if config_name else f"exp_gpu{first}_x{len(gpu_ids)}"
        )

    print(f"\n{'='*70}")
    if len(gpu_ids) == 1:
        print(f"🚀 Disparando experimento na GPU {gpu_ids[0]}")
    else:
        print(f"🚀 Disparando experimento nas GPUs {gpu_ids}")
    print(f"{'='*70}")
    print(f"Comando: {command}")
    if len(gpu_ids) == 1:
        print(f"GPU: {gpu_ids[0]}")
    else:
        print(f"GPUs: {gpu_ids}")
    if use_screen:
        print(f"Screen Session: {screen_session_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_str = (
            "".join(str(i) for i in gpu_ids)
            if len(gpu_ids) == 1
            else "x".join(str(i) for i in gpu_ids)
        )
        log_file = log_dir / f"auto_gpu{id_str}_{timestamp}.log"
        print(f"Log: {log_file}")

        with open(log_file, "w") as f:
            f.write(f"Auto-launched experiment\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            if len(gpu_ids) == 1:
                f.write(f"GPU: {gpu_ids[0]}\n")
            else:
                f.write(f"GPUs: {gpu_ids}\n")

        # Abrir log em modo append
        log_handle = open(log_file, "a")
        stdout = log_handle
        stderr = subprocess.STDOUT
    else:
        stdout = None
        stderr = None

    print(f"{'='*70}\n")

    # Executar comando
    if use_screen:
        # Verificar se screen está instalado
        try:
            subprocess.run(["screen", "-v"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  'screen' não encontrado. Instalando...")
            try:
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "screen"], check=True
                )
                print("✅ screen instalado com sucesso!\n")
            except subprocess.CalledProcessError:
                print("❌ Erro ao instalar screen. Executando sem screen.\n")
                use_screen = False

    if use_screen:
        # Construir comando screen
        # screen -dmS <nome> bash -c "export CUDA_VISIBLE_DEVICES=X && comando"

        # Detectar se há ambiente virtual ativo
        venv_activate = ""
        if "VIRTUAL_ENV" in os.environ:
            venv_path = os.environ["VIRTUAL_ENV"]
            venv_activate = f"source {venv_path}/bin/activate && "

        visible = ",".join(str(i) for i in gpu_ids)
        screen_command = (
            f"screen -dmS {screen_session_name} bash -c "
            f'"{venv_activate}export CUDA_VISIBLE_DEVICES={visible} && {command}'
        )

        if log_dir:
            # Redirecionar output para log file
            screen_command += f" 2>&1 | tee -a {log_file}"

        screen_command += '"'

        # Executar screen
        subprocess.run(screen_command, shell=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"\n[{timestamp}] ✅ Experimento iniciado na sessão screen: {screen_session_name}"
        )
        if len(gpu_ids) == 1:
            print(f"[{timestamp}] 🔌 GPU {gpu_ids[0]} alocada")
        else:
            print(f"[{timestamp}] 🔌 GPUs {gpu_ids} alocadas")

        if log_dir:
            print(f"[{timestamp}] 📄 Log: {log_file}")

        print(f"   screen -ls")
        print(f"\n🔌 Para desanexar (dentro da sessão):")
        print(f"   Ctrl+A, depois D")
        print(f"\n❌ Para matar a sessão:")
        print(f"   screen -X -S {screen_session_name} quit")

        return None
    else:
        # Executar comando sem screen (modo original)
        process = subprocess.Popen(
            command, shell=True, env=env, stdout=stdout, stderr=stderr
        )
        return process


def main():
    parser = argparse.ArgumentParser(
        description="Monitora GPUs e dispara experimento automaticamente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Experimento CIFAR-100 com ANT
  python auto_run_on_free_gpu.py \\
      --command "python main.py train --exp-configs configs/all_in_one/cifar100_10-10_ant_beta0.5_margin0.5_local_resnet18.yaml"
  
  # Experimento ImageNet100 usando duas GPUs (usará torchrun internamente)
  python auto_run_on_free_gpu.py \\
      --command "torchrun --nproc_per_node=2 python main.py train --exp-configs configs/all_in_one/imagenet100_10-10_baseline_local_resnet18.yaml" \\
      --gpus 2 \\
      --threshold 5.0 \\
      --interval 15
  
  # Sem monitoramento contínuo após disparar
  python auto_run_on_free_gpu.py \\
      --command "python main.py train --exp-configs configs/all_in_one/cifar100_50-10_ant_beta0.5_margin0.5_local_resnet18.yaml" \\
      --no-wait
        """,
    )

    parser.add_argument(
        "--command",
        "-c",
        required=True,
        help="Comando a executar quando GPU ficar disponível",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=1.0,
        help="Threshold de utilização GPU (%%) para considerar livre (default: 1.0)",
    )

    parser.add_argument(
        "--gpus",
        "-g",
        type=int,
        default=1,
        help="Número de GPUs necessárias simultaneamente (default: 1)",
    )

    parser.add_argument(
        "--memory-threshold",
        "-m",
        type=float,
        default=None,
        help="Threshold de memória ocupada (%%) para considerar livre (default: None = ignorar memória)",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Intervalo entre checagens em segundos (default: 30)",
    )

    parser.add_argument(
        "--log-dir",
        "-l",
        type=str,
        default=None,
        help="Diretório para salvar logs do experimento",
    )

    parser.add_argument(
        "--wait",
        "-w",
        action="store_true",
        default=True,
        help="Aguardar conclusão do experimento (default: True)",
    )

    parser.add_argument(
        "--no-wait",
        action="store_false",
        dest="wait",
        help="Não aguardar conclusão (disparar e sair)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Modo silencioso (menos mensagens)"
    )

    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="Não usar sessão screen (executa diretamente)",
    )

    args = parser.parse_args()

    # Criar monitor
    monitor = GPUMonitor(
        threshold=args.threshold,
        memory_threshold=args.memory_threshold,
        interval=args.interval,
        verbose=not args.quiet,
    )

    # Aguardar GPUs disponíveis
    required = args.gpus
    free_gpus = monitor.wait_for_free_gpu(count=required)

    if not free_gpus:
        print("❌ Monitoramento interrompido ou não houve GPUs livres suficientes.")
        sys.exit(1)

    # Disparar experimento
    use_screen = not args.no_screen
    process = run_experiment(args.command, free_gpus, args.log_dir, use_screen)

    # exibir quais GPUs foram alocadas se não estiver em screen
    allocated_str = ",".join(str(x) for x in free_gpus)
    print(f"\n🚀 Comando lançado com GPUs: {allocated_str}")

    # Se usar screen, não há processo para aguardar
    if use_screen:
        allocated = ",".join(str(x) for x in free_gpus)
        print(f"\n✅ Experimento disparado em sessão screen nas GPUs {allocated}")
        sys.exit(0)

    if args.wait:
        print(f"\n⏳ Aguardando conclusão do experimento (PID: {process.pid})...")
        print("   Pressione Ctrl+C para desanexar (experimento continuará rodando)\n")

        try:
            return_code = process.wait()

            if return_code == 0:
                print(f"\n✅ Experimento concluído com sucesso!")
            else:
                print(f"\n⚠️  Experimento finalizado com código: {return_code}")

            sys.exit(return_code)

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Desanexado do experimento (PID: {process.pid})")
            print(f"   O experimento continuará rodando em background.")
            print(f"   Use 'ps aux | grep {process.pid}' para verificar status")
            sys.exit(0)
    else:
        print(f"\n✅ Experimento disparado (PID: {process.pid})")
        print(f"   Rodando em background na GPU {free_gpu}")
        sys.exit(0)


if __name__ == "__main__":
    main()
