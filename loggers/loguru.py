import os
from pathlib import Path
from loguru import logger


class LoguruLogger:
    def __init__(
        self, configs, disable_output_files=False, tqdm_out=False
    ) -> None:  # loguru_configs
        super().__init__()
        self.configs = configs
        self.disable_output_files = disable_output_files

        self.logger = logger
        if tqdm_out:
            level = 0 if self.configs.get("debug") else "INFO"
            self._set_tqdm(level=level)
        self._set_logfiles()

    def _set_logfiles(self):
        if self.disable_output_files:
            return

        log_dir = self.configs.get("log_dir")
        if log_dir is not None:
            log_dir = Path(log_dir)

            # Create log directory if it doesn't exist
            log_dir.mkdir(parents=True, exist_ok=True)

            rank = int(os.environ.get("RANK", 0))
            logfile_prefix = self.configs.get("output_file_prefix", "loguru")

            filename = f"{logfile_prefix}_stdlog{rank}.log"
            self.logger.add(
                log_dir / filename,
                level="INFO",
                filter=lambda r: not r["extra"].get("sim_debug", False),
            )

            if self.configs.get("debug"):
                debug_filename = f"{logfile_prefix}_debuglog{rank}.log"
                self.logger.add(log_dir / debug_filename, level=0)

            # Always create a debug log for similarity matrices and loss components
            debug_log_filename = f"{logfile_prefix}_debug{rank}.log"
            self.logger.add(
                log_dir / debug_log_filename,
                level="DEBUG",
                filter=lambda record: "Matrix" in record["message"]
                or "Loss components" in record["message"]
                or "ANT distance stats" in record["message"]
                or "ANT flattening" in record["message"]
                or "Contrastive stats" in record["message"],
            )

            if rank == 0:
                performance_logname = f"{logfile_prefix}_gistlog.log"
                self.logger.add(log_dir / performance_logname, level="SUCCESS")

    def _set_tqdm(self, level="INFO"):
        from tqdm import tqdm
        from functools import partial

        self.logger.remove()
        self.logger.add(
            partial(tqdm.write, end=""),
            colorize=True,
            level=level,
            filter=lambda r: not r["extra"].get("sim_debug", False),
        )

    def log_loss_components(
        self, components, prefix="", task=None, epoch=None, batch=None
    ):
        """
        Log individual loss components with task/epoch/batch context.

        Args:
            components: dict - Dictionary of loss component names to values
            prefix: str - Optional prefix for the component names
            task: int - Current task number
            epoch: int - Current epoch number
            batch: int - Current batch number
        """
        if prefix:
            prefix = f"{prefix}_"

        # Build context string
        context_parts = []
        if task is not None:
            context_parts.append(f"T{task}")
        if epoch is not None:
            context_parts.append(f"E{epoch}")
        if batch is not None:
            context_parts.append(f"B{batch}")

        context = f"[{' '.join(context_parts)}] " if context_parts else ""

        log_msg = " | ".join([f"{prefix}{k}: {v:.4f}" for k, v in components.items()])
        self.logger.debug(f"{context}Loss components: {log_msg}")

    def log_contrastive_stats(self, stats, task=None, epoch=None, batch=None):
        """
        Log general contrastive learning statistics (applicable to all experiments).

        Args:
            stats: dict - Dictionary containing basic similarity statistics:
                - pos_mean, pos_std: Positive pair similarities
                - neg_mean, neg_std: Negative pair similarities
                - current_gap: Natural gap (pos_mean - neg_mean)
            task: int - Current task number
            epoch: int - Current epoch number
            batch: int - Current batch number
        """
        # Build context string
        context_parts = []
        if task is not None:
            context_parts.append(f"T{task}")
        if epoch is not None:
            context_parts.append(f"E{epoch}")
        if batch is not None:
            context_parts.append(f"B{batch}")

        context = f"[{' '.join(context_parts)}] " if context_parts else ""

        # Format basic statistics (always logged)
        log_parts = [
            f"pos_mean: {stats['pos_mean']:.4f}",
            f"pos_std: {stats['pos_std']:.4f}",
            f"neg_mean: {stats['neg_mean']:.4f}",
            f"neg_std: {stats['neg_std']:.4f}",
            f"current_gap: {stats['current_gap']:.4f}",
        ]

        log_msg = " | ".join(log_parts)
        self.logger.debug(f"{context}Contrastive stats: {log_msg}")

    def log_ant_distance_stats(self, stats, task=None, epoch=None, batch=None):
        """
        Log detailed ANT-specific statistics (only when ANT is active).

        Args:
            stats: dict - Dictionary containing ANT-specific statistics:
                - pos_min, pos_max: Min/max positive similarities
                - neg_min, neg_max: Min/max negative similarities
                - gap_mean, gap_std, gap_min, gap_max: Gap statistics (distance from margin)
                - margin: the margin value used
                - violation_pct: percentage of pairs violating margin
                - ant_loss: the computed ANT loss value
            task: int - Current task number
            epoch: int - Current epoch number
            batch: int - Current batch number
        """
        # Build context string
        context_parts = []
        if task is not None:
            context_parts.append(f"T{task}")
        if epoch is not None:
            context_parts.append(f"E{epoch}")
        if batch is not None:
            context_parts.append(f"B{batch}")

        context = f"[{' '.join(context_parts)}] " if context_parts else ""

        # Format ANT-specific statistics (only logged when ANT is active)
        log_parts = [
            f"pos_min: {stats['pos_min']:.4f}",
            f"pos_max: {stats['pos_max']:.4f}",
            f"neg_min: {stats['neg_min']:.4f}",
            f"neg_max: {stats['neg_max']:.4f}",
            f"gap_mean: {stats['gap_mean']:.4f}",
            f"gap_std: {stats['gap_std']:.4f}",
            f"gap_min: {stats['gap_min']:.4f}",
            f"gap_max: {stats['gap_max']:.4f}",
            f"margin: {stats['margin']:.4f}",
            f"violation_pct: {stats['violation_pct']:.2f}%",
            f"ant_loss: {stats['ant_loss']:.4f}",
        ]

        log_msg = " | ".join(log_parts)
        self.logger.debug(f"{context}ANT distance stats: {log_msg}")

    def log_ant_flattening_diagnostics(self, stats, task=None, epoch=None, batch=None):
        """
        Log ANT count-floor diagnostics to diagnose the flattening phenomenon.

        Compares the raw ANT loss against a count-floor-adjusted version and
        reports active violation statistics and hard-negative geometry.

        Args:
            stats: dict containing:
                - num_neg_mean: mean number of valid negatives per anchor
                - ant_loss_raw: raw ANT loss (logsumexp-based floor included)
                - ant_loss_adj: count-floor-adjusted ANT loss
                - active_ratio: fraction of valid negatives with active violations
                - active_count_mean: mean count of active violations per anchor
                - viol_mean_all: mean ReLU(raw_v) over all valid pairs
                - viol_mean_act: mean ReLU(raw_v) over actively violating pairs only
                - viol_max_act: max ReLU(raw_v) over actively violating pairs
                  NOTE: uninformative when ant_max_global=True (always == gamma)
                - raw_v_p90: 90th percentile of raw_v over valid pairs (pre-ReLU)
                - raw_v_p95: 95th percentile of raw_v over valid pairs (pre-ReLU)
                - hard_neg_sim: mean hardest-negative similarity per anchor
                - sim_gap_mean: mean(pos_sim - hardest_neg_sim)
                - sim_gap_min: min(pos_sim - hardest_neg_sim)
                - raw_v_mean: mean raw violation value (pre-ReLU, over valid pairs)
                - raw_v_min: min raw violation value (pre-ReLU, over valid pairs)
                - raw_v_max: max raw violation value (pre-ReLU, over valid pairs)
            task, epoch, batch: context for log message
        """
        context_parts = []
        if task is not None:
            context_parts.append(f"T{task}")
        if epoch is not None:
            context_parts.append(f"E{epoch}")
        if batch is not None:
            context_parts.append(f"B{batch}")
        context = f"[{' '.join(context_parts)}] " if context_parts else ""

        log_parts = [
            f"num_neg_mean: {stats['num_neg_mean']:.1f}",
            f"ant_loss_raw: {stats['ant_loss_raw']:.4f}",
            f"ant_loss_adj: {stats['ant_loss_adj']:.4f}",
            f"active_ratio: {stats['active_ratio']:.4f}",
            f"active_count_mean: {stats['active_count_mean']:.2f}",
            f"viol_mean_all: {stats['viol_mean_all']:.4f}",
            f"viol_mean_act: {stats['viol_mean_act']:.4f}",
            f"viol_max_act: {stats['viol_max_act']:.4f}",
            f"raw_v_p90: {stats['raw_v_p90']:.4f}",
            f"raw_v_p95: {stats['raw_v_p95']:.4f}",
            f"raw_v_mean: {stats['raw_v_mean']:.4f}",
            f"raw_v_min: {stats['raw_v_min']:.4f}",
            f"raw_v_max: {stats['raw_v_max']:.4f}",
            f"hard_neg_sim: {stats['hard_neg_sim']:.4f}",
            f"sim_gap_mean: {stats['sim_gap_mean']:.4f}",
            f"sim_gap_min: {stats['sim_gap_min']:.4f}",
        ]
        log_msg = " | ".join(log_parts)
        self.logger.debug(f"{context}ANT flattening: {log_msg}")
