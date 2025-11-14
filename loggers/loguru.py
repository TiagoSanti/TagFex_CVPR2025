import os
from pathlib import Path
from loguru import logger
import torch
import numpy as np
import matplotlib.pyplot as plt


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
            self.logger.add(log_dir / filename, level="INFO")

            if self.configs.get("debug"):
                debug_filename = f"{logfile_prefix}_debuglog{rank}.log"
                self.logger.add(log_dir / debug_filename, level=0)

            # Always create a matrix debug log for similarity matrices
            matrix_debug_filename = f"{logfile_prefix}_matrix_debug{rank}.log"
            self.logger.add(
                log_dir / matrix_debug_filename,
                level="DEBUG",
                filter=lambda record: "Matrix" in record["message"]
                or "Loss components" in record["message"]
                or "ANT distance stats" in record["message"],
            )

            if rank == 0:
                performance_logname = f"{logfile_prefix}_gistlog.log"
                self.logger.add(log_dir / performance_logname, level="SUCCESS")

    def _set_tqdm(self, level="INFO"):
        from tqdm import tqdm
        from functools import partial

        self.logger.remove()
        self.logger.add(partial(tqdm.write, end=""), colorize=True, level=level)

    def log_similarity_matrix(self, matrix, name="similarity", description=""):
        """
        Placeholder for similarity matrix logging (currently disabled).

        Args:
            matrix: torch.Tensor or np.ndarray - The similarity matrix to visualize
            name: str - Name identifier for the matrix
            description: str - Description of what the matrix represents
        """
        return

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

    def log_ant_distance_stats(self, stats, task=None, epoch=None, batch=None):
        """
        Log detailed ANT distance statistics including positive/negative distances and gaps.

        Args:
            stats: dict - Dictionary containing distance statistics:
                - pos_distances: tensor of positive pair distances
                - neg_distances: tensor of negative pair distances
                - gaps: tensor of gaps (pos - neg)
                - margin: the margin value used
                - violations: percentage of pairs violating margin
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

        # Format statistics
        log_parts = [
            f"pos_mean: {stats['pos_mean']:.4f}",
            f"pos_std: {stats['pos_std']:.4f}",
            f"pos_min: {stats['pos_min']:.4f}",
            f"pos_max: {stats['pos_max']:.4f}",
            f"neg_mean: {stats['neg_mean']:.4f}",
            f"neg_std: {stats['neg_std']:.4f}",
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
