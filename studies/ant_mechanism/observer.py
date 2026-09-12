"""Runtime observer for the ANT mechanism study.

The observer is opt-in and receives detached similarity matrices for numerical
analysis.  The only retained graph objects are loss scalars at explicitly
selected gradient-probe batches.
"""

from __future__ import annotations

import atexit
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from .analytics import analyze_contrastive_geometry
from .schema import ALL_ANT_VARIANTS, ANTVariant, SCHEMA_VERSION, validate_metric_record


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


class ANTStudyObserver:
    """Collect sampled, structured ANT/InfoNCE diagnostics."""

    def __init__(self, configs: dict[str, Any], logger=None) -> None:
        cfg = dict(configs.get("ant_study") or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.logger = logger
        self._closed = False
        self._handles: dict[str, Any] = {}
        self._batch_metadata: dict[str, Any] = {}
        self._batch_payloads: dict[str, dict[str, torch.Tensor]] = {}
        self._pending_losses: dict[str, dict[str, torch.Tensor]] = {}
        self._parameter_snapshot: dict[str, torch.Tensor] | None = None
        self._snapshot_bytes = 0
        self.parameter_gradient_probes = False
        self.parameter_update_probes = False
        self.snapshot_image_count = 0

        if not self.enabled:
            return

        if configs.get("ffcv"):
            raise ValueError("ant_study currently requires ffcv: false so sample identities remain observable")
        if configs.get("distributed"):
            raise ValueError("ant_study is currently intended for single-GPU diagnostic runs")

        self.scalar_every = max(1, int(cfg.get("scalar_every_n_batches", 10)))
        self.snapshot_batches = {int(v) for v in cfg.get("snapshot_batches", [1])}
        self.snapshot_tasks = {int(v) for v in cfg.get("snapshot_tasks", [1, 2, 3])}
        self.init_snapshot_epochs = {
            int(v) for v in cfg.get("init_snapshot_epochs", [1, 5, 20, 60, 120, 170, 200])
        }
        self.inc_snapshot_epochs = {
            int(v) for v in cfg.get("inc_snapshot_epochs", [1, 5, 20, 80, 120, 150, 170])
        }
        self.shadow_variants = bool(cfg.get("shadow_variants", True))
        self.save_snapshots = bool(cfg.get("save_snapshots", True))
        self.parameter_gradient_probes = bool(cfg.get("parameter_gradient_probes", True))
        self.parameter_update_probes = bool(cfg.get("parameter_update_probes", True))
        self.snapshot_image_count = max(0, int(cfg.get("snapshot_image_count", 8)))
        self.min_free_gb = float(cfg.get("min_free_gb", 50.0))
        self.max_output_gb = float(cfg.get("max_output_gb", 10.0))

        output_root = Path(cfg.get("output_root", "./study_outputs/ant_mechanism_20260909"))
        run_name = cfg.get("run_name") or Path(str(configs.get("log_dir", "run"))).name
        seed = configs.get("seed", "unknown")
        seed_suffix = f"_s{seed}"
        run_leaf = str(run_name) if str(run_name).endswith(seed_suffix) else f"{run_name}{seed_suffix}"
        self.run_dir = output_root.expanduser() / run_leaf
        self.snapshot_dir = self.run_dir / "snapshots"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._write_manifest(configs, cfg)
        atexit.register(self.close)
        self._log(f"ANT study observer enabled: {self.run_dir}")

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _write_manifest(self, configs: dict[str, Any], study_cfg: dict[str, Any]) -> None:
        selected = {
            key: configs.get(key)
            for key in (
                "dataset_name", "dataset_root", "scenario", "class_order", "seed",
                "method", "init_epochs", "inc_epochs", "infonce_temp",
                "infonce_kd_temp", "nce_alpha", "ant_beta", "ant_margin",
                "ant_max_global", "ant_symmetric_full", "ant_detach_reference",
                "ant_formulation", "contrast_factor", "contrast_kd_factor",
                "memory_configs", "trainloader_params", "backbone_configs",
                "network_configs",
            )
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "instrumentation_source_sha256": self._instrumentation_source_hash(),
            "study": study_cfg,
            "training": selected,
            "variants": [variant.name for variant in ALL_ANT_VARIANTS],
        }
        with (self.run_dir / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False, default=_json_default)

    @staticmethod
    def _instrumentation_source_hash() -> str:
        repo_root = Path(__file__).resolve().parents[2]
        paths = [
            repo_root / "methods" / "tagfex" / "tagfex.py",
            repo_root / "loggers" / "loguru.py",
            repo_root / "modules" / "data" / "dataloader.py",
        ]
        paths.extend(sorted((repo_root / "studies" / "ant_mechanism").glob("*.py")))
        digest = hashlib.sha256()
        for path in sorted(paths):
            digest.update(str(path.relative_to(repo_root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _handle(self, name: str):
        if name not in self._handles:
            self._handles[name] = gzip.open(
                self.run_dir / f"{name}.jsonl.gz", "at", encoding="utf-8"
            )
        return self._handles[name]

    def _write_record(self, stream_name: str, record: dict[str, Any]) -> None:
        handle = self._handle(stream_name)
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()

    def set_batch_metadata(self, metadata: dict[str, Any]) -> None:
        if self.enabled:
            self._batch_metadata = metadata
            self._batch_payloads.clear()

    def set_dataset_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with (self.run_dir / "dataset_metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False, default=_json_default)

    def set_batch_payload(
        self,
        branch: str,
        *,
        task: int,
        epoch: int,
        batch: int,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        if not self.is_snapshot(task, epoch, batch):
            return
        self._batch_payloads[branch] = {
            name: tensor.detach().cpu()
            for name, tensor in tensors.items()
            if tensor is not None
        }

    def is_snapshot(self, task: int, epoch: int, batch: int) -> bool:
        if not self.enabled or batch not in self.snapshot_batches or task not in self.snapshot_tasks:
            return False
        epochs = self.init_snapshot_epochs if task == 1 else self.inc_snapshot_epochs
        return epoch in epochs

    def should_capture(self, task: int, epoch: int, batch: int) -> bool:
        return self.enabled and (batch == 1 or batch % self.scalar_every == 0 or self.is_snapshot(task, epoch, batch))

    @staticmethod
    def _actual_variant(params: dict[str, Any]) -> ANTVariant:
        return ANTVariant(
            "FS" if params["ant_symmetric_full"] else "IV",
            "GR" if params["ant_max_global"] else "AR",
            bool(params.get("ant_detach_reference", False)),
        )

    def _semantic_metrics(self, analysis) -> dict[str, float]:
        targets = self._batch_metadata.get("continual_target")
        replay = self._batch_metadata.get("is_replay")
        if targets is None:
            return {}
        valid = analysis.tensors["valid_mask"]
        active = analysis.tensors["active_mask"]
        reference = analysis.tensors["reference_mask"]
        rows, cols = valid.shape
        targets_t = torch.as_tensor(targets, device=valid.device)[: max(rows, cols)]
        same_class = targets_t[:rows, None] == targets_t[:cols][None, :]

        def ratio(mask: torch.Tensor, domain: torch.Tensor) -> float:
            count = domain.sum()
            if count == 0:
                return 0.0
            return float((mask & domain).sum().float().div(count).item())

        metrics = {
            "same_class_negative_ratio": ratio(same_class, valid),
            "same_class_active_ratio": ratio(same_class, active),
            "same_class_reference_ratio": ratio(same_class, reference),
        }
        if replay is not None:
            replay_t = torch.as_tensor(replay, dtype=torch.bool, device=valid.device)[: max(rows, cols)]
            row_replay = replay_t[:rows, None]
            col_replay = replay_t[:cols][None, :]
            metrics.update(
                {
                    "active_current_current_ratio": ratio(~row_replay & ~col_replay, active),
                    "active_replay_replay_ratio": ratio(row_replay & col_replay, active),
                    "active_current_replay_ratio": ratio(row_replay ^ col_replay, active),
                }
            )
        return metrics

    def record_contrastive(
        self,
        cos_sim: torch.Tensor,
        *,
        branch: str,
        task: int,
        epoch: int,
        batch: int,
        params: dict[str, Any],
        nce_loss_tensor: torch.Tensor,
        ant_loss_tensor: torch.Tensor,
    ) -> None:
        if not self.should_capture(task, epoch, batch):
            return

        actual = self._actual_variant(params)
        snapshot = self.is_snapshot(task, epoch, batch)
        baseline = float(params["ant_beta"]) == 0.0
        analysis_items = [("Baseline-InfoNCE" if baseline else actual.name, actual, "actual")]
        if snapshot and self.shadow_variants:
            analysis_items.extend(
                (variant.name, variant, "shadow")
                for variant in ALL_ANT_VARIANTS
                if baseline or variant != actual
            )

        analyses = {}
        for label, variant, mode in analysis_items:
            analysis = analyze_contrastive_geometry(
                cos_sim,
                temperature=float(params["temperature"]),
                margin=float(params["ant_margin"]),
                variant=variant,
                nce_alpha=float(params["nce_alpha"]),
                ant_beta=float(params["ant_beta"]),
            )
            analyses[label] = analysis
            record = {
                "schema_version": SCHEMA_VERSION,
                "task": int(task),
                "epoch": int(epoch),
                "batch": int(batch),
                "branch": branch,
                "variant": label,
                "mode": mode,
                "trained_ant_beta": float(params["ant_beta"]),
                **analysis.metrics,
                **self._semantic_metrics(analysis),
            }
            validate_metric_record(record)
            self._write_record("geometry_metrics", record)

        if snapshot and self.save_snapshots and self._storage_allows_snapshot():
            self._save_snapshot(analyses, branch=branch, task=task, epoch=epoch, batch=batch)

        if snapshot and self.parameter_gradient_probes:
            self._pending_losses[branch] = {
                "nce": nce_loss_tensor,
                "ant": ant_loss_tensor,
            }

    def _storage_allows_snapshot(self) -> bool:
        if self._snapshot_bytes >= self.max_output_gb * 1024**3:
            self._log("ANT study snapshot quota reached; scalar collection continues")
            return False
        free_gb = shutil.disk_usage(self.run_dir).free / 1024**3
        if free_gb < self.min_free_gb:
            self._log(f"ANT study snapshot disabled: only {free_gb:.1f} GiB free")
            return False
        return True

    def _save_snapshot(self, analyses, *, branch: str, task: int, epoch: int, batch: int) -> None:
        arrays: dict[str, np.ndarray] = {}
        first = next(iter(analyses.values()))
        for tensor_name in (
            "cos_sim",
            "nce_grad",
            "nce_probabilities",
            "nce_positive_mask",
            "nce_negative_mask",
            "nce_loss_per_anchor",
        ):
            tensor = first.tensors[tensor_name].cpu()
            if tensor.dtype == torch.bool:
                arrays[tensor_name] = tensor.numpy()
            else:
                arrays[tensor_name] = tensor.numpy().astype(
                    np.float16 if tensor_name == "cos_sim" else np.float32
                )
        for name, analysis in analyses.items():
            key = name.lower().replace("-", "_")
            for tensor_name in (
                "valid_mask",
                "active_mask",
                "reference_mask",
                "raw_violation",
                "ant_weights",
                "ant_loss_per_anchor",
                "ant_grad",
                "combined_grad",
            ):
                tensor = analysis.tensors[tensor_name].cpu()
                if tensor.dtype == torch.bool:
                    arrays[f"{key}__{tensor_name}"] = tensor.numpy()
                else:
                    arrays[f"{key}__{tensor_name}"] = tensor.numpy().astype(np.float32)

        for key, value in self._batch_metadata.items():
            if isinstance(value, (list, tuple, np.ndarray)) or torch.is_tensor(value):
                arrays[f"meta__{key}"] = np.asarray(
                    value.detach().cpu().numpy() if torch.is_tensor(value) else value
                )

        for name, tensor in self._batch_payloads.get(branch, {}).items():
            value = tensor
            if name in {"view1", "view2"}:
                value = value[: self.snapshot_image_count]
            arrays[f"payload__{name}"] = value.numpy().astype(np.float16)

        path = self.snapshot_dir / f"T{task:02d}_E{epoch:03d}_B{batch:04d}_{branch}.npz"
        np.savez_compressed(path, **arrays)
        self._snapshot_bytes += path.stat().st_size

    def record_training_objective(
        self,
        *,
        task: int,
        epoch: int,
        batch: int,
        values: dict[str, Any],
    ) -> None:
        if not self.should_capture(task, epoch, batch):
            return
        record = {
            "schema_version": SCHEMA_VERSION,
            "task": int(task),
            "epoch": int(epoch),
            "batch": int(batch),
            **{
                key: float(value.detach().item()) if torch.is_tensor(value) else float(value)
                for key, value in values.items()
                if value is not None
            },
        }
        self._write_record("training_objective", record)

    def record_evaluation(
        self,
        *,
        phase: str,
        task: int,
        epoch: int | None,
        values: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "task": int(task),
            "epoch": None if epoch is None else int(epoch),
        }
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().cpu()
                record[key] = float(value.item()) if value.ndim == 0 else value.tolist()
            else:
                record[key] = value
        self._write_record("evaluations", record)

    @staticmethod
    def _parameter_group(name: str) -> str:
        for group in ("ta_net", "projector", "predictor", "classifier", "ts_nets"):
            if group in name:
                return group
        return "other"

    def record_component_parameter_gradients(
        self,
        model: torch.nn.Module,
        *,
        task: int,
        epoch: int,
        batch: int,
        branch_outer_weights: dict[str, float],
        ant_beta: float,
        nce_alpha: float,
    ) -> None:
        if not self.enabled or not self.parameter_gradient_probes or not self.is_snapshot(task, epoch, batch):
            self._pending_losses.clear()
            return

        named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        params = [p for _, p in named_params]
        component_grads: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for branch, losses in self._pending_losses.items():
            outer = float(branch_outer_weights.get(branch, 0.0))
            for component, inner in (("nce", nce_alpha), ("ant", ant_beta)):
                scalar = losses[component] * outer * float(inner)
                grads = torch.autograd.grad(
                    scalar,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                component_grads[f"{branch}_{component}"] = grads

        groups = sorted({self._parameter_group(name) for name, _ in named_params})
        for component, grads in component_grads.items():
            sums = {group: 0.0 for group in groups}
            for (name, _), grad in zip(named_params, grads):
                if grad is not None:
                    sums[self._parameter_group(name)] += float(grad.detach().pow(2).sum().item())
            for group, squared_norm in sums.items():
                self._write_record(
                    "parameter_gradients",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task": task,
                        "epoch": epoch,
                        "batch": batch,
                        "component": component,
                        "parameter_group": group,
                        "grad_norm": squared_norm**0.5,
                    },
                )

        for branch in self._pending_losses:
            left = component_grads.get(f"{branch}_nce")
            right = component_grads.get(f"{branch}_ant")
            if left is None or right is None:
                continue
            for group in groups:
                dot = nce_sq = ant_sq = 0.0
                for (name, _), g_nce, g_ant in zip(named_params, left, right):
                    if self._parameter_group(name) != group or g_nce is None or g_ant is None:
                        continue
                    dot += float((g_nce.detach() * g_ant.detach()).sum().item())
                    nce_sq += float(g_nce.detach().pow(2).sum().item())
                    ant_sq += float(g_ant.detach().pow(2).sum().item())
                cosine = dot / max((nce_sq * ant_sq) ** 0.5, 1e-30)
                self._write_record(
                    "parameter_gradient_alignment",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "task": task,
                        "epoch": epoch,
                        "batch": batch,
                        "branch": branch,
                        "parameter_group": group,
                        "nce_ant_grad_cosine": cosine,
                        "ant_to_nce_grad_ratio": (ant_sq / max(nce_sq, 1e-30)) ** 0.5,
                    },
                )
        self._pending_losses.clear()

    def capture_parameter_state(self, model: torch.nn.Module, *, task: int, epoch: int, batch: int) -> None:
        if self.enabled and self.parameter_update_probes and self.is_snapshot(task, epoch, batch):
            self._parameter_snapshot = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    def record_total_parameter_gradients(
        self,
        model: torch.nn.Module,
        *,
        task: int,
        epoch: int,
        batch: int,
        stage: str,
    ) -> None:
        if not self.enabled or not self.is_snapshot(task, epoch, batch):
            return
        totals: dict[str, float] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            group = self._parameter_group(name)
            totals[group] = totals.get(group, 0.0) + float(
                parameter.grad.detach().pow(2).sum().item()
            )
        for group, squared_norm in totals.items():
            self._write_record(
                "total_parameter_gradients",
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "epoch": epoch,
                    "batch": batch,
                    "stage": stage,
                    "parameter_group": group,
                    "grad_norm": squared_norm**0.5,
                },
            )

    def record_parameter_update(self, model: torch.nn.Module, *, task: int, epoch: int, batch: int) -> None:
        if self._parameter_snapshot is None:
            return
        totals: dict[str, list[float]] = {}
        for name, parameter in model.named_parameters():
            before = self._parameter_snapshot.get(name)
            if before is None:
                continue
            after = parameter.detach().cpu()
            group = self._parameter_group(name)
            values = totals.setdefault(group, [0.0, 0.0])
            values[0] += float((after - before).pow(2).sum().item())
            values[1] += float(before.pow(2).sum().item())
        for group, (delta_sq, parameter_sq) in totals.items():
            self._write_record(
                "parameter_updates",
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "epoch": epoch,
                    "batch": batch,
                    "parameter_group": group,
                    "update_norm": delta_sq**0.5,
                    "parameter_norm": parameter_sq**0.5,
                    "relative_update_norm": (delta_sq / max(parameter_sq, 1e-30)) ** 0.5,
                },
            )
        self._parameter_snapshot = None

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._closed = True
