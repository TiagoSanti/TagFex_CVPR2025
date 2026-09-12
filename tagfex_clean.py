"""Versão enxuta do TagFex.

Mantém a lógica de treinamento incremental, Avg-K teacher, SBS, herding,
InfoNCE e ANT. Foram removidos logs, métricas de acompanhamento, heatmaps,
barras de progresso, geração de nomes de diretórios e checkpoints.
"""

import collections
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from modules import HerdingIndicesLearner, get_loaders
from modules import optimizer_dispatch, scheduler_dispatch
from modules.data.dataset import WithIndexDataset
from .tagfexnet import TagFexNet


EPSILON = 1e-8


def _sbs_keep_mask(
    speeds: np.ndarray,
    q: float,
    s: float,
    min_keep: int,
) -> np.ndarray:
    """Mantém a faixa intermediária das velocidades de aprendizagem."""
    n = len(speeds)
    n_drop_slow = int(np.floor(n * s))
    n_drop_fast = int(np.floor(n * q))

    if n_drop_slow + n_drop_fast >= n - min_keep:
        return np.ones(n, dtype=bool)

    order = np.argsort(speeds)
    dropped = set(order[:n_drop_slow].tolist())

    if n_drop_fast > 0:
        dropped.update(order[n - n_drop_fast :].tolist())

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[list(dropped)] = False
    return keep_mask


@torch.no_grad()
def _avg_state_dicts(state_dicts: list[dict]) -> dict:
    """Calcula a média elemento a elemento de vários state_dicts."""
    averaged = {}

    for key in state_dicts[0]:
        reference = state_dicts[0][key]

        if torch.is_floating_point(reference):
            values = [state[key].float() for state in state_dicts]
            averaged[key] = torch.stack(values).mean(dim=0).to(reference.dtype)
        else:
            averaged[key] = state_dicts[-1][key].clone()

    return averaged


class TagFex(HerdingIndicesLearner):
    def __init__(
        self,
        data_maganger,
        configs: dict,
        device,
        distributed=False,
    ) -> None:
        super().__init__(data_maganger, configs, device, distributed)

        self._init_network(
            self.configs.get("backbone_configs", {}),
            self.configs.get("network_configs", {}),
        )

        if self.distributed is not None:
            self._init_ddp()

        self.ordered_index_map = torch.from_numpy(
            self.data_manager.ordered_index_map
        ).to(self.device)

        self._sbs_q = float(self.configs.get("sbs_q", 0.0))
        self._sbs_s = float(self.configs.get("sbs_s", 0.0))
        self._sbs_tracking = False
        self._sbs_correct = None
        self._sbs_total = None
        self._sbs_n_new = 0
        self._sbs_task_new_abs = None
        self._sbs_speed_map = {}

    def _init_network(self, backbone_configs: dict, network_configs: dict) -> None:
        if backbone_configs["name"] == "resnet18":
            params = {
                "dataset_name": self.data_manager.dataset_name,
                "small_base": (
                    self.data_manager.init_num_cls
                    == self.data_manager.inc_num_cls
                ),
            }
            backbone_configs.update(params=params)

            if network_configs.get("new_backbone_configs"):
                network_configs["new_backbone_configs"].update(params=params)

        self.network = TagFexNet(
            backbone_configs,
            network_configs,
            self.device,
        )
        self.local_network = self.network

        self.last_ta_net = None
        self.last_projector = None

        self._avg_last_k = self.configs.get("avg_last_k", 0)
        self._ckpt_buf_ta = None
        self._ckpt_buf_proj = None

    def _init_ddp(self) -> None:
        self.configs["trainloader_params"]["batch_size"] //= self.distributed[
            "world_size"
        ]
        torch.distributed.barrier()

    def _model_to_ddp(self) -> None:
        self.network = nn.parallel.DistributedDataParallel(
            self.local_network,
            device_ids=[self.distributed["rank"]],
            find_unused_parameters=self.configs["debug"],
        )

    @torch.no_grad()
    def extract_herding_features(
        self,
        dataset: torch.utils.data.Dataset,
    ) -> torch.Tensor:
        if self.configs["ffcv"]:
            from modules.data.ffcv.loader import (
                OrderOption,
                default_transform_dict,
                get_ffcv_loader,
            )

            dataset_name = self.configs["dataset_name"].lower()
            pipeline_name = self.configs.get(
                "test_transform",
                default_transform_dict[dataset_name.strip("0123456789")][0],
            )
            data_loader = get_ffcv_loader(
                dataset,
                self.configs["train_beton_path"],
                pipeline_name,
                self.device,
                order=OrderOption.SEQUENTIAL,
                seed=self.configs["seed"],
                **self.configs["testloader_params"],
            )
        else:
            data_loader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.configs["testloader_params"]["batch_size"],
                num_workers=0,
            )

        self.local_network.eval()
        features = []

        for samples, _ in data_loader:
            samples = samples.to(self.device, non_blocking=True)
            output = self.local_network(samples.contiguous())
            batch_features = torch.cat(output["ts_features"], dim=-1)
            features.append(batch_features)

        features = torch.cat(features)
        return F.normalize(features, dim=-1, eps=EPSILON)

    def train(self) -> None:
        self.update_state(
            run_state="train",
            num_tasks=self.data_manager.num_tasks,
        )

        for task_id, (task_train, task_test) in enumerate(
            self.data_manager.tasks
        ):
            current_classes = self.data_manager.task_num_cls[task_id]
            total_classes = sum(
                self.data_manager.task_num_cls[: task_id + 1]
            )
            learned_classes = total_classes - current_classes

            self.update_state(
                cur_task=task_id + 1,
                cur_task_num_classes=current_classes,
                sofar_num_classes=total_classes,
                learned_num_classes=learned_classes,
            )

            if task_id > 0:
                if self._avg_last_k > 0 and self._ckpt_buf_ta:
                    self.last_ta_net = self._build_avg_ta()
                    self.last_projector = self._build_avg_projector()
                else:
                    self.last_ta_net = (
                        self.local_network.get_freezed_copy_ta()
                    )
                    self.last_projector = (
                        self.local_network.get_freezed_copy_projector()
                    )

            self.local_network.update_network(current_classes)
            self.local_network.freeze_old_backbones()

            if self.distributed is not None:
                self._model_to_ddp()

            sbs_enabled = (
                (self._sbs_q > 0 or self._sbs_s > 0)
                and not self.configs.get("ffcv")
                and self.distributed is None
            )

            if sbs_enabled:
                self._sbs_n_new = len(task_train.indices)
                self._sbs_task_new_abs = task_train.indices.copy()

            memory_indices = self.get_memory()
            task_train.indices = np.concatenate(
                (task_train.indices, memory_indices)
            )

            train_loader, test_loader = self._build_task_loaders(
                task_train,
                task_test,
                sbs_enabled,
            )

            self.train_task(train_loader)

            if self._sbs_tracking:
                self._sbs_speed_map = self._sbs_finalize_speeds()

            self.reduce_memory()
            self.update_memory()

            if task_id > 0:
                self.local_network.weight_align(current_classes)

            self.eval_epoch(test_loader)

        self.update_state(run_state="finished")

    def _build_task_loaders(
        self,
        task_train,
        task_test,
        sbs_enabled: bool,
    ):
        trainloader_params = self.configs["trainloader_params"].copy()

        if self.configs["ffcv"]:
            from modules.data.ffcv.loader import get_ffcv_loaders

            return get_ffcv_loaders(
                task_train,
                task_test,
                trainloader_params,
                self.configs["testloader_params"],
                self.device,
                self.configs,
                self.distributed is not None,
            )

        train_source = (
            WithIndexDataset(task_train) if sbs_enabled else task_train
        )

        if sbs_enabled:
            n_total = len(task_train)
            self._sbs_correct = np.zeros(n_total, dtype=np.int32)
            self._sbs_total = np.zeros(n_total, dtype=np.int32)
            self._sbs_tracking = True

        return get_loaders(
            train_source,
            task_test,
            trainloader_params,
            self.configs["testloader_params"],
            self.distributed,
        )

    def train_task(self, train_loader) -> None:
        first_task = self.state["cur_task"] == 1

        optimizer_configs = self.configs[
            "init_optimizer_configs" if first_task else "inc_optimizer_configs"
        ]
        scheduler_configs = self.configs[
            "init_scheduler_configs" if first_task else "inc_scheduler_configs"
        ]
        num_epochs = self.configs[
            "init_epochs" if first_task else "inc_epochs"
        ]

        optimizer = optimizer_dispatch(
            self.local_network.parameters(),
            optimizer_configs,
        )
        scheduler = scheduler_dispatch(optimizer, scheduler_configs)

        if self.configs["debug"]:
            num_epochs = 5

        self.update_state(cur_task_num_epochs=num_epochs)

        if self._avg_last_k > 0:
            self._reset_ckpt_buffer()

        for epoch in range(num_epochs):
            self.update_state(
                cur_epoch=epoch + 1,
                num_batches=len(train_loader),
            )
            self.add_state(accumulated_cur_epoch=1)

            if self.distributed is not None and not self.configs["ffcv"]:
                train_loader.sampler.set_epoch(epoch)

            self.train_epoch(train_loader, optimizer)
            scheduler.step()

            if self._avg_last_k > 0 and epoch >= num_epochs - self._avg_last_k:
                self._append_ckpt()

    def _reset_ckpt_buffer(self) -> None:
        self._ckpt_buf_ta = collections.deque(maxlen=self._avg_last_k)
        self._ckpt_buf_proj = collections.deque(maxlen=self._avg_last_k)

    @torch.no_grad()
    def _append_ckpt(self) -> None:
        ta_state = {
            key: value.detach().clone()
            for key, value in self.local_network.ta_net.state_dict().items()
        }
        projector_state = {
            key: value.detach().clone()
            for key, value in self.local_network.projector.state_dict().items()
        }
        self._ckpt_buf_ta.append(ta_state)
        self._ckpt_buf_proj.append(projector_state)

    def _build_avg_ta(self):
        averaged_state = _avg_state_dicts(list(self._ckpt_buf_ta))
        teacher = copy.deepcopy(self.local_network.ta_net)
        teacher.load_state_dict(averaged_state)

        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        return teacher.eval()

    def _build_avg_projector(self):
        averaged_state = _avg_state_dicts(list(self._ckpt_buf_proj))
        projector = copy.deepcopy(self.local_network.projector)
        projector.load_state_dict(averaged_state)

        for parameter in projector.parameters():
            parameter.requires_grad_(False)

        return projector.eval()

    def train_epoch(self, train_loader, optimizer) -> None:
        self.network.train()

        for batch_index, batch_data in enumerate(train_loader):
            batch_data = tuple(
                value.to(self.device, non_blocking=True)
                for value in batch_data
            )

            if self._sbs_tracking:
                local_idx, sample1, sample2, targets = batch_data
            elif self.configs.get("ffcv"):
                sample1, targets, sample2 = batch_data
            else:
                sample1, sample2, targets = batch_data

            targets = self.ordered_index_map[targets.flatten()]
            samples = torch.cat((sample1, sample2))
            targets = torch.cat((targets, targets))

            self.update_state(cur_batch=batch_index + 1)

            output = self.network(samples.contiguous())
            logits = output["logits"]
            cls_loss = F.cross_entropy(logits, targets)

            ant_parameters = self._get_ant_parameters()
            contrastive_loss = infoNCE_loss(
                output["embedding"],
                self.configs["infonce_temp"],
                **ant_parameters,
            )

            if output.get("aux_logits") is None:
                loss = (
                    cls_loss
                    + self.configs["contrast_factor"] * contrastive_loss
                )
            else:
                loss = self._compute_incremental_loss(
                    output,
                    samples,
                    targets,
                    logits,
                    cls_loss,
                    contrastive_loss,
                    ant_parameters,
                )

            optimizer.zero_grad()
            loss.backward()

            grad_clip_norm = self.configs.get("grad_clip_norm")
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.local_network.parameters(),
                    max_norm=grad_clip_norm,
                )

            optimizer.step()

            if self._sbs_tracking:
                self._sbs_record_batch(local_idx, logits, targets)

    def _get_ant_parameters(self) -> dict:
        ant_beta = self.configs.get("ant_beta", 0.0)
        ant_max_global = self.configs.get("ant_max_global", True)

        return {
            "nce_alpha": self.configs.get("nce_alpha", 1.0),
            "ant_beta": ant_beta,
            "ant_margin": self.configs.get("ant_margin", 0.1),
            "ant_max_global": ant_max_global,
            "ant_symmetric_full": self.configs.get(
                "ant_symmetric_full", False
            ),
            "ant_formulation": self.configs.get(
                "ant_formulation", "logsumexp"
            ),
            "ant_tau": self.configs.get("ant_tau", 0.1),
            "ant_topk": self.configs.get("ant_topk", 32),
        }

    def _compute_incremental_loss(
        self,
        output,
        samples,
        targets,
        logits,
        cls_loss,
        contrastive_loss,
        ant_parameters,
    ):
        learned_classes = self.state["learned_num_classes"]
        total_classes = self.state["sofar_num_classes"]

        aux_targets = torch.where(
            targets >= learned_classes,
            targets - learned_classes + 1,
            0,
        )
        aux_loss = F.cross_entropy(output["aux_logits"], aux_targets)

        with torch.no_grad():
            teacher_features = self.last_ta_net(
                samples.contiguous()
            )["features"]
            teacher_projection = self.last_projector(teacher_features)

        kd_loss = infoNCE_distill_loss(
            self.last_projector(output["predicted_feature"]),
            teacher_projection,
            self.configs["infonce_kd_temp"],
            **ant_parameters,
        )

        current_task_mask = targets >= learned_classes
        trans_logits = output["trans_logits"]
        trans_cls_loss = F.cross_entropy(
            trans_logits[current_task_mask],
            targets[current_task_mask] - learned_classes,
        )

        if trans_cls_loss < cls_loss:
            temperature = self.configs["kd_temp"]
            transfer_loss = F.kl_div(
                (
                    logits[current_task_mask][:, learned_classes:]
                    / temperature
                ).log_softmax(dim=1),
                (
                    trans_logits.detach()[current_task_mask]
                    / temperature
                ).softmax(dim=1),
                reduction="batchmean",
            )
        else:
            transfer_loss = torch.zeros((), device=self.device)

        auto_kd_factor = learned_classes / total_classes

        return (
            cls_loss
            + self.configs["aux_factor"] * aux_loss
            + self.configs["contrast_factor"]
            * (
                contrastive_loss * (1 - auto_kd_factor)
                + self.configs["contrast_kd_factor"]
                * kd_loss
                * auto_kd_factor
            )
            + self.configs["trans_cls_factor"] * trans_cls_loss
            + self.configs["transfer_factor"] * transfer_loss
        )

    @torch.no_grad()
    def _sbs_record_batch(
        self,
        local_idx: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        batch_size = local_idx.shape[0]
        predictions = logits[:batch_size].argmax(dim=1)
        correct = (
            predictions == targets[:batch_size]
        ).cpu().numpy().astype(np.int32)
        indices = local_idx.cpu().numpy()

        new_samples = indices < self._sbs_n_new

        if new_samples.any():
            np.add.at(
                self._sbs_correct,
                indices[new_samples],
                correct[new_samples],
            )
            np.add.at(
                self._sbs_total,
                indices[new_samples],
                1,
            )

    def _sbs_finalize_speeds(self) -> dict[int, float]:
        n_new = self._sbs_n_new
        attempts = np.maximum(self._sbs_total[:n_new], 1)
        speeds = (
            self._sbs_correct[:n_new].astype(np.float32) / attempts
        )

        return {
            int(index): float(speed)
            for index, speed in zip(self._sbs_task_new_abs, speeds)
        }

    @torch.no_grad()
    def update_memory(self) -> None:
        if not self._sbs_tracking or (
            self._sbs_q == 0.0 and self._sbs_s == 0.0
        ):
            super().update_memory()
            self._sbs_tracking = False
            return

        speed_by_abs_index = self._sbs_speed_map
        self._sbs_tracking = False

        total_classes = self.state["sofar_num_classes"]
        current_classes = self.state["cur_task_num_classes"]
        first_current_class = total_classes - current_classes
        current_class_ids = np.arange(first_current_class, total_classes)

        selected_indices = []
        class_means = []

        for class_id in current_class_ids:
            original_class_id = self.data_manager.class_order[
                class_id
            ].item()
            class_dataset = self.data_manager.get_dataset_by_class_ids(
                [original_class_id],
                split="train",
                mode="test",
            )
            absolute_indices = np.asarray(class_dataset.indices)
            speeds = np.asarray(
                [
                    speed_by_abs_index.get(int(index), 0.5)
                    for index in absolute_indices
                ],
                dtype=np.float32,
            )

            keep_mask = _sbs_keep_mask(
                speeds,
                self._sbs_q,
                self._sbs_s,
                self.num_exemplars_per_class,
            )
            filtered_indices = absolute_indices[keep_mask]

            if len(filtered_indices) < self.num_exemplars_per_class:
                raise ValueError(
                    "O filtro SBS manteve menos amostras que o número "
                    "necessário de exemplares por classe."
                )

            filtered_dataset = Subset(
                class_dataset.dataset,
                filtered_indices,
            )
            class_features = self.extract_herding_features(
                filtered_dataset
            )
            class_mean = class_features.mean(dim=0, keepdim=True)

            chosen_indices = []
            selected_mean = torch.zeros(
                self.local_network.feature_dim,
                device=self.device,
            )

            for n_selected in range(
                1,
                self.num_exemplars_per_class + 1,
            ):
                candidate_means = (
                    (n_selected - 1) * selected_mean + class_features
                ) / n_selected
                chosen = (
                    candidate_means - class_mean
                ).norm(dim=-1).argmin().item()

                selected_mean = candidate_means[chosen]
                chosen_indices.append(int(filtered_indices[chosen]))

                remaining = (
                    torch.arange(len(class_features), device=self.device)
                    != chosen
                )
                class_features = class_features[remaining]
                filtered_indices = np.delete(filtered_indices, chosen)

            selected_indices.append(chosen_indices)
            class_means.append(F.normalize(selected_mean, dim=0))

        self.memory_samples.extend(selected_indices)
        self.class_means.extend(class_means)

    @torch.no_grad()
    def eval_epoch(self, data_loader) -> dict[str, torch.Tensor]:
        self.network.eval()

        logits_all = []
        targets_all = []
        nme_logits_all = []

        use_nme = len(self.class_means) == self.state["sofar_num_classes"]

        for samples, targets in data_loader:
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            targets = self.ordered_index_map[targets.flatten()]

            output = self.network(samples.contiguous())
            logits_all.append(output["logits"])
            targets_all.append(targets)

            if use_nme:
                features = torch.cat(output["ts_features"], dim=-1)
                features = F.normalize(features, dim=-1, eps=EPSILON)
                distances = torch.cdist(
                    features,
                    torch.stack(self.class_means),
                )
                nme_logits_all.append(-distances)

        result = {
            "logits": torch.cat(logits_all),
            "targets": torch.cat(targets_all),
        }

        if use_nme:
            result["nme_logits"] = torch.cat(nme_logits_all)

        return result


def _build_ant_matrix(
    cos_sim: torch.Tensor,
    symmetric_full: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constrói a matriz usada pelo ANT e a máscara de negativos válidos."""
    device = cos_sim.device
    n = cos_sim.shape[0]

    if symmetric_full:
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = self_mask.roll(shifts=n // 2, dims=0)
        valid_negatives = ~(self_mask | positive_mask)
        ant_matrix = cos_sim.masked_fill(
            ~valid_negatives,
            -torch.inf,
        )
    else:
        batch_size = n // 2
        ant_matrix = cos_sim[:batch_size, :batch_size]
        self_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=device,
        )
        valid_negatives = ~self_mask
        ant_matrix = ant_matrix.masked_fill(self_mask, -torch.inf)

    return ant_matrix, valid_negatives


def _compute_ant_loss(
    ant_matrix: torch.Tensor,
    valid_negatives: torch.Tensor,
    margin: float,
    max_global: bool,
    formulation: str,
    tau: float,
    topk: int,
) -> torch.Tensor:
    """Calcula uma das formulações da perda ANT."""
    if max_global:
        reference = ant_matrix.max()
    else:
        reference = ant_matrix.max(dim=-1, keepdim=True).values

    violations = ant_matrix - reference + margin
    negatives_per_anchor = valid_negatives.sum(dim=-1).float()

    if formulation == "logsumexp":
        values = torch.relu(violations)
        values = values.masked_fill(~valid_negatives, -torch.inf)
        loss_per_anchor = torch.logsumexp(values, dim=-1)

    elif formulation == "expm1":
        values = torch.expm1(torch.relu(violations))
        values = values.masked_fill(~valid_negatives, 0.0)
        loss_per_anchor = torch.log1p(values.sum(dim=-1))

    elif formulation == "softplus":
        values = F.softplus(violations / tau)
        values = values.masked_fill(~valid_negatives, 0.0)
        loss_per_anchor = (
            values.sum(dim=-1)
            / negatives_per_anchor.clamp_min(1.0)
        )

    elif formulation == "topk":
        k = min(topk, int(negatives_per_anchor.min().item()))

        if k == 0:
            loss_per_anchor = torch.zeros(
                ant_matrix.shape[0],
                device=ant_matrix.device,
            )
        else:
            values = violations.masked_fill(
                ~valid_negatives,
                -torch.inf,
            )
            values = torch.topk(values, k=k, dim=-1).values
            loss_per_anchor = torch.logsumexp(
                torch.relu(values),
                dim=-1,
            )

    elif formulation == "active_only":
        values = torch.relu(violations)
        values = values.masked_fill(~valid_negatives, 0.0)
        active_count = (values > 0).sum(dim=-1).clamp_min(1)
        loss_per_anchor = values.sum(dim=-1) / active_count

    else:
        raise ValueError(
            f"Formulação ANT desconhecida: {formulation}. "
            "Use logsumexp, expm1, softplus, topk ou active_only."
        )

    return loss_per_anchor.mean()


def _compute_infonce_loss(
    cos_sim: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Calcula a formulação original e única da InfoNCE."""
    n = cos_sim.shape[0]
    device = cos_sim.device

    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    positive_mask = self_mask.roll(shifts=n // 2, dims=0)

    logits = cos_sim.masked_fill(self_mask, -9e15) / temperature

    negative_log_likelihood = (
        -logits[positive_mask]
        + torch.logsumexp(logits, dim=-1)
    )
    return negative_log_likelihood.mean()


def _compute_contrastive_loss_base(
    cos_sim: torch.Tensor,
    temperature: float,
    nce_alpha: float = 1.0,
    ant_beta: float = 0.0,
    ant_margin: float = 0.1,
    ant_max_global: bool = True,
    infonce_max_global: bool = True,
    ant_symmetric_full: bool = False,
    ant_formulation: str = "logsumexp",
    ant_tau: float = 0.1,
    ant_topk: int = 32,
) -> torch.Tensor:
    """Combina InfoNCE e ANT."""
    ant_matrix, valid_negatives = _build_ant_matrix(
        cos_sim,
        ant_symmetric_full,
    )
    ant_loss = _compute_ant_loss(
        ant_matrix,
        valid_negatives,
        margin=ant_margin,
        max_global=ant_max_global,
        formulation=ant_formulation,
        tau=ant_tau,
        topk=ant_topk,
    )
    infonce_loss = _compute_infonce_loss(
        cos_sim,
        temperature,
    )

    return nce_alpha * infonce_loss + ant_beta * ant_loss


def infoNCE_loss(
    feats: torch.Tensor,
    t: float,
    nce_alpha: float = 1.0,
    ant_beta: float = 0.0,
    ant_margin: float = 0.1,
    ant_max_global: bool = True,
    infonce_max_global: bool = True,
    ant_symmetric_full: bool = False,
    ant_formulation: str = "logsumexp",
    ant_tau: float = 0.1,
    ant_topk: int = 32,
) -> torch.Tensor:
    """InfoNCE sobre duas visões do mesmo batch, com ANT opcional."""
    # normalize→matmul avoids the [N,N,D] intermediate tensor (OOM for large images)
    feats_n = F.normalize(feats, dim=-1)
    cos_sim = feats_n @ feats_n.T

    return _compute_contrastive_loss_base(
        cos_sim,
        temperature=t,
        nce_alpha=nce_alpha,
        ant_beta=ant_beta,
        ant_margin=ant_margin,
        ant_max_global=ant_max_global,
        infonce_max_global=infonce_max_global,
        ant_symmetric_full=ant_symmetric_full,
        ant_formulation=ant_formulation,
        ant_tau=ant_tau,
        ant_topk=ant_topk,
    )


def infoNCE_distill_loss(
    p_feats: torch.Tensor,
    z_feats: torch.Tensor,
    t: float,
    nce_alpha: float = 1.0,
    ant_beta: float = 0.0,
    ant_margin: float = 0.1,
    ant_max_global: bool = True,
    infonce_max_global: bool = True,
    ant_symmetric_full: bool = False,
    ant_formulation: str = "logsumexp",
    ant_tau: float = 0.1,
    ant_topk: int = 32,
) -> torch.Tensor:
    """InfoNCE de destilação entre as features do aluno e do professor."""
    # normalize→matmul avoids the [M,N,D] intermediate tensor (OOM for large images)
    p_n = F.normalize(p_feats, dim=-1)
    z_n = F.normalize(z_feats, dim=-1)
    cos_sim = p_n @ z_n.T

    return _compute_contrastive_loss_base(
        cos_sim,
        temperature=t,
        nce_alpha=nce_alpha,
        ant_beta=ant_beta,
        ant_margin=ant_margin,
        ant_max_global=ant_max_global,
        infonce_max_global=infonce_max_global,
        ant_symmetric_full=ant_symmetric_full,
        ant_formulation=ant_formulation,
        ant_tau=ant_tau,
        ant_topk=ant_topk,
    )
