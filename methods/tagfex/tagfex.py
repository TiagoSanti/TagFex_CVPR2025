import collections
import copy
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from torch.utils.data import Subset
from modules import HerdingIndicesLearner
from modules.data.dataset import WithIndexDataset
from .tagfexnet import TagFexNet
from modules import (
    Accuracy,
    MeanMetric,
    CatMetric,
    select_metrics,
    forward_metrics,
    get_metrics,
)
from modules import optimizer_dispatch, scheduler_dispatch, get_loaders
from utils.funcs import parameter_count

from loggers import LoguruLogger, loguru

EPSILON = 1e-8


def _sbs_keep_mask(speeds: np.ndarray, q: float, s: float, min_keep: int) -> np.ndarray:
    """Return a boolean mask keeping the intermediate learning-speed band.

    Drops the bottom *s* fraction (slow-learned, likely noisy outliers) and the
    top *q* fraction (fast-learned, trivial samples already well-generalised).
    Implements the SBS-fix criterion from Hacohen & Tuytelaars (ICML 2025).

    Falls back to keeping all samples if the filter would leave fewer than
    *min_keep* samples (safety guard for small classes).
    """
    n = len(speeds)
    n_drop_s = int(np.floor(n * s))   # drop slowest
    n_drop_q = int(np.floor(n * q))   # drop fastest
    if n_drop_s + n_drop_q >= n - min_keep:
        return np.ones(n, dtype=bool)
    order = np.argsort(speeds)         # ascending: index 0 = slowest
    drop = set(order[:n_drop_s].tolist())
    if n_drop_q > 0:
        drop.update(order[n - n_drop_q:].tolist())
    mask = np.ones(n, dtype=bool)
    for i in drop:
        mask[i] = False
    return mask


@torch.no_grad()
def _avg_state_dicts(state_dicts):
    """Average a list of state dicts element-wise (float tensors averaged; others use last)."""
    keys = list(state_dicts[0].keys())
    n = len(state_dicts)
    avg = {}
    for k in keys:
        ref = state_dicts[0][k]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for sd in state_dicts:
                acc.add_(sd[k].float())
            avg[k] = (acc / n).to(ref.dtype)
        else:
            avg[k] = state_dicts[-1][k].clone()
    return avg


class TagFex(HerdingIndicesLearner):
    def __init__(self, data_maganger, configs: dict, device, distributed=False) -> None:
        super().__init__(data_maganger, configs, device, distributed)

        self._init_network(
            self.configs.get("backbone_configs", dict()),
            self.configs.get("network_configs", dict()),
        )

        if self.distributed is not None:
            self._init_ddp()

        self._adjust_log_dir_with_loss_params()
        self._init_loggers()
        self._init_similarity_debug_logger()
        self.print_logger.info(configs)

        self.print_logger.info(f"class order: {self.data_manager.class_order.tolist()}")
        self.ordered_index_map = torch.from_numpy(
            self.data_manager.ordered_index_map
        ).to(self.device)

        # ── SBS (Speed-Based Sampling) state ─────────────────────────────────
        self._sbs_q: float = float(self.configs.get("sbs_q", 0.0))  # drop fastest q%
        self._sbs_s: float = float(self.configs.get("sbs_s", 0.0))  # drop slowest s%
        self._sbs_tracking: bool = False          # True during SBS-enabled task
        self._sbs_correct: np.ndarray | None = None   # per-local-idx correct count
        self._sbs_total: np.ndarray | None = None     # per-local-idx attempt count
        self._sbs_n_new: int = 0                  # #new-class samples before memory
        self._sbs_task_new_abs: np.ndarray | None = None  # absolute base-dataset idxs
        self._sbs_speed_map: dict = {}            # abs_idx → learning_speed (post-task)

    def _init_network(self, backbone_configs, network_configs):
        if backbone_configs["name"] == "resnet18":
            params = {
                "dataset_name": self.data_manager.dataset_name,
                "small_base": (
                    self.data_manager.init_num_cls == self.data_manager.inc_num_cls
                ),
            }
            backbone_configs.update(params=params)
            if network_configs.get("new_backbone_configs"):
                network_configs["new_backbone_configs"].update(params=params)

        self.network = TagFexNet(backbone_configs, network_configs, self.device)
        self.local_network = self.network  # for ddp compatibility

        self.last_ta_net = None
        self.last_projector = None

        # Avg-K teacher buffers (populated during train_task)
        self._avg_last_k = self.configs.get("avg_last_k", 0)
        self._ckpt_buf_ta = None
        self._ckpt_buf_proj = None

    def _init_ddp(self):
        self.configs["trainloader_params"]["batch_size"] //= self.distributed[
            "world_size"
        ]
        torch.distributed.barrier()

    def _model_to_ddp(self):
        self.network = nn.parallel.DistributedDataParallel(
            self.local_network,
            device_ids=[self.distributed["rank"]],
            find_unused_parameters=self.configs["debug"],
        )

    @torch.no_grad()
    def extract_herding_features(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.Tensor:
        # self.print_logger.debug(f'dataset length: {len(dataset.indices)}')
        if self.configs["ffcv"]:
            from modules.data.ffcv.loader import (
                get_ffcv_loader,
                default_transform_dict,
                OrderOption,
            )

            dataset_name = self.configs["dataset_name"].lower()
            pipline_name = self.configs.get(
                "test_transform",
                default_transform_dict[dataset_name.strip("0123456789")][0],
            )
            data_loader = get_ffcv_loader(
                dataset,
                self.configs["train_beton_path"],
                pipline_name,
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
                num_workers=0,  # extremely slow when setting to >0
            )

        # self.print_logger.debug(f'dataloader length: {len(data_loader)}')
        self.local_network.eval()
        features = []
        for samples, targets in data_loader:
            samples = samples.to(self.device, non_blocking=True)
            feats = self.local_network(samples.contiguous())["ts_features"]
            feats = torch.cat(feats, dim=-1)
            features.append(feats)
        features = torch.cat(features)
        features /= features.norm(dim=-1, keepdim=True) + EPSILON
        return features

    @loguru.logger.catch(reraise=True)
    def train(self) -> None:
        # >>> @train_start
        self.update_state(run_state="train", num_tasks=self.data_manager.num_tasks)
        self.run_metrics = {
            "acc1_curve": CatMetric(sync_on_compute=False).to(self.device),
            "acc5_curve": CatMetric(sync_on_compute=False).to(self.device),
            "nme1_curve": CatMetric(sync_on_compute=False).to(self.device),
            "nme5_curve": CatMetric(sync_on_compute=False).to(self.device),
            "avg_acc1": MeanMetric().to(self.device),
            "avg_acc5": MeanMetric().to(self.device),
            "avg_nme1": MeanMetric().to(self.device),
            "avg_nme5": MeanMetric().to(self.device),
        }
        # <<< @train_start

        for task_id, (task_train, task_test) in enumerate(self.data_manager.tasks):
            # >>> @train_task_start
            cur_task_num_classes = self.data_manager.task_num_cls[task_id]
            sofar_num_classes = sum(self.data_manager.task_num_cls[: task_id + 1])
            learned_num_classes = sofar_num_classes - cur_task_num_classes
            self.update_state(
                cur_task=task_id + 1,
                cur_task_num_classes=cur_task_num_classes,
                sofar_num_classes=sofar_num_classes,
                learned_num_classes=learned_num_classes,
            )

            if task_id > 0:
                if self._avg_last_k > 0 and self._ckpt_buf_ta is not None and len(self._ckpt_buf_ta) > 0:
                    self.last_ta_net = self._build_avg_ta()
                    self.last_projector = self._build_avg_projector()
                    self.print_logger.info(
                        f"Avg-K teacher built from {len(self._ckpt_buf_ta)} checkpoints (K={self._avg_last_k})."
                    )
                else:
                    self.last_ta_net = self.local_network.get_freezed_copy_ta()
                    self.last_projector = self.local_network.get_freezed_copy_projector()

            self.local_network.update_network(cur_task_num_classes)
            self.local_network.freeze_old_backbones()
            if (
                self.configs["ckpt_path"] is not None
                and self.configs["ckpt_task"] is not None
                and task_id + 1 <= self.configs["ckpt_task"]
            ):
                if task_id + 1 == self.configs["ckpt_task"]:
                    self._load_checkpoint(self.configs["ckpt_path"])
                continue

            total, trainable = parameter_count(self.local_network)
            self.print_logger.info(
                f"{self._get_status()} | parameters: {total} in total, {trainable} trainable."
            )

            if self.distributed is not None:
                self._model_to_ddp()

            # add memory into training set.
            # >>> @sbs_pre_memory — save new-class count before memory is concatenated
            _sbs_enabled = (self._sbs_q > 0 or self._sbs_s > 0) and not self.configs.get("ffcv") and self.distributed is None
            if _sbs_enabled:
                self._sbs_n_new = len(task_train.indices)
                self._sbs_task_new_abs = task_train.indices.copy()
            # <<< @sbs_pre_memory
            memory_indices = self.get_memory()
            new_indices = np.concatenate((task_train.indices, memory_indices))
            task_train.indices = new_indices

            trainloader_params = self.configs["trainloader_params"].copy()

            # get dataloaders
            if self.configs["ffcv"]:
                from modules.data.ffcv.loader import get_ffcv_loaders

                train_loader, test_loader = get_ffcv_loaders(
                    task_train,
                    task_test,
                    trainloader_params,
                    self.configs["testloader_params"],
                    self.device,
                    self.configs,
                    self.distributed is not None,
                )
            else:
                # >>> @sbs_wrap_loader
                _train_src = WithIndexDataset(task_train) if _sbs_enabled else task_train
                if _sbs_enabled:
                    n_total = len(task_train)
                    self._sbs_correct = np.zeros(n_total, dtype=np.int32)
                    self._sbs_total = np.zeros(n_total, dtype=np.int32)
                    self._sbs_tracking = True
                    self.print_logger.info(
                        f"SBS enabled (q={self._sbs_q}, s={self._sbs_s}): "
                        f"tracking {self._sbs_n_new} new-class samples."
                    )
                # <<< @sbs_wrap_loader
                train_loader, test_loader = get_loaders(
                    _train_src,
                    task_test,
                    trainloader_params,
                    self.configs["testloader_params"],
                    self.distributed,
                )
            # <<< @train_task_start

            self.train_task(train_loader, test_loader)

            # >>> @sbs_finalize — build speed map after training, before memory update
            if self._sbs_tracking:
                self._sbs_speed_map = self._sbs_finalize_speeds()
            # <<< @sbs_finalize

            # >>> @train_task_end
            self.print_logger.info(
                f"Adjust memory to {self.num_exemplars_per_class} per class ({self.num_exemplars_per_class * self.state['sofar_num_classes']} in total)."
            )
            self.reduce_memory()
            self.update_memory()

            if task_id > 0:
                self.print_logger.info("Weight align before task end evaluation.")
                self.local_network.weight_align(cur_task_num_classes)

            # evaluation as task end
            results = self.eval_epoch(test_loader)
            forward_metrics(
                select_metrics(self.run_metrics, "acc1"), results["eval_acc1"]
            )
            forward_metrics(
                select_metrics(self.run_metrics, "acc5"), results["eval_acc5"]
            )
            forward_metrics(
                select_metrics(self.run_metrics, "nme1"), results["eval_nme1"]
            )
            forward_metrics(
                select_metrics(self.run_metrics, "nme5"), results["eval_nme5"]
            )
            self.print_logger.success(
                f"{self._get_status()}\n├> {self._metric_repr(results | get_metrics(self.run_metrics))}"
            )
            if (
                self.configs["ckpt_dir"] is not None
                and not self.configs["disable_save_ckpt"]
                and task_id + 1 in self.configs["save_ckpt_tasks"]
            ):
                self._save_checkpoint()
            # <<< @train_task_end

        # >>> @train_end
        results = get_metrics(self.run_metrics)
        self.print_logger.success(
            f"{self._get_status()}\n├> {self._metric_repr(results)}"
        )

        self.update_state(run_state="finished")
        # <<< @train_end

    def train_task(self, train_loader, test_loader):
        # get optimizer and lr scheduler
        if self.state["cur_task"] == 1:
            optimizer = optimizer_dispatch(
                self.local_network.parameters(), self.configs["init_optimizer_configs"]
            )
            scheduler = scheduler_dispatch(
                optimizer, self.configs["init_scheduler_configs"]
            )

            num_epochs = self.configs["init_epochs"]  # FIXME: config category
        else:
            optimizer = optimizer_dispatch(
                self.local_network.parameters(), self.configs["inc_optimizer_configs"]
            )
            scheduler = scheduler_dispatch(
                optimizer, self.configs["inc_scheduler_configs"]
            )

            num_epochs = self.configs["inc_epochs"]  # FIXME: config category

        # >>> @after_train_task_setups
        if self.configs["debug"]:
            num_epochs = 5
        self.update_state(cur_task_num_epochs=num_epochs)
        if self._avg_last_k > 0:
            self._reset_ckpt_buffer()
        # <<< @after_train_task_setups

        rank = 0 if self.distributed is None else self.distributed["rank"]
        prog_bar = (
            tqdm(
                range(num_epochs),
                desc=f"Task {self.state['cur_task']}/{self.state['num_tasks']}",
            )
            if rank == 0
            else range(num_epochs)
        )
        for epoch in prog_bar:
            # >>> @train_epoch_start
            self.update_state(cur_epoch=epoch + 1, num_batches=len(train_loader))
            self.add_state(accumulated_cur_epoch=1)
            if self.distributed is not None and not self.configs["ffcv"]:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)

            # <<< @train_epoch_start

            train_results = self.train_epoch(train_loader, optimizer, scheduler)

            # >>> @train_epoch_end
            if epoch % self.configs["eval_interval"] == 0:
                eval_results = self.eval_epoch(test_loader)
                self.print_logger.info(
                    f"{self._get_status()} | {self._metric_repr(train_results)} {self._metric_repr(eval_results)}"
                )
            else:
                self.print_logger.info(
                    f"{self._get_status()} | {self._metric_repr(train_results)}"
                )
            # <<< @train_epoch_end
            # >>> @avg_k_checkpoint
            if self._avg_last_k > 0 and epoch >= num_epochs - self._avg_last_k:
                self._append_ckpt()
            # <<< @avg_k_checkpoint
        if rank == 0:
            prog_bar.close()

    def _reset_ckpt_buffer(self):
        self._ckpt_buf_ta = collections.deque(maxlen=self._avg_last_k)
        self._ckpt_buf_proj = collections.deque(maxlen=self._avg_last_k)

    @torch.no_grad()
    def _append_ckpt(self):
        net = self.local_network
        ta_sd = {k: v.detach().clone() for k, v in net.ta_net.state_dict().items()}
        proj_sd = {k: v.detach().clone() for k, v in net.projector.state_dict().items()}
        self._ckpt_buf_ta.append(ta_sd)
        self._ckpt_buf_proj.append(proj_sd)

    def _build_avg_ta(self):
        avg_sd = _avg_state_dicts(list(self._ckpt_buf_ta))
        ta_copy = copy.deepcopy(self.local_network.ta_net)
        ta_copy.load_state_dict(avg_sd)
        for p in ta_copy.parameters():
            p.requires_grad_(False)
        return ta_copy.eval()

    def _build_avg_projector(self):
        avg_sd = _avg_state_dicts(list(self._ckpt_buf_proj))
        proj_copy = copy.deepcopy(self.local_network.projector)
        proj_copy.load_state_dict(avg_sd)
        for p in proj_copy.parameters():
            p.requires_grad_(False)
        return proj_copy.eval()

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.network.train()
        num_classes = self.state["sofar_num_classes"].item()
        metrics = {
            "loss": MeanMetric().to(self.device),
            "cls_loss": MeanMetric().to(self.device),
            "contrast_loss": MeanMetric().to(self.device),
            "train_acc1": Accuracy(task="multiclass", num_classes=num_classes).to(
                self.device
            ),
        }
        if self.state["cur_task"] > 1:
            metrics.update(
                {
                    "trans_cls_loss": MeanMetric().to(self.device),
                    "transfer_loss": MeanMetric().to(self.device),
                    "aux_loss": MeanMetric().to(self.device),
                    "kd_loss": MeanMetric().to(self.device),
                }
            )

        for batch, batch_data in enumerate(train_loader):
            batch_data = tuple(
                data.to(self.device, non_blocking=True) for data in batch_data
            )

            if self._sbs_tracking:
                local_idx, sample1, sample2, targets = batch_data
            elif self.configs.get("ffcv"):
                sample1, targets, sample2 = batch_data
            else:
                sample1, sample2, targets = batch_data
            targets = self.ordered_index_map[
                targets.flatten()
            ]  # map to continual class id.
            samples = torch.cat((sample1, sample2))
            targets = torch.cat((targets, targets))
            # self.print_logger.debug(f'train {batch}/{len(train_loader)}', samples.device, targets.device)
            # self.print_logger.debug(f'batch shape {samples.shape}')

            # >>> @train_batch_start
            self.update_state(cur_batch=batch + 1)
            # <<< @train_batch_start

            # >>> @train_forward
            out = self.network(samples.contiguous())
            logits = out["logits"]
            # self.print_logger.debug(f'rank {self.distributed["rank"]}, batch {batch}, logits: {logits.shape}, has NaN: {torch.isnan(logits).any()}')
            cls_loss = F.cross_entropy(logits, targets)
            # self.print_logger.debug(f'rank {self.distributed["rank"]}, batch {batch}, cls_loss: {cls_loss}')

            embedding = out["embedding"]

            # Get debug parameters if similarity debugging is enabled, gated by
            # epoch-interval and max-batches-per-epoch sampling controls.
            debug_logger = None
            heatmap_dir = None
            if self.configs.get("debug_similarity", False):
                _interval = self.configs.get("debug_similarity_epoch_interval", 1)
                _max_batches = self.configs.get(
                    "debug_similarity_batches_per_epoch", 9999
                )
                _cur_task = self.state["cur_task"]
                _cur_epoch = self.state["cur_epoch"]
                _max_epoch = self.configs.get(
                    "init_epochs" if _cur_task == 1 else "inc_epochs", 200
                )
                _epoch_sampled = (
                    _cur_epoch == 1
                    or _cur_epoch % _interval == 0
                    or _cur_epoch == _max_epoch
                )
                if _epoch_sampled and (batch + 1) <= _max_batches:
                    debug_logger = getattr(self, "similarity_debug_logger", None)
                    heatmap_dir = getattr(self, "heatmap_dir", None)

            ant_beta = self.configs.get("ant_beta", 0.0)
            ant_max_global = self.configs.get("ant_max_global", True)
            infonce_max_global_cfg = self.configs.get("infonce_max_global", None)
            ant_symmetric_full = self.configs.get("ant_symmetric_full", False)
            ant_formulation = self.configs.get("ant_formulation", "logsumexp")
            ant_tau = self.configs.get("ant_tau", 0.1)
            ant_topk = self.configs.get("ant_topk", 32)

            # Backward compatibility:
            # - ANT disabled: InfoNCE follows ant_max_global (legacy baseline behavior)
            # - ANT enabled: InfoNCE stays global unless explicitly configured
            if infonce_max_global_cfg is None:
                infonce_max_global = ant_max_global if ant_beta == 0.0 else True
            else:
                infonce_max_global = infonce_max_global_cfg

            infonce_loss = infoNCE_loss(
                embedding,
                self.configs["infonce_temp"],
                self.configs.get("nce_alpha", 1.0),
                ant_beta,
                self.configs.get("ant_margin", 0.1),
                ant_max_global,
                infonce_max_global,
                ant_symmetric_full,
                ant_formulation=ant_formulation,
                ant_tau=ant_tau,
                ant_topk=ant_topk,
                logger=self.loguru_logger,
                task=self.state["cur_task"],
                epoch=self.state["cur_epoch"],
                batch=batch + 1,
                debug_logger=debug_logger,
                heatmap_dir=heatmap_dir,
            )

            if (aux_logits := out.get("aux_logits")) is not None:
                learned_num_classes = self.state["learned_num_classes"]
                aux_targets = torch.where(
                    targets >= learned_num_classes, targets - learned_num_classes + 1, 0
                )
                aux_loss = F.cross_entropy(aux_logits, aux_targets)
                # self.print_logger.debug(f'rank {self.distributed["rank"]}, batch {batch}, aux_loss: {aux_loss}')

                predicted_feature = out["predicted_feature"]
                old_ta_feature = self.last_ta_net(samples.contiguous())["features"]
                kd_loss = infoNCE_distill_loss(
                    self.last_projector(predicted_feature),
                    self.last_projector(old_ta_feature),
                    self.configs["infonce_kd_temp"],
                    self.configs.get("nce_alpha", 1.0),
                    ant_beta,
                    self.configs.get("ant_margin", 0.1),
                    ant_max_global,
                    infonce_max_global,
                    ant_symmetric_full,
                    ant_formulation=ant_formulation,
                    ant_tau=ant_tau,
                    ant_topk=ant_topk,
                    logger=self.loguru_logger,
                    task=self.state["cur_task"],
                    epoch=self.state["cur_epoch"],
                    batch=batch + 1,
                    debug_logger=debug_logger,
                    heatmap_dir=heatmap_dir,
                )

                trans_logits = out["trans_logits"]
                cur_task_mask = targets >= learned_num_classes
                # DEBUG: log trans_logits stats at first batch of each incremental task
                if batch == 0 and self.state["cur_epoch"] == 1:
                    _tl = trans_logits[cur_task_mask]
                    _mf = out.get("merged_feature")
                    self.print_logger.warning(
                        f"[DEBUG T{self.state['cur_task']}E1B1] trans_logits: shape={_tl.shape} "
                        f"mean={_tl.mean():.4f} std={_tl.std():.4f} min={_tl.min():.4f} max={_tl.max():.4f} "
                        f"| cur_task_mask n_selected={cur_task_mask.sum().item()} "
                        f"| learned_num_classes={learned_num_classes.item() if hasattr(learned_num_classes, 'item') else learned_num_classes}"
                    )
                    if _mf is not None:
                        self.print_logger.warning(
                            f"[DEBUG T{self.state['cur_task']}E1B1] merged_feature: shape={_mf.shape} "
                            f"mean={_mf.mean():.4f} std={_mf.std():.4f} min={_mf.min():.4f} max={_mf.max():.4f}"
                        )
                trans_cls_loss = F.cross_entropy(
                    trans_logits[cur_task_mask],
                    targets[cur_task_mask] - learned_num_classes,
                )
                # self.print_logger.debug(f'rank {self.distributed["rank"]}, batch {batch}, trans_cls_loss: {trans_cls_loss}')

                if trans_cls_loss < cls_loss:
                    T = self.configs["kd_temp"]
                    transfer_loss = F.kl_div(
                        (
                            logits[cur_task_mask][:, learned_num_classes:] / T
                        ).log_softmax(dim=1),
                        (trans_logits.detach()[cur_task_mask] / T).softmax(dim=1),
                        reduction="batchmean",
                    )
                else:
                    transfer_loss = torch.tensor(0.0, device=self.device)

                sofar_num_classes = self.state["sofar_num_classes"]
                auto_kd_factor = learned_num_classes / sofar_num_classes
                loss = (
                    cls_loss
                    + self.configs["aux_factor"] * aux_loss
                    + self.configs["contrast_factor"]
                    * (
                        infonce_loss * (1 - auto_kd_factor)
                        + self.configs["contrast_kd_factor"] * kd_loss * auto_kd_factor
                    )
                    + self.configs["trans_cls_factor"] * trans_cls_loss
                    + self.configs["transfer_factor"] * transfer_loss
                )
            else:
                loss = cls_loss + self.configs["contrast_factor"] * infonce_loss
            # <<< @train_forward

            # >>> @train_backward
            optimizer.zero_grad()
            loss.backward()
            grad_clip_norm = self.configs.get("grad_clip_norm", None)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.local_network.parameters(), max_norm=grad_clip_norm
                )
            optimizer.step()
            # >>> @sbs_record — accumulate per-sample correctness (new-class samples only)
            if self._sbs_tracking:
                self._sbs_record_batch(local_idx, logits, targets)
            # <<< @sbs_record
            # <<< @train_backward

            # >>> @train_batch_end
            metrics["loss"].update(loss.detach())
            metrics["cls_loss"].update(cls_loss.detach())
            metrics["contrast_loss"].update(infonce_loss.detach())
            metrics["train_acc1"].update(logits.detach(), targets.detach())
            if self.state["cur_task"] > 1:
                metrics["trans_cls_loss"].update(trans_cls_loss.detach())
                metrics["transfer_loss"].update(transfer_loss.detach())
                metrics["aux_loss"].update(aux_loss.detach())
                metrics["kd_loss"].update(kd_loss.detach())

            # <<< @train_batch_end

        scheduler.step()
        train_results = get_metrics(metrics)
        return train_results

    # ── SBS helpers ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def _sbs_record_batch(
        self,
        local_idx: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate per-sample correctness for SBS speed tracking.

        Both *logits* and *targets* are the doubled version produced by
        ``torch.cat((sample1_batch, sample2_batch))`` — only the first half
        (corresponding to *sample1*) is used here to avoid double-counting.
        *local_idx* always has batch-size B (not 2B).
        """
        bs = local_idx.shape[0]
        preds = logits[:bs].argmax(dim=1)
        correct = (preds == targets[:bs]).cpu().numpy().astype(np.int32)
        idx_np = local_idx.cpu().numpy()
        # Only accumulate for new-class positions (local_idx < _sbs_n_new).
        new_mask = idx_np < self._sbs_n_new
        if new_mask.any():
            np.add.at(self._sbs_correct, idx_np[new_mask], correct[new_mask])
            np.add.at(self._sbs_total, idx_np[new_mask], 1)

    def _sbs_finalize_speeds(self) -> dict:
        """Compute learning_speed = correct_count / epoch_count for new-class samples.

        Returns a mapping ``{abs_dataset_idx: float speed}`` that aligns with
        the absolute CIFAR-100 (or other base-dataset) indices stored in
        ``self._sbs_task_new_abs``.
        """
        n = self._sbs_n_new
        seen = np.maximum(self._sbs_total[:n], 1)
        speeds = self._sbs_correct[:n].astype(np.float32) / seen
        self.print_logger.info(
            f"SBS: new-class learning_speed — "
            f"mean={speeds.mean():.3f}  std={speeds.std():.3f}  "
            f"p10={np.percentile(speeds, 10):.3f}  p90={np.percentile(speeds, 90):.3f}"
        )
        return {int(ai): float(sp) for ai, sp in zip(self._sbs_task_new_abs, speeds)}

    # ── SBS override of update_memory ─────────────────────────────────────────

    @torch.no_grad()
    def update_memory(self) -> None:
        """Herding with optional SBS pre-filtering.

        When SBS is disabled (``sbs_q == sbs_s == 0``) or the run uses FFCV /
        distributed training, this delegates entirely to the parent-class
        implementation.  Otherwise it applies the SBS filter to each new class's
        candidate pool before running the standard iCaRL greedy herding.
        """
        if not self._sbs_tracking or (self._sbs_q == 0.0 and self._sbs_s == 0.0):
            super().update_memory()
            self._sbs_tracking = False
            return

        speed_by_abs = self._sbs_speed_map
        self._sbs_tracking = False  # reset before any early-return

        num_classes = self.state["sofar_num_classes"]
        cur_task_num_classes = self.state["cur_task_num_classes"]
        num_seen_classes = num_classes - cur_task_num_classes
        local_classes = np.arange(num_seen_classes, num_classes)

        selected_indices: list = []
        class_means: list = []

        prog_bar_desc = (
            f"Task {self.state['cur_task']}/{self.state['num_tasks']} "
            f"updmem-SBS [{local_classes[0]}~{local_classes[-1]}]"
            f"[{self.num_exemplars_per_class}]"
        )
        prog_bar = tqdm(local_classes, desc=prog_bar_desc)

        for class_id in prog_bar:
            inherent_class_id = self.data_manager.class_order[class_id].item()
            class_dataset = self.data_manager.get_dataset_by_class_ids(
                [inherent_class_id], split="train", mode="test"
            )
            abs_indices = np.array(class_dataset.indices)

            # Map absolute indices to SBS speeds; default 0.5 for unseen samples.
            speeds = np.array(
                [speed_by_abs.get(int(ai), 0.5) for ai in abs_indices],
                dtype=np.float32,
            )
            keep_mask = _sbs_keep_mask(
                speeds, self._sbs_q, self._sbs_s, self.num_exemplars_per_class
            )
            n_kept = int(keep_mask.sum())
            self.print_logger.info(
                f"SBS [class {class_id}]: {len(speeds)} → {n_kept} kept "
                f"(q={self._sbs_q}, s={self._sbs_s}) | "
                f"speed mean={speeds.mean():.3f} min={speeds.min():.3f} max={speeds.max():.3f}"
            )

            filtered_abs = abs_indices[keep_mask]
            assert n_kept >= self.num_exemplars_per_class, (
                f"SBS filter left only {n_kept} samples for class {class_id} "
                f"but need {self.num_exemplars_per_class}. "
                f"Reduce sbs_q/sbs_s or check class size."
            )

            filtered_dataset = Subset(class_dataset.dataset, filtered_abs)
            class_features = self.extract_herding_features(filtered_dataset)

            class_mean = class_features.mean(dim=0, keepdim=True)
            class_selected_indices: list = []
            selected_mean = torch.zeros(
                (self.local_network.feature_dim,), device=self.device
            )

            for n_sel in range(1, self.num_exemplars_per_class + 1):
                mu_p = ((n_sel - 1) * selected_mean + class_features) / n_sel
                idx = (mu_p - class_mean).norm(dim=-1).argmin().item()
                selected_mean = mu_p[idx]
                class_selected_indices.append(int(filtered_abs[idx]))
                mask = torch.arange(len(class_features)) != idx
                class_features = class_features[mask]
                filtered_abs = np.delete(filtered_abs, idx)

            selected_indices.append(class_selected_indices)
            selected_mean /= selected_mean.norm()
            class_means.append(selected_mean)

        prog_bar.close()
        self.memory_samples.extend(selected_indices)
        self.class_means.extend(class_means)

    @torch.no_grad()
    def eval_epoch(self, data_loader):
        # >>> @eval_start
        prev_run_state = self.state.get("run_state")
        self.update_state(run_state="eval")
        self.network.eval()
        # <<< @eval_start

        num_classes = self.state["sofar_num_classes"].item()
        acc_metrics = {
            "eval_acc1": Accuracy(task="multiclass", num_classes=num_classes).to(
                self.device
            ),
            "eval_acc5": Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            ).to(self.device),
            "eval_acc1_per_class": Accuracy(
                task="multiclass", average=None, num_classes=num_classes
            ).to(self.device),
            "eval_acc5_per_class": Accuracy(
                task="multiclass", average=None, num_classes=num_classes, top_k=5
            ).to(self.device),
        }
        if len(self.class_means) == self.state["sofar_num_classes"]:
            nme_metrics = {
                "eval_nme1": Accuracy(task="multiclass", num_classes=num_classes).to(
                    self.device
                ),
                "eval_nme5": Accuracy(
                    task="multiclass", num_classes=num_classes, top_k=5
                ).to(self.device),
                "eval_nme1_per_class": Accuracy(
                    task="multiclass", average=None, num_classes=num_classes
                ).to(self.device),
                "eval_nme5_per_class": Accuracy(
                    task="multiclass", average=None, num_classes=num_classes, top_k=5
                ).to(self.device),
            }

        # >>> @eval_epoch_start
        self.update_state(eval_num_batches=len(data_loader))
        # <<< @eval_epoch_start

        for batch, batch_data in enumerate(data_loader):
            batch_data = tuple(
                data.to(self.device, non_blocking=True) for data in batch_data
            )

            # >>> @eval_batch_start
            self.update_state(eval_cur_batch=batch + 1)
            # <<< @eval_batch_start

            samples, targets = batch_data
            targets = self.ordered_index_map[
                targets.flatten()
            ]  # map to continual class id.
            # self.print_logger.debug(f'eval {batch}/{len(data_loader)}', samples.device, targets.device)
            outs = self.network(samples.contiguous())
            logits = outs["logits"]
            forward_metrics(acc_metrics, logits, targets)

            if len(self.class_means) == self.state["sofar_num_classes"]:
                features = outs["ts_features"]
                features = torch.cat(features, dim=-1)
                features /= features.norm(dim=-1, keepdim=True) + EPSILON
                dists = torch.cdist(features, torch.stack(self.class_means))
                forward_metrics(nme_metrics, -dists, targets)

        acc_metric_results = get_metrics(acc_metrics)
        if len(self.class_means) == self.state["sofar_num_classes"]:
            nme_metric_results = get_metrics(nme_metrics)
        else:
            nme_metric_results = dict()

        # >>> @eval_end
        self.update_state(run_state=prev_run_state)
        # >>> @eval_end

        return acc_metric_results | nme_metric_results

    def _metric_repr(self, metric_results: dict):
        def merge_to_task(acc_per_cls):
            display_value = []
            accumuated_num_cls = 0
            for num_cls in self.data_manager.task_num_cls:
                tot_num_cls = accumuated_num_cls + num_cls
                if tot_num_cls > len(acc_per_cls):
                    break
                task_acc = acc_per_cls[accumuated_num_cls:tot_num_cls].mean()
                display_value.append(task_acc.item())
                accumuated_num_cls = tot_num_cls
            return display_value

        scalars = []
        vectors = []
        for key, value in metric_results.items():
            if "acc" in key or "nme" in key:
                display_value = value * 100
            else:
                display_value = value

            if value.dim() > 0:
                if "per_class" in key:
                    display_value = merge_to_task(display_value)
                    key = key.replace("per_class", "per_task")
                else:
                    display_value = display_value.cpu().tolist()
                [f"{v:.2f}" for v in display_value]
                r = f'{key} [{" ".join([f"{v:.2f}" for v in display_value])}]'
                vectors.append(r)
            else:
                r = f"{key} {display_value.item():.2f}"
                scalars.append(r)

        componets = []
        if len(scalars) > 0:
            componets.append(" ".join(scalars))
        if len(vectors) > 0:
            componets.append("\n├> ".join(vectors))
        s = "\n├> ".join(componets)
        return "\n└>".join(s.rsplit("\n├>", 1))

    def _init_similarity_debug_logger(self):
        """Initialize separate logger for similarity matrix debugging."""
        if not self.configs.get("debug_similarity", False):
            return

        from loguru import logger

        log_dir = Path(self.configs.get("log_dir"))
        debug_log_path = log_dir / "similarity_debug.log"

        # Add new handler for similarity debugging and store the handler ID
        self.similarity_debug_handler_id = logger.add(
            debug_log_path,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            enqueue=True,
        )

        self.similarity_debug_logger = logger.bind(sim_debug=True)
        self.print_logger.info(f"Similarity debug logging to: {debug_log_path}")

        # Create directory for heatmaps
        self.heatmap_dir = log_dir / "similarity_heatmaps"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        self.print_logger.info(
            f"Similarity heatmaps will be saved to: {self.heatmap_dir}"
        )

    def _adjust_log_dir_with_loss_params(self):
        """Adjust log directory name based on loss parameters (ANT, contrast factors, etc.)"""
        log_dir = self.configs.get("log_dir")
        if log_dir is None:
            return

        from pathlib import Path

        log_dir = Path(log_dir)

        # Build suffix based on parameters that differ from defaults
        suffix_parts = []

        # ANT parameters
        ant_beta = self.configs.get("ant_beta", 0.0)
        suffix_parts.append(f"antB{ant_beta:.3f}".rstrip("0").rstrip("."))

        nce_alpha = self.configs.get("nce_alpha", 1.0)
        suffix_parts.append(f"nceA{nce_alpha:.3f}".rstrip("0").rstrip("."))

        # Only include ant_margin if ant_beta > 0 (ANT is active)
        if ant_beta > 0:
            ant_margin = self.configs.get("ant_margin", 0.1)
            suffix_parts.append(f"antM{ant_margin:.3f}".rstrip("0").rstrip("."))

        ant_max_global = self.configs.get("ant_max_global", True)
        infonce_max_global = self.configs.get("infonce_max_global", ant_max_global)

        # Check if symmetric_full is enabled
        ant_symmetric_full = self.configs.get("ant_symmetric_full", False)
        if ant_symmetric_full:
            suffix_parts.append("antSymmetricFull")
        else:
            suffix_parts.append("antGlobal" if ant_max_global else "antLocal")
        suffix_parts.append("nceGlobal" if infonce_max_global else "nceLocal")

        # ANT loss formulation (only if non-default)
        ant_formulation = self.configs.get("ant_formulation", "logsumexp")
        if ant_formulation != "logsumexp":
            suffix_parts.append(f"form{ant_formulation}")

        # Avg-K teacher (only if enabled)
        avg_last_k = self.configs.get("avg_last_k", 0)
        if avg_last_k > 0:
            suffix_parts.append(f"avgK{avg_last_k}")

        # SBS (only if enabled)
        sbs_q = self.configs.get("sbs_q", 0.0)
        sbs_s = self.configs.get("sbs_s", 0.0)
        if sbs_q > 0 or sbs_s > 0:
            suffix_parts.append(f"sbsQ{sbs_q:.2f}S{sbs_s:.2f}".rstrip("0").rstrip("."))

        # Always append seed so every run has a unique, traceable directory
        seed = self.configs.get("seed", 1993)
        suffix_parts.append(f"s{seed}")

        # Contrast factors (optional - you can enable these if needed)
        if self.configs.get("include_contrast_in_logdir", False):
            contrast_factor = self.configs.get("contrast_factor", 1.0)
            if contrast_factor != 1.0:
                suffix_parts.append(f"cf{contrast_factor:.2f}".rstrip("0").rstrip("."))

            contrast_kd_factor = self.configs.get("contrast_kd_factor", 2.0)
            if contrast_kd_factor != 2.0:
                suffix_parts.append(
                    f"ckf{contrast_kd_factor:.2f}".rstrip("0").rstrip(".")
                )

        # Build new log directory path
        if suffix_parts:
            suffix = "_" + "_".join(suffix_parts)

            # If log_dir is just a base directory (like './logs'), create experiment subdirectory
            # Check if log_dir name is 'logs' or ends with 'logs'
            if log_dir.name == "logs" or str(log_dir) in ["./logs", "logs"]:
                # Create experiment name from dataset and scenario
                dataset_name = self.configs.get("dataset_name", "dataset")
                scenario = self.configs.get("scenario", "").split()[-1]
                exp_name = f"exp_{dataset_name}_{scenario}{suffix}"
                new_log_dir = log_dir / exp_name
            else:
                # Append suffix to existing experiment name
                new_log_dir = log_dir.parent / (log_dir.name + suffix)

            # Check if directory already exists and create a unique name if needed
            new_log_dir = self._get_unique_log_dir(new_log_dir)

            self.configs["log_dir"] = new_log_dir

    def _get_unique_log_dir(self, base_dir):
        """
        Get a unique log directory name. If the directory already exists,
        append a version number (v2, v3, etc.) to avoid overwriting.

        Args:
            base_dir: Path - The base directory path

        Returns:
            Path - A unique directory path that doesn't exist yet
        """
        from pathlib import Path

        base_dir = Path(base_dir)

        # If directory doesn't exist, use it as is
        if not base_dir.exists():
            return base_dir

        # Directory exists, find next available version
        version = 2
        while True:
            versioned_dir = base_dir.parent / f"{base_dir.name}_v{version}"
            if not versioned_dir.exists():
                return versioned_dir
            version += 1

    def _init_loggers(self):
        self.loguru_logger = LoguruLogger(
            self.configs, self.configs["disable_log_file"], tqdm_out=True
        )
        self.print_logger = self.loguru_logger.logger  # the actual logger

    def _get_status(self):
        if self.distributed is None:
            rank, world_size = 0, 1
        else:
            rank, world_size = self.distributed["rank"], self.distributed["world_size"]
        run_state = self.state.get("run_state")
        num_tasks = self.state.get("num_tasks")
        cur_task = self.state.get("cur_task")
        cur_task_num_classes = self.state.get("cur_task_num_classes")
        sofar_num_classes = self.state.get("sofar_num_classes")
        cur_task_num_epochs = self.state.get("cur_task_num_epochs")
        cur_epoch = self.state.get("cur_epoch")
        num_batches = self.state.get("num_batches")
        cur_batch = self.state.get("cur_batch")
        eval_num_batches = self.state.get("eval_num_batches")
        eval_cur_batch = self.state.get("eval_cur_batch")

        if run_state == "train":
            status = f"R{rank}T[{cur_task}/{num_tasks}]E[{cur_epoch}/{cur_task_num_epochs}] {run_state}"
        elif run_state == "eval":
            status = f"R{rank}T[{cur_task}/{num_tasks}]E[{cur_epoch}/{cur_task_num_epochs}] {run_state}"

        return status

    def state_dict(self) -> dict:
        super_dict = super().state_dict()
        d = {
            "network_state_dict": self.local_network.state_dict(),
            "run_metrics": {
                name: metric.state_dict() for name, metric in self.run_metrics.items()
            },
        }
        return super_dict | d

    def load_state_dict(self, d) -> None:
        super().load_state_dict(d)
        network_state_dict = d["network_state_dict"]
        self.local_network.load_state_dict(network_state_dict)

        run_metrics = d.get("run_metrics", dict())
        for name, state_dict in run_metrics.items():
            self.run_metrics[name].load_state_dict(state_dict)

    def _save_checkpoint(self):
        save_dict = self.state_dict()

        cur_task = self.state["cur_task"]
        num_tasks = self.state["num_tasks"]
        dataset_name = self.data_manager.dataset_name
        task_name, scenario = self.data_manager.scenario.split(" ")
        method = self.configs["method"]

        ckpt_file_name = (
            f"{method}_{dataset_name}_{scenario}_[{cur_task}_{num_tasks}].ckpt"
        )
        ckpt_dir = self.configs["ckpt_dir"]
        ckpt_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

        torch.save(save_dict, ckpt_dir / ckpt_file_name)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, self.device)
        self.load_state_dict(ckpt)


def _compute_contrastive_loss_base(
    cos_sim,
    t,
    nce_alpha=1.0,
    ant_beta=0.0,
    ant_margin=0.1,
    ant_max_global=True,
    infonce_max_global=True,
    ant_symmetric_full=False,
    ant_formulation="logsumexp",
    ant_tau=0.1,
    ant_topk=32,
    logger=None,
    log_prefix="contrast",
    task=None,
    epoch=None,
    batch=None,
    debug_logger=None,
    heatmap_dir=None,
):
    """
    Base function for computing contrastive loss with ANT.
    Used by both infoNCE_loss and infoNCE_distill_loss to avoid code duplication.

    Args:
        cos_sim: Pre-computed cosine similarity matrix [N, N] where N = 2 * batch_size
        t: Temperature parameter for InfoNCE loss
        nce_alpha: Weight for InfoNCE loss component
        ant_beta: Weight for ANT (Adaptive Negative Thresholding) loss component
        ant_margin: Margin threshold for ANT loss
        ant_max_global: If True, ANT uses global maximum across anchors; if False, per-anchor maximum
        infonce_max_global: If True, InfoNCE uses global normalization; if False, per-anchor normalization
        ant_symmetric_full: If True, ANT uses full symmetric matrix; if False, uses intra-view only
        ant_formulation: ANT loss variant - one of:
            "logsumexp"  (default) log(sum(exp(relu(v)))) — has count-floor = log(N)
            "expm1"      log1p(sum(expm1(relu(v)))) — zero contribution from non-violating
            "softplus"   mean(softplus(v/tau)) over valid pairs — smooth, count-normalised
            "topk"       logsumexp on top-k violations — focuses on hard negatives
            "active_only" mean(relu(v)) over active violations — direct violation severity
        ant_tau: Temperature for softplus formulation (default 0.1)
        ant_topk: Number of hard negatives for topk formulation (default 32)
        logger: Logger instance for recording statistics
        log_prefix: Prefix for log messages ("contrast" or "kd")
        task: Current task number for logging
        epoch: Current epoch number for logging
        batch: Current batch number for logging
        debug_logger: Separate logger for detailed similarity matrix debugging
        heatmap_dir: Directory to save heatmap visualizations

    Returns:
        total_loss: Combined loss value (InfoNCE + ANT)
    """
    device = cos_sim.device

    # Always define pos_start for positive similarities calculation
    pos_start = cos_sim.shape[0] // 2

    # Debug similarity matrices if enabled
    if debug_logger is not None and heatmap_dir is not None:
        _debug_similarity_matrices(
            cos_sim=cos_sim,
            ant_margin=ant_margin,
            max_global=ant_max_global,
            debug_logger=debug_logger,
            heatmap_dir=heatmap_dir,
            task=task,
            epoch=epoch,
            batch=batch,
            log_prefix=log_prefix,
        )

    # ANT (Adaptive Negative Thresholding) — build similarity matrix and valid-negative mask
    if ant_symmetric_full:
        # Full symmetric matrix: exclude self and positive pairs
        N = cos_sim.shape[0]
        self_mask_ant = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask_ant = self_mask_ant.roll(shifts=N // 2, dims=0)
        valid_neg_mask = ~(self_mask_ant | pos_mask_ant)  # [N, N]
        ant_sim_matrix = cos_sim.masked_fill(~valid_neg_mask, -float("inf"))
    else:
        # Original: intra-view block only (first half × first half)
        cos_sim_q1 = cos_sim[:pos_start, :pos_start]
        diag_mask_q1 = torch.eye(cos_sim_q1.shape[0], dtype=torch.bool, device=device)
        valid_neg_mask = ~diag_mask_q1  # [B, B]
        ant_sim_matrix = cos_sim_q1.masked_fill(diag_mask_q1, -float("inf"))

    # Positive similarities for gap / logging
    pos_sims = torch.diagonal(cos_sim[:pos_start, pos_start:])

    # Per-anchor valid-negative count — needed for floor correction
    num_negatives = valid_neg_mask.sum(dim=-1).float()  # [N] or [B]

    # Reference similarity per anchor (global or local max of valid negatives)
    if ant_max_global:
        ant_max = ant_sim_matrix.max()
    else:
        ant_max = ant_sim_matrix.max(dim=-1, keepdim=True).values  # [N,1] or [B,1]

    # Raw violation values before ReLU: v_i = neg_sim_i - ref_sim + margin
    raw_v = ant_sim_matrix - ant_max + ant_margin  # same shape as ant_sim_matrix

    # Compute ANT loss per anchor based on chosen formulation
    if ant_formulation == "logsumexp":
        # log(sum(exp(relu(v)))) over valid negatives only.
        # After relu, masked positions (-inf → 0) would contribute exp(0)=1 each,
        # inflating the count floor. Re-mask them to -inf so only valid pairs enter
        # the logsumexp. Non-violating valid negatives still contribute exp(0)=1,
        # so the count floor remains log(num_valid_negatives) — but no longer
        # includes self/positive slots.
        mq = torch.relu(raw_v)
        mq = mq.masked_fill(~valid_neg_mask, float("-inf"))
        ant_loss_per_anchor = torch.logsumexp(mq, dim=-1)

    elif ant_formulation == "expm1":
        # log1p(sum(expm1(relu(v)))) — non-violating and masked contribute exactly 0
        relu_v = torch.relu(raw_v)
        violation_mass = torch.expm1(relu_v)  # exp(relu(v)) - 1; 0 when v <= 0
        ant_loss_per_anchor = torch.log1p(violation_mass.sum(dim=-1))

    elif ant_formulation == "softplus":
        # mean(softplus(v/tau)) over valid pairs — smooth, no hard ReLU floor
        sp = F.softplus(raw_v / ant_tau)
        sp = sp * valid_neg_mask.float()  # zero masked positions explicitly
        ant_loss_per_anchor = sp.sum(dim=-1) / num_negatives.clamp_min(1.0)

    elif ant_formulation == "topk":
        # logsumexp on top-k hardest negatives — focuses pressure on hard pairs
        k = min(ant_topk, int(num_negatives.min().item()))
        if k > 0:
            v_for_topk = raw_v.masked_fill(~valid_neg_mask, float("-inf"))
            topk_v = torch.topk(v_for_topk, k=k, dim=-1).values
            mq_topk = torch.relu(topk_v)
            ant_loss_per_anchor = torch.logsumexp(mq_topk, dim=-1)
        else:
            ant_loss_per_anchor = torch.zeros(ant_sim_matrix.shape[0], device=device)

    elif ant_formulation == "active_only":
        # mean(relu(v)) over active violators — direct violation severity, no floor
        relu_v = torch.relu(raw_v)
        active_counts = (relu_v > 0).sum(dim=-1).clamp_min(1).float()
        ant_loss_per_anchor = relu_v.sum(dim=-1) / active_counts

    else:
        raise ValueError(
            f"Unknown ant_formulation '{ant_formulation}'. "
            "Choose from: logsumexp, expm1, softplus, topk, active_only"
        )

    ant_loss = ant_loss_per_anchor.mean()

    # Valid negative similarities for statistics
    neg_sims = ant_sim_matrix[valid_neg_mask]

    # Always log basic contrastive statistics for monitoring
    if logger is not None:
        pos_mean = pos_sims.mean()
        neg_mean = neg_sims.mean()
        basic_stats = {
            "pos_mean": pos_mean.item(),
            "pos_std": pos_sims.std().item(),
            "neg_mean": neg_mean.item(),
            "neg_std": neg_sims.std().item(),
            "current_gap": (pos_mean - neg_mean).item(),
        }
        logger.log_contrastive_stats(basic_stats, task=task, epoch=epoch, batch=batch)

    # Log detailed ANT-specific statistics when ANT is active
    if logger is not None and ant_beta > 0:
        # Gap: distance of each valid negative from the reference (ant_max)
        if ant_max_global:
            gaps = neg_sims - ant_max
        else:
            ant_max_expanded = ant_max.expand_as(ant_sim_matrix)
            gaps = (ant_sim_matrix - ant_max_expanded)[valid_neg_mask]

        # Classic violation percentage (mq > 0 in logsumexp terms, or raw_v > 0)
        raw_v_valid = raw_v[valid_neg_mask]
        violations = (raw_v_valid > 0).float()
        violation_pct = violations.mean().item() * 100

        ant_stats = {
            "pos_min": pos_sims.min().item(),
            "pos_max": pos_sims.max().item(),
            "neg_min": neg_sims.min().item(),
            "neg_max": neg_sims.max().item(),
            "gap_mean": gaps.mean().item(),
            "gap_std": gaps.std().item(),
            "gap_min": gaps.min().item(),
            "gap_max": gaps.max().item(),
            "margin": ant_margin,
            "violation_pct": violation_pct,
            "ant_loss": ant_loss.item(),
        }
        logger.log_ant_distance_stats(ant_stats, task=task, epoch=epoch, batch=batch)

        # --- ANT flattening diagnostics (Section 4 of investigation plan) ---
        # 4.1 Negative-set size
        num_neg_mean = num_negatives.mean()

        # 4.2 Raw violation values (pre-ReLU, over valid pairs only)
        raw_v_mean = raw_v_valid.mean()
        raw_v_min = raw_v_valid.min()
        raw_v_max = raw_v_valid.max()

        # 4.3-4.4 Active violation mask and severity
        active_mask = valid_neg_mask & (raw_v > 0)
        active_count_per_anchor = active_mask.sum(dim=-1).float()
        active_count_mean = active_count_per_anchor.mean()
        active_ratio_mean = (active_count_per_anchor / num_negatives.clamp_min(1.0)).mean()

        relu_v_all = torch.relu(raw_v_valid)  # [K_valid]
        viol_mean_all = relu_v_all.mean()

        if active_mask.any():
            relu_v_act = torch.relu(raw_v[active_mask])
            viol_mean_act = relu_v_act.mean()
            viol_max_act = relu_v_act.max()
        else:
            viol_mean_act = torch.zeros(1, device=device).squeeze()
            viol_max_act = torch.zeros(1, device=device).squeeze()

        # Percentile tail metrics of raw_v over valid negatives.
        # More informative than viol_max_act for ant_max_global=True, where the
        # max violator is always exactly gamma (structural artifact of the
        # local-max reference: v_max = s_max - s_max + gamma = gamma).
        raw_v_p90 = torch.quantile(raw_v_valid, 0.90)
        raw_v_p95 = torch.quantile(raw_v_valid, 0.95)

        # 4.5 Count-floor-adjusted ANT loss (only meaningful for logsumexp formulation)
        ant_floor_per_anchor = torch.log(num_negatives.clamp_min(1.0))
        ant_loss_adj = (ant_loss_per_anchor - ant_floor_per_anchor).mean()

        # 4.6 Hardest-negative similarity and positive–negative gap
        # Use first pos_start rows (one view) for comparison with pos_sims [B]
        hardest_neg_per_anchor = ant_sim_matrix[:pos_start].max(dim=-1).values  # [B]
        hard_neg_sim_mean = hardest_neg_per_anchor.mean()
        sim_gap = pos_sims - hardest_neg_per_anchor  # [B]
        sim_gap_mean = sim_gap.mean()
        sim_gap_min = sim_gap.min()

        flattening_stats = {
            "num_neg_mean": num_neg_mean.item(),
            "ant_loss_raw": ant_loss.item(),
            "ant_loss_adj": ant_loss_adj.item(),
            "active_ratio": active_ratio_mean.item(),
            "active_count_mean": active_count_mean.item(),
            "viol_mean_all": viol_mean_all.item(),
            "viol_mean_act": viol_mean_act.item(),
            "viol_max_act": viol_max_act.item(),
            "raw_v_p90": raw_v_p90.item(),
            "raw_v_p95": raw_v_p95.item(),
            "raw_v_mean": raw_v_mean.item(),
            "raw_v_min": raw_v_min.item(),
            "raw_v_max": raw_v_max.item(),
            "hard_neg_sim": hard_neg_sim_mean.item(),
            "sim_gap_mean": sim_gap_mean.item(),
            "sim_gap_min": sim_gap_min.item(),
        }
        logger.log_ant_flattening_diagnostics(
            flattening_stats, task=task, epoch=epoch, batch=batch
        )

    # Mask out cosine similarity to itself for InfoNCE
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Positive pair mask: batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    # InfoNCE loss with optional local anchor normalization
    cos_sim = cos_sim / t

    # Apply local anchor normalization for InfoNCE when configured.
    # This strategy is independent from the ANT anchor selection.
    if not infonce_max_global:
        cos_sim_neg = cos_sim.clone()
        cos_sim_neg[pos_mask] = -float("inf")
        max_neg_per_anchor = cos_sim_neg.max(dim=-1, keepdim=True).values
        cos_sim = cos_sim - max_neg_per_anchor

    # Compute InfoNCE loss
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll_mean = nll.mean()
    total_loss = nce_alpha * nll_mean + ant_beta * ant_loss

    # Log partial loss values
    if logger is not None:
        loss_components = {
            "nll": nll_mean.item(),
            "ant_loss": ant_loss.item(),
            "nce_weighted": (nce_alpha * nll_mean).item(),
            "ant_weighted": (ant_beta * ant_loss).item(),
            "total": total_loss.item(),
        }
        logger.log_loss_components(
            loss_components,
            prefix=log_prefix,
            task=task,
            epoch=epoch,
            batch=batch,
        )

    return total_loss


def infoNCE_loss(
    feats,
    t,
    nce_alpha=1.0,
    ant_beta=0.0,
    ant_margin=0.1,
    ant_max_global=True,
    infonce_max_global=True,
    ant_symmetric_full=False,
    ant_formulation="logsumexp",
    ant_tau=0.1,
    ant_topk=32,
    logger=None,
    task=None,
    epoch=None,
    batch=None,
    debug_logger=None,
    heatmap_dir=None,
):
    """
    InfoNCE contrastive loss with optional ANT.

    Args:
        feats: Feature embeddings [2*batch_size, feature_dim]
        t: Temperature parameter
        nce_alpha: Weight for InfoNCE loss
        ant_beta: Weight for ANT loss
        ant_margin: Margin threshold for ANT
        ant_max_global: If True, ANT uses global max; if False, per-anchor max
        infonce_max_global: If True, InfoNCE uses global normalization; if False, per-anchor normalization
        ant_symmetric_full: If True, ANT uses full symmetric matrix; if False, uses intra-view only
        ant_formulation: ANT loss variant (logsumexp, expm1, softplus, topk, active_only)
        ant_tau: Temperature for softplus formulation
        ant_topk: k for topk formulation
        logger: Logger instance
        task: Current task number
        epoch: Current epoch number
        batch: Current batch number
        debug_logger: Separate logger for detailed similarity matrix debugging
        heatmap_dir: Directory to save heatmap visualizations

    Returns:
        total_loss: Combined contrastive loss
    """
    # Compute cosine similarity matrix
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

    # Use base function to compute loss
    return _compute_contrastive_loss_base(
        cos_sim=cos_sim,
        t=t,
        nce_alpha=nce_alpha,
        ant_beta=ant_beta,
        ant_margin=ant_margin,
        ant_max_global=ant_max_global,
        infonce_max_global=infonce_max_global,
        ant_symmetric_full=ant_symmetric_full,
        ant_formulation=ant_formulation,
        ant_tau=ant_tau,
        ant_topk=ant_topk,
        logger=logger,
        log_prefix="contrast",
        task=task,
        epoch=epoch,
        batch=batch,
        debug_logger=debug_logger,
        heatmap_dir=heatmap_dir,
    )


def infoNCE_distill_loss(
    p_feats,
    z_feats,
    t,
    nce_alpha=1.0,
    ant_beta=0.0,
    ant_margin=0.1,
    ant_max_global=True,
    infonce_max_global=True,
    ant_symmetric_full=False,
    ant_formulation="logsumexp",
    ant_tau=0.1,
    ant_topk=32,
    logger=None,
    task=None,
    epoch=None,
    batch=None,
    debug_logger=None,
    heatmap_dir=None,
):
    """
    InfoNCE distillation loss with optional ANT.
    Used for knowledge distillation between predicted and old features.

    Args:
        p_feats: Predicted feature embeddings [2*batch_size, feature_dim]
        z_feats: Old (teacher) feature embeddings [2*batch_size, feature_dim]
        t: Temperature parameter
        nce_alpha: Weight for InfoNCE loss
        ant_beta: Weight for ANT loss
        ant_margin: Margin threshold for ANT
        ant_max_global: If True, ANT uses global max; if False, per-anchor max
        infonce_max_global: If True, InfoNCE uses global normalization; if False, per-anchor normalization
        ant_symmetric_full: If True, ANT uses full symmetric matrix; if False, uses intra-view only
        ant_formulation: ANT loss variant (logsumexp, expm1, softplus, topk, active_only)
        ant_tau: Temperature for softplus formulation
        ant_topk: k for topk formulation
        logger: Logger instance
        task: Current task number
        epoch: Current epoch number
        batch: Current batch number
        debug_logger: Separate logger for detailed similarity matrix debugging
        heatmap_dir: Directory to save heatmap visualizations

    Returns:
        total_loss: Combined distillation loss
    """
    # Compute cosine similarity matrix between predicted and old features
    cos_sim = F.cosine_similarity(p_feats[:, None, :], z_feats[None, :, :], dim=-1)

    # Use base function to compute loss
    return _compute_contrastive_loss_base(
        cos_sim=cos_sim,
        t=t,
        nce_alpha=nce_alpha,
        ant_beta=ant_beta,
        ant_margin=ant_margin,
        ant_max_global=ant_max_global,
        infonce_max_global=infonce_max_global,
        ant_symmetric_full=ant_symmetric_full,
        ant_formulation=ant_formulation,
        ant_tau=ant_tau,
        ant_topk=ant_topk,
        logger=logger,
        log_prefix="kd",
        task=task,
        epoch=epoch,
        batch=batch,
        debug_logger=debug_logger,
        heatmap_dir=heatmap_dir,
    )


def _debug_similarity_matrices(
    cos_sim,
    ant_margin,
    max_global,
    debug_logger,
    heatmap_dir,
    task,
    epoch,
    batch,
    log_prefix="contrast",
):
    """
    Debug and visualize similarity matrices with detailed analysis.

    This function creates:
    1. Text logs with matrix values, highlighting anchors and margin violations
    2. Heatmap visualizations showing similarity patterns
    3. Statistical summaries

    Args:
        cos_sim: Cosine similarity matrix [N, N]
        ant_margin: Margin threshold for ANT
        max_global: Whether using global or local max
        debug_logger: Logger instance for debug output
        heatmap_dir: Directory to save heatmap images
        task: Current task number
        epoch: Current epoch
        batch: Current batch
        log_prefix: Prefix for file naming
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert to numpy for easier manipulation
    cos_sim_np = cos_sim.detach().cpu().numpy()
    N = cos_sim_np.shape[0]
    batch_size = N // 2

    # Context string for logging
    context = f"T{task}_E{epoch}_B{batch}_{log_prefix}"

    debug_logger.info(f"\n{'='*80}")
    debug_logger.info(f"SIMILARITY MATRIX DEBUG - {context}")
    debug_logger.info(f"Matrix shape: {cos_sim_np.shape}")
    debug_logger.info(
        f"ANT margin: {ant_margin}, Max strategy: {'Global' if max_global else 'Local'}"
    )
    debug_logger.info(f"{'='*80}\n")

    # Split into first half (anchors) for analysis
    cos_sim_anchors = cos_sim_np[:batch_size, :batch_size]

    # Mask diagonal (self-similarity)
    mask_diag = np.eye(batch_size, dtype=bool)

    # Compute max for each anchor (or global)
    if max_global:
        global_max = np.max(cos_sim_anchors[~mask_diag])
        threshold = global_max - ant_margin
        debug_logger.info(f"Global max: {global_max:.4f}")
        debug_logger.info(f"Threshold (max - margin): {threshold:.4f}\n")
    else:
        local_maxs = np.max(np.where(mask_diag, -np.inf, cos_sim_anchors), axis=1)
        debug_logger.info(
            f"Local max per anchor: min={local_maxs.min():.4f}, "
            f"max={local_maxs.max():.4f}, mean={local_maxs.mean():.4f}\n"
        )

    # Analyze each anchor (show first 8 for readability)
    num_show = min(8, batch_size)
    for anchor_idx in range(num_show):
        debug_logger.info(f"--- Anchor {anchor_idx} ---")

        # Get similarities for this anchor (excluding self)
        sims = cos_sim_anchors[anchor_idx].copy()
        sims[anchor_idx] = np.nan  # Mark self-similarity as NaN

        # Compute threshold for this anchor
        if max_global:
            anchor_threshold = threshold
            anchor_max = global_max
        else:
            anchor_max = local_maxs[anchor_idx]
            anchor_threshold = anchor_max - ant_margin

        # Find positive pair (should be at batch_size + anchor_idx)
        pos_sim = cos_sim_np[anchor_idx, batch_size + anchor_idx]

        # Categorize values
        above_threshold = []
        below_threshold = []

        for neg_idx in range(batch_size):
            if neg_idx == anchor_idx:
                continue
            sim_val = sims[neg_idx]
            if sim_val >= anchor_threshold:
                above_threshold.append((neg_idx, sim_val))
            else:
                below_threshold.append((neg_idx, sim_val))

        # Log anchor info
        debug_logger.info(
            f"  Positive pair (idx {batch_size + anchor_idx}): {pos_sim:.4f}"
        )
        debug_logger.info(
            f"  Anchor max: {anchor_max:.4f}, Threshold: {anchor_threshold:.4f}"
        )
        debug_logger.info(
            f"  Above threshold: {len(above_threshold)}, Below threshold: {len(below_threshold)}"
        )

        # Show values above threshold (potential margin violations)
        if above_threshold:
            above_threshold.sort(key=lambda x: x[1], reverse=True)
            debug_logger.info(f"  Values ABOVE threshold:")
            for neg_idx, sim_val in above_threshold[:5]:  # Show top 5
                gap = sim_val - anchor_threshold
                debug_logger.info(f"    idx {neg_idx}: {sim_val:.4f} (gap: +{gap:.4f})")

        # Show some values below threshold
        if below_threshold and len(below_threshold) > 0:
            below_threshold.sort(key=lambda x: x[1], reverse=True)
            debug_logger.info(f"  Top values BELOW threshold:")
            for neg_idx, sim_val in below_threshold[:3]:  # Show top 3
                gap = anchor_threshold - sim_val
                debug_logger.info(f"    idx {neg_idx}: {sim_val:.4f} (gap: -{gap:.4f})")

        debug_logger.info("")

    # Overall statistics
    all_neg_sims = cos_sim_anchors[~mask_diag].flatten()
    pos_sims = np.array([cos_sim_np[i, batch_size + i] for i in range(batch_size)])

    debug_logger.info(f"--- Overall Statistics ---")
    debug_logger.info(
        f"Positive pairs: min={pos_sims.min():.4f}, max={pos_sims.max():.4f}, "
        f"mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}"
    )
    debug_logger.info(
        f"Negative pairs: min={all_neg_sims.min():.4f}, max={all_neg_sims.max():.4f}, "
        f"mean={all_neg_sims.mean():.4f}, std={all_neg_sims.std():.4f}"
    )
    debug_logger.info(
        f"Gap (pos_mean - neg_mean): {pos_sims.mean() - all_neg_sims.mean():.4f}"
    )

    if max_global:
        violations = (all_neg_sims >= threshold).sum()
        debug_logger.info(
            f"Margin violations: {violations} / {len(all_neg_sims)} "
            f"({100 * violations / len(all_neg_sims):.2f}%)"
        )

    debug_logger.info(f"\n{'='*80}\n")

    # Log the complete similarity matrix in text format
    _log_similarity_matrix(
        cos_sim_np=cos_sim_np,
        debug_logger=debug_logger,
        context=context,
        ant_margin=ant_margin,
        max_global=max_global,
        local_maxs=local_maxs if not max_global else None,
        threshold=threshold if max_global else None,
    )

def _log_similarity_matrix(
    cos_sim_np,
    debug_logger,
    context,
    ant_margin,
    max_global,
    local_maxs=None,
    threshold=None,
):
    """
    Log the complete similarity matrix in a formatted text representation.

    Args:
        cos_sim_np: Similarity matrix as numpy array
        debug_logger: Logger instance
        context: Context string for identification
        ant_margin: Margin value
        max_global: Whether using global or local max
        local_maxs: Per-anchor maximum values (for local strategy)
        threshold: Global threshold (for global strategy)
    """
    N = cos_sim_np.shape[0]
    batch_size = N // 2

    debug_logger.info(f"\n{'='*80}")
    debug_logger.info(f"COMPLETE SIMILARITY MATRIX - {context}")
    debug_logger.info(f"{'='*80}\n")

    # Create header row
    header = "     |"
    for j in range(N):
        header += f"  {j:2d}  |"
    debug_logger.info(header)
    debug_logger.info("-" * len(header))

    # Log each row with highlighting
    for i in range(N):
        row_str = f" {i:2d}  |"

        # Determine if this is an anchor row (first half)
        is_anchor = i < batch_size

        # Get threshold for this anchor
        if is_anchor:
            if max_global:
                anchor_threshold = threshold
            else:
                anchor_threshold = local_maxs[i] - ant_margin

        for j in range(N):
            val = cos_sim_np[i, j]

            # Determine highlighting
            marker = ""
            if i == j:
                # Self-similarity (diagonal)
                marker = "*"
            elif is_anchor and j == i + batch_size:
                # Positive pair for anchor i
                marker = "+"
            elif is_anchor and j < batch_size and j != i:
                # Negative within batch - check if above threshold
                if val >= anchor_threshold:
                    marker = "!"  # Violation

            row_str += f" {val:5.2f}{marker}|"

        debug_logger.info(row_str)

    # Add legend
    debug_logger.info("\nLegend:")
    debug_logger.info("  * = Self-similarity (diagonal, always 1.0)")
    debug_logger.info("  + = Positive pair (augmentation)")
    debug_logger.info("  ! = Margin violation (negative above threshold)")
    debug_logger.info(f"\n{'='*80}\n")