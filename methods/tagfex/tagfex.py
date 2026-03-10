import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from modules import HerdingIndicesLearner
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

    def _init_ddp(self):
        self.configs["trainloader_params"]["batch_size"] //= self.distributed[
            "world_size"
        ]
        torch.distributed.barrier()

    def _should_use_debug_batch(self):
        """Check if we should use reduced batch size for similarity debug."""
        return self.configs.get("debug_similarity", False)

    def _get_debug_batch_size(self):
        """Get batch size for similarity debugging (creates 16x16 matrices)."""
        return self.configs.get("debug_similarity_batch_size", 16)

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

    @loguru.logger.catch
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
            memory_indices = self.get_memory()
            new_indices = np.concatenate((task_train.indices, memory_indices))
            task_train.indices = new_indices

            # Adjust batch size for similarity debugging if enabled
            trainloader_params = self.configs["trainloader_params"].copy()
            if self._should_use_debug_batch():
                debug_batch_size = self._get_debug_batch_size()
                self.print_logger.info(
                    f"Debug mode: Using reduced batch size {debug_batch_size} for similarity analysis"
                )
                trainloader_params["batch_size"] = debug_batch_size

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
                train_loader, test_loader = get_loaders(
                    task_train,
                    task_test,
                    trainloader_params,
                    self.configs["testloader_params"],
                    self.distributed,
                )
            # <<< @train_task_start

            self.train_task(train_loader, test_loader)

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
        if rank == 0:
            prog_bar.close()

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

            if self.configs.get("ffcv"):
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
            
            # Get debug parameters if similarity debugging is enabled
            debug_logger = getattr(self, 'similarity_debug_logger', None) if self.configs.get("debug_similarity", False) else None
            heatmap_dir = getattr(self, 'heatmap_dir', None) if self.configs.get("debug_similarity", False) else None
            
            infonce_loss = infoNCE_loss(
                embedding,
                self.configs["infonce_temp"],
                self.configs.get("nce_alpha", 1.0),
                self.configs.get("ant_beta", 0.0),
                self.configs.get("ant_margin", 0.1),
                self.configs.get("ant_max_global", True),
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
                    self.configs.get("ant_beta", 0.0),
                    self.configs.get("ant_margin", 0.1),
                    self.configs.get("ant_max_global", True),
                    logger=self.loguru_logger,
                    task=self.state["cur_task"],
                    epoch=self.state["cur_epoch"],
                    batch=batch + 1,
                    debug_logger=debug_logger,
                    heatmap_dir=heatmap_dir,
                )

                trans_logits = out["trans_logits"]
                cur_task_mask = targets >= learned_num_classes
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
            optimizer.step()
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
        
        self.similarity_debug_logger = logger
        self.print_logger.info(f"Similarity debug logging to: {debug_log_path}")
        
        # Create directory for heatmaps
        self.heatmap_dir = log_dir / "similarity_heatmaps"
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        self.print_logger.info(f"Similarity heatmaps will be saved to: {self.heatmap_dir}")

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

        if ant_max_global:
            suffix_parts.append("antGlobal")
        else:
            suffix_parts.append("antLocal")

        # Append seed when non-default so multi-seed runs get distinct directories
        seed = self.configs.get("seed", 1993)
        if seed != 1993:
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
    max_global=True,
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
        max_global: If True, use global maximum across all anchors; if False, use per-anchor maximum
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

    # Debug similarity matrices if enabled
    if debug_logger is not None and heatmap_dir is not None:
        _debug_similarity_matrices(
            cos_sim=cos_sim,
            ant_margin=ant_margin,
            max_global=max_global,
            debug_logger=debug_logger,
            heatmap_dir=heatmap_dir,
            task=task,
            epoch=epoch,
            batch=batch,
            log_prefix=log_prefix,
        )

    # ANT (Adaptive Negative Thresholding) logic
    # Split into first half (anchors) for computing negative statistics
    pos_start = cos_sim.shape[0] // 2
    cos_sim_q1 = cos_sim[:pos_start, :pos_start]

    mask_q1 = torch.eye(cos_sim_q1.shape[0], dtype=bool, device=device)
    q1 = cos_sim_q1.masked_fill(mask_q1, 0.0)  # ignore self-similarity

    # Compute positive similarities for gap calculation and logging
    pos_sims = torch.diagonal(cos_sim[:pos_start, pos_start:])

    # Compute ANT loss based on max_global strategy
    if max_global:
        # Global maximum across all anchors
        q1_max = q1.max()
        mq1 = F.relu_(q1 - q1_max + ant_margin)
    else:
        # Maximum per anchor (per row)
        q1_max = q1.max(dim=-1, keepdim=True).values
        mq1 = F.relu_(q1 - q1_max + ant_margin)

    # Compute base ANT loss
    ant_loss = torch.logsumexp(mq1, dim=-1).mean()

    # Get non-zero q1 values (excluding masked diagonal) for statistics
    q1_nonzero = q1[~mask_q1]
    neg_sims = q1_nonzero

    # Always log basic contrastive statistics for monitoring
    if logger is not None:
        # Calculate current gap (pos_mean - neg_mean)
        pos_mean = pos_sims.mean()
        neg_mean = neg_sims.mean()
        current_gap_computed = pos_mean - neg_mean

        # Prepare basic statistics
        basic_stats = {
            "pos_mean": pos_mean.item(),
            "pos_std": pos_sims.std().item(),
            "neg_mean": neg_mean.item(),
            "neg_std": neg_sims.std().item(),
            "current_gap": current_gap_computed.item(),
        }

        logger.log_contrastive_stats(basic_stats, task=task, epoch=epoch, batch=batch)

    # Log detailed ANT-specific statistics when ANT is active
    if logger is not None and ant_beta > 0:
        # Compute gaps: distance from max negative to margin threshold
        if max_global:
            gaps = q1_nonzero - q1_max
        else:
            # Expand q1_max to match q1 shape
            q1_max_expanded = q1_max.expand_as(q1)
            gaps = (q1 - q1_max_expanded)[~mask_q1]

        # Count margin violations (mq1 > 0 means margin was violated)
        mq1_nonzero = mq1[~mask_q1]
        violations = (mq1_nonzero > 0).float()
        violation_pct = violations.mean().item() * 100

        # Prepare detailed ANT statistics (only logged when ANT is active)
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

        # Log detailed ANT statistics
        logger.log_ant_distance_stats(ant_stats, task=task, epoch=epoch, batch=batch)

    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    # InfoNCE loss with optional local anchor normalization
    cos_sim = cos_sim / t

    # Apply local anchor normalization when ANT is disabled but max_global=False
    # Each anchor is normalized by its own maximum negative similarity
    if ant_beta == 0.0 and not max_global:
        # Find max negative similarity per anchor (excluding positives)
        cos_sim_neg = cos_sim.clone()
        cos_sim_neg[pos_mask] = -float("inf")

        # Normalize by per-anchor maximum
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
    max_global=True,
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
        max_global: If True, use global max; if False, use per-anchor max
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
        max_global=max_global,
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
    max_global=True,
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
        max_global: If True, use global max; if False, use per-anchor max
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
        max_global=max_global,
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
    debug_logger.info(f"ANT margin: {ant_margin}, Max strategy: {'Global' if max_global else 'Local'}")
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
        debug_logger.info(f"Local max per anchor: min={local_maxs.min():.4f}, "
                         f"max={local_maxs.max():.4f}, mean={local_maxs.mean():.4f}\n")
    
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
        debug_logger.info(f"  Positive pair (idx {batch_size + anchor_idx}): {pos_sim:.4f}")
        debug_logger.info(f"  Anchor max: {anchor_max:.4f}, Threshold: {anchor_threshold:.4f}")
        debug_logger.info(f"  Above threshold: {len(above_threshold)}, Below threshold: {len(below_threshold)}")
        
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
    debug_logger.info(f"Positive pairs: min={pos_sims.min():.4f}, max={pos_sims.max():.4f}, "
                     f"mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}")
    debug_logger.info(f"Negative pairs: min={all_neg_sims.min():.4f}, max={all_neg_sims.max():.4f}, "
                     f"mean={all_neg_sims.mean():.4f}, std={all_neg_sims.std():.4f}")
    debug_logger.info(f"Gap (pos_mean - neg_mean): {pos_sims.mean() - all_neg_sims.mean():.4f}")
    
    if max_global:
        violations = (all_neg_sims >= threshold).sum()
        debug_logger.info(f"Margin violations: {violations} / {len(all_neg_sims)} "
                         f"({100 * violations / len(all_neg_sims):.2f}%)")
    
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
    
    # Create and save heatmap
    # _save_similarity_heatmap(
    #     cos_sim_np=cos_sim_np,
    #     heatmap_dir=heatmap_dir,
    #     context=context,
    #     ant_margin=ant_margin,
    #     max_global=max_global,
    # )

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


def _save_similarity_heatmap(
    cos_sim_np,
    heatmap_dir,
    context,
    ant_margin,
    max_global,
):
    """
    Create and save heatmap visualization of similarity matrix.
    
    Args:
        cos_sim_np: Similarity matrix as numpy array
        heatmap_dir: Directory to save heatmap
        context: Context string for filename
        ant_margin: Margin value for title
        max_global: Whether using global or local max
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        cos_sim_np,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Cosine Similarity"},
        ax=ax,
    )
    
    # Add title with parameters
    max_strategy = "Global" if max_global else "Local"
    ax.set_title(f"Similarity Matrix - {context}\nMargin: {ant_margin}, Max: {max_strategy}", 
                 fontsize=12, pad=20)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Sample Index (Anchor)", fontsize=10)
    
    # Highlight diagonal
    N = cos_sim_np.shape[0]
    batch_size = N // 2
    
    # Draw lines to separate augmented pairs
    ax.axhline(y=batch_size, color='blue', linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=batch_size, color='blue', linewidth=2, linestyle='--', alpha=0.7)
    
    # Save figure
    filename = f"sim_heatmap_{context}.png"
    filepath = Path(heatmap_dir) / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

