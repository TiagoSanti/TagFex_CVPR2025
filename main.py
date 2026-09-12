import os, datetime
import sys
from pathlib import Path
import torch
import utils.funcs as utlf
from utils.configuration import load_configs, load_yaml, training_config
from utils.provenance import safe_finalize_experiment, safe_start_experiment

from modules.data.manager import ContinualDataManager
from modules.learner.base import ContinualLearner
from methods import method_dispatch

class ContinualLauncher:
    def __init__(self, args) -> None:
        self.args = args

        self._init_torch_dist()

        if self.distributed is not None:
            self.device = torch.device(f'{args.device}:{self.distributed["rank"]}')
            utlf.set_seed(args.seed + self.distributed['rank'])
        else:
            self.device = torch.device(args.device)
            utlf.set_seed(args.seed)

    def _get_train_configs(self):
        self.configs = training_config(self.args)

    def train(self):
        self._get_train_configs()
        tracker = None
        is_primary = self.distributed is None or self.distributed["rank"] == 0
        writes_logs = not self.configs.get("terminal_only", False) and not self.configs.get("disable_log_file", False)

        try:
            data_manager = ContinualDataManager(self.configs, self.distributed)

            learner: ContinualLearner = method_dispatch(self.configs['method'], data_manager, self.configs, self.device, self.distributed)

            # Some learners resolve a parameterised/seed-specific log_dir in
            # their constructor.  Capture provenance only after that point so
            # the manifest is colocated with the actual experiment artifacts.
            if is_primary and writes_logs and self.configs.get("log_dir") is not None:
                tracker = safe_start_experiment(
                    repo=Path(__file__).resolve().parent,
                    output_dir=Path(self.configs["log_dir"]),
                    effective_configuration=self.configs,
                    config_paths=self.args.exp_configs,
                    command=[sys.executable, *sys.argv],
                )

            learner.train()
        except BaseException as error:
            safe_finalize_experiment(tracker, "failed", error)
            raise
        else:
            safe_finalize_experiment(tracker, "completed")

    def _get_evaluate_configs(self):
        exp_configs = load_configs(self.args.exp_configs)

        self.configs = vars(self.args)
        self.configs.update(**exp_configs)

    def evaluate(self):
        pass

    def reconstruct(self):
        pass

    def _init_torch_dist(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.distributed = dict()
            self.distributed['rank'] = int(os.environ['RANK'])
            self.distributed['world_size'] = int(os.environ['WORLD_SIZE'])
            self.distributed['local_rank'] = int(os.environ['LOCAL_RANK'])
        else:
            self.distributed = None
            return

        import multiprocessing
        multiprocessing.set_start_method('spawn')

        torch.cuda.set_device(self.distributed['local_rank'])
        torch.distributed.init_process_group(
            init_method='env://',
            backend=self.args.dist_backend, 
            world_size=self.distributed['world_size'], 
            rank=self.distributed['rank'],
            # timeout=datetime.timedelta(seconds=5)
        )
        torch.distributed.barrier()

if __name__ == '__main__':
    from utils.argument import get_args
    args = get_args()
    launcher = ContinualLauncher(args)

    if args.command == 'train':
        launcher.train()
    elif args.command == 'evaluate':
        launcher.evaluate()
    else:
        raise ValueError(f'Unknown command {args.command}')
