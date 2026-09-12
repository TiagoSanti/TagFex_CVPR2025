import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main


class MainProvenanceIntegrationTests(unittest.TestCase):
    def test_manifest_uses_log_directory_resolved_by_learner(self):
        launcher = main.ContinualLauncher.__new__(main.ContinualLauncher)
        launcher.args = SimpleNamespace(exp_configs=[])
        launcher.distributed = None
        launcher.device = "cpu"

        def load_configuration():
            launcher.configs = {
                "method": "tagfex",
                "terminal_only": False,
                "disable_log_file": False,
                "log_dir": Path("logs/base"),
            }

        learner = Mock()

        def construct_learner(_method, _data, configs, _device, _distributed):
            configs["log_dir"] = Path("logs/resolved_s1993")
            return learner

        tracker = Mock()
        with (
            patch.object(launcher, "_get_train_configs", side_effect=load_configuration),
            patch.object(main, "ContinualDataManager", return_value=Mock()),
            patch.object(main, "method_dispatch", side_effect=construct_learner),
            patch.object(main, "safe_start_experiment", return_value=tracker) as start,
            patch.object(main, "safe_finalize_experiment") as finalize,
        ):
            launcher.train()

        self.assertEqual(start.call_args.kwargs["output_dir"], Path("logs/resolved_s1993"))
        learner.train.assert_called_once_with()
        finalize.assert_called_once_with(tracker, "completed")


if __name__ == "__main__":
    unittest.main()
