import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from methods.tagfex.tagfex import _compute_contrastive_loss_base
from studies.ant_mechanism.observer import ANTStudyObserver
from studies.ant_mechanism.schema import validate_metric_record


class ANTMechanismObserverTests(unittest.TestCase):
    def test_observer_writes_complete_sampled_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs = {
                "seed": 1993,
                "log_dir": "observer_test",
                "ffcv": False,
                "ant_study": {
                    "enabled": True,
                    "output_root": tmp,
                    "scalar_every_n_batches": 1,
                    "snapshot_tasks": [1],
                    "snapshot_epochs": [1],
                    "init_snapshot_epochs": [1],
                    "snapshot_batches": [1],
                    "shadow_variants": True,
                    "save_snapshots": True,
                    "parameter_gradient_probes": True,
                    "parameter_update_probes": True,
                    "min_free_gb": 0,
                    "max_output_gb": 1,
                },
            }
            observer = ANTStudyObserver(configs)
            observer.set_batch_metadata(
                {
                    "absolute_index": [10, 11, 12, 13, 10, 11, 12, 13],
                    "continual_target": [0, 1, 2, 3, 0, 1, 2, 3],
                    "view": [0, 0, 0, 0, 1, 1, 1, 1],
                    "is_replay": [False] * 8,
                }
            )
            observer.set_batch_payload(
                "current",
                task=1,
                epoch=1,
                batch=1,
                tensors={
                    "view1": torch.randn(4, 3, 8, 8),
                    "view2": torch.randn(4, 3, 8, 8),
                },
            )

            model = torch.nn.Linear(5, 5, bias=False, dtype=torch.float64)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            inputs = torch.randn(8, 5, dtype=torch.float64)
            features = torch.nn.functional.normalize(model(inputs), dim=-1)
            scores = features @ features.T
            loss = _compute_contrastive_loss_base(
                scores,
                t=0.2,
                nce_alpha=1.0,
                ant_beta=0.5,
                ant_margin=0.5,
                ant_max_global=False,
                ant_symmetric_full=True,
                ant_detach_reference=False,
                ant_formulation="logsumexp",
                task=1,
                epoch=1,
                batch=1,
                study_observer=observer,
            )
            observer.record_component_parameter_gradients(
                model,
                task=1,
                epoch=1,
                batch=1,
                branch_outer_weights={"current": 1.0, "kd": 0.0},
                ant_beta=0.5,
                nce_alpha=1.0,
            )
            observer.capture_parameter_state(model, task=1, epoch=1, batch=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            observer.record_parameter_update(model, task=1, epoch=1, batch=1)
            observer.close()

            run_dir = Path(tmp) / "observer_test_s1993"
            self.assertTrue((run_dir / "manifest.json").is_file())
            snapshot_path = run_dir / "snapshots" / "T01_E001_B0001_current.npz"
            self.assertTrue(snapshot_path.is_file())
            with np.load(snapshot_path) as snapshot:
                self.assertIn("payload__view1", snapshot.files)
                self.assertIn("meta__absolute_index", snapshot.files)
            with gzip.open(run_dir / "geometry_metrics.jsonl.gz", "rt", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(len(records), 8)
            for record in records:
                validate_metric_record(record)
            self.assertTrue((run_dir / "parameter_gradients.jsonl.gz").is_file())
            self.assertTrue((run_dir / "parameter_gradient_alignment.jsonl.gz").is_file())
            self.assertTrue((run_dir / "parameter_updates.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
