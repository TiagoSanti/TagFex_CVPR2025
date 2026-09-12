import json
import tempfile
import unittest
from pathlib import Path

from utils.provenance import (
    artifact_record,
    canonical_hash,
    dataset_record,
    records_hash,
)


class ProvenanceTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_insertion_order(self):
        left = {"seed": 1993, "config": {"beta": 0.5, "alpha": 1.0}}
        right = {"config": {"alpha": 1.0, "beta": 0.5}, "seed": 1993}
        self.assertEqual(canonical_hash(left), canonical_hash(right))

    def test_full_dataset_hash_rechecks_same_size_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            dataset = workspace / "dataset"
            dataset.mkdir()
            (dataset / "a.bin").write_bytes(b"abc")
            (dataset / "labels.txt").write_text("a 0\n", encoding="utf-8")

            first = dataset_record(dataset, cache_dir=workspace / "cache", mode="full")
            second = dataset_record(dataset, cache_dir=workspace / "cache", mode="full")

            self.assertEqual(first["dataset_hash"], second["dataset_hash"])
            self.assertFalse(first["cache_reused"])
            self.assertFalse(second["cache_reused"])

            (dataset / "a.bin").write_bytes(b"abd")
            changed = dataset_record(dataset, cache_dir=workspace / "cache", mode="full")
            self.assertNotEqual(first["dataset_hash"], changed["dataset_hash"])

    def test_artifact_hash_excludes_manifests_to_avoid_self_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "result.txt").write_text("result\n", encoding="utf-8")
            (output / "provenance.json").write_text("{}\n", encoding="utf-8")
            provenance_dir = output / "provenance"
            provenance_dir.mkdir()
            (provenance_dir / "run.json").write_text("{}\n", encoding="utf-8")

            artifacts = artifact_record(output)

            self.assertEqual(artifacts["file_count"], 1)
            self.assertEqual(artifacts["files"][0]["path"], "result.txt")
            self.assertEqual(
                artifacts["artifacts_hash"], records_hash(artifacts["files"])
            )


if __name__ == "__main__":
    unittest.main()
