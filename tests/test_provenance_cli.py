import tempfile
import unittest
from pathlib import Path

from provenance_cli import create_bundle, verify_manifest


class ProvenanceCliTests(unittest.TestCase):
    def test_bundle_connects_and_verifies_inputs_parents_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            source = repo / "source.tex"
            parent = repo / "parent.json"
            artifact = repo / "document.pdf"
            manifest = repo / "provenance.json"
            source.write_text("source\n", encoding="utf-8")
            parent.write_text("{}\n", encoding="utf-8")
            artifact.write_bytes(b"pdf")

            created = create_bundle(
                bundle_kind="test-document",
                inputs=[source],
                artifacts=[artifact],
                parents=[parent],
                output=manifest,
                repo=repo,
            )

            self.assertEqual(created["kind"], "lineage_bundle")
            self.assertEqual(verify_manifest(manifest), [])

            source.write_text("changed\n", encoding="utf-8")
            self.assertTrue(verify_manifest(manifest))


if __name__ == "__main__":
    unittest.main()
