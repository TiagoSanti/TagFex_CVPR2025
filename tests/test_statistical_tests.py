import tempfile
import unittest
from pathlib import Path

from statistical_tests import (
    CONFIG_NAME,
    _canonical_config,
    _parse_final_metrics,
    discover_candidates,
    select_candidates,
)


class StatisticalInputTests(unittest.TestCase):
    def test_detached_ant_is_named_canonical_and_connected_ant_is_legacy(self):
        detached = "antB0.5_nceA1_antM0.5_antSymmetricFullGlobal_refDetached"
        connected = "antB0.5_nceA1_antM0.5_antSymmetricFullGlobal"

        self.assertEqual(CONFIG_NAME[detached], "ANT-FS-GR")
        self.assertIn("legado: ref. conectada", CONFIG_NAME[connected])

    def test_legacy_infonce_tokens_have_one_canonical_identity(self):
        global_name = "antB0_nceA1_antGlobal_nceGlobal"
        local_name = "antB0_nceA1_antLocal_nceLocal"
        self.assertEqual(_canonical_config(global_name), _canonical_config(local_name))

    def test_teacheravg_suffix_survives_baseline_canonicalization(self):
        plain = _canonical_config("antB0_nceA1_antGlobal_nceGlobal")
        averaged = _canonical_config(
            "antB0_nceA1_antGlobal_nceGlobal_avgK5"
        )

        self.assertEqual(plain, "antB0_nceA1_baselineInfoNCE")
        self.assertEqual(averaged, "antB0_nceA1_baselineInfoNCE_avgK5")

    def test_parser_uses_final_complete_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "exp_gistlog.log"
            log.write_text(
                "avg_acc1 1.0 acc1_curve [1 2]\n"
                "avg_acc1 3.5 acc1_curve [1 2 3]\n",
                encoding="utf-8",
            )
            self.assertEqual(_parse_final_metrics(log), (3.5, 3))

    def test_selection_is_deterministic_and_prefers_nglobal(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            logs = repo / "logs"
            names = (
                "exp_cifar100_50-10_antB0_nceA1_antGlobal_nceGlobal_s1993",
                "debug_exp_cifar100_50-10_antB0_nceA1_antLocal_nceLocal_s1993",
            )
            for index, name in enumerate(names):
                directory = logs / name
                directory.mkdir(parents=True)
                curve = " ".join(str(value) for value in range(6))
                (directory / "exp_gistlog.log").write_text(
                    f"avg_acc1 {70 + index}.0 acc1_curve [{curve}]\n",
                    encoding="utf-8",
                )

            candidates = discover_candidates(logs, repo)
            selected = select_candidates(candidates)
            winner = next(iter(selected.values()))

            self.assertIn("nceGlobal", winner.raw_config)
            self.assertEqual(winner.avg_acc1, 70.0)

    def test_versioned_seed_directory_is_discovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            logs = repo / "logs"
            directory = logs / (
                "exp_tiny_imagenet_20-20_"
                "antB0.5_nceA1_antM0.5_antGlobal_nceGlobal_s1995_v2"
            )
            directory.mkdir(parents=True)
            curve = " ".join(str(value) for value in range(10))
            (directory / "exp_gistlog.log").write_text(
                f"avg_acc1 4.5 acc1_curve [{curve}]\n",
                encoding="utf-8",
            )

            candidates = discover_candidates(logs, repo)

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].seed, "s1995")
            self.assertEqual(candidates[0].task_count, 10)


if __name__ == "__main__":
    unittest.main()
