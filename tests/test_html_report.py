import unittest

import numpy as np
import pandas as pd

from generate_html_pdf_report import (
    _build_arg_parser,
    _build_html_document,
    _complementary_method_sort_key,
    _cross_dataset_section_html,
    _is_complementary_method,
    _is_legacy_method,
    _ranking_html,
    deduplicate_equivalent_runs,
    filter_complete,
    is_true_baseline,
    short_name,
)


class HtmlReportSelectionTests(unittest.TestCase):
    def test_short_report_is_the_cli_default_and_full_is_opt_in(self):
        parser = _build_arg_parser()

        self.assertTrue(parser.parse_args([]).short)
        self.assertTrue(parser.parse_args(["--short"]).short)
        self.assertFalse(parser.parse_args(["--full"]).short)

    def test_beta_zero_teacheravg_is_a_separate_ablation(self):
        experiment = (
            "exp_cifar100_10-10_antB0_nceA1_"
            "antGlobal_nceGlobal_avgK5_s1993"
        )

        self.assertEqual(
            short_name(experiment), "Baseline InfoNCE + TeacherAvg-5"
        )
        self.assertFalse(is_true_baseline(experiment))
        self.assertTrue(is_true_baseline(experiment.replace("_avgK5", "")))

    def test_detached_ant_is_canonical_and_connected_ant_is_legacy(self):
        stem = (
            "exp_cifar100_10-10_antB0.5_nceA1_antM0.5_"
            "antSymmetricFullGlobal_nceGlobal"
        )

        self.assertEqual(
            short_name(stem + "_refDetached_s1993"),
            "ANT-FS-GR (β=0.5)",
        )
        self.assertEqual(
            short_name(stem + "_s1993"),
            "ANT-FS-GR (β=0.5) [legado: ref. conectada]",
        )

    def test_section_zero_separates_central_and_complementary_methods(self):
        rows = []
        methods = [
            ("Baseline InfoNCE", True, 0.0),
            ("ANT-IV-AR (β=0.5)", False, 1.0),
            ("ANT-IV-AR (β=0.5) + TeacherAvg-5", False, 1.5),
            ("ANT-FS-AR (β=0.5) + SBS", False, 1.2),
            (
                "ANT-FS-AR (β=0.5) [legado: ref. conectada]",
                False,
                0.8,
            ),
        ]
        for dataset in (
            "cifar100_10-10",
            "cifar100_50-10",
            "tiny_imagenet_20-20",
        ):
            for label, is_baseline, gain in methods:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": 1993,
                        "label": label,
                        "is_baseline": is_baseline,
                        "avg_acc": 70.0 + gain,
                        "avg_nme": 60.0 + gain,
                        "fgt_acc": 10.0 - gain,
                    }
                )

        html = _cross_dataset_section_html(pd.DataFrame(rows))
        central_start = html.index("Métodos centrais da pesquisa")
        complementary_start = html.index("Métodos complementares")
        legacy_start = html.index("ANT legado — referência conectada")
        central_html = html[central_start:complementary_start]
        complementary_html = html[complementary_start:legacy_start]
        legacy_html = html[legacy_start:]

        self.assertIn("ANT-IV-AR (β=0.5)", central_html)
        self.assertNotIn(
            '<td class="experiment" rowspan="3">'
            'ANT-IV-AR (β=0.5) + TeacherAvg-5</td>',
            central_html,
        )
        self.assertNotIn(
            '<td class="experiment" rowspan="3">'
            'ANT-FS-AR (β=0.5) [legado: ref. conectada]</td>',
            central_html,
        )
        self.assertIn("TeacherAvg-5", complementary_html)
        self.assertIn("SBS", complementary_html)
        self.assertIn("legado: ref. conectada", legacy_html)
        self.assertEqual(
            html.count(
                '<td class="experiment" rowspan="3">Baseline InfoNCE</td>'
            ),
            3,
        )
        self.assertIn("Ranking estatístico — métodos centrais", central_html)
        self.assertIn(
            "Ranking estatístico — métodos complementares",
            complementary_html,
        )
        complementary_ranking = complementary_html.split(
            "Ranking estatístico — métodos complementares", 1
        )[1]
        self.assertIn("ANT-FS-AR (β=0.5) + SBS", complementary_ranking)
        self.assertIn("Ranking estatístico — ANT legado", legacy_html)

    def test_complementary_method_classification(self):
        self.assertFalse(_is_complementary_method("ANT-FS-AR (β=0.5)"))
        self.assertTrue(
            _is_complementary_method("ANT-FS-AR (β=0.5) + TeacherAvg-5")
        )
        self.assertTrue(_is_complementary_method("ANT-FS-AR (β=0.5) + SBS"))
        self.assertFalse(
            _is_complementary_method(
                "ANT-FS-AR (β=0.5) [legado: ref. conectada]"
            )
        )
        self.assertFalse(_is_legacy_method("ANT-FS-AR (β=0.5)"))
        self.assertTrue(
            _is_legacy_method(
                "ANT-FS-AR (β=0.5) [legado: ref. conectada]"
            )
        )

    def test_complementary_methods_sort_by_ablation_then_natural_name(self):
        labels = [
            "ANT-IV-AR (β=0.5) + SBS",
            "ANT-IV-AR (β=0.5) + TeacherAvg-10",
            "ANT-FS-AR (β=0.5) + TeacherAvg-5",
            "ANT-IV-AR (β=0.5) + TeacherAvg-3",
        ]

        self.assertEqual(
            sorted(labels, key=_complementary_method_sort_key),
            [
                "ANT-FS-AR (β=0.5) + TeacherAvg-5",
                "ANT-IV-AR (β=0.5) + TeacherAvg-3",
                "ANT-IV-AR (β=0.5) + TeacherAvg-10",
                "ANT-IV-AR (β=0.5) + SBS",
            ],
        )

    def test_ranking_includes_required_ant_fs_gr_on_paired_intersection(self):
        rows = []
        datasets = (
            "cifar100_10-10",
            "cifar100_50-10",
            "tiny_imagenet_20-20",
            "tiny_imagenet_100-20",
        )
        for dataset in datasets:
            for seed in (1993, 1994, 1995):
                for label, gain, baseline in (
                    ("Baseline InfoNCE", 0.0, True),
                    ("ANT-FS-AR (β=0.5)", 0.5, False),
                ):
                    rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "label": label,
                        "is_baseline": baseline,
                        "avg_acc": 70.0 + gain,
                        "avg_nme": 60.0 + gain,
                        "fgt_acc": 10.0 - gain,
                    })
            for seed in (1993, 1994):
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "label": "ANT-FS-GR (β=0.5)",
                    "is_baseline": False,
                    "avg_acc": 70.8,
                    "avg_nme": 60.8,
                    "fgt_acc": 9.2,
                })

        html = _ranking_html(
            pd.DataFrame(rows),
            method_labels={
                "Baseline InfoNCE",
                "ANT-FS-AR (β=0.5)",
                "ANT-FS-GR (β=0.5)",
            },
            required_method_labels={"ANT-FS-GR (β=0.5)"},
        )

        self.assertIn("ANT-FS-GR (β=0.5)", html)
        self.assertIn("8 observações × 3 métodos", html)
        self.assertIn(">8</td>", html)

    def test_provenance_index_has_no_links_and_uses_nowrap_layout(self):
        html = _build_html_document(
            "<section></section>",
            "",
            "test-time",
            {
                "lineage_hash": "a" * 64,
                "sidecar_name": "report.provenance.json",
                "selected_experiment_count": 1,
                "selected_without_run_manifest_count": 1,
                "mode": "short",
                "selected_experiments": [
                    {
                        "dataset": "tiny_imagenet_100-20",
                        "method": "ANT-FS-AR (β=0.5)",
                        "seed": 1993,
                        "experiment": "exp_example",
                        "gistlog": {"path": "logs/example.log", "sha256": "b" * 64},
                        "run_manifests": [],
                    }
                ],
            },
        )

        self.assertNotIn("<a ", html)
        self.assertIn(".overview-table th, .overview-table td", html)
        self.assertIn(".provenance-index th:nth-child(-n+3)", html)
        self.assertIn(
            ".overview-table { width: 100%; table-layout: fixed; }", html
        )
        self.assertIn(
            ".provenance-index table { width: 100%; table-layout: fixed; }", html
        )
        self.assertIn(
            ".cross-dataset .table-wrap, .provenance-index .table-wrap "
            "{ overflow-x: visible; }",
            html,
        )
        self.assertIn("container-type: inline-size", html)
        self.assertIn("font-size: clamp(11px, 0.95cqw, 15px)", html)
        self.assertIn("font-size: clamp(12px, 0.9cqw, 15px)", html)
        self.assertNotIn("width: max-content", html)

    def test_completeness_uses_scenario_contract_not_observed_maximum(self):
        rows = []
        for tasks in (9, 10):
            row = {
                "dataset": "cifar100_10-10",
                "exp": f"exp_{tasks}",
                "num_tasks": tasks,
            }
            for task in range(1, 11):
                row[f"acc_T{task}"] = 1.0 if task <= tasks else np.nan
            rows.append(row)

        complete = filter_complete(pd.DataFrame(rows))

        self.assertEqual(complete["exp"].tolist(), ["exp_10"])

    def test_deduplication_prefers_non_debug_name_after_infonce_priority(self):
        base = "exp_cifar100_10-10_antB0.5_nceA1_antM0.5_antLocal_nceGlobal_s1993"
        frame = pd.DataFrame(
            [
                {"dataset": "cifar100_10-10", "exp": "debug_" + base, "seed": 1993},
                {"dataset": "cifar100_10-10", "exp": base, "seed": 1993},
            ]
        )

        selected = deduplicate_equivalent_runs(frame)

        self.assertEqual(selected["exp"].tolist(), [base])

    def test_ranking_delta_uses_matching_dataset_baseline(self):
        datasets = [
            "cifar100_10-10",
            "cifar100_50-10",
            "tiny_imagenet_20-20",
        ]
        baselines = [90.0, 60.0, 50.0]
        rows = []
        for dataset, baseline in zip(datasets, baselines):
            rows.extend(
                [
                    {
                        "dataset": dataset,
                        "seed": 1993,
                        "label": "Baseline InfoNCE",
                        "is_baseline": True,
                        "avg_acc": baseline,
                        "avg_nme": baseline,
                        "fgt_acc": 10.0,
                    },
                    {
                        "dataset": dataset,
                        "seed": 1993,
                        "label": "ANT",
                        "is_baseline": False,
                        "avg_acc": baseline + 1.0,
                        "avg_nme": baseline + 1.0,
                        "fgt_acc": 9.0,
                    },
                ]
            )

        html = _ranking_html(pd.DataFrame(rows))

        self.assertIn("<td>+1.00</td>", html)

    def test_cross_dataset_baseline_has_means_for_all_metrics(self):
        frame = pd.DataFrame(
            [
                {
                    "dataset": "cifar100_10-10",
                    "seed": 1993,
                    "label": "Baseline InfoNCE",
                    "is_baseline": True,
                    "avg_acc": 80.0,
                    "avg_nme": 70.0,
                    "fgt_acc": 10.0,
                },
                {
                    "dataset": "cifar100_50-10",
                    "seed": 1993,
                    "label": "Baseline InfoNCE",
                    "is_baseline": True,
                    "avg_acc": 60.0,
                    "avg_nme": 50.0,
                    "fgt_acc": 20.0,
                },
            ]
        )

        html = _cross_dataset_section_html(frame)

        self.assertIn("<strong>70.00</strong>", html)
        self.assertIn("<strong>60.00</strong>", html)
        self.assertIn("<strong>15.00</strong>", html)


if __name__ == "__main__":
    unittest.main()
