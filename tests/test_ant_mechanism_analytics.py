import math
import unittest

import torch

from methods.tagfex.tagfex import _compute_contrastive_loss_base
from studies.ant_mechanism.analytics import analyze_contrastive_geometry
from studies.ant_mechanism.schema import ALL_ANT_VARIANTS, ANTVariant


class ANTMechanismAnalyticsTests(unittest.TestCase):
    def _scores(self) -> torch.Tensor:
        torch.manual_seed(7)
        features = torch.randn(8, 5, dtype=torch.float64)
        features = torch.nn.functional.normalize(features, dim=-1)
        return (features @ features.T).detach().requires_grad_(True)

    def _autograd(
        self,
        scores: torch.Tensor,
        variant: ANTVariant,
        *,
        nce_alpha: float,
        ant_beta: float,
    ):
        loss = _compute_contrastive_loss_base(
            scores.clone(),
            t=0.2,
            nce_alpha=nce_alpha,
            ant_beta=ant_beta,
            ant_margin=0.5,
            ant_max_global=variant.max_global,
            ant_symmetric_full=variant.symmetric_full,
            ant_detach_reference=variant.detach,
            ant_formulation="logsumexp",
        )
        return loss.detach(), torch.autograd.grad(loss, scores)[0]

    def test_analytical_combined_gradient_matches_autograd(self):
        for variant in ALL_ANT_VARIANTS:
            with self.subTest(variant=variant.name):
                scores = self._scores()
                expected_loss, expected_grad = self._autograd(
                    scores, variant, nce_alpha=1.0, ant_beta=0.5
                )
                analysis = analyze_contrastive_geometry(
                    scores,
                    temperature=0.2,
                    margin=0.5,
                    variant=variant,
                    nce_alpha=1.0,
                    ant_beta=0.5,
                )
                torch.testing.assert_close(
                    analysis.tensors["combined_grad"],
                    expected_grad,
                    rtol=1e-10,
                    atol=1e-10,
                )
                self.assertTrue(
                    math.isclose(
                        analysis.metrics["combined_loss"],
                        expected_loss.item(),
                        abs_tol=1e-10,
                    )
                )

    def test_mask_cardinality_matches_monograph(self):
        for variant in ALL_ANT_VARIANTS:
            with self.subTest(variant=variant.name):
                scores = self._scores()
                analysis = analyze_contrastive_geometry(
                    scores, temperature=0.2, margin=0.5, variant=variant
                )
                expected = (
                    scores.shape[0] - 2
                    if variant.symmetric_full
                    else scores.shape[0] // 2 - 1
                )
                self.assertEqual(analysis.metrics["valid_negative_count_mean"], expected)

    def test_detach_preserves_forward_and_changes_reference_gradient(self):
        for coverage in ("IV", "FS"):
            for reference in ("GR", "AR"):
                with self.subTest(coverage=coverage, reference=reference):
                    scores = self._scores()
                    connected = analyze_contrastive_geometry(
                        scores,
                        temperature=0.2,
                        margin=0.5,
                        variant=ANTVariant(coverage, reference, False),
                    )
                    detached = analyze_contrastive_geometry(
                        scores,
                        temperature=0.2,
                        margin=0.5,
                        variant=ANTVariant(coverage, reference, True),
                    )
                    self.assertEqual(
                        connected.metrics["ant_loss_raw"],
                        detached.metrics["ant_loss_raw"],
                    )
                    torch.testing.assert_close(
                        connected.tensors["active_mask"],
                        detached.tensors["active_mask"],
                    )
                    self.assertFalse(
                        torch.equal(
                            connected.tensors["ant_grad"],
                            detached.tensors["ant_grad"],
                        )
                    )

    def test_connected_ant_is_invariant_to_uniform_similarity_shift(self):
        for reference in ("GR", "AR"):
            with self.subTest(reference=reference):
                analysis = analyze_contrastive_geometry(
                    self._scores(),
                    temperature=0.2,
                    margin=0.5,
                    variant=ANTVariant("FS", reference, False),
                    nce_alpha=0.0,
                    ant_beta=1.0,
                )
                self.assertAlmostEqual(
                    analysis.tensors["ant_grad"].sum().item(), 0.0, places=12
                )

    def test_monograph_infonce_toy_values(self):
        positive = torch.tensor(0.9, dtype=torch.float64)
        negatives = torch.tensor(
            [0.72, 0.62, 0.58, 0.30, 0.05], dtype=torch.float64
        )
        logits = torch.cat((positive.view(1), negatives))
        denominator = logits.exp().sum()
        loss = -positive + denominator.log()
        negative_gradients = negatives.exp() / denominator

        self.assertAlmostEqual(denominator.item(), 10.56, places=2)
        self.assertAlmostEqual(loss.item(), 1.46, places=2)
        torch.testing.assert_close(
            negative_gradients,
            torch.tensor([0.195, 0.176, 0.169, 0.128, 0.100], dtype=torch.float64),
            rtol=0,
            atol=0.001,
        )


if __name__ == "__main__":
    unittest.main()
