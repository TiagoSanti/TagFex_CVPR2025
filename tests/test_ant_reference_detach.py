import torch

from methods.tagfex.tagfex import _compute_contrastive_loss_base


def _scores():
    # N=6 gives an intra-view ANT block with three anchors.  Row 0 has two
    # active negatives and a unique reference at column 1.
    values = torch.tensor(
        [
            [1.00, 0.90, 0.85, 0.20, 0.10, 0.00],
            [0.40, 1.00, 0.20, 0.00, 0.20, 0.10],
            [0.10, 0.30, 1.00, 0.10, 0.00, 0.20],
            [0.20, 0.00, 0.10, 1.00, 0.40, 0.30],
            [0.10, 0.20, 0.00, 0.40, 1.00, 0.20],
            [0.00, 0.10, 0.20, 0.30, 0.20, 1.00],
        ],
        dtype=torch.float64,
    )
    return values.requires_grad_()


def _loss_and_grad(detach, *, ant_beta=1.0, nce_alpha=0.0):
    scores = _scores()
    loss = _compute_contrastive_loss_base(
        scores.clone(),
        t=0.2,
        nce_alpha=nce_alpha,
        ant_beta=ant_beta,
        ant_margin=0.5,
        ant_max_global=False,
        ant_symmetric_full=False,
        ant_formulation="logsumexp",
        ant_detach_reference=detach,
    )
    grad = torch.autograd.grad(loss, scores)[0]
    return loss.detach(), grad


def test_detach_changes_only_backward_value_on_same_scores():
    connected_loss, _ = _loss_and_grad(False)
    detached_loss, _ = _loss_and_grad(True)
    torch.testing.assert_close(connected_loss, detached_loss, rtol=0, atol=0)


def test_detach_reverses_reference_gradient_and_preserves_nonreference_gradient():
    _, connected_grad = _loss_and_grad(False)
    _, detached_grad = _loss_and_grad(True)

    # Row 0 reference is s[0,1]; s[0,2] is another active negative.
    assert connected_grad[0, 1] < 0
    assert detached_grad[0, 1] > 0
    torch.testing.assert_close(
        connected_grad[0, 2], detached_grad[0, 2], rtol=1e-12, atol=1e-12
    )


def test_detach_is_inert_when_ant_is_disabled():
    connected_loss, connected_grad = _loss_and_grad(
        False, ant_beta=0.0, nce_alpha=1.0
    )
    detached_loss, detached_grad = _loss_and_grad(
        True, ant_beta=0.0, nce_alpha=1.0
    )
    torch.testing.assert_close(connected_loss, detached_loss, rtol=0, atol=0)
    torch.testing.assert_close(connected_grad, detached_grad, rtol=0, atol=0)
