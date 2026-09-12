"""Pure, side-effect-free analytical diagnostics for InfoNCE and ANT.

The functions here intentionally operate on an already-computed similarity
matrix.  They never participate in the optimizer graph and can therefore be
used for both the trained condition and counterfactual ANT variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .schema import ANTVariant


@dataclass
class GeometryAnalysis:
    metrics: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return values.std(unbiased=False)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    denom = a_flat.norm() * b_flat.norm()
    if denom <= 0:
        return 0.0
    return float(torch.dot(a_flat, b_flat).div(denom).item())


def _info_nce(cos_sim: torch.Tensor, temperature: float) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    if temperature <= 0:
        raise ValueError("InfoNCE temperature must be positive")
    n = cos_sim.shape[0]
    if cos_sim.ndim != 2 or cos_sim.shape[1] != n or n % 2:
        raise ValueError("cos_sim must be an even, square [2B, 2B] matrix")

    rows = torch.arange(n, device=cos_sim.device)
    pos_idx = (rows + n // 2) % n
    self_mask = torch.eye(n, dtype=torch.bool, device=cos_sim.device)
    pos_mask = torch.zeros_like(self_mask)
    pos_mask[rows, pos_idx] = True
    neg_mask = ~(self_mask | pos_mask)

    logits = (cos_sim / temperature).masked_fill(self_mask, -torch.inf)
    log_partition = torch.logsumexp(logits, dim=-1)
    positive_logits = logits[rows, pos_idx]
    nll = -positive_logits + log_partition
    probabilities = torch.softmax(logits, dim=-1)

    grad = probabilities / (n * temperature)
    grad[rows, pos_idx] -= 1.0 / (n * temperature)
    grad = grad.masked_fill(self_mask, 0.0)

    positive_prob = probabilities[rows, pos_idx]
    negative_probabilities = probabilities.masked_fill(~neg_mask, 0.0)
    negative_mass = negative_probabilities.sum(dim=-1)
    conditional_neg = negative_probabilities / negative_mass.unsqueeze(1).clamp_min(1e-30)
    entropy = -(conditional_neg * conditional_neg.clamp_min(1e-30).log()).sum(dim=-1)
    effective_negatives = entropy.exp()

    negative_values = cos_sim.masked_fill(~neg_mask, -torch.inf)
    positive_values = cos_sim[rows, pos_idx]
    positive_rank = 1 + (negative_values > positive_values.unsqueeze(1)).sum(dim=-1)
    top_k = min(5, n - 2)
    top_mass = torch.topk(negative_probabilities, k=top_k, dim=-1).values.sum(dim=-1)

    metrics = {
        "nce_loss": float(nll.mean().item()),
        "nce_alignment_term": float((-positive_logits).mean().item()),
        "nce_log_partition": float(log_partition.mean().item()),
        "positive_similarity_mean": float(positive_values.mean().item()),
        "positive_similarity_std": float(_safe_std(positive_values).item()),
        "negative_similarity_mean": float(cos_sim[neg_mask].mean().item()),
        "negative_similarity_std": float(_safe_std(cos_sim[neg_mask]).item()),
        "positive_probability_mean": float(positive_prob.mean().item()),
        "positive_rank_mean": float(positive_rank.float().mean().item()),
        "negative_above_positive_ratio": float((positive_rank > 1).float().mean().item()),
        "negative_probability_mass_mean": float(negative_mass.mean().item()),
        "negative_top5_mass_mean": float(top_mass.mean().item()),
        "negative_weight_entropy_mean": float(entropy.mean().item()),
        "effective_negative_count_mean": float(effective_negatives.mean().item()),
        "nce_grad_norm": float(grad.norm().item()),
    }
    return metrics, {
        "nce_grad": grad,
        "nce_probabilities": probabilities,
        "nce_positive_mask": pos_mask,
        "nce_negative_mask": neg_mask,
        "nce_loss_per_anchor": nll,
    }


def _ant(
    cos_sim: torch.Tensor,
    margin: float,
    variant: ANTVariant,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    n = cos_sim.shape[0]
    b = n // 2
    if variant.symmetric_full:
        ant_values = cos_sim
        self_mask = torch.eye(n, dtype=torch.bool, device=cos_sim.device)
        pos_mask = self_mask.roll(shifts=b, dims=0)
        valid = ~(self_mask | pos_mask)
    else:
        ant_values = cos_sim[:b, :b]
        valid = ~torch.eye(b, dtype=torch.bool, device=cos_sim.device)

    masked_values = ant_values.masked_fill(~valid, -torch.inf)
    anchors = masked_values.shape[0]
    num_negatives = valid.sum(dim=-1).to(cos_sim.dtype)

    if variant.max_global:
        reference = masked_values.max()
        reference_for_rows = reference.expand(anchors, 1)
        global_ref_mask = valid & (masked_values == reference)
        reference_mask = global_ref_mask
        reference_count = int(global_ref_mask.sum().item())
    else:
        reference_for_rows, reference_idx = masked_values.max(dim=-1, keepdim=True)
        reference = reference_for_rows
        reference_mask = torch.zeros_like(valid)
        reference_mask.scatter_(1, reference_idx, True)
        reference_count = anchors

    raw_v = masked_values - reference_for_rows + margin
    active = valid & (raw_v > 0)
    relu_v = torch.relu(raw_v).masked_fill(~valid, -torch.inf)
    ant_loss_per_anchor = torch.logsumexp(relu_v, dim=-1)
    ant_loss = ant_loss_per_anchor.mean()
    floor = torch.log(num_negatives.clamp_min(1))

    weights = torch.softmax(relu_v, dim=-1)
    direct = weights * active.to(weights.dtype) / anchors
    ant_grad_subset = direct.clone()
    if not variant.detach:
        if variant.max_global:
            reference_mass = direct.sum()
            ant_grad_subset[reference_mask] -= reference_mass / max(reference_count, 1)
        else:
            row_mass = direct.sum(dim=-1, keepdim=True)
            ant_grad_subset -= reference_mask.to(direct.dtype) * row_mass

    ant_grad = torch.zeros_like(cos_sim)
    if variant.symmetric_full:
        ant_grad.copy_(ant_grad_subset)
    else:
        ant_grad[:b, :b] = ant_grad_subset

    valid_raw = raw_v[valid]
    active_raw = raw_v[active]
    active_count = active.sum(dim=-1).to(cos_sim.dtype)
    grad_active_anchor = ant_grad_subset.norm(dim=-1) > 1e-12
    ref_values = (
        reference.expand(1)
        if variant.max_global
        else reference_for_rows.flatten()
    )

    ref_grad_values = ant_grad_subset[reference_mask]
    rows = torch.arange(anchors, device=cos_sim.device)
    if variant.symmetric_full:
        ant_positive = cos_sim[rows, (rows + b) % n]
    else:
        ant_positive = cos_sim[rows, rows + b]
    hardest_negative = masked_values.max(dim=-1).values
    active_nonreference = active & ~reference_mask
    if active_nonreference.any():
        expanded_reference = reference_for_rows.expand_as(masked_values)
        ref_nonref_gap = (
            expanded_reference[active_nonreference]
            - masked_values[active_nonreference]
        ).mean()
    else:
        ref_nonref_gap = torch.zeros((), dtype=cos_sim.dtype, device=cos_sim.device)
    if int(num_negatives.min().item()) >= 2:
        top2 = torch.topk(masked_values, k=2, dim=-1).values
        top1_top2_gap = (top2[:, 0] - top2[:, 1]).mean()
    else:
        top1_top2_gap = torch.zeros((), dtype=cos_sim.dtype, device=cos_sim.device)
    nonzero_ant = ant_grad.abs() > 1e-12
    metrics = {
        "ant_loss_raw": float(ant_loss.item()),
        "ant_count_floor": float(floor.mean().item()),
        "ant_loss_adjusted": float((ant_loss_per_anchor - floor).mean().item()),
        "valid_negative_count_mean": float(num_negatives.mean().item()),
        "active_count_mean": float(active_count.mean().item()),
        "active_ratio": float((active_count / num_negatives.clamp_min(1)).mean().item()),
        "effective_anchor_ratio": float(grad_active_anchor.float().mean().item()),
        "violation_mean_all": float(torch.relu(valid_raw).mean().item()),
        "violation_mean_active": float(active_raw.mean().item()) if active_raw.numel() else 0.0,
        "violation_p90": float(torch.quantile(valid_raw, 0.90).item()),
        "violation_p95": float(torch.quantile(valid_raw, 0.95).item()),
        "reference_similarity_mean": float(ref_values.mean().item()),
        "reference_similarity_std": float(_safe_std(ref_values).item()),
        "threshold_mean": float((reference_for_rows - margin).mean().item()),
        "threshold_std": float(_safe_std(reference_for_rows.flatten() - margin).item()),
        "reference_count": float(reference_count),
        "hardest_negative_similarity_mean": float(hardest_negative.mean().item()),
        "positive_reference_gap_mean": float((ant_positive - reference_for_rows.flatten()).mean().item()),
        "reference_active_nonreference_gap_mean": float(ref_nonref_gap.item()),
        "top1_top2_negative_gap_mean": float(top1_top2_gap.item()),
        "reference_grad_mean": float(ref_grad_values.mean().item()) if ref_grad_values.numel() else 0.0,
        "reference_repel_ratio": float((ref_grad_values > 0).float().mean().item()) if ref_grad_values.numel() else 0.0,
        "ant_grad_nonzero_ratio": float(nonzero_ant.float().mean().item()),
        "ant_grad_norm": float(ant_grad.norm().item()),
    }
    tensors = {
        "valid_mask": valid,
        "active_mask": active,
        "reference_mask": reference_mask,
        "raw_violation": raw_v,
        "ant_weights": weights,
        "ant_loss_per_anchor": ant_loss_per_anchor,
        "ant_grad": ant_grad,
    }
    return metrics, tensors


@torch.no_grad()
def analyze_contrastive_geometry(
    cos_sim: torch.Tensor,
    *,
    temperature: float,
    margin: float,
    variant: ANTVariant,
    nce_alpha: float = 1.0,
    ant_beta: float = 0.5,
) -> GeometryAnalysis:
    """Return scalar diagnostics and analytical dL/dS tensors.

    Gradients include the mean reductions used by the training implementation.
    A positive dL/ds means gradient descent tends to reduce that similarity.
    """

    scores = cos_sim.detach().to(dtype=torch.float64)
    nce_metrics, nce_tensors = _info_nce(scores, temperature)
    nce_grad = nce_tensors["nce_grad"]
    ant_metrics, ant_tensors = _ant(scores, margin, variant)
    ant_grad = ant_tensors["ant_grad"]
    combined_grad = nce_alpha * nce_grad + ant_beta * ant_grad

    overlap = (nce_grad.abs() > 1e-12) & (ant_grad.abs() > 1e-12)
    if overlap.any():
        sign_agreement = (torch.sign(nce_grad[overlap]) == torch.sign(ant_grad[overlap])).float().mean()
    else:
        sign_agreement = torch.zeros((), device=scores.device)

    metrics: dict[str, Any] = {**nce_metrics, **ant_metrics}
    metrics.update(
        {
            "weighted_nce_loss": nce_alpha * nce_metrics["nce_loss"],
            "weighted_ant_loss": ant_beta * ant_metrics["ant_loss_raw"],
            "weighted_ant_loss_adjusted": ant_beta * ant_metrics["ant_loss_adjusted"],
            "combined_loss": nce_alpha * nce_metrics["nce_loss"] + ant_beta * ant_metrics["ant_loss_raw"],
            "weighted_nce_grad_norm": float((nce_alpha * nce_grad).norm().item()),
            "weighted_ant_grad_norm": float((ant_beta * ant_grad).norm().item()),
            "combined_grad_norm": float(combined_grad.norm().item()),
            "ant_to_nce_grad_ratio": float(
                (ant_beta * ant_grad).norm().div((nce_alpha * nce_grad).norm().clamp_min(1e-30)).item()
            ),
            "nce_ant_grad_cosine": _cosine(nce_alpha * nce_grad, ant_beta * ant_grad),
            "nce_ant_sign_agreement": float(sign_agreement.item()),
        }
    )
    tensors = {
        **ant_tensors,
        "cos_sim": scores,
        **nce_tensors,
        "combined_grad": combined_grad,
    }
    return GeometryAnalysis(metrics=metrics, tensors=tensors)
