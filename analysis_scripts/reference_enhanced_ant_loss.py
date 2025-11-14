"""
Enhanced ANT Loss Implementation
Implements adaptive margin and gap maximization strategies
"""

import torch
import torch.nn.functional as F


def compute_adaptive_margin(
    epoch,
    max_epochs,
    initial_margin=0.1,
    target_margin=0.5,
    current_gap=None,
    gap_ratio=0.8,
):
    """
    Compute adaptive margin based on training progress and current gap.

    Args:
        epoch: Current epoch
        max_epochs: Total epochs for the task
        initial_margin: Starting margin value
        target_margin: Maximum margin value
        current_gap: Current gap between pos and neg (optional)
        gap_ratio: Target ratio of margin to gap (default 0.8 = 80% of gap)

    Returns:
        Adaptive margin value
    """
    # Linear schedule based on epoch
    progress = min(epoch / max_epochs, 1.0)
    scheduled_margin = initial_margin + (target_margin - initial_margin) * progress

    # If we have gap information, adjust margin to be challenging but achievable
    if current_gap is not None:
        gap_based_margin = current_gap * gap_ratio
        # Use the minimum to ensure margin doesn't exceed reasonable bounds
        return min(scheduled_margin, gap_based_margin)

    return scheduled_margin


def ant_loss_with_gap_maximization(
    feats,
    t,
    nce_alpha,
    ant_beta,
    ant_margin,
    max_global,
    logger=None,
    task=None,
    epoch=None,
    batch=None,
    enable_gap_max=True,
    gap_target=0.7,
    gap_beta=0.5,
):
    """
    Enhanced ANT loss with gap maximization.

    Args:
        feats: Feature embeddings
        t: Temperature
        nce_alpha: NCE loss weight
        ant_beta: ANT loss weight
        ant_margin: Margin for ANT
        max_global: Use global max vs per-anchor max
        logger: Logger instance
        task, epoch, batch: Training context
        enable_gap_max: Enable gap maximization loss
        gap_target: Target gap value (pos - neg)
        gap_beta: Weight for gap maximization loss

    Returns:
        Dictionary with loss components
    """
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

    # ANT (Adaptive Negative Thresholding) logic
    pos_start = cos_sim.shape[0] // 2
    cos_sim_q1 = cos_sim[:pos_start, :pos_start]

    mask_q1 = torch.eye(cos_sim_q1.shape[0], dtype=bool, device=cos_sim.device)
    q1 = cos_sim_q1.masked_fill(mask_q1, 0.0)  # ignore self-similarity

    # Compute positive similarities
    pos_sims = torch.diagonal(cos_sim[:pos_start, pos_start:])

    # Compute current gap for adaptive margin
    pos_mean = pos_sims.mean()
    neg_mean = q1[~mask_q1].mean()
    current_gap = pos_mean - neg_mean

    if max_global:
        # Global maximum across all anchors
        q1_max = q1.max()
        mq1 = F.relu_(q1 - q1_max + ant_margin)
    else:
        # Maximum per anchor (per row)
        q1_max = q1.max(dim=-1, keepdim=True).values
        mq1 = F.relu_(q1 - q1_max + ant_margin)

    # Compute base ANT loss
    ant_loss = torch.logsumexp(mq1, dim=-1).mean()

    # Gap maximization loss
    gap_loss = torch.tensor(0.0, device=feats.device)
    if enable_gap_max:
        # Penalize if gap is below target
        gap_deficit = F.relu(gap_target - current_gap)
        gap_loss = gap_deficit

    # Combined ANT loss
    total_ant_loss = ant_loss + gap_beta * gap_loss

    # Compute NCE loss for comparison
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim_masked = cos_sim.masked_fill(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    logits = cos_sim_masked / t
    labels = torch.where(pos_mask)[1]
    nce_loss = F.cross_entropy(logits, labels)

    # Combined loss
    combined_loss = nce_alpha * nce_loss + ant_beta * total_ant_loss

    # Log detailed statistics
    if logger is not None:
        q1_nonzero = q1[~mask_q1]

        if max_global:
            gaps = q1_nonzero - q1_max
        else:
            q1_max_expanded = q1_max.expand_as(q1)
            gaps = (q1 - q1_max_expanded)[~mask_q1]

        mq1_nonzero = mq1[~mask_q1]
        violations = (mq1_nonzero > 0).float()
        violation_pct = violations.mean().item() * 100

        stats = {
            "pos_mean": pos_mean.item(),
            "pos_std": pos_sims.std().item(),
            "pos_min": pos_sims.min().item(),
            "pos_max": pos_sims.max().item(),
            "neg_mean": neg_mean.item(),
            "neg_std": q1_nonzero.std().item(),
            "neg_min": q1_nonzero.min().item(),
            "neg_max": q1_nonzero.max().item(),
            "gap_mean": gaps.mean().item(),
            "gap_std": gaps.std().item(),
            "gap_min": gaps.min().item(),
            "gap_max": gaps.max().item(),
            "margin": ant_margin,
            "violation_pct": violation_pct,
            "ant_loss": ant_loss.item(),
            "gap_loss": gap_loss.item(),
            "current_gap": current_gap.item(),
        }

        logger.log_ant_distance_stats(stats, task=task, epoch=epoch, batch=batch)

        # Log loss components
        loss_components = {
            "infoNCE_nll": nce_loss.item(),
            "infoNCE_ant_loss": ant_loss.item(),
            "infoNCE_gap_loss": gap_loss.item(),
            "infoNCE_total_ant": total_ant_loss.item(),
            "infoNCE_nce_weighted": (nce_alpha * nce_loss).item(),
            "infoNCE_ant_weighted": (ant_beta * total_ant_loss).item(),
            "infoNCE_total": combined_loss.item(),
        }
        logger.log_loss_components(
            loss_components, prefix="contrast", task=task, epoch=epoch, batch=batch
        )

    return {
        "total": combined_loss,
        "nce_loss": nce_loss,
        "ant_loss": ant_loss,
        "gap_loss": gap_loss,
        "total_ant_loss": total_ant_loss,
        "current_gap": current_gap,
    }


def hard_negative_mining_ant_loss(
    feats,
    t,
    nce_alpha,
    ant_beta,
    ant_margin,
    max_global,
    hard_negative_ratio=0.5,
    logger=None,
    task=None,
    epoch=None,
    batch=None,
):
    """
    ANT loss with hard negative mining - focus only on hardest negatives.

    Args:
        hard_negative_ratio: Fraction of hardest negatives to use (0.5 = top 50%)
    """
    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

    pos_start = cos_sim.shape[0] // 2
    cos_sim_q1 = cos_sim[:pos_start, :pos_start]

    mask_q1 = torch.eye(cos_sim_q1.shape[0], dtype=bool, device=cos_sim.device)
    q1 = cos_sim_q1.masked_fill(mask_q1, 0.0)

    if max_global:
        # Select top-k hardest negatives globally
        q1_flat = q1[~mask_q1]
        k = int(len(q1_flat) * hard_negative_ratio)
        hard_neg_vals, _ = torch.topk(q1_flat, k)
        q1_max = hard_neg_vals.max()

        # Only compute loss on hard negatives
        hard_mask = q1 >= hard_neg_vals.min()
        hard_mask = hard_mask & ~mask_q1

        mq1 = torch.zeros_like(q1)
        mq1[hard_mask] = F.relu(q1[hard_mask] - q1_max + ant_margin)
    else:
        # Per-anchor hard negative mining
        k_per_anchor = int(q1.shape[1] * hard_negative_ratio)

        # Get top-k per row
        topk_vals, topk_idx = torch.topk(q1, k_per_anchor, dim=-1)
        q1_max = topk_vals.max(dim=-1, keepdim=True).values

        # Create mask for hard negatives
        hard_mask = torch.zeros_like(q1, dtype=bool)
        hard_mask.scatter_(1, topk_idx, True)
        hard_mask = hard_mask & ~mask_q1

        mq1 = torch.zeros_like(q1)
        q1_max_expanded = q1_max.expand_as(q1)
        mq1[hard_mask] = F.relu(q1[hard_mask] - q1_max_expanded[hard_mask] + ant_margin)

    # Compute ANT loss
    ant_loss = torch.logsumexp(mq1, dim=-1).mean()

    # NCE loss
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim_masked = cos_sim.masked_fill(self_mask, -9e15)
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

    logits = cos_sim_masked / t
    labels = torch.where(pos_mask)[1]
    nce_loss = F.cross_entropy(logits, labels)

    combined_loss = nce_alpha * nce_loss + ant_beta * ant_loss

    return {
        "total": combined_loss,
        "nce_loss": nce_loss,
        "ant_loss": ant_loss,
    }
