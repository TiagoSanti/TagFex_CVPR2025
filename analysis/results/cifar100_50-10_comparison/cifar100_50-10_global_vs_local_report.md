# CIFAR-100 50-10: Global vs Local Anchor Comparison

## Experiment Configuration

**Scenario:** CIFAR-100, 50 initial classes + 10 incremental tasks

**Compared Experiments:**
- **Global:** exp_cifar100_50-10_antB0_nceA1_antM0_antGlobal
- **Local:** exp_cifar100_50-10_antB0_nceA1_antM0_antLocal

**Key Settings:**
- `ant_beta = 0` (InfoNCE only, no ANT regularization)
- `ant_margin = 0` (no margin constraint)
- `nce_alpha = 1` (InfoNCE weight)

**Goal:** Validate whether local anchors provide benefit even without explicit ANT loss in a harder scenario (50-10 vs 10-10).

## Summary Statistics

| Metric | Global | Local | Δ (Local - Global) |
|--------|--------|-------|--------------------|
| Final NME1 | 7011.00% | 6935.00% | -76.00% |
| Avg Gap | 0.8754 | 0.8772 | +0.0018 |
| Avg Violation % | 0.00% | 0.00% | +0.00% |

## Per-Task NME1 Comparison

| Task | Global NME1 | Local NME1 | Δ (Local - Global) |
|------|-------------|------------|--------------------|
| 1 | 8384.00% | 8384.00% | +0.00% |
| 2 | 8043.00% | 8122.00% | +79.00% |
| 3 | 7824.00% | 7846.00% | +22.00% |
| 4 | 7456.00% | 7455.00% | -1.00% |
| 5 | 7198.00% | 7146.00% | -52.00% |
| 6 | 7011.00% | 6935.00% | -76.00% |

## Analysis

### ⚠ Local anchors show **-76.00%** degradation

Local anchors underperform global anchors in the 50-10 scenario. This might indicate:

1. **Harder scenario requires explicit ANT loss** (ant_beta > 0)
2. **Local anchors alone insufficient** for larger initial task
3. **May need margin or gap maximization** for 50-10 setup

### Gap Evolution

- **Global average gap:** 0.8754
- **Local average gap:** 0.8772
- **Difference:** +0.0018

Local anchors maintain **larger gaps** on average, suggesting better separation.

## Conclusion

This comparison validates the impact of anchor type (global vs local) in the challenging 50-10 scenario when using InfoNCE without ANT loss. Results guide whether to:

1. Keep ant_beta=0 with local anchors
2. Add ANT loss (ant_beta > 0) for 50-10
3. Explore gap maximization or margin settings

