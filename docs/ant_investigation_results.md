# ANT Loss Investigation — Results Report

**Dataset**: CIFAR-100, 10-task split (10 classes/task)  
**Backbone**: ResNet-18  
**Seed**: 1995  
**ANT config**: β=0.5, margin γ=0.5, symmetric full, InfoNCE local anchor  
**Run date**: 2026-05-19 / 2026-05-20, quaTII (RTX 4090)

---

## 1. Motivation

The raw ANT loss appeared nearly constant throughout training (~5.57), raising the
question of whether the loss function was actually doing anything. Investigation
revealed a **count-floor phenomenon**: `logsumexp(relu(v))` has a structural floor
at `log(N_valid) ≈ log(254) ≈ 5.537` because non-violating valid negatives contribute
`exp(relu(0)) = exp(0) = 1` each to the sum.

A secondary masking bug was also fixed: after `relu(-inf) = 0`, self and positive-pair
slots were contributing `exp(0) = 1` each, slightly inflating the floor. Fixed with
`mq.masked_fill(~valid_neg_mask, float("-inf"))` before `logsumexp`.

---

## 2. Experiment Overview

| # | Config | Formulation | Status | Final avg_acc1 | Final avg_nme1 |
|---|--------|-------------|--------|:--------------:|:--------------:|
| 1 | `stage1_diag` | logsumexp + diagnostics | ✅ Complete | **79.67** | **76.17** |
| 2 | `stage2_logsumexp` | logsumexp (clean) | ✅ Complete | **79.67** | **76.17** |
| 3 | `stage2_expm1` | expm1 | ✅ Complete | 79.39 | 75.87 |
| 4 | `stage2_softplus_tau0.1` | softplus τ=0.1 | ✅ Complete | 79.43 | 76.03 |
| 5 | `stage2_topk32` | top-k (k=32) | ✅ Complete | 79.32 | 75.82 |
| 6 | `stage2_activeonly` | active_only | ✅ Complete | 79.56 | 76.14 |

> Stage 1 and Stage 2 produce identical numbers: the diagnostic overhead has zero
> effect on training.

---

## 3. Final Performance (Completed Experiments)

### 3.1 Average Accuracy Over All Tasks (avg\_acc1 / avg\_nme1)

| Formulation | avg\_acc1 | avg\_acc5 | avg\_nme1 | avg\_nme5 | Δ acc1 vs logsumexp |
|-------------|:---------:|:---------:|:---------:|:---------:|:-------------------:|
| **logsumexp** (baseline) | **79.67** | **95.17** | **76.17** | **94.26** | — |
| expm1       | 79.39 | 95.04 | 75.87 | 94.39 | −0.28 |
| softplus τ=0.1 | 79.43 | 95.26 | 76.03 | 94.43 | −0.24 |
| topk-32 (k=32) | 79.32 | 95.06 | 75.82 | 94.19 | −0.35 |
| active_only | 79.56 | 95.17 | 76.14 | 94.30 | −0.11 |

**Conclusion**: None of the alternative formulations beat `logsumexp` on avg_acc1 or
avg_nme1. The differences are within ±0.3 pp — well within run-to-run variability for
a single seed. The original formulation is fine.

---

### 3.2 Per-Task acc1 Curves

| Task | logsumexp | expm1 | softplus τ=0.1 | topk-32 | active_only |
|------|:---------:|:-----:|:--------------:|:-----------------:|
| T1   | 93.40 | 93.40 | 93.40 | 93.40 |
| T2   | 85.55 | 86.05 | 86.15 | 85.65 |
| T3   | 85.63 | 84.17 | 84.53 | 84.87 |
| T4   | 81.72 | 81.18 | 82.02 | 81.85 |
| T5   | 79.60 | 78.92 | 79.70 | 78.98 |
| T6   | 77.98 | 77.80 | 77.87 | 77.78 |
| T7   | 76.60 | 76.27 | 76.40 | 75.56 |
| T8   | 74.12 | 73.86 | 73.08 | 73.72 |
| T9   | 71.54 | 71.63 | 70.77 | 71.12 |
| T10  | 70.51 | 70.61 | 70.36 | 70.30 |

### 3.3 Per-Task nme1 Curves

| Task | logsumexp | expm1 | softplus τ=0.1 | topk-32 | active_only |
|------|:---------:|:-----:|:--------------:|:-------:|:-----------:|
| T1   | 93.00 | 93.00 | 93.00 | 93.00 | 93.00 |
| T2   | 85.55 | 85.25 | 86.30 | 85.70 | 86.35 |
| T3   | 83.30 | 83.20 | 83.30 | 83.07 | 83.90 |
| T4   | 79.83 | 78.60 | 79.28 | 79.03 | 79.10 |
| T5   | 76.30 | 75.86 | 76.52 | 75.96 | 75.84 |
| T6   | 73.82 | 73.78 | 73.63 | 73.45 | 73.72 |
| T7   | 71.66 | 71.41 | 71.47 | 70.74 | 71.40 |
| T8   | 68.28 | 68.01 | 67.74 | 67.85 | 68.20 |
| T9   | 65.74 | 65.37 | 65.08 | 65.37 | 65.46 |
| T10  | 64.23 | 64.17 | 63.96 | 64.05 | 64.44 |

---

## 4. ANT Flattening Diagnostics (Stage 1 — Task 1, logsumexp)

Data from `stage1_diag_debug0.log`, first batch (B1) of each epoch, full batches only
(num_neg_mean = 254). Floor: `log(254) ≈ 5.537`.

| Epoch | raw\_loss | adj\_loss | active\_ratio | hard\_neg\_sim | sim\_gap\_mean |
|-------|:---------:|:---------:|:-------------:|:--------------:|:--------------:|
| E1    | 5.9815 | 0.4441 | 100.0% | 0.943 | −0.046 |
| E3    | 5.6272 | 0.0899 | 39.6%  | 0.942 | −0.289 |
| E10   | 5.5801 | 0.0427 | 21.6%  | 0.835 | −0.333 |
| E20   | 5.5731 | 0.0357 | 18.4%  | 0.750 | −0.111 |
| E30   | 5.5713 | 0.0340 | 18.0%  | 0.738 | −0.056 |
| E60   | 5.5683 | 0.0310 | 15.8%  | 0.724 | −0.003 |
| E100  | 5.5717 | 0.0344 | 19.6%  | 0.663 | +0.133 |
| E150  | 5.5675 | 0.0301 | 17.7%  | 0.675 | +0.107 |
| E200  | 5.5666 | 0.0293 | 17.2%  | 0.678 | +0.099 |

### Interpretation

- **raw_loss**: barely moves (5.98 → 5.57), appearing flat. This is the count-floor
  artifact. The floor log(254) ≈ 5.537 dominates.
- **adj_loss** (= raw_loss − log(N_valid)): drops from 0.444 → 0.029, a **~93% reduction**,
  revealing the actual learning signal.
- **active_ratio**: falls from 100% → ~17%. Most negatives stop violating the margin.
- **hard_neg_sim**: falls from 0.943 → 0.678. The hardest negatives are progressively
  pushed away.
- **sim_gap_mean** (pos_sim − hardest_neg_sim): transitions from negative (−0.33) to
  positive (+0.13), meaning by E100 the average anchor's positive is more similar to
  it than the hardest negative. ANT is successfully organizing the feature space.

> **Key finding**: `raw_loss` is not a useful diagnostic metric for the `logsumexp`
> formulation. Use `ant_loss_adj`, `active_ratio`, and `hard_neg_sim` instead.

### Note on viol\_max\_act

When `ant_max_global=True`, `viol_max_act` is structurally fixed at `γ = 0.5` because
the anchor-max reference pair always has `v = s_max − s_max + γ = γ`. It is not an
informative metric in this mode. Better tail metrics: `raw_v_p90`, `raw_v_p95`
(added in this investigation).

---

## 5. Main Findings

1. **The logsumexp count-floor is real but benign for optimization.** The adjusted
   loss and active ratio confirm ANT is actively reshaping the feature space. The
   issue is diagnostic only — the optimizer sees gradients from the violation terms,
   not from the constant floor.

2. **The logsumexp masking fix (exp(0) from self/positive slots) was a correctness
   bug.** It inflated the floor by `log(256/254) ≈ 0.008`. Fixed. Does not change
   final performance.

3. **None of the alternative formulations improve over logsumexp** on CIFAR-100 10×10
   with seed 1995. Differences vs baseline: expm1 −0.28, softplus −0.24, topk-32 −0.35,
   active_only −0.11 avg_acc1. All are within ±0.4 pp — within single-seed noise.
   The count floor is not harming generalization in this setting.

4. **Recommendation**: Keep the current `logsumexp` formulation. Report training
   progress using `ant_loss_adj` and `active_ratio` rather than `ant_loss_raw`.

---

## 6. Configuration Details

All experiments use:

```
backbone:        ResNet-18
dataset:         CIFAR-100, 10-task (10 classes/task)
epochs_task1:    200
epochs_task2+:   170
batch_size:      128 (effective: 256 with symmetric views)
ant_beta:        0.5
ant_margin:      0.5
ant_symmetric_full: true
infonce_max_global: false  (local anchor for InfoNCE)
ant_max_global:  true
seed:            1995
```

| Experiment | ant_formulation | ant_tau | ant_topk |
|------------|:---------------:|:-------:|:--------:|
| logsumexp  | logsumexp | — | — |
| expm1      | expm1     | — | — |
| softplus   | softplus  | 0.1 | — |
| topk-32    | topk      | — | 32 |
| active_only| active_only | — | — |

---

*Generated: 2026-05-20 — all 6 experiments complete*
