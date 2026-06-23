# ANT Loss Flattening Investigation Plan

## 1. Context

Recent experiments show that `β=0.5 aSymFull nLocal` is one of the strongest ANT variants, especially on Tiny-ImageNet 100+20×5. However, debug logs show a suspicious pattern:

- InfoNCE / NLL loss decreases normally through training.
- Raw ANT loss remains almost constant after the first few epochs.
- `violation_pct` and `gap_mean` change, but `loss_ant_loss` itself barely decreases.
- Accuracy and NME sometimes improve despite the apparent ANT loss plateau.

This suggests that the current ANT scalar loss may not be a reliable diagnostic, or that the ANT objective is not applying sufficiently focused optimization pressure.

The goal of this investigation is to determine whether the ANT loss is:

1. Correctly implemented but poorly scaled/logged.
2. Dominated by a constant floor from the number of negatives.
3. Saturated because of the ReLU/log-sum-exp formulation.
4. Too weak relative to InfoNCE gradients.
5. Miscalibrated due to margin, temperature, or negative-set size.

---

## 2. Main Hypothesis

The current ANT loss may be dominated by a count-dependent lower bound:

\[
L_{ANT} = \log \sum_i \exp(\operatorname{ReLU}(v_i))
\]

where:

\[
v_i = s_{neg,i} - s_{ref} + \gamma
\]

For non-violating negatives:

\[
v_i \leq 0
\]

so:

\[
\operatorname{ReLU}(v_i) = 0
\]

but:

\[
\exp(0) = 1
\]

Therefore, every non-violating negative still contributes `1` to the log-sum-exp. This creates a lower bound approximately equal to:

\[
\log(|N|)
\]

where \(|N|\) is the number of negatives.

This may explain why the raw ANT loss remains almost constant even when the representation improves.

---

## 3. Investigation Goals

### Goal 1 — Verify whether ANT is actually optimizing violations

Add diagnostics that separate:

- raw ANT loss,
- count-floor-adjusted ANT loss,
- number of active violations,
- severity of active violations,
- hardest-negative behavior.

### Goal 2 — Determine whether the current ANT loss is count-floor dominated

Compare:

\[
L_{ANT}
\]

against:

\[
L_{ANT} - \log(|N|)
\]

If the adjusted loss decreases while raw loss stays flat, then the current scalar logging is misleading.

### Goal 3 — Determine whether the ANT formulation needs modification

If neither adjusted loss nor active violation severity decreases, then the current ANT objective may be weak or saturated. In that case, test alternative formulations.

---

## 4. Diagnostics to Add

Add the following metrics to `exp_debug0.log` or equivalent debug logger.

### 4.1 Negative-set size

```python
num_negatives = valid_negative_mask.sum(dim=-1)
num_negatives_mean = num_negatives.float().mean()
```

Purpose:

- Needed to estimate the expected log-count floor.
- Helps compare `aLocal` vs `aSymFull`.

---

### 4.2 Raw violation values

Assuming the ANT violation term is approximately:

```python
raw_violation = neg_sim - reference_sim + gamma
```

Log:

```python
raw_violation_mean = raw_violation[valid_negative_mask].mean()
raw_violation_min = raw_violation[valid_negative_mask].min()
raw_violation_max = raw_violation[valid_negative_mask].max()
```

Purpose:

- Shows whether negatives are actually moving farther below the margin.

---

### 4.3 Active violation mask

```python
active_mask = valid_negative_mask & (raw_violation > 0)
active_count = active_mask.sum(dim=-1)
active_count_mean = active_count.float().mean()
active_ratio = active_count.float() / num_negatives.clamp_min(1).float()
active_ratio_mean = active_ratio.mean()
```

Purpose:

- Directly measures how many negatives violate the ANT constraint.

---

### 4.4 Violation severity

```python
relu_violation = torch.relu(raw_violation)

violation_mean_all = relu_violation[valid_negative_mask].mean()

if active_mask.any():
    violation_mean_active = relu_violation[active_mask].mean()
    violation_max_active = relu_violation[active_mask].max()
else:
    violation_mean_active = torch.tensor(0.0, device=relu_violation.device)
    violation_max_active = torch.tensor(0.0, device=relu_violation.device)
```

Purpose:

- `violation_mean_all`: average violation diluted by easy negatives.
- `violation_mean_active`: average severity among actually violating negatives.
- `violation_max_active`: hardest violation.

If `violation_mean_active` decreases, ANT is doing something even if raw loss is flat.

---

### 4.5 Count-floor-adjusted ANT loss

```python
ant_floor = torch.log(num_negatives.float().clamp_min(1))
ant_loss_adjusted = ant_loss_per_anchor - ant_floor
ant_loss_adjusted_mean = ant_loss_adjusted.mean()
```

If ANT loss is reduced over all anchors as a scalar, compute:

```python
global_num_negatives = valid_negative_mask.sum()
global_ant_floor = torch.log(global_num_negatives.float().clamp_min(1))
ant_loss_adjusted_global = ant_loss_raw - global_ant_floor
```

Purpose:

- Tests whether raw ANT loss is mostly `log(number_of_negatives)`.

---

### 4.6 Positive and hard-negative similarity

```python
positive_sim_mean = positive_sim.mean()

masked_neg_sim = neg_sim.masked_fill(~valid_negative_mask, float("-inf"))
hardest_negative_sim = masked_neg_sim.max(dim=-1).values
hardest_negative_sim_mean = hardest_negative_sim.mean()

sim_gap = positive_sim - hardest_negative_sim
sim_gap_mean = sim_gap.mean()
sim_gap_min = sim_gap.min()
```

Purpose:

- Confirms whether the representation is improving geometrically.
- If positive similarity increases and hardest-negative similarity decreases, ANT may be helping even if raw loss is flat.

---

### 4.7 Gradient contribution diagnostics

At least for one debug run, estimate gradient magnitudes separately:

```python
# Pseudocode only
grad_norm_nce = compute_grad_norm(loss_nce)
grad_norm_ant = compute_grad_norm(beta * loss_ant)
grad_ratio_ant_to_nce = grad_norm_ant / (grad_norm_nce + 1e-12)
```

Purpose:

- Determines whether ANT gradients are too small compared to InfoNCE.
- If `grad_ratio_ant_to_nce` is near zero, ANT is mostly decorative.
- If it is too large, ANT may destabilize representation learning.

---

## 5. Expected Diagnostic Patterns

### Case A — Raw ANT loss flat, adjusted loss decreases

Interpretation:

- ANT is working.
- Raw ANT scalar is misleading because of the count floor.
- Keep formulation but report better diagnostics.

Action:

- Use `ant_loss_adjusted`, `violation_mean_active`, and `hardest_negative_sim` in analysis.
- Do not overinterpret raw `loss_ant_loss`.

---

### Case B — Raw and adjusted ANT losses both flat, but active violation severity decreases

Interpretation:

- ANT may still be working, but log-sum-exp is insensitive.
- Active-violation metrics are better indicators.

Action:

- Consider changing the loss to better track active violations.
- Test `exp(ReLU(v)) - 1` or top-k variants.

---

### Case C — Raw loss, adjusted loss, and active violation severity all flat

Interpretation:

- ANT is not being effectively optimized.
- Possible causes:
  - margin too hard,
  - margin too easy,
  - gradients too weak,
  - masking bug,
  - detach bug,
  - negative set too large and diluted.

Action:

- Inspect implementation.
- Run loss formulation ablations.
- Check gradient flow.

---

### Case D — ANT metrics improve, but Acc1 does not

Interpretation:

- ANT improves geometry but does not help classifier performance.
- Could still explain NME gains.

Action:

- Compare Acc1 vs NME1.
- Inspect classifier bias and class-prototype quality.

---

## 6. Code-Level Checks

Before changing the objective, verify the implementation.

### 6.1 Check for accidental detach

Search for:

```python
.detach()
with torch.no_grad()
tensor.data
```

around:

- similarity matrix construction,
- positive similarity extraction,
- negative similarity extraction,
- ANT loss computation.

ANT must have gradient flow into the embeddings.

---

### 6.2 Check masking correctness

Verify that the negative mask excludes:

- self-comparisons,
- positive pairs,
- invalid task/class entries,
- any excluded views.

For each batch, log:

```python
num_self_masked
num_positive_masked
num_valid_negatives
num_total_candidates
```

---

### 6.3 Check shape differences between `aLocal` and `aSymFull`

Log:

```python
similarity_matrix_shape
num_anchors
num_negatives_per_anchor_mean
num_negatives_per_anchor_min
num_negatives_per_anchor_max
```

Expected:

- `aSymFull` should use more negative relations than `aLocal`.
- If the negative count differs, raw loss values are not directly comparable.

---

### 6.4 Check margin scale

Log distribution summaries:

```python
positive_sim_mean
positive_sim_std
negative_sim_mean
negative_sim_std
hardest_negative_sim_mean
raw_violation_mean
raw_violation_max
```

Purpose:

- Determine whether `gamma` is appropriate for the similarity scale.

---

## 7. Loss Formulation Ablations

Run these only after diagnostics confirm the flattening source.

---

### 7.1 Current ANT

Baseline formulation for comparison.

```python
loss = torch.logsumexp(torch.relu(v), dim=-1).mean()
```

Expected issue:

- Non-violating negatives contribute `exp(0)=1`.
- Count floor may dominate.

---

### 7.2 Count-floor-corrected logging only

Do not change training loss. Only log:

```python
loss_adjusted = loss - torch.log(num_negatives.float()).mean()
```

Purpose:

- Determine whether the current method is actually improving hidden metrics.

---

### 7.3 Zero contribution from non-violating negatives

Change:

\[
\exp(\operatorname{ReLU}(v))
\]

to:

\[
\exp(\operatorname{ReLU}(v)) - 1
\]

Stable version:

```python
relu_v = torch.relu(v)
violation_mass = torch.expm1(relu_v)  # exp(relu_v) - 1
loss = torch.log1p(violation_mass.sum(dim=-1)).mean()
```

Expected benefit:

- Non-violating negatives contribute exactly zero.
- Loss should better reflect actual violation mass.

---

### 7.4 Softplus ANT

```python
loss = F.softplus(v / tau).mean()
```

Suggested temperatures:

```text
tau ∈ {0.05, 0.1, 0.2}
```

Expected benefit:

- Smoother gradients near the margin.
- Avoids hard ReLU plateau.

---

### 7.5 Top-k hard negative ANT

```python
v_masked = v.masked_fill(~valid_negative_mask, float("-inf"))
topk_v = torch.topk(v_masked, k=k, dim=-1).values
loss = torch.logsumexp(torch.relu(topk_v), dim=-1).mean()
```

Suggested values:

```text
k ∈ {16, 32, 64}
```

Expected benefit:

- Focuses ANT pressure on hard negatives.
- Prevents easy negatives from dominating the scalar.

---

### 7.6 Active-only ANT

```python
relu_v = torch.relu(v)
active = relu_v > 0

loss_per_anchor = relu_v.sum(dim=-1) / active.sum(dim=-1).clamp_min(1)
loss = loss_per_anchor.mean()
```

Expected benefit:

- Directly optimizes average violation severity.
- Less sensitive to negative-set size.

Potential issue:

- If few active negatives exist, gradients may become noisy.

---

## 8. Hyperparameter Sweeps

Only sweep after choosing the most promising corrected ANT formulation.

### 8.1 Margin sweep

```text
gamma ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
```

Desired behavior:

```text
early violation_pct: 50–70%
late violation_pct: 10–25%
```

If late violation remains too high:

- margin may be too hard,
- β may be too low,
- loss may be too diluted.

If violation collapses too early:

- margin may be too easy,
- ANT may stop contributing too soon.

---

### 8.2 β sweep

```text
β ∈ {0.1, 0.25, 0.5, 0.75, 1.0}
```

Track:

- Avg Acc1,
- Avg NME1,
- forgetting,
- ANT adjusted loss,
- gradient ratio ANT/NCE.

---

### 8.3 β schedule

Instead of fixed β, test:

#### Warmup schedule

```text
β = 0 for first 10–20 epochs
β = 0.5 afterward
```

#### Linear ramp

```text
β linearly increases from 0 to 0.5 during first 30 epochs
```

Rationale:

- Early embeddings are noisy.
- ANT may be more useful after InfoNCE creates a basic structure.

---

## 9. Minimal Experiment Matrix

Avoid a large combinatorial explosion.

### Stage 1 — Single-seed diagnostics

Dataset:

```text
CIFAR-100 10×10
```

Seed:

```text
1995
```

Run:

```text
β=0.5 aSymFull nLocal
```

Add all diagnostics from Section 4.

Goal:

- Determine whether raw ANT flattening is a logging artifact or real optimization failure.

---

### Stage 2 — Loss formulation comparison

Dataset:

```text
CIFAR-100 10×10
```

Seed:

```text
1995
```

Compare:

```text
1. current ANT
2. expm1/log1p ANT
3. softplus ANT
4. top-k ANT, k=32
```

Goal:

- Identify whether a corrected formulation produces a more meaningful ANT trajectory and better Acc1/NME1.

---

### Stage 3 — Validate on harder setting

Dataset:

```text
Tiny-ImageNet 100+20×5
```

Seeds:

```text
1993, 1994, 1995
```

Compare:

```text
1. baseline β=0 aGlobal nGlobal
2. current β=0.5 aSymFull nLocal
3. best corrected ANT formulation
```

Goal:

- Check whether the corrected ANT increases the already strong mean gain.

---

### Stage 4 — Final multi-seed confirmation

Run only the final selected variants:

```text
1. β=0 aGlobal nGlobal
2. β=0.5 aLocal nLocal
3. β=0.5 aSymFull nLocal current
4. β=best corrected aSymFull nLocal
```

Datasets:

```text
CIFAR-100 10×10
CIFAR-100 50+10×5
Tiny-ImageNet 100+20×5
Tiny-ImageNet 20×10
```

Goal:

- Determine whether corrected ANT increases effect size beyond seed-level variance.

---

## 10. Statistical Analysis Plan

Use paired seed comparisons wherever possible.

For each seed:

```text
delta_seed = metric_variant_seed - metric_baseline_seed
```

Report:

```text
mean(delta_seed)
std(delta_seed)
number of seeds improved
```

Prefer this over only comparing aggregate means.

Example table:

| Dataset | Variant | Mean Δ Acc1 | Std Δ Acc1 | Improved seeds |
|---|---:|---:|---:|---:|
| CIFAR-100 10×10 | aSymFull | ... | ... | ... |
| Tiny-ImageNet 100+20×5 | aSymFull | ... | ... | ... |

Interpretation:

- If all seeds improve but the absolute delta is small, claim consistent but modest improvement.
- If only one seed drives the mean, claim instability.
- If Tiny-ImageNet improves across all seeds, emphasize that as the strongest evidence.

---

## 11. Decision Criteria

### Keep current ANT formulation if:

- `ant_loss_adjusted` decreases,
- active violation severity decreases,
- NME improves consistently,
- no corrected formulation improves mean Acc1/NME1.

### Replace ANT formulation if:

- raw and adjusted ANT losses are flat,
- active violation severity is flat,
- corrected loss produces larger mean gains,
- corrected loss improves NME without hurting Acc1.

### Prefer top-k or active-only ANT if:

- `aSymFull` has too many easy negatives,
- raw loss is dominated by negative count,
- hard-negative metrics improve more clearly than full-negative metrics.

### Prefer softplus ANT if:

- ReLU causes abrupt saturation,
- violation severity is near zero but gradients vanish too early,
- training becomes smoother with better NME.

---

## 12. Expected Outcomes

### Best-case outcome

A corrected ANT formulation increases the mean effect size, especially on CIFAR-100, while preserving or improving the strong Tiny-ImageNet 100+20×5 result.

Expected narrative:

> The original ANT formulation already suggested representation benefits, but its raw loss was dominated by a negative-count floor. After correcting the objective to focus on active margin violations, ANT produced a clearer optimization signal and stronger mean improvements.

### Neutral outcome

Diagnostics show that ANT was working, but raw loss was misleading. Corrected formulations do not improve performance.

Expected narrative:

> ANT behaves more like a constraint-style regularizer than a conventional decreasing loss. Its scalar value is not directly interpretable because of the log-sum-exp count floor, so violation-based diagnostics are more appropriate.

### Negative outcome

Diagnostics show ANT gradients are weak and corrected losses do not improve performance.

Expected narrative:

> The current ANT formulation provides limited additional optimization pressure beyond local InfoNCE. Its observed gains may be due to mild regularization rather than a strongly optimized negative-space constraint.

---

## 13. Recommended Immediate Next Step

Implement diagnostic logging first, without changing the training objective.

Priority metrics:

```text
ant_loss_raw
ant_loss_adjusted
num_negatives_mean
active_ratio_mean
violation_mean_all
violation_mean_active
violation_max_active
positive_sim_mean
hardest_negative_sim_mean
sim_gap_mean
grad_ratio_ant_to_nce
```

Run:

```text
CIFAR-100 10×10
β=0.5 aSymFull nLocal
seed=1995
```

Then decide whether the loss needs reformulation.
