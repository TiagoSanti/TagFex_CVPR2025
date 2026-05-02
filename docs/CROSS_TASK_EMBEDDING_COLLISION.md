# Cross-Task Embedding Collision in Continual Learning

**Author:** Tiago  
**Date:** April 25, 2026  
**Context:** Idea sparked at ICLR 2026, to be discussed with advisors/professors.

---

## 1. Core Intuition

In continual learning, catastrophic forgetting is typically attributed to the model overwriting old knowledge when learning new tasks. However, a subtler and underexplored failure mode may also be occurring:

> **When training on a new task, the contrastive loss pushes negative samples away from the anchor. But it provides no constraint on *where* those negatives land. A new class (e.g., lion) may drift toward the already-learned prototype of a semantically adjacent class (e.g., cat) — not because the model is forgetting cat, but because that region of embedding space was already shaped into a "good neighborhood" for similar visual features.**

This is not a forgetting event by itself, but it may *trigger* forgetting: once the lion embedding occupies space near cat's prototype, the classifier head and backbone updates start conflating the two, corrupting cat's decision boundary and causing accuracy drops on old tasks.

**The phenomenon is distinct from classical forgetting** because:
- The old class features may not degrade directly.
- The new class is not maliciously placed — it follows the gradient path of least resistance.
- It is invisible to the training loss, which only monitors within-task negatives.

---

## 2. Why This Happens in Our Setup (TagFex)

### 2.1 The Gap in the Current Loss

The current `kd_loss` (knowledge distillation from the frozen old TA-net) preserves the old backbone's feature representations for old data. However:

- It creates **no cross-task repulsion**: nothing prevents a new-task embedding from settling near an old class prototype.
- The InfoNCE + ANT loss only sees **within-batch negatives**, which are all from the current task.
- The within-task repulsion from ANT is local — it cannot see old-class regions of the embedding space.

### 2.2 Mechanism

```
During task T (lion, tiger):
  - InfoNCE pushes lion away from tiger (within-task ✓)
  - ANT adaptively thresholds hard negatives within task T (✓)
  - But: lion embedding slides toward cat prototype (old task, task 1) — UNDETECTED ✗
  
  Result: lion ≈ cat in embedding space
  → Classifier head updates begin conflating them
  → Weight alignment amplifies the problem
  → Cat accuracy drops: appears as "forgetting"
```

### 2.3 What We Already Have

The `class_means` list already stores normalized prototype vectors for all learned classes (used in NME evaluation). These prototypes live in the same embedding space as training embeddings. This means:

- **Detection is essentially free** — computing `cosine_similarity(new_embedding, old_class_means)` requires no new infrastructure.
- The existing `debug_similarity` logging pipeline is already in place for heatmaps and statistics.

---

## 3. Two Implementation Directions

### Direction A — Diagnostic (Non-invasive)

**Goal:** Empirically verify that cross-task embedding collision is actually occurring before designing a new loss term.

**Approach:** During `train_epoch`, after computing the batch `embedding`, compute:

$$\text{sim}(z_i, \mu_c) = \frac{z_i \cdot \mu_c}{\|z_i\| \|\mu_c\|}, \quad \forall c \in \{1, \ldots, C_{\text{old}}\}$$

Log and visualize when any new-task embedding exceeds a percentile threshold with respect to old-class prototypes. Track which class pairs are most affected.

**Advantages:**
- Zero impact on training dynamics — purely observational.
- Can be validated on **existing logs and checkpoints** without retraining.
- Provides empirical evidence to support or refute the hypothesis before committing to a new loss term.

---

### Direction B — Preventive Loss Term

**Goal:** Add a regularization term that directly penalizes proximity between new-task embeddings and old-class prototypes.

Two candidate formulations:

#### B.1 — Prototype-Augmented InfoNCE

Treat old class means as additional hard negatives in the InfoNCE denominator:

$$\mathcal{L}_{proto} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\exp(\text{sim}(z_i, z_i^+)/\tau) + \sum_{j \neq i} \exp(\text{sim}(z_i, z_j)/\tau) + \sum_{c=1}^{C_{\text{old}}} \exp(\text{sim}(z_i, \mu_c)/\tau)}$$

- Naturally integrates into the existing `_compute_contrastive_loss_base` function.
- Old prototypes act as persistent memory anchors in the contrastive landscape.

#### B.2 — Margin-Based Prototype Repulsion (Softer)

$$\mathcal{L}_{repel} = \frac{1}{N \cdot C_{\text{old}}} \sum_{i=1}^{N} \sum_{c=1}^{C_{\text{old}}} \text{ReLU}\left(\text{sim}(z_i, \mu_c) - \delta\right)$$

where $\delta$ is a margin threshold (similar in spirit to the existing `ant_margin`).

- More surgical: **only activates when encroachment actually occurs** (sim > δ).
- Easier to tune and interpret than augmenting the full InfoNCE denominator.
- Analogous to the existing ANT formulation — would fit architecturally.

---

## 4. Open Questions for Discussion

### 4.1 Scope of Embeddings to Monitor/Regularize
- Should the repulsion apply only to **new-task samples**, or also to **exemplars** in the replay buffer?
- Exemplars are supposed to already be stable — but do they drift under the new backbone update?

### 4.2 Scalability with Large $C_{\text{old}}$
- At large task counts (e.g., CIFAR-100 with 50+10 split → up to 90 old classes), computing pairwise sim against all old prototypes per batch may be expensive.
- A **top-k filtering** strategy (only repel against the k closest old prototypes per anchor) would reduce cost and focus regularization where it matters.
- What is the right value of k? Is the top-1 closest prototype sufficient?

### 4.3 Prototype Staleness
- `class_means` are updated after each task ends (herding-based). They are therefore **one task behind** during training.
- Is this staleness acceptable? Could stale prototypes mislead the repulsion term?
- Alternative: use the live class means computed from exemplar features at the start of each task.

### 4.4 Class Semantic Structure
- The collision hypothesis is strongest for **semantically adjacent classes** (lion ↔ cat, truck ↔ car).
- Should the repulsion weight be modulated by semantic similarity (e.g., from a pre-trained text embedding or class hierarchy)?
- Or is the embedding distance itself a sufficient proxy?

### 4.5 Interaction with Weight Alignment
- Weight alignment (post-task) is already correcting for classifier bias. Could a prototype repulsion term interfere with this correction?

### 4.6 Diagnostic First or Loss Term First?
- Running a diagnostic pass on **existing experiment logs and checkpoints** may confirm or refute the hypothesis without any new training runs.
- If the collision is not empirically observed, the loss term is not warranted.
- **Recommended order:** Diagnostic → confirm collision → design loss term.

---

## 5. Connection to Related Work

| Paper Theme | Relevance |
|---|---|
| CNN layer visualization of memorization vs. generalization (ICLR 2026) | Motivating observation — feature collapse visible not just in loss |
| iCaRL / class-mean classifiers | `class_means` already available; prototype-based methods understand this space |
| SupCon / supervised contrastive learning | Adding prototypes as negatives has precedent in supervised contrastive setups |
| Dark Experience Replay / DER++ | Replay + distillation doesn't solve cross-task feature space collisions explicitly |
| ANT (our work) | ANT repels within-task negatives; this extends the idea to cross-task prototype repulsion |

---

## 6. Proposed Next Steps

1. **[ ]** Run diagnostic logging on existing checkpoints — compute `sim(new_embedding, old_class_means)` across tasks and visualize.
2. **[ ]** Identify which class pairs show highest cross-task similarity (especially semantically adjacent ones).
3. **[ ]** Decide: Direction B.1 (InfoNCE augmentation) or B.2 (margin repulsion) based on diagnostic findings and professor feedback.
4. **[ ]** Implement chosen direction as a gated config flag (e.g., `proto_repel_beta: 0.0`) so it can be ablated cleanly.
5. **[ ]** Run ablation: baseline vs. +proto_repel on CIFAR-100 10-10 and 50-10 splits.
