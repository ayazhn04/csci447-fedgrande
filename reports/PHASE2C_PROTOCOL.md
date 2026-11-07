# Phase 2C protocol: shared GRANDE structure, personalized outputs (frozen before the final runs)

**Status:** frozen. The frozen commit is the commit that adds this file. The configuration is
`configs/phase2c/personalization.json`. This is the final experimental extension.

## Question

Under extreme label skew (Dirichlet α = 0.1), can GRANDE keep its globally shared tree structure while personalizing only
prediction-related parameters? And does any benefit go beyond a one-parameter client-intercept recalibration?

## Starting points: the Phase 2B global models, not retrained

Phase 2B saved predictions, round logs and split selections for every job, but not model states. The 15 Dirichlet
FedAvg-GRANDE best-global states (5 C2 folds × seeds 42, 447, 2026) were therefore **regenerated** with the unchanged
Phase 2B job code. The only addition was an optional `--save-state` flag.

They were accepted as *the* Phase 2B checkpoints only after a bitwise check against Phase 2B
(`results/phase2c/checkpoint_regeneration.json`; 15 of 15 verified). The check covered:
- validation and test probabilities;
- per-round validation AUC and loss;
- selected split features and thresholds;
- best round.

Every Phase 2C run asserts again that its loaded checkpoint reproduces the Phase 2B predictions. The comparison is at
float32 precision because the network outputs are float32 and Phase 2B's CSV stores them as decimal text, which rounds at
about 1e-17.

## Data (identical to Phase 2B)

- The same per-fold preprocessing caches (`data/derived/phase2b/prep_C2_outer{k}.npz`).
- The same Dirichlet α = 0.1 partition (`partitions_dirichlet_v1.csv`).
- The same inner fit / validation / test rows, with leakage-group assertions.
- No client-specific preprocessing and no vocabulary rebuilding.

## Methods

All four methods are evaluated on the same rows. Each validation or test row is predicted by **its own client's** model.

| Method | What changes | Fitted on |
|---|---|---|
| **P0** Global FedAvg GRANDE | nothing (the Phase 2B best global model) | – |
| **P1** FedAvg GRANDE + client intercept calibration | one additive logit intercept b_c per client: logit(p) = logit(p_global) + b_c | client FIT rows only |
| **P2** personalized leaves | `leaf_classes_array` only | client FIT rows; client validation for selection |
| **P3** personalized leaves + estimator weights | `leaf_classes_array` and `estimator_weights` | client FIT rows; client validation for selection |

**P1 details**
- b_c = argmin_b Σ_i CE(y_i, σ(z_i + b)) + (λ/2) b², with **λ = 1.0**.
  - z_i is the global model's logit, log(p/(1−p)), on the client's fit rows.
  - CE is unweighted cross-entropy.
  - This is equivalent to a MAP estimate with a N(0, 1) prior on b.
- Solved by Newton's method (at most 100 iterations, tolerance 1e-12). It is deterministic, and b is finite for
  all-positive and all-negative clients.
- P1 cannot change the ranking within a client.

**P2 and P3: frozen parameters**
- `split_index_array`, `split_values` and all buffers (`features_by_estimator`, `path_identifier_list`,
  `internal_node_index_list`) are frozen (`requires_grad=False`).
- In P2, `estimator_weights` is frozen as well.
- After personalization, every client model is checked **bitwise**: frozen tensors and buffers must be identical to the
  global checkpoint, and only the intended tensors may differ.

**P2 and P3: training**
- Objective: unweighted cross-entropy on the client's fit rows. This is the same objective family as P1, so that both
  adapt to the client's own label prior.
- Optimizer: Adam with betas (0.9, 0.95). Learning rates are the GRANDE 1.0 values for these groups: leaves 0.01,
  estimator weights 0.005. One fixed rate per group; no tuning.
- Budget: **max 20 epochs, patience 5**, min_delta 1e-4.
- Batches: `max(1, min(256, n//2))`, shuffled with a generator seeded by (seed, client, epoch), drop_last.
- Selection: client validation ROC-AUC if the client's validation rows contain both classes. Otherwise, a
  **predeclared deterministic fallback**: validation cross-entropy.
  - Epoch 0 (the unpersonalized global state) is always a candidate, and the best state is restored.

**Budget decision (from the smoke runs on fold 0, seed 42; validation and runtime only)**
- For single-class-validation clients (19 of 21 in that fold), validation loss decreases monotonically up to any budget
  that was tried (20 and 50 epochs). This happens *by construction*: cross-entropy on a single label pushes logits toward
  that label without limit. No budget is ever non-binding for these clients, so the epoch cap acts as an implicit
  regularizer, much like λ in P1.
- For mixed-class clients, patience stopped training by epoch 9 at the latest.
- So the suggested 20 epochs / patience 5 is kept.
- Smoke artifacts are in `results/phase2c/smoke*` and are not results.

## Thresholds

For each (method, fold, seed), the validation macro-F1 is maximized on the pooled validation predictions over the grid
0.05..0.95 (step 0.01), ties going to the value closest to 0.5. This is the same procedure as Phase 2B.

## Metrics

- **Per outer fold** (primary): ROC-AUC, macro-F1 at 0.5, macro-F1 at the validation threshold, log loss (probabilities
  clipped to [1e-7, 1−1e-7]), Brier score, and ECE (10 equal-width bins).
- **Per seed:** mean over the 5 folds (with the SD across folds). Across seeds: **mean ± SD over the 3 seeds**.
- **Paired differences:** fold-level differences P2−P1, P3−P1, P1−P0 and P3−P2, reported descriptively. No p-values.
- **Pooled OOF:** also reported, but not a headline metric (Phase 2B amendment A3).
- **Within-client** (pooled OOF test predictions per Dirichlet client and seed):
  - For clients whose test rows contain both classes: ROC-AUC, macro-F1 at 0.5 and at the run's validation threshold,
    log loss, and Brier score.
  - For single-class clients: log loss, Brier and accuracy only. **No AUC is computed; undefined AUCs are never
    averaged as 0.**
  - The numbers of mixed and single-class clients are stated.
- **How to read the results:**
  - P1 cannot change within-client ranking. P2 or P3 beating P1 on *within-client AUC* is evidence of changed
    discrimination.
  - Gains only in calibration or pooled metrics indicate prior/calibration adaptation.

## Interpretation rules (fixed in advance)

- **Tree-specific personalization helps:** P2 or P3 materially beats P1, especially in within-client discrimination, or
  consistently across per-fold metrics.
- **Recalibration explains the benefit:** P1 captures nearly all gains, and P2/P3 do not meaningfully improve on it.
- **Personalization is harmful or unsupported:** P2/P3 are worse than P0 and P1.

## Communication and runtime

- Personalization runs locally after the global model has been distributed. Nothing is sent back, so the additional
  communication is **0 bytes**. The global FedAvg cost is taken from Phase 2B.
- Runtime is recorded per client (P1, P2, P3), per fold-seed run, and in total. Projected total is about 12 min for the
  15 runs.

## Excluded

Personalization of split features or thresholds, HPO, other α values, other partitions, FedProx, FedAdam, new datasets,
new neural models, DP, secure aggregation, and participant reconstruction.

## Amendments

None at freeze time.
