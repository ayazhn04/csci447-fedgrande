# Phase 2E protocol: structure-aware Federated GRANDE (frozen before the final runs)

**Status:** frozen. The frozen commit is the commit that adds this file. The pre-freeze engineering state is `2da1d8f`.
The configuration is `configs/phase2e/structure.json`.

The full matrix was approved by the user after the runtime gate (projected about 7.4 h; see Benchmark). This is the
final experimental phase.

## Question

After feature-selection consensus becomes high, which parts of GRANDE still need continued federated optimization:
feature identities, split thresholds, or only prediction parameters? And how much communication does freezing them save?

## Gate 0 (`results/phase2e/gate0.json`): passed

The model is 256 trees, depth 5, 502 inputs, V = 50, with 809,984 trainable parameters (3.24 MB in FP32).

| Tensor | Share of params and payload |
|---|---|
| `split_values` | 48.99% |
| `split_index_array` | 48.99% |
| `estimator_weights` | 1.01% |
| `leaf_classes_array` | 1.01% |

Post-freeze steady-state traffic per round, against F0's 136.08 MB (21 clients, down + up):

| Variant | MB per round | Reduction |
|---|---|---|
| F1 | 69.41 | −49.0% |
| F2 | 2.75 | −98.0% |
| F3 | 1.72 | −98.7% |

All three are well above the 10% gate.

## Unchanged from Phase 2B

Everything below is identical to Phase 2B:
- preprocessing (502 features), the C2 folds with inner validation, and the natural and IID partitions;
- seeds 42, 447 and 2026;
- FedAvg with local Adam (betas (0.9, 0.95), per-group learning rates), 1 local epoch, full participation, a fresh
  optimizer every round, and at most 100 rounds;
- selection on validation AUC (patience 10, warm-up 10), thresholds, metrics, and 6 threads at "highest" precision.

The code path is `phase2b/run_job.py` → `fl/train.fedavg`.

## Variants

| | Rounds 1–20 | From round 21: frozen | From round 21: still optimized and communicated | Uplink after the freeze |
|---|---|---|---|---|
| **F0** | full FedAvg | – | all | FP32 (**Phase 2B results reused**) |
| **F1** | full FedAvg | `split_index_array` | `split_values`, `leaf_classes_array`, `estimator_weights` | FP32 |
| **F2** | full FedAvg | `split_index_array`, `split_values` | `leaf_classes_array`, `estimator_weights` | FP32 |
| **F3** | full FedAvg | `split_index_array`, `split_values` | `leaf_classes_array`, `estimator_weights` | INT8, using the Phase 2D codec unchanged |

- **The freeze round is 20**, fixed a priori from Phase 2B, where node consensus was already about 0.98 at round 20.
  It is not tuned, and there is no adaptive or consensus-triggered freezing.
- **Rounds 1–20 run as ordinary FP32 FedAvg** in every variant. They were verified identical to F0 in the benchmark.
  Freezing begins at round 21.

## Freeze implementation (`fl/train.fedavg`, `cfg["freeze"]`)

From round 21 onward, every round, for every client:
1. The client loads the global state, so the frozen tensors equal the round-20 global values.
2. Frozen parameters are set to `requires_grad=False`, so no gradients are computed for them.
3. The local Adam is rebuilt **only over the active parameter groups**, with the same per-group learning rates and betas.
   So no optimizer state or momentum ever touches a frozen tensor. Optimizer state is still reset every round.
4. After local training, the code asserts that the frozen tensors are bitwise unchanged on the client.
5. **Uplink and aggregation cover only the active tensors.** Frozen tensors are copied unchanged from the round-20 global
   model, so they are never averaged; averaging could otherwise change bits through floating-point rounding. The code
   asserts every round that the global frozen tensors are unchanged.
6. In F3, the active deltas pass through the Phase 2D INT8 round-trip (per-tensor scale, round-half-to-even, clip at ±127).

The `features_by_estimator` and path/node buffers are the same as in Phase 2B and are checked unchanged every round.

## Communication accounting (conceptual tensor bytes, computed round by round)

- **Initialization:** `features_by_estimator` (int64) is sent once to each client.
- **Rounds 1–20:** FP32 downlink and FP32 uplink of all four tensors, the same as F0.
- **Round 21:** the downlink still carries **all four tensors**, a final broadcast of the aggregated round-20 structure
  that clients do not yet hold. The uplink carries only the active tensors (INT8 plus 4-byte scales in F3).
- **Round 22 onward:** the downlink and uplink carry only the active tensors.
- **Reported:** bytes per round before and after the freeze; the post-freeze reduction; cumulative bytes through round 20,
  to the best round and to the final round; downlink and uplink separately; and the ratio to F0.
- **F0** is recomputed with the same definitions, which is the Phase 2D `accounting_v2`.
- **Consistency check:** the aggregator asserts that cumulative bytes = initialization bytes + the sum of per-round bytes.

## Per-round dynamics recorded

- Validation AUC and loss, and the weighted training loss.
- Active groups, bytes down and up.
- The relative movement of each global tensor, ‖θ_after − θ_before‖ / ‖θ_before‖.
- Node consensus and unanimity, and update cosine.
- The INT8 error, for F3.

## Structural-drift diagnostics (labelled DIAGNOSTIC; `phase2e/stability.py`)

**Why reruns are needed.** Phase 2B stored no intermediate states. So the ordinary F0 is rerun for seed 42, folds 0–4,
natural and IID: 10 runs. Each rerun saves global snapshots at rounds 20, 40 and 60 and at the best round, and is
**checked bitwise against Phase 2B** (test predictions and per-round validation AUC). One of these runs already exists
from the consistency check.

**Metrics, each relative to round 20:**
- **Feature-identity stability:** the fraction of nodes whose selected original feature is unchanged at round 40, at
  round 60 and at the best round.
- **Threshold drift:** for nodes whose selected feature is unchanged, |threshold_r − threshold_20|. For numeric features
  this is in training-fold SD units, because inputs are z-scored; one-hot features are reported separately on the 0/1
  scale.
- **Relative norm drift:** for `leaf_classes_array`, `estimator_weights`, `split_values` and `split_index_array`.

## Metrics

- Pooled-OOF ROC-AUC (primary), macro-F1 at 0.5, macro-F1 at the validation threshold, accuracy and confusion matrices.
- Per seed, then mean ± SD over the 3 seeds.
- ΔAUC and AUC retention against F0, paired by partition and seed.
- Paired fold/seed test-AUC differences (F1−F0, F2−F0, F3−F0, F2−F1, F3−F2), descriptive only. No significance tests.

## Benchmark (pre-freeze, fold 0, seed 42; runtime and validation only)

- **F0 consistency rerun:** bitwise identical to Phase 2B.
- **Wall time:** F1 356 / 378 s, F2 214 / 212 s, F3 220 / 215 s (natural / IID). Seconds per round after the freeze:
  F1 about 3.6, F2 and F3 about 1.75 (F0 about 3.5).
- **Projection:** about 6.6 h for the 90 jobs, plus about 0.8 h of diagnostics, about **7.4 h** in total. This is above the
  4.5 h convenience gate and was explicitly approved by the user.
- The benchmark runs are development artifacts (`results/phase2e/benchmark/`). The final matrix reruns those jobs.

## Interpretation (descriptive bands fixed in advance)

**AUC bands:**
- |ΔAUC| ≤ 0.005: essentially preserved.
- 0.005–0.010: small tradeoff.
- Larger: stated plainly.

**Outcome cases:**
- **A:** F1 ≈ F0 and F2 ≈ F0, so full tree structure stabilizes early.
- **B:** F1 ≈ F0 but F2 is worse, so feature identities stabilize but thresholds remain important.
- **C:** F1 and F2 both worse, so structural optimization remains necessary despite high consensus.
- **D:** F3 ≈ F2, so INT8 remains complementary after structural freezing.

The best Phase 2E configuration is compared with the existing Phase 2D C2 and C3 results, which are not rerun.

## Excluded

Other freeze rounds, adaptive freezing, 64-tree variants, Dirichlet, other quantization settings, sparsification, error
feedback, other optimizers, new models, personalization, privacy, and new data.

## Amendments

None at freeze time.
