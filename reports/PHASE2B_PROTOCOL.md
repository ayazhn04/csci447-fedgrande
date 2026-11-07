# Phase 2B protocol: Federated GRANDE on SpeedDating (frozen before the final runs)

**Status:** frozen. The frozen commit is the commit that adds this file. Every final result records `project_commit`
and `config_sha256`. If anything changes after the freeze, it is recorded below under *Amendments*, with a reason.

## 1. Questions

- **Primary.** Can GRANDE's differentiable hard-tree parameters be trained effectively by federated parameter
  averaging across heterogeneous clients? How much performance is lost relative to matched centralized training, and
  how does the loss depend on client heterogeneity (natural waves vs a size-matched IID control)?
- **Mechanistic.** Does heterogeneity make clients learn incompatible tree structures, especially different split-feature
  selections? Does lower structural consensus go with a larger centralized-to-federated gap?

This is not an attempt to beat state-of-the-art tabular models on accuracy.

## 2. Data, features and leakage groups

- **Data.** SpeedDating, OpenML 40536 v1. Input is the checksummed ARFF from Phase 1, read offline by `fedgrande.data`.
- **Features.** The 120 OpenML predictors minus `wave`, which leaves **119 raw features: 58 numeric and 61 nominal**
  (types from the ARFF schema).
- **Never used as features.** `match` (the target), `decision` and `decision_o` (label proxies), and `wave`. `wave` is
  kept only as client metadata. Reason for dropping it: it is the natural client identifier, and keeping it would
  confound the natural-vs-IID comparison. No other predictor is removed.
- **Leakage groups.** `leakage_group_id` is the Phase 2A connected-component id (`dyads_v1.csv`, column `dyad_id`).
  There are 4,189 groups:
  - 4,175 **dyads**: 4,173 mutual pairs and 2 one-way pairs;
  - 4 **ambiguous leakage groups** of 4–6 rows each. These are *not* confirmed dyads;
  - 10 singletons.
- **Outer evaluation.** Phase 2A C2 split (`splits_inner_v1.csv`): group-level stratified 5-fold. Each fold has inner
  fit / validation / test roles, also grouped.
- **Assertions.** Every job asserts that no leakage group crosses fit/val, fit/test or val/test. Every partition asserts
  that no group crosses clients.
- **Fold sizes.** Fit rows: 5,353–5,364 per fold. Validation: 1,339–1,350. Test: 1,669–1,681.

## 3. Federated-safe preprocessing

The preprocessing is shared by all five methods and uses no labels (`fedgrande/fl/preprocess.py`). It is fitted
separately in each outer fold, on the **inner-fit rows only**; validation and test rows never contribute.

- **Numeric columns.**
  - Each client sends, for each column: non-missing count, sum, and sum of squares (float64).
  - The server computes the global mean and population SD; the SD is set to 1 if the variance is 0.
  - Missing values are imputed with the global mean, then values are z-scored.
- **Categorical columns.**
  - Each client sends per-category counts (no labels). The vocabulary is the set of categories seen in aggregated
    training data, in schema order.
  - Encoding is one-hot. A dedicated MISSING column exists only if training data contains missing values for that column.
  - Unseen categories, and missing values in a column without a MISSING column, map to UNK, which is an all-zero block.
- **Not used.** Target, mean or leave-one-out encoding, or any transformation that uses `y`.
- **Computation.** The statistics are computed from the 21 natural clients' sufficient statistics. A unit test shows this
  equals pooled train-only computation and is partition-invariant (natural vs IID clients give the same schema and
  values within 1e-5).
- **Resulting dimensions.** **502 encoded features in every fold**: 58 z-scored numeric columns and 444 one-hot columns,
  including MISSING columns for `race`, `race_o` and `field`. `field` keeps all 259 levels in every fold. There were 0
  unseen validation or test categories in every fold (`results/phase2b/prepare/prepare_report.json`).

## 4. Client partitions (fixed; independent of model seeds and folds)

All partitions are stored in `data/derived/phase2b/partitions_v1.csv`. Within a fold, a client's fit, validation and test
rows are its rows intersected with that fold's roles.

- **Natural.** 21 clients, one per wave (`wave` is metadata only).
- **IID size-matched.**
  - 21 synthetic clients. Their target sizes are the natural wave sizes.
  - Whole leakage groups are dealt by largest remaining positive/negative row quota. Positive-row targets are
    proportional to client size (largest remainder). Partition seed: **7**.
  - Achieved: maximum size deviation 1 row, client positive rates 0.160–0.169 (natural clients: 0.083–0.310). Each IID
    client mixes 12–21 waves.
- **Dirichlet label skew (α = 0.1).** Not part of the core freeze. If it is run after the core matrix (B3), it will be
  added by amendment as a controlled synthetic heterogeneity stress test, with leakage groups intact and a fixed seed.

## 5. Seeds

- Outer split: Phase 2A C2 (seed 42, frozen file).
- Client partition seed: **7**.
- Model and training seeds: **42, 447, 2026**. Phase 2A used only seed 42, so there was no existing three-seed
  convention; the specification's default set is adopted.

Partitions are never regenerated per seed. Data order within an epoch uses a generator seeded by
(seed, round or epoch, client).

## 6. Models

| Model | Specification |
|---|---|
| GRANDE | GRANDE 1.0.0 `GRANDE_Module` at commit db79c860, used unmodified. 256 estimators, depth 5, `selected_variables` 0.8 (V = min(int(0.8·502), 50) = **50**), dropout 0, `data_subset_fraction` 1, bootstrap false, `missing_values` false, no embeddings, **no compile**. **809,984 trainable parameters.** |
| MLP | Linear(502, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 2). No dropout and no batch normalization, so it is FedAvg-safe. PyTorch default initialization after `torch.manual_seed(seed)`. **72,770 trainable parameters.** |

The settings below apply to every method.

- **GRANDE optimizer.** Adam with betas (0.9, 0.95). Learning rates (the GRANDE 1.0 defaults): `split_values` 0.01,
  `split_index_array` 0.01, `estimator_weights` 0.005, `leaf_classes_array` 0.01.
- **MLP optimizer.** Adam, learning rate 1e-3, default betas.
- **Loss.** Class-weighted cross-entropy. The weights come from **global training-fold** class counts:
  w_k = (1/n_k) / Σ_j (1/n_j). This is the same formula as GRANDE 1.0.
- **Batches.** Each data holder (the central pool, or each client) uses batch size `max(1, min(256, n // 2))`, shuffled,
  with drop_last. This is GRANDE 1.0's rule.
- **Runtime.** 6 torch CPU threads, float32, `float32_matmul_precision` = "highest".

## 7. Methods

1. **Centralized GRANDE (matched).** Pooled fit rows, the same preprocessing, features, class weights, architecture and
   optimizer, and the same fit/validation split. At most 100 epochs. This is the comparator that measures the cost of
   federation.
2. **Local-only GRANDE.** One model per client, no collaboration, the same initialization seed and the same architecture.
   Model selection per client uses that client's validation ROC-AUC. Edge cases are handled explicitly; no client is
   silently dropped:
   - **Validation has one class:** select on weighted validation loss instead.
   - **No validation rows:** train for the median best epoch of the other clients in that fold and seed.
   - **Fewer than 2 fit rows:** the model stays untrained.

   Each validation or test row is predicted by its own client's model.
3. **FedAvg GRANDE.** Described in §8.
4. **Centralized MLP.** Same rules as method 1.
5. **FedAvg MLP.** Same rules as method 3.

## 8. Federated algorithm: FedAvg with local Adam

This is **FedAvg with local Adam**, not FedAdam.

- **Initialization.** One global model is initialized from the seed. Every client receives the identical global state,
  including identical non-trainable buffers.
- **Each round, for each of the 21 clients (full participation):**
  1. Load the global trainable parameters.
  2. Create a **fresh local Adam**: optimizer state is reset every round.
  3. Train **1 local epoch** on the client's fit rows.
  4. Return the trainable parameters.
- **Aggregation.** Trainable parameters are averaged, weighted by **client training-row count**. Clients with fewer than
  2 fit rows get weight 0; this never occurred in the development folds.
- **Buffers.** Every round asserts that the GRANDE buffers (`features_by_estimator` and the path/node index lists) are
  unchanged. They are never averaged.
- **Rounds.** At most 100.

## 9. Validation, stopping and thresholds

- **Selection.** After each epoch (centralized/local) or round (FedAvg), compute pooled validation ROC-AUC.
  - An improvement means AUC > best + 1e-4.
  - Stop once epoch/round ≥ 10 (warm-up) and there has been no improvement for **10** (patience).
  - The best-validation state is restored. Test data is never used for stopping.
- **Threshold.** Per run, maximize validation macro-F1 over the grid 0.05..0.95 (step 0.01), ties going to the value
  closest to 0.5. The threshold is applied to that run's test predictions. Macro-F1 is reported both at 0.5 and at this
  validation threshold.
- **Development observation, frozen as-is.** In the development run (FedAvg GRANDE, natural, fold 0, seed 42; only
  validation and runtime inspected), validation AUC was still slowly improving at round 100 (best round 93). The 100-round
  cap is kept as specified, so FedAvg results are **round-budget-limited**.

## 10. Metrics and aggregation

- **Global.** Test predictions are pooled out-of-fold over the 5 folds, separately for each (method, partition, seed).
  - Primary metric: ROC-AUC.
  - Also: macro-F1 at 0.5, macro-F1 at the validation threshold (each fold's own threshold), accuracy (secondary) and
    the confusion matrix.
  - Aggregation: **mean ± SD over the 3 seeds**. These are repeated runs on the same folds, not independent datasets, so
    no p-values are reported.
  - Paired per-fold differences are reported descriptively.
- **Per client.**
  - For every method, metrics are computed per natural wave from that seed's pooled OOF predictions.
  - Across waves: mean, median, SD and minimum per seed, then mean ± SD over seeds.
  - Small waves (6, 18, 20, with 3–5 positive dyads) are reported, not hidden, and are flagged as unstable.
- **Gaps.**
  - Centralized − FedAvg, for GRANDE and for the MLP.
  - Centralized − local, and FedAvg − local.
  - IID − natural for each federated method.
  - All are computed per seed on pooled OOF AUC (and on F1 at the validation threshold).
- **Structural consensus (FedAvg GRANDE, every round).** Computed from each client's locally trained parameters before
  aggregation. A node's selected *original encoded* feature is `features_by_estimator[e, argmax split_index_array[e, i]]`;
  the buffer is shared, so identities are comparable across clients.
  - Per node: modal feature and the fraction of clients choosing it. Summaries: mean and median consensus, fraction of
    unanimous nodes, and the fraction of modal features that are numeric.
  - Global agreement: the fraction of client/node selections equal to the new global model's selection.
  - Threshold dispersion: the SD of thresholds among clients that chose the modal feature. It is reported separately for
    numeric-feature nodes (in training-fold SD units, because inputs are z-scored) and one-hot nodes (0/1 scale).
  - Update disagreement: client update norm mean and SD, aggregate update norm, cosine of each client update to the
    weighted aggregate (mean and minimum), and the same cosine restricted to `split_index_array`.
- **Structural diagnostics outside FedAvg.**
  - Local-only GRANDE: consensus among the 21 final local models.
  - FedAvg-vs-centralized: the fraction of nodes whose final selected feature matches the centralized model with the
    same fold and seed. Both share the same initialization and feature subsets.
- **Relation to the gap.** Spearman correlation between per-run consensus and the per-fold centralized − FedAvg AUC gap,
  across all runs and within each partition. It is descriptive only.
- **Communication.** Counted from the actual tensors: float32 trainable parameters, server→clients and clients→server
  each round.
  - `features_by_estimator` (int64) is counted **once** at initialization.
  - The path/node index lists are regenerated from depth and cost 0 bytes.
  - Reported per round, cumulative to the best round, and cumulative to the final round.
- **Runtime and memory.** Wall time per job and peak RSS per job process.

## 11. Execution

- **B2 core.**
  - 5 methods; natural and IID partitions for the partitioned methods; C2 folds 0–4; seeds 42, 447 and 2026.
  - That is **120 jobs**, run serially with one subprocess per job via `src/fedgrande/phase2b/run_matrix.py --stage core`.
    The runner stops if free disk falls below 8 GiB.
  - Projected about 5.5–6.5 h (development run: FedAvg GRANDE 368 s for 100 rounds; peak RSS 2.7 GB).
- **Declared auxiliary analyses** (`configs/phase2b/auxiliary.json`; outside the matched comparison):
  - **N. Preprocessing check.** Stock GRANDE 1.0 preprocessing vs federated-safe preprocessing, both trained with the
    *same* Phase 2B centralized loop, on C2 fold 0 with 3 seeds.
  - **J. Stock GRANDE 1.0 reference.** `GRANDEClassifier` with its own preprocessing and loop; 1,024 estimators, depth 5,
    embeddings off, class weights on, compile off; seed 42, 5 folds, `wave` removed. Labelled "Stock GRANDE 1.0
    reference".
- **B3** (only after B2 passes and only if the budget allows): Dirichlet α = 0.1, by amendment.
- **Excluded:** differential privacy, secure aggregation, federated XGBoost, vertical FL, participant reconstruction,
  personalization, broad HPO, entmax, extra datasets.

## 12. Development-only artifacts

`results/phase2b/smoke/` (3-step runs) and `results/phase2b/dev_runs/` (one full FedAvg GRANDE run on fold 0, seed 42)
come from pre-freeze engineering. They are not part of any result.

## Amendments

### A1 — Dirichlet α = 0.1 stress test (added after B2 completed and was verified; before any B3 run)

- **Why.** The specification asked for this optional B3 step. B2 finished in 4.1 h of compute with 0 failures, and disk
  stayed at 24 GiB free. B2 showed natural ≈ IID for FedAvg, so a controlled stronger heterogeneity is needed to
  separate the hypotheses.
- **Design.**
  - 21 clients with **the natural wave sizes**. Quantity skew is therefore identical to the natural and IID partitions;
    only the label distribution changes.
  - Each client's share of the positive rows is drawn from Dirichlet(0.1·1₂₁), using partition seed 7 with stream id 1.
  - Positive targets are water-filled under client-size caps; negative target = size − positive target.
  - Whole leakage groups are dealt by quota, using the same routine as the IID partition.
- **Outcome** (`results/phase2b/prepare/partition_dirichlet0.1.json`): sizes within 1 row of target; 12 clients have
  0 positives and 4 are all positive; no leakage group crosses clients.
- **Files.** The partition is stored separately in `data/derived/phase2b/partitions_dirichlet_v1.csv`. The frozen
  `partitions_v1.csv` is untouched.
- **Code changes.** All in the same commit as this amendment:
  - `partitions.dirichlet_size_matched`;
  - `run_job.py` reads the separate partition file when the partition is `dirichlet0.1`;
  - `run_matrix.py` gains a `dirichlet` stage;
  - `aggregate.py` gains Dirichlet gap entries.
- **Unchanged.** Everything else (preprocessing, folds, seeds, models, training, stopping, thresholds and metrics) is
  unchanged from `core.json` (`configs/phase2b/amendment_A1_dirichlet.json`).
- **Runs.** Only the partitioned methods are run: local, FedAvg GRANDE and FedAvg MLP, × 5 folds × 3 seeds = 45 jobs.
  The centralized B2 runs are reused because they are partition-independent.
- **Label.** "Controlled synthetic heterogeneity stress test". The local-only edge-case rules of §7 apply unchanged.
  Many clients have single-class validation sets and therefore use validation-loss selection.

### A2: Post-hoc within-client AUC (added during B3, after observing a pooled-metric artifact; no runs changed)

- **Observation.** During B3 execution, local-only GRANDE on the Dirichlet α = 0.1 partition reached pooled test ROC-AUC
  of about 0.98.
- **Cause.** That partition assigns clients *by label*, so client identity encodes the label prior. Each local model
  learns its client's base rate. When test rows from different clients are pooled, the ranking is dominated by client
  identity, not by features. This is a client-identity shortcut in the pooled metric; it is not information leakage
  across train and test.
- **Implication.** The frozen pooled metrics remain the primary metrics and are reported unchanged. But pooled AUC is
  **not interpretable for local-only under Dirichlet**. It is also distorted, more mildly, whenever client priors differ.
- **Added analysis, secondary and labelled post hoc.** Within-client ROC-AUC on each partition's *own* clients, computed
  from pooled-OOF test rows, only for clients whose test rows contain both classes. It is computed for every method in
  each partition view, centralized methods included.
- **Code.** `aggregate.py`, section "POST-HOC". No training runs, configs or primary metrics changed.
