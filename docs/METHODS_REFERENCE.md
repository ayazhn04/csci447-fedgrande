# Methods reference

This is the authoritative technical reference for report writing. Every value below was taken from the frozen configs
(`configs/`) and protocols (`reports/PHASE2*_PROTOCOL.md`), not from memory. If this page and a frozen protocol ever
disagree, **the protocol wins**.

Related pages:
- [EXPERIMENT_MAP.md](EXPERIMENT_MAP.md): what was run.
- [RESULTS_LEDGER.md](RESULTS_LEDGER.md): the numbers.

## 1. Dataset

| | |
|---|---|
| **Source** | SpeedDating, **OpenML 40536**, version 1 (uploaded 2016-11-16; licence "Public"). ARFF MD5 `0c5590a0c81750687ee1b0ee68168a5d`, checked on download (`results/phase1/speeddating_download_manifest.json`). |
| **Rows** | **8,378**. Each row is one participant's view of one speed date. |
| **Target** | `match` (binary): class 0 = 6,998 rows, class 1 = 1,380 rows (**16.5% positive**). |
| **Predictors (OpenML)** | **120**, after OpenML's ignore attributes `decision` and `decision_o`. |
| **Predictors used (Phase 2B onward)** | **119 raw features (58 numeric, 61 nominal)**. That is the 120 minus `wave`, which is kept only as client metadata. |
| **Never used as features** | `match` (the target); `decision` and `decision_o` (label proxies: `match` = both decisions positive); `wave` (the natural client identifier; dropping it keeps the natural-vs-IID comparison clean). |
| **Loader** | `src/fedgrande/data.py` reads the checksummed ARFF offline. It was verified identical across both pinned GRANDE environments (`results/phase1/phase1_summary.json`). |

## 2. Leakage grouping (Phase 2A, `data/derived/phase2/dyads_v1.csv`)

- **Mirror rows.**
  - Most dates appear twice, once from each participant's perspective.
  - The two rows mirror each other's partner-perspective columns and share the same `match` label.
  - Exact mirror candidates were detected from those columns.
- **Leakage groups.** These are the **connected components** of the mirror-candidate graph. The group id is column
  `dyad_id`, called `leakage_group_id` in the code. There are **4,189 groups**:

  | Kind | Groups | Rows | Meaning |
  |---|---|---|---|
  | `pair_mutual` | 4,173 | 8,346 | Two rows, each the other's unique mirror: a confirmed dyad. |
  | `pair_one_way` | 2 | 4 | Two rows where the mirror relation holds in one direction only. |
  | `ambiguous_group` | 4 | 18 | Components of 4–6 rows with several candidates. These are **not confirmed dyads**, but they are kept together. |
  | `singleton` | 10 | 10 | No mirror found (all in wave 5). |

- **Terminology.** "4,175 dyads" means the 4,173 mutual plus 2 one-way pairs. **Do not call every connected component
  a dyad.** Ambiguous groups and singletons are leakage groups, not dyads.
- **Validation.** All pairs are same-wave, opposite-gender and same-`match`. All groups lie within a single wave.
  `decision` equals the mirror's `decision_o` (used only to validate the grouping, never as a feature).
- **The problem with row-wise CV** (scheme C1, the historical split):
  - **6,686 of 8,378 test rows (79.8%)** had their mirror in the training folds.
  - Under the grouped split (C2) this falls to **0**.

## 3. Evaluation protocol

- **Outer split C2.**
  - Group-level stratified 5-fold CV with seed 42 (`data/derived/phase2/splits_outer_v1.csv`).
  - No leakage group crosses folds.
  - The fold positive rate stays within 0.00065 of the overall rate.
- **Inner split.** Each outer fold's training part is split again by group into **fit** and **validation**
  (`splits_inner_v1.csv`). Per fold: fit 5,353–5,364 rows, validation 1,339–1,350, test 1,669–1,681. Every job asserts
  that no group crosses fit, validation or test.
- **Seeds.** Model and training seeds **42, 447, 2026**. Client-partition seed **7**. The outer-split seed (42) is frozen
  in the file.
- **Model selection.**
  - Pooled validation ROC-AUC after every epoch (centralized) or round (FedAvg).
  - An improvement must exceed 1e-4.
  - Stop after **patience 10**, with a **warm-up of 10**; the best-validation state is restored.
  - Test data is never used for stopping.
- **Threshold.** Per run, maximize validation macro-F1 over the grid 0.05..0.95 (step 0.01); ties go to the value closest
  to 0.5. The threshold is applied to that run's test predictions.
- **Aggregation.**
  - **Pooled out-of-fold (OOF):** test predictions of the 5 folds are pooled per (method, partition, seed). ROC-AUC is
    primary, plus macro-F1 at 0.5 and at the validation threshold, accuracy and the confusion matrix.
  - Results are reported as **mean ± SD over the 3 seeds**.
  - Fold × seed runs are **not independent**, so **no p-values** are reported anywhere.
- **Per-fold view** (Phase 2B amendment A3). Mean test metric over the 5 folds per seed, then mean ± SD over seeds.
  Report it next to pooled OOF whenever score levels shift between folds (Dirichlet; the early-stopped G2 runs).
- **Within-client AUC** (2B amendment A2; 2C amendment A1). AUC computed inside each client. In Phase 2C it is computed
  per (client, fold) cell, because pooling a client across folds mixes 5 different models.

## 4. Federated clients (`data/derived/phase2b/`)

| Partition | Definition | Role |
|---|---|---|
| **Natural** | 21 clients = the 21 waves (events). Sizes 50–968 rows (19.4× spread); positive rate 0.083–0.310. | Primary (realistic cross-silo). |
| **IID size-matched** | 21 synthetic clients with the natural sizes. Whole leakage groups are dealt by largest remaining positive/negative quota, seed 7. Positive rate 0.160–0.169; each client mixes 12–21 waves. | Control: same quantity skew, no label or feature shift. |
| **Dirichlet α = 0.1** | Natural sizes. Each client's share of positives is drawn from Dirichlet(0.1·1₂₁) (seed 7, stream 1), then whole groups are dealt. 12 clients have no positives and 4 are all-positive. | Stress test only (2B amendment A1); Phase 2C. |

No leakage group crosses clients in any partition.

## 5. Preprocessing (federated-safe, label-free; `src/fedgrande/fl/preprocess.py`)

Fitted separately in each outer fold on the **inner-fit rows only**, from the 21 natural clients' sufficient statistics.
It is identical to pooled computation and partition-invariant (unit-tested).

- **Numeric (58 columns).**
  - Clients send the per-column non-missing count, sum and sum of squares.
  - The server forms the global mean and population SD (SD = 1 if the variance is 0).
  - Missing values are imputed with the mean, then values are z-scored.
- **Categorical (61 columns).**
  - Clients send label-free category counts. The vocabulary is every category seen in training, in schema order.
  - Encoding is one-hot. A MISSING column exists only where training data had missing values (`race`, `race_o`,
    `field`).
  - Unseen categories, and missing values without a MISSING column, map to UNK, which is an all-zero block.
  - No target or leave-one-out encoding.
- **Result.** **502 encoded features** in every fold: 58 numeric plus 444 one-hot. 0 unseen validation/test categories.
- **Check.** On fold 0, the same centralized loop scored test AUC 0.884 with this preprocessing and 0.874 with stock
  GRANDE 1.0 preprocessing, which uses leave-one-out encoding of `field`.

## 6. GRANDE configuration (all federated and centralized GRANDE runs)

| Setting | Value |
|---|---|
| Implementation | GRANDE 1.0.0 `GRANDE_Module`, commit **db79c86060f06c15f46c3bfb9207dc30c3cb91ae**, used unmodified. Bitwise forward and training equivalence with the historical commit 07f7278b was verified in Phase 2A (E0a/E0b). |
| Estimators / depth | **256 / 5**: 31 internal nodes and 32 leaves per tree. Phase 2D C1/C3 use 64 trees. |
| `selected_variables` | 0.8. Effective V = min(max(min(int(0.8·F), 50), 10), F) = **50** features per estimator. |
| Trainable parameters | **809,984** (3.24 MB FP32): `split_values` [256,31,50] and `split_index_array` [256,31,50] have 396,800 each (48.99% each); `estimator_weights` [256,32] and `leaf_classes_array` [256,32] have 8,192 each (1.01% each). |
| Buffers | `features_by_estimator` [256,50] (int64), and the path and internal-node index lists. Never averaged; asserted unchanged every round. |
| Other | dropout 0, `data_subset_fraction` 1, bootstrap false, `missing_values` false, **no embeddings**, **no `torch.compile`**. |
| Loss | **Class-weighted cross-entropy**, w_k = (1/n_k) / Σ_j (1/n_j) from global training-fold counts. |
| Optimizer | **Adam, betas (0.9, 0.95)**, learning rates `split_values` 0.01, `split_index_array` 0.01, `estimator_weights` 0.005, `leaf_classes_array` 0.01. |
| Batches | max(1, min(256, n // 2)) per data holder, shuffled with a generator seeded by (seed, round/epoch, client), drop_last. |
| Budget | Centralized: at most 100 epochs. FedAvg: at most **100 rounds** of **1 local epoch**. FedAvg runs on natural/IID were still improving slowly at round 100, so they are **round-budget-limited**. |
| Runtime | 6 CPU threads, float32, `float32_matmul_precision` "highest". |

## 7. MLP baseline

- **Architecture.** Linear(502,128) → ReLU → Linear(128,64) → ReLU → Linear(64,2). No dropout or batch normalization.
  **72,770 parameters.**
- **Initialization.** PyTorch default, after `torch.manual_seed(seed)`.
- **Optimizer.** Adam, learning rate 1e-3, default betas.
- **Everything else** (loss, batches, selection, thresholds) matches GRANDE.

## 8. Training procedures

- **Centralized.** Pooled fit rows, at most 100 epochs.
- **Local-only.** One model per client.
  - Selection on that client's validation AUC.
  - If its validation set has one class: weighted validation loss instead.
  - If it has no validation rows: the median best epoch of the other clients.
  - Fewer than 2 fit rows: the model stays untrained.
  - Each row is scored by its own client's model.
- **FedAvg with local Adam** (not FedAdam; `fl/train.fedavg`):
  - Full participation of 21 clients.
  - Each round, every client loads the global trainable parameters, builds a **fresh Adam** (optimizer state is reset
    every round), trains 1 local epoch and returns its parameters.
  - The server averages trainable parameters **weighted by training-row count**. Clients with fewer than 2 fit rows get
    weight 0, which never happened.

## 9. Structural consensus (Phase 2B diagnostics; the Phase 2F gates)

- **Selected feature at node j.** For client k, node j = (estimator e, internal node i), the selected feature is
  `features_by_estimator[e, argmax_v split_index_array_k[e,i,v]]`. It is computed from the client's locally trained
  parameters before aggregation. The buffer is shared, so feature identities are comparable across clients.
- **Node consensus C_j.** The fraction of clients choosing the modal feature at node j (`fl/adaptive.node_consensus`).
- **Summaries.**
  - Mean consensus (mean_j C_j).
  - Unanimous fraction (share of nodes with C_j = 1).
  - Global agreement (share of client/node selections equal to the aggregated model's selection).
  - Threshold SD among clients choosing the modal feature (numeric nodes in training-fold SD units).
- **Update geometry.** Cosine of each client update to the weighted aggregate update (mean and minimum), plus update norms.

## 10. Communication accounting (conceptual tensor bytes; `fl/comm.accounting_v2`, plus per-round v3 in 2E/2F)

- Only tensor payloads are counted, with **no serialization or protocol overhead**. All 21 clients participate.
- **Downlink per client per round:** the FP32 trainable global parameters (4 bytes × parameters).
- **Uplink per client per round:**
  - FP32: 4 bytes × parameters.
  - INT8: 1 byte × parameters, plus a 4-byte scale per transmitted tensor.
- **At initialization:** `features_by_estimator` (int64, 256 × 50 × 8 bytes) once per client, i.e. 2,150,400 bytes in
  total. Index lists are regenerated from depth and cost 0 bytes.
- **F0 per round:** 21 × 2 × 3,239,936 = **136,077,312 bytes** (136.08 MB).
- **Reported:** cumulative GB to the best validation round and to the final round, downlink and uplink separately.
  "Compression" is a ratio of these conceptual bytes, **not measured network traffic**.

## 11. INT8 client-update compression (Phase 2D; `fl/compress.py`)

1. Clients train in FP32.
2. For each trainable tensor, Δ = θ_client − θ_global.
3. Symmetric per-tensor quantization: s = max|Δ|/127 and q = clip(round(Δ/s), −127, 127) as int8. `torch.round` rounds
   half to even. If Δ is all zeros, s = 0 and q = 0.
4. Uplink: q plus s (FP32) per tensor.
5. The server reconstructs θ̂ = θ_global + s·q in FP32, then runs ordinary weighted FedAvg.
6. The downlink stays FP32.
7. Not used: error feedback, stochastic rounding, sparsification or entropy coding.
8. Observed maximum relative L2 quantization error: about 0.4–0.5% per round.

## 12. Fixed structural freezing (Phase 2E; `fl/train.fedavg`, `cfg["freeze"]`)

| | Frozen from round 21 | Still trained and sent | Uplink after the freeze |
|---|---|---|---|
| **F0** | – | all | FP32 (Phase 2B reused) |
| **F1** | `split_index_array` | `split_values`, leaves, estimator weights | FP32 |
| **F2** | `split_index_array`, `split_values` | leaves, estimator weights | FP32 |
| **F3** | as F2 | as F2 | INT8 (Phase 2D codec) |

- The **freeze round is 20**, fixed a priori from the Phase 2B consensus (about 0.98 at round 20). It is not tuned.
- Rounds 1–20 are ordinary FedAvg, verified identical to F0.
- **Mechanics.**
  - Frozen tensors are set to `requires_grad=False`, and Adam is built only over the active groups.
  - Frozen tensors are copied unchanged from the round-20 global model; they are never averaged. They are asserted
    bitwise unchanged on every client and on the server.
- **Accounting.** Round 21's downlink still carries all tensors: a final broadcast of the round-20 structure. After
  that, only active tensors travel.
- **Post-freeze traffic per round:** F1 69.41 MB (−49.0%), F2 2.75 MB (−98.0%), F3 1.72 MB (−98.7%).

## 13. Consensus-gated progressive structural freezing, CGSF (Phase 2F; `fl/adaptive.py`)

All gate values were fixed in `configs/phase2f/cgsf.json` before any final run and never tuned. Decisions use
training-side information only: client-local parameters, global parameters and training-row feature SDs.

- **G1, global gate.**
  - After round r, for **r > 10**: if mean_j C_j ≥ **0.98** AND the unanimous fraction ≥ **0.85**, for **3 consecutive
    rounds**, then **all** nodes are fully frozen (feature logits and thresholds) from round r+1.
  - The counter resets on any failing round.
  - Earliest possible trigger: round 13.
- **G2, node-wise gates.**
  - *Feature gate:* node j's `split_index_array[e,i,:]` slice freezes after C_j ≥ **0.95** for **3 consecutive rounds**
    (r > 10).
  - *Threshold gate:* a node that is already feature-frozen additionally freezes its `split_values[e,i,:]` slice after
    **3 consecutive rounds** in which both:
    - D_j = |t_j(r) − t_j(r−1)| / (sd_f + 1e-6) ≤ **0.01** (drift of the global threshold), and
    - S_j = SD over clients of the client thresholds / (sd_f + 1e-6) ≤ **0.05** (dispersion).

    Here sd_f is the SD of the node's encoded feature over the inner-fit rows. Only rounds that start with the node
    already feature-frozen count.
  - Invariants: threshold-frozen ⊆ feature-frozen; freezing is irreversible.
- **Frozen-slice mechanics.**
  - Gradients are zeroed before every Adam step. A fresh Adam every round means its moments stay exactly 0.
  - Every step asserts the slice bitwise unchanged.
  - The server copies frozen slices from the previous global state (masked aggregation).
  - A fully frozen tensor leaves autograd entirely.
- **Communication.**
  - Slice-level: each [e,i,:] slice is 50 values.
  - A newly frozen slice is broadcast once more in the next downlink, then never again.
  - The uplink excludes it from the round after the decision.
- **G3.** G2 plus INT8 uplink.
  - Predeclared eligibility: G2's pooled-AUC loss against F0 ≤ 0.005 in both partitions AND G2's GB to best < F2's.
  - **Both failed, so G3 was not run** (decision record D1).
- **Interpretation bands** (fixed in advance; also used in 2E): |ΔAUC| ≤ 0.005 "essentially preserved"; 0.005–0.010
  "small tradeoff"; > 0.010 "meaningful degradation".

## 14. Personalization (Phase 2C; Dirichlet α = 0.1 only)

Starting point: the 15 Phase 2B Dirichlet FedAvg-GRANDE best-global states, regenerated and bitwise-verified against
Phase 2B.

| | Adapted | Frozen | Fitted on |
|---|---|---|---|
| **P0** | nothing | everything | – |
| **P1** | one logit intercept b_c per client | the whole model | client fit rows |
| **P2** | `leaf_classes_array` | everything else | client fit rows; client validation for selection |
| **P3** | `leaf_classes_array`, `estimator_weights` | split features, thresholds, buffers | client fit rows; client validation for selection |

- **P1 objective.** b_c = argmin Σ CE(y, σ(z + b)) + (λ/2) b² with **λ = 1.0**, solved by Newton's method. It cannot
  change the ranking within a client.
- **P2 and P3 training.**
  - Unweighted CE; Adam (0.9, 0.95); learning rates: leaves 0.01, estimator weights 0.005.
  - **At most 20 epochs, patience 5.**
  - Selection on client validation AUC, or validation CE if that client's validation set has one class.
  - Epoch 0 is always a candidate.
- **Communication.** Personalization is local, so it adds 0 bytes.
