# Phase 2F protocol: consensus-gated progressive structural freezing (CGSF), frozen before the final runs

**Status:** frozen. The frozen commit is the commit that adds this file. The pre-freeze implementation checkpoint is
`906aef5` (the user-requested pause). The configuration is `configs/phase2f/cgsf.json`.

This is the final experimental phase. No Phase 2G follows.

## Question

Can the fixed round-20 structural freeze of Phase 2E (F2) be replaced by a freeze that is *triggered by client
consensus*, either:
- globally (G1), or
- node by node, with feature identities first and thresholds second (G2),

and does that give equal AUC at lower cumulative communication, or other useful behaviour?

## Unchanged from Phases 2B and 2E

Everything below is identical to Phases 2B and 2E:
- preprocessing (502 encoded features; label-free and federated-safe; fit on inner-fit rows only);
- the C2 grouped outer folds 0–4 with grouped inner validation;
- the natural-wave partition (21 clients) and the size-matched IID partition (21 clients, partition seed 7);
- model seeds 42, 447 and 2026;
- GRANDE 1.0.0 `GRANDE_Module` (db79c860): 256 trees, depth 5, `selected_variables` 0.8 (V = 50);
- FedAvg with local Adam: betas (0.9, 0.95), per-group learning rates, a fresh optimizer every round, 1 local epoch,
  full participation, weights = training-row counts, at most 100 rounds;
- selection on pooled validation AUC (patience 10, warm-up 10, min_delta 1e-4);
- the validation-threshold rule, metrics and pooled-OOF aggregation;
- 6 threads at "highest" float32 matmul precision.

The third-party GRANDE source is unmodified.

## Methods

| ID | Description | Source |
|---|---|---|
| **F0** | FedAvg-GRANDE, no freezing | Phase 2B `fedavg_grande`, **reused** |
| **F2** | fixed freeze of `split_index_array` and `split_values` after round 20 | Phase 2E `fedavg_grande_frzstruct`, **reused** |
| **G1** | adaptive global structural freeze | new: `fedavg_grande_cgsf_global` |
| **G2** | progressive node-wise structural freeze | new: `fedavg_grande_cgsf_node` |
| **G3** | G2 plus INT8 uplink (Phase 2D codec) | **conditional** (see the G3 section) |

Leaves (`leaf_classes_array`) and `estimator_weights` are always trained and aggregated in every method.

## Signals (computed after each round's local training, from training-side information only)

For each internal node j = (estimator e, node i):

- **Selected feature.** The feature that client k's split selects at node j is
  `features_by_estimator[e, argmax(split_index_array[e,i,:])]`, computed from the client's locally trained parameters.
- **Node consensus C_j.** The fraction of active clients that choose the modal feature at node j
  (`fl/adaptive.node_consensus`).
- **Global threshold t_j.** The threshold of the aggregated global model at node j, i.e. `split_values` at the selected
  slot.
- **Threshold drift D_j** = |t_j(r) − t_j(r−1)| / (sd_f + ε), where:
  - f is the node's selected global feature;
  - sd_f is the standard deviation of that encoded feature over the inner-fit rows. This is obtainable from aggregated
    sufficient statistics. Inputs are z-scored, so sd ≈ 1 for numeric features and √(p(1−p)) for one-hot features;
  - ε = 1e-6.
- **Threshold dispersion S_j** = SD over clients of the client-local thresholds at node j, divided by (sd_f + ε).

None of these signals use validation or test data. Validation is used only for best-round selection, exactly as in F0.
A unit test checks that freezing decisions do not depend on validation data.

## G1: adaptive global freeze (exact rule)

- After round r, for **r > 10** (warm-up 10), check:
  - mean_j C_j ≥ **0.98**, AND
  - the fraction of nodes with C_j = 1 (unanimous) ≥ **0.85**.
- If the check holds for **3 consecutive rounds**, every node becomes fully frozen (feature logits and thresholds) from
  round r + 1 on. The counter resets to 0 on any failing round.
- The trigger round is r. The earliest possible trigger is round 13, so freezing takes effect from round 14 at the
  earliest.
- Freezing is irreversible.
- If the check never holds for 3 consecutive rounds, G1 is identical to F0 and the run is reported as "never triggered".

## G2: progressive node-wise freeze (exact rules)

**Stage A: feature gate.**
- Node j is **feature-frozen** (its slice `split_index_array[e,i,:]`) after round r when, for r > 10,
  C_j ≥ **0.95** for **3 consecutive rounds**.
- The node's counter resets to 0 on any failing round.

**Stage B: threshold gate.**
- A node that is already feature-frozen is additionally **threshold-frozen** (its slice `split_values[e,i,:]`) after
  round r when D_j ≤ **0.01** AND S_j ≤ **0.05** for **3 consecutive rounds**.
- Only rounds that *start* with the node already feature-frozen count toward these 3 rounds. So thresholds can freeze at
  the earliest 3 rounds after the feature freeze took effect.
- The counter resets to 0 on any failing round.

**Invariants (asserted in code):**
- threshold-frozen ⊆ feature-frozen;
- freezing is irreversible;
- decisions after round r take effect from round r + 1.

**What a frozen slice means:**
- Clients zero its gradient before every optimizer step. A fresh Adam is built every round, so its moments stay exactly
  0. The slice is asserted bitwise unchanged after every step.
- The server copies the slice from the previous global state instead of averaging it, which is masked aggregation.
- When an entire tensor is frozen, it is removed from autograd and from the optimizer, as in Phase 2E.

## G3: G2 with INT8 uplink (conditional, decided after the final G1/G2 runs)

- **Eligibility (predeclared).** Both conditions must hold:
  1. G2's pooled-OOF AUC loss against F0 is ≤ 0.005 in both partitions (mean over seeds, i.e. ΔAUC ≥ −0.005), AND
  2. G2's mean cumulative GB to the best round is below F2's in both partitions.

  This is computed by `phase2f/aggregate.py`, `G3_eligibility`.
- **If eligible.** An amendment commit adds `fedavg_grande_cgsf_node_int8`, specified exactly as in `G3_conditional` in
  the config: G2's gates, plus the Phase 2D symmetric per-tensor INT8 codec applied to the *transmitted* (non-frozen)
  slices of all four tensors' deltas. G3 then runs on the same 30 cells (2 partitions × 5 folds × 3 seeds).
- **If not eligible.** G3 is not run, and the report says why.

## Communication accounting (conceptual tensor bytes, slice level, round by round)

- **Initialization:** `features_by_estimator` is sent once to each client (as in Phases 2D and 2E).
- **Downlink in round r:**
  - all non-frozen slices (FP32);
  - plus the slices that became frozen by the decision after round r − 1, broadcast once more because clients do not yet
    hold their aggregated value;
  - plus leaves and estimator weights.
- **Uplink in round r:**
  - only slices that were not frozen at the start of round r, plus leaves and estimator weights;
  - FP32 (G1, G2), or INT8 plus one 4-byte scale per transmitted tensor (G3).
- Each element is one [E, I] slice of length V = 50.
- **F0 and F2** are recomputed with the same definitions (Phase 2D `accounting_v2` and the Phase 2E per-round
  accounting).
- **Consistency check:** the aggregator checks that cumulative bytes = initialization + the sum of per-round bytes.

## Recorded per round

- validation AUC and loss, and the weighted training loss;
- mean consensus and unanimous fraction;
- the fractions of nodes feature-frozen and fully frozen at the start of the round;
- newly frozen counts;
- for G2, medians of D and S over feature-frozen nodes;
- slice-level bytes down and up for each tensor;
- the INT8 error (G3).

Per run, `freeze_rounds.npz` stores each node's feature-freeze and threshold-freeze round (−1 = never).

## Metrics and reporting

- **Accuracy:**
  - pooled-OOF ROC-AUC (primary), macro-F1 at 0.5 and at the validation threshold, accuracy;
  - per seed, then mean ± SD over the 3 seeds;
  - the per-fold mean AUC as secondary;
  - ΔAUC and AUC retention against F0, paired by (partition, seed);
  - paired fold-level differences (G1−F0, G2−F0, G2−F2, G1−F2, G2−G1, and G3−G2 if run), descriptive only;
  - fold × seed runs are not treated as independent, and **no p-values** are reported.
- **Communication:** cumulative GB to the best round and to the final round, MB per round at the end of training,
  compression relative to F0, and paired GB ratios (G2/F2, G1/F2, G2/G1).
- **Freezing dynamics:**
  - the fraction of nodes feature-frozen and fully frozen at rounds 10, 20, 30, 40 and 60, at the best round and at the
    final round;
  - distributions of freeze rounds (median and IQR);
  - the fraction of nodes never feature-frozen and never fully frozen;
  - the lag between feature and threshold freezes;
  - the G1 trigger round (mean, SD, and the number of runs never triggered).
- **Runtime and memory:** seconds per job and peak RSS.
- **Figures** (`reports/figures/phase2f/`):
  - A: progressive freezing curves;
  - B: the AUC versus communication frontier, including Phase 2D C2/C3 and Phase 2E F3 for reference (not rerun);
  - C: freeze timing.

## Interpretation (bands fixed in advance; unchanged from Phase 2E)

**AUC bands:**
- |ΔAUC| ≤ 0.005: essentially preserved.
- 0.005–0.010: small tradeoff.
- More than 0.010: meaningful degradation.

**Exactly one primary conclusion will be chosen:**
1. Progressive consensus-gated freezing improves on the fixed schedule.
2. Consensus gating reproduces the fixed schedule automatically but does not improve it.
3. Global adaptive freezing is sufficient; node-wise complexity provides no useful gain.
4. Consensus is not a reliable enough signal for adaptive structural freezing.

## Engineering gate (pre-freeze; fold 0, seed 42; G1 and G2 × natural and IID; runtime and validation only)

All four runs completed with no assertion failure. The machine-generated summary is in
`results/phase2f/engineering_gate.json` and the log in `results/phase2f/bench.log`.

| Run | Wall s | Peak RSS | Rounds (best) | Rounds 1–13 identical to F0 | First round ≠ F0 | G1 trigger | Feature-frozen at end | Fully frozen at end | MB per round, first → last |
|---|---|---|---|---|---|---|---|---|---|
| G1 natural | 202 | 2.02 GB | 100 (100) | yes (val AUC, val loss, train loss bitwise) | 24 | 23 | 100% | 100% | 136.08 → 2.75 |
| G1 IID | 189 | 1.89 GB | 100 (99) | yes | 20 | 19 | 100% | 100% | 136.08 → 2.75 |
| G2 natural | 338 | 1.90 GB | 100 (99) | yes | 14 | – | 99.7% | 83.9% | 136.08 → 13.74 |
| G2 IID | 331 | 2.34 GB | 100 (98) | yes | 14 | – | 99.8% | 84.4% | 136.08 → 13.45 |

- **Pre-freeze identity holds.** Every run is bitwise identical to Phase 2B F0 up to the round before its first freeze
  takes effect. For G2 that is round 14, the earliest possible.
- **Accounting.**
  - G1's post-freeze traffic, 2.75 MB per round, equals F2's Phase 2E steady state.
  - G2 keeps sending the thresholds of nodes that are not yet threshold-frozen, plus feature logits for the few nodes
    not yet feature-frozen.
- **Freeze timing in G2.**
  - Median feature freeze is at round 13 (effective from round 14) and median threshold freeze at round 16, both the
    earliest possible.
  - About 84% of nodes become fully frozen.
- **Runtime projection.** 30 × ~196 s (G1) + 30 × ~335 s (G2) ≈ **4.4 h** for the 60 jobs. This is under the 5 h report
  threshold.
  - G2 gets no Phase 2E-style speed-up because its structural tensors stay partly trainable, so gradients are still
    computed for them.
- **Memory.** Peak RSS at most 2.45 GB (`/usr/bin/time`).
- **Disclosure.** The job runner's standard final stdout line prints the fold-0 test AUC, so it appeared in the
  benchmark log. It was not used for any decision, and no setting was changed after the pause.

These benchmark runs are development artifacts (`results/phase2f/benchmark/`). The final matrix reruns those cells. No
outer-test metric from the benchmark was examined to set or change any threshold. All gate values above were fixed in
the config before the pause (`906aef5`) and are unchanged.

## Operational note

The matrix runner gained a `--min-free-gib` option (default 8, unchanged). Free disk space on the machine fell below
8 GiB because of activity outside the project; the project uses about 0.45 GB, and Phase 2F adds about 30 MB. The final
matrix is run with `--min-free-gib 3`. This has no scientific effect.

## Excluded

- Dirichlet partitions;
- new HPO and other gate thresholds, warm-ups or consecutive-round counts;
- other freeze rounds, 64-tree variants, sparsification, error feedback;
- other optimizers or models;
- personalization, privacy, and new data.

## Amendments

None at freeze time.
