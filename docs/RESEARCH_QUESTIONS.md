# Research questions

The project is organized as seven research questions (RQs). Each RQ maps almost directly onto a Results/Discussion
subsection of the report.

**Conventions:**
- Numbers are pooled out-of-fold (OOF) ROC-AUC, mean over 3 seeds, unless stated otherwise. Exact values and SDs are in
  [RESULTS_LEDGER.md](RESULTS_LEDGER.md).
- What may and may not be claimed is in [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md).

```text
RQ1  Does FedAvg GRANDE work?                         ─┐
RQ2  How much does heterogeneity hurt?                 ├─ core federated study (Phase 2B)
RQ3  Does structural disagreement explain it?         ─┘
RQ4  Personalization vs calibration under label skew    ── mechanistic follow-up (Phase 2C)
RQ5  Static communication compression                  ─┐
RQ6  Freeze structure once it is stable                ├─ communication extensions (Phases 2D, 2E, 2F)
RQ7  Let consensus decide when to freeze               ─┘
```

---

## RQ1: Can GRANDE be effectively trained with FedAvg across heterogeneous clients?

- **Hypothesis.** GRANDE's hard trees are trained by gradient descent through differentiable splits, so FedAvg (averaging
  parameters) should work, with some loss against centralized training.
- **Experiment.** Phase 2B core:
  - centralized, local-only and FedAvg GRANDE, plus centralized and FedAvg MLP;
  - natural and IID partitions, 5 grouped folds × 3 seeds;
  - matched preprocessing, architecture and optimizer.
- **Result.**

  | Model | Pooled OOF AUC |
  |---|---|
  | Centralized GRANDE | **0.857** |
  | FedAvg GRANDE | **0.844** (natural) / **0.845** (IID) |
  | Local-only GRANDE | **0.741** / **0.737** |
  | FedAvg MLP | 0.850 / 0.847 |
  | Centralized MLP | 0.848 |

  The centralized − FedAvg gap is +0.013 (natural) and +0.012 (IID).
- **Interpretation.**
  - FedAvg retains about 98.5% of centralized AUC and gains about +0.10 over training alone.
  - GRANDE is FedAvg-compatible, but not better than a small MLP in this setting.
- **Limitation.**
  - One dataset.
  - FedAvg runs are round-budget-limited (100 rounds; validation still improving slowly).
  - No hyperparameter search was done for any method.

## RQ2: How does natural client heterogeneity compare with a size-matched IID control and extreme label skew?

- **Hypothesis.** Real wave heterogeneity (19× size spread, positive rate 0.08–0.31, feature shift) degrades FedAvg
  relative to IID clients of the same sizes. Extreme label skew degrades it much more.
- **Experiment.**
  - Natural against size-matched IID (same sizes, mixed waves, equalized positive rate).
  - Dirichlet α = 0.1 label skew as a stress test (2B amendment A1).
- **Result.**
  - FedAvg GRANDE natural 0.844 ≈ IID 0.845 (IID − natural +0.0008 ± 0.0011).
  - Dirichlet: pooled OOF 0.644, but per-fold AUC 0.827. The pooled number collapses because federated models become
    badly miscalibrated under label skew: F1 at 0.5 is 0.455, and validation thresholds fall to 0.28–0.43. Score scales
    then differ across folds.
- **Interpretation.**
  - The natural wave heterogeneity of SpeedDating is **not harmful enough to matter** for FedAvg GRANDE.
  - Extreme label skew is, and it damages calibration more than ranking.
- **Limitation.**
  - The natural partition has fairly mild label skew.
  - Dirichlet is synthetic.
  - Pooled-OOF AUC is distorted under label skew, so the per-fold view (2B amendment A3) is needed.

## RQ3: Does structural disagreement explain degradation under severe heterogeneity?

- **Hypothesis.** Heterogeneous clients pick incompatible split features. Lower consensus should go with a larger
  centralized − FedAvg gap.
- **Experiment.** Per-round node consensus, unanimity, global agreement and update cosine for every FedAvg GRANDE run,
  plus Spearman correlation with the per-fold gap (descriptive).
- **Result.**
  - Consensus rises from about 0.94 (round 1) to **0.982 (natural) / 0.983 (IID)** at round 20, but only to **0.971**
    under Dirichlet.
  - Across all 45 runs, round-20 consensus vs gap: Spearman ρ = −0.48.
  - **Within** partitions the relation is weak or positive: natural +0.19, IID +0.46, Dirichlet −0.28.
  - Local-only models agree far less (final consensus about 0.75).
- **Interpretation.**
  - Label skew lowers structural agreement *and* widens the gap.
  - But consensus does not explain gap variation within a heterogeneity regime. The evidence is an association between
    regimes, not a mechanism.
- **Limitation.**
  - Correlational; 15 runs per partition.
  - The Dirichlet gap is dominated by calibration (RQ4), not structure.

## RQ4: Is extreme label-skew degradation better explained by tree personalization or by client calibration?

- **Hypothesis A (personalization).** Per-client leaves (and estimator weights) on the shared structure improve
  within-client discrimination.
- **Hypothesis B (calibration).** One intercept per client recovers the benefit.
- **Experiment.** Phase 2C, Dirichlet only:
  - P0 global; P1 + client intercept; P2 + personalized leaves; P3 + leaves and estimator weights;
  - on the 15 bitwise-verified Phase 2B checkpoints.
- **Result.**
  - Within-client AUC per (client, fold) cell: P0 = P1 = **0.770**, P2 **0.769**, P3 **0.768**.
  - Test log loss: P0 **0.542**, P1 **0.099**, P2 0.344, P3 0.339.
  - Aggregate per-fold AUC jumps from 0.83 to 0.99 for every P1–P3, but that is the client-identity effect described
    under limitations.
- **Interpretation.** **Recalibration explains the benefit.** Tree-output personalization does not improve
  discrimination within clients and is worse calibrated than P1.
- **Limitation.**
  - Only 5 of 21 Dirichlet clients have both classes in test (48 client × fold cells), so the within-client evidence is
    small.
  - The aggregate AUC is inflated by client identity, and the frozen pooled within-client metric was contaminated by
    cross-fold offsets (2C amendment A1).

## RQ5: Can GRANDE's communication cost be reduced through smaller ensembles or INT8 client-update compression?

- **Hypothesis.** The 256-tree ensemble is redundant, and client updates tolerate 8-bit quantization.
- **Experiment.** Phase 2D: C0 (256 trees, FP32), C1 (64 trees), C2 (INT8 uplink), C3 (64 trees + INT8); natural and
  IID, 5 folds × 3 seeds.
- **Result** (natural / IID):

  | | ΔAUC vs C0 | Communication reduction to best round |
  |---|---|---|
  | C2 (INT8) | +0.000 / +0.001 | 1.50× / 1.53× |
  | C1 (64 trees) | −0.010 / −0.006 | 5.1× / 4.6× |
  | C3 (both) | −0.009 / −0.009 | 8.5× / 7.9× |

- **Interpretation.**
  - INT8 uplink compression is essentially free.
  - Shrinking the ensemble trades a small AUC loss for large savings.
  - The two are complementary.
- **Limitation.**
  - Bytes are conceptual tensor bytes.
  - INT8 is applied to the uplink only.
  - 64-tree runs have higher seed variance.

## RQ6: Once tree structure becomes stable, does it still need to be federated?

- **Hypothesis.** After consensus reaches about 0.98 (round 20), split features and thresholds can be frozen. Then only
  the 2% of parameters in leaves and estimator weights need to be federated.
- **Experiment.**
  - Phase 2E Gate 0 (structure share of the payload).
  - F1 (freeze feature logits), F2 (freeze features and thresholds), F3 (F2 + INT8), all after round 20.
  - Drift diagnostics on F0 reruns.
- **Result.**
  - Structure is **97.98%** of the payload.
  - F1 ΔAUC −0.0002 / −0.0009. **F2 −0.0018 / −0.0022**, using **3.01 GB** to best against 10.7–12.5 GB (3.6–4.1× less).
  - F3 ≈ F2 at 2.93 GB.
  - In unfrozen F0, 95% of nodes keep their round-20 feature at round 40, and about 90% at the best round. Thresholds
    keep drifting slowly (about 0.07–0.18 SD).
- **Interpretation.**
  - **Full tree structure stabilizes early enough to be frozen.** Federating it after round 20 buys at most about 0.002
    AUC ("essentially preserved" band).
  - Structure does *not* stop changing entirely. Continued changes simply contribute little.
- **Limitation.**
  - The freeze round was chosen from Phase 2B observation, not tuned, but specific to this dataset.
  - Other rounds were not tested.

## RQ7: Can structural consensus automatically decide when tree structure should stop being federated?

- **Hypothesis.** A consensus gate can find the freeze point without a hand-chosen round. Node-wise gating could freeze
  more and earlier.
- **Experiment.** Phase 2F, with predeclared gates:
  - **G1:** global freeze when mean consensus ≥ 0.98 and unanimity ≥ 0.85 for 3 rounds.
  - **G2:** node-wise feature gate C_j ≥ 0.95, then threshold gate D ≤ 0.01 and S ≤ 0.05, each for 3 rounds.
  - Conditional **G3**.
  - F0 and F2 reused.
- **Result.**
  - **G1 triggers at round 20.4 ± 1.4** (18–23; all 30 runs triggered). It matches F2: ΔAUC −0.0015 / −0.0021, GB to
    best 3.16 / 2.96 against F2's 3.01.
  - **G2:**
    - Freezes features at the earliest allowed round (median 13) and thresholds at 16.
    - 17% of thresholds never pass the gate.
    - Uses **3.63 / 3.65 GB** (1.21× F2).
    - Pooled ΔAUC −0.0078 / −0.0057, driven by 2 early-stopped seed-447 runs whose score level shifted. Its per-fold AUC
      equals F2/G1.
  - G3 was not eligible.
- **Interpretation (primary conclusion, chosen from four predeclared options).** *Consensus gating reproduces the fixed
  schedule automatically but does not improve it.* Node-wise gating adds cost and instability without benefit.
- **Limitation.**
  - Gate thresholds were not tuned. A stricter G2 might behave differently, but that was explicitly excluded.
  - Pooled OOF is sensitive to score-level shifts across folds.
