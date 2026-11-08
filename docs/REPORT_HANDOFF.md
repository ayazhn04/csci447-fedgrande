# Report handoff (blueprint, not prose)

This is the plan for the final course report. **It is not the report.** Each section lists:
- its purpose;
- the points to make;
- where the facts live;
- figures and tables;
- what to stress and what to avoid.

Sources:
- Numbers: [RESULTS_LEDGER.md](RESULTS_LEDGER.md).
- Claim boundaries: [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md) (S = safe, Q = qualify, N = never).
- RQ write-ups: [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md).

**Working title:** *Federated GRANDE: Heterogeneity, Calibration, and Structure-Aware Communication in Differentiable
Tree Ensembles.*

**Suggested length:** sections 6–11 carry the results. Keep 2C (personalization) and G2 (node-wise gating) as
supporting findings, not as headlines.

---

## 1. Introduction

- **Purpose.** Motivate federated tabular learning with tree ensembles and state the questions.
- **Points to make.**
  - GBDTs dominate tabular data but are hard to federate.
  - GRANDE learns hard axis-aligned trees by gradient descent, so it can in principle be trained with FedAvg.
  - SpeedDating has a natural federation: 21 events (waves).
  - Contributions:
    1. a leakage-aware evaluation;
    2. FedAvg GRANDE vs centralized, local and MLP baselines;
    3. a heterogeneity analysis, including a label-skew stress test;
    4. a calibration diagnosis;
    5. the finding that tree structure is 98% of the communication and stabilizes early, which enables freezing;
    6. consensus-gated freezing that rediscovers the fixed schedule.
- **Sources.** RESEARCH_QUESTIONS (overview), PROJECT_TIMELINE.
- **Figures/tables.** fig01 (could also go in §5).
- **Emphasize.** The progression problem → federation → heterogeneity → communication → structure.
- **Avoid.** Privacy claims (N1); "first to …" claims that have not been checked against the literature.

## 2. Related Work

- **Purpose.** Position the work.
- **Points to make.**
  - Differentiable and soft trees, and GRANDE (Marton et al.).
  - Federated learning and FedAvg (McMahan et al.).
  - Federated tree ensembles (federated GBDT / XGBoost variants).
  - Non-IID FL and label skew (Dirichlet partitions are standard).
  - Personalization vs calibration in FL.
  - Communication compression (quantization, model size).
  - Partial or layer freezing in FL.
- **Sources.** Literature still needs to be collected; the repository contains none. `archive/checkpoint1/` has the
  paper-to-code notes, with caveats in its ERRATA.
- **Emphasize.** What is specific here: hard-tree structure that can be federated and frozen.
- **Avoid.** Citing the GRANDE paper's numbers as if we reproduced them (N9).

## 3. Dataset and Leakage-Aware Evaluation

- **Purpose.** Establish trust in the evaluation.
- **Points to make.**
  - SpeedDating: 8,378 rows, 16.5% positive.
  - Excluded features: `decision`, `decision_o` (label proxies) and `wave` (client id).
  - Mirror rows; connected-component leakage groups with the correct terminology (4,175 dyads + 4 ambiguous groups
    + 10 singletons).
  - Row-wise CV puts 79.8% of test rows' mirrors in training.
  - Grouped 5-fold CV with grouped inner validation; 3 seeds; validation-only selection and thresholds.
  - Pooled OOF vs per-fold metrics.
- **Sources.** METHODS_REFERENCE §1–3, Phase 2A report §B–C.
- **Figures/tables.**
  - A small table of group kinds.
  - Optionally a wave table (size and positive rate; Phase 2A §D).
- **Emphasize.** S1, S2, S15.
- **Avoid.** N2: do not claim that the historical leakage inflated a specific number. N11.

## 4. Federated GRANDE Method

- **Purpose.** Define what is averaged and how.
- **Points to make.**
  - The GRANDE parameterization: split feature logits, thresholds, leaves, estimator weights; the V = 50 feature subset
    per tree.
  - FedAvg with local Adam: fresh optimizer every round, 1 local epoch, weighted by rows, full participation.
  - Federated-safe, label-free preprocessing from sufficient statistics.
  - Class weighting.
  - Structural consensus as a diagnostic.
- **Sources.** METHODS_REFERENCE §5–9.
- **Figures/tables.** A parameter table (shapes and payload share), linking forward to §9–10. fig06 could go here.
- **Emphasize.** Buffers are shared and never averaged, so feature identities are comparable across clients.
- **Avoid.** Calling it FedAdam.

## 5. Experimental Setup

- **Purpose.** Make the evaluation reproducible from the text.
- **Points to make.**
  - The three partitions: natural, IID control, Dirichlet stress.
  - The methods and baselines (EXPERIMENT_MAP).
  - Budgets: 100 rounds/epochs, patience 10, warm-up 10.
  - Seeds; metrics; no HPO; no p-values.
  - Communication accounting definition.
  - Hardware and software; protocol freezing before each phase.
- **Sources.** METHODS_REFERENCE, EXPERIMENT_MAP, REPRODUCIBILITY.
- **Figures/tables.** fig01 (if not used in §1); an experiment table condensed from EXPERIMENT_MAP.
- **Emphasize.** Predeclared protocols and interpretation bands.
- **Avoid.** Implying tuning on test data. None was done.

## 6. Main Federated Results

- **Purpose.** Answer RQ1.
- **Points to make.**
  - FedAvg GRANDE 0.844/0.845 against central 0.857 and local 0.741/0.737.
  - FedAvg MLP 0.850/0.847; central MLP 0.848.
  - Gaps: central − FedAvg +0.013/+0.012; FedAvg − local +0.10.
- **Sources.** Ledger "Main performance"; `results/curated/main_performance.csv`.
- **Figures/tables.** **fig02**, plus the main table with pooled and per-fold AUC and F1 at the validation threshold.
- **Emphasize.** S3, S4.
- **Avoid.** Implying GRANDE beats the MLP under federation (Q10). Mention the round budget (Q5).

## 7. Heterogeneity and Structural Analysis

- **Purpose.** Answer RQ2 and RQ3.
- **Points to make.**
  - Natural ≈ IID.
  - Dirichlet: pooled 0.644 vs per-fold 0.827; the F1 collapse means calibration failure.
  - Consensus about 0.94 → 0.98 by round 20, lower under label skew (0.971).
  - Local-only consensus about 0.75.
  - The consensus–gap association holds across regimes, not within them.
- **Sources.** Ledger "Heterogeneity"; RESEARCH_QUESTIONS RQ2/RQ3.
- **Figures/tables.** **fig03**; fig04 (supporting); optional per-wave AUC (supplementary).
- **Emphasize.** S5, S6, S7.
- **Avoid.** Causal language about consensus (Q1). Do not present the Dirichlet pooled AUC without the per-fold number
  (Q4).

## 8. Calibration and Personalization

- **Purpose.** Answer RQ4, as a supporting finding.
- **Points to make.**
  - The P0–P3 design.
  - The within-client AUC is unchanged by P2/P3.
  - P1 log loss 0.10 vs P0 0.54; P2/P3 are worse calibrated.
  - The aggregate AUC jump to 0.99 is a client-identity effect.
- **Sources.** Ledger "Personalization"; Phase 2C report (amendment A1).
- **Figures/tables.** **fig05**.
- **Emphasize.** S8: recalibration explains the benefit.
- **Avoid.** N8. Q3: do not quote 0.99 as discrimination.

## 9. Communication Efficiency

- **Purpose.** Answer RQ5: the static options.
- **Points to make.**
  - FedAvg GRANDE sends 136 MB per round, about 10.7–12.5 GB to the best round.
  - INT8 uplink: free at 1.5×. 64 trees: −0.006/−0.010 at 4.6–5.1×. Both: about −0.009 at 7.9–8.5×.
- **Sources.** Ledger "Static compression".
- **Figures/tables.** fig08 (C points), or a compression table.
- **Emphasize.** S9, S10; complementarity.
- **Avoid.** N6. State the accounting definition (Q2).

## 10. Structure-Aware Freezing

- **Purpose.** Answer RQ6: the central communication finding.
- **Points to make.**
  - Structure is 98% of the payload (Gate 0).
  - Freeze after round 20 (fixed a priori from 2B consensus).
  - F1 ≈ F0; **F2 ΔAUC −0.0018/−0.0022 at 3.0 GB (3.6–4.1×)**; F3 ≈ F2.
  - Drift diagnostics: feature identities mostly settle; thresholds drift slowly; the extra change buys almost no AUC.
- **Sources.** Ledger "Fixed structural freezing"; Phase 2E report.
- **Figures/tables.** **fig08**, fig06 and fig07 (supporting), fig09/fig10 (supporting).
- **Emphasize.** S11, S12.
- **Avoid.** N4 (structure does not literally stop changing; Q7).

## 11. Adaptive Consensus-Gated Freezing

- **Purpose.** Answer RQ7: can the freeze round be chosen automatically?
- **Points to make.**
  - The predeclared G1 and G2 gates.
  - G1 triggers at 20.4 ± 1.4 and matches F2.
  - G2 freezes features at round 13 and thresholds at 16, but 17% of thresholds never freeze, giving 1.21× F2's GB. Two
    early-stopped runs lower its pooled AUC; per-fold AUC equals F2.
  - G3 was not eligible.
  - **Conclusion:** consensus gating reproduces the fixed schedule automatically but does not improve it.
- **Sources.** Ledger "Adaptive freezing"; Phase 2F protocol (D1, A1) and report.
- **Figures/tables.** **fig11**, fig08, fig12 (supporting), fig13 (supplementary).
- **Emphasize.** S13, S14. The value of G1 is that it removes a hand-chosen hyperparameter and matches it.
- **Avoid.** N3, N5. Q8: explain G2's pooled number honestly.

## 12. Discussion

- **Purpose.** Synthesize.
- **Points to make.**
  - Under natural heterogeneity, GRANDE federates well.
  - Severe label skew is a calibration problem more than a structure problem.
  - Communication is dominated by structure. Structure stabilizes early, so most of it does not need to be federated
    for long.
  - Consensus is a usable *timing* signal at the global level, not a reliable per-node signal with these gates.
  - Practical recipe: FedAvg GRANDE + G1 (or a freeze at about round 20) + INT8 for the remaining tensors (F3 ≈ F2).
- **Sources.** RESEARCH_QUESTIONS (interpretations).
- **Figures/tables.** fig08 as the summary.
- **Emphasize.** The coherent story from observation (consensus) to mechanism (payload) to method (freeze).
- **Avoid.** Overgeneralizing (N7).

## 13. Limitations

- **Purpose.** Be honest.
- **Points to make.** CLAIMS_AND_LIMITATIONS §4, items 1–9.
- **Sources.** That page.
- **Emphasize.** One dataset; no HPO; conceptual bytes; no privacy.
- **Avoid.** Burying the G2 early-stopping observation. Mention it.

## 14. Conclusion

- **Purpose.** Restate the answers in 4–6 sentences.
- **Points to make.**
  - FedAvg GRANDE works (about 98.5% of centralized AUC).
  - Wave heterogeneity is benign.
  - Label skew miscalibrates, and an intercept fixes it.
  - Freezing structure after it stabilizes cuts communication about 4× at no meaningful cost.
  - A consensus gate finds that point automatically.
- **Sources.** README "Main findings".
- **Emphasize.** The final experimental conclusion, worded as written.
- **Avoid.** New claims.

## Appendix candidates

- Per-wave AUC.
- Phase 2A equivalence (E0a/E0b).
- Stock preprocessing check.
- Within-client AUC tables.
- G2 freeze timing (fig13).
- The full protocol amendments list.
