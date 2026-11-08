# Claims and limitations

Use this page while writing the report and preparing for questions at the defense. Every safe claim cites where its
number lives:
- [RESULTS_LEDGER.md](RESULTS_LEDGER.md) for the numbers;
- [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for the reasoning.

## 1. Safe claims (directly supported)

| # | Claim | Evidence |
|---|---|---|
| S1 | Row-wise CV on SpeedDating is structurally leaky: 79.8% of row-wise test rows have their mirror row in training. Grouped CV removes this. | Phase 2A §C |
| S2 | GRANDE 1.0.0 is bitwise-equivalent to the historical 0.2.0 implementation for forward passes and training. | Phase 2A §A |
| S3 | FedAvg GRANDE retained most of centralized performance: pooled OOF AUC 0.844 (natural) / 0.845 (IID) against 0.857 centralized, about 98.5% retention. | Ledger, main performance |
| S4 | Collaboration matters: FedAvg beats local-only training by about +0.10 AUC. | Ledger, gaps |
| S5 | Natural-wave heterogeneity performed like the size-matched IID control in this dataset (IID − natural +0.0008 ± 0.0011). | Ledger, heterogeneity |
| S6 | Extreme synthetic label skew (Dirichlet α = 0.1) produced a much larger gap, mainly by miscalibrating the global model (F1 at 0.5 = 0.455; pooled AUC 0.644 against per-fold 0.827). | 2B A1/A3 |
| S7 | Split-feature consensus across clients rises quickly (about 0.98 by round 20 on natural/IID) and is lower under label skew (0.971). | Ledger, heterogeneity |
| S8 | Under label skew, a per-client intercept (P1) captured the Phase 2C benefit. Leaf or estimator-weight personalization (P2/P3) did not improve within-client discrimination (0.769/0.768 against 0.770) and calibrated worse (log loss 0.34 against 0.10). | Ledger, personalization |
| S9 | INT8 uplink quantization had negligible predictive cost in this experiment (ΔAUC +0.000/+0.001) at 1.5× less conceptual communication. | Ledger, compression |
| S10 | A 64-tree ensemble cuts communication 4.6–5.1× at −0.006 to −0.010 AUC. Combined with INT8: 7.9–8.5× at about −0.009. | Ledger, compression |
| S11 | Tree structure (split thresholds plus split feature logits) is 97.98% of GRANDE's trainable payload. | Gate 0 |
| S12 | Freezing structure after round 20 (F2) preserved performance within the predeclared band (ΔAUC −0.0018/−0.0022) while cutting cumulative communication 3.6–4.1×. | Ledger, structure freezing |
| S13 | G1 (global consensus gate) automatically rediscovered approximately the fixed round-20 schedule (trigger 20.4 ± 1.4, 30/30 runs) and matched F2 in AUC and communication. | Ledger, adaptive freezing |
| S14 | Node-wise gating (G2) did not improve on F2: about 1.21× more communication, equal per-fold AUC, and two early-stopped runs. | Ledger, adaptive freezing |
| S15 | All preprocessing used was label-free and fitted on training rows only, from aggregatable statistics. | Methods §5 |

## 2. Claims that need qualification

| # | Statement | Required qualification |
|---|---|---|
| Q1 | "Lower structural consensus goes with a larger federation gap." | **Associative, and only across heterogeneity regimes.** All runs: ρ = −0.48. Within natural and within IID the sign is positive (+0.19, +0.46). Not causal. |
| Q2 | Any communication number or compression ratio | **Conceptual tensor bytes** (parameters × bytes, 21 clients, full participation). No serialization, protocol or transport overhead; not measured network traffic. |
| Q3 | Phase 2C aggregate AUC (0.83 → 0.99) | **Affected by client identity.** Clients are label-sorted under Dirichlet, so any per-client offset ranks clients. Use within-client AUC. |
| Q4 | Pooled-OOF AUC comparisons | **Can be distorted by cross-fold calibration shifts** (Dirichlet; G2 seed-447 early-stopped runs). Quote per-fold AUC alongside. |
| Q5 | "FedAvg GRANDE performance" | **Round-budget-limited** (100 rounds, still improving slowly). No HPO for any method. |
| Q6 | "Natural heterogeneity does not hurt." | **In this dataset.** SpeedDating waves have moderate label skew (0.08–0.31); other federations may differ. |
| Q7 | "The structure is stable after round 20." | Feature identity is about 95% unchanged at round 40 and about 90% at the best round. Thresholds keep drifting slowly. The point is that further change adds almost no AUC, not that change stops. |
| Q8 | G2's ΔAUC (−0.0078 / −0.0057) | Driven by 2 of 30 runs that early-stopped after aggressive freezing. Per-fold AUC equals F2/G1. The predeclared G3 rule used the pooled number and is reported as such. |
| Q9 | Seed variability | Mean ± SD over 3 seeds on the same folds; not independent samples, so no significance tests. |
| Q10 | MLP vs GRANDE | The FedAvg MLP matched or slightly beat FedAvg GRANDE (0.850 / 0.847). GRANDE's advantage here is centralized (0.857 against 0.848), not federated. |

## 3. Claims we must NOT make

| # | Do not claim | Why |
|---|---|---|
| N1 | "Federated learning provides privacy." | No differential privacy, secure aggregation or attack evaluation was done. Parameter updates can leak information. |
| N2 | "Historical row-wise leakage inflated accuracy." | Not shown. The single-fold diagnostic gave *lower* AUC under row-wise CV (0.829 against 0.874). We showed the split is structurally leaky, not that a specific number was inflated. |
| N3 | "CGSF / consensus gating improves FedAvg" or "universally improves" anything | G1 only matched the fixed schedule; G2 was worse in communication. One dataset. |
| N4 | "Tree structure literally stops changing after round 20." | See Q7. |
| N5 | "G2 is better because it is more complex / finer-grained." | It was not better on any primary metric. |
| N6 | "Our compression ratios equal real network savings." | See Q2. |
| N7 | "Results generalize to all tabular datasets / all FL settings." | One dataset, 21 clients, full participation, cross-silo. |
| N8 | "Personalization improves discrimination under label skew." | Within-client AUC did not improve (S8). |
| N9 | "We reproduced the GRANDE paper's results." | Not attempted. The Phase 2A baseline is explicitly "not a paper reproduction". |
| N10 | "Statistically significant" | No significance tests were run, by design. |
| N11 | "Every connected leakage group is a dyad." | 4 ambiguous groups and 10 singletons are not confirmed dyads. |

## 4. Main limitations (for the report's Limitations section)

1. **One dataset** (SpeedDating) and one natural federation of 21 clients.
2. **No hyperparameter search**, by design. The 100-round FedAvg budget is binding.
3. **3 seeds on fixed folds.** Variability is descriptive.
4. **Synthetic stress test.** Dirichlet label skew is artificial. Its pooled metrics need the per-fold view.
5. **Conceptual communication accounting.** No real network measurement, no partial participation, no stragglers.
6. **No privacy mechanisms or threat model.**
7. **Fixed predeclared thresholds** in Phases 2E/2F. Robustness to other freeze rounds or gate values is untested.
8. **Phase 2C small sample.** Only 5 Dirichlet clients have both classes in test.
9. **CPU only, single machine.** Runtimes are indicative.
