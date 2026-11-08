# Presentation handoff (storyline, not slides)

This is a likely 12-slide talk. Each slide lists only its purpose, core message, recommended visual and source.

The storyline:

> problem → federation → heterogeneity → communication bottleneck → structural observation → fixed freeze → adaptive
> global consensus rule.

Phase 2C (personalization) and G2 (node-wise gating) are supporting findings: each gets one line or a backup slide, not
a full slide. Numbers come from [RESULTS_LEDGER.md](RESULTS_LEDGER.md); check wording against
[CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md).

| # | Slide purpose | Core message | Recommended figure/table | Source artifact |
|---|---|---|---|---|
| 1 | Title | Federated GRANDE: heterogeneity, calibration and structure-aware communication | – | README |
| 2 | Problem | Tabular data is often siloed; tree ensembles are hard to federate; GRANDE learns hard trees by gradient descent, so FedAvg is possible | simple schematic (GRANDE tree + clients) | METHODS_REFERENCE §6 |
| 3 | Data and evaluation | SpeedDating has 21 natural clients (waves); row-wise CV leaks mirror rows (79.8%), so we evaluate with grouped folds | fig01 (study design) + one leakage number | Phase 2A report §C |
| 4 | Federation works | FedAvg GRANDE 0.844 vs central 0.857 vs local 0.741 | **fig02** | `results/curated/main_performance.csv` |
| 5 | Heterogeneity | Natural ≈ IID; extreme label skew hurts, mostly through calibration | fig03 or a 3-row table (natural / IID / Dirichlet, pooled vs per-fold) | `results/curated/heterogeneity.csv` |
| 6 | Calibration (supporting) | Under label skew, one intercept per client beats personalizing trees | **fig05** (one slide, or a backup) | `results/curated/personalization.csv` |
| 7 | Communication bottleneck | 136 MB per round, about 11–12 GB to convergence; INT8 is free (1.5×), 64 trees costs a little AUC | fig08 restricted to C points, or a small table | `results/curated/compression.csv` |
| 8 | Structural observation | Structure is 98% of the payload, and mean client consensus on split features reaches 0.98 by round 20 | **fig06** + fig03 (round-20 marker) | `results/phase2e/gate0.json`; fig03 |
| 9 | Fixed freeze | Freezing structure after round 20 costs ≈0.002 AUC (−0.0018 / −0.0022) and cuts communication 3.6–4.1× | **fig08** (F points highlighted) | `results/curated/structure_freezing.csv` |
| 10 | Adaptive global rule | A consensus gate finds the freeze point automatically: round 20.4 ± 1.4, matching F2 | **fig11** + the G1 point on fig08 | `results/curated/adaptive_freezing.csv` |
| 11 | What did not help (supporting) | Node-wise gating (G2): more bytes (1.21×), less stable; personalization: no within-client gain | one table or bullet list; fig12 as a backup | Phase 2F report; ledger |
| 12 | Takeaways and limitations | Recipe: FedAvg + structure freeze (consensus-timed) + INT8. Limits: one dataset, conceptual bytes, no privacy | text | CLAIMS_AND_LIMITATIONS |

**Backup slides:** the G2 early-stopping explanation (Phase 2F protocol A1); per-wave AUC; drift diagnostics (fig07);
per-round traffic (fig09); the GRANDE version-equivalence check.

**Likely questions to prepare** (answers are in CLAIMS_AND_LIMITATIONS):
- Is this private? (N1)
- Did leakage inflate the earlier results? (N2)
- Why is the MLP as good as GRANDE? (Q10)
- Are the bytes real network bytes? (Q2)
- Why not tune the freeze round? (protocol: fixed a priori)
- Why did G2 lose AUC? (Q8)
