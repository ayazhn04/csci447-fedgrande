# Figure catalog

This page inventories every scientific figure.

**Status labels:**
- **MAIN:** goes in the report and slides.
- **SUPPORTING:** report body or backup slide.
- **SUPPLEMENTARY:** appendix.
- **DO NOT USE:** superseded or misleading.

Final-style versions (PNG + PDF + plotted data) are in `reports/figures/final_style/`. Their provenance is in
[`MANIFEST.md`](../reports/figures/final_style/MANIFEST.md). The phase-time originals stay in `reports/figures/phase2*/`.

## Final-style figures (use these)

| File (`reports/figures/final_style/`) | RQ | What it shows | Status | Caveat |
|---|---|---|---|---|
| `fig01_study_design` | all | Pipeline: data → leakage groups → 21 clients → FedAvg GRANDE → extensions | **MAIN** | Schematic, no data. |
| `fig02_main_performance` | RQ1 | Pooled OOF AUC (± SD) for central, FedAvg and local GRANDE and the MLPs | **MAIN** | x-axis starts at 0.70. |
| `fig03_consensus_by_round` | RQ2/3 | Mean node consensus by round for natural, IID and Dirichlet | **MAIN** | Mean over runs still active; Dirichlet runs stop early (34 rounds on average). |
| `fig04_consensus_vs_gap` | RQ3 | Per-run round-20 consensus vs centralized − FedAvg gap | SUPPORTING | Association between regimes only; within partitions the sign flips. |
| `fig05_calibration_vs_personalization` | RQ4 | P0–P3: within-client AUC (unchanged) and log loss (P1 best) | **MAIN** | Dirichlet stress test; 48 client × fold cells. |
| `fig06_payload_share` | RQ6 | Share of the payload per GRANDE tensor (structure 98%) | **MAIN** (slides) / SUPPORTING (report) | Analytic. |
| `fig07_post_round20_drift` | RQ6 | F0 after round 20: feature-identity stability and threshold drift | SUPPORTING | Diagnostic reruns, seed 42 only. |
| `fig08_communication_frontier` | RQ5–7 | AUC vs cumulative GB to best: F0–F3, G1, G2, with C1–C3 faded | **MAIN** | Log x-axis; conceptual bytes. |
| `fig09_traffic_per_round` | RQ6/7 | MB per round for F0, F2, G1, G2 | SUPPORTING | Mean over runs still active. |
| `fig10_cumulative_communication` | RQ6/7 | Cumulative GB by round for F0, F2, G1, G2 | SUPPORTING | Stopped runs carried forward. |
| `fig11_g1_trigger_rounds` | RQ7 | Histogram of G1 trigger rounds against the fixed round 20 | **MAIN** | – |
| `fig12_g2_progressive_freezing` | RQ7 | Fraction of nodes feature- and threshold-frozen over rounds (G2) | SUPPORTING | – |
| `fig13_g2_freeze_timing` | RQ7 | Freeze decisions per round, all G2 runs | SUPPLEMENTARY | Symlog y-axis. |

## Presentation figures

Two figures in the final deck were regenerated after the professor's review. They use the deck's chart style, not the
final-style theme. Files are in `reports/figures/presentation/` (PNG = the exact image in the deck, PDF = vector, plotted
data in `data/`). Rebuild them with `make slide-figures` (`scripts/make_presentation_figures.py`); the script reads only
saved results.

| File (`reports/figures/presentation/`) | Slide | What it shows | Caveat |
|---|---|---|---|
| `slide19_training_curves` | 19 | Hold-out loss and validation ROC-AUC, centralized vs FedAvg GRANDE (fold 0, seed 42) | Centralized best checkpoint is epoch 51; training stops at epoch 61 (patience 10). FedAvg runs to the 100-round budget; the selected checkpoint is round 93. Accuracy was not logged per round. |
| `slide21_consensus_by_round` | 21 | Mean node consensus per round for natural, IID and Dirichlet(0.1) | Mean over runs still active (same data as fig03). Natural and IID reach round 100 and nearly overlap; the last Dirichlet run ends at round 53. |

## Phase-time originals

| File | RQ | What it shows | Status | Caveat |
|---|---|---|---|---|
| `phase2b/fedavg_grande_consensus_rounds.png` | RQ3 | Consensus, agreement, validation AUC by round (3 panels) | SUPPLEMENTARY (superseded by fig03) | Original style. |
| `phase2b/consensus_fixed_rounds.png` | RQ3 | Consensus, unanimity, cosine at rounds 1/5/10/20 | SUPPLEMENTARY | Useful backup for the update-cosine claim. |
| `phase2b/consensus_vs_gap.png` | RQ3 | Same data as fig04 | DO NOT USE (superseded by fig04) | – |
| `phase2b/fedavg_val_auc_rounds.png` | RQ1 | Validation AUC by round for FedAvg GRANDE and MLP | SUPPLEMENTARY | Survivor bias in the tails (MLP lines jump as runs stop). |
| `phase2b/per_wave_auc_natural.png` | RQ2 | Per-wave AUC for all methods | SUPPLEMENTARY | Small waves (6, 18, 20) are unstable; dense. |
| `phase2d/frontier.png` | RQ5 | C0–C3 frontier | DO NOT USE (merged into fig08) | – |
| `phase2d/val_auc_vs_gb.png` | RQ5 | Validation AUC against cumulative GB | SUPPLEMENTARY | Mean over runs still active. |
| `phase2e/frontier.png` | RQ6 | F0–F3 frontier | DO NOT USE (merged into fig08) | – |
| `phase2e/post_round20_drift.png` | RQ6 | 3-panel drift diagnostic | SUPPLEMENTARY (fig07 restyles panels 1–2) | Third panel (relative norm drift) only here. |
| `phase2f/A_progressive_freezing.png` | RQ7 | Traffic per round + G2 freezing | DO NOT USE (split into fig09 + fig12) | – |
| `phase2f/B_frontier.png` | RQ7 | 2F frontier | DO NOT USE (merged into fig08) | – |
| `phase2f/C_freeze_timing.png` | RQ7 | G2 freeze timing (20% node sample) | DO NOT USE (fig13 uses exact counts) | – |
| `phase2f/D_cumulative_communication.png` | RQ7 | Cumulative GB | DO NOT USE (superseded by fig10) | – |

## Recommended shortlist (6 figures)

1. **`fig01_study_design`**: the design in one picture, including the leakage-aware split.
2. **`fig02_main_performance`**: federation works (RQ1).
3. **`fig03_consensus_by_round`**: heterogeneity and the structural observation (RQ2/3). It motivates the freeze.
4. **`fig05_calibration_vs_personalization`**: the label-skew diagnosis (RQ4).
5. **`fig08_communication_frontier`**: compression, fixed freeze and adaptive freeze on one axis (RQ5–7).
6. **`fig11_g1_trigger_rounds`**: consensus rediscovers round about 20 (RQ7).

For slides, `fig06_payload_share` is the best single image for *why* structure freezing matters.
