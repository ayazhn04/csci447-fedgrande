# Final-style figures: manifest

## How these figures are made

- **Generator:** `scripts/regenerate_figures.py`, which calls `src/fedgrande/plotting/figures.py`.
  - Run it with `make figures`.
  - Check that nothing changed with `make check`.
- **Style:** `src/fedgrande/plotting/theme.py`, matched to `ayazhn04/benchmark_figures`. See
  [docs/FIGURE_STYLE.md](../../../docs/FIGURE_STYLE.md).
- **Formats:** each figure is a PNG (450 dpi) plus a vector PDF with embedded TrueType fonts. The exact plotted numbers
  are in `data/<figure>.csv`.
- **Data:** read only from saved, git-tracked result files of the frozen experiments (tag `experiments-final`). Nothing
  is retrained or re-evaluated.
- **Checks:** every figure asserts anchor values against the original phase summaries before it is written. Examples
  are listed in the last column of the table.

The phase-time figures in `reports/figures/phase2{b,d,e,f}/` are the original experiment artifacts and are kept
unchanged.

## Figures

| Figure | Function | Source result files | Replaces / restyles | Anchor check |
|---|---|---|---|---|
| `fig01_study_design` | `fig01_pipeline` | none (schematic) | new | – |
| `fig02_main_performance` | `fig02_main_performance` | `results/curated/main_performance.csv`, from `results/phase2b/summary/oof_summary_all.csv` | new view of the Phase 2B main table | every point equals `oof_summary_all.csv` |
| `fig03_consensus_by_round` | `fig03_consensus_rounds` | `results/phase2b/runs/fedavg_grande__*/rounds.csv` (45 runs) | `phase2b/fedavg_grande_consensus_rounds.png`, left panel | round-20 means equal `consensus_fixed_rounds_summary_all.csv` (1e-12) |
| `fig04_consensus_vs_gap` | `fig04_consensus_vs_gap` | `results/phase2b/summary/consensus_fixed_rounds_by_run_all.csv`, `fold_gaps_all.csv` | `phase2b/consensus_vs_gap.png` | gaps equal `fold_gaps_all.csv` exactly |
| `fig05_calibration_vs_personalization` | `fig05_calibration` | `results/curated/personalization.csv`, from `results/phase2c/summary/` | new (Phase 2C had only tables) | the builder asserts agreement with the frozen 2C table |
| `fig06_payload_share` | `fig06_payload_share` | `results/phase2e/gate0.json` | new (Gate 0 had only a table) | parameter total = 809,984 |
| `fig07_post_round20_drift` | `fig07_post_round20_drift` | `results/phase2e/stability_by_run.csv` | `phase2e/post_round20_drift.png`, panels 1–2 | equals `stability_summary.json` (1e-12) |
| `fig08_communication_frontier` | `fig08_frontier` | `results/curated/{structure_freezing,adaptive_freezing,compression}.csv` | merges `phase2d/frontier.png`, `phase2e/frontier.png` and `phase2f/B_frontier.png` | values copied from the curated tables, which are built from the phase main tables |
| `fig09_traffic_per_round` | `fig09_10_communication_curves` | `rounds.csv` of F0 (2B), F2 (2E), G1 and G2 (2F) | `phase2f/A_progressive_freezing.png`, left panel | per-run cumulative GB at the best round equals `phase2f/summary/runs_table.csv` (1e-9) for all 120 runs |
| `fig10_cumulative_communication` | same | same | `phase2f/D_cumulative_communication.png` | same |
| `fig11_g1_trigger_rounds` | `fig11_g1_trigger` | `results/phase2f/summary/phase2f_summary.json` | new | mean trigger round equals the summary |
| `fig12_g2_progressive_freezing` | `fig12_g2_freezing` | `results/phase2f/runs/fedavg_grande_cgsf_node__*/rounds.csv` | `phase2f/A_progressive_freezing.png`, right panel | per-run frozen-node counts equal the run table's never-frozen fractions |
| `fig13_g2_freeze_timing` | same | same | `phase2f/C_freeze_timing.png`, which used a 20% node sample; this figure uses exact counts over all nodes | same |

## Phase figures not restyled

These are catalogued as supplementary in [docs/FIGURE_CATALOG.md](../../../docs/FIGURE_CATALOG.md):
- `phase2b/consensus_fixed_rounds.png`
- `phase2b/fedavg_val_auc_rounds.png`
- `phase2b/per_wave_auc_natural.png`
- `phase2d/val_auc_vs_gb.png`
