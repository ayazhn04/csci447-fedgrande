# Presentation figures

The two figures regenerated for the final deck after the professor's review. Rebuild with `make slide-figures`
(`scripts/make_presentation_figures.py`). The script reads only saved results; nothing is retrained.

| Figure | Slide | Source |
|---|---|---|
| `slide19_training_curves` | 19 | `submission/artifacts/holdout_training_curves.csv`, `core_run_info.csv` (from `results/phase2b/runs/central_grande__none__f0__s42` and `fedavg_grande__natural__f0__s42`) |
| `slide21_consensus_by_round` | 21 | `reports/figures/final_style/data/fig03_consensus_by_round.csv` |

- `.png`: 200 dpi, byte-identical to the image in the deck.
- `.pdf`: vector version for the report.
- `data/*.csv`: the plotted values.

The script asserts the training-curve facts it annotates: centralized best checkpoint at epoch 51, early stop at epoch 61
(patience 10); FedAvg selected checkpoint at round 93 of the 100-round budget.
