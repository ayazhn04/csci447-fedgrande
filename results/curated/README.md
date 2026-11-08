# Curated results

This folder holds small, report-ready tables built **programmatically** by `scripts/build_curated_results.py`
(`make summaries`) from the saved phase summaries of the frozen experiments (tag `experiments-final`). Nothing in it
is typed by hand.
- Regenerate with `make summaries`.
- Verify with `make check`.

## Provenance columns (every row of every CSV)

| Column | Meaning |
|---|---|
| `source_file` | The phase summary files the row was read from (`;`-separated). |
| `source_sha256` | The first 16 hex digits of each source file's SHA-256. |
| `phase` | The experiment phase, including amendments used. |
| `results_commit` | The commit, at or before `experiments-final`, that last wrote the source file. |
| `config_sha256` | The first 16 hex digits of the SHA-256 of the frozen phase config. The full hashes are in `PROVENANCE.json` and match the `config_sha256` stored in every run's `result.json`. |

## Files

| File | Phase | Contents | Use it for |
|---|---|---|---|
| `main_performance.csv` | 2B | Central, FedAvg and local GRANDE and MLP on natural/IID: pooled OOF AUC (mean, SD), per-fold AUC, macro-F1, accuracy | RQ1: federation vs centralized vs local |
| `heterogeneity.csv` | 2B (+A1, A3) | FedAvg GRANDE by partition, including Dirichlet: pooled and per-fold AUC, central gap, consensus and unanimity at round 20 and at best, update cosine, local-only consensus, overlap with the centralized structure | RQ2, RQ3 |
| `personalization.csv` | 2C (+A1) | P0–P3: within-client per-fold AUC, aggregate fold AUC, F1, log loss, Brier, ECE (mean and SD over seeds); frozen contaminated metric for reference | RQ4 |
| `compression.csv` | 2D | C0–C3: AUC, ΔAUC, retention, GB to best and final, compression ×, runtime, memory | RQ5 |
| `structure_freezing.csv` | 2E | F0–F3: AUC, ΔAUC, retention, MB per round before and after the freeze, GB, compression ×, runtime; structure share of the payload | RQ6 |
| `adaptive_freezing.csv` | 2F | F0, F2, G1, G2: AUC, ΔAUC, fold AUC, GB, trigger round, freeze-round medians, never-frozen fractions, frozen fractions at round 20 and final | RQ7 |
| `final_key_numbers.json` | all | About 80 single numbers with their source file, for prose and slides | Quoting numbers |
| `PROVENANCE.json` | all | The full SHA-256 and commit of every frozen config | Audit |

## Caveats to carry into any use

- **Pooled OOF vs per-fold.**
  - Pooled-OOF AUC is the frozen primary metric.
  - Under label skew (Dirichlet), and for two early-stopped G2 runs, score levels differ across folds and pooled AUC
    understates discrimination. Quote the per-fold column alongside it.
- **Phase 2C aggregate AUC is inflated by client identity.** Use `within_client_auc_per_fold_A1` for discrimination.
- **GB are conceptual tensor bytes**, not network measurements.
