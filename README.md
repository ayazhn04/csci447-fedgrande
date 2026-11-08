# Federated GRANDE: Heterogeneity, Calibration, and Structure-Aware Communication in Differentiable Tree Ensembles

**CSCI447 Machine Learning · Project CSCI447P016**
Team: **Ayazhan Uzbekbay**, **Nuray Amanzholova**
Base model: **GRANDE**, gradient-based decision-tree ensembles for tabular data (Marton et al.)

> **Status: the experimental stage is complete through Phase 2F. No further experiments are planned.**
> The frozen experimental state is tag `experiments-final` (commit `e38c7a9`). Everything after it is documentation and
> packaging. See [PROJECT_STATE.md](PROJECT_STATE.md).

## Overview

GRANDE learns ensembles of hard, axis-aligned decision trees by gradient descent. That makes it trainable with standard
federated averaging (FedAvg) without any special tree-aggregation protocol.

We train GRANDE across the **21 natural clients** (speed-dating events) of the **SpeedDating** dataset (OpenML 40536),
under a **leakage-aware grouped evaluation**. We compare it with centralized, local-only and MLP baselines. We then ask:
- how client heterogeneity and label skew affect it;
- whether the damage is a structure problem or a calibration problem;
- how its very high communication cost can be reduced.

The key observation is that **tree structure is 98% of what FedAvg sends, and clients agree on it early**. Structure
can therefore stop being federated once it stabilizes, and a consensus signal can pick that moment automatically.

## Final research question

The project started as "can FedAvg train a differentiable hard-tree ensemble?". It evolved into a study of four linked
issues in federated GRANDE:
- **client heterogeneity**;
- **calibration** under label skew;
- **communication efficiency**;
- **structural stabilization**, i.e. when tree structure stops needing federation.

There are seven research questions: [docs/RESEARCH_QUESTIONS.md](docs/RESEARCH_QUESTIONS.md).

## Main findings

1. **Leakage-aware evaluation matters.** SpeedDating has mirror rows (both sides of each date). Row-wise CV puts 79.8%
   of test rows' mirrors in training. All results use grouped folds over leakage groups.
2. **Federation works.**

   | Model | Pooled out-of-fold AUC |
   |---|---|
   | Centralized GRANDE | 0.857 |
   | FedAvg GRANDE | 0.844 (natural) / 0.845 (IID) |
   | Local-only GRANDE | 0.741 / 0.737 |

   Collaboration recovers about +0.10 AUC over training alone.
3. **Natural heterogeneity is benign here.** Natural waves ≈ the size-matched IID control. Extreme synthetic label skew
   (Dirichlet α = 0.1) widens the gap mainly by **miscalibrating** the model.
4. **Calibration, not personalization.** Under label skew, one intercept per client cuts log loss from 0.54 to 0.10.
   Personalizing leaves or tree weights adds no within-client discrimination.
5. **Static compression.**
   - INT8 client updates: free (ΔAUC ≈ 0, 1.5× less communication).
   - 64 trees: 4.6–5.1× less for −0.006 to −0.010 AUC.
   - Both together: about 8×.
6. **Structure-aware freezing.**
   - Freezing split features and thresholds after round 20 costs about 0.002 AUC and cuts communication 3.6–4.1×.
   - A consensus-gated global rule (G1) freezes at **round 20.4 ± 1.4** and matches that result.
   - *Consensus gating reproduces the fixed schedule automatically but does not improve it.*

Exact numbers: [docs/RESULTS_LEDGER.md](docs/RESULTS_LEDGER.md). What we can and cannot claim:
[docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md).

## Repository map

```text
README.md                 this page
PROJECT_STATE.md          authoritative status snapshot
docs/                     START_HERE, onboarding, methods, results ledger, claims, figures, report/slide handoffs
configs/                  frozen per-phase experiment configs (phase2, phase2b … phase2f)
src/fedgrande/            code: data, leakage groups, splits, fl/ (preprocess, FedAvg, compression, freezing),
                          phase*/ (runners, aggregators), plotting/ (theme + final figures)
scripts/                  safe entry points: build_curated_results, regenerate_figures, build_data_caches, validate_project
tests/                    unit tests (70)
results/                  saved per-run outputs and per-phase summaries; results/curated/ = report-ready tables
reports/                  frozen protocols (PHASE2*_PROTOCOL.md), generated phase reports, figures/
data/derived/             leakage groups, grouped splits, client partitions (raw data is downloaded, not committed)
env/                      exact environment snapshots and pip constraints
archive/checkpoint1/      Checkpoint 1 evidence (unchanged) + ERRATA
environment.yml           minimal environment (+ requirements-dev.txt)
Makefile                  safe workflow targets; expensive targets require CONFIRM=yes
```

## Quick start (no training)

```bash
# 1. clone
git clone https://github.com/ayazhn04/csci447-fedgrande.git && cd csci447-fedgrande
# 2. environment
conda env create -f environment.yml && conda activate fedgrande-dev
# 3. data + pinned GRANDE (needed only for the unit tests)
make grande && pip install --no-deps -e third_party/grande_dev_db79c860
make data
# 4. validate setup and run the tests
make validate
# 5. regenerate tables and figures from saved results (no training)
make summaries figures
make check
```

`make` on its own prints the list of targets. **No default target runs experiments.**

## Reproducing experiments

Re-running the experiment matrices takes hours of CPU and is never needed for the report. The levels of reproduction,
exact versions, seeds, commits and guarded commands are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Key artifacts

| What | Where |
|---|---|
| Curated, report-ready tables (with provenance) | [results/curated/](results/curated/README.md) |
| Exact numbers for the report and slides | [docs/RESULTS_LEDGER.md](docs/RESULTS_LEDGER.md) |
| Final-style figures (PNG + PDF) | `reports/figures/final_style/`; see [docs/FIGURE_CATALOG.md](docs/FIGURE_CATALOG.md) |
| Frozen protocols | `reports/PHASE2B_PROTOCOL.md` … `reports/PHASE2F_PROTOCOL.md` |
| Generated phase reports | `reports/phase2a_report.md` … `reports/phase2f_report.md` |
| Where to start reading | [docs/START_HERE.md](docs/START_HERE.md) |

## Status

**Experimental stage complete through Phase 2F. No further experiments are planned.**

Remaining work: the course report, the presentation, defense preparation, and final submission packaging. See
[PROJECT_STATE.md](PROJECT_STATE.md).

Third-party GRANDE code is used unmodified from pinned commits. It is cloned into `third_party/` by `make grande` and
is not part of this repository.
