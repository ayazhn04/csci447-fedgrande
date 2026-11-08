# Reproducibility

There are four levels of reproduction. Levels 1–3 never train a model. **The report and slides need only Levels 1–2.**

| Level | What you get | Needs | Time |
|---|---|---|---|
| 1. Inspect | Read every number and figure | a clone | 0 |
| 2. Regenerate | Rebuild curated tables and final figures from saved results; check that nothing changed | Python environment | about 1 min |
| 3. Validate | Repository checks plus the full unit-test suite | environment + data + pinned GRANDE | about 2 min |
| 4. Re-run experiments | Recompute the matrices of Phases 2B–2F | everything above + hours of CPU | about 6 h (2B), 4 h (2D), 7.4 h (2E), 4 h (2F) |

## Software

| Item | Value |
|---|---|
| Python | 3.12.14 (conda-forge), macOS arm64 (Apple silicon) |
| Core packages | torch 2.7.1, numpy 2.2.6, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2, matplotlib 3.11.2, category-encoders 2.9.0, openml 0.15.1 |
| GRANDE (experiments) | 1.0.0, commit `db79c86060f06c15f46c3bfb9207dc30c3cb91ae`, used unmodified from a git worktree |
| GRANDE (historical reference) | 0.2.0, commit `07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb` (environment `fedgrande-ref`; Phase 1/2A only) |
| Dataset | OpenML **40536** (SpeedDating v1), ARFF MD5 `0c5590a0c81750687ee1b0ee68168a5d` |
| Model seeds | 42, 447, 2026; partition seed 7; outer split seed 42 (frozen in `data/derived/phase2/splits_outer_v1.csv`) |
| Threads / precision | 6 CPU threads, float32, matmul precision "highest" |

## Environment files

| File | What it is | When to use it |
|---|---|---|
| `environment.yml` + `requirements-dev.txt` | **Minimal specification**: Python and the direct dependencies of this code, pinned to the verified versions | Creating a working environment |
| `env/snapshots/fedgrande-dev.conda-explicit.txt` | **Exact historical snapshot**: the conda package URLs of the environment that ran every experiment | Bit-level forensic re-creation (macOS arm64 only) |
| `env/snapshots/fedgrande-dev.pip-freeze.txt`, `.conda-export.yml`, `.json` | Exact pip list; conda export; importlib metadata, commit and platform | Auditing versions |
| `env/snapshots/fedgrande-ref.*` | Same for the historical-reference environment | Phase 1/2A only |
| `env/dev_shared_constraints.txt`, `env/ref_historical_constraints.txt` | The pip constraints used when the environments were built | Rebuilding exactly as in Phase 1 |

The minimal specification gives the same versions of everything this code imports. The exact snapshot also pins every
transitive package and build string.

## Level 1: inspect (no setup)

- Numbers: [RESULTS_LEDGER.md](RESULTS_LEDGER.md) and `results/curated/`.
- Full phase reports: `reports/phase2{a..f}_report.md`.
- Protocols: `reports/PHASE2{B..F}_PROTOCOL.md`.
- Figures: `reports/figures/final_style/` (see [FIGURE_CATALOG.md](FIGURE_CATALOG.md)).
- Per-run outputs: `results/phase2*/runs/<job>/` (`result.json`, `rounds.csv`, `predictions.csv`).

## Level 2: regenerate tables and figures (no training)

```bash
conda env create -f environment.yml
conda activate fedgrande-dev
make summaries    # results/curated/* and the generated tables in docs/RESULTS_LEDGER.md
make figures      # reports/figures/final_style/*.png|pdf + data/*.csv
make check        # fails if the tables or figure data differ from the saved results
```

These read only the git-tracked result files. They need neither the dataset nor GRANDE.

## Level 3: validate (no training)

```bash
make grande                                             # clone the pinned GRANDE commits into third_party/
pip install --no-deps -e third_party/grande_dev_db79c860
make data                                               # download the ARFF (MD5-checked) and rebuild the caches
make validate                                           # repository checks + 70 unit tests
```

- `make data` downloads from OpenML, then rebuilds the 5 gitignored preprocessing caches
  (`data/derived/phase2b/prep_C2_outer*.npz`). It verifies them against the recorded array digests of the caches used
  in the experiments (`data/derived/phase2b/cache_digests.json`). It never rewrites tracked files.
- `make validate` checks:
  - the `experiments-final` tag;
  - the key artifacts;
  - that curated tables and figure data are unchanged;
  - that the safe tooling imports no training code;
  - that the GRANDE worktrees are at their pinned commits with no changes;
  - package versions against the snapshot.

  It then runs `python -m unittest discover -s tests`.
- Expected: 70 tests, `OK`. Two Phase 2C tests are skipped when the gitignored `.pt` checkpoints are absent.

## Level 4: re-run experiments (expensive, optional)

**Not needed for the report.** Every runner is resumable and skips any job that already has a `result.json`. On a full
checkout these commands therefore do nothing unless you delete or move results first. Run them in a separate copy of
the repository if you really want to recompute.

| Phase | Command | Jobs | Approximate CPU time |
|---|---|---|---|
| 2B core + A1 | `make experiments-phase2b CONFIRM=yes` | 120 + 45 | 5.7 h |
| 2C | `python -I src/fedgrande/phase2c/regenerate.py`, then `python -I src/fedgrande/phase2c/run_all.py` | 15 | about 0.3 h, plus regeneration |
| 2D | `make experiments-phase2d CONFIRM=yes` | 90 (60 new) | about 4 h |
| 2E | `make experiments-phase2e CONFIRM=yes` | 90 | about 6.6 h, plus diagnostics (`phase2e/stability.py`) |
| 2F | `make experiments-phase2f CONFIRM=yes` | 60 | about 4 h |

- **Aggregation after a re-run:** `python -I src/fedgrande/phase2{b,c,d,e,f}/aggregate.py`, plus
  `src/fedgrande/phase2b/report.py` for 2B. These overwrite the phase reports and summaries.
- **Phase 2A gates:** the scripts in `src/fedgrande/phase2/` (see `src/fedgrande/phase2/README.md`).
- **Determinism:**
  - Runs are deterministic on the same machine and software: Phase 2C/2E reruns matched Phase 2B bitwise.
  - Across different CPUs or BLAS builds, small floating-point differences are possible.

## Protocol and result commits

See [PROJECT_TIMELINE.md](PROJECT_TIMELINE.md). The experiment boundary is the tag `experiments-final` = `7f7f6a4`.
