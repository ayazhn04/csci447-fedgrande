# Phase 2A: run order

Run from the repository root with the environment's own interpreter (`"$CONDA_PREFIX/bin/python" -I ...`).
Use `fedgrande-dev` unless stated otherwise.

1. `gate_b_dyads.py` writes `data/derived/phase2/dyads_v1.csv` and `results/phase2/gate_b/`.
2. `gate_c_splits.py` writes `data/derived/phase2/splits_{outer,inner}_v1.csv` and `results/phase2/gate_c/`.
3. `gate_d_waves.py` writes `results/phase2/gate_d/`.
4. E0a (forward equivalence):
   - `e0a_compare.py --prepare`
   - `e0a_forward.py --impl dev` (run in `fedgrande-dev`)
   - `e0a_forward.py --impl ref` (run in `fedgrande-ref`)
   - `e0a_compare.py --compare`
   - Each script takes `--config configs/phase2/e0a_forward.json --workdir results/phase2/gate_a/e0a`.
5. E0b (training sanity):
   - `e0b_prepare.py`
   - `e0b_train.py --impl dev` (run in `fedgrande-dev`)
   - `e0b_train.py --impl ref` (run in `fedgrande-ref`)
   - `e0b_compare.py`
6. `gate_ef_run.py` runs every job in `configs/phase2/gate_ef_runs.json`, each through `src/fedgrande/grande_runner.py` in its own process. It can be resumed.
7. `gate_ef_summarize.py`, then `phase2a_summary.py`. These write `results/phase2/phase2a_summary.json` and `reports/phase2a_report.md`.

Tests: from `tests/`, run `python -m unittest test_data test_dyads test_splits test_e0_mapping` in either environment.

The `.npz` intermediates are git-ignored because they can be regenerated. Their SHA256 or contents are recorded in the JSON reports.
