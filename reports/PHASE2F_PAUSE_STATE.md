# Phase 2F pause state (user-requested pause, 2026-11-08)

## Current stage

Phase 2F (CGSF: consensus-gated progressive structural freezing), stage **E0 implementation → engineering gate**.

- **Implementation:** complete and tested.
- **Engineering gate:** started but **not completed**. It was interrupted during its first run.
- **Protocol freeze:** **NOT done.** `reports/PHASE2F_PROTOCOL.md` has not been written, and no protocol commit exists.
- **Final 60-job matrix:** **not started.** No final Phase 2F result exists.
- **G3:** not applicable yet. Its eligibility can only be decided after the final G2 results.

## Git

- Last commit before the pause: `26b1c0c` (Phase 2E results).
- Pause checkpoint commit: **"checkpoint: pause Phase 2F before final execution"**, the commit that adds this file
  (`git log -1`).
  - It contains the Phase 2F implementation, config, tests and aggregator.
  - It is **not** a scientific or protocol freeze.
- No push has been made to any remote.
- Third-party GRANDE worktrees: 0 tracked modifications (checked at the pause).

## What is implemented (checkpoint commit)

- `src/fedgrande/fl/adaptive.py`: `fedavg_adaptive` and `node_consensus`.
  - G1 global trigger.
  - G2 node-wise feature and threshold gates.
  - Slice-level masks with gradient zeroing, a fresh Adam each round, and bitwise assertions after every optimizer step.
  - Masked aggregation.
  - Slice-level byte accounting, with newly frozen slices broadcast once in the next downlink.
  - Optional INT8 uplink for G3, using the Phase 2D codec.
- `src/fedgrande/phase2b/run_job.py`: dispatches methods that carry an `adaptive` config to `fedavg_adaptive`, saves
  `freeze_rounds.npz`, and does per-round communication accounting.
- `src/fedgrande/phase2b/run_matrix.py`: new stage `adaptive`, output to `results/phase2f/runs`.
- `configs/phase2f/cgsf.json`:
  - G1 `fedavg_grande_cgsf_global`: warm-up 10, 3 consecutive rounds, mean consensus ≥ 0.98, unanimous fraction ≥ 0.85.
  - G2 `fedavg_grande_cgsf_node`: warm-up 10, 3 consecutive rounds, C_j ≥ 0.95, D ≤ 0.01, S ≤ 0.05, ε = 1e-6.
  - A conditional G3 specification and its eligibility rule.
- `src/fedgrande/phase2f/aggregate.py`: tables, G3 eligibility, figures A/B/C and the report. Not yet run.
- `tests/test_phase2f.py`: 9 tests.

## Tests at the pause

The full suite passes: **70 of 70** (`results/phase2f/tests_pause.log`). This includes the 9 Phase 2F tests on
consensus, slice masking under Adam, gate ordering and boundaries, the global trigger and its accounting, determinism,
pre-freeze identity with FedAvg, mixed-mask aggregation, and freezing decisions being independent of validation data.

## Engineering-gate jobs (fold 0, seed 42; runtime and validation only)

| Job | Status |
|---|---|
| `fedavg_grande_cgsf_global__natural__f0__s42` | **INTERRUPTED** after about 2.5 min. Its directory holds only `INCOMPLETE.txt`; no outputs were written, because the job writes only at the end. **Must be rerun.** |
| `fedavg_grande_cgsf_global__iid__f0__s42` | not started |
| `fedavg_grande_cgsf_node__natural__f0__s42` | not started |
| `fedavg_grande_cgsf_node__iid__f0__s42` | not started |

Completed Phase 2F jobs: **none**. Remaining jobs:
- the 4 engineering-gate runs;
- then the final 60-job matrix (G1 and G2 × natural and IID × 5 folds × 3 seeds);
- then G3 (30 runs) only if eligible.

## How to resume (only on the user's explicit instruction)

```bash
cd ~/csci447-fedgrande
source env/conda_hook.zsh && conda activate fedgrande-dev
PY="$CONDA_PREFIX/bin/python"
```

**1. Rerun the engineering gate.** Delete the empty interrupted directory first, or keep it; the job recreates its own
outputs.

```bash
for m in fedavg_grande_cgsf_global fedavg_grande_cgsf_node; do for p in natural iid; do "$PY" -I src/fedgrande/phase2b/run_job.py --method $m --partition $p --fold 0 --seed 42 --config configs/phase2f/cgsf.json --out-root results/phase2f/benchmark; done; done
```

**What to check:**
- Rounds 1–13 must be identical to F0's Phase 2B `rounds.csv`, because no freeze can take effect before round 14.
- Trigger round (G1); frozen fractions and byte curves (G2).
- Runtime projection for 60 jobs: report to the user if it is above about 5 h.
- Peak memory.
- No assertion failures.

**2. Freeze the protocol.**
- Write `reports/PHASE2F_PROTOCOL.md`.
- Run the full test suite.
- Make a local protocol commit **before** any final run.

**3. Run the final matrix.** It is resumable and skips jobs that already have a `result.json`.

```bash
"$PY" -I src/fedgrande/phase2b/run_matrix.py --stage adaptive --config configs/phase2f/cgsf.json
```

**4. Decide on G3.**
- `"$PY" -I src/fedgrande/phase2f/aggregate.py` computes `G3_eligibility`.
- If G3 is eligible: make an amendment commit that adds the `fedavg_grande_cgsf_node_int8` method to the config (the
  specification is already under `G3_conditional`), then run it.

**5. Aggregate and report.** Rerun the aggregator, then write the final report.

## Warnings

- **The interrupted engineering run left no partial outputs**, only `INCOMPLETE.txt`. `results/phase2f/bench.log` is
  empty.
- **The runtime projection is still unknown.** The plan expected roughly G1 ≈ 4–6 min and G2 ≈ 6 min per run, so about
  5 h for 60 jobs. G2 computes gradients for partially frozen tensors and will not get Phase 2E F2's post-freeze speed-up
  unless a whole tensor becomes frozen.
- **Disk space** fell from about 24 to **22 GiB free** through activity outside this project. That is still well above
  the 8 GiB guard.
- **No background process remains.** The engineering-gate shell chain and its Python job were stopped, and no watcher,
  monitor or queued job is left.
