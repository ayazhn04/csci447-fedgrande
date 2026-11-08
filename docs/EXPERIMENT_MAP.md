# Experiment map

This page lists every experiment that produced a reported result. Exact settings are in
[METHODS_REFERENCE.md](METHODS_REFERENCE.md), numbers in [RESULTS_LEDGER.md](RESULTS_LEDGER.md), and the phase
chronology in [PROJECT_TIMELINE.md](PROJECT_TIMELINE.md).

**Notation:**
- 5 × 3 = 5 grouped outer folds (C2) × 3 model seeds (42, 447, 2026).
- "Partition" refers to the 21-client split.
- *Reused* means the result comes from an earlier phase and was not rerun.

**Labels:**
- **PRIMARY:** headline comparisons.
- **CONTROL:** isolates a factor.
- **STRESS:** synthetic extreme condition.
- **MECHANISTIC:** explains a primary result.
- **COMM:** communication-efficiency extension.

| ID | Method | Label | Purpose | Partition | Folds × seeds (jobs) | Main output |
|---|---|---|---|---|---|---|
| central_grande | Centralized GRANDE (256 trees, depth 5) | PRIMARY | Upper reference: cost of federation | none (pooled) | 5×3 (15) | `results/phase2b/runs/central_grande__none__*` |
| fedavg_grande | FedAvg GRANDE, local Adam, ≤100 rounds | PRIMARY | Main method | natural, IID | 5×3 each (30) | `results/phase2b/runs/fedavg_grande__*` |
| local_grande | Local-only GRANDE (one model per client) | CONTROL | Lower reference: no collaboration | natural, IID | 5×3 each (30) | `results/phase2b/runs/local_grande__*` |
| central_mlp | Centralized MLP 502-128-64-2 | CONTROL | Model-class reference | none | 5×3 (15) | `results/phase2b/runs/central_mlp__*` |
| fedavg_mlp | FedAvg MLP | CONTROL | Is the gap specific to GRANDE? | natural, IID | 5×3 each (30) | `results/phase2b/runs/fedavg_mlp__*` |
| IID partition | size-matched IID clients | CONTROL | Isolates wave heterogeneity from size skew | iid | (within the rows above) | `data/derived/phase2b/partitions_v1.csv` |
| Dirichlet stress | local, FedAvg GRANDE, FedAvg MLP | STRESS | Extreme label skew (2B amendment A1) | dirichlet0.1 | 5×3 × 3 methods (45) | `results/phase2b/runs/*__dirichlet0.1__*` |
| Structural diagnostics | consensus, agreement, update cosine | MECHANISTIC | RQ3 | all FedAvg GRANDE runs | (logged per round) | `results/phase2b/summary/consensus_*` |
| P0 | Global FedAvg GRANDE (Phase 2B checkpoint) | MECHANISTIC | Baseline under label skew | dirichlet0.1 | 5×3 (15 regenerated, bitwise-verified checkpoints) | `results/phase2c/` |
| P1 | + per-client logit intercept | MECHANISTIC | Calibration hypothesis | dirichlet0.1 | same 15 | `results/phase2c/` |
| P2 | + personalized leaves | MECHANISTIC | Personalization hypothesis | dirichlet0.1 | same 15 | `results/phase2c/` |
| P3 | + personalized leaves and estimator weights | MECHANISTIC | Personalization hypothesis | dirichlet0.1 | same 15 | `results/phase2c/` |
| C0 | 256 trees, FP32 (= Phase 2B fedavg_grande) | COMM (reference) | Baseline cost | natural, IID | reused | `results/phase2b/runs/fedavg_grande__*` |
| C1 | 64 trees, FP32 | COMM | Ensemble redundancy | natural, IID | 5×3 each (30) | `results/phase2d/runs/fedavg_grande64__*` |
| C2 | 256 trees, INT8 uplink | COMM | Update quantization | natural, IID | 5×3 each (30) | `results/phase2d/runs/fedavg_grande_int8__*` |
| C3 | 64 trees + INT8 uplink | COMM | Combined | natural, IID | 5×3 each (30) | `results/phase2d/runs/fedavg_grande64_int8__*` |
| Gate 0 | payload share per tensor | COMM (analysis) | Is structure the bottleneck? | – | analytic | `results/phase2e/gate0.json` |
| F0 | no freeze (= Phase 2B fedavg_grande) | COMM (reference) | Baseline | natural, IID | reused | `results/phase2b/runs/fedavg_grande__*` |
| F1 | freeze feature logits after round 20 | COMM | Do feature identities still need federation? | natural, IID | 5×3 each (30) | `results/phase2e/runs/fedavg_grande_frzidx__*` |
| F2 | freeze features + thresholds after round 20 | COMM | Does structure still need federation? | natural, IID | 5×3 each (30) | `results/phase2e/runs/fedavg_grande_frzstruct__*` |
| F3 | F2 + INT8 uplink | COMM | Complementarity with 2D | natural, IID | 5×3 each (30) | `results/phase2e/runs/fedavg_grande_frzstruct_int8__*` |
| F0 drift | F0 reruns with snapshots | MECHANISTIC (diagnostic) | What still changes after round 20? | natural, IID | 5 folds × seed 42 (10) | `results/phase2e/stability_by_run.csv` |
| G1 | consensus-gated global freeze | COMM | Automatic freeze timing | natural, IID | 5×3 each (30) | `results/phase2f/runs/fedavg_grande_cgsf_global__*` |
| G2 | consensus-gated node-wise freeze | COMM | Progressive, per-node freezing | natural, IID | 5×3 each (30) | `results/phase2f/runs/fedavg_grande_cgsf_node__*` |
| G3 | G2 + INT8 | COMM (conditional) | Only if G2 qualified | – | **not run** (ineligible) | `reports/PHASE2F_PROTOCOL.md` (D1) |

## Auxiliary analyses (not part of the matched comparisons)

| ID | Purpose | Output |
|---|---|---|
| Phase 2A E0a/E0b | GRANDE 0.2.0 (07f7278b) vs 1.0.0 (db79c860): bitwise-identical forward pass and training | `results/phase2/gate_a/` |
| Phase 2A Gate E/F | First 1,024-tree baseline (fold 0) and depth/ensemble runtime benchmarks | `results/phase2/gate_ef/` |
| Phase 2A C1 vs C2 diagnostic | Row-wise vs grouped split, single fold (descriptive) | `reports/phase2a_report.md` §C |
| Phase 2B N | Stock GRANDE 1.0 preprocessing vs federated-safe preprocessing (same centralized loop, fold 0, 3 seeds) | `results/phase2b/prep_check/` |
| Phase 2B J | Stock `GRANDEClassifier` (1,024 trees) reference, 5 folds | `results/phase2b/stock_reference/` |

## Development-only artifacts (never results)

- `results/phase2b/smoke`, `dev_runs`;
- `results/phase2c/smoke*`;
- `results/phase2{d,e,f}/benchmark`;
- `results/phase2f/benchmark/INTERRUPTED_*`.
