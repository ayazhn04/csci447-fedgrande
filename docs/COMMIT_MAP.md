# Commit map (old → canonical)

On 2026-11-09 the project history was rewritten to normalize commit metadata: tool co-author trailers were removed from
commit messages. **Only commit metadata changed.** Every commit keeps its exact file tree, author, timestamps, order and
parents, so every file and every scientific artifact is byte-identical. Because commit metadata is part of a commit's
hash, every project commit received a new SHA.

## Where old SHAs still appear

Frozen experiment records written *at run time* keep the identifiers that were valid when they were produced. These
include the `project_commit` fields in `results/phase2*/runs/*/result.json`, the phase summaries and run tables, the
generated phase reports, and the Phase 2E/2F protocol files. They are part of the frozen experimental tree (tag
`experiments-final`), so they are intentionally left unchanged.
- Translate any old SHA with this table, or with the machine-readable [`commit_map.tsv`](commit_map.tsv).
- The documentation layer (`README.md`, `PROJECT_STATE.md`, `docs/`, `results/curated/`, `scripts/`) uses the canonical SHAs.

External GRANDE commits (`db79c860…`, `07f7278b…`) belong to the third-party GRANDE repository and are unaffected.

## Map

| Old SHA | Canonical SHA | Commit |
|---|---|---|
| `1f4a038` | `2552fcb` | Phase 0-1: archive Checkpoint 1 evidence, pin GRANDE envs, audit SpeedDating |
| `20aad5e` | `5166c1a` | Phase 2A: GRANDE version equivalence, dyads, leakage-safe splits, wave audit, baseline, benchmarks |
| `615aeeb` | `570ce34` | Phase 2B protocol freeze: federated-safe preprocessing, partitions, FedAvg/local/central GRANDE and MLP |
| `f286965` | `bccbd45` | Phase 2B amendment A1: size-matched Dirichlet(0.1) label-skew stress-test partition (after B2 verified) |
| `834eb07` | `e05bb9c` | Phase 2B amendment A2: post-hoc within-client AUC (client-identity shortcut under Dirichlet) |
| `0170922` | `30765cb` | Phase 2B results: core matrix (natural, IID), Dirichlet stress test, auxiliary analyses, report |
| `53f718a` | `74a3a1b` | Phase 2C protocol freeze: P0-P3 personalization on Dirichlet(0.1), regenerated bitwise-verified checkpoints |
| `cd4ab90` | `81a4fcd` | Phase 2C results: P0-P3 personalization on Dirichlet(0.1), post-hoc per-fold within-client AUC (A1), report |
| `e17dcf4` | `aba6cbb` | Phase 2D protocol freeze: 64-tree and INT8-uplink FedAvg-GRANDE, communication accounting v2 |
| `db0f831` | `20c6393` | Phase 2D results: 64-tree and INT8-uplink FedAvg-GRANDE (90 jobs), communication frontier, report |
| `2da1d8f` | `cd687a6` | Phase 2E pre-freeze: Gate 0 (structure = 97.98% of payload), freeze scheduler, per-round accounting, tests, benchmark (runtime gate reached) |
| `ff3302f` | `80a55c3` | Phase 2E protocol freeze: F1/F2/F3 structure freezing after round 20 (full matrix approved) |
| `26b1c0c` | `2fe710b` | Phase 2E results: structure freezing after round 20 (F1/F2/F3, 90 jobs), drift diagnostics, Phase 2D comparison, report |
| `906aef5` | `279fcac` | checkpoint: pause Phase 2F before final execution |
| `1f243a3` | `c552f66` | Phase 2F protocol freeze: CGSF G1 (global) and G2 (node-wise) consensus-gated structural freezing |
| `7f7f6a4` | `e38c7a9` | Phase 2F results: CGSF G1/G2 final matrix (60 jobs), aggregation, figures, report; G3 not eligible |
| `40ad41c` | `b6d1069` | style: standardize scientific figures |
| `3d4d6fd` | `0824c8c` | chore: prepare reproducible teammate workflow |
| `4d2f1e1` | `beb043e` | fix: precise fig08 title wording and curated role labels |
| `7ab12f7` | `eda57f4` | docs: add final project handoff and research map |
| `42d1f01` | `3cc71ca` | docs: finalize repository packaging |

Full 40-character SHAs are in [`commit_map.tsv`](commit_map.tsv).
