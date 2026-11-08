# Project state

**Experiment stage: COMPLETE** (Phases 1, 2A–2F). No further experiments are planned or allowed.

## Commits and tags

| Item | Value |
|---|---|
| Final experiment commit | `e38c7a9b84c6a40dadd3049b9ff1c5c9ba4329e2` (`e38c7a9`), annotated tag **`experiments-final`** |
| Final Phase 2F protocol commit | `c552f66` |
| Documentation / packaging | commits after `experiments-final`; the final packaging commit carries the tag **`documentation-ready`** (`git rev-parse documentation-ready`) |
| Pinned GRANDE (experiments) | 1.0.0 @ `db79c86060f06c15f46c3bfb9207dc30c3cb91ae` |
| Pinned GRANDE (historical reference) | 0.2.0 @ `07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb` |
| GitHub | https://github.com/ayazhn04/csci447-fedgrande (**private**), default branch `main`; tags `experiments-final`, `documentation-ready` |
| Environment | `fedgrande-dev`: Python 3.12.14, torch 2.7.1 (see `environment.yml`, `env/snapshots/`) |

## Snapshot

| | |
|---|---|
| **Dataset** | SpeedDating, OpenML 40536: 8,378 rows, 16.5% positive, 21 natural clients (waves); grouped 5-fold CV over mirror-row leakage groups; seeds 42/447/2026 |
| **Main baseline** | Centralized GRANDE (256 trees, depth 5), pooled OOF AUC 0.857. Also local-only GRANDE (0.741) and MLPs |
| **Main method** | FedAvg GRANDE with local Adam: 0.844 (natural) / 0.845 (IID) |
| **Final recommended method** | FedAvg GRANDE with **structure freezing once consensus is high**: G1 (consensus-timed, round 20.4 ± 1.4) or equivalently a freeze after round 20 (F2). Optionally INT8 for the remaining tensors (F3 ≈ F2) |
| **Strongest result** | Structure is 98% of the payload; freezing it after it stabilizes costs about 0.002 AUC (−0.0018 / −0.0022) for 3.6–4.1× less communication; G1 finds the freeze point automatically |
| **Final Phase 2F conclusion** | *Consensus gating reproduces the fixed schedule automatically but does not improve it.* |
| **Main limitations** | One dataset; no HPO (100-round budget binding); 3 seeds on fixed folds (descriptive); conceptual tensor bytes, not network traffic; no privacy mechanisms; synthetic label-skew stress test; fixed predeclared freeze round and gate thresholds |

## Where things live

| What | Where |
|---|---|
| Report-ready numbers | `docs/RESULTS_LEDGER.md`, `results/curated/` |
| Per-phase summaries and runs | `results/phase2{,b,c,d,e,f}/` |
| Protocols and phase reports | `reports/PHASE2*_PROTOCOL.md`, `reports/phase2*_report.md` |
| Final figures | `reports/figures/final_style/` (catalog: `docs/FIGURE_CATALOG.md`) |
| Phase-time figures | `reports/figures/phase2{b,d,e,f}/` |
| Documentation hub | `docs/START_HERE.md` |

## Remaining work

- Write the final course report (blueprint: `docs/REPORT_HANDOFF.md`).
- Prepare the final presentation (storyline: `docs/PRESENTATION_HANDOFF.md`).
- Prepare the defense / Q&A (`docs/CLAIMS_AND_LIMITATIONS.md`).
- Invite the teammate to the private GitHub repository once her GitHub username is known, and coordinate contributions
  (`docs/TEAMMATE_ONBOARDING.md`, section "Getting access").
- Package the final submission.

No further experiments.
