# Checkpoint 1 errata

Written 2026-11-07. The archived evidence files are NOT edited. Corrections live only here.
"Verified" means checked against the GRANDE repository cloned on 2026-11-07 (third_party/), not recalled from memory.

E1. Q14 did not run SpeedDating.
    The as-is notebook `GRANDE_minimal_example_with_comparison_BINARY.ipynb` at commit 07f7278b calls
    `openml.datasets.get_dataset(46915, ...)`, which is the churn dataset (verified). It uses a random
    train/valid/test split and a single run, so it is neither SpeedDating nor the paper's 5-fold protocol.

E2. No GRANDE-on-SpeedDating result exists from Checkpoint 1. No GRANDE metric from our own Q14 run survives
    either. The archived "last 10 notebook output lines" contain only XGBoost and CatBoost metrics, and the executed
    notebook (GRANDE_as_is_executed.ipynb) was lost with the old machine.
    Side note: the repository notebook already stores the author's outputs. Its CatBoost ROC AUC is
    0.9137693329656831 (verified present at 07f7278b), while our evidence shows 0.9140701376181285.
    So our Q14 output was a genuine re-execution, not a copy of the stored output.

E3. category-encoders version: 2.8.1 versus 2.9.0.
    - Q14_GRANDE_execution_evidence.txt prints 2.8.1. The Moodle Q14 answer states 2.9.0.
    - The category-encoders 2.9.0 wheel sets the module attribute `__version__ = '2.8.1'`, while its distribution
      metadata says 2.9.0.
    - requirements.txt at 07f7278b allows `category-encoders>=2.6.4,<=2.9.0`.
    Status: the most likely explanation is that 2.9.0 was installed and the evidence script printed the stale
    module attribute. This cannot be proven because the old environment is lost. From Phase 1 onward, versions are
    read with importlib.metadata only. Both values are recorded in env/snapshots/*.json.

E4. Q15 mapped a different implementation from the one Q14 executed.
    - Q15 cited line numbers in the TensorFlow-era `GRANDE/GRANDE.py` (commit fff4cbce).
    - Q14 executed the PyTorch rewrite at 07f7278b. That rewrite selects split features with softmax, and its
      entmax15 call is commented out (07f7278b GRANDE/GRANDE.py lines 304-307, verified).
    - The paper's own experiments imported a third copy, `experiments_paper/utilities/GRANDE.py`. That directory
      was removed from the repository before 07f7278b.
    - Equation-to-code mappings from Q15 must not be assumed to hold for 07f7278b or db79c860.

E5. Q11 reproduced the same kind of outer split, but not proven identical folds.
    - What was reproduced: StratifiedKFold(5, shuffle=True, random_state=42) on the OpenML row order, giving the
      same fold sizes as the paper and zero train/test overlap.
    - What was not established: that the fold indices are identical to the paper's folds. The paper pipeline
      preprocesses and possibly reorders rows before splitting.
    - Q11's phrase "reproduced the paper's 5-fold stratified outer split" should be read as "same procedure and
      sizes".

E6. The repository HEAD changed after Checkpoint 1.
    - Q13 correctly reported 07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb (2025-12-16, package 0.2.0) as the latest
      commit at that time.
    - Current master is db79c86060f06c15f46c3bfb9207dc30c3cb91ae (2026-10-20): "GRANDE 1.0.0: scikit-learn
      style package with inspection tools", with a pyproject.toml build and no AutoGluon dependency (verified).
    - Both commits are pinned as separate worktrees in third_party/.

E7. NEW audit finding, not part of Checkpoint 1: mirror/dyad leakage. Must be re-verified locally in Phase 2.
    - The audit of 2026-11-06 found that about 99.8% of rows (8,364 / 8,378) appear to have an exact
      partner-perspective mirror row with the same `match` label.
    - If confirmed, row-wise CV (including Q11's split) puts both perspectives of the same date into different
      folds, which leaks label information.
    - Q10 discussed participant overlap but not this dyad mirroring.
    - Status: UNVERIFIED in this project tree. Phase 1 deliberately does not reconstruct dyads or participants.

Also verified, related to the audit (not a Checkpoint 1 error):
GRANDE caps `selected_variables` at 50 features per estimator for fractional settings, with a floor of 10:
- 07f7278b: GRANDE/GRANDE.py lines 190-193
- db79c860: GRANDE/model.py lines 69-72

With 120 inputs and selected_variables=0.8, each estimator therefore uses 50 features, not 96. Any
communication-cost estimate must come from the actual trainable model state.

Addendum 2026-11-07 (Phase 1 observation, supports E3):
In both new environments (fedgrande-ref, fedgrande-dev), `importlib.metadata.version("category-encoders")` returns
2.9.0, while `category_encoders.__version__` returns 2.8.1. See env/snapshots/fedgrande-*.json. This reproduces the
Checkpoint 1 discrepancy exactly, so the 2.8.1 printed in Q14 is consistent with 2.9.0 having been installed.
It is still not proof about the lost environment.
