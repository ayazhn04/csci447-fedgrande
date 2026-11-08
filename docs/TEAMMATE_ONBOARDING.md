# Teammate onboarding (for Nuray)

Welcome! The experiments are finished. This repository now exists so that you can understand them, regenerate tables
and figures, and help write the report and slides. **You never need to train a model.**

## If you have 10 minutes

Read these, in order:
1. [README](../README.md)
2. [START_HERE](START_HERE.md)
3. [RESULTS_LEDGER](RESULTS_LEDGER.md)
4. [FIGURE_CATALOG](FIGURE_CATALOG.md)

## If you want to understand the methods

- [METHODS_REFERENCE](METHODS_REFERENCE.md): exact settings.
- [RESEARCH_QUESTIONS](RESEARCH_QUESTIONS.md): what we asked and found.
- [EXPERIMENT_MAP](EXPERIMENT_MAP.md): every experiment and where its output lives.
- [PROJECT_TIMELINE](PROJECT_TIMELINE.md): the order in which things happened, and why.

## If you want to help write the report

- [REPORT_HANDOFF](REPORT_HANDOFF.md): section-by-section blueprint.
- [CLAIMS_AND_LIMITATIONS](CLAIMS_AND_LIMITATIONS.md): check every sentence against it.

## If you want to help make slides

- [PRESENTATION_HANDOFF](PRESENTATION_HANDOFF.md): 12-slide storyline.
- [FIGURE_CATALOG](FIGURE_CATALOG.md) and `reports/figures/final_style/`: PNG for slides, PDF for the report.

## Reproducing the environment

You need conda (Miniforge recommended) and git. The experiments ran on macOS arm64 with Python 3.12.14.

```bash
git clone https://github.com/ayazhn04/csci447-fedgrande.git
cd csci447-fedgrande
conda env create -f environment.yml
conda activate fedgrande-dev
```

That is enough for tables and figures. For the unit tests, also install the pinned GRANDE and fetch the data:

```bash
make grande
pip install --no-deps -e third_party/grande_dev_db79c860
make data
```

## Regenerating tables and figures (safe: no training)

```bash
make check       # verify that tables/figures match the saved results
make summaries   # rebuild results/curated/ and the tables in docs/RESULTS_LEDGER.md
make figures     # rebuild reports/figures/final_style/
make validate    # repository checks + all unit tests (needs `make data` and `make grande`)
```

These commands only read saved result files. `make figures` and `make summaries` are deterministic: if you have not
changed any code, `git status` stays clean afterwards.

## Do not run

| Command | Why not |
|---|---|
| `make experiments-*` | Re-runs whole experiment matrices: hours of CPU. They are guarded and refuse to run without `CONFIRM=yes`. |
| `python src/fedgrande/phase2b/run_matrix.py ...` / `run_job.py` | Trains models. |
| `python src/fedgrande/phase2*/aggregate.py`, `phase2b/report.py` | Rewrite the frozen phase reports and summaries. Not needed: use `make summaries`. |
| `python src/fedgrande/phase2b/prepare.py`, `prepare_dirichlet.py` | Rewrite tracked partition and provenance files. Use `make data` instead, which never touches tracked files. |
| `python src/fedgrande/phase2c/regenerate.py`, `run_all.py` | Regenerate checkpoints and personalization runs. |
| Anything under `third_party/` | Pinned third-party code; must stay unmodified. |

**Rule of thumb:** if a command is not in the "safe" block above, ask first.

## Collaborating with git

- **`main`** is the stable, documented project. Do not push to it directly.
- **Your work goes on a branch:**

  ```bash
  git switch -c report/intro-draft        # or slides/storyline, figures/fig02-tweak, docs/fix-typo
  # ...edit...
  git add -A && git commit -m "report: draft introduction"
  git push -u origin report/intro-draft
  ```

- **Open a pull request** on GitHub and ask Ayazhan to review. Merge after review, then delete the branch.
- **Branch prefixes:** `report/`, `slides/`, `figures/`, `docs/`.
- Keep pull requests small and focused.
- **Never move or delete the `experiments-final` tag.** It marks the frozen experimental state (7f7f6a4).
- If you change plotting code, run `make check`. It fails if figure *data* would change; data should never change.

## Getting access (for the repository owner)

The repository is private.

**To add Nuray:**
- **Web:** GitHub → `ayazhn04/csci447-fedgrande` → Settings → Collaborators → Add people → enter her exact GitHub
  username → role "Write".
- **CLI**, once her username is known:

  ```bash
  gh api -X PUT repos/ayazhn04/csci447-fedgrande/collaborators/<her-username> -f permission=push
  ```

She must accept the email invitation before she can clone.
