# CSCI447P016 Federated GRANDE: teammate workflow.
#
# SAFE targets (no model training, no model evaluation):
#   make validate   : repository checks + full unit-test suite (needs `make data` once for the data-dependent tests)
#   make check      : verify curated tables and figure data are unchanged versus the saved results
#   make summaries  : rebuild results/curated/*.csv|json and the tables in docs/RESULTS_LEDGER.md from saved results
#   make figures    : rebuild reports/figures/final_style/* (PNG + PDF + plotted data) from saved results
#   make slide-figures : rebuild reports/figures/presentation/* (the two regenerated deck figures) from saved results
#   make test       : unit tests only
#   make data       : download the OpenML 40536 ARFF (MD5-checked) and rebuild the gitignored preprocessing caches
#   make grande     : clone the pinned GRANDE commits as worktrees in third_party/ (needed for tests and experiments)
#
# EXPENSIVE targets (re-run frozen experiments; hours of CPU; NOT part of any default target):
#   make experiments-phase2b CONFIRM=yes   (~6 h)   ... see docs/REPRODUCIBILITY.md, Level 4, for every phase
#
# The default target only prints this help. Nothing here runs experiments unless you type CONFIRM=yes.

PY ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,python)

.DEFAULT_GOAL := help
.PHONY: help validate check summaries figures slide-figures test data grande experiments-phase2b experiments-phase2d experiments-phase2e experiments-phase2f _confirm

help:
	@sed -n '1,17p' Makefile | sed 's/^# \{0,1\}//'

validate:
	$(PY) -I scripts/validate_project.py
	$(PY) -m unittest discover -s tests

check:
	$(PY) -I scripts/build_curated_results.py --check
	$(PY) -I scripts/regenerate_figures.py --check

summaries:
	$(PY) -I scripts/build_curated_results.py

figures:
	$(PY) -I scripts/regenerate_figures.py

slide-figures:
	$(PY) -I scripts/make_presentation_figures.py

test:
	$(PY) -m unittest discover -s tests

data:
	$(PY) scripts/build_data_caches.py

grande:
	test -d third_party/GRANDE-repo || git clone --no-checkout https://github.com/s-marton/GRANDE.git third_party/GRANDE-repo
	test -d third_party/grande_dev_db79c860 || git -C third_party/GRANDE-repo worktree add --detach ../grande_dev_db79c860 db79c86060f06c15f46c3bfb9207dc30c3cb91ae
	test -d third_party/grande_ref_07f7278b || git -C third_party/GRANDE-repo worktree add --detach ../grande_ref_07f7278b 07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb

# ---------------------------------------------------------------------------------------------------------------------
# EXPENSIVE: re-runs of the frozen experiment matrices. Resumable (finished jobs with result.json are skipped), so on a
# complete checkout they do nothing. Never needed for the report, slides or figures.
_confirm:
	@test "$(CONFIRM)" = "yes" || (echo "Refusing: this re-runs experiments for hours. Re-run with CONFIRM=yes."; exit 2)

experiments-phase2b: _confirm
	$(PY) -I src/fedgrande/phase2b/run_matrix.py --stage core
	$(PY) -I src/fedgrande/phase2b/run_matrix.py --stage dirichlet --config configs/phase2b/amendment_A1_dirichlet.json

experiments-phase2d: _confirm
	$(PY) -I src/fedgrande/phase2b/run_matrix.py --stage compression --config configs/phase2d/compression.json

experiments-phase2e: _confirm
	$(PY) -I src/fedgrande/phase2b/run_matrix.py --stage structure --config configs/phase2e/structure.json

experiments-phase2f: _confirm
	$(PY) -I src/fedgrande/phase2b/run_matrix.py --stage adaptive --config configs/phase2f/cgsf.json
