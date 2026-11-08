#!/usr/bin/env python3
"""Validate the repository without training anything.

Checks:
1. the `experiments-final` tag, when tags are present, points to 7f7f6a4;
2. key experiment artifacts exist;
3. the curated results and figure data are up to date with the saved results (both builders run in --check mode);
4. the safe tooling never imports training code or torch;
5. the pinned GRANDE worktrees, when present, sit at their pinned commits with no tracked changes;
6. environment versions match the historical snapshot (reported as warnings only);
7. local data (ARFF and caches) is present, otherwise a hint to run `make data`.

The unit tests run separately (`make test`); `make validate` runs both.
Exit code 0 = all hard checks pass.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FINAL = "7f7f6a42fd1929efc4fbfef3749e304c607eeaab"
PINS = {"third_party/grande_dev_db79c860": "db79c86060f06c15f46c3bfb9207dc30c3cb91ae",
        "third_party/grande_ref_07f7278b": "07f7278b30ab9ebbbdc544e5f9a91f1b5df7fecb"}
ARTIFACTS = ["reports/PHASE2B_PROTOCOL.md", "reports/PHASE2C_PROTOCOL.md", "reports/PHASE2D_PROTOCOL.md", "reports/PHASE2E_PROTOCOL.md",
             "reports/PHASE2F_PROTOCOL.md", "reports/phase2a_report.md", "reports/phase2b_report.md", "reports/phase2c_report.md",
             "reports/phase2d_report.md", "reports/phase2e_report.md", "reports/phase2f_report.md",
             "results/phase2b/summary/summary_all.json", "results/phase2c/summary/phase2c_summary.json",
             "results/phase2d/summary/phase2d_summary.json", "results/phase2e/summary/phase2e_summary.json",
             "results/phase2f/summary/phase2f_summary.json", "results/phase2e/gate0.json",
             "data/derived/phase2/dyads_v1.csv", "data/derived/phase2/splits_inner_v1.csv", "data/derived/phase2b/partitions_v1.csv"]
fails, warns = [], []


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check(ok, msg, hard=True):
    print(("PASS " if ok else ("FAIL " if hard else "WARN ")) + msg)
    if not ok:
        (fails if hard else warns).append(msg)


def main():
    tag = git("rev-parse", "-q", "--verify", "experiments-final^{commit}")
    if tag:
        check(tag == FINAL, f"tag experiments-final -> {tag[:7]} (expected {FINAL[:7]})")
    else:
        check(False, "tag experiments-final not present locally (run: git fetch --tags)", hard=False)
    missing = [p for p in ARTIFACTS if not (ROOT / p).exists()]
    check(not missing, f"key artifacts present ({len(ARTIFACTS) - len(missing)}/{len(ARTIFACTS)})" + (f" missing: {missing}" if missing else ""))
    for script in ("build_curated_results.py", "regenerate_figures.py"):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / script), "--check"], cwd=ROOT, capture_output=True, text=True)
        check(r.returncode == 0, f"{script} --check: {(r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else ''}")
    before = set(sys.modules)
    importlib.import_module("fedgrande.plotting.figures")
    loaded = set(sys.modules) - before
    bad = sorted(m for m in loaded if m == "torch" or m.startswith(("torch.", "fedgrande.fl", "fedgrande.phase2b.run")))
    check(not bad, "safe figure tooling imports no training code or torch" + (f" (found {bad[:5]})" if bad else ""))
    for wt, pin in PINS.items():
        p = ROOT / wt
        if not p.exists():
            check(False, f"{wt} absent (only needed for experiments; see docs/REPRODUCIBILITY.md)", hard=False)
            continue
        head = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(p), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        check(head == pin and not dirty, f"{wt} at {head[:8]} with {len(dirty.splitlines()) if dirty else 0} tracked changes")
    snap = json.loads((ROOT / "env/snapshots/fedgrande-dev.json").read_text())
    want = {**snap.get("all_distributions_metadata", {}), **snap.get("key_packages_metadata", {})}
    for pkg in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib"):
        try:
            v = md.version(pkg)
        except md.PackageNotFoundError:
            v = None
        w = want.get(pkg)
        check(w is None or v == w, f"{pkg} {v} (snapshot {w})", hard=False)
    from fedgrande.paths import ARFF
    caches = [ROOT / f"data/derived/phase2b/prep_C2_outer{k}.npz" for k in range(5)]
    check(ARFF.exists() and all(c.exists() for c in caches), "local data present (ARFF + 5 preprocessing caches); otherwise run `make data`", hard=False)
    print(f"\n{len(fails)} failure(s), {len(warns)} warning(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
