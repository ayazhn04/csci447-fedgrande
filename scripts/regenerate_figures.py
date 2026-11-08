#!/usr/bin/env python3
"""Regenerate the final-style figures from saved, git-tracked results. Safe: no training, no model evaluation.

Writes reports/figures/final_style/<fig>.png (450 dpi) + .pdf and the exact plotted data in
reports/figures/final_style/data/<fig>.csv. Each figure asserts anchor values against the frozen phase summaries.

Usage: python -I scripts/regenerate_figures.py [--check]
--check regenerates the plotted-data CSVs in memory-equivalent form and fails if any differs from the committed copy.
"""
import argparse
import filecmp
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedgrande.plotting import figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify committed plotted-data CSVs are unchanged; images are not overwritten")
    a = ap.parse_args()
    if a.check:
        committed = figures.DATA
        tmp = Path(tempfile.mkdtemp())
        figures.OUT, figures.DATA = tmp, tmp / "data"
        figures.build_all()
        diffs = [p.name for p in sorted(committed.glob("*.csv")) if not filecmp.cmp(p, tmp / "data" / p.name, shallow=False)]
        missing = [p.name for p in (tmp / "data").glob("*.csv") if not (committed / p.name).exists()]
        shutil.rmtree(tmp)
        print("figure data unchanged" if not (diffs or missing) else f"CHANGED: {diffs} MISSING: {missing}")
        sys.exit(1 if diffs or missing else 0)
    out = figures.build_all()
    print(f"wrote {len(out)} files to {figures.OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
