"""Phase 1: compare the unchanged historical loader's stdout against the archived Checkpoint 1 values.

Standard library only. Prints MATCH/DIFF per item, writes JSON, exits 1 on any DIFF.
"""
import argparse
import json
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader-output", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = open(a.loader_output, encoding="utf-8").read()
    E = json.load(open(a.expected))
    lines = out.splitlines()
    res = []

    def check(name, expected, observed):
        res.append({"check": name, "expected": expected, "observed": observed,
                    "status": "MATCH" if expected == observed else "DIFF"})

    def grab(pat, cast=str):
        m = re.search(pat, out, re.M)
        return cast(m.group(1)) if m else None

    check("OpenML ID", E["openml_id"], grab(r"^OpenML ID: (\d+)$", int))
    check("Version", E["openml_version"], grab(r"^Version: (\d+)$", int))
    check("X shape", E["X_shape"], [int(v) for v in re.findall(r"\d+", grab(r"^X shape: (.*)$") or "")])
    check("y shape", E["y_shape"], [int(v) for v in re.findall(r"\d+", grab(r"^y shape: (.*)$") or "")])
    check("X+y shape", E["Xy_shape"], [int(v) for v in re.findall(r"\d+", grab(r"^Combined X\+y shape: (.*)$") or "")])
    i = lines.index("=== FIRST FIVE ROWS (X + target) ===") if "=== FIRST FIVE ROWS (X + target) ===" in lines else None
    check("first five rows (header + 5 lines, byte-equal text)", E["first_five_rows_csv_lines"],
          lines[i + 1:i + 7] if i is not None else None)
    check("class 0 count", E["class_counts"]["0"], grab(r"^class 0: (\d+) \(", int))
    check("class 1 count", E["class_counts"]["1"], grab(r"^class 1: (\d+) \(", int))
    check("minority percentage", E["minority_pct_reported"], grab(r"^Minority percentage: ([\d.]+)%$"))
    check("total missing predictor cells", E["missing_total"], grab(r"^Total missing predictor cells: (\d+)$", int))
    check("predictor columns with missing", E["missing_columns"],
          grab(r"^Predictor columns with missing values: (\d+)$", int))
    j = lines.index("=== MISSING VALUES ===")
    k = next(n for n in range(j, len(lines)) if lines[n].startswith("Target missing values"))
    table = {m.group(1): int(m.group(2)) for l in lines[j + 3:k] for m in [re.match(r"^(\S+)\s+(\d+)$", l)] if m}
    check("per-column missing table", E["missing_by_column"], table)
    check("target missing", E["target_missing"], grab(r"^Target missing values: (\d+)$", int))
    folds = [{"fold": int(f), "train": int(tr), "test": int(te), "overlap": int(o)} for f, tr, te, o in
             re.findall(r"^fold (\d): train=(\d+), test=(\d+), train/test overlap=(\d+)$", out, re.M)]
    check("fold lines", E["folds"], folds)
    check("held out exactly once", E["held_out_exactly_once"],
          grab(r"^Samples held out exactly once across the 5 outer folds: (\d+)$", int))
    check("held out more than once", 0, grab(r"^Samples held out more than once: (\d+)$", int))
    check("never held out", 0, grab(r"^Samples never held out: (\d+)$", int))

    n_diff = sum(r["status"] == "DIFF" for r in res)
    print(f"Historical loader output vs Checkpoint 1 [{a.label}]")
    for r in res:
        shown = (lambda v: (str(v)[:90] + "...") if len(str(v)) > 90 else v)
        print(f"{r['status']:6s}  {r['check']}: {shown(r['expected'])} -> {shown(r['observed'])}")
    print(f"\n{len(res)} checks, {n_diff} DIFF")
    json.dump({"label": a.label, "loader_output": a.loader_output, "checks": res, "n_diff": n_diff},
              open(a.out, "w"), indent=2)
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
