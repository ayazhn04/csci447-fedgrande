"""Phase 1: audit SpeedDating (OpenML 40536) against the Checkpoint 1 numbers.

Every check prints EXPECTED vs OBSERVED. Nothing is overwritten or silently corrected.
Exit code 1 if any check differs.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATASET_ID = 40536
SEED = 42
RESULTS = []


def check(name, expected, observed, source):
    ok = expected == observed
    RESULTS.append({"check": name, "expected": expected, "observed": observed, "source": source,
                    "status": "MATCH" if ok else "DIFF"})


def to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def get_data(ds, **kw):
    try:
        return ds.get_data(dataset_format="dataframe", **kw)
    except TypeError:
        return ds.get_data(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="folder written by fetch_speeddating.py")
    ap.add_argument("--cache-dir", required=True, help="project-local OpenML cache")
    ap.add_argument("--expected", required=True, help="results/phase1/expected/checkpoint1_expected.json")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.cache_dir, exist_ok=True)
    if hasattr(openml.config, "set_root_cache_directory"):
        openml.config.set_root_cache_directory(os.path.abspath(a.cache_dir))
    else:
        openml.config.cache_directory = os.path.abspath(a.cache_dir)

    with open(a.expected) as f:
        E = json.load(f)
    HIST_HEADER = E["header"]
    HIST_MISSING = E["missing_by_column"]
    HIST_FOLDS = [(d["train"], d["test"]) for d in E["folds"]]
    with open(os.path.join(a.raw_dir, "download_manifest.json")) as f:
        manifest = json.load(f)

    ds = openml.datasets.get_dataset(DATASET_ID, download_data=True, download_qualities=True,
                                     download_features_meta_data=True)
    X, y, cat_ind, names = get_data(ds, target=ds.default_target_attribute)

    # Metadata
    check("openml dataset id", E["openml_id"], int(ds.dataset_id), "Moodle Q7/Q11")
    check("dataset name", "SpeedDating", ds.name, "Moodle Q11")
    check("dataset version", E["openml_version"], int(ds.version), "Moodle Q11")
    check("default target", "match", ds.default_target_attribute, "Moodle Q7")
    check("upload date (OpenML, not collection date)", "2016-11-16", str(ds.upload_date)[:10], "Moodle Q12")
    check("licence", "Public", ds.licence, "Moodle Q12")
    check("ignore attributes", ["decision", "decision_o"], sorted(ds.ignore_attribute or []), "Phase-1 spec / audit 2026-11-06")
    check("openml md5 (API via openml-python)", "0c5590a0c81750687ee1b0ee68168a5d", ds.md5_checksum, "Phase-1 spec / audit 2026-11-06")
    check("downloaded ARFF md5 matches OpenML", True, bool(manifest["md5_ok"]), "fetch_speeddating.py")

    # Shapes and leakage columns
    check("X shape", tuple(E["X_shape"]), tuple(X.shape), "Moodle Q11")
    check("y shape", tuple(E["y_shape"]), tuple(y.shape), "Moodle Q11")
    df = X.copy()
    df[ds.default_target_attribute] = y
    check("X+y shape", tuple(E["Xy_shape"]), tuple(df.shape), "Moodle Q11")
    check("column order equals historical header", True, list(df.columns) == HIST_HEADER, "Moodle Q11 header line")
    check("'decision' absent from X", True, "decision" not in X.columns, "Phase-1 spec / audit 2026-11-06")
    check("'decision_o' absent from X", True, "decision_o" not in X.columns, "Phase-1 spec / audit 2026-11-06")
    check("'match' absent from X", True, "match" not in X.columns, "sanity")

    # Classes
    counts = y.astype(str).value_counts().sort_index()
    check("class 0 count", E["class_counts"]["0"], int(counts.get("0", -1)), "Moodle Q8/Q11")
    check("class 1 count", E["class_counts"]["1"], int(counts.get("1", -1)), "Moodle Q8/Q11")
    minority_pct = 100 * counts.min() / len(y)
    check("minority % (as printed in Q11, 14 dp)", E["minority_pct_reported"], f"{minority_pct:.14f}", "Moodle Q11")
    check("minority % rounded to paper (2 dp)", "16.47", f"{minority_pct:.2f}", "paper Table 5 (as quoted in Moodle Q8)")
    check("target missing", E["target_missing"], int(y.isna().sum()), "Moodle Q11")

    # Missingness
    miss = X.isna().sum()
    miss = miss[miss > 0]
    check("missing predictor cells", E["missing_total"], int(miss.sum()), "Moodle Q11")
    check("predictor columns with missing", E["missing_columns"], int(len(miss)), "Moodle Q11")
    obs_miss = {k: int(v) for k, v in miss.items()}
    diff_cols = {k: {"expected": HIST_MISSING.get(k), "observed": obs_miss.get(k)}
                 for k in sorted(set(HIST_MISSING) | set(obs_miss)) if HIST_MISSING.get(k) != obs_miss.get(k)}
    check("per-column missing counts identical (60 cols)", {}, diff_cols, "Moodle Q11 table")
    check("rows with >=1 missing predictor", 7330, int(X.isna().any(axis=1).sum()), "audit 2026-11-06 (OpenML quality)")

    # Feature types
    n_nominal = int(sum(bool(c) for c in cat_ind))
    check("nominal predictors (openml categorical_indicator)", 61, n_nominal, "Phase-1 spec / audit 2026-11-06")
    check("numeric predictors (openml categorical_indicator)", 59, int(len(cat_ind) - n_nominal), "Phase-1 spec / audit 2026-11-06")
    n_cat_dtype = int(sum(str(t) == "category" for t in X.dtypes))
    check("pandas 'category' dtype columns", 61, n_cat_dtype, "cross-check of the above")
    q = ds.qualities or {}
    check("OpenML quality NumberOfNumericFeatures", 59, int(q.get("NumberOfNumericFeatures", -1)), "Phase-1 spec / audit 2026-11-06")
    check("OpenML quality NumberOfSymbolicFeatures (incl. target)", 62, int(q.get("NumberOfSymbolicFeatures", -1)), "Phase-1 spec / audit 2026-11-06")

    # Ignored attributes: verify they are a label proxy (verification only; never used for modeling)
    full, _, _, _ = get_data(ds, include_ignore_attribute=True)
    m = full["match"].astype(str) == "1"
    dd = (full["decision"].astype(str) == "1") & (full["decision_o"].astype(str) == "1")
    check("match == decision AND decision_o on all rows", True, bool((m == dd).all()), "Phase-1 spec / audit 2026-11-06")

    # Content of openml-python's loaded file vs the checksummed ARFF
    import arff  # liac-arff, an openml dependency
    with open(os.path.join(a.raw_dir, manifest["arff_file"])) as f:
        A = arff.load(f)
    raw = pd.DataFrame(A["data"], columns=[c[0] for c in A["attributes"]])
    mismatched_cells = 0
    for col in raw.columns:
        r, o = raw[col], full[col]
        if pd.api.types.is_numeric_dtype(o) and not isinstance(o.dtype, pd.CategoricalDtype):
            eq = np.isclose(pd.to_numeric(r, errors="coerce").to_numpy(float), o.to_numpy(float), equal_nan=True, rtol=0, atol=1e-9)
        else:
            eq = r.astype(object).where(r.notna(), "<NA>").astype(str).to_numpy() == \
                 o.astype(object).where(o.notna(), "<NA>").astype(str).to_numpy()
        mismatched_cells += int((~eq).sum())
    check("loaded data == checksummed ARFF (mismatched cells)", 0, mismatched_cells, "Phase-1 spec / audit 2026-11-06")

    # Historical outer split (same procedure as the Checkpoint loader)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_seen = np.zeros(len(X), dtype=int)
    fold_id = np.full(len(X), -1)
    sizes, overlaps = [], []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        sizes.append((len(tr), len(te)))
        overlaps.append(int(np.intersect1d(tr, te).size))
        test_seen[te] += 1
        fold_id[te] = k
    check("fold (train, test) sizes", [list(s) for s in HIST_FOLDS], [list(s) for s in sizes], "Moodle Q11")
    check("train/test overlap per fold", [d["overlap"] for d in E["folds"]], overlaps, "Moodle Q11")
    check("rows held out exactly once", E["held_out_exactly_once"], int((test_seen == 1).sum()), "Moodle Q11")
    fold_hash = hashlib.sha256(fold_id.astype(np.int8).tobytes()).hexdigest()

    # Report
    width = max(len(r["check"]) for r in RESULTS)
    print(f"\nSpeedDating audit [{a.label}]")
    print(f"{'STATUS':6s}  {'CHECK':{width}s}  EXPECTED -> OBSERVED   (source)")
    for r in RESULTS:
        print(f"{r['status']:6s}  {r['check']:{width}s}  {r['expected']} -> {r['observed']}   ({r['source']})")
    n_diff = sum(r["status"] == "DIFF" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks, {n_diff} DIFF")
    print(f"fold-assignment sha256 (seed 42, record only, no historical value): {fold_hash}")

    out = {"label": a.label, "expected_file": a.expected, "expected_source_sha256": E["source_sha256"], "checks": RESULTS,
           "n_diff": n_diff, "fold_assignment_sha256_seed42": fold_hash,
           "missing_by_column_observed": obs_miss, "arff_sha256": manifest["sha256_observed"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=to_jsonable)
    print(f"written: {a.out}")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
