#!/usr/bin/env python3
"""Fetch the SpeedDating ARFF (if missing) and rebuild the gitignored per-fold preprocessing caches.

Safe: data download and label-free preprocessing only. No training, no evaluation, and no tracked file is written.

- The ARFF is downloaded from OpenML by the Phase 1 fetcher (src/fedgrande/phase1/fetch_speeddating.py), which checks
  OpenML's MD5. It goes to data/raw/openml_40536/.
- The caches data/derived/phase2b/prep_C2_outer{0..4}.npz are rebuilt with exactly the code path of
  src/fedgrande/phase2b/prepare.py (FederatedPreprocessor fitted on the inner-fit rows of the natural clients). They are
  then verified against the array digests recorded in data/derived/phase2b/cache_digests.json.
- Unlike prepare.py, this script does NOT rewrite the tracked partition files or the Phase 2B prepare report. It only
  checks that the recomputed partitions equal the tracked data/derived/phase2b/partitions_v1.csv.

Usage: python scripts/build_data_caches.py [--skip-fetch] [--record-digests]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DIGESTS = ROOT / "data" / "derived" / "phase2b" / "cache_digests.json"
KEYS = ("Z", "roles", "feature_names", "n_numeric", "mean", "sd")


def digest(npz_path: Path) -> dict:
    import numpy as np
    z = np.load(npz_path)
    return {k: hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--record-digests", action="store_true", help="(maintainers) record digests of the existing caches")
    a = ap.parse_args()
    out = ROOT / "data" / "derived" / "phase2b"
    if a.record_digests:
        DIGESTS.write_text(json.dumps({f"prep_C2_outer{k}.npz": digest(out / f"prep_C2_outer{k}.npz") for k in range(5)}, indent=1) + "\n")
        print(f"recorded {DIGESTS.relative_to(ROOT)}")
        return
    from fedgrande.paths import ARFF, RAW_DIR
    if not ARFF.exists():
        if a.skip_fetch:
            sys.exit(f"missing {ARFF.relative_to(ROOT)}; run without --skip-fetch")
        subprocess.run([sys.executable, "-I", str(ROOT / "src/fedgrande/phase1/fetch_speeddating.py"), "--out-dir", str(RAW_DIR)], check=True)

    import numpy as np
    import pandas as pd
    from fedgrande.fl import partitions as P
    from fedgrande.fl.features import load_modeling, schema
    from fedgrande.fl.preprocess import FederatedPreprocessor
    from fedgrande.paths import DERIVED

    cfg = json.load(open(ROOT / "configs" / "phase2b" / "core.json"))
    X, y, meta = load_modeling()
    S = schema(X)
    nat = P.natural(meta)
    iid = P.iid_size_matched(meta, y, seed=cfg["seeds"]["partition_seed"])
    tracked = pd.read_csv(out / "partitions_v1.csv")
    assert (tracked.client_natural.to_numpy() == nat).all() and (tracked.client_iid.to_numpy() == iid).all(), "partitions differ from tracked file"
    sizes = np.bincount(nat)
    inner = pd.read_csv(DERIVED / "splits_inner_v1.csv")
    order = {c: list(X[c].cat.categories) for c in S["categorical"]}
    expected = json.loads(DIGESTS.read_text())
    for k in cfg["outer_split"]["folds"]:
        s = inner[(inner.scheme == "C2") & (inner.outer_fold == k)].sort_values("row_index")
        roles = s.role.to_numpy()
        fit = roles == "fit"
        pp = FederatedPreprocessor(S["numeric"], S["categorical"], order).fit_from_clients([X[fit & (nat == c)] for c in range(len(sizes))])
        Z = pp.transform(X)
        f = out / f"prep_C2_outer{k}.npz"
        np.savez(f, Z=Z, roles=roles.astype("U4"), feature_names=np.array(pp.feature_names_, dtype="U128"), n_numeric=np.array(len(S["numeric"])),
                 mean=pp.params_["mean"], sd=pp.params_["sd"])
        got = digest(f)
        bad = [key for key in KEYS if got[key] != expected[f.name][key]]
        assert not bad, f"{f.name}: arrays differ from the recorded experiment caches: {bad}"
        print(f"{f.relative_to(ROOT)}: rebuilt, {Z.shape[1]} encoded features, identical to the experiment cache")
    print("data caches OK")


if __name__ == "__main__":
    main()
