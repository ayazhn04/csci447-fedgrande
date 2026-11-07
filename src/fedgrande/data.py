"""Offline loader for SpeedDating (OpenML 40536 v1) from the checksummed ARFF of Phase 1.

Reads the local ARFF (no network). Nominal attributes become pandas ``category`` columns with the categories in
ARFF-declared order, numeric attributes float64 -- the same representation openml-python produces
(Phase 1 verified openml's loaded data equals this ARFF cell by cell; tests/test_data.py re-checks the dtypes).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import ARFF, MANIFEST
from .provenance import sha256_file

TARGET = "match"
IGNORED = ("decision", "decision_o")  # OpenML ignore attributes: label proxies, never model features


def load_full(verify: bool = True) -> pd.DataFrame:
    """All 123 attributes in file order (including the ignored label proxies, for verification only)."""
    import arff  # liac-arff, installed as an openml dependency

    if verify:
        manifest = json.loads(MANIFEST.read_text())
        observed = sha256_file(ARFF)
        if observed != manifest["sha256_observed"]:
            raise RuntimeError(f"ARFF sha256 {observed} != Phase 1 manifest {manifest['sha256_observed']}")
    with open(ARFF) as f:
        A = arff.load(f)
    df = pd.DataFrame(A["data"], columns=[a[0] for a in A["attributes"]])
    for name, kind in A["attributes"]:
        if isinstance(kind, list):
            df[name] = pd.Categorical(df[name], categories=kind)
        else:
            df[name] = pd.to_numeric(df[name], errors="raise").astype(np.float64)
    return df


def load_xy(verify: bool = True):
    """Modeling view: X (120 predictors, label proxies and target removed), y (int 0/1), and the full frame."""
    full = load_full(verify=verify)
    X = full.drop(columns=[TARGET, *IGNORED])
    y = full[TARGET].astype(str).astype(int).to_numpy()
    assert X.shape == (8378, 120), X.shape
    assert not set(IGNORED) & set(X.columns) and TARGET not in X.columns
    return X, y, full
