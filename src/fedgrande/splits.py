"""Outer CV schemes and their verification.

C1 ``rowwise``: StratifiedKFold(5, shuffle=True, random_state=42) on rows (the Checkpoint 1 / paper style).
C2 ``dyad``:    group-level StratifiedKFold(5, shuffle=True, random_state=42): whole dyad groups (dyad_id) are
               stratified by their label (max of ``match`` in the group) and assigned to folds, so both rows of a date
               (and every ambiguous candidate group) always fall on the same side. Chosen over sklearn's
               StratifiedGroupKFold(shuffle=True), whose fold positive rates deviated by up to 2.0 percentage points
               on this data (results/phase2/gate_c/splits_report.json, "alternatives"); the group-level method is
               seedable and deviates by ~0.1 pp because dyads are label-homogeneous.
Inner validation (for early stopping/thresholds) is carved out of the outer-training rows with the same grouped
method, so validation is never row-wise either (for C1, groups are single rows, reproducing a row-wise hold-out).
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

N_SPLITS = 5
SEED = 42


def fold_assignment_rowwise(y, n_splits=N_SPLITS, seed=SEED) -> np.ndarray:
    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        fold[te] = k
    return fold


def _group_labels(y, groups):
    """Unique groups (sorted) and their stratification label (1 if any row in the group is positive)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    lab = np.zeros(len(uniq), dtype=int)
    np.maximum.at(lab, inv, np.asarray(y, dtype=int))
    return uniq, inv, lab


def fold_assignment_grouped(y, groups, n_splits=N_SPLITS, seed=SEED) -> np.ndarray:
    """Group-level stratified K-fold: every row of a group gets the group's fold."""
    uniq, inv, lab = _group_labels(y, groups)
    gfold = np.full(len(uniq), -1)
    for k, (_, te) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(uniq)), lab)):
        gfold[te] = k
    return gfold[inv]


def fold_assignment_sgkf(y, groups, n_splits=N_SPLITS, shuffle=True, seed=SEED) -> np.ndarray:
    """sklearn StratifiedGroupKFold (kept for comparison only)."""
    fold = np.full(len(y), -1)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold[te] = k
    return fold


def inner_train_val(train_idx, y, groups, seed=SEED, n_splits=N_SPLITS):
    """Grouped stratified hold-out (fold 0 of a group-level K-fold) of the outer-training rows: (fit_idx, val_idx)."""
    train_idx = np.asarray(train_idx)
    f = fold_assignment_grouped(y[train_idx], groups[train_idx], n_splits=n_splits, seed=seed)
    return np.sort(train_idx[f != 0]), np.sort(train_idx[f == 0])


def fold_hash(fold: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(fold, dtype=np.int8).tobytes()).hexdigest()


def verify(fold, y, groups, mirror) -> dict:
    """Per-fold invariants and the mirror-leakage count (test rows whose mirror row is in that fold's training set)."""
    fold = np.asarray(fold)
    n = len(fold)
    out = {"all_rows_assigned_once": bool((fold >= 0).all()), "n_folds": int(fold.max() + 1), "folds": []}
    has_mirror = mirror >= 0
    for k in range(out["n_folds"]):
        te = np.flatnonzero(fold == k)
        tr = np.flatnonzero(fold != k)
        g_te, g_tr = set(groups[te].tolist()), set(groups[tr].tolist())
        te_m = te[has_mirror[te]]
        leak = int((fold[mirror[te_m]] != k).sum())
        out["folds"].append({
            "fold": k,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "row_overlap": int(len(np.intersect1d(tr, te))),
            "train_plus_test_equals_n": bool(len(tr) + len(te) == n),
            "dyad_group_overlap": int(len(g_te & g_tr)),
            "test_pos": int(y[te].sum()), "test_neg": int((1 - y[te]).sum()),
            "test_pos_rate": float(y[te].mean()), "train_pos_rate": float(y[tr].mean()),
            "test_rows_with_mirror": int(len(te_m)),
            "test_rows_whose_mirror_is_in_train": leak,
            "test_rows_whose_mirror_is_in_train_frac": float(leak / len(te)),
        })
    f = out["folds"]
    out["total_test_rows_whose_mirror_is_in_train"] = int(sum(x["test_rows_whose_mirror_is_in_train"] for x in f))
    out["total_dyad_group_overlap"] = int(sum(x["dyad_group_overlap"] for x in f))
    out["max_abs_pos_rate_deviation"] = float(max(abs(x["test_pos_rate"] - y.mean()) for x in f))
    out["fold_hash"] = fold_hash(fold)
    return out
