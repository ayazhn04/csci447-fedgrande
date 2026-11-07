"""Metrics and the single deterministic validation-only threshold procedure used for every model."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def select_threshold(y_val, p_val) -> float:
    """Maximize validation macro-F1 over a fixed grid 0.05..0.95 step 0.01; ties -> closest to 0.5."""
    if len(y_val) == 0:
        return 0.5
    scores = np.array([f1_score(y_val, (p_val >= t).astype(int), average="macro", zero_division=0) for t in THRESHOLD_GRID])
    best = np.flatnonzero(scores == scores.max())
    return float(THRESHOLD_GRID[best[np.argmin(np.abs(THRESHOLD_GRID[best] - 0.5))]])


def binary_metrics(y, p, thr=0.5, pred=None) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int) if pred is None else np.asarray(pred).astype(int)
    return {"n": int(len(y)), "pos": int(y.sum()), "roc_auc": safe_auc(y, p),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "confusion_[[tn,fp],[fn,tp]]": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
