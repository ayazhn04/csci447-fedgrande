"""Federated-safe, label-free preprocessing shared by every Phase 2B model.

Fitting uses only TRAINING (inner-fit) rows of the outer-training portion; validation and test rows never
contribute. Each client computes label-free sufficient statistics; the server aggregates them:

numeric column j:   n_j = #non-missing, s_j = sum x, q_j = sum x^2  (float64)
                    mean_j = s_j / n_j ; var_j = q_j / n_j - mean_j^2 (population) ; sd_j = sqrt(var_j) or 1 if var_j <= 0
                    transform: impute missing with mean_j, then (x - mean_j) / sd_j
categorical col c:  per-client category counts (no labels). Vocabulary = categories with count > 0 in the
                    aggregated training counts, in schema order. A dedicated MISSING indicator column is added iff
                    the aggregated training data contains missing values for c. Unseen categories (and missing
                    values in a column without a MISSING column) map to UNK = all-zero block.
Output: float32 [numeric block (z-scores) | one-hot blocks], identical schema for every client and model.
No function in this module accepts or reads the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def client_statistics(Xc: pd.DataFrame, numeric: list, categorical: list) -> dict:
    """Label-free sufficient statistics of one client's training rows."""
    A = Xc[numeric].to_numpy(dtype=np.float64)
    mask = ~np.isnan(A)
    A0 = np.where(mask, A, 0.0)
    cats = {}
    for c in categorical:
        vc = Xc[c].astype(object).where(Xc[c].notna(), None)
        counts = {}
        for v in vc:
            key = "__MISSING__" if v is None else str(v)
            counts[key] = counts.get(key, 0) + 1
        cats[c] = counts
    return {"n_rows": int(len(Xc)), "num_count": mask.sum(0), "num_sum": A0.sum(0), "num_sumsq": (A0 ** 2).sum(0), "cat_counts": cats}


def aggregate(stats: list, numeric: list, categorical: list, category_order: dict) -> dict:
    n = np.sum([s["num_count"] for s in stats], axis=0).astype(np.float64)
    s1 = np.sum([s["num_sum"] for s in stats], axis=0)
    s2 = np.sum([s["num_sumsq"] for s in stats], axis=0)
    if (n == 0).any():
        raise ValueError(f"numeric columns without training values: {[numeric[i] for i in np.flatnonzero(n == 0)]}")
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    sd = np.sqrt(var)
    sd = np.where(var <= 1e-12, 1.0, sd)
    vocab, has_missing, counts_total = {}, {}, {}
    for c in categorical:
        tot = {}
        for s in stats:
            for k, v in s["cat_counts"][c].items():
                tot[k] = tot.get(k, 0) + v
        counts_total[c] = tot
        vocab[c] = [str(k) for k in category_order[c] if tot.get(str(k), 0) > 0]
        extra = sorted(k for k in tot if k != "__MISSING__" and k not in set(map(str, category_order[c])))
        if extra:
            raise ValueError(f"category values outside the schema in {c}: {extra[:5]}")
        has_missing[c] = tot.get("__MISSING__", 0) > 0
    return {"numeric": list(numeric), "categorical": list(categorical), "mean": mean, "sd": sd, "vocab": vocab,
            "has_missing": has_missing, "n_train_rows": int(sum(s["n_rows"] for s in stats)), "category_counts": counts_total}


class FederatedPreprocessor:
    """Fit from client statistics (or pooled rows, which is the 1-client special case); transform any rows."""

    def __init__(self, numeric: list, categorical: list, category_order: dict):
        self.numeric, self.categorical, self.category_order = list(numeric), list(categorical), category_order

    def fit_from_clients(self, client_frames: list) -> "FederatedPreprocessor":
        stats = [client_statistics(Xc, self.numeric, self.categorical) for Xc in client_frames]
        self.params_ = aggregate(stats, self.numeric, self.categorical, self.category_order)
        self._build_names()
        return self

    def fit_pooled(self, X: pd.DataFrame) -> "FederatedPreprocessor":
        return self.fit_from_clients([X])

    def _build_names(self):
        names = list(self.numeric)
        self.blocks_ = {}
        for c in self.categorical:
            start = len(names)
            names += [f"{c}={v}" for v in self.params_["vocab"][c]]
            if self.params_["has_missing"][c]:
                names.append(f"{c}=__MISSING__")
            self.blocks_[c] = (start, len(names))
        self.feature_names_ = names
        self.n_features_ = len(names)
        self.numeric_index_ = list(range(len(self.numeric)))

    def transform(self, X: pd.DataFrame, return_unseen: bool = False):
        p = self.params_
        A = X[self.numeric].to_numpy(dtype=np.float64)
        A = np.where(np.isnan(A), p["mean"][None, :], A)
        out = np.zeros((len(X), self.n_features_), dtype=np.float32)
        out[:, : len(self.numeric)] = ((A - p["mean"]) / p["sd"]).astype(np.float32)
        unseen = {}
        for c in self.categorical:
            start, end = self.blocks_[c]
            lookup = {v: start + i for i, v in enumerate(p["vocab"][c])}
            miss_col = end - 1 if p["has_missing"][c] else None
            vals = X[c].astype(object).to_numpy()
            n_unseen = 0
            for r, v in enumerate(vals):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    if miss_col is not None:
                        out[r, miss_col] = 1.0
                    else:
                        n_unseen += 1      # missing but no MISSING column: UNK (all zeros)
                else:
                    j = lookup.get(str(v))
                    if j is None:
                        n_unseen += 1      # unseen category: UNK (all zeros)
                    else:
                        out[r, j] = 1.0
            if n_unseen:
                unseen[c] = n_unseen
        return (out, unseen) if return_unseen else out

    def summary(self) -> dict:
        return {"n_numeric": len(self.numeric), "n_categorical": len(self.categorical), "encoded_dim": self.n_features_,
                "one_hot_columns": self.n_features_ - len(self.numeric),
                "categories_per_column": {c: len(self.params_["vocab"][c]) for c in self.categorical},
                "missing_indicator_columns": [c for c in self.categorical if self.params_["has_missing"][c]],
                "n_train_rows": self.params_["n_train_rows"]}
