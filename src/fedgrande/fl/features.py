"""Phase 2B modeling view of SpeedDating and leakage-group metadata.

Model features = the 120 OpenML predictors minus ``wave`` (client identity metadata only) -> 119 raw columns.
``match`` (target), ``decision`` and ``decision_o`` (label proxies) are never features.
Column types come from the ARFF/OpenML schema: nominal -> categorical, numeric -> numeric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import load_xy
from ..paths import DERIVED

METADATA_ONLY = ("wave",)
NEVER_FEATURES = ("match", "decision", "decision_o", "wave")
GROUP_KIND = {"pair_mutual": "dyad", "pair_one_way": "dyad", "ambiguous_group": "ambiguous_leakage_group",
              "singleton": "singleton"}


def load_modeling():
    """Return (X [8378 x 119], y [int 0/1], meta DataFrame with wave, leakage_group_id, group_kind)."""
    X120, y, full = load_xy()
    X = X120.drop(columns=list(METADATA_ONLY))
    d = pd.read_csv(DERIVED / "dyads_v1.csv")
    assert (d.row_index.to_numpy() == np.arange(len(y))).all()
    meta = pd.DataFrame({
        "row_index": np.arange(len(y)),
        "wave": X120["wave"].astype(int).to_numpy(),
        "leakage_group_id": d["dyad_id"].to_numpy(),          # Phase 2A column name 'dyad_id' = connected component id
        "group_kind": d["dyad_status"].map(GROUP_KIND).to_numpy(),
    })
    assert not set(NEVER_FEATURES) & set(X.columns)
    return X, y, meta


def schema(X: pd.DataFrame) -> dict:
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    return {"categorical": cat, "numeric": num, "n_raw": X.shape[1], "n_categorical": len(cat), "n_numeric": len(num)}
